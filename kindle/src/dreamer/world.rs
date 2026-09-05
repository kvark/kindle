//! Categorical DreamerV3 world model graphs.

use meganeura::{Graph, graph::NodeId};

use super::config::DreamerConfig;
use super::networks::{
    MlpHead, ObservationDecoder, Prior, Representation, RssmCore, categorical_kl, feature,
    mixed_probabilities, scale, straight_through_sample, sum,
};
use crate::vision::{OBSERVATION_CHANNELS, OBSERVATION_GRID};

pub const LOSS_TOTAL: usize = 0;
pub const LOSS_RECONSTRUCTION: usize = 1;
pub const LOSS_DYNAMICS: usize = 2;
pub const LOSS_REPRESENTATION: usize = 3;
pub const LOSS_REWARD: usize = 4;
pub const LOSS_CONTINUATION: usize = 5;
pub const LOSS_REPLAY_VALUE: usize = 6;
pub const RAW_KL: usize = 7;
pub const LOSS_FUTURE_PREDICTION: usize = 8;

struct Dynamics {
    core: RssmCore,
    prior: Prior,
}

impl Dynamics {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        Self {
            core: RssmCore::new(graph, config),
            prior: Prior::new(graph, config),
        }
    }
}

struct WorldHeads {
    reward: MlpHead,
    continuation: MlpHead,
}

impl WorldHeads {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            reward: MlpHead::new(
                graph,
                "world.reward",
                config.feature_dim(),
                units,
                1,
                config.value_bins,
            ),
            continuation: MlpHead::new(
                graph,
                "world.continuation",
                config.feature_dim(),
                units,
                1,
                1,
            ),
        }
    }

    fn forward(&self, graph: &mut Graph, state: NodeId) -> (NodeId, NodeId) {
        let reward = self.reward.forward(graph, state);
        let continuation = self.continuation.forward(graph, state);
        let continuation = graph.sigmoid(continuation);
        (reward, continuation)
    }
}

struct WorldModel {
    dynamics: Dynamics,
    representation: Representation,
    decoder: Option<ObservationDecoder>,
    future_predictor: Option<MlpHead>,
    heads: WorldHeads,
    /// The behavior optimizer owns these parameters. They are frozen in this
    /// graph so replay-value gradients only shape the posterior/RSSM path.
    replay_value: MlpHead,
}

impl WorldModel {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            dynamics: Dynamics::new(graph, config),
            representation: Representation::new(graph, config),
            decoder: (config.loss_scales.reconstruction > 0.0)
                .then(|| ObservationDecoder::new(graph, config)),
            future_predictor: (config.loss_scales.future_prediction > 0.0)
                .then(|| future_predictor(graph, config)),
            heads: WorldHeads::new(graph, config),
            replay_value: MlpHead::new(
                graph,
                "behavior.value",
                config.feature_dim(),
                units,
                3,
                config.value_bins,
            ),
        }
    }
}

fn future_predictor(graph: &mut Graph, config: &DreamerConfig) -> MlpHead {
    MlpHead::new(
        graph,
        "world.future_predictor",
        config.network().deter,
        config.network().units,
        2,
        config.observation_dim(),
    )
}

fn weighted_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    weight: NodeId,
) -> NodeId {
    let log_probability = graph.log_softmax(logits);
    let product = graph.mul(target, log_probability);
    let per_row = graph.sum_inner(product);
    let per_row = graph.neg(per_row);
    let weighted = graph.mul(per_row, weight);
    graph.mean_all(weighted)
}

/// Full sequence loss with externally sampled hard posterior states.
///
/// The hard samples are produced immediately before this graph runs by the
/// observe graph. Inside this graph they are combined with posterior
/// probabilities using `hard + probs - stop_gradient(probs)`, exactly D3's
/// straight-through categorical estimator.
pub fn build_training_graph(config: &DreamerConfig, length: usize) -> Graph {
    config.validate();
    assert!(length > 0 && length <= config.batch_length);
    let batch = config.batch_size;
    let size = config.network();
    let patches = OBSERVATION_GRID * OBSERVATION_GRID;

    let mut graph = Graph::new();
    let model = WorldModel::new(&mut graph, config);
    let mut deter = graph.input("initial_deter", &[batch, size.deter]);
    let mut stoch = graph.input("initial_stoch", &[batch * size.stoch, size.classes]);

    let mut reconstruction_losses = Vec::with_capacity(length);
    let mut future_prediction_losses = Vec::with_capacity(length);
    let mut dynamics_losses = Vec::with_capacity(length);
    let mut representation_losses = Vec::with_capacity(length);
    let mut raw_kl_metrics = Vec::with_capacity(length);
    let mut reward_losses = Vec::with_capacity(length);
    let mut continuation_losses = Vec::with_capacity(length);
    let mut replay_value_losses = Vec::with_capacity(length);
    let reward_weight = graph.constant(vec![1.0; batch], &[batch, 1]);

    for time in 0..length {
        let observation = graph.input(
            &format!("observation_{time}"),
            &[batch * patches, OBSERVATION_CHANNELS],
        );
        let action = graph.input(
            &format!("previous_action_{time}"),
            &[batch, config.action_count],
        );
        let keep_deter = graph.input(&format!("keep_deter_{time}"), &[batch, size.deter]);
        let keep_stoch = graph.input(
            &format!("keep_stoch_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let keep_action = graph.input(
            &format!("keep_action_{time}"),
            &[batch, config.action_count],
        );
        let hard_sample = graph.input(
            &format!("posterior_sample_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let reward_target = graph.input(
            &format!("reward_target_{time}"),
            &[batch, config.value_bins],
        );
        let continuation_target = graph.input(&format!("continuation_target_{time}"), &[batch, 1]);
        let replay_value_target = graph.input(
            &format!("replay_value_target_{time}"),
            &[batch, config.value_bins],
        );
        let replay_slow_target = graph.input(
            &format!("replay_slow_target_{time}"),
            &[batch, config.value_bins],
        );
        let replay_value_weight = graph.input(&format!("replay_value_weight_{time}"), &[batch, 1]);

        let masked_deter = graph.mul(deter, keep_deter);
        let masked_stoch = graph.mul(stoch, keep_stoch);
        let masked_action = graph.mul(action, keep_action);
        deter = model.dynamics.core.forward(
            &mut graph,
            masked_deter,
            masked_stoch,
            masked_action,
            batch,
        );
        let prior_logits = model.dynamics.prior.forward(&mut graph, deter, batch);
        if let Some(predictor) = &model.future_predictor {
            let prediction = predictor.forward(&mut graph, deter);
            let target = graph.reshape(observation, &[batch, config.observation_dim()]);
            let target = graph.stop_gradient(target);
            let negative_target = graph.neg(target);
            let residual = graph.add(prediction, negative_target);
            let squared = graph.mul(residual, residual);
            let per_row = graph.sum_inner(squared);
            // Reset observations have no preceding action-conditioned state.
            // Their zero keep mask removes them without cutting valid sequence
            // or microbatch boundaries. Normalize over the same B*T as D3.
            let keep = graph.sum_inner(keep_action);
            let normalization =
                graph.constant(vec![1.0 / config.action_count as f32; batch], &[batch, 1]);
            let keep = graph.mul(keep, normalization);
            let masked = graph.mul(per_row, keep);
            future_prediction_losses.push(graph.mean_all(masked));
        }
        let encoded = model
            .representation
            .encoder
            .forward(&mut graph, observation, batch);
        let posterior_logits = model
            .representation
            .posterior
            .forward(&mut graph, deter, encoded, batch);
        let probabilities = mixed_probabilities(
            &mut graph,
            posterior_logits,
            batch * size.stoch,
            size.classes,
            config.unimix,
        );
        stoch = straight_through_sample(&mut graph, hard_sample, probabilities);

        let state = feature(&mut graph, deter, stoch, batch, config);
        if let Some(decoder) = &model.decoder {
            let reconstruction = decoder.forward(&mut graph, state, batch);
            let negative_target = graph.neg(observation);
            let residual = graph.add(reconstruction, negative_target);
            let squared = graph.mul(residual, residual);
            let reconstruction_loss = graph.sum_all(squared);
            // Sum event dimensions and average B*T, as in the D3 control.
            reconstruction_losses.push(scale(&mut graph, reconstruction_loss, 1.0 / batch as f32));
        }

        let dynamics_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            true,
            false,
            config.dynamics_free_nats(),
        );
        dynamics_losses.push(dynamics_kl.loss);
        raw_kl_metrics.push(dynamics_kl.raw);
        let representation_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            false,
            true,
            config.free_nats,
        );
        representation_losses.push(representation_kl.loss);

        let (reward, continuation) = model.heads.forward(&mut graph, state);
        // Keep the reduction explicit because this value is composed into the
        // joint loss and exported as a metric. The fused backend loss stores
        // per-row partials and is only scalar when used as a terminal output.
        reward_losses.push(weighted_cross_entropy(
            &mut graph,
            reward,
            reward_target,
            reward_weight,
        ));
        continuation_losses.push(graph.bce_loss(continuation, continuation_target));
        let replay_value_state = if config.replay_value_gradient {
            state
        } else {
            graph.stop_gradient(state)
        };
        let value = model
            .replay_value
            .forward_frozen(&mut graph, replay_value_state);
        let value_target =
            weighted_cross_entropy(&mut graph, value, replay_value_target, replay_value_weight);
        let slow_target =
            weighted_cross_entropy(&mut graph, value, replay_slow_target, replay_value_weight);
        replay_value_losses.push(graph.add(value_target, slow_target));
    }

    let average = 1.0 / length as f32;
    let reconstruction = sum_or_zero(&mut graph, &reconstruction_losses);
    let reconstruction = scale(&mut graph, reconstruction, average);
    let future_prediction = sum_or_zero(&mut graph, &future_prediction_losses);
    let future_prediction = scale(&mut graph, future_prediction, average);
    let dynamics = sum(&mut graph, &dynamics_losses);
    let dynamics = scale(&mut graph, dynamics, average);
    let representation = sum(&mut graph, &representation_losses);
    let representation = scale(&mut graph, representation, average);
    let raw_kl = sum(&mut graph, &raw_kl_metrics);
    let raw_kl = scale(&mut graph, raw_kl, average);
    let reward = sum(&mut graph, &reward_losses);
    let reward = scale(&mut graph, reward, average);
    let continuation = sum(&mut graph, &continuation_losses);
    let continuation = scale(&mut graph, continuation, average);
    let replay_value = sum(&mut graph, &replay_value_losses);
    let replay_value = scale(&mut graph, replay_value, average);

    let scales = config.loss_scales;
    let weighted = [
        scale(&mut graph, reconstruction, scales.reconstruction),
        scale(&mut graph, future_prediction, scales.future_prediction),
        scale(&mut graph, dynamics, scales.dynamics),
        scale(&mut graph, representation, scales.representation),
        scale(&mut graph, reward, scales.reward),
        scale(&mut graph, continuation, scales.continuation),
        scale(&mut graph, replay_value, scales.replay_value),
    ];
    let total = sum(&mut graph, &weighted);
    graph.set_outputs(vec![
        total,
        reconstruction,
        dynamics,
        representation,
        reward,
        continuation,
        replay_value,
        raw_kl,
        future_prediction,
    ]);
    graph
}

fn sum_or_zero(graph: &mut Graph, values: &[NodeId]) -> NodeId {
    if values.is_empty() {
        graph.constant(vec![0.0], &[1])
    } else {
        sum(graph, values)
    }
}

/// One posterior update used both by the live actor and the pre-training
/// categorical sampling pass.
///
/// Outputs: next deterministic state, posterior logits, prior logits, and the
/// trainable observation-encoder output used by the posterior.
pub fn build_observe_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let patches = OBSERVATION_GRID * OBSERVATION_GRID;
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let representation = Representation::new(&mut graph, config);
    let previous_deter = graph.input("previous_deter", &[batch, size.deter]);
    let previous_stoch = graph.input("previous_stoch", &[batch * size.stoch, size.classes]);
    let previous_action = graph.input("previous_action", &[batch, config.action_count]);
    let observation = graph.input("observation", &[batch * patches, OBSERVATION_CHANNELS]);
    let keep_deter = graph.input("keep_deter", &[batch, size.deter]);
    let keep_stoch = graph.input("keep_stoch", &[batch * size.stoch, size.classes]);
    let keep_action = graph.input("keep_action", &[batch, config.action_count]);
    let previous_deter = graph.mul(previous_deter, keep_deter);
    let previous_stoch = graph.mul(previous_stoch, keep_stoch);
    let previous_action = graph.mul(previous_action, keep_action);
    let deter = dynamics.core.forward(
        &mut graph,
        previous_deter,
        previous_stoch,
        previous_action,
        batch,
    );
    let prior = dynamics.prior.forward(&mut graph, deter, batch);
    let encoded = representation
        .encoder
        .forward(&mut graph, observation, batch);
    let posterior = representation
        .posterior
        .forward(&mut graph, deter, encoded, batch);
    graph.set_outputs(vec![deter, posterior, prior, encoded]);
    graph
}

/// One prior transition for imagination.
///
/// Outputs: next deterministic state and prior logits. Categorical sampling
/// happens on the CPU between this graph and the next imagined step.
pub fn build_transition_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let action = graph.input("action", &[batch, config.action_count]);
    let next_deter = dynamics
        .core
        .forward(&mut graph, deter, stoch, action, batch);
    let prior = dynamics.prior.forward(&mut graph, next_deter, batch);
    graph.set_outputs(vec![next_deter, prior]);
    graph
}

/// Reward and continuation predictions for posterior or imagined states.
pub fn build_head_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let heads = WorldHeads::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let (reward, continuation) = heads.forward(&mut graph, state);
    graph.set_outputs(vec![reward, continuation]);
    graph
}

/// Evaluate the configured observation head. Reconstruction reads the full
/// state; prediction-only reads the deterministic state before observation.
/// This graph is constructed lazily by diagnostics.
pub fn build_decoder_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    if config.loss_scales.reconstruction == 0.0 {
        assert!(
            config.loss_scales.future_prediction > 0.0,
            "no observation prediction head"
        );
        let predictor = future_predictor(&mut graph, config);
        let deter = graph.input("deter", &[batch, size.deter]);
        graph.input("stoch", &[batch * size.stoch, size.classes]);
        let observation = predictor.forward(&mut graph, deter);
        graph.set_outputs(vec![observation]);
        return graph;
    }
    let decoder = ObservationDecoder::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let observation = decoder.forward(&mut graph, state, batch);
    graph.set_outputs(vec![observation]);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_graphs_expose_d3_shapes() {
        let config = DreamerConfig::tiny(3);
        let size = config.network();
        let training = build_training_graph(&config, config.world_backprop_length);
        assert_eq!(training.outputs().len(), 9);
        let grouped_weight = training
            .nodes()
            .iter()
            .find_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name }
                    if name == "world.dynamics.core.dynhid0.weight" =>
                {
                    Some(node.ty.shape.as_slice())
                }
                _ => None,
            })
            .expect("grouped RSSM weight");
        assert_eq!(
            grouped_weight,
            [
                size.blocks,
                size.deter / size.blocks + 3 * size.hidden,
                size.deter / size.blocks,
            ]
        );
        let observe = build_observe_graph(&config, 2);
        assert_eq!(
            observe.node(observe.outputs()[0]).ty.shape,
            vec![2, size.deter]
        );
        assert_eq!(
            observe.node(observe.outputs()[1]).ty.shape,
            vec![2 * size.stoch, size.classes]
        );
        assert_eq!(
            observe.node(observe.outputs()[3]).ty.shape,
            vec![2, OBSERVATION_GRID * OBSERVATION_GRID * size.vision_depth]
        );
        let transition = build_transition_graph(&config, 5);
        assert_eq!(
            transition.node(transition.outputs()[0]).ty.shape,
            vec![5, size.deter]
        );
        let heads = build_head_graph(&config, 5);
        assert_eq!(
            heads.node(heads.outputs()[0]).ty.shape,
            vec![5, config.value_bins]
        );
        assert_eq!(heads.node(heads.outputs()[1]).ty.shape, vec![5, 1]);
        let decoder = build_decoder_graph(&config, 5);
        assert_eq!(
            decoder.node(decoder.outputs()[0]).ty.shape,
            vec![
                5 * OBSERVATION_GRID * OBSERVATION_GRID,
                OBSERVATION_CHANNELS
            ]
        );
    }

    #[test]
    fn prediction_only_omits_decoder_parameters() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let names = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!names.iter().any(|name| name.starts_with("world.decoder.")));
        assert!(
            names
                .iter()
                .any(|name| name.starts_with("world.future_predictor."))
        );
    }

    #[test]
    fn tiny_world_training_graph_autodiffs_and_compiles() {
        let config = DreamerConfig::tiny(3);
        let graph = build_training_graph(&config, config.world_backprop_length);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }
}
