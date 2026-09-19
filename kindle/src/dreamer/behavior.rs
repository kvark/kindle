//! DreamerV3 actor and distributional value model.

use meganeura::{Graph, graph::NodeId};

use super::config::DreamerConfig;
use super::networks::{MlpHead, scale, sum, weighted_cross_entropy};

pub const LOSS_TOTAL: usize = 0;
pub const LOSS_POLICY: usize = 1;
pub const LOSS_VALUE: usize = 2;
pub const LOSS_REPLAY_VALUE: usize = 3;
pub const POLICY_ENTROPY: usize = 4;
pub const WEIGHTED_POLICY_ENTROPY: usize = 5;

struct BehaviorModel {
    actor: MlpHead,
    value: MlpHead,
}

impl BehaviorModel {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            actor: MlpHead::new(
                graph,
                "behavior.actor",
                config.feature_dim(),
                units,
                3,
                config.action_count,
            ),
            value: MlpHead::new(
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

fn value_loss(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    slow_target: NodeId,
    weight: NodeId,
) -> NodeId {
    let target_loss = weighted_cross_entropy(graph, logits, target, weight);
    let slow_loss = weighted_cross_entropy(graph, logits, slow_target, weight);
    graph.add(target_loss, slow_loss)
}

/// Joint actor/value update over imagined states plus D3's replay-value loss.
///
/// `action_target` is `discount_weight * normalized_advantage * one_hot`.
/// Cross entropy is linear in this target, yielding the exact score-function
/// policy gradient for positive and negative advantages.
pub fn build_training_graph(
    config: &DreamerConfig,
    imagined_rows: usize,
    replay_rows: usize,
) -> Graph {
    config.validate();
    assert!(imagined_rows > 0 && replay_rows > 0);
    let mut graph = Graph::new();
    let model = BehaviorModel::new(&mut graph, config);

    let imagined_feature = graph.input("imagined_feature", &[imagined_rows, config.feature_dim()]);
    let action_target = graph.input("action_target", &[imagined_rows, config.action_count]);
    let imagined_weight = graph.input("imagined_weight", &[imagined_rows, 1]);
    let imagined_value_target =
        graph.input("imagined_value_target", &[imagined_rows, config.value_bins]);
    let imagined_slow_target =
        graph.input("imagined_slow_target", &[imagined_rows, config.value_bins]);

    let actor_logits = model.actor.forward(&mut graph, imagined_feature);
    let value_logits = model.value.forward(&mut graph, imagined_feature);
    let probabilities = graph.softmax(actor_logits);
    let retained = graph.constant(
        vec![1.0 - config.actor_unimix; imagined_rows * config.action_count],
        &[imagined_rows, config.action_count],
    );
    let probabilities = graph.mul(probabilities, retained);
    let uniform = graph.constant(
        vec![config.actor_unimix / config.action_count as f32; imagined_rows * config.action_count],
        &[imagined_rows, config.action_count],
    );
    let probabilities = graph.add(probabilities, uniform);
    let mixed_logits = graph.log(probabilities);
    // Keep this reduction explicit. Meganeura's fused cross-entropy stores
    // per-row partials for terminal losses, whereas this loss is composed with
    // entropy and value terms and must remain a true scalar metric.
    let score_terms = graph.mul(action_target, mixed_logits);
    let score_loss = graph.sum_inner(score_terms);
    let score_loss = graph.mean_all(score_loss);
    let score_loss = graph.neg(score_loss);
    let log_probabilities = graph.log(probabilities);
    let p_log_p = graph.mul(probabilities, log_probabilities);
    let entropy = graph.sum_inner(p_log_p);
    let entropy = graph.neg(entropy);
    let mean_entropy = graph.mean_all(entropy);
    let weighted_entropy = graph.mul(entropy, imagined_weight);
    let weighted_entropy = graph.mean_all(weighted_entropy);
    let entropy_bonus = scale(&mut graph, weighted_entropy, -config.actor_entropy);
    let policy = graph.add(score_loss, entropy_bonus);
    let value = value_loss(
        &mut graph,
        value_logits,
        imagined_value_target,
        imagined_slow_target,
        imagined_weight,
    );

    let replay_feature = graph.input("replay_feature", &[replay_rows, config.feature_dim()]);
    let replay_weight = graph.input("replay_weight", &[replay_rows, 1]);
    let replay_target = graph.input("replay_value_target", &[replay_rows, config.value_bins]);
    let replay_slow_target = graph.input("replay_slow_target", &[replay_rows, config.value_bins]);
    let replay_logits = model.value.forward(&mut graph, replay_feature);
    let replay_value = value_loss(
        &mut graph,
        replay_logits,
        replay_target,
        replay_slow_target,
        replay_weight,
    );

    let scales = config.loss_scales;
    let weighted_policy = scale(&mut graph, policy, scales.policy);
    let weighted_value = scale(&mut graph, value, scales.value);
    let weighted_replay = scale(&mut graph, replay_value, scales.replay_value);
    let total = sum(
        &mut graph,
        &[weighted_policy, weighted_value, weighted_replay],
    );
    graph.set_outputs(vec![
        total,
        policy,
        value,
        replay_value,
        mean_entropy,
        weighted_entropy,
    ]);
    graph
}

/// Current actor and value prediction. Slow-value and live-policy sessions
/// use this same graph and differ only in how parameters are synchronized.
pub fn build_inference_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let mut graph = Graph::new();
    let model = BehaviorModel::new(&mut graph, config);
    let state = graph.input("feature", &[batch, config.feature_dim()]);
    let actor = model.actor.forward(&mut graph, state);
    let value = model.value.forward(&mut graph, state);
    graph.set_outputs(vec![actor, value]);
    graph
}

pub fn build_actor_inference_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let mut graph = Graph::new();
    let actor = MlpHead::new(
        &mut graph,
        "behavior.actor",
        config.feature_dim(),
        config.network().units,
        3,
        config.action_count,
    );
    let state = graph.input("feature", &[batch, config.feature_dim()]);
    let logits = actor.forward(&mut graph, state);
    graph.set_outputs(vec![logits]);
    graph
}

pub fn build_value_inference_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let mut graph = Graph::new();
    let value = MlpHead::new(
        &mut graph,
        "behavior.value",
        config.feature_dim(),
        config.network().units,
        3,
        config.value_bins,
    );
    let state = graph.input("feature", &[batch, config.feature_dim()]);
    let logits = value.forward(&mut graph, state);
    graph.set_outputs(vec![logits]);
    graph
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use meganeura::Mode;

    use super::*;
    use crate::dreamer::runtime::{build_session, configure_d3_optimizer, initialize_d3};

    #[test]
    fn behavior_graphs_have_d3_heads() {
        let config = DreamerConfig::tiny(5);
        let training = build_training_graph(&config, 12, 6);
        assert_eq!(training.outputs().len(), 6);
        let inference = build_inference_graph(&config, 7);
        assert_eq!(inference.node(inference.outputs()[0]).ty.shape, vec![7, 5]);
        assert_eq!(
            inference.node(inference.outputs()[1]).ty.shape,
            vec![7, config.value_bins]
        );
        let actor = build_actor_inference_graph(&config, 1);
        assert_eq!(actor.node(actor.outputs()[0]).ty.shape, vec![1, 5]);
        let value = build_value_inference_graph(&config, 2);
        assert_eq!(
            value.node(value.outputs()[0]).ty.shape,
            vec![2, config.value_bins]
        );
    }

    #[test]
    fn tiny_behavior_training_graph_autodiffs_and_compiles() {
        let config = DreamerConfig::tiny(3);
        let graph = build_training_graph(&config, 12, 6);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn size12m_actor_first_layer_covers_dense_and_categorical_features() {
        let mut config = DreamerConfig::new(18);
        config.learning_rate_warmup = 0;
        let rows = 1_024;
        let graph = build_training_graph(&config, rows, rows);
        let gpu = Arc::new(crate::init_gpu_context().expect("Vulkan compute device"));
        let mut session = build_session(&graph, &gpu, Mode::Training, false);
        initialize_d3(&mut session, &graph, config.seed ^ 0x5eed_0000_0000_0001);

        let network = config.network();
        let mut feature_rng = 0xd1b5_4a32_d192_ed03_u64;
        let mut feature = vec![0.0; rows * config.feature_dim()];
        for row in 0..rows {
            for column in 0..network.deter {
                feature_rng = feature_rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((feature_rng >> 40) as f32 + 0.5) / (1_u32 << 24) as f32;
                feature[row * config.feature_dim() + column] = 2.0 * unit - 1.0;
            }
            for variable in 0..network.stoch {
                feature_rng = feature_rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let class = ((feature_rng >> 32) as usize) % network.classes;
                let column = network.deter + variable * network.classes + class;
                feature[row * config.feature_dim() + column] = 1.0;
            }
        }
        let mut action_target = vec![0.0; rows * config.action_count];
        let mut target_rng = 0x94d0_49bb_1331_11eb_u64;
        for row in 0..rows {
            target_rng = target_rng
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let action = (target_rng as usize) % config.action_count;
            let signed = ((target_rng >> 32) as u32) as f32 / u32::MAX as f32;
            action_target[row * config.action_count + action] = 2.0 * signed - 1.0;
        }
        let mut value_target = vec![0.0; rows * config.value_bins];
        for row in 0..rows {
            value_target[row * config.value_bins + config.value_bins / 2 + row % 7] = 1.0;
        }
        let weights = vec![1.0; rows];

        session.set_input("imagined_feature", &feature);
        session.set_input("action_target", &action_target);
        session.set_input("imagined_weight", &weights);
        session.set_input("imagined_value_target", &value_target);
        session.set_input("imagined_slow_target", &value_target);
        session.set_input("replay_feature", &feature);
        session.set_input("replay_weight", &weights);
        session.set_input("replay_value_target", &value_target);
        session.set_input("replay_slow_target", &value_target);
        configure_d3_optimizer(&mut session, &config, 1, config.behavior_learning_rate());
        session.step();
        session.wait();

        let name = "behavior.actor.layer0.weight";
        let mut momentum = vec![0.0; session.param_size(name).expect("actor parameter")];
        session.read_adam_m(name, &mut momentum);
        assert!(momentum.iter().all(|value| value.is_finite()));
        let zero_count = momentum.iter().filter(|value| **value == 0.0).count();
        let units = network.units;
        let deterministic_end = network.deter * units;
        let deterministic_zero_count = momentum[..deterministic_end]
            .iter()
            .filter(|value| **value == 0.0)
            .count();
        let categorical_zero_count = momentum[deterministic_end..]
            .iter()
            .filter(|value| **value == 0.0)
            .count();
        let missing_hidden_channels = (0..units)
            .filter(|output| {
                (0..network.deter).all(|input| momentum[input * units + output] == 0.0)
            })
            .collect::<Vec<_>>();
        let categorical_inputs_without_gradient = momentum[deterministic_end..]
            .chunks_exact(units)
            .enumerate()
            .filter_map(|(input, values)| values.iter().all(|value| *value == 0.0).then_some(input))
            .collect::<Vec<_>>();
        assert_eq!(
            deterministic_zero_count,
            0,
            "dense deterministic features left {deterministic_zero_count} actor weights without \
             gradient; total zeros {zero_count}/{}, categorical zeros {categorical_zero_count}",
            momentum.len()
        );
        assert!(
            missing_hidden_channels.is_empty(),
            "actor gradient missed hidden channels {missing_hidden_channels:?}"
        );
        assert!(
            categorical_inputs_without_gradient.is_empty(),
            "sampled categorical inputs without any gradient: \
             {categorical_inputs_without_gradient:?}"
        );
    }
}
