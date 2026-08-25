//! DreamerV3 actor and distributional value model.

use meganeura::{Graph, graph::NodeId};

use super::config::DreamerConfig;
use super::networks::{MlpHead, scale, sum};

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
    let actor_update_scale = graph.input("actor_update_scale", &[1]);
    let weighted_policy = graph.mul(weighted_policy, actor_update_scale);
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
    use super::*;

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
}
