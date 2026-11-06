//! Policy expectation over every discrete action with detached sampled values.

use meganeura::{
    Graph,
    graph::{NodeId, Op},
};

use crate::dreamer::{
    config::DreamerConfig,
    distributions::TwoHotBins,
    exploration::Disagreement,
    networks::{
        MlpHead, Prior, RssmCore, feature, gumbel_sample, mixed_probabilities, repeat_rows,
        slice_columns,
    },
};

fn branch_inputs(
    graph: &mut Graph,
    config: &DreamerConfig,
) -> (NodeId, NodeId, NodeId, NodeId, NodeId) {
    let rows = config.batch_size * config.batch_length;
    let size = config.network();
    let actions = config.action_count;
    let latent = size.stoch * size.classes;
    let deter = graph.input("expected_deter", &[rows, size.deter]);
    let stoch = graph.input("expected_stoch", &[rows * size.stoch, size.classes]);
    let uniforms = graph.input(
        "expected_latent_uniforms",
        &[rows * size.stoch, size.classes],
    );
    let start = feature(graph, deter, stoch, rows, config);
    let start = graph.stop_gradient(start);
    let deter = repeat_rows(graph, deter, actions);
    let stoch = graph.reshape(stoch, &[rows, latent]);
    let stoch = repeat_rows(graph, stoch, actions);
    let stoch = graph.reshape(stoch, &[rows * actions * size.stoch, size.classes]);
    let uniforms = graph.reshape(uniforms, &[rows, latent]);
    let uniforms = repeat_rows(graph, uniforms, actions);
    let uniforms = graph.reshape(uniforms, &[rows * actions * size.stoch, size.classes]);
    let one_hot = (0..rows * actions * actions)
        .map(|i| {
            if i / actions % actions == i % actions {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let action = graph.constant(one_hot, &[rows * actions, actions]);
    (start, deter, stoch, action, uniforms)
}

fn objective(
    graph: &mut Graph,
    logits: NodeId,
    values: NodeId,
    weight: NodeId,
    return_scale: NodeId,
    unimix: f32,
) -> NodeId {
    let shape = graph.node(logits).ty.shape.clone();
    let (rows, actions) = (shape[0], shape[1]);
    assert_eq!(graph.node(values).ty.shape, shape);
    assert_eq!(graph.node(weight).ty.shape, [rows, 1]);
    let values = graph.stop_gradient(values);
    let mean = graph.sum_inner(values);
    let mean = graph.scale(mean, -1.0 / actions as f32);
    let ones = graph.constant(vec![1.0; actions], &[1, actions]);
    let negative_mean = graph.matmul(mean, ones);
    let centered = graph.add(values, negative_mean);
    let probabilities = mixed_probabilities(graph, logits, rows, actions, unimix);
    let expected = graph.mul(probabilities, centered);
    let expected = graph.sum_inner(expected);
    let weight = graph.stop_gradient(weight);
    let weighted = graph.mul(expected, weight);
    let total = graph.mean_all(weighted);
    let return_scale = graph.stop_gradient(return_scale);
    let normalized = graph.div(total, return_scale);
    graph.neg(normalized)
}

fn values(graph: &mut Graph, config: &DreamerConfig, value: &MlpHead) -> (NodeId, NodeId) {
    let rows = config.batch_size * config.batch_length;
    let branches = rows * config.action_count;
    let size = config.network();
    let core = RssmCore::new_frozen(graph, config);
    let prior = Prior::new_frozen(graph, config);
    let reward = MlpHead::new(
        graph,
        "world.reward",
        config.feature_dim(),
        size.units,
        1,
        config.value_bins,
    );
    let continuation = MlpHead::new(
        graph,
        "world.continuation",
        config.feature_dim(),
        size.units,
        1,
        1,
    );
    let (start, deter, stoch, action, uniforms) = branch_inputs(graph, config);
    let next_deter = core.forward(graph, deter, stoch, action, branches);
    let prior_logits = prior.forward(graph, next_deter, branches);
    let next_stoch = gumbel_sample(
        graph,
        prior_logits,
        uniforms,
        branches * size.stoch,
        size.classes,
        config.unimix,
    );
    let arrival = feature(graph, next_deter, next_stoch, branches, config);
    let bins = TwoHotBins::new(config.value_bins);
    let reward_logits = reward.forward_frozen(graph, arrival);
    let mut reward = super::dynamics::scalar_mean(graph, reward_logits, &bins);
    if config.uses_disagreement() {
        let ensemble = Disagreement::new(graph, config);
        let repeated_start = repeat_rows(graph, start, config.action_count);
        let bonus = ensemble.bonus(graph, repeated_start, action);
        let bonus = graph.scale(bonus, config.intrinsic_reward_scale);
        reward = graph.add(reward, bonus);
    }
    let value_logits = value.forward_frozen(graph, arrival);
    let value = super::dynamics::scalar_mean(graph, value_logits, &bins);
    let continuation_logits = continuation.forward_frozen(graph, arrival);
    let continuation = graph.sigmoid(continuation_logits);
    let future = graph.mul(continuation, value);
    let values = graph.add(reward, future);
    let values = graph.reshape(values, &[rows, config.action_count]);
    (start, values)
}

pub(super) fn loss(
    graph: &mut Graph,
    config: &DreamerConfig,
    actor: &MlpHead,
    value: &MlpHead,
    weight: NodeId,
) -> NodeId {
    let rows = config.batch_size * config.batch_length;
    let (start, values) = values(graph, config, value);
    let logits = actor.forward(graph, start);
    let weight = slice_columns(graph, weight, 1, rows * config.imagination_length, 0, rows);
    let weight = graph.reshape(weight, &[rows, 1]);
    let return_scale = graph.input("expected_return_scale", &[1]);
    for node in graph.nodes_mut() {
        if let Op::Parameter { name } = &node.op
            && name.starts_with("world.")
        {
            node.op = Op::Input {
                name: format!("expected_parameter_{name}"),
            };
        }
    }
    objective(
        graph,
        logits,
        values,
        weight,
        return_scale,
        config.actor_unimix,
    )
}

#[cfg(test)]
pub(crate) fn build_values_graph(config: &DreamerConfig) -> Graph {
    let mut graph = Graph::new();
    let value = MlpHead::new(
        &mut graph,
        "behavior.value",
        config.feature_dim(),
        config.network().units,
        3,
        config.value_bins,
    );
    let (start, values) = values(&mut graph, config, &value);
    graph.set_outputs(vec![start, values]);
    graph
}

#[cfg(test)]
mod tests;
