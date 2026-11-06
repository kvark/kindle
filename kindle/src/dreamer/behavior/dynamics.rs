//! Actor gradients through a pre-update world snapshot. This is a biased
//! straight-through estimator for discrete actions and latents, not an exact
//! derivative of categorical sampling. The ordinary score-function loss stays.

use meganeura::{
    Graph,
    graph::{NodeId, Op},
};

use crate::dreamer::{
    config::DreamerConfig,
    distributions::TwoHotBins,
    networks::{
        MlpHead, Prior, RssmCore, feature, gumbel_sample, mixed_probabilities, slice_columns,
        straight_through_sample,
    },
};

/// Pair opposite support bins before summing. Uniform logits then decode to
/// exact zero even with the large raw-value endpoints of Dreamer's support.
fn scalar_mean(graph: &mut Graph, logits: NodeId, bins: &TwoHotBins) -> NodeId {
    let rows = graph.node(logits).ty.shape[0];
    let half = bins.len() / 2;
    let probabilities = graph.softmax(logits);
    let lower = slice_columns(graph, probabilities, rows, bins.len(), 0, half);
    let transposed = graph.transpose(probabilities);
    let indices = graph.constant_u32(
        &(0..half)
            .map(|i| (bins.len() - 1 - i) as u32)
            .collect::<Vec<_>>(),
        &[half],
    );
    let upper = graph.embedding(indices, transposed);
    let upper = graph.transpose(upper);
    let difference = graph.sub(upper, lower);
    let support = graph.constant(bins.values()[..half].iter().map(|x| -x).collect(), &[half]);
    let weighted = graph.bias_mul(difference, support);
    graph.sum_inner(weighted)
}

fn sample(
    graph: &mut Graph,
    logits: NodeId,
    uniforms: NodeId,
    rows: usize,
    classes: usize,
    unimix: f32,
) -> NodeId {
    let hard = gumbel_sample(graph, logits, uniforms, rows, classes, unimix);
    let probabilities = mixed_probabilities(graph, logits, rows, classes, unimix);
    straight_through_sample(graph, hard, probabilities)
}

/// r[t+1] and c[t+1] belong to a[t]. Continuation already contains the
/// configured time discount, just as in the existing imagined targets.
fn lambda_objective(
    graph: &mut Graph,
    reward: &[NodeId],
    continuation: &[NodeId],
    value: &[NodeId],
    weight: NodeId,
    return_scale: NodeId,
    lambda: f32,
) -> NodeId {
    let horizon = reward.len();
    assert!(horizon > 0);
    assert_eq!(continuation.len(), horizon);
    assert_eq!(value.len(), horizon);
    let rows = graph.node(reward[0]).ty.shape[0];
    let mut next = value[horizon - 1];
    let mut losses = Vec::with_capacity(horizon);
    for time in (0..horizon).rev() {
        let trace = graph.scale(next, lambda);
        let bootstrap = graph.scale(value[time], 1.0 - lambda);
        let future = graph.add(trace, bootstrap);
        let future = graph.mul(continuation[time], future);
        next = graph.add(reward[time], future);
        let weight = slice_columns(graph, weight, 1, rows * horizon, time * rows, rows);
        let weight = graph.reshape(weight, &[rows, 1]);
        let weight = graph.stop_gradient(weight);
        let weighted = graph.mul(weight, next);
        losses.push(graph.mean_all(weighted));
    }
    let total = super::sum(graph, &losses);
    let return_scale = graph.stop_gradient(return_scale);
    let normalized = graph.div(total, return_scale);
    graph.scale(normalized, -1.0 / horizon as f32)
}

pub(super) fn loss(
    graph: &mut Graph,
    config: &DreamerConfig,
    actor: &MlpHead,
    value: &MlpHead,
    weight: NodeId,
) -> NodeId {
    let rows = config.batch_size * config.batch_length;
    let horizon = config.imagination_length;
    let size = config.network();
    let core = RssmCore::new_frozen(graph, config);
    let prior = Prior::new_frozen(graph, config);
    let reward_head = MlpHead::new(
        graph,
        "world.reward",
        config.feature_dim(),
        size.units,
        1,
        config.value_bins,
    );
    let continuation_head = MlpHead::new(
        graph,
        "world.continuation",
        config.feature_dim(),
        size.units,
        1,
        1,
    );
    let bins = TwoHotBins::new(config.value_bins);
    let deter = graph.input("dynamics_deter", &[rows, size.deter]);
    let stoch = graph.input("dynamics_stoch", &[rows * size.stoch, size.classes]);
    let mut deter = graph.stop_gradient(deter);
    let mut stoch = graph.stop_gradient(stoch);
    assert_eq!(graph.node(weight).ty.shape, [rows * horizon, 1]);
    let return_scale = graph.input("dynamics_return_scale", &[1]);
    let bonus = config.uses_disagreement().then(|| {
        let bonus = graph.input("dynamics_bonus", &[rows * horizon, 1]);
        graph.stop_gradient(bonus)
    });
    let mut rewards = Vec::with_capacity(horizon);
    let mut continuations = Vec::with_capacity(horizon);
    let mut values = Vec::with_capacity(horizon);
    for time in 0..horizon {
        let state = feature(graph, deter, stoch, rows, config);
        let logits = actor.forward(graph, state);
        let uniforms = graph.input(
            &format!("dynamics_action_uniforms_{time}"),
            &[rows, config.action_count],
        );
        let action = sample(
            graph,
            logits,
            uniforms,
            rows,
            config.action_count,
            config.actor_unimix,
        );
        deter = core.forward(graph, deter, stoch, action, rows);
        let logits = prior.forward(graph, deter, rows);
        let uniforms = graph.input(
            &format!("dynamics_latent_uniforms_{time}"),
            &[rows * size.stoch, size.classes],
        );
        stoch = sample(
            graph,
            logits,
            uniforms,
            rows * size.stoch,
            size.classes,
            config.unimix,
        );
        let arrival = feature(graph, deter, stoch, rows, config);
        let reward = reward_head.forward_frozen(graph, arrival);
        let mut reward = scalar_mean(graph, reward, &bins);
        if let Some(bonus) = bonus {
            let bonus = slice_columns(graph, bonus, 1, rows * horizon, time * rows, rows);
            let bonus = graph.reshape(bonus, &[rows, 1]);
            let bonus = graph.scale(bonus, config.intrinsic_reward_scale);
            reward = graph.add(reward, bonus);
        }
        rewards.push(reward);
        let continuation = continuation_head.forward_frozen(graph, arrival);
        continuations.push(graph.sigmoid(continuation));
        let value = value.forward_frozen(graph, arrival);
        values.push(scalar_mean(graph, value, &bins));
    }
    // World weights are private snapshot inputs, never optimizer parameters.
    // Reclassifying leaves preserves node order and every forward dependency.
    for node in graph.nodes_mut() {
        if let Op::Parameter { name } = &node.op
            && name.starts_with("world.")
        {
            node.op = Op::Input {
                name: format!("dynamics_parameter_{name}"),
            };
        }
    }
    lambda_objective(
        graph,
        &rewards,
        &continuations,
        &values,
        weight,
        return_scale,
        config.lambda,
    )
}

#[cfg(test)]
mod tests;
