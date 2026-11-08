//! Proper reward and continuation losses on self-observed imagined states.
use super::*;

pub(super) fn event_losses(
    graph: &mut Graph,
    reward: NodeId,
    continuation: NodeId,
    reward_target: NodeId,
    continuation_target: NodeId,
    weight: NodeId,
) -> [NodeId; 2] {
    let weight = graph.stop_gradient(weight);
    let reward_target = graph.stop_gradient(reward_target);
    let continuation_target = graph.stop_gradient(continuation_target);
    let reward = weighted_cross_entropy(graph, reward, reward_target, weight);
    let normalizer = graph.softplus(continuation, 1.0);
    let selected = graph.mul(continuation, continuation_target);
    let error = graph.sub(normalizer, selected);
    let error = graph.mul(error, weight);
    [reward, graph.mean_all(error)]
}

#[cfg(test)]
mod tests;
