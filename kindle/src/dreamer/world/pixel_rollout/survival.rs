//! Cumulative continuation scores; labels and masks never change a rollout state.
use super::*;

pub(super) fn build(
    graph: &mut Graph,
    logits: &[NodeId],
    targets: &[NodeId],
    weights: &[NodeId],
    gamma: f32,
    batch: usize,
) -> NodeId {
    assert!(!logits.is_empty() && logits.len() == targets.len() && logits.len() == weights.len());
    assert!(gamma > 0.0 && gamma <= 1.0 && batch > 0);
    let discount = graph.constant(vec![gamma; batch], &[batch, 1]);
    let mut predicted = graph.constant(vec![1.0; batch], &[batch, 1]);
    let mut actual = predicted;
    let mut losses = Vec::with_capacity(logits.len());
    for ((&logit, &target), &weight) in logits.iter().zip(targets).zip(weights) {
        let probability = graph.sigmoid(logit);
        let probability = graph.div(probability, discount);
        let probability = graph.clamp(probability, 0.0, 1.0);
        predicted = graph.mul(predicted, probability);
        let target = graph.stop_gradient(target);
        let target = graph.div(target, discount);
        let target = graph.clamp(target, 0.0, 1.0);
        actual = graph.mul(actual, target);
        let error = graph.sub(predicted, actual);
        let squared = graph.mul(error, error);
        let weight = graph.stop_gradient(weight);
        let weighted = graph.mul(squared, weight);
        losses.push(graph.mean_all(weighted));
    }
    let loss = sum(graph, &losses);
    graph.scale(loss, 1.0 / logits.len() as f32)
}

#[cfg(test)]
mod tests;
