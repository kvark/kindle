//! Outcome-conditioned reconstruction through the decoder's own image memory.
use super::*;

pub(super) struct Sequence<'a> {
    pub posterior: &'a [NodeId],
    pub actions: &'a [NodeId],
    pub observations: &'a [NodeId],
    pub keeps: &'a [NodeId],
    pub history: NodeId,
}

pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    sequence: Sequence<'_>,
) -> pixel_rollout::PixelRollout {
    let Sequence {
        posterior,
        actions,
        observations,
        keeps,
        mut history,
    } = sequence;
    assert!(!posterior.is_empty());
    assert_eq!(posterior.len(), actions.len());
    assert_eq!(keeps.len(), actions.len());
    assert_eq!(observations.len(), actions.len() + 1);
    let batch = config.batch_size;
    let width = config.observation_dim();
    let decoder = model.decoder.as_ref().unwrap();
    let mut losses = Vec::new();
    #[cfg(test)]
    let mut predictions = Vec::new();
    for (t, &state) in posterior.iter().enumerate() {
        let keep = graph.stop_gradient(keeps[t]);
        let keep = graph.broadcast_inner(keep, 2 * width);
        history = graph.mul(history, keep);
        let logits =
            decoder.logits_with_action_history(graph, state, history, Some(actions[t]), batch);
        let image = super::super::networks::centered_pixels(graph, logits);
        history = super::super::spatial_history::shift(graph, history, image, batch, width);
        #[cfg(test)]
        predictions.push(image);
        let weight = graph.input(&format!("pixel_rollout_weight_{t}"), &[batch, 1]);
        let errors =
            super::super::networks::pixel_cross_entropy(graph, logits, observations[t + 1]);
        let motion = super::super::networks::pixel_motion_weights(
            graph,
            observations[t + 1],
            observations[t],
            weight,
        );
        let weighted = graph.mul(errors, motion);
        let per_row = graph.sum_inner(weighted);
        let weight = graph.stop_gradient(weight);
        let weighted = graph.mul(per_row, weight);
        losses.push(graph.sum_all(weighted));
    }
    let loss = sum(graph, &losses);
    let loss = graph.scale(loss, 1.0 / (batch * actions.len()) as f32);
    pixel_rollout::PixelRollout {
        loss,
        events: None,
        centroid: None,
        #[cfg(test)]
        states: posterior.to_vec(),
        #[cfg(test)]
        predictions,
    }
}

#[cfg(test)]
mod tests;
