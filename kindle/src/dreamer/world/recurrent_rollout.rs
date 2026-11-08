//! Two conditional recurrent-state futures, with stopped image-informed targets.

use super::*;

pub(super) struct Losses {
    pub energy: NodeId,
    pub target_distance: NodeId,
    pub sample_distance: NodeId,
    pub valid: NodeId,
    pub events: Option<[NodeId; 2]>,
    #[cfg(test)]
    states: Vec<NodeId>,
}

fn distance(graph: &mut Graph, a: NodeId, b: NodeId, width: usize) -> NodeId {
    let residual = graph.sub(a, b);
    let squared = graph.mul(residual, residual);
    let squared = graph.sum_inner(squared);
    let squared = graph.scale(squared, 1.0 / width as f32);
    let rows = graph.node(squared).ty.shape[0];
    let epsilon = graph.constant(vec![1e-6; rows], &[rows, 1]);
    let squared = graph.add(squared, epsilon);
    graph.sqrt(squared)
}

fn score(
    graph: &mut Graph,
    first: NodeId,
    second: NodeId,
    target: NodeId,
    keep: NodeId,
    width: usize,
) -> Losses {
    let target = graph.stop_gradient(target);
    let keep = graph.stop_gradient(keep);
    let a = distance(graph, first, target, width);
    let b = distance(graph, second, target, width);
    let target_distance = graph.add(a, b);
    let target_distance = graph.scale(target_distance, 0.5);
    let sample_distance = distance(graph, first, second, width);
    let half_pair = graph.scale(sample_distance, 0.5);
    let energy = graph.sub(target_distance, half_pair);
    let mut masked = |x| {
        let weighted = graph.mul(x, keep);
        graph.mean_all(weighted)
    };
    let energy = masked(energy);
    let target_distance = masked(target_distance);
    let sample_distance = masked(sample_distance);
    let valid = graph.mean_all(keep);
    Losses {
        energy,
        target_distance,
        sample_distance,
        valid,
        events: None,
        #[cfg(test)]
        states: Vec::new(),
    }
}

pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    targets: &[NodeId],
    weight_prefix: &str,
) -> Losses {
    build_with_events(
        graph,
        config,
        model,
        initial,
        actions,
        targets,
        weight_prefix,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_with_events(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    targets: &[NodeId],
    weight_prefix: &str,
    event_targets: Option<super::pixel_rollout::EventTargets<'_>>,
) -> Losses {
    assert!(!actions.is_empty() && actions.len() == targets.len());
    if let Some(events) = &event_targets {
        assert_eq!(events.reward.len(), actions.len());
        assert_eq!(events.continuation.len(), actions.len());
    }
    let size = config.network();
    let batch = config.batch_size;
    let seed_h = graph.stop_gradient(initial.0);
    let seed_z = graph.stop_gradient(initial.1);
    let mut h = [seed_h; 2];
    let mut z = [seed_z; 2];
    let mut losses = Vec::new();
    let mut event_losses = Vec::new();
    #[cfg(test)]
    let mut states = Vec::new();
    for (time, (&action, &target)) in actions.iter().zip(targets).enumerate() {
        let mut event_states = Vec::new();
        for i in 0..2 {
            h[i] = model
                .dynamics
                .core
                .forward(graph, h[i], z[i], action, batch);
            #[cfg(test)]
            states.push(h[i]);
            if time + 1 < actions.len() || event_targets.is_some() {
                let logits = model.dynamics.prior.forward(graph, h[i], batch);
                let name = if i == 0 {
                    "recurrent_first"
                } else {
                    "recurrent_second"
                };
                let uniforms = graph.input(
                    &format!("{name}_uniforms_{time}"),
                    &[batch * size.stoch, size.classes],
                );
                let sample = gumbel_sample(
                    graph,
                    logits,
                    uniforms,
                    batch * size.stoch,
                    size.classes,
                    config.unimix,
                );
                let sample = graph.stop_gradient(sample);
                let probabilities = mixed_probabilities(
                    graph,
                    logits,
                    batch * size.stoch,
                    size.classes,
                    config.unimix,
                );
                z[i] = straight_through_sample(graph, sample, probabilities);
            }
            if event_targets.is_some() {
                event_states.push(feature(graph, h[i], z[i], batch, config));
            }
        }
        let keep = graph.input(&format!("{weight_prefix}_{time}"), &[batch, 1]);
        losses.push(score(graph, h[0], h[1], target, keep, size.deter));
        if let Some(events) = &event_targets {
            for state in event_states {
                let reward = model.heads.reward.forward(graph, state);
                let continuation = model.heads.continuation.forward(graph, state);
                event_losses.push(super::pixel_rollout::event_losses(
                    graph,
                    reward,
                    continuation,
                    events.reward[time],
                    events.continuation[time],
                    keep,
                ));
            }
        }
    }
    let mut average = |field: fn(&Losses) -> NodeId| {
        let values = losses.iter().map(field).collect::<Vec<_>>();
        let total = sum(graph, &values);
        graph.scale(total, 1.0 / actions.len() as f32)
    };
    let energy = average(|x| x.energy);
    let target_distance = average(|x| x.target_distance);
    let sample_distance = average(|x| x.sample_distance);
    let valid = average(|x| x.valid);
    let events = event_targets.map(|_| {
        [0, 1].map(|i| {
            let values = event_losses.iter().map(|row| row[i]).collect::<Vec<_>>();
            let loss = sum(graph, &values);
            graph.scale(loss, 1.0 / (2 * actions.len()) as f32)
        })
    });
    Losses {
        energy,
        target_distance,
        sample_distance,
        valid,
        events,
        #[cfg(test)]
        states,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod event_tests;
