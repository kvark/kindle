//! Causal multi-step priors matched to detached, image-informed posteriors.

use super::*;

fn masked_kl(
    graph: &mut Graph,
    config: &DreamerConfig,
    target: NodeId,
    prior: NodeId,
    keep: NodeId,
) -> NodeId {
    let size = config.network();
    let batch = config.batch_size;
    let target = graph.stop_gradient(target);
    let posterior = mixed_probabilities(
        graph,
        target,
        batch * size.stoch,
        size.classes,
        config.unimix,
    );
    let prior = mixed_probabilities(
        graph,
        prior,
        batch * size.stoch,
        size.classes,
        config.unimix,
    );
    let posterior_log = graph.log(posterior);
    let prior_log = graph.log(prior);
    let ratio = graph.sub(posterior_log, prior_log);
    let elements = graph.mul(posterior, ratio);
    let per_variable = graph.sum_inner(elements);
    let per_variable = graph.reshape(per_variable, &[batch, size.stoch]);
    let per_state = graph.sum_inner(per_variable);
    let keep = graph.stop_gradient(keep);
    let masked = graph.mul(per_state, keep);
    graph.mean_all(masked)
}

pub(super) struct LatentRollout {
    pub loss: NodeId,
    #[cfg(test)]
    priors: Vec<NodeId>,
}

pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    targets: &[NodeId],
) -> LatentRollout {
    assert!(!actions.is_empty() && actions.len() == targets.len());
    let (mut deter, mut stoch) = (
        graph.stop_gradient(initial.0),
        graph.stop_gradient(initial.1),
    );
    let size = config.network();
    let batch = config.batch_size;
    let mut losses = Vec::new();
    #[cfg(test)]
    let mut priors = Vec::new();
    for (time, (&action, &target)) in actions.iter().zip(targets).enumerate() {
        deter = model
            .dynamics
            .core
            .forward(graph, deter, stoch, action, batch);
        let logits = model.dynamics.prior.forward(graph, deter, batch);
        #[cfg(test)]
        priors.push(logits);
        let keep = graph.input(&format!("pixel_rollout_weight_{time}"), &[batch, 1]);
        losses.push(masked_kl(graph, config, target, logits, keep));
        if time + 1 < actions.len() {
            let uniforms = graph.input(
                &format!("latent_rollout_uniforms_{time}"),
                &[batch * size.stoch, size.classes],
            );
            let sampled = gumbel_sample(
                graph,
                logits,
                uniforms,
                batch * size.stoch,
                size.classes,
                config.unimix,
            );
            let sampled = graph.stop_gradient(sampled);
            let probabilities = mixed_probabilities(
                graph,
                logits,
                batch * size.stoch,
                size.classes,
                config.unimix,
            );
            stoch = straight_through_sample(graph, sampled, probabilities);
        }
    }
    let loss = sum(graph, &losses);
    let loss = graph.scale(loss, 1.0 / actions.len() as f32);
    LatentRollout {
        loss,
        #[cfg(test)]
        priors,
    }
}

#[cfg(test)]
mod tests;
