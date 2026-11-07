//! RGB errors through causal prior transitions after a real observed prefix.

use super::*;

pub(super) struct PixelRollout {
    pub loss: NodeId,
    #[cfg(test)]
    pub states: Vec<NodeId>,
}

pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    mut deter: NodeId,
    mut stoch: NodeId,
    actions: &[NodeId],
    observations: &[NodeId],
) -> PixelRollout {
    assert!(!actions.is_empty() && actions.len() == observations.len());
    let size = config.network();
    let batch = config.batch_size;
    let decoder = model
        .decoder
        .as_ref()
        .expect("RGB rollout requires decoder");
    let mut losses = Vec::new();
    #[cfg(test)]
    let mut states = Vec::new();
    for (time, (&action, &observation)) in actions.iter().zip(observations).enumerate() {
        deter = model
            .dynamics
            .core
            .forward(graph, deter, stoch, action, batch);
        let logits = model.dynamics.prior.forward(graph, deter, batch);
        let uniforms = graph.input(
            &format!("pixel_rollout_uniforms_{time}"),
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
        let state = feature(graph, deter, stoch, batch, config);
        #[cfg(test)]
        states.push(state);
        let prediction = decoder.forward(graph, state, batch);
        let target = graph.stop_gradient(observation);
        let residual = graph.sub(prediction, target);
        let squared = graph.mul(residual, residual);
        losses.push(graph.sum_all(squared));
    }
    let loss = sum(graph, &losses);
    let loss = graph.scale(loss, 1.0 / (batch * actions.len()) as f32);
    PixelRollout {
        loss,
        #[cfg(test)]
        states,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn future_images_are_targets_only_and_prior_parameters_receive_gradients() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        config.actor_critic_gradient = false;
        config.loss_scales.replay_value = 0.0;
        let mut graph = Graph::new();
        let model = WorldModel::with_behavior(&mut graph, &config, false);
        let size = config.network();
        let deter = graph.input("prefix_deter", &[config.batch_size, size.deter]);
        let stoch = graph.input(
            "prefix_stoch",
            &[config.batch_size * size.stoch, size.classes],
        );
        let actions = (0..3)
            .map(|i| {
                graph.input(
                    &format!("action{i}"),
                    &[config.batch_size, config.action_count],
                )
            })
            .collect::<Vec<_>>();
        let images = (0..3)
            .map(|i| {
                graph.input(
                    &format!("target{i}"),
                    &config.observation_shape(config.batch_size),
                )
            })
            .collect::<Vec<_>>();
        let rollout = build(&mut graph, &config, &model, deter, stoch, &actions, &images);
        for state in rollout.states {
            let mut seen = HashSet::new();
            let mut pending = vec![state];
            while let Some(node) = pending.pop() {
                if seen.insert(node) {
                    pending.extend(graph.node(node).inputs.iter().copied());
                }
            }
            assert!(images.iter().all(|image| !seen.contains(image)));
            assert!(seen.contains(&deter) && seen.contains(&stoch));
        }
        graph.set_outputs(vec![rollout.loss]);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        for prefix in [
            "world.dynamics.core.",
            "world.dynamics.prior",
            "world.decoder.",
        ] {
            assert!(
                plan.param_buffers
                    .iter()
                    .any(|(name, buffer)| name.starts_with(prefix)
                        && plan
                            .param_grad_pairs
                            .iter()
                            .any(|(parameter, _)| parameter == buffer)),
                "{prefix}"
            );
        }
    }
}
