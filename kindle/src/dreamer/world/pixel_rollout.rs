//! RGB errors through causal prior transitions after a real observed prefix.

use super::*;

pub(super) struct PixelRollout {
    pub loss: NodeId,
    #[cfg(test)]
    pub states: Vec<NodeId>,
}

fn motion_loss(
    graph: &mut Graph,
    errors: NodeId,
    current: NodeId,
    previous: NodeId,
    keep: NodeId,
) -> NodeId {
    let weights = super::super::networks::pixel_motion_weights(graph, current, previous, keep);
    let weighted = graph.mul(errors, weights);
    let per_row = graph.sum_inner(weighted);
    let keep = graph.stop_gradient(keep);
    let weighted = graph.mul(per_row, keep);
    graph.sum_all(weighted)
}

pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    observations: &[NodeId],
    pixel_loss: PixelLoss,
) -> PixelRollout {
    let (mut deter, mut stoch) = initial;
    assert!(!actions.is_empty() && actions.len() + 1 == observations.len());
    let size = config.network();
    let batch = config.batch_size;
    let decoder = model
        .decoder
        .as_ref()
        .expect("RGB rollout requires decoder");
    let mut losses = Vec::new();
    #[cfg(test)]
    let mut states = Vec::new();
    for (time, (&action, pair)) in actions.iter().zip(observations.windows(2)).enumerate() {
        let (previous, observation) = (pair[0], pair[1]);
        deter = model
            .dynamics
            .core
            .forward(graph, deter, stoch, action, batch);
        let logits = model.dynamics.prior.forward(graph, deter, batch);
        let uniforms = graph.input(
            &format!("pixel_rollout_uniforms_{time}"),
            &[batch * size.stoch, size.classes],
        );
        stoch = if config.world_gaussian {
            super::super::gaussian::sample(
                graph,
                logits,
                uniforms,
                batch * size.stoch,
                size.classes,
            )
        } else {
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
            straight_through_sample(graph, sampled, probabilities)
        };
        let state = feature(graph, deter, stoch, batch, config);
        #[cfg(test)]
        states.push(state);
        let squared = if pixel_loss != PixelLoss::SquaredError {
            decoder.cross_entropy(graph, state, observation, batch)
        } else {
            let prediction = decoder.forward(graph, state, batch);
            let target = graph.stop_gradient(observation);
            let residual = graph.sub(prediction, target);
            graph.mul(residual, residual)
        };
        let loss = if pixel_loss.motion_attention() {
            let keep = graph.input(&format!("pixel_rollout_weight_{time}"), &[batch, 1]);
            motion_loss(graph, squared, observation, previous, keep)
        } else {
            let per_row = graph.sum_inner(squared);
            let weight = graph.input(&format!("pixel_rollout_weight_{time}"), &[batch, 1]);
            let weight = graph.stop_gradient(weight);
            let weighted = graph.mul(per_row, weight);
            graph.sum_all(weighted)
        };
        losses.push(loss);
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
mod motion_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn future_images_are_targets_only_and_prior_parameters_receive_gradients() {
        for (pixel_loss, gaussian) in [
            (PixelLoss::SquaredError, false),
            (PixelLoss::CrossEntropy, false),
            (PixelLoss::MotionCrossEntropy, false),
            (PixelLoss::MotionFixedTeacherImage, false),
            (PixelLoss::MotionCrossEntropy, true),
        ] {
            let mut config = DreamerConfig::tiny(3);
            config.observation_kind = ObservationKind::Rgb64;
            config.actor_critic_gradient = false;
            config.world_gaussian = gaussian;
            config.loss_scales.policy = 0.0;
            config.loss_scales.value = 0.0;
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
            let images = (0..4)
                .map(|i| {
                    graph.input(
                        &format!("target{i}"),
                        &config.observation_shape(config.batch_size),
                    )
                })
                .collect::<Vec<_>>();
            let rollout = build(
                &mut graph,
                &config,
                &model,
                (deter, stoch),
                &actions,
                &images,
                pixel_loss,
            );
            for state in rollout.states {
                let mut seen = HashSet::new();
                let mut pending = vec![state];
                while let Some(node) = pending.pop() {
                    if seen.insert(node) {
                        pending.extend(graph.node(node).inputs.iter().copied());
                    }
                }
                assert!(images.iter().all(|image| !seen.contains(image)));
                assert!(
                    seen.iter().all(|&id| !matches!(&graph.node(id).op,
                meganeura::graph::Op::Input { name } if name.starts_with("pixel_rollout_weight_")))
                );
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
}
