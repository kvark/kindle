//! RGB errors through causal prior transitions after a real observed prefix.

use super::*;

pub(super) struct PixelRollout {
    pub loss: NodeId,
    pub events: Option<[NodeId; 2]>,
    #[cfg(test)]
    pub states: Vec<NodeId>,
    #[cfg(test)]
    pub predictions: Vec<NodeId>,
}

pub(super) struct EventTargets<'a> {
    pub reward: &'a [NodeId],
    pub continuation: &'a [NodeId],
}

mod feedback;

pub(super) fn event_losses(
    graph: &mut Graph,
    reward: NodeId,
    continuation: NodeId,
    reward_target: NodeId,
    continuation_target: NodeId,
    weight: NodeId,
) -> [NodeId; 2] {
    feedback::event_losses(
        graph,
        reward,
        continuation,
        reward_target,
        continuation_target,
        weight,
    )
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

#[cfg(test)]
pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    observations: &[NodeId],
    pixel_loss: PixelLoss,
) -> PixelRollout {
    build_with_history(
        graph,
        config,
        model,
        initial,
        actions,
        observations,
        pixel_loss,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn build_with_history(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    observations: &[NodeId],
    pixel_loss: PixelLoss,
    history: Option<NodeId>,
) -> PixelRollout {
    build_with_feedback(
        graph,
        config,
        model,
        initial,
        actions,
        observations,
        pixel_loss,
        history,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_with_feedback(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    initial: (NodeId, NodeId),
    actions: &[NodeId],
    observations: &[NodeId],
    pixel_loss: PixelLoss,
    mut history: Option<NodeId>,
    targets: Option<EventTargets<'_>>,
) -> PixelRollout {
    assert_eq!(config.rgb_spatial_decoder, history.is_some());
    assert_eq!(pixel_loss.rollout_events(), targets.is_some());
    if let Some(targets) = &targets {
        assert_eq!(targets.reward.len(), actions.len());
        assert_eq!(targets.continuation.len(), actions.len());
    }
    let (mut deter, mut stoch) = initial;
    assert!(!actions.is_empty() && actions.len() + 1 == observations.len());
    let size = config.network();
    let batch = config.batch_size;
    let decoder = model
        .decoder
        .as_ref()
        .expect("RGB rollout requires decoder");
    let mut losses = Vec::new();
    let mut event_losses = Vec::new();
    #[cfg(test)]
    let mut states = Vec::new();
    #[cfg(test)]
    let mut predictions = Vec::new();
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
        let squared = if let Some(past) = history {
            let logits = decoder.logits_with_history(graph, state, past, batch);
            let next = super::super::networks::centered_pixels(graph, logits);
            #[cfg(test)]
            predictions.push(next);
            history = Some(super::super::spatial_history::shift(
                graph,
                past,
                next,
                batch,
                config.observation_dim(),
            ));
            if config.world_image_feedback {
                let encoded = model.representation.encoder.forward(graph, next, batch);
                let logits = model
                    .representation
                    .posterior
                    .forward(graph, deter, encoded, batch);
                let noise = graph.input(
                    &format!("pixel_feedback_uniforms_{time}"),
                    &[batch * size.stoch, size.classes],
                );
                let sampled = gumbel_sample(
                    graph,
                    logits,
                    noise,
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
            super::super::networks::pixel_cross_entropy(graph, logits, observation)
        } else if pixel_loss != PixelLoss::SquaredError {
            decoder.cross_entropy(graph, state, observation, batch)
        } else {
            let prediction = decoder.forward(graph, state, batch);
            let target = graph.stop_gradient(observation);
            let residual = graph.sub(prediction, target);
            graph.mul(residual, residual)
        };
        let belief = if config.world_image_feedback {
            feature(graph, deter, stoch, batch, config)
        } else {
            state
        };
        #[cfg(test)]
        states.push(belief);
        // Read targets only after the complete causal state and image exist.
        let mut event_weight = None;
        if let Some(targets) = &targets {
            let weight = graph.input(&format!("pixel_rollout_weight_{time}"), &[batch, 1]);
            event_weight = Some(weight);
            let reward = model.heads.reward.forward(graph, belief);
            let continuation = model.heads.continuation.forward(graph, belief);
            event_losses.push(feedback::event_losses(
                graph,
                reward,
                continuation,
                targets.reward[time],
                targets.continuation[time],
                weight,
            ));
        }
        let loss = if pixel_loss.motion_attention() {
            let weight = event_weight.unwrap_or_else(|| {
                graph.input(&format!("pixel_rollout_weight_{time}"), &[batch, 1])
            });
            motion_loss(graph, squared, observation, previous, weight)
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
    let events = targets.map(|_| {
        [0, 1].map(|i| {
            let values = event_losses.iter().map(|row| row[i]).collect::<Vec<_>>();
            let loss = sum(graph, &values);
            graph.scale(loss, 1.0 / actions.len() as f32)
        })
    });
    PixelRollout {
        loss,
        events,
        #[cfg(test)]
        states,
        #[cfg(test)]
        predictions,
    }
}

#[cfg(test)]
mod image_feedback_tests;
#[cfg(test)]
mod motion_tests;
#[cfg(test)]
mod suffix_event_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn spatial_forecasts_use_own_images_without_future_targets_or_reset_masks() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.rgb_spatial_decoder = true;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        let mut graph = Graph::new();
        let model = WorldModel::with_behavior(&mut graph, &config, false);
        let size = config.network();
        let deter = graph.input("seed_deter", &[config.batch_size, size.deter]);
        let stoch = graph.input(
            "seed_stoch",
            &[config.batch_size * size.stoch, size.classes],
        );
        let history = graph.input(
            "seed_history",
            &[config.batch_size, 2 * config.observation_dim()],
        );
        let images = (0..4)
            .map(|i| {
                graph.input(
                    &format!("target{i}"),
                    &[config.batch_size, config.observation_dim()],
                )
            })
            .collect::<Vec<_>>();
        let actions = (0..3)
            .map(|i| {
                graph.input(
                    &format!("action{i}"),
                    &[config.batch_size, config.action_count],
                )
            })
            .collect::<Vec<_>>();
        let rollout = build_with_history(
            &mut graph,
            &config,
            &model,
            (deter, stoch),
            &actions,
            &images,
            PixelLoss::MotionCrossEntropy,
            Some(history),
        );
        assert_eq!(rollout.predictions.len(), 3);
        for (i, &prediction) in rollout.predictions.iter().enumerate() {
            let mut seen = HashSet::new();
            let mut pending = vec![prediction];
            while let Some(node) = pending.pop() {
                if seen.insert(node) {
                    pending.extend(graph.node(node).inputs.iter().copied());
                }
            }
            assert!(images.iter().all(|n| !seen.contains(n)));
            assert!(seen.contains(&history) && seen.contains(&deter) && seen.contains(&stoch));
            assert!(seen.iter().all(|&n| !matches!(&graph.node(n).op,meganeura::graph::Op::Input{name} if name.starts_with("pixel_rollout_weight_"))));
            if i > 0 {
                assert!(seen.contains(&rollout.predictions[i - 1]));
            }
        }
        graph.set_outputs(vec![rollout.loss]);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        for prefix in [
            "world.dynamics.core.",
            "world.dynamics.prior",
            "world.decoder.motion0.",
            "world.decoder.motion1.",
            "world.decoder.motion2.",
            "world.decoder.imgout.",
        ] {
            assert!(
                plan.param_buffers
                    .iter()
                    .any(|(name, buffer)| name.starts_with(prefix)
                        && plan.param_grad_pairs.iter().any(|(p, _)| p == buffer)),
                "{prefix}"
            );
        }
    }

    #[test]
    fn future_images_are_targets_only_and_prior_parameters_receive_gradients() {
        for (pixel_loss, gaussian, dim) in [
            (PixelLoss::SquaredError, false, None),
            (PixelLoss::CrossEntropy, false, None),
            (PixelLoss::MotionCrossEntropy, false, None),
            (PixelLoss::MotionFixedTeacherImage, false, None),
            (PixelLoss::MotionCrossEntropy, true, None),
            (PixelLoss::MotionCrossEntropy, true, Some(4)),
        ] {
            let mut config = DreamerConfig::tiny(3);
            config.observation_kind = ObservationKind::Rgb64;
            config.actor_critic_gradient = false;
            config.world_gaussian = gaussian;
            config.world_gaussian_dim = dim;
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
