//! Direct one-step world targets from a prior that has not observed the arrival.

use super::*;

pub(super) struct Losses {
    pub image: NodeId,
    pub reward: NodeId,
    pub continuation: NodeId,
    pub valid: NodeId,
}

impl Losses {
    pub(super) fn scaled(&self, graph: &mut Graph, config: &DreamerConfig) -> NodeId {
        let image = graph.scale(self.image, config.loss_scales.reconstruction);
        let reward = graph.scale(self.reward, config.loss_scales.reward);
        let continuation = graph.scale(self.continuation, config.loss_scales.continuation);
        let total = graph.add(image, reward);
        graph.add(total, continuation)
    }
}

pub(super) fn image_loss(
    graph: &mut Graph,
    elements: NodeId,
    target: NodeId,
    previous: NodeId,
    keep: NodeId,
) -> NodeId {
    let weights = super::super::networks::pixel_motion_weights(graph, target, previous, keep);
    let elements = graph.mul(elements, weights);
    let per_row = graph.sum_inner(elements);
    let keep = graph.stop_gradient(keep);
    let masked = graph.mul(per_row, keep);
    graph.mean_all(masked)
}

fn head_losses(
    graph: &mut Graph,
    reward_logits: NodeId,
    continuation_logits: NodeId,
    reward_target: NodeId,
    continuation_target: NodeId,
    keep: NodeId,
) -> (NodeId, NodeId) {
    let reward_target = graph.stop_gradient(reward_target);
    let continuation_target = graph.stop_gradient(continuation_target);
    let reward = weighted_cross_entropy(graph, reward_logits, reward_target, keep);
    let continuation =
        prior_continuation::loss(graph, continuation_logits, continuation_target, keep);
    (reward, continuation)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn losses(
    graph: &mut Graph,
    model: &WorldModel,
    state: NodeId,
    target: NodeId,
    previous: NodeId,
    keep_action: NodeId,
    reward_target: NodeId,
    continuation_target: NodeId,
    rows: usize,
) -> Losses {
    let keep = graph.mean_inner(keep_action);
    let keep = graph.stop_gradient(keep);
    let elements = model
        .decoder
        .as_ref()
        .unwrap()
        .cross_entropy(graph, state, target, rows);
    let image = image_loss(graph, elements, target, previous, keep);
    let reward_logits = model.heads.reward.forward(graph, state);
    let logits = model.heads.continuation.forward(graph, state);
    let (reward, continuation) = head_losses(
        graph,
        reward_logits,
        logits,
        reward_target,
        continuation_target,
        keep,
    );
    let valid = graph.mean_all(keep);
    Losses {
        image,
        reward,
        continuation,
        valid,
    }
}

pub(super) fn average(graph: &mut Graph, values: &[Losses]) -> Losses {
    let mut mean = |select: fn(&Losses) -> NodeId| {
        let nodes = values.iter().map(select).collect::<Vec<_>>();
        let total = sum(graph, &nodes);
        graph.scale(total, 1.0 / values.len() as f32)
    };
    let image = mean(|v| v.image);
    let reward = mean(|v| v.reward);
    let continuation = mean(|v| v.continuation);
    let valid = mean(|v| v.valid);
    Losses {
        image,
        reward,
        continuation,
        valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meganeura::{
        graph::Op,
        reference::{Feeds, evaluate},
    };
    use std::collections::HashSet;

    fn ancestors(graph: &Graph, root: NodeId) -> HashSet<NodeId> {
        let mut seen = HashSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if seen.insert(id) {
                pending.extend(&graph.node(id).inputs);
            }
        }
        seen
    }

    #[test]
    fn predictive_image_loss_matches_f64_and_masks_missing_transitions() {
        use super::super::super::networks::pixel_cross_entropy;
        for mask in [[0., 1.], [0., 0.]] {
            let mut graph = Graph::new();
            let logits = graph.parameter("logits", &[2, 4]);
            let target = graph.parameter("target", &[2, 4]);
            let previous = graph.parameter("previous", &[2, 4]);
            let keep = graph.constant(mask.to_vec(), &[2, 1]);
            let elements = pixel_cross_entropy(&mut graph, logits, target);
            let loss = image_loss(&mut graph, elements, target, previous, keep);
            graph.set_outputs(vec![loss]);
            let backward = meganeura::autodiff::differentiate(&graph);
            let x = [-2., -1., 1., 2., -3., -0.5, 0.5, 3.];
            let y = [-0.5, -0.2, 0.2, 0.5, -0.3, -0.1, 0.1, 0.3];
            let before = [0.; 8];
            let mut feeds = Feeds::new();
            feeds
                .set("logits", &x)
                .set("target", &y)
                .set("previous", &before);
            let actual = evaluate(&backward, &feeds).unwrap();
            let mut expected = 0.;
            let mut gradients = Vec::new();
            for row in 0..2 {
                let weights = (0..4)
                    .map(|i| 1. + 100. * f64::from(y[row * 4 + i]).abs())
                    .collect::<Vec<_>>();
                let mean = weights.iter().sum::<f64>() / 4.;
                for (column, weight) in weights.iter().enumerate() {
                    let index = row * 4 + column;
                    let x = f64::from(x[index]);
                    let y = f64::from(y[index]) + 0.5;
                    let weight = weight / mean * f64::from(mask[row]) / 2.;
                    expected += (x.max(0.) + (-x.abs()).exp().ln_1p() - x * y) * weight;
                    gradients.push((1. / (1. + (-x).exp()) - y) * weight);
                }
            }
            assert!((actual[backward.outputs()[0] as usize].data[0] - expected).abs() < 1e-10);
            for (actual, expected) in actual[backward.outputs()[1] as usize]
                .data
                .iter()
                .zip(gradients)
            {
                assert!((actual - expected).abs() < 1e-10);
            }
            for &output in &backward.outputs()[2..] {
                assert!(actual[output as usize].data.iter().all(|&x| x == 0.));
            }
        }
    }

    #[test]
    fn predictive_head_losses_match_f64_and_detach_masked_targets() {
        let mut graph = Graph::new();
        let reward = graph.parameter("reward", &[2, 3]);
        let continuation = graph.parameter("continuation", &[2, 1]);
        let reward_target = graph.parameter("reward_target", &[2, 3]);
        let continuation_target = graph.parameter("continuation_target", &[2, 1]);
        let keep = graph.constant(vec![0., 1.], &[2, 1]);
        let (r, c) = head_losses(
            &mut graph,
            reward,
            continuation,
            reward_target,
            continuation_target,
            keep,
        );
        let total = graph.add(r, c);
        graph.set_outputs(vec![total]);
        let backward = meganeura::autodiff::differentiate(&graph);
        let logits = [-1., 0., 1., 1., 0.2, -0.5];
        let rewards = [1., 0., 0., 0.2, 0.8, 0.];
        let continuations = [-2., 0.7];
        let targets = [0., 1. - 1. / 333.];
        let mut feeds = Feeds::new();
        feeds
            .set("reward", &logits)
            .set("continuation", &continuations)
            .set("reward_target", &rewards)
            .set("continuation_target", &targets);
        let actual = evaluate(&backward, &feeds).unwrap();
        let exps = logits[3..]
            .iter()
            .map(|&x| f64::from(x).exp())
            .collect::<Vec<_>>();
        let denominator = exps.iter().sum::<f64>();
        let target_sum = rewards[3..].iter().map(|&v| f64::from(v)).sum::<f64>();
        let mut expected = 0.;
        let mut expected_grad = vec![0.; 3];
        for (i, &exp) in exps.iter().enumerate() {
            let probability = exp / denominator;
            expected -= f64::from(rewards[3 + i]) * probability.ln() / 2.;
            expected_grad.push((probability * target_sum - f64::from(rewards[3 + i])) / 2.);
        }
        let x = f64::from(continuations[1]);
        expected += (x.max(0.) + (-x.abs()).exp().ln_1p() - x * f64::from(targets[1])) / 2.;
        assert!((actual[backward.outputs()[0] as usize].data[0] - expected).abs() < 1e-10);
        for (a, b) in actual[backward.outputs()[1] as usize]
            .data
            .iter()
            .zip(expected_grad)
        {
            assert!((a - b).abs() < 1e-10);
        }
        let gradients = &actual[backward.outputs()[2] as usize].data;
        assert_eq!(gradients[0], 0.);
        assert!(
            (gradients[1] - (1. / (1. + (-x).exp()) - f64::from(targets[1])) / 2.).abs() < 1e-10
        );
        for &output in &backward.outputs()[3..] {
            assert!(actual[output as usize].data.iter().all(|&x| x == 0.));
        }
    }

    #[test]
    fn predictive_states_exclude_arrival_targets_and_auxiliary_reaches_shared_parameters() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        let mut graph = Graph::new();
        let model = WorldModel::with_behavior(&mut graph, &config, false);
        let batch = config.batch_size;
        let size = config.network();
        let prefix = graph.input("prefix", &[batch, size.deter]);
        let logits = model.dynamics.prior.forward(&mut graph, prefix, batch);
        let state = prior_continuation::state(&mut graph, &config, prefix, logits, 1, "predictive");
        let target = graph.input("target", &[batch, config.observation_dim()]);
        let previous = graph.input("previous", &[batch, config.observation_dim()]);
        let keep = graph.input("keep", &[batch, config.action_count]);
        let reward = graph.input("reward", &[batch, config.value_bins]);
        let continuation = graph.input("continuation", &[batch, 1]);
        let forward = ancestors(&graph, state);
        for input in [target, previous, keep, reward, continuation] {
            assert!(!forward.contains(&input));
        }
        assert!(forward.contains(&prefix));
        let losses = losses(
            &mut graph,
            &model,
            state,
            target,
            previous,
            keep,
            reward,
            continuation,
            batch,
        );
        let total = losses.scaled(&mut graph, &config);
        graph.set_outputs(vec![total]);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        for prefix in [
            "world.dynamics.prior",
            "world.decoder.",
            "world.reward.",
            "world.continuation.",
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

    #[test]
    fn complete_predictive_world_preserves_parameter_shapes_and_compiles() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.batch_length = 2;
        config.world_backprop_length = 2;
        config.actor_critic_gradient = false;
        config.disagreement_bonus = false;
        config.loss_scales.policy = 0.;
        config.loss_scales.value = 0.;
        config.loss_scales.replay_value = 0.;
        let normal =
            build_world_only_pixel_training_graph(&config, 0, PixelLoss::MotionCrossEntropy);
        let predictive = build_world_only_pixel_training_graph(
            &config,
            0,
            PixelLoss::PredictiveMotionCrossEntropy,
        );
        let params = |g: &Graph| {
            g.nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(params(&normal), params(&predictive));
        assert!(
            params(&predictive)
                .iter()
                .all(|(name, _)| name.starts_with("world."))
        );
        assert_eq!(predictive.outputs().len(), LOSS_PREDICTIVE_WORLD + 6);
        let (plan, _) = meganeura::compile_training_graph(&predictive);
        assert!(!plan.dispatches.is_empty());
    }
}
