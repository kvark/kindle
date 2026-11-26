use super::*;
use meganeura::graph::Op;
use std::collections::HashSet;

fn ancestors(graph: &Graph, node: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![node];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(graph.node(n).inputs.iter().copied());
        }
    }
    seen
}
#[test]
fn self_observed_states_use_generated_images_and_never_future_labels_or_reset_masks() {
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.world_image_feedback = true;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    let mut graph = Graph::new();
    let model = WorldModel::with_behavior(&mut graph, &c, false);
    let b = c.batch_size;
    let n = c.network();
    let h = graph.input("seed_h", &[b, n.deter]);
    let z = graph.input("seed_z", &[b * n.stoch, n.classes]);
    let history = graph.input("seed_history", &[b, 2 * c.observation_dim()]);
    let images = (0..3)
        .map(|i| graph.input(&format!("true_future_image{i}"), &[b, c.observation_dim()]))
        .collect::<Vec<_>>();
    let actions = (0..2)
        .map(|i| graph.input(&format!("future_action{i}"), &[b, c.action_count]))
        .collect::<Vec<_>>();
    let rewards = (0..2)
        .map(|i| graph.input(&format!("true_future_reward{i}"), &[b, c.value_bins]))
        .collect::<Vec<_>>();
    let continuations = (0..2)
        .map(|i| graph.input(&format!("true_future_continuation{i}"), &[b, 1]))
        .collect::<Vec<_>>();
    let rollout = build_with_feedback(
        &mut graph,
        &c,
        &model,
        (h, z),
        &actions,
        &images,
        PixelLoss::MotionImageFeedback,
        Some(history),
        Some(EventTargets {
            reward: &rewards,
            continuation: &continuations,
        }),
    );
    for (t, &state) in rollout.states.iter().enumerate() {
        let seen = ancestors(&graph, state);
        assert!(seen.contains(&rollout.predictions[t]));
        assert!(
            images
                .iter()
                .chain(&rewards)
                .chain(&continuations)
                .all(|n| !seen.contains(n))
        );
        assert!(seen.iter().all(|&n|!matches!(&graph.node(n).op,Op::Input{name} if name.starts_with("pixel_rollout_weight_"))));
        if t > 0 {
            assert!(seen.contains(&rollout.predictions[t - 1]));
        }
    }
    for &image in &rollout.predictions {
        let seen = ancestors(&graph, image);
        assert!(
            images
                .iter()
                .chain(&rewards)
                .chain(&continuations)
                .all(|n| !seen.contains(n))
        );
    }
    let [reward, continuation] = rollout.events.unwrap();
    let a = graph.add(rollout.loss, reward);
    let total = graph.add(a, continuation);
    graph.set_outputs(vec![total]);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    for prefix in [
        "world.representation.encoder.",
        "world.representation.posterior.",
        "world.dynamics.core.",
        "world.dynamics.prior",
        "world.decoder.motion",
        "world.reward.",
        "world.continuation.",
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
