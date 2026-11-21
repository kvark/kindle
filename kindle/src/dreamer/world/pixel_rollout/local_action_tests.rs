use super::*;
use crate::dreamer::{ObservationKind, runtime};
use meganeura::{Mode, graph::Op};
use std::{collections::HashSet, sync::Arc};

fn ancestors(graph: &Graph, node: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![node];
    while let Some(node) = todo.pop() {
        if seen.insert(node) {
            todo.extend(graph.node(node).inputs.iter().copied());
        }
    }
    seen
}

fn fixture() -> (DreamerConfig, Graph, PixelRollout) {
    let mut c = DreamerConfig::tiny(3);
    c.batch_size = 2;
    c.value_bins = 5;
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    let mut g = Graph::new();
    let model = WorldModel::with_behavior(&mut g, &c, false);
    let b = c.batch_size;
    let n = c.network();
    let h = g.input("seed_h", &[b, n.deter]);
    let z = g.input("seed_z", &[b * n.stoch, n.classes]);
    let history = g.input("seed_history", &[b, 2 * c.observation_dim()]);
    let images = (0..3)
        .map(|i| g.input(&format!("true_image{i}"), &[b, c.observation_dim()]))
        .collect::<Vec<_>>();
    let actions = (0..2)
        .map(|i| g.input(&format!("action{i}"), &[b, c.action_count]))
        .collect::<Vec<_>>();
    let rewards = (0..2)
        .map(|i| g.input(&format!("true_reward{i}"), &[b, c.value_bins]))
        .collect::<Vec<_>>();
    let continuations = (0..2)
        .map(|i| g.input(&format!("true_continuation{i}"), &[b, 1]))
        .collect::<Vec<_>>();
    let rollout = build_with_feedback(
        &mut g,
        &c,
        &model,
        (h, z),
        &actions,
        &images,
        PixelLoss::MotionEventRollout,
        Some(history),
        Some(EventTargets {
            reward: &rewards,
            continuation: &continuations,
        }),
    );
    (c, g, rollout)
}

#[test]
fn local_action_forecasts_keep_action_time_and_target_causality() {
    let (_, mut g, rollout) = fixture();
    for (t, &state) in rollout.states.iter().enumerate() {
        let seen = ancestors(&g, state);
        assert!(seen.iter().all(|&n| !matches!(&g.node(n).op, Op::Input { name } if name.starts_with("true_") || name.starts_with("pixel_rollout_weight_"))));
        assert!(rollout.predictions.iter().all(|n| !seen.contains(n)));
        if t > 0 {
            assert!(seen.iter().any(|&n| matches!(&g.node(n).op, Op::Input { name } if name == "pixel_rollout_uniforms_0")));
        }
    }
    for (time, &image) in rollout.predictions.iter().enumerate() {
        let action_seen = ancestors(&g, image);
        assert!(
            action_seen.iter().any(
                |&n| matches!(&g.node(n).op,Op::Input{name} if name==&format!("action{time}"))
            )
        );
        assert!(!action_seen.iter().any(
            |&n| matches!(&g.node(n).op,Op::Input{name} if name==&format!("action{}",time+1))
        ));
        let seen = ancestors(&g, image);
        assert!(seen.iter().all(|&n| !matches!(&g.node(n).op, Op::Input { name } if name.starts_with("true_") || name.starts_with("pixel_rollout_weight_"))));
    }
    assert!(ancestors(&g, rollout.predictions[1]).contains(&rollout.predictions[0]));
    let [reward, continuation] = rollout.events.unwrap();
    let events = g.add(reward, continuation);
    let seen = ancestors(&g, events);
    assert!(rollout.states.iter().all(|n| seen.contains(n)));
    assert!(rollout.predictions.iter().all(|n| !seen.contains(n)));
    g.set_outputs(vec![events]);
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.reward.",
        "world.continuation.",
        "world.dynamics.core.",
        "world.dynamics.prior",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(name, buffer)| name.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
        );
    }
    assert!(
        plan.param_buffers
            .iter()
            .filter(|(name, _)| name.starts_with("world.decoder."))
            .all(|(_, buffer)| !plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
    );
}

#[test]
#[ignore = "requires separately declared guarded local-action causal controls"]
fn local_action_future_targets_and_masks_do_not_change_images_or_states() {
    let (c, mut g, rollout) = fixture();
    let [reward, continuation] = rollout.events.unwrap();
    let mut outputs = vec![rollout.loss, reward, continuation];
    outputs.extend(rollout.states.iter().copied());
    outputs.extend(rollout.predictions.iter().copied());
    let widths = outputs
        .iter()
        .map(|&n| g.node(n).ty.num_elements())
        .collect::<Vec<_>>();
    g.set_outputs(outputs);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = runtime::build_session(&g, &gpu, Mode::Inference, false);
    runtime::initialize_d3(&mut session, &g, 47);
    session.set_parameter("world.reward.out.bias", &[0.2, -0.1, 0.7, -0.4, 0.3]);
    session.set_parameter("world.continuation.out.bias", &[0.3]);
    let n = c.network();
    let h = (0..2 * n.deter)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect::<Vec<_>>();
    session.set_input("seed_h", &h);
    let mut z = vec![0.; 2 * n.stoch * n.classes];
    for i in 0..2 * n.stoch {
        z[i * n.classes + i % n.classes] = 1.;
    }
    session.set_input("seed_z", &z);
    let history = (0..4 * c.observation_dim())
        .map(|i| (i % 19) as f32 / 24. - 0.4)
        .collect::<Vec<_>>();
    session.set_input("seed_history", &history);
    for t in 0..2 {
        session.set_input(&format!("action{t}"), &[1., 0., 0., 0., 1., 0.]);
        session.set_input(
            &format!("pixel_rollout_uniforms_{t}"),
            &vec![0.37; 2 * n.stoch * n.classes],
        );
    }
    let baseline = {
        let mut run = |kind| {
            for t in 0..3 {
                session.set_input(
                    &format!("true_image{t}"),
                    &vec![if kind == 1 { 0.35 } else { -0.25 }; 2 * c.observation_dim()],
                );
            }
            for t in 0..2 {
                let mut target = vec![0.; 2 * c.value_bins];
                for lane in 0..2 {
                    target[lane * c.value_bins + usize::from(kind == 2)] = 1.;
                }
                session.set_input(&format!("true_reward{t}"), &target);
                session.set_input(
                    &format!("true_continuation{t}"),
                    &[if kind == 3 {
                        0.
                    } else {
                        c.continuation_discount()
                    }; 2],
                );
                session.set_input(
                    &format!("pixel_rollout_weight_{t}"),
                    &[if kind == 4 { 0. } else { 1. }; 2],
                );
            }
            session.step();
            session.wait();
            widths
                .iter()
                .enumerate()
                .map(|(i, &width)| {
                    let mut values = vec![0.; width];
                    session.read_output_by_index(i, &mut values);
                    assert!(values.iter().all(|x| x.is_finite()));
                    values
                })
                .collect::<Vec<_>>()
        };
        let baseline = run(0);
        for kind in 1..5 {
            let changed = run(kind);
            assert_eq!(baseline[3..], changed[3..]);
            if kind == 1 {
                assert_ne!(baseline[0], changed[0]);
            }
            if kind == 2 {
                assert_ne!(baseline[1], changed[1]);
            }
            if kind == 3 {
                assert_ne!(baseline[2], changed[2]);
            }
            if kind == 4 {
                assert!(changed[..3].iter().all(|x| x[0] == 0.));
            }
        }
        baseline
    };
    let mut changed_h = h;
    for value in &mut changed_h[..n.deter] {
        *value += 0.15;
    }
    session.set_input("seed_h", &changed_h);
    session.step();
    session.wait();
    let mut changed_lane = false;
    for (i, &width) in widths.iter().enumerate().skip(3) {
        let mut value = vec![0.; width];
        session.read_output_by_index(i, &mut value);
        assert_eq!(value[width / 2..], baseline[i][width / 2..]);
        changed_lane |= value[..width / 2] != baseline[i][..width / 2];
    }
    assert!(changed_lane);
    println!("suffix event target/image/mask independence and lane isolation pass");
}
