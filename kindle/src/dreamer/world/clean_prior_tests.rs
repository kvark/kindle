use super::*;
use crate::dreamer::{ObservationKind, runtime};
use meganeura::{Mode, graph::Op};
use std::{collections::HashSet, sync::Arc};

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.batch_size = 2;
    c.batch_length = 2;
    c.world_backprop_length = 2;
    c.world_microbatch_size = Some(2);
    c.value_bins = 5;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

fn ancestors(g: &Graph, node: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![node];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(g.node(n).inputs.iter().copied());
        }
    }
    seen
}

fn readouts(g: &Graph, name: &str) -> Vec<NodeId> {
    let weight = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op,Op::Parameter{name:n} if n==name))
        .unwrap()
        .id;
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul | Op::Conv2d { .. }) && n.inputs.contains(&weight))
        .map(|n| n.id)
        .collect()
}

#[test]
fn clean_main_rgb_and_event_readouts_exclude_the_current_image_and_posterior_sample() {
    let c = config();
    let g = training_graph(&c, 2, 1, true, false, 1, PixelLoss::MotionCleanPrior);
    for name in [
        "world.decoder.imgout.weight",
        "world.reward.out.weight",
        "world.continuation.out.weight",
    ] {
        let nodes = readouts(&g, name);
        assert_eq!(nodes.len(), 3, "{name}");
        for (time, &output) in nodes[..2].iter().enumerate() {
            let dependencies = ancestors(&g, output);
            for n in g.nodes() {
                if let Op::Input { name } = &n.op {
                    if name == &format!("observation_{time}")
                        || name == &format!("uniforms_{time}")
                        || name.starts_with("reward_target_")
                        || name.starts_with("continuation_target_")
                    {
                        assert!(!dependencies.contains(&n.id), "{name} enters {output}");
                    }
                    if time == 1 && name == "observation_0" {
                        assert!(dependencies.contains(&n.id));
                    }
                }
            }
        }
    }
    let mut g = g;
    g.set_outputs(vec![g.outputs()[LOSS_RECONSTRUCTION]]);
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.representation.encoder.",
        "world.representation.posterior.",
        "world.dynamics.core.",
        "world.dynamics.prior",
        "world.decoder.",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(n, b)| n.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(p, _)| p == b)),
            "{prefix}"
        );
    }
}

#[test]
fn clean_prior_preserves_parameters_one_decoder_pass_and_reports_validity() {
    let c = config();
    let old = build_world_only_pixel_training_graph(&c, 1, PixelLoss::MotionEventRollout);
    let g = build_world_only_pixel_training_graph(&c, 1, PixelLoss::MotionCleanPrior);
    let specs = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| {
                if let Op::Parameter { name } = &n.op {
                    Some((name.clone(), n.ty.shape.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(specs(&old), specs(&g));
    assert_eq!(
        readouts(&old, "world.decoder.imgout.weight").len(),
        readouts(&g, "world.decoder.imgout.weight").len()
    );
    assert_eq!(g.outputs().len(), CLEAN_PRIOR_VALID_FRACTION + 3);
    let inputs = g
        .nodes()
        .iter()
        .filter_map(|n| {
            if let Op::Input { name } = &n.op {
                Some(name)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(inputs.len(), inputs.iter().collect::<HashSet<_>>().len());
    assert_eq!(
        inputs
            .iter()
            .filter(|n| n.starts_with("clean_prior_uniforms_"))
            .count(),
        2
    );
    assert!(
        inputs
            .iter()
            .all(|n| !n.starts_with("pixel_feedback_uniforms_"))
    );
    let total = ancestors(&g, g.outputs()[0]);
    for i in [
        LOSS_RECONSTRUCTION,
        LOSS_REWARD,
        LOSS_CONTINUATION,
        LOSS_WORLD_ROLLOUT,
        LOSS_IMAGE_FEEDBACK,
        LOSS_IMAGE_FEEDBACK + 1,
    ] {
        assert!(total.contains(&g.outputs()[i]));
    }
    assert!(!total.contains(&g.outputs()[CLEAN_PRIOR_VALID_FRACTION]));
}

#[test]
#[ignore = "requires separately guarded clean-prior main-readout controls"]
fn clean_main_native_readouts_are_causal_lane_local_and_zero_masked() {
    let c = config();
    let mut g = training_graph(&c, 2, 2, true, false, 1, PixelLoss::MotionCleanPrior);
    let mut outputs = vec![
        g.outputs()[LOSS_RECONSTRUCTION],
        g.outputs()[LOSS_REWARD],
        g.outputs()[LOSS_CONTINUATION],
        g.outputs()[LOSS_WORLD_ROLLOUT],
        g.outputs()[LOSS_IMAGE_FEEDBACK],
        g.outputs()[LOSS_IMAGE_FEEDBACK + 1],
        g.outputs()[CLEAN_PRIOR_VALID_FRACTION],
    ];
    for name in [
        "world.decoder.imgout.weight",
        "world.reward.out.weight",
        "world.continuation.out.weight",
    ] {
        outputs.extend(readouts(&g, name));
    }
    let widths = outputs
        .iter()
        .map(|&n| g.node(n).ty.num_elements())
        .collect::<Vec<_>>();
    outputs.extend_from_slice(g.outputs());
    g.set_outputs(outputs);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = runtime::build_session(&g, &gpu, Mode::Inference, false);
    runtime::initialize_d3(&mut session, &g, 61);
    session.set_parameter("world.reward.out.bias", &[0.2, -0.1, 0.7, -0.4, 0.3]);
    session.set_parameter("world.continuation.out.bias", &[0.3]);
    let n = c.network();
    session.set_input("initial_deter", &vec![0.; 2 * n.deter]);
    session.set_input("initial_stoch", &vec![0.; 2 * n.stoch * n.classes]);
    let mut run = |kind| {
        for t in 0..2 {
            let keep = f32::from(t == 1 && kind != 4);
            session.set_input(&format!("keep_deter_{t}"), &vec![keep; 2 * n.deter]);
            session.set_input(
                &format!("keep_stoch_{t}"),
                &vec![keep; 2 * n.stoch * n.classes],
            );
            session.set_input(&format!("keep_action_{t}"), &vec![keep; 2 * c.action_count]);
            session.set_input(&format!("previous_action_{t}"), &[1., 0., 0., 0., 1., 0.]);
            let mut image = (0..2 * c.observation_dim())
                .map(|i| (i % 23) as f32 / 30. - 0.4)
                .collect::<Vec<_>>();
            if kind == 1 && t == 1 {
                image.fill(0.3);
            }
            if kind == 5 && t == 0 {
                image[..c.observation_dim()].fill(-0.3);
            }
            session.set_input(&format!("observation_{t}"), &image);
            let mut reward = vec![0.; 2 * c.value_bins];
            for lane in 0..2 {
                reward[lane * c.value_bins + if kind == 2 { 3 } else { 1 }] = 1.;
            }
            session.set_input(&format!("reward_target_{t}"), &reward);
            session.set_input(
                &format!("continuation_target_{t}"),
                &[if kind == 3 {
                    0.
                } else {
                    c.continuation_discount()
                }; 2],
            );
            for prefix in ["uniforms", "clean_prior_uniforms"] {
                session.set_input(
                    &format!("{prefix}_{t}"),
                    &vec![0.37; 2 * n.stoch * n.classes],
                );
            }
        }
        session.set_input(
            "pixel_rollout_uniforms_0",
            &vec![0.57; 2 * n.stoch * n.classes],
        );
        session.set_input(
            "pixel_rollout_weight_0",
            &[if kind == 4 { 0. } else { 1. }; 2],
        );
        session.step();
        session.wait();
        widths
            .iter()
            .enumerate()
            .map(|(i, &width)| {
                let mut x = vec![0.; width];
                session.read_output_by_index(i, &mut x);
                assert!(x.iter().all(|v| v.is_finite()));
                x
            })
            .collect::<Vec<_>>()
    };
    let base = run(0);
    assert_eq!(base[6], [0.5]);
    for kind in 1..4 {
        let changed = run(kind);
        assert_eq!(base[7..], changed[7..]);
        if kind == 1 {
            assert_ne!(base[0], changed[0]);
        }
        if kind == 2 {
            assert_ne!(base[1], changed[1]);
        }
        if kind == 3 {
            assert_ne!(base[2], changed[2]);
        }
    }
    let zero = run(4);
    assert!(zero[..7].iter().all(|x| x[0] == 0.));
    let changed = run(5);
    let mut affected = false;
    for (i, &width) in widths.iter().enumerate().skip(7) {
        let rows = if (i - 7) % 2 == 0 { 4 } else { 2 };
        let width = width / rows;
        for row in 0..rows {
            let range = row * width..(row + 1) * width;
            if row % 2 == 1 {
                assert_eq!(base[i][range.clone()], changed[i][range]);
            } else {
                affected |= base[i][range.clone()] != changed[i][range];
            }
        }
    }
    assert!(affected);
    println!("actual clean-prior RGB/event causality, lane isolation and zero masks pass");
}
