use super::*;
use meganeura::graph::Op;
use std::collections::HashSet;

fn ancestors(g: &Graph, n: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![n];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(g.node(n).inputs.iter().copied());
        }
    }
    seen
}

#[test]
fn rgb_event_suffix_reads_own_images_but_never_corrects_prior_states() {
    suffix_causality(None);
    suffix_causality(Some(10.));
}

fn suffix_causality(survival_scale: Option<f32>) {
    // A reduced graph fixture exercises the shared readout wiring, not a
    // supported production preset. Public construction is separately restricted.
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_event_heads = true;
    c.world_survival_scale = survival_scale;
    c.value_bins = 5;
    let mut g = Graph::new();
    let b = c.batch_size;
    let size = c.network();
    let model = WorldModel::with_behavior(&mut g, &c, false);
    let h = g.input("seed_h", &[b, size.deter]);
    let z = g.input("seed_z", &[b * size.stoch, size.classes]);
    let history = g.input("seed_history", &[b, 2 * c.observation_dim()]);
    let actions = (0..2)
        .map(|t| g.input(&format!("action{t}"), &[b, 3]))
        .collect::<Vec<_>>();
    let images = (0..3)
        .map(|t| g.input(&format!("truth_rgb{t}"), &[b, c.observation_dim()]))
        .collect::<Vec<_>>();
    let rewards = (0..2)
        .map(|t| g.input(&format!("truth_reward{t}"), &[b, c.value_bins]))
        .collect::<Vec<_>>();
    let continuations = (0..2)
        .map(|t| g.input(&format!("truth_continuation{t}"), &[b, 1]))
        .collect::<Vec<_>>();
    let rollout = pixel_rollout::build_with_feedback(
        &mut g,
        &c,
        &model,
        (h, z),
        &actions,
        &images,
        PixelLoss::MotionEventRollout,
        Some(history),
        Some(pixel_rollout::EventTargets {
            reward: &rewards,
            continuation: &continuations,
        }),
    );
    for (t, &state) in rollout.states.iter().enumerate() {
        let seen = ancestors(&g, state);
        assert!(seen.contains(&h) && seen.contains(&z) && seen.contains(&actions[t]));
        assert!(rollout.predictions.iter().all(|n| !seen.contains(n)));
        assert!(actions[t + 1..].iter().all(|n| !seen.contains(n)));
        assert!(seen.iter().all(|&n|!matches!(&g.node(n).op,Op::Input{name} if name.starts_with("truth_") || name.starts_with("pixel_rollout_weight_"))));
    }
    let [reward, continuation] = rollout.events.unwrap();
    let mut events = g.add(reward, continuation);
    if let Some(loss) = rollout.survival {
        let seen = ancestors(&g, loss);
        assert!(continuations.iter().all(|n| seen.contains(n)));
        assert!(images.iter().all(|n| !seen.contains(n)));
        assert!(rollout.predictions.iter().all(|n| seen.contains(n)));
        events = g.add(events, loss);
    }
    for loss in [reward, continuation] {
        let seen = ancestors(&g, loss);
        assert!(rollout.predictions.iter().all(|n| seen.contains(n)));
        assert!(images.iter().all(|n| !seen.contains(n)));
    }
    for &image in &rollout.predictions {
        let seen = ancestors(&g, image);
        assert!(seen.iter().all(|&n|!matches!(&g.node(n).op,Op::Input{name} if name.starts_with("truth_") || name.starts_with("pixel_rollout_weight_"))));
    }
    assert!(ancestors(&g, rollout.predictions[1]).contains(&rollout.predictions[0]));
    g.set_outputs(vec![events]);
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.reward.",
        "world.continuation.",
        "world.representation.encoder.",
        "world.decoder.motion0.",
        "world.decoder.motion1.",
        "world.decoder.motion2.",
        "world.decoder.fusion.",
        "world.decoder.imgout.",
        "world.dynamics.core.",
        "world.dynamics.prior",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(name, p)| name.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(x, _)| x == p)),
            "{prefix}"
        );
    }
    assert!(
        plan.param_buffers
            .iter()
            .filter(|(name, _)| name.starts_with("world.representation.posterior."))
            .all(|(_, p)| !plan.param_grad_pairs.iter().any(|(x, _)| x == p))
    );
}

#[test]
fn rgb_event_observed_losses_read_decoded_posterior_images_and_shared_encoder() {
    let mut c = DreamerConfig::tiny(18);
    c.model_size = super::super::ModelSize::Size1M;
    c.observation_kind = ObservationKind::Rgb64;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_event_heads = true;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    let g = build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionEventRollout);
    for index in [LOSS_REWARD, LOSS_CONTINUATION] {
        let seen = ancestors(&g, g.outputs()[index]);
        for prefix in [
            "world.decoder.motion0.",
            "world.decoder.imgout.",
            "world.representation.encoder.",
            "world.representation.posterior.",
        ] {
            assert!(
                seen.iter().any(
                    |&n| matches!(&g.node(n).op,Op::Parameter{name} if name.starts_with(prefix))
                ),
                "{index} {prefix}"
            );
        }
    }
}

struct ReadoutFixture {
    graph: Graph,
    feeds: meganeura::reference::Feeds,
    parameters: Vec<(String, Vec<f32>)>,
}

fn readout_fixture() -> ReadoutFixture {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(902);
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_event_heads = true;
    c.value_bins = 5;
    let mut g = Graph::new();
    let encoder = super::super::networks::ObservationEncoder::new(&mut g, &c);
    let heads = WorldHeads::new(&mut g, &c);
    let state = g.parameter("fixture.state", &[2, c.feature_dim()]);
    let logits = g.parameter("fixture.rgb_logits", &[2, c.observation_dim()]);
    let image = super::super::networks::centered_pixels(&mut g, logits);
    let event_state = rgb_event_features(&mut g, &encoder, state, image, 2);
    let (reward, continuation) = heads.forward(&mut g, event_state);
    let outputs = concat_columns(&mut g, reward, continuation, 2, 5, 1);
    let loss = meganeura::reference::gradients::weighted_loss(&mut g, outputs, 901, 1.);
    g.set_outputs(vec![loss, outputs]);
    let parameters = g
        .nodes()
        .iter()
        .filter_map(|n| {
            let Op::Parameter { name } = &n.op else {
                return None;
            };
            let values = if name == "fixture.rgb_logits" {
                (0..n.ty.num_elements())
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            } else if name.ends_with("norm.weight") {
                vec![1.; n.ty.num_elements()]
            } else if name.ends_with("bias") {
                vec![0.; n.ty.num_elements()]
            } else {
                (0..n.ty.num_elements())
                    .map(|i| ((i * 13 + name.len() * 7) % 37) as f32 / 370. - 0.05)
                    .collect()
            };
            Some((name.clone(), values))
        })
        .collect::<Vec<_>>();
    let mut feeds = meganeura::reference::Feeds::new();
    for (name, values) in &parameters {
        feeds.set(name, values);
    }
    ReadoutFixture {
        graph: g,
        feeds,
        parameters,
    }
}

fn pool_winners(g: &Graph, values: &[meganeura::reference::Tensor]) -> Vec<usize> {
    let mut winners = Vec::new();
    for n in g.nodes() {
        if let Op::MaxPool2d {
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        } = n.op
        {
            let (c, h, w, kh, kw, s, p) = (
                channels as usize,
                in_h as usize,
                in_w as usize,
                kernel_h as usize,
                kernel_w as usize,
                stride as usize,
                padding as isize,
            );
            let x = &values[n.inputs[0] as usize].data;
            let oh = (h + 2 * padding as usize - kh) / s + 1;
            let ow = (w + 2 * padding as usize - kw) / s + 1;
            for bc in 0..x.len() / (h * w) {
                for y in 0..oh {
                    for z in 0..ow {
                        let mut best = None;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (y * s + ky) as isize - p;
                                let ix = (z * s + kx) as isize - p;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let index = bc * h * w + iy as usize * w + ix as usize;
                                if best.is_none_or(|old| x[index] > x[old]) {
                                    best = Some(index);
                                }
                            }
                        }
                        winners.push(best.unwrap());
                    }
                }
            }
            assert_eq!(x.len() % (c * h * w), 0);
        }
    }
    winners
}

#[test]
fn rgb_event_readout_f64_derivatives_and_lane_isolation() {
    use meganeura::reference::evaluate;
    let mut f = readout_fixture();
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let expected = evaluate(&backward, &f.feeds).unwrap();
    assert!(expected.iter().flat_map(|v| &v.data).all(|x| x.is_finite()));
    for name in [
        "fixture.rgb_logits",
        "world.representation.encoder.cnn0.weight",
        "world.reward.layer0.weight",
        "world.continuation.layer0.weight",
    ] {
        let i = f.parameters.iter().position(|(n, _)| n == name).unwrap();
        let values = f.parameters[i].1.clone();
        let grad = &expected[backward.outputs()[2 + i] as usize].data;
        assert!(grad.iter().any(|x| x.abs() > 1e-10), "{name}");
        let direction = (0..values.len())
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect::<Vec<f64>>();
        let norm = (values.len() as f64).sqrt();
        let baseline_winners = pool_winners(&f.graph, &expected);
        let mut checked = false;
        for epsilon in [1e-3, 1e-4, 1e-5, 1e-6] {
            let changed = [-1., 1.].map(|sign| {
                values
                    .iter()
                    .zip(&direction)
                    .map(|(&v, d)| (f64::from(v) + sign * epsilon * d / norm) as f32)
                    .collect::<Vec<_>>()
            });
            assert!(changed[0].iter().zip(&changed[1]).any(|(a, b)| a != b));
            let result = changed.each_ref().map(|v| {
                f.feeds.set(name, v);
                evaluate(&f.graph, &f.feeds).unwrap()
            });
            let switches = result
                .iter()
                .map(|v| {
                    pool_winners(&f.graph, v)
                        .iter()
                        .zip(&baseline_winners)
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum::<usize>();
            eprintln!("{name} epsilon={epsilon} pool_winner_changes={switches}");
            if switches != 0 {
                continue;
            }
            let measured = (result[1][f.graph.outputs()[0] as usize].data[0]
                - result[0][f.graph.outputs()[0] as usize].data[0])
                / (2. * epsilon);
            // Feeds deliberately round operands to f32. Compare against the
            // actual rounded direction rather than its ideal f64 proposal.
            let derivative = grad
                .iter()
                .zip(changed[0].iter().zip(&changed[1]))
                .map(|(g, (&lo, &hi))| g * (f64::from(hi) - f64::from(lo)) / (2. * epsilon))
                .sum::<f64>();
            assert!(
                (measured - derivative).abs() <= 3e-6 + 3e-3 * derivative.abs(),
                "{name}: {measured} vs {derivative}"
            );
            checked = true;
            break;
        }
        f.feeds.set(name, &values);
        assert!(
            checked,
            "no stable nonzero finite-difference interval for {name}"
        );
    }
    let (_, pixels) = f
        .parameters
        .iter()
        .find(|(n, _)| n == "fixture.rgb_logits")
        .unwrap();
    let mut changed = pixels.clone();
    let half = changed.len() / 2;
    for x in &mut changed[half..] {
        *x += 0.3;
    }
    f.feeds.set("fixture.rgb_logits", &changed);
    let actual = evaluate(&f.graph, &f.feeds).unwrap();
    let old = &expected[backward.outputs()[1] as usize].data;
    let new = &actual[f.graph.outputs()[1] as usize].data;
    assert_eq!(old[..6], new[..6]);
    assert!(
        old[6..]
            .iter()
            .zip(&new[6..])
            .any(|(a, b)| (a - b).abs() > 1e-8)
    );
}

#[test]
#[ignore = "requires separately declared guarded RGB event-head qualification"]
fn rgb_event_readout_native_values_derivatives_and_lane_isolation() {
    use meganeura::{Mode, reference::evaluate};
    use std::sync::Arc;
    let mut f = readout_fixture();
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = super::super::runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    assert_eq!(
        session.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    session.clear_optimizer();
    let mut first_lane = None;
    for offset in [0., 0.3, -0.2] {
        for (name, values) in &f.parameters {
            let mut changed = values.clone();
            if name == "fixture.rgb_logits" {
                let half = changed.len() / 2;
                for x in &mut changed[half..] {
                    *x += offset;
                }
            }
            f.feeds.set(name, &changed);
            session.set_parameter(name, &changed);
        }
        let expected = evaluate(&backward, &f.feeds).unwrap();
        session.step();
        session.wait();
        let check = |actual: &[f32], expected: &[f64]| {
            assert_eq!(actual.len(), expected.len());
            for (&a, &b) in actual.iter().zip(expected) {
                assert!(
                    a.is_finite() && (f64::from(a) - b).abs() <= 2e-4 + 2e-3 * b.abs(),
                    "{a} vs {b}"
                );
            }
        };
        for (i, &node) in f.graph.outputs().iter().enumerate() {
            let mut actual = vec![0.; f.graph.node(node).ty.num_elements()];
            session.read_output_by_index(i, &mut actual);
            check(&actual, &expected[backward.outputs()[i] as usize].data);
            if i == 1 {
                let lane = actual[..6].to_vec();
                if let Some(prior) = &first_lane {
                    assert_eq!(&lane, prior);
                } else {
                    first_lane = Some(lane);
                }
            }
        }
        for (i, (name, values)) in f.parameters.iter().enumerate() {
            let mut actual = vec![0.; values.len()];
            session.read_param_grad(name, &mut actual);
            check(&actual, &expected[backward.outputs()[2 + i] as usize].data);
        }
    }
    assert_eq!(session.adam_step_count(), 0);
}
