use super::*;
use meganeura::{
    graph::Op,
    reference::{Feeds, evaluate},
};
use std::{collections::HashSet, sync::Arc};

fn reference(
    first: &[f64],
    second: &[f64],
    target: &[f64],
    keep: &[f64],
    width: usize,
) -> ([f64; 4], [Vec<f64>; 2]) {
    let mut reports = [0.; 4];
    let mut gradients = [vec![0.; first.len()], vec![0.; first.len()]];
    let epsilon = f64::from(1e-6_f32);
    let denominator = (keep.len() * width) as f64;
    for row in 0..keep.len() {
        let range = row * width..(row + 1) * width;
        let a = &first[range.clone()];
        let b = &second[range.clone()];
        let y = &target[range.clone()];
        let distance = |x: &[f64], z: &[f64]| {
            ((x.iter().zip(z).map(|(x, z)| (x - z).powi(2)).sum::<f64>() / width as f64) + epsilon)
                .sqrt()
        };
        let da = distance(a, y);
        let db = distance(b, y);
        let pair = distance(a, b);
        let weight = keep[row] / keep.len() as f64;
        reports[0] += weight * (0.5 * (da + db) - 0.5 * pair);
        reports[1] += weight * 0.5 * (da + db);
        reports[2] += weight * pair;
        reports[3] += weight;
        for j in 0..width {
            gradients[0][row * width + j] =
                0.5 * keep[row] / denominator * ((a[j] - y[j]) / da - (a[j] - b[j]) / pair);
            gradients[1][row * width + j] =
                0.5 * keep[row] / denominator * ((b[j] - y[j]) / db - (b[j] - a[j]) / pair);
        }
    }
    (reports, gradients)
}

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    values: [Vec<f32>; 4],
    expected: [f64; 4],
    gradients: [Vec<f64>; 2],
    width: usize,
}

fn fixture(batch: usize, width: usize, zero: bool) -> Fixture {
    let a = (0..batch * width)
        .map(|i| ((i * 7) % 19) as f32 * 0.125 - 1.)
        .collect::<Vec<_>>();
    let b = (0..batch * width)
        .map(|i| ((i * 3 + 5) % 17) as f32 * 0.125 - 1.)
        .collect::<Vec<_>>();
    let y = (0..batch * width)
        .map(|i| ((i * 5 + 2) % 23) as f32 * 0.0625 - 0.5)
        .collect::<Vec<_>>();
    let keep = (0..batch)
        .map(|i| f32::from(!zero && i % 3 != 0))
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let first = graph.parameter("first", &[batch, width]);
    let second = graph.parameter("second", &[batch, width]);
    let target = graph.parameter("target", &[batch, width]);
    let mask = graph.parameter("keep", &[batch, 1]);
    let losses = score(&mut graph, first, second, target, mask, width);
    graph.set_outputs(vec![
        losses.energy,
        losses.target_distance,
        losses.sample_distance,
        losses.valid,
    ]);
    let mut feeds = Feeds::new();
    feeds
        .set("first", &a)
        .set("second", &b)
        .set("target", &y)
        .set("keep", &keep);
    let f64v = |x: &[f32]| x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let (expected, gradients) = reference(&f64v(&a), &f64v(&b), &f64v(&y), &f64v(&keep), width);
    Fixture {
        graph,
        feeds,
        values: [a, b, y, keep],
        expected,
        gradients,
        width,
    }
}

#[test]
fn recurrent_energy_matches_f64_values_gradients_and_detached_targets() {
    for zero in [false, true] {
        let f = fixture(4, 8, zero);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let values = evaluate(&backward, &f.feeds).unwrap();
        let outputs = backward.outputs();
        for (i, v) in f.expected.iter().enumerate() {
            assert!((values[outputs[i] as usize].data[0] - v).abs() < 1e-10);
        }
        for (i, g) in f.gradients.iter().enumerate() {
            for (a, b) in values[outputs[4 + i] as usize].data.iter().zip(g) {
                assert!((a - b).abs() < 1e-10);
            }
        }
        for &id in &outputs[6..] {
            assert!(values[id as usize].data.iter().all(|&v| v == 0.));
        }
    }
}

#[test]
fn recurrent_energy_derivatives_match_finite_differences() {
    let f = fixture(4, 8, false);
    let mut vectors = f
        .values
        .iter()
        .map(|v| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for sample in 0..2 {
        for i in (0..vectors[sample].len()).step_by(3) {
            let old = vectors[sample][i];
            vectors[sample][i] = old + 1e-6;
            let plus = reference(&vectors[0], &vectors[1], &vectors[2], &vectors[3], f.width).0[0];
            vectors[sample][i] = old - 1e-6;
            let minus = reference(&vectors[0], &vectors[1], &vectors[2], &vectors[3], f.width).0[0];
            vectors[sample][i] = old;
            assert!(((plus - minus) / 2e-6 - f.gradients[sample][i]).abs() < 1e-8);
        }
    }
}

fn assert_gradient_scope(graph: &Graph) {
    let backward = meganeura::autodiff::differentiate(graph);
    let parameters = graph.nodes().iter().filter_map(|n| match &n.op {
        Op::Parameter { name } => Some(name),
        _ => None,
    });
    for (name, &gradient) in parameters.zip(&backward.outputs()[1..]) {
        if !name.starts_with("world.dynamics.") {
            assert!(
                matches!(&backward.node(gradient).op,Op::Constant{data} if data.iter().all(|&v| v==0.)),
                "{name}"
            );
        }
    }
    let (plan, _) = meganeura::compile_training_graph(graph);
    for prefix in ["world.dynamics.core.", "world.dynamics.prior"] {
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
fn recurrent_prior_states_exclude_future_targets_masks_and_actions() {
    let config = DreamerConfig::tiny(3);
    let size = config.network();
    let batch = config.batch_size;
    let mut graph = Graph::new();
    let model = WorldModel::with_behavior(&mut graph, &config, false);
    let h = graph.input("prefix_h", &[batch, size.deter]);
    let z = graph.input("prefix_z", &[batch * size.stoch, size.classes]);
    let actions = (0..3)
        .map(|i| graph.input(&format!("action{i}"), &[batch, config.action_count]))
        .collect::<Vec<_>>();
    let targets = (0..3)
        .map(|i| graph.input(&format!("target{i}"), &[batch, size.deter]))
        .collect::<Vec<_>>();
    let losses = build(
        &mut graph,
        &config,
        &model,
        (h, z),
        &actions,
        &targets,
        "pixel_rollout_weight",
    );
    for (i, &state) in losses.states.iter().enumerate() {
        let time = i / 2;
        let mut seen = HashSet::new();
        let mut pending = vec![state];
        while let Some(id) = pending.pop() {
            if seen.insert(id) {
                pending.extend(&graph.node(id).inputs);
            }
        }
        assert!(targets.iter().all(|t| !seen.contains(t)));
        assert!(actions[..=time].iter().all(|a| seen.contains(a)));
        assert!(actions[time + 1..].iter().all(|a| !seen.contains(a)));
        assert!(seen.iter().all(|&id| !matches!(&graph.node(id).op,Op::Input{name} if name.starts_with("pixel_rollout_weight_"))));
    }
    graph.set_outputs(vec![losses.energy]);
    assert_gradient_scope(&graph);
}

#[test]
fn recurrent_objective_preserves_shapes_and_detaches_observed_target_branch() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb64;
    config.batch_length = 4;
    config.world_backprop_length = 4;
    config.actor_critic_gradient = false;
    config.disagreement_bonus = false;
    config.loss_scales.policy = 0.;
    config.loss_scales.value = 0.;
    config.loss_scales.replay_value = 0.;
    let normal = build_world_only_pixel_training_graph(&config, 0, PixelLoss::MotionCrossEntropy);
    let mut recurrent =
        build_world_only_pixel_training_graph(&config, 2, PixelLoss::MotionRecurrentEnergy);
    let parameters = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(parameters(&normal), parameters(&recurrent));
    meganeura::compile_training_graph(&recurrent);
    let loss = recurrent.outputs()[LOSS_RECURRENT_WORLD];
    recurrent.set_outputs(vec![loss]);
    assert_gradient_scope(&recurrent);
}

#[test]
#[ignore = "requires guarded GPU; recurrent RMS energy values and derivatives"]
fn recurrent_energy_native_matches_f64() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for zero in [false, true] {
        let f = fixture(256, 64, zero);
        let mut session = crate::dreamer::runtime::build_session(
            &f.graph,
            &gpu,
            meganeura::Mode::Training,
            false,
        );
        for (name, values) in ["first", "second", "target", "keep"]
            .into_iter()
            .zip(&f.values)
        {
            session.set_parameter(name, values);
        }
        session.step();
        session.wait();
        assert_eq!(session.adam_step_count(), 0);
        for (i, expected) in f.expected.iter().enumerate() {
            let mut out = [0.];
            session.read_output_by_index(i, &mut out);
            assert!((f64::from(out[0]) - expected).abs() < 2e-5);
        }
        for (name, expected) in ["first", "second"].into_iter().zip(&f.gradients) {
            let mut out = vec![0.; expected.len()];
            session.read_param_grad(name, &mut out);
            for (a, b) in out.iter().zip(expected) {
                assert!((f64::from(*a) - b).abs() < 2e-6);
            }
        }
        for (name, values) in ["first", "second", "target", "keep"]
            .into_iter()
            .zip(&f.values)
        {
            let mut out = vec![0.; values.len()];
            session.read_param(name, &mut out);
            assert_eq!(&out, values);
            if matches!(name, "target" | "keep") && session.has_param_grad(name) {
                session.read_param_grad(name, &mut out);
                assert!(out.iter().all(|&v| v == 0.));
            }
        }
    }
}
