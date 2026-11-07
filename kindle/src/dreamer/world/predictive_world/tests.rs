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

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
    expected: [f64; 4],
    gradients: [Vec<f64>; 2],
}

fn fixture(rows: usize, width: usize, zero_mask: bool) -> Fixture {
    let mut graph = Graph::new();
    let mut parameters = Vec::new();
    let mut feeds = Feeds::new();
    let mut values = Vec::new();
    for (name, offset) in [("first", 0), ("second", 3), ("target", 7)] {
        let data = (0..rows * (width + 2))
            .map(|i| ((i * 7 + offset) % 19) as f32 * 0.0625 - 0.5)
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        for (suffix, start, end, columns) in [
            ("rgb", 0, width, width),
            ("reward", width, width + 1, 1),
            ("continuation", width + 1, width + 2, 1),
        ] {
            let name = format!("{name}_{suffix}");
            let v = data
                .chunks_exact(width + 2)
                .flat_map(|row| row[start..end].iter().copied())
                .collect::<Vec<_>>();
            nodes.push(graph.parameter(&name, &[rows, columns]));
            feeds.set(&name, &v);
            parameters.push((name, v));
        }
        values.push(data.into_iter().map(f64::from).collect::<Vec<_>>());
        assert_eq!(nodes.len(), 3);
    }
    let nodes = graph
        .nodes()
        .iter()
        .filter_map(|node| matches!(node.op, Op::Parameter { .. }).then_some(node.id))
        .collect::<Vec<_>>();
    let sample = |i| Sample {
        rgb: nodes[i],
        reward: nodes[i + 1],
        continuation: nodes[i + 2],
    };
    let weights = (0..rows * width)
        .map(|i| 0.5 + (i % 5) as f32 * 0.25)
        .collect::<Vec<_>>();
    let keep = (0..rows)
        .map(|i| f32::from(!zero_mask && i % 3 != 0))
        .collect::<Vec<_>>();
    let w = graph.parameter("weights", &[rows, width]);
    let k = graph.parameter("keep", &[rows, 1]);
    feeds.set("weights", &weights).set("keep", &keep);
    parameters.push(("weights".into(), weights.clone()));
    parameters.push(("keep".into(), keep.clone()));
    let losses = score(
        &mut graph,
        [sample(0), sample(3)],
        sample(6),
        w,
        k,
        [2., 0.5],
    );
    graph.set_outputs(vec![
        losses.energy,
        losses.target_distance,
        losses.sample_distance,
        losses.valid,
    ]);
    let mut expected = [0.; 4];
    let mut gradients = [vec![0.; rows * (width + 2)], vec![0.; rows * (width + 2)]];
    for row in 0..rows {
        let range = row * (width + 2)..(row + 1) * (width + 2);
        let a = &values[0][range.clone()];
        let b = &values[1][range.clone()];
        let y = &values[2][range.clone()];
        let metric = weights[row * width..(row + 1) * width]
            .iter()
            .map(|&x| f64::from(x))
            .chain([2., 0.5])
            .collect::<Vec<_>>();
        let distance = |x: &[f64], y: &[f64]| {
            (x.iter()
                .zip(y)
                .zip(&metric)
                .map(|((&a, &b), &w)| w * (a - b).powi(2))
                .sum::<f64>()
                + f64::from(1e-6_f32))
            .sqrt()
        };
        let ay = distance(a, y);
        let by = distance(b, y);
        let ab = distance(a, b);
        let mask = f64::from(keep[row]) / rows as f64;
        expected[0] += mask * (0.5 * (ay + by) - 0.5 * ab);
        expected[1] += mask * 0.5 * (ay + by);
        expected[2] += mask * ab;
        expected[3] += mask;
        for i in 0..width + 2 {
            gradients[0][range.start + i] = f64::from(keep[row])
                * f64::from(1.0 / rows as f32)
                * 0.5
                * metric[i]
                * ((a[i] - y[i]) / ay - (a[i] - b[i]) / ab);
            gradients[1][range.start + i] = f64::from(keep[row])
                * f64::from(1.0 / rows as f32)
                * 0.5
                * metric[i]
                * ((b[i] - y[i]) / by + (a[i] - b[i]) / ab);
        }
    }
    Fixture {
        graph,
        feeds,
        parameters,
        expected,
        gradients,
    }
}

#[test]
fn joint_energy_matches_independent_f64_derivatives_masks_and_detachment() {
    for zero_mask in [false, true] {
        let f = fixture(7, 4, zero_mask);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let values = evaluate(&backward, &f.feeds).unwrap();
        for (&out, expected) in backward.outputs()[..4].iter().zip(f.expected) {
            assert!((values[out as usize].data[0] - expected).abs() < 1e-10);
        }
        let mut index = 4;
        for gradient in &f.gradients {
            for (start, end) in [(0, 4), (4, 5), (5, 6)] {
                let expected = gradient
                    .chunks_exact(6)
                    .flat_map(|row| row[start..end].iter());
                for (&a, &b) in values[backward.outputs()[index] as usize]
                    .data
                    .iter()
                    .zip(expected)
                {
                    assert!((a - b).abs() < 1e-10, "{a} != {b}");
                }
                index += 1;
            }
        }
        for &out in &backward.outputs()[index..] {
            assert!(values[out as usize].data.iter().all(|&x| x == 0.));
        }
    }
}

#[test]
fn categorical_samples_have_hard_support_and_softmax_surrogate_derivatives() {
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[2, 3]);
    let uniforms = graph.input("uniforms", &[2, 3]);
    let sample = sample_coordinate(&mut graph, logits, uniforms, &[-2., 0., 3.], 2);
    let loss = graph.sum_all(sample);
    graph.set_outputs(vec![loss, sample]);
    let backward = meganeura::autodiff::differentiate(&graph);
    let mut feeds = Feeds::new();
    let logits = [-1., 0., 1., 1., 0., -1.];
    feeds
        .set("logits", &logits)
        .set("uniforms", &[0.99, 0.1, 0.1, 0.1, 0.1, 0.99]);
    let actual = evaluate(&backward, &feeds).unwrap();
    assert_eq!(actual[backward.outputs()[1] as usize].data, [-2., 3.]);
    let mut expected = Vec::new();
    for row in logits.chunks_exact(3) {
        let probabilities = row.iter().map(|&x| f64::from(x).exp()).collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f64>();
        let mean = probabilities
            .iter()
            .zip([-2., 0., 3.])
            .map(|(&p, v)| p / total * v)
            .sum::<f64>();
        expected.extend(
            probabilities
                .iter()
                .zip([-2., 0., 3.])
                .map(|(&p, v)| p / total * (v - mean)),
        );
    }
    for (&a, b) in actual[backward.outputs()[2] as usize]
        .data
        .iter()
        .zip(expected)
    {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn expected_energy_prefers_true_two_point_distribution_over_mean_collapse() {
    let distance = |a: f64, b: f64| ((a - b).powi(2) + 1e-6).sqrt();
    let expected = |predictions: &[f64]| {
        let target = predictions
            .iter()
            .flat_map(|&x| [-1., 1.].map(|y| distance(x, y)))
            .sum::<f64>()
            / (2 * predictions.len()) as f64;
        let pair = predictions
            .iter()
            .flat_map(|&x| predictions.iter().map(move |&y| distance(x, y)))
            .sum::<f64>()
            / (predictions.len() * predictions.len()) as f64;
        target - 0.5 * pair
    };
    assert!(expected(&[-1., 1.]) < expected(&[0.]) - 0.49);
    assert!(expected(&[-1., 1.]) < expected(&[-1.]) - 0.49);
}

#[test]
fn sampled_prior_excludes_targets_and_joint_loss_reaches_shared_world_parameters() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    let mut graph = Graph::new();
    let model = WorldModel::with_behavior(&mut graph, &config, false);
    let batch = config.batch_size;
    let prefix = graph.input("prefix", &[batch, config.network().deter]);
    let logits = model.dynamics.prior.forward(&mut graph, prefix, batch);
    let states = [
        prior_continuation::state(&mut graph, &config, prefix, logits, 1, "predictive"),
        prior_continuation::state(&mut graph, &config, prefix, logits, 1, "predictive_second"),
    ];
    let mut input = |name, width| graph.input(name, &[batch, width]);
    let target = input("target", config.observation_dim());
    let previous = input("previous", config.observation_dim());
    let before_previous = input("before_previous", config.observation_dim());
    let keep = input("keep", config.action_count);
    let past_keep = input("past_keep", config.action_count);
    let uniforms = Uniforms {
        reward: [
            input("reward0", config.value_bins),
            input("reward1", config.value_bins),
        ],
        continuation: [input("continuation0", 2), input("continuation1", 2)],
        reward_target: input("reward_target", 1),
        continuation_target: input("continuation_target", 1),
    };
    for state in states {
        let dependencies = ancestors(&graph, state);
        assert!(dependencies.contains(&prefix));
        for input in [
            target,
            previous,
            before_previous,
            keep,
            past_keep,
            uniforms.reward_target,
            uniforms.continuation_target,
        ] {
            assert!(!dependencies.contains(&input));
        }
    }
    let losses = losses(
        &mut graph,
        &config,
        &model,
        states,
        target,
        previous,
        before_previous,
        past_keep,
        keep,
        uniforms,
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
fn complete_energy_world_preserves_parameter_shapes_and_compiles() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    config.batch_length = 2;
    config.world_backprop_length = 2;
    config.actor_critic_gradient = false;
    config.disagreement_bonus = false;
    config.loss_scales.policy = 0.;
    config.loss_scales.value = 0.;
    config.loss_scales.replay_value = 0.;
    let control = build_world_only_pixel_training_graph(&config, 0, PixelLoss::MotionCrossEntropy);
    let energy =
        build_world_only_pixel_training_graph(&config, 0, PixelLoss::PredictiveMotionEnergy);
    let params = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|node| match &node.op {
                Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(params(&control), params(&energy));
    assert!(
        params(&energy)
            .iter()
            .all(|(name, _)| name.starts_with("world."))
    );
    assert_eq!(energy.outputs().len(), LOSS_PREDICTIVE_WORLD + 6);
    let (plan, _) = meganeura::compile_training_graph(&energy);
    assert!(!plan.dispatches.is_empty());
}

#[test]
#[ignore = "requires guarded GPU; joint energy values and derivatives against independent F64"]
fn joint_energy_native_matches_f64() {
    let f = fixture(513, 16, false);
    let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session =
        crate::dreamer::runtime::build_session(&f.graph, &gpu, meganeura::Mode::Training, false);
    for (name, values) in &f.parameters {
        session.set_parameter(name, values);
    }
    session.step();
    session.wait();
    for (index, expected) in f.expected.into_iter().enumerate() {
        let mut value = [0.];
        session.read_output_by_index(index, &mut value);
        assert!(
            (f64::from(value[0]) - expected).abs() < 2e-5,
            "metric {index}: {} != {expected}",
            value[0]
        );
    }
    for (name, values) in f
        .parameters
        .iter()
        .filter(|(name, _)| name.starts_with("first_") || name.starts_with("second_"))
    {
        let mut actual = vec![0.; values.len()];
        session.read_param_grad(name, &mut actual);
        let expected = if name.starts_with("first_") || name.starts_with("second_") {
            let sample = usize::from(name.starts_with("second_"));
            let (start, end) = if name.ends_with("rgb") {
                (0, 16)
            } else if name.ends_with("reward") {
                (16, 17)
            } else {
                (17, 18)
            };
            f.gradients[sample]
                .chunks_exact(18)
                .flat_map(|row| row[start..end].iter().copied())
                .collect::<Vec<_>>()
        } else {
            vec![0.; values.len()]
        };
        for (&a, b) in actual.iter().zip(expected) {
            assert!((f64::from(a) - b).abs() < 2e-6, "{name}: {a} != {b}");
        }
    }
}
