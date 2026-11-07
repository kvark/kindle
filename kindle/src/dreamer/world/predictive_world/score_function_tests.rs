use super::*;
use meganeura::reference::{Feeds, evaluate};

const ROWS: usize = 256;
const UNIMIX: f32 = 0.01;
const SUPPORT: [f32; 3] = [-1., 0., 1.];
const BLOCKS: [(&str, usize, usize); 4] = [
    ("prior", 1, 2),
    ("reward", 2, 3),
    ("continuation", 2, 2),
    ("rgb", 2, 2),
];

fn parameters() -> Vec<Vec<f32>> {
    vec![
        vec![-0.5, 0.25],
        vec![0.25, -0.5, 0.75, -0.25, 0.5, -0.75],
        vec![-0.5, 0.25, 0.5, -0.25],
        vec![-0.25, 0.375, 0.25, -0.125],
    ]
}

fn probabilities(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
    let total = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= total;
    }
    p
}

fn reference(parameters: &[Vec<f64>]) -> f64 {
    let mut prior = probabilities(&parameters[0]);
    for p in &mut prior {
        *p = *p * f64::from(1. - UNIMIX) + f64::from(UNIMIX / 2.);
    }
    let normalizer = prior.iter().sum::<f64>();
    for p in &mut prior {
        *p /= normalizer;
    }
    let mut worlds = Vec::new();
    for (z, &pz) in prior.iter().enumerate() {
        let reward = probabilities(&parameters[1][z * 3..z * 3 + 3]);
        let continuation = probabilities(&parameters[2][z * 2..z * 2 + 2]);
        for (ri, &pr) in reward.iter().enumerate() {
            for (ci, &pc) in continuation.iter().enumerate() {
                worlds.push((
                    pz * pr * pc,
                    [
                        parameters[3][z * 2],
                        parameters[3][z * 2 + 1],
                        f64::from(SUPPORT[ri]),
                        ci as f64,
                    ],
                ));
            }
        }
    }
    let distance = |a: [f64; 4], b: [f64; 4]| {
        (a.into_iter()
            .zip(b)
            .zip([1., 2., 1.25, 0.75])
            .map(|((a, b), w)| w * (a - b).powi(2))
            .sum::<f64>()
            + f64::from(1e-6_f32))
        .sqrt()
    };
    let target = [0.125, -0.25, 0., 1.];
    let cross = worlds
        .iter()
        .map(|(p, x)| p * distance(*x, target))
        .sum::<f64>();
    let mut pair = 0.;
    for (p, x) in &worlds {
        for (q, y) in &worlds {
            pair += p * q * distance(*x, *y);
        }
    }
    cross - 0.5 * pair
}

fn expected() -> (f64, Vec<Vec<f64>>) {
    let values = parameters()
        .iter()
        .map(|row| row.iter().copied().map(f64::from).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut gradient = values.clone();
    for block in 0..values.len() {
        for column in 0..values[block].len() {
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[block][column] += 1e-5;
            minus[block][column] -= 1e-5;
            gradient[block][column] = (reference(&plus) - reference(&minus)) / 2e-5;
        }
    }
    (reference(&values), gradient)
}

fn fixture(score_function: bool, zero_mask: bool) -> (Graph, Feeds, Vec<Vec<f32>>) {
    let values = parameters();
    let mut graph = Graph::new();
    let mut feeds = Feeds::new();
    let nodes = BLOCKS
        .into_iter()
        .zip(&values)
        .map(|((name, m, n), values)| {
            feeds.set(name, values);
            graph.parameter(name, &[m, n])
        })
        .collect::<Vec<_>>();
    let prior_logits = graph.broadcast_to(nodes[0], &[ROWS, 2]);
    let mut log_probabilities = Vec::new();
    let mut samples = Vec::new();
    for draw in 0..2 {
        let mut uniform = [
            vec![1e-6; ROWS * 2],
            vec![1e-6; ROWS * 3],
            vec![1e-6; ROWS * 2],
        ];
        for row in 0..ROWS {
            let world = if row < 144 {
                if draw == 0 { row / 12 } else { row % 12 }
            } else {
                0
            };
            let (z, reward, continuation) = (world / 6, (world % 6) / 2, world % 2);
            uniform[0][row * 2 + z] = 0.999999;
            uniform[1][row * 3 + reward] = 0.999999;
            uniform[2][row * 2 + continuation] = 0.999999;
        }
        let mut inputs = Vec::new();
        for (i, width) in [2, 3, 2].into_iter().enumerate() {
            let name = format!("uniforms_{draw}_{i}");
            let node = graph.input(&name, &[ROWS, width]);
            feeds.set(&name, &uniform[i]);
            inputs.push(node);
        }
        let (z, z_logp) = categorical(
            &mut graph,
            prior_logits,
            inputs[0],
            ROWS,
            2,
            UNIMIX,
            score_function,
        );
        let reward_logits = graph.matmul(z, nodes[1]);
        let continuation_logits = graph.matmul(z, nodes[2]);
        let (reward, r_logp) = sample_coordinate(
            &mut graph,
            reward_logits,
            inputs[1],
            &SUPPORT,
            ROWS,
            score_function,
        );
        let (continuation, c_logp) = sample_coordinate(
            &mut graph,
            continuation_logits,
            inputs[2],
            &[0., 1.],
            ROWS,
            score_function,
        );
        let rgb = graph.matmul(z, nodes[3]);
        samples.push(Sample {
            rgb,
            reward,
            continuation,
        });
        // Enumeration weights are numeric probabilities, detached by score().
        let hard_z = gumbel_sample(&mut graph, prior_logits, inputs[0], ROWS, 2, UNIMIX);
        let z_logs = mixed_log_probabilities(&mut graph, prior_logits, ROWS, 2, UNIMIX);
        let selected = graph.mul(hard_z, z_logs);
        let z_weight = graph.sum_inner(selected);
        let hard_r = gumbel_sample(&mut graph, reward_logits, inputs[1], ROWS, 3, 0.);
        let r_logs = graph.log_softmax(reward_logits);
        let selected = graph.mul(hard_r, r_logs);
        let r_weight = graph.sum_inner(selected);
        let hard_c = gumbel_sample(&mut graph, continuation_logits, inputs[2], ROWS, 2, 0.);
        let c_logs = graph.log_softmax(continuation_logits);
        let selected = graph.mul(hard_c, c_logs);
        let c_weight = graph.sum_inner(selected);
        let heads = graph.add(r_weight, c_weight);
        let weight = graph.add(z_weight, heads);
        let gradient_logp = z_logp.map(|z| {
            let heads = graph.add(r_logp.unwrap(), c_logp.unwrap());
            graph.add(z, heads)
        });
        log_probabilities.push((weight, gradient_logp));
    }
    let joint = graph.add(log_probabilities[0].0, log_probabilities[1].0);
    let probability = graph.exp(joint);
    let prior = mixed_probabilities(&mut graph, prior_logits, ROWS, 2, UNIMIX);
    let normalizer = graph.sum_inner(prior);
    let normalizer = graph.mul(normalizer, normalizer);
    let probability = graph.div(probability, normalizer);
    let probability = graph.scale(probability, ROWS as f32);
    let mask = graph.constant(
        (0..ROWS)
            .map(|i| f32::from(i < 144 && !zero_mask))
            .collect(),
        &[ROWS, 1],
    );
    let keep = graph.mul(probability, mask);
    let target = Sample {
        rgb: graph.constant([0.125, -0.25].repeat(ROWS), &[ROWS, 2]),
        reward: graph.constant(vec![0.; ROWS], &[ROWS, 1]),
        continuation: graph.constant(vec![1.; ROWS], &[ROWS, 1]),
    };
    let weights = graph.constant([1., 2.].repeat(ROWS), &[ROWS, 2]);
    let logp = score_function.then(|| {
        [
            log_probabilities[0].1.unwrap(),
            log_probabilities[1].1.unwrap(),
        ]
    });
    let losses = score(
        &mut graph,
        [samples[0], samples[1]],
        target,
        weights,
        keep,
        [1.25, 0.75],
        logp,
    );
    let loss = losses.energy;
    graph.set_outputs(vec![
        loss,
        losses.target_distance,
        losses.sample_distance,
        losses.valid,
    ]);
    (graph, feeds, values)
}

#[test]
fn conditional_categorical_energy_matches_exact_enumeration_and_finite_differences() {
    let (expected, gradients) = expected();
    let mut old_error = 0.;
    for corrected in [false, true] {
        let (graph, feeds, _) = fixture(corrected, false);
        let backward = meganeura::autodiff::differentiate(&graph);
        let values = evaluate(&backward, &feeds).unwrap();
        assert!((values[backward.outputs()[0] as usize].data[0] - expected).abs() < 1e-10);
        for (&out, expected) in backward.outputs()[4..].iter().zip(&gradients) {
            for (&actual, &expected) in values[out as usize].data.iter().zip(expected) {
                if corrected {
                    assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
                } else {
                    old_error += (actual - expected).powi(2);
                }
            }
        }
    }
    assert!(old_error.sqrt() > 0.01);
}

#[test]
fn zero_mask_cancels_score_function_and_direct_gradients() {
    let (graph, feeds, _) = fixture(true, true);
    let backward = meganeura::autodiff::differentiate(&graph);
    let actual = evaluate(&backward, &feeds).unwrap();
    for &out in backward.outputs() {
        assert!(actual[out as usize].data.iter().all(|&v| v == 0.));
    }
}

#[test]
fn corrected_prior_stops_sample_derivatives_and_returns_grouped_log_probability() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    let mut graph = Graph::new();
    let logits = graph.parameter(
        "logits",
        &[
            config.batch_size * config.network().stoch,
            config.network().classes,
        ],
    );
    let deter = graph.input("deter", &[config.batch_size, config.network().deter]);
    let (state, logp) = state(&mut graph, &config, deter, logits, 0, "predictive", true);
    let loss = graph.sum_all(state);
    graph.set_outputs(vec![loss]);
    let backward = meganeura::autodiff::differentiate(&graph);
    assert_ne!(
        backward.node(*backward.outputs().last().unwrap()).ty.shape,
        graph.node(logits).ty.shape
    );
    assert!(logp.is_some());
}

#[test]
#[ignore = "requires guarded GPU; conditional categorical energy and gradients against exact enumeration"]
fn conditional_categorical_energy_native_matches_f64() {
    let (graph, feeds, values) = fixture(true, false);
    let (expected, gradients) = expected();
    let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session =
        crate::dreamer::runtime::build_session(&graph, &gpu, meganeura::Mode::Training, false);
    for ((name, _, _), values) in BLOCKS.into_iter().zip(values) {
        session.set_parameter(name, &values);
    }
    for draw in 0..2 {
        for i in 0..3 {
            let name = format!("uniforms_{draw}_{i}");
            session.set_input(&name, &feeds.f32(&name).unwrap());
        }
    }
    session.step();
    session.wait();
    let mut actual = [0.];
    session.read_output_by_index(0, &mut actual);
    assert!(
        (f64::from(actual[0]) - expected).abs() < 2e-5,
        "{} != {expected}",
        actual[0]
    );
    for ((name, _, _), expected) in BLOCKS.into_iter().take(4).zip(gradients) {
        let mut actual = vec![0.; expected.len()];
        session.read_param_grad(name, &mut actual);
        for (&actual, expected) in actual.iter().zip(expected) {
            assert!(
                (f64::from(actual) - expected).abs() < 2e-5,
                "{name}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn symmetric_reward_variance_counterexample_receives_the_true_score_gradient() {
    let logits_values = [-std::f32::consts::LN_2, 0., -std::f32::consts::LN_2];
    let p = probabilities(&logits_values.map(f64::from));
    let distance = |a: f64, b: f64| ((a - b).powi(2) + f64::from(1e-6_f32)).sqrt();
    let direct = (0..3)
        .map(|i| {
            distance(f64::from(SUPPORT[i]), 0.)
                - (0..3)
                    .map(|j| p[j] * distance(f64::from(SUPPORT[i]), f64::from(SUPPORT[j])))
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mean = p.iter().zip(&direct).map(|(p, d)| p * d).sum::<f64>();
    let expected = p
        .iter()
        .zip(&direct)
        .map(|(p, d)| p * (d - mean))
        .collect::<Vec<_>>();
    assert!(expected[0] > 0.06 && expected[1] < -0.12 && expected[2] > 0.06);
    for corrected in [false, true] {
        let mut graph = Graph::new();
        let logits = graph.parameter("logits", &[1, 3]);
        let logits = graph.broadcast_to(logits, &[16, 3]);
        let mut samples = Vec::new();
        let mut logps = Vec::new();
        for draw in 0..2 {
            let mut uniform = vec![1e-6; 16 * 3];
            for row in 0..16 {
                let index = if row < 9 {
                    if draw == 0 { row / 3 } else { row % 3 }
                } else {
                    0
                };
                uniform[row * 3 + index] = 0.999999;
            }
            let uniforms = graph.constant(uniform, &[16, 3]);
            let (reward, logp) =
                sample_coordinate(&mut graph, logits, uniforms, &SUPPORT, 16, corrected);
            let logp = logp.unwrap_or_else(|| {
                categorical(&mut graph, logits, uniforms, 16, 3, 0., true)
                    .1
                    .unwrap()
            });
            samples.push(Sample {
                rgb: graph.constant(vec![0.; 16], &[16, 1]),
                reward,
                continuation: graph.constant(vec![0.; 16], &[16, 1]),
            });
            logps.push(logp);
        }
        let joint = graph.add(logps[0], logps[1]);
        let weight = graph.exp(joint);
        let weight = graph.scale(weight, 16.);
        let mask = graph.constant((0..16).map(|i| f32::from(i < 9)).collect(), &[16, 1]);
        let keep = graph.mul(weight, mask);
        let target = Sample {
            rgb: graph.constant(vec![0.; 16], &[16, 1]),
            reward: graph.constant(vec![0.; 16], &[16, 1]),
            continuation: graph.constant(vec![0.; 16], &[16, 1]),
        };
        let weights = graph.constant(vec![1.; 16], &[16, 1]);
        let loss = score(
            &mut graph,
            [samples[0], samples[1]],
            target,
            weights,
            keep,
            [1., 1.],
            corrected.then(|| [logps[0], logps[1]]),
        );
        graph.set_outputs(vec![loss.energy]);
        let backward = meganeura::autodiff::differentiate(&graph);
        let mut feeds = Feeds::new();
        feeds.set("logits", &logits_values);
        let actual = evaluate(&backward, &feeds).unwrap();
        for (&a, &b) in actual[backward.outputs()[1] as usize]
            .data
            .iter()
            .zip(&expected)
        {
            assert!(
                (a - if corrected { b } else { 0. }).abs() < 1e-10,
                "{corrected}: {a} != {b}"
            );
        }
    }
}

#[test]
fn grouped_prior_score_gradient_matches_mixed_categorical_reference_with_mask() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 2;
    let size = config.network();
    let rows = config.batch_size * size.stoch;
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[rows, size.classes]);
    let deter = graph.input("deter", &[config.batch_size, size.deter]);
    let (_, logp) = state(&mut graph, &config, deter, logits, 0, "predictive", true);
    let mask = graph.constant(vec![0., 1.], &[2, 1]);
    let value = graph.mul(logp.unwrap(), mask);
    let loss = graph.mean_all(value);
    graph.set_outputs(vec![loss]);
    let backward = meganeura::autodiff::differentiate(&graph);
    let mut uniforms = vec![1e-6; rows * size.classes];
    for row in 0..rows {
        uniforms[row * size.classes + row % size.classes] = 0.999999;
    }
    let mut feeds = Feeds::new();
    feeds
        .set("logits", &vec![0.; rows * size.classes])
        .set("deter", &vec![0.; 2 * size.deter])
        .set("predictive_uniforms_0", &uniforms);
    let actual = evaluate(&backward, &feeds).unwrap();
    let probability = f64::from(1. - config.unimix) / size.classes as f64
        + f64::from(config.unimix / size.classes as f32);
    let expected_loss = 0.5 * size.stoch as f64 * probability.ln();
    assert!((actual[backward.outputs()[0] as usize].data[0] - expected_loss).abs() < 1e-10);
    let factor = 0.5 * f64::from(1. - config.unimix) / (size.classes as f64 * probability);
    for (index, &actual) in actual[backward.outputs()[1] as usize]
        .data
        .iter()
        .enumerate()
    {
        let row = index / size.classes;
        let column = index % size.classes;
        let expected = if row < size.stoch {
            0.
        } else {
            factor * (f64::from(column == row % size.classes) - 1. / size.classes as f64)
        };
        assert!(
            (actual - expected).abs() < 1e-10,
            "{row}/{column}: {actual} != {expected}"
        );
    }
}
