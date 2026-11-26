use super::*;

const ACTION_WEIGHT: [f64; 3] = [-0.4, 0.3, 0.6];
const STATE_WEIGHT: [f64; 3] = [0.2, 0.1, -0.3];
const LOGIT_WEIGHT: [f64; 3] = [0.5, -0.8, 0.2];
const BIAS: [f64; 3] = [0.1, -0.1, 0.2];
const REWARD_WEIGHT: [f64; 3] = [0.3, -0.5, 0.7];
const CONTINUATION_WEIGHT: [f64; 3] = [-0.2, 0.1, 0.3];
const VALUE_WEIGHT: [f64; 3] = [0.1, 0.3, -0.2];
const UNIFORMS: [[f64; 3]; 3] = [[0.3, 0.7, 0.5], [0.8, 0.2, 0.4], [0.6, 0.4, 0.7]];

fn vector(graph: &mut Graph, values: &[f64; 3]) -> NodeId {
    graph.constant(values.iter().map(|&x| x as f32).collect(), &[1, 3])
}

fn dot(graph: &mut Graph, node: NodeId, values: &[f64; 3]) -> NodeId {
    let weights = vector(graph, values);
    let product = graph.mul(node, weights);
    graph.sum_inner(product)
}

fn toy_graph() -> Graph {
    let mut graph = Graph::new();
    let mut h = graph.input("initial_deter", &[1, 1]);
    let mut z = graph.input("initial_stoch", &[1, 3]);
    let mut rewards = Vec::new();
    let mut continuations = Vec::new();
    for time in 0..3 {
        let action = graph.parameter(&format!("action_{time}"), &[1, 3]);
        let previous = graph.scale(h, 0.6);
        let contribution = dot(&mut graph, action, &ACTION_WEIGHT);
        let previous = graph.add(previous, contribution);
        let contribution = dot(&mut graph, z, &STATE_WEIGHT);
        let previous = graph.add(previous, contribution);
        h = graph.tanh(previous);
        let weight = vector(&mut graph, &LOGIT_WEIGHT);
        let logits = graph.matmul(h, weight);
        let bias = vector(&mut graph, &BIAS);
        let logits = graph.add(logits, bias);
        let uniforms = graph.input(&format!("uniforms_{time}"), &[1, 3]);
        z = sample(&mut graph, logits, uniforms, 1, 3, 0.07);
        let reward = dot(&mut graph, z, &REWARD_WEIGHT);
        let linear = graph.scale(h, 0.2);
        rewards.push(graph.add(reward, linear));
        let logit = dot(&mut graph, z, &CONTINUATION_WEIGHT);
        let linear = graph.scale(h, 0.4);
        let logit = graph.add(logit, linear);
        continuations.push(graph.sigmoid(logit));
    }
    let value = dot(&mut graph, z, &VALUE_WEIGHT);
    let linear = graph.scale(h, 0.7);
    let value = graph.add(value, linear);
    let scores = objectives(&mut graph, &rewards, &continuations, value);
    outputs(&mut graph, scores);
    graph
}

#[derive(Clone)]
struct Anchor {
    hard: Vec<f64>,
    probability: Vec<f64>,
}

fn reference(actions: &[f64; 9], anchors: Option<&[Anchor]>) -> ([f64; 2], Vec<Anchor>) {
    let mut h = 0.15_f64;
    let mut z = vec![1.0, 0.0, 0.0];
    let mut live = 1.0;
    let mut total = 0.0;
    let mut draws = Vec::new();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    for time in 0..3 {
        h = (0.6 * h
            + dot(&actions[time * 3..time * 3 + 3], &ACTION_WEIGHT)
            + dot(&z, &STATE_WEIGHT))
        .tanh();
        let logits = (0..3)
            .map(|i| h * LOGIT_WEIGHT[i] + BIAS[i])
            .collect::<Vec<_>>();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
        let sum: f64 = exponentials.iter().sum();
        let p = exponentials
            .iter()
            .map(|v| 0.93 * v / sum + 0.07 / 3.0)
            .collect::<Vec<_>>();
        let index = (0..3)
            .max_by(|&a, &b| {
                let score = |i: usize| p[i].ln() - (-UNIFORMS[time][i].ln()).ln();
                score(a).total_cmp(&score(b))
            })
            .unwrap();
        let mut hard = vec![0.0; 3];
        hard[index] = 1.0;
        z = match anchors {
            Some(anchors) => (0..3)
                .map(|i| anchors[time].hard[i] + p[i] - anchors[time].probability[i])
                .collect(),
            None => hard.clone(),
        };
        draws.push(Anchor {
            hard,
            probability: p,
        });
        let reward = dot(&z, &REWARD_WEIGHT) + 0.2 * h;
        let continuation = 1.0 / (1.0 + (-(dot(&z, &CONTINUATION_WEIGHT) + 0.4 * h)).exp());
        total += live * reward;
        live *= continuation;
    }
    total += live * (dot(&z, &VALUE_WEIGHT) + 0.7 * h);
    ([live, total], draws)
}

#[test]
fn action_probe_reference_tracks_survival_and_terminal_value() {
    let graph = toy_graph();
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert_eq!(plan.param_grad_pairs.len(), 3);
    let actions = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let (base, anchors) = reference(&actions, None);
    assert_eq!(base, reference(&actions, Some(&anchors)).0);
    assert!(base[0] > 0.0 && base[0] < 1.0 && base[1].is_finite());
    for index in 0..9 {
        let mut plus = actions;
        let mut minus = actions;
        plus[index] += 1e-5;
        minus[index] -= 1e-5;
        for score in 0..2 {
            let derivative = (reference(&plus, Some(&anchors)).0[score]
                - reference(&minus, Some(&anchors)).0[score])
                / 2e-5;
            assert!(derivative.is_finite() && derivative.abs() > 1e-8);
        }
    }
}

#[test]
#[ignore = "requires separately guarded action-probe numerical qualification"]
fn action_probe_surrogate_matches_f64_and_existing_forward_graphs() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let actions = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let (expected, anchors) = reference(&actions, None);
    let batch = Batch {
        deter: vec![0.15],
        stoch: vec![1.0, 0.0, 0.0],
        actions: actions
            .chunks_exact(3)
            .map(|r| r.iter().map(|&x| x as f32).collect())
            .collect(),
        uniforms: UNIFORMS
            .iter()
            .map(|r| r.iter().map(|&x| x as f32).collect())
            .collect(),
    };
    let mut session = build_session(&toy_graph(), &gpu, Mode::Training, false);
    session.clear_optimizer();
    session.disable_grad_clip();
    for objective in 0..2 {
        let (scores, gradients) = execute(&mut session, &batch, objective);
        for score in 0..2 {
            assert!((f64::from(scores[score][0]) - expected[score]).abs() < 2e-6);
        }
        for index in 0..9 {
            let mut plus = actions;
            let mut minus = actions;
            plus[index] += 1e-5;
            minus[index] -= 1e-5;
            let derivative = (reference(&plus, Some(&anchors)).0[objective]
                - reference(&minus, Some(&anchors)).0[objective])
                / 2e-5;
            let actual = f64::from(gradients[index / 3][index % 3]);
            assert!(
                (actual - derivative).abs() < 3e-5 * (1.0 + derivative.abs()),
                "objective {objective}, action {index}: {actual} vs {derivative}"
            );
        }
    }
    drop(session);
    let config = DreamerConfig::tiny(3);
    let graph = graph(&config, 2, 3);
    let mut rng = StdRng::seed_from_u64(71);
    let weights = graph
        .nodes()
        .iter()
        .filter_map(|node| {
            let Op::Input { name } = &node.op else {
                return None;
            };
            if !name.starts_with("world.") && !name.starts_with("behavior.") {
                return None;
            }
            let values = (0..node.ty.num_elements())
                .map(|_| {
                    if name.ends_with("norm.weight") {
                        1.0
                    } else {
                        rng.random_range(-0.02..0.02)
                    }
                })
                .collect();
            Some((name.clone(), values))
        })
        .collect::<Weights>();
    let size = config.network();
    let batch = Batch {
        deter: vec![0.1; 2 * size.deter],
        stoch: (0..2 * size.stoch * size.classes)
            .map(|i| f32::from(i % size.classes == 0))
            .collect(),
        actions: vec![vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; 3],
        uniforms: (0..3)
            .map(|_| {
                (0..2 * size.stoch * size.classes)
                    .map(|_| rng.random_range(1e-5..1.0 - 1e-5))
                    .collect()
            })
            .collect(),
    };
    let mut session = build_session(&graph, &gpu, Mode::Training, false);
    session.clear_optimizer();
    session.disable_grad_clip();
    for (name, values) in &weights {
        session.set_input(name, values);
    }
    let (actual, _) = execute(&mut session, &batch, 1);
    let expected = forward_reference(&config, &gpu, &weights, &batch);
    let error = check_forward(&actual, &expected);
    let mut readback = Readback::new(Arc::clone(&gpu));
    for (name, expected) in &weights {
        let mut actual = vec![0.0; expected.len()];
        readback.read_regions(&mut [(session.input_buffer(name).unwrap(), &mut actual)]);
        assert_eq!(&actual, expected);
    }
    eprintln!(
        "Action probe: 18 F64 surrogate derivatives, two objectives, frozen inputs, zero optimizer steps; Tiny forward relative error {error}"
    );
}
