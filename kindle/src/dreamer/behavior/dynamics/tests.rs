use super::*;
use crate::dreamer::runtime::build_session;
use meganeura::{Mode, graph::Op};
use std::sync::Arc;

#[test]
fn dynamics_scale_preserves_legacy_serialization_and_checks_weighted_scale() {
    let mut config = DreamerConfig::tiny(3);
    let legacy = serde_json::to_value(&config).unwrap();
    assert!(legacy.get("actor_dynamics_scale").is_none());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
        config
    );
    for bad in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        config.actor_dynamics_scale = Some(bad);
        assert!(config.check().unwrap_err().contains("actor dynamics"));
    }
    config.actor_dynamics_scale = Some(0.5);
    assert!(config.check().is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&config).unwrap()).unwrap(),
        config
    );
    config.actor_dynamics_scale = Some(f32::MAX);
    config.loss_scales.policy = 2.0;
    assert!(config.check().unwrap_err().contains("actor dynamics"));
}

#[test]
fn dynamics_graph_only_differentiates_actor_parameters() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 1;
    config.batch_length = 2;
    config.world_backprop_length = 2;
    config.imagination_length = 3;
    let plain = super::super::build_training_graph(&config, 6, 1);
    assert!(plain.nodes().iter().all(|node| !matches!(&node.op,
        Op::Parameter { name } if name.starts_with("world."))));
    config.actor_dynamics_scale = Some(0.5);
    let mut graph = Graph::new();
    let model = super::super::BehaviorModel::new(&mut graph, &config);
    let weights = graph.input("weights", &[6, 1]);
    let loss = loss(&mut graph, &config, &model.actor.base, &model.value, weights);
    graph.set_outputs(vec![loss]);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    let mut actor_count = 0;
    let mut frozen_count = 0;
    for (name, buffer) in &plan.param_buffers {
        let gradient = plan
            .param_grad_pairs
            .iter()
            .any(|(parameter, _)| parameter == buffer);
        if name.starts_with("behavior.actor.") {
            assert!(gradient, "missing actor gradient: {name}");
            actor_count += 1;
        } else {
            assert!(!gradient, "non-actor gradient: {name}");
            frozen_count += 1;
        }
    }
    assert_eq!(actor_count, 11);
    assert_eq!(frozen_count, 11);
    assert!(
        plan.input_buffers
            .iter()
            .any(|(name, _)| name.starts_with("dynamics_parameter_world."))
    );
    let combined = super::super::build_training_graph(&config, 6, 1);
    assert_eq!(
        combined.outputs().len(),
        super::super::dynamics_actor_output(&config) + 1
    );
    let mut names = std::collections::HashSet::new();
    for node in combined.nodes() {
        if let Op::Input { name } = &node.op {
            assert!(names.insert(name), "duplicate input {name}");
        }
    }
}

const ACTOR: [f64; 6] = [0.4, -0.6, 0.2, 0.1, -0.2, 0.3];
const WORLD: [f64; 5] = [0.6, 0.4, 0.2, 0.15, 0.7];
const REWARD: [f64; 5] = [-0.2, 0.1, 0.0, 0.3, -0.1];
const CRITIC: [f64; 5] = [-0.1, 0.2, 0.0, -0.2, 0.1];
const WEIGHTS: [f64; 3] = [0.9, 0.4, 0.2];
const UNIFORMS: [[f64; 3]; 6] = [
    [0.3, 0.7, 0.5],
    [0.8, 0.2, 0.4],
    [0.6, 0.4, 0.7],
    [0.3, 0.8, 0.5],
    [0.4, 0.6, 0.9],
    [0.7, 0.3, 0.5],
];

fn probabilities(logits: &[f64], unimix: f64) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
    let sum: f64 = exponentials.iter().sum();
    exponentials
        .iter()
        .map(|x| (1.0 - unimix) * x / sum + unimix / logits.len() as f64)
        .collect()
}

#[derive(Clone)]
struct Draw {
    hard: Vec<f64>,
    probabilities: Vec<f64>,
}

// For perturbed parameters, hold hard draws and detached probabilities fixed
// at their unperturbed values. This differentiates the declared surrogate,
// rather than taking meaningless finite differences through hard argmax.
fn reference(
    actor: &[f64],
    fixed: Option<&[Draw]>,
    bins: &[f64],
    bias: &[f64],
) -> (f64, Vec<Draw>) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let mut draws = Vec::new();
    let mut draw = |logits: &[f64]| {
        let p = probabilities(logits, 0.01);
        let index = draws.len();
        let hard = if let Some(fixed) = fixed {
            fixed[index].hard.clone()
        } else {
            let chosen = (0..3)
                .max_by(|&a, &b| {
                    let score = |i: usize| p[i].ln() - (-UNIFORMS[index][i].ln()).ln();
                    score(a).total_cmp(&score(b))
                })
                .unwrap();
            (0..3)
                .map(|i| if i == chosen { 1.0 } else { 0.0 })
                .collect()
        };
        let result = hard
            .iter()
            .enumerate()
            .map(|(i, &h)| h + fixed.map_or(0.0, |fixed| p[i] - fixed[index].probabilities[i]))
            .collect::<Vec<_>>();
        draws.push(Draw {
            hard,
            probabilities: p,
        });
        result
    };
    let mut state = 0.23;
    let mut rewards = Vec::new();
    let mut continuations = Vec::new();
    let mut values = Vec::new();
    for _ in 0..3 {
        let action = draw(
            &(0..3)
                .map(|i| state * actor[i] + actor[i + 3])
                .collect::<Vec<_>>(),
        );
        state = (WORLD[0] * state + dot(&action, &[-0.8, 0.3, 0.9])).tanh();
        let latent = draw(&[-0.5 * state + 0.2, 0.2 * state - 0.1, 0.7 * state + 0.1]);
        let latent = dot(&latent, &[-0.3, 0.1, 0.8]);
        let feature = state + WORLD[1] * latent;
        let decode = |weights: &[f64]| {
            let logits = weights
                .iter()
                .zip(bias)
                .map(|(w, b)| w * feature + b)
                .collect::<Vec<_>>();
            dot(&probabilities(&logits, 0.0), bins)
        };
        rewards.push(decode(&REWARD));
        values.push(decode(&CRITIC));
        continuations.push(1.0 / (1.0 + (-WORLD[2] * feature - WORLD[4]).exp()));
        state += WORLD[3] * latent;
    }
    let mut next = values[2];
    let mut objective = 0.0;
    for time in (0..3).rev() {
        next = rewards[time] + continuations[time] * (0.3 * values[time] + 0.7 * next);
        objective -= WEIGHTS[time] * next / 6.0;
    }
    (objective, draws)
}

#[test]
#[ignore = "requires separately guarded actor dynamics F64 surrogate qualification"]
fn recurrent_dynamics_actor_matches_f64_surrogate_and_freezes_model() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let bins = TwoHotBins::new(5);
    let bias = bins
        .values()
        .iter()
        .map(|x| -x.abs().ln_1p() - if *x == 0.0 { 0.0 } else { 2.0 })
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let actor = graph.parameter("actor", &[6]);
    let actor_w = slice_columns(&mut graph, actor, 1, 6, 0, 3);
    let actor_b = slice_columns(&mut graph, actor, 1, 6, 3, 3);
    let actor_b = graph.reshape(actor_b, &[3]);
    let world = graph.parameter("world", &[5]);
    let world = graph.stop_gradient(world);
    let world = (0..5)
        .map(|i| slice_columns(&mut graph, world, 1, 5, i, 1))
        .collect::<Vec<_>>();
    let reward = graph.parameter("reward", &[1, 5]);
    let reward = graph.stop_gradient(reward);
    let critic = graph.parameter("critic", &[1, 5]);
    let critic = graph.stop_gradient(critic);
    let weight = graph.parameter("weight", &[3, 1]);
    let norm = graph.input("normalizer", &[1]);
    let bias_node = graph.constant(bias.clone(), &[5]);
    let action_effect = graph.constant(vec![-0.8, 0.3, 0.9], &[3, 1]);
    let latent_effect = graph.constant(vec![-0.3, 0.1, 0.8], &[3, 1]);
    let prior_w = graph.constant(vec![-0.5, 0.2, 0.7], &[1, 3]);
    let prior_b = graph.constant(vec![0.2, -0.1, 0.1], &[3]);
    let mut state = graph.constant(vec![0.23], &[1, 1]);
    let mut rewards = Vec::new();
    let mut continuations = Vec::new();
    let mut values = Vec::new();
    for time in 0..3 {
        let logits = graph.matmul(state, actor_w);
        let logits = graph.bias_add(logits, actor_b);
        let uniforms = graph.constant(
            UNIFORMS[2 * time].iter().map(|&x| x as f32).collect(),
            &[1, 3],
        );
        let action = sample(&mut graph, logits, uniforms, 1, 3, 0.01);
        let effect = graph.matmul(action, action_effect);
        state = graph.mul(state, world[0]);
        state = graph.add(state, effect);
        state = graph.tanh(state);
        let logits = graph.matmul(state, prior_w);
        let logits = graph.bias_add(logits, prior_b);
        let uniforms = graph.constant(
            UNIFORMS[2 * time + 1].iter().map(|&x| x as f32).collect(),
            &[1, 3],
        );
        let latent = sample(&mut graph, logits, uniforms, 1, 3, 0.01);
        let latent = graph.matmul(latent, latent_effect);
        let effect = graph.mul(latent, world[1]);
        let feature = graph.add(state, effect);
        let logits = graph.matmul(feature, reward);
        let logits = graph.bias_add(logits, bias_node);
        rewards.push(scalar_mean(&mut graph, logits, &bins));
        let logits = graph.matmul(feature, critic);
        let logits = graph.bias_add(logits, bias_node);
        values.push(scalar_mean(&mut graph, logits, &bins));
        let continuation = graph.mul(feature, world[2]);
        let continuation = graph.add(continuation, world[4]);
        continuations.push(graph.sigmoid(continuation));
        let effect = graph.mul(latent, world[3]);
        state = graph.add(state, effect);
    }
    let loss = lambda_objective(
        &mut graph,
        &rewards,
        &continuations,
        &values,
        weight,
        norm,
        0.7,
    );
    graph.set_outputs(vec![loss]);
    let mut session = build_session(&graph, &gpu, Mode::Training, false);
    for (name, values) in [
        ("actor", ACTOR.as_slice()),
        ("world", WORLD.as_slice()),
        ("reward", REWARD.as_slice()),
        ("critic", CRITIC.as_slice()),
        ("weight", WEIGHTS.as_slice()),
    ] {
        session.set_parameter(name, &values.iter().map(|&x| x as f32).collect::<Vec<_>>());
    }
    assert!(session.has_param_grad("actor"));
    session.set_input("normalizer", &[2.0]);
    let frozen = ["world", "reward", "critic", "weight"];
    assert!(frozen.iter().all(|name| !session.has_param_grad(name)));
    let before = session.read_params(&frozen);
    session.set_adam(1e-4, 0.9, 0.999, 1e-8);
    session.step();
    session.wait();
    let support = bins
        .values()
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    let bias = bias.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let (expected, draws) = reference(&ACTOR, None, &support, &bias);
    let mut actual = [0.0];
    session.read_output_by_index(0, &mut actual);
    assert!(
        (f64::from(actual[0]) - expected).abs() < 2e-5,
        "loss {} != {expected}",
        actual[0]
    );
    let mut gradients = [0.0; 6];
    session.read_param_grad("actor", &mut gradients);
    for index in 0..6 {
        let mut plus = ACTOR;
        let mut minus = ACTOR;
        plus[index] += 1e-5;
        minus[index] -= 1e-5;
        let expected = (reference(&plus, Some(&draws), &support, &bias).0
            - reference(&minus, Some(&draws), &support, &bias).0)
            / 2e-5;
        assert!(expected.abs() > 1e-5);
        assert!(
            (f64::from(gradients[index]) - expected).abs() < 2e-5 * (1.0 + expected.abs()),
            "gradient {index}: {} != {expected}",
            gradients[index]
        );
    }
    assert_eq!(session.read_params(&frozen), before);
    assert_ne!(session.read_params(&["actor"])[0], ACTOR.map(|x| x as f32));
    session.set_parameter("weight", &[0.0; 3]);
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut actual);
    session.read_param_grad("actor", &mut gradients);
    assert_eq!(actual[0], 0.0);
    assert!(gradients.iter().all(|&x| x == 0.0));
    eprintln!(
        "F64 recurrent surrogate: value, six actor derivatives, frozen model/critic and zero masks pass"
    );
    check_full_scalar_support(&gpu);
}

fn check_full_scalar_support(gpu: &Arc<blade_graphics::Context>) {
    let bins = TwoHotBins::new(255);
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[2, bins.len()]);
    let means = scalar_mean(&mut graph, logits, &bins);
    let loss = graph.mean_all(means);
    graph.set_outputs(vec![loss, means]);
    let mut session = build_session(&graph, gpu, Mode::Training, false);
    let mut logits = vec![0.0; bins.len()];
    logits.extend(bins.values().iter().enumerate().map(|(i, x)| {
        -x.abs().ln_1p() - if *x == 0.0 { 0.0 } else { 2.0 } + 0.1 * (i as f32 * 0.3).sin()
    }));
    session.set_parameter("logits", &logits);
    session.set_adam(0.0, 0.9, 0.999, 1e-8);
    session.step();
    session.wait();
    let mut actual = [0.0; 2];
    session.read_output_by_index(1, &mut actual);
    assert_eq!(
        actual[0], 0.0,
        "uniform scalar distribution must decode to exact zero"
    );
    let mut gradients = vec![0.0; logits.len()];
    session.read_param_grad("logits", &mut gradients);
    for row in 0..2 {
        let p = probabilities(
            &logits[row * bins.len()..(row + 1) * bins.len()]
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
            0.0,
        );
        let expected = p
            .iter()
            .zip(bins.values())
            .map(|(p, b)| p * f64::from(*b))
            .sum::<f64>();
        assert!((f64::from(actual[row]) - expected).abs() < 2e-5);
        for i in 0..bins.len() {
            let expected = 0.5 * p[i] * (f64::from(bins.values()[i]) - expected);
            assert!(
                (f64::from(gradients[row * bins.len() + i]) - expected).abs()
                    < 2e-5 * (1.0 + expected.abs())
            );
        }
    }
    eprintln!("F64 255-bin decoder: uniform zero, localized mean and all 510 derivatives pass");
}
