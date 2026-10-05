use super::*;
use crate::dreamer::runtime::{build_session, initialize_d3, share_matching};
use meganeura::Mode;
use std::sync::Arc;

fn gpu() -> Arc<blade_graphics::Context> {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        assert_eq!(gpu.device_information().device_name, expected);
        assert!(!gpu.device_information().is_software_emulated);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
    gpu
}

#[test]
fn zero_scale_has_no_exploration_graph_or_parameters() {
    let mut config = DreamerConfig::tiny(3);
    config.disagreement_bonus = true;
    assert!(!config.uses_disagreement());
    for graph in [
        crate::dreamer::world::build_training_graph(&config, config.batch_length),
        crate::dreamer::world::build_posterior_graph(&config),
        crate::dreamer::world::build_imagination_graph(&config),
    ] {
        assert!(!graph.nodes().iter().any(|node| matches!(&node.op,
            meganeura::graph::Op::Parameter { name } if name.starts_with("world.exploration."))));
    }
    config.intrinsic_reward_scale = -1.0;
    assert!(config.check().is_err());
    config.intrinsic_reward_scale = 1.0;
    config.visitation_bonus = true;
    assert!(config.check().is_err());
}

#[test]
fn ensemble_target_marginalizes_the_current_categorical_draw() {
    use meganeura::graph::Op;

    let mut config = DreamerConfig::tiny(3);
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    // One transition isolates its target from earlier sampled input states.
    let graph = crate::dreamer::world::build_training_graph(&config, 1);
    let mut pending = vec![graph.outputs()[crate::dreamer::world::LOSS_EXPLORATION]];
    let mut visited = std::collections::HashSet::new();
    let mut inputs = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if visited.insert(id) {
            let node = graph.node(id);
            if let Op::Input { name } = &node.op {
                inputs.insert(name.as_str());
            }
            pending.extend_from_slice(&node.inputs);
        }
    }
    assert!(inputs.contains("observation_0"));
    assert!(inputs.contains("initial_stoch"));
    assert!(!inputs.contains("posterior_sample_0"));

    let probabilities = [0.1_f64, 0.3, 0.6];
    let prediction = [-0.3_f64, 0.1, 0.7];
    for (coordinate, (&p, &x)) in probabilities.iter().zip(&prediction).enumerate() {
        let expected_gradient = probabilities
            .iter()
            .enumerate()
            .map(|(sample, &weight)| weight * 2.0 * (x - f64::from(sample == coordinate)))
            .sum::<f64>();
        assert!((expected_gradient - 2.0 * (x - p)).abs() < 1e-12);
    }
}

#[test]
fn enabled_graphs_have_exploration_heads() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = crate::ObservationKind::Rgb64;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    for graph in [
        crate::dreamer::world::build_training_graph(&config, config.batch_length),
        crate::dreamer::world::build_posterior_graph(&config),
        crate::dreamer::world::build_imagination_graph(&config),
    ] {
        assert!(graph.nodes().iter().any(|node| matches!(&node.op,
            meganeura::graph::Op::Parameter { name } if name.starts_with("world.exploration."))));
    }
}

#[test]
#[ignore = "requires GPU; independent F64 bonus, masked loss and raw derivatives"]
fn tiny_disagreement_matches_scalar_values_and_gradients() {
    const ROWS: usize = 3;
    const WIDTH: usize = 5;
    let gpu = gpu();
    let predictions = (0..MEMBERS)
        .map(|member| {
            (0..ROWS * WIDTH)
                .map(|i| match i / WIDTH {
                    0 => (i % WIDTH) as f32 * 0.1,
                    1 => 0.2 + (member * (i % WIDTH + 1)) as f32 * 1e-6,
                    _ => ((member * 11 + i * 7) % 19) as f32 * 0.07 - 0.4,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = (0..ROWS * WIDTH)
        .map(|i| (i % 7) as f32 * 0.12)
        .collect::<Vec<_>>();
    let weights = [1.0, 0.0, 0.5];
    let mut graph = Graph::new();
    let inputs = (0..MEMBERS)
        .map(|i| graph.parameter(&format!("head{i}"), &[ROWS, WIDTH]))
        .collect::<Vec<_>>();
    let bonus = disagreement(&mut graph, &inputs);
    graph.set_outputs(vec![bonus]);
    let mut inference = build_session(&graph, &gpu, Mode::Inference, false);
    for (i, values) in predictions.iter().enumerate() {
        inference.set_parameter(&format!("head{i}"), values);
    }
    inference.step();
    inference.wait();
    let mut actual = [0.0; ROWS];
    inference.read_output_by_index(0, &mut actual);
    for row in 0..ROWS {
        let expected = (0..WIDTH)
            .map(|column| {
                let values = predictions
                    .iter()
                    .map(|p| f64::from(p[row * WIDTH + column]))
                    .collect::<Vec<_>>();
                let mean = values.iter().sum::<f64>() / MEMBERS as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / MEMBERS as f64;
                (variance + 1e-8).sqrt() - 1e-4
            })
            .sum::<f64>()
            / WIDTH as f64;
        assert!((f64::from(actual[row]) - expected).abs() < 2e-6);
    }

    let target = graph.parameter("target", &[ROWS, WIDTH]);
    let weight = graph.constant(weights.to_vec(), &[ROWS, 1]);
    let loss = masked_loss(&mut graph, &inputs, target, weight);
    graph.set_outputs(vec![loss]);
    let mut training = build_session(&graph, &gpu, Mode::Training, false);
    training.set_parameter("target", &targets);
    for (i, values) in predictions.iter().enumerate() {
        training.set_parameter(&format!("head{i}"), values);
    }
    training.step();
    training.wait();
    assert!(!training.has_param_grad("target"));
    let mut expected_loss = 0.0_f64;
    let divisor = (MEMBERS * ROWS * WIDTH) as f64;
    for (head, values) in predictions.iter().enumerate() {
        let mut actual = vec![0.0; ROWS * WIDTH];
        training.read_param_grad(&format!("head{head}"), &mut actual);
        for i in 0..actual.len() {
            let residual = f64::from(values[i]) - f64::from(targets[i]);
            let weight = f64::from(weights[i / WIDTH]);
            expected_loss += weight * residual * residual / divisor;
            let gradient = 2.0 * residual * weight / divisor;
            assert!((f64::from(actual[i]) - gradient).abs() < 2e-7);
        }
    }
    let mut loss = [0.0];
    training.read_output_by_index(0, &mut loss);
    assert!((f64::from(loss[0]) - expected_loss).abs() < 2e-6);
}

#[test]
#[ignore = "requires GPU; detached ensemble learning and repeated/novel states"]
fn tiny_disagreement_learns_repeated_states_without_world_gradients() {
    let gpu = gpu();
    let config = DreamerConfig::tiny(3);
    let width = config.network().stoch * config.network().classes;
    let mut graph = Graph::new();
    let ensemble = Disagreement::new(&mut graph, &config);
    let state = graph.parameter("state", &[1, config.feature_dim()]);
    let action = graph.parameter("action", &[1, 3]);
    let target = graph.parameter("target", &[1, width]);
    let weight = graph.constant(vec![1.0], &[1, 1]);
    let loss = ensemble.loss(&mut graph, state, action, target, weight);
    graph.set_outputs(vec![loss]);
    let mut training = build_session(&graph, &gpu, Mode::Training, false);
    initialize_d3(&mut training, &graph, 113);
    let mut readout_graph = Graph::new();
    let ensemble = Disagreement::new(&mut readout_graph, &config);
    let state = readout_graph.input("state", &[2, config.feature_dim()]);
    let action = readout_graph.input("action", &[2, 3]);
    let bonus = ensemble.bonus(&mut readout_graph, state, action);
    readout_graph.set_outputs(vec![bonus]);
    let mut readout = build_session(&readout_graph, &gpu, Mode::Inference, false);
    share_matching(&mut training, &mut readout, "world.exploration.");
    let repeated = (0..config.feature_dim())
        .map(|i| (i % 11) as f32 * 0.1 - 0.5)
        .collect::<Vec<_>>();
    let novel = (0..config.feature_dim())
        .map(|i| (i * 7 % 13) as f32 * 0.1 - 0.6)
        .collect::<Vec<_>>();
    training.set_parameter("state", &repeated);
    training.set_parameter("action", &[0.0, 1.0, 0.0]);
    training.set_parameter(
        "target",
        &(0..width)
            .map(|i| f32::from(i % 4 == 0))
            .collect::<Vec<_>>(),
    );
    readout.set_input("state", &[repeated, novel].concat());
    readout.set_input("action", &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let read = |session: &mut meganeura::Session| {
        session.step();
        session.wait();
        let mut result = [0.0; 2];
        session.read_output_by_index(0, &mut result);
        result
    };
    let before = read(&mut readout);
    training.set_laprop(0.003, 0.9, 0.999, 1e-20);
    for _ in 0..256 {
        training.step();
        training.wait();
    }
    for name in ["state", "action", "target"] {
        assert!(!training.has_param_grad(name));
    }
    crate::dreamer::runtime::sync_matching(&training, &mut readout, "world.exploration.");
    let after = read(&mut readout);
    assert!(after[0] < before[0] * 0.25, "{before:?} -> {after:?}");
    assert!(after[1] > after[0] * 2.0, "{before:?} -> {after:?}");
    eprintln!("disagreement repeated/novel: {before:?} -> {after:?}");
}
