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
fn ensemble_target_uses_the_encoding_not_the_posterior() {
    use meganeura::graph::Op;

    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = crate::ObservationKind::Rgb64;
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    // One transition isolates its target from earlier sampled input states.
    let graph = crate::dreamer::world::build_training_graph(&config, 1);
    let mut pending = vec![graph.outputs()[crate::dreamer::world::LOSS_EXPLORATION]];
    let mut visited = std::collections::HashSet::new();
    let mut inputs = std::collections::HashSet::new();
    let mut parameters = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if visited.insert(id) {
            let node = graph.node(id);
            if let Op::Input { name } = &node.op {
                inputs.insert(name.as_str());
            }
            if let Op::Parameter { name } = &node.op {
                parameters.insert(name.as_str());
                if name.starts_with("world.exploration.") && name.ends_with(".out.bias") {
                    assert_eq!(node.ty.shape, [config.encoded_observation_dim()]);
                }
            }
            pending.extend_from_slice(&node.inputs);
        }
    }
    assert!(inputs.contains("observation_0"));
    assert!(inputs.contains("initial_stoch"));
    assert!(!inputs.contains("posterior_sample_0"));
    assert!(parameters.contains("world.representation.encoder.cnn0.weight"));
    assert!(
        !parameters
            .iter()
            .any(|name| name.starts_with("world.representation.posterior."))
    );
}

#[test]
#[ignore = "requires GPU; full embedding-target graph keeps world parameters detached"]
fn tiny_embedding_disagreement_has_no_world_parameter_gradients() {
    use meganeura::graph::Op;

    let gpu = gpu();
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = crate::ObservationKind::Rgb64;
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    let mut graph = crate::dreamer::world::build_training_graph(&config, 1);
    graph.set_outputs(vec![
        graph.outputs()[crate::dreamer::world::LOSS_EXPLORATION],
    ]);
    let backward = meganeura::autodiff::differentiate(&graph);
    for (parameter, &gradient) in graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::Parameter { .. }))
        .zip(&backward.outputs()[1..])
    {
        if let Op::Parameter { name } = &parameter.op
            && !name.starts_with("world.exploration.")
        {
            assert!(
                matches!(&backward.node(gradient).op,
                Op::Constant { data } if data == &[0.0]),
                "{name}"
            );
        }
    }
    let session = build_session(&graph, &gpu, Mode::Training, false);
    let mut encoder = 0;
    let mut ensemble = 0;
    for name in session.param_names() {
        if name.starts_with("world.exploration.") {
            assert!(session.has_param_grad(name), "{name}");
            ensemble += 1;
        } else {
            // Autodiff's dead-parameter zero sentinel can have the same shape
            // as an unused scalar parameter, hence still own a gradient buffer.
            if session.has_param_grad(name) {
                assert_eq!(session.param_size(name), Some(1), "{name}");
                let mut value = [f32::NAN];
                session.read_param_grad(name, &mut value);
                assert_eq!(value, [0.0], "{name}");
            }
            encoder += usize::from(name.starts_with("world.representation.encoder."));
        }
    }
    assert!(encoder > 0 && ensemble > 0);
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
#[ignore = "requires GPU; independent action centering, state-offset and permutation invariance"]
fn tiny_action_effect_disagreement_matches_reference() {
    const ROWS: usize = 3;
    const ACTIONS: usize = 3;
    const WIDTH: usize = 5;
    let gpu = gpu();
    let predictions = (0..MEMBERS)
        .map(|member| {
            (0..ROWS * ACTIONS * WIDTH)
                .map(|i| {
                    let row = i / (ACTIONS * WIDTH);
                    let action = i / WIDTH % ACTIONS;
                    let column = i % WIDTH;
                    let effect = match row {
                        0 => 0.0,
                        1 => action as f32 * 0.03,
                        _ => (member * (action + 1) * (column + 1)) as f32 * 0.007,
                    };
                    (member * 7 + row * 3 + column) as f32 * 0.1 + effect
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let inputs = (0..MEMBERS)
        .map(|i| graph.input(&format!("head{i}"), &[ROWS * ACTIONS, WIDTH]))
        .collect::<Vec<_>>();
    let bonus = action_effect_disagreement(&mut graph, &inputs, ACTIONS);
    graph.set_outputs(vec![bonus]);
    let mut session = build_session(&graph, &gpu, Mode::Inference, false);
    let read = |session: &mut meganeura::Session, values: &[Vec<f32>]| {
        for (i, value) in values.iter().enumerate() {
            session.set_input(&format!("head{i}"), value);
        }
        session.step();
        session.wait();
        let mut output = vec![0.0; ROWS * ACTIONS];
        session.read_output_by_index(0, &mut output);
        output
    };
    let actual = read(&mut session, &predictions);
    for row in 0..ROWS {
        for action in 0..ACTIONS {
            let mut expected = 0.0;
            for column in 0..WIDTH {
                let effects = predictions
                    .iter()
                    .map(|p| {
                        let mean = (0..ACTIONS)
                            .map(|a| f64::from(p[(row * ACTIONS + a) * WIDTH + column]))
                            .sum::<f64>()
                            / ACTIONS as f64;
                        f64::from(p[(row * ACTIONS + action) * WIDTH + column]) - mean
                    })
                    .collect::<Vec<_>>();
                let mean = effects.iter().sum::<f64>() / MEMBERS as f64;
                let variance =
                    effects.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / MEMBERS as f64;
                expected += ((variance + 1e-8).sqrt() - 1e-4) / WIDTH as f64;
            }
            assert!((f64::from(actual[row * ACTIONS + action]) - expected).abs() < 2e-6);
        }
    }
    assert!(actual[..2 * ACTIONS].iter().all(|v| v.abs() < 2e-6));
    let permutation = [2, 0, 1];
    let shifted = predictions
        .iter()
        .enumerate()
        .map(|(member, p)| {
            (0..p.len())
                .map(|i| {
                    let row = i / (ACTIONS * WIDTH);
                    let action = permutation[i / WIDTH % ACTIONS];
                    p[(row * ACTIONS + action) * WIDTH + i % WIDTH]
                        + (member + row + 1) as f32 * 0.2
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let shifted = read(&mut session, &shifted);
    for row in 0..ROWS {
        for action in 0..ACTIONS {
            assert!(
                (shifted[row * ACTIONS + action] - actual[row * ACTIONS + permutation[action]])
                    .abs()
                    < 2e-6
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; detached ensemble learning and repeated/novel states"]
fn tiny_disagreement_learns_repeated_states_without_world_gradients() {
    let gpu = gpu();
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = crate::ObservationKind::Rgb64;
    let width = config.encoded_observation_dim();
    let mut graph = Graph::new();
    let ensemble = Disagreement::new(&mut graph, &config);
    let state = graph.parameter("state", &[3, config.feature_dim()]);
    let action = graph.parameter("action", &[3, 3]);
    let target = graph.parameter("target", &[3, width]);
    let weight = graph.constant(vec![1.0; 3], &[3, 1]);
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
    training.set_parameter("state", &repeated.repeat(3));
    training.set_parameter("action", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    training.set_parameter(
        "target",
        &(0..3 * width)
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
