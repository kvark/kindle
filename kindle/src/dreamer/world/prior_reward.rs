use super::*;
use meganeura::graph::Op;
use std::collections::HashSet;

fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.cdp_centered = true;
    config.batch_length = 2;
    config.world_backprop_length = 2;
    config.prior_reward_scale = Some(1.0);
    config
}

#[test]
fn prior_reward_excludes_arrival_but_reaches_dynamics_and_reward() {
    let mut graph = build_training_graph(&config(), 1);
    graph.set_outputs(vec![graph.outputs()[LOSS_PRIOR_REWARD]]);
    let mut dependencies = HashSet::new();
    let mut pending = graph.outputs().to_vec();
    while let Some(id) = pending.pop() {
        if dependencies.insert(id) {
            pending.extend(&graph.node(id).inputs);
        }
    }
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            match name.as_str() {
                "observation_0" | "posterior_sample_0" | "continuation_target_0" => {
                    assert!(!dependencies.contains(&node.id), "arrival input {name}");
                }
                "previous_action_0"
                | "prior_reward_uniforms_0"
                | "reward_target_0"
                | "keep_action_0" => {
                    assert!(dependencies.contains(&node.id), "prior input {name}");
                }
                _ => {}
            }
        }
    }
    let backward = meganeura::autodiff::differentiate(&graph);
    for (node, &gradient) in graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::Parameter { .. }))
        .zip(&backward.outputs()[1..])
    {
        let Op::Parameter { name } = &node.op else {
            unreachable!()
        };
        if name.starts_with("world.dynamics.") || name.starts_with("world.reward.") {
            assert_eq!(
                backward.node(gradient).ty.shape,
                node.ty.shape,
                "connected {name}"
            );
        } else {
            assert_ne!(
                backward.node(gradient).ty.shape,
                node.ty.shape,
                "detached {name}"
            );
        }
    }
}

#[test]
fn prior_reward_preserves_parameters_and_inference_and_compiles() {
    let enabled = config();
    let mut disabled = enabled.clone();
    disabled.prior_reward_scale = None;
    let graph = build_training_graph(&enabled, 1);
    let control = build_training_graph(&disabled, 1);
    let parameters = |graph: &Graph| {
        graph
            .nodes()
            .iter()
            .filter_map(|node| {
                if let Op::Parameter { name } = &node.op {
                    Some((name.clone(), node.ty.shape.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(parameters(&graph), parameters(&control));
    for builder in [build_observe_graph, build_imagination_head_graph] {
        assert_eq!(
            format!("{:?}", builder(&enabled, 2).nodes()),
            format!("{:?}", builder(&disabled, 2).nodes())
        );
    }
    assert_eq!(
        format!("{:?}", build_imagination_graph(&enabled).nodes()),
        format!("{:?}", build_imagination_graph(&disabled).nodes())
    );
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert!(!plan.dispatches.is_empty());
    let staged = build_staged_training_graph(&enabled);
    assert_eq!(staged.outputs().len(), PRIOR_REWARD_VALID_FRACTION + 3);
    let (plan, _) = meganeura::compile_training_graph(&staged);
    assert!(!plan.dispatches.is_empty());
}

#[test]
#[ignore = "requires guarded prior-reward F64 fixture"]
fn prior_reward_masked_ce_and_gradients_match_f64() {
    use crate::dreamer::runtime::build_session;
    let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!gpu.device_information().is_software_emulated);
    let classes = 255;
    let mut maximum_loss_error = 0.0_f64;
    let mut maximum_gradient_error = 0.0_f64;
    for rows in [1, 17, 256, 513] {
        let logits = (0..rows * classes)
            .map(|i| match i % 113 {
                0 => -30.0,
                1 => 30.0,
                _ => (i as f32 * 0.17).sin() * 3.0,
            })
            .collect::<Vec<_>>();
        let mut targets = vec![0.0; rows * classes];
        for i in 0..rows {
            let bin = i * 7 % (classes - 1);
            targets[i * classes + bin] = 0.25;
            targets[i * classes + bin + 1] = 0.75;
        }
        let keep = (0..rows).map(|i| f32::from(i % 5 != 0)).collect::<Vec<_>>();
        let mut graph = Graph::new();
        let z = graph.parameter("logits", &[rows, classes]);
        let target = graph.input("target", &[rows, classes]);
        let mask = graph.input("mask", &[rows, 1]);
        let loss = weighted_cross_entropy(&mut graph, z, target, mask);
        let loss = graph.scale(loss, 0.5);
        graph.set_outputs(vec![loss]);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        session.set_parameter("logits", &logits);
        session.set_input("target", &targets);
        session.set_input("mask", &keep);
        session.step();
        session.wait();
        let mut actual = [0.0];
        let mut gradients = vec![0.0; logits.len()];
        session.read_output_by_index(0, &mut actual);
        session.read_param_grad("logits", &mut gradients);
        let mut expected = 0.0;
        for i in 0..rows {
            let row = &logits[i * classes..(i + 1) * classes];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum = row.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>();
            for (j, &logit) in row.iter().enumerate() {
                let index = i * classes + j;
                let log_probability = f64::from(logit) - max - sum.ln();
                let weight = 0.5 * f64::from(keep[i]) / rows as f64;
                expected -= weight * f64::from(targets[index]) * log_probability;
                let gradient = weight * (log_probability.exp() - f64::from(targets[index]));
                let error = (f64::from(gradients[index]) - gradient).abs();
                maximum_gradient_error = maximum_gradient_error.max(error);
                assert!(gradients[index].is_finite() && error <= 1e-7 + 3e-6 * gradient.abs());
                if keep[i] == 0.0 {
                    assert_eq!(gradients[index], 0.0);
                }
            }
        }
        let error = (f64::from(actual[0]) - expected).abs();
        maximum_loss_error = maximum_loss_error.max(error);
        assert!(actual[0].is_finite() && error <= 5e-5);
    }
    println!(
        "{}",
        serde_json::json!({"status":"passed","cases":4,"maximum_loss_error":maximum_loss_error,"maximum_gradient_error":maximum_gradient_error})
    );
}

#[test]
#[ignore = "requires guarded complete prior-reward gradient fixture"]
fn prior_reward_complete_gradients_arrival_invariance_and_reset() {
    use crate::dreamer::runtime::{build_session, initialize_d3};
    let config = config();
    let mut graph = build_training_graph(&config, 1);
    graph.set_outputs(vec![graph.outputs()[LOSS_PRIOR_REWARD]]);
    let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!gpu.device_information().is_software_emulated);
    let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
    initialize_d3(&mut session, &graph, 19);
    // A fresh reward output is zero-initialized; use a nonzero head to check upstream routing.
    let count = session.param_size("world.reward.out.weight").unwrap();
    session.set_parameter(
        "world.reward.out.weight",
        &(0..count)
            .map(|i| (i as f32 * 0.13).sin() * 0.02)
            .collect::<Vec<_>>(),
    );
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            let count = node.ty.shape.iter().product::<usize>();
            let value = if name.starts_with("keep_") {
                1.0
            } else if name.starts_with("prior_reward_uniforms_") {
                0.37
            } else {
                0.1
            };
            let mut values = vec![value; count];
            if name.starts_with("reward_target_") {
                values.fill(0.0);
                for row in values.chunks_mut(config.value_bins) {
                    row[config.value_bins / 2 + 1] = 1.0;
                }
            }
            session.set_input(name, &values);
        }
    }
    session.step();
    session.wait();
    let mut before = [0.0];
    session.read_output_by_index(0, &mut before);
    assert!(before[0].is_finite() && before[0] > 0.0);
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for prefix in [
        "world.dynamics.prior",
        "world.dynamics.core.",
        "world.reward.",
    ] {
        let mut total = 0.0_f64;
        let mut count = 0;
        for name in names.iter().filter(|n| n.starts_with(prefix)) {
            assert!(session.has_param_grad(name));
            let mut grad = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut grad);
            assert!(grad.iter().all(|v| v.is_finite()));
            total += grad.iter().map(|v| f64::from(v.abs())).sum::<f64>();
            count += 1;
        }
        assert!(count > 0 && total > 0.0, "gradient prefix {prefix}");
    }
    session.set_input(
        "observation_0",
        &vec![0.9; config.batch_size * config.observation_dim()],
    );
    session.set_input(
        "posterior_sample_0",
        &vec![0.8; config.batch_size * config.network().stoch * config.network().classes],
    );
    session.step();
    session.wait();
    let mut after = [0.0];
    session.read_output_by_index(0, &mut after);
    assert_eq!(before, after);
    session.set_input(
        "keep_action_0",
        &vec![0.0; config.batch_size * config.action_count],
    );
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut after);
    assert_eq!(after, [0.0]);
    for name in names {
        if session.has_param_grad(&name) {
            let mut grad = vec![0.0; session.param_size(&name).unwrap()];
            session.read_param_grad(&name, &mut grad);
            assert!(grad.iter().all(|&v| v == 0.0), "masked gradient {name}");
        }
    }
    println!(
        "{}",
        serde_json::json!({"status":"passed","loss":before[0],"arrival_invariant":true,"reset_loss_and_gradients_zero":true})
    );
}
