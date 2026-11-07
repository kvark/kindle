use super::*;

pub(super) fn state(
    graph: &mut Graph,
    config: &DreamerConfig,
    deter: NodeId,
    logits: NodeId,
    time: usize,
    name: &str,
) -> NodeId {
    let size = config.network();
    let rows = config.batch_size * size.stoch;
    let uniforms = graph.input(&format!("{name}_uniforms_{time}"), &[rows, size.classes]);
    let sample = gumbel_sample(graph, logits, uniforms, rows, size.classes, config.unimix);
    let sample = graph.stop_gradient(sample);
    let probabilities = mixed_probabilities(graph, logits, rows, size.classes, config.unimix);
    let stoch = straight_through_sample(graph, sample, probabilities);
    feature(graph, deter, stoch, config.batch_size, config)
}

pub(super) fn loss(graph: &mut Graph, logits: NodeId, target: NodeId, keep: NodeId) -> NodeId {
    let positive = graph.softplus(logits, 1.0);
    let product = graph.mul(logits, target);
    let per_row = graph.sub(positive, product);
    let masked = graph.mul(per_row, keep);
    graph.mean_all(masked)
}

#[cfg(test)]
mod tests {
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
        config.prior_continuation_scale = Some(1.0);
        config
    }

    fn ancestors(graph: &Graph, root: NodeId) -> HashSet<NodeId> {
        let mut found = HashSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if found.insert(id) {
                pending.extend(&graph.node(id).inputs);
            }
        }
        found
    }

    #[test]
    fn prior_loss_excludes_arrival_observation_but_reaches_prior_and_head() {
        let config = config();
        let mut graph = build_training_graph(&config, 1);
        graph.set_outputs(vec![graph.outputs()[LOSS_PRIOR_CONTINUATION]]);
        let dependencies = ancestors(&graph, graph.outputs()[0]);
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op {
                if name == "observation_0" || name == "posterior_sample_0" {
                    assert!(!dependencies.contains(&node.id), "arrival input {name}");
                } else if name == "previous_action_0" || name == "prior_continuation_uniforms_0" {
                    assert!(dependencies.contains(&node.id), "prior input {name}");
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
            if name.starts_with("world.dynamics.") || name.starts_with("world.continuation.") {
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
    fn auxiliary_preserves_parameters_inference_and_compiles_both_training_paths() {
        let enabled = config();
        let mut disabled = enabled.clone();
        disabled.prior_continuation_scale = None;
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
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
        let staged = build_staged_training_graph(&enabled);
        assert_eq!(
            staged.outputs().len(),
            PRIOR_CONTINUATION_VALID_FRACTION + 3
        );
        let (plan, _) = meganeura::compile_training_graph(&staged);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    #[ignore = "requires guarded prior-continuation F64 fixture"]
    fn masked_loss_and_gradients_match_f64() {
        use crate::dreamer::runtime::build_session;
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        let gamma = 1.0_f32 - 1.0 / 333.0;
        let mut max_loss_error = 0.0_f64;
        let mut max_gradient_error = 0.0_f64;
        for rows in [1, 17, 256, 513] {
            let z = (0..rows)
                .map(|i| match i % 11 {
                    0 => -30.0,
                    1 => 30.0,
                    _ => (i as f32 * 0.17).sin() * 3.0,
                })
                .collect::<Vec<_>>();
            let target = (0..rows)
                .map(|i| if i % 3 == 0 { 0.0 } else { gamma })
                .collect::<Vec<_>>();
            let keep = (0..rows).map(|i| f32::from(i % 5 != 0)).collect::<Vec<_>>();
            let mut graph = Graph::new();
            let logits = graph.parameter("logits", &[rows, 1]);
            let t = graph.input("target", &[rows, 1]);
            let k = graph.input("keep", &[rows, 1]);
            let value = loss(&mut graph, logits, t, k);
            let scaled = graph.scale(value, 0.3);
            graph.set_outputs(vec![scaled]);
            let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
            session.set_parameter("logits", &z);
            session.set_input("target", &target);
            session.set_input("keep", &keep);
            session.step();
            session.wait();
            let mut actual = [0.0];
            let mut gradients = vec![0.0; rows];
            session.read_output_by_index(0, &mut actual);
            session.read_param_grad("logits", &mut gradients);
            let mut expected = 0.0;
            for i in 0..rows {
                let z = f64::from(z[i]);
                let t = f64::from(target[i]);
                let k = f64::from(keep[i]);
                expected += 0.3 * k * (z.max(0.0) + (-z.abs()).exp().ln_1p() - t * z) / rows as f64;
                let gradient = 0.3 * k * (1.0 / (1.0 + (-z).exp()) - t) / rows as f64;
                let error = (f64::from(gradients[i]) - gradient).abs();
                max_gradient_error = max_gradient_error.max(error);
                assert!(gradients[i].is_finite() && error <= 1e-7 + 2e-6 * gradient.abs());
                if k == 0.0 {
                    assert_eq!(gradients[i], 0.0);
                }
            }
            let error = (f64::from(actual[0]) - expected).abs();
            max_loss_error = max_loss_error.max(error);
            assert!(actual[0].is_finite() && error <= 5e-5);
        }
        println!(
            "{}",
            serde_json::json!({"status":"passed","cases":4,"maximum_loss_error":max_loss_error,"maximum_gradient_error":max_gradient_error})
        );
    }

    #[test]
    #[ignore = "requires guarded complete prior-continuation gradient fixture"]
    fn complete_prior_loss_gradients_and_arrival_invariance() {
        use crate::dreamer::runtime::{build_session, initialize_d3};
        let config = config();
        let mut graph = build_training_graph(&config, 1);
        graph.set_outputs(vec![graph.outputs()[LOSS_PRIOR_CONTINUATION]]);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!gpu.device_information().is_software_emulated);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        initialize_d3(&mut session, &graph, 19);
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op {
                let size = node.ty.shape.iter().product::<usize>();
                let value = if name.starts_with("keep_") {
                    1.0
                } else if name.starts_with("prior_continuation_uniforms_") {
                    0.37
                } else {
                    0.1
                };
                session.set_input(name, &vec![value; size]);
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
            "world.continuation.",
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
        println!(
            "{}",
            serde_json::json!({"status":"passed","loss":before[0],"arrival_invariant":true,"reset_mask_zero":true})
        );
    }
}
