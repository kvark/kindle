use meganeura::{Graph, graph::NodeId};

/// Targets are zero at true terminals and exactly `discount` otherwise.
/// Keep the ordinary row denominator so continuing-row losses retain weight one.
pub(super) fn loss(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    rows: usize,
    discount: f32,
    terminal_weight: f32,
) -> NodeId {
    let discount_target = graph.constant(vec![discount; rows], &[rows, 1]);
    let terminal = graph.sub(discount_target, target);
    let extra_weight = graph.scale(terminal, (terminal_weight - 1.0) / discount);
    let ones = graph.constant(vec![1.0; rows], &[rows, 1]);
    let weight = graph.add(ones, extra_weight);
    let positive = graph.softplus(logits, 1.0);
    let product = graph.mul(logits, target);
    let per_row = graph.sub(positive, product);
    let weighted = graph.mul(per_row, weight);
    graph.mean_all(weighted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_continuation_graph_autodiffs_and_compiles() {
        let mut graph = Graph::new();
        let logits = graph.parameter("logits", &[513, 1]);
        let target = graph.input("target", &[513, 1]);
        let loss = loss(&mut graph, logits, target, 513, 1.0 - 1.0 / 333.0, 10.0);
        graph.set_outputs(vec![loss]);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    fn terminal_weighted_world_graph_compiles() {
        use super::super::super::config::{DreamerConfig, ObservationKind};
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.terminal_continuation_weight = Some(10.0);
        config.check().unwrap();
        let graph = super::super::build_training_graph(&config, config.world_backprop_length);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    fn unit_weight_preserves_default_training_graph() {
        use super::super::super::config::{DreamerConfig, ObservationKind};
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        let default = super::super::build_training_graph(&config, config.world_backprop_length);
        config.terminal_continuation_weight = Some(1.0);
        let explicit = super::super::build_training_graph(&config, config.world_backprop_length);
        assert_eq!(
            format!("{:?}", default.nodes()),
            format!("{:?}", explicit.nodes())
        );
    }

    #[test]
    #[ignore = "requires the separately guarded terminal-weight numerical fixture"]
    fn terminal_weight_loss_and_gradients_match_f64() {
        use super::super::super::runtime::build_session;
        use meganeura::Mode;
        use std::sync::Arc;

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let discount = 1.0_f32 - 1.0 / 333.0;
        let mut maximum_loss_error = 0.0_f64;
        let mut maximum_gradient_error = 0.0_f64;
        for rows in [1, 17, 256, 513, 1024] {
            let logits = (0..rows)
                .map(|i| match i % 13 {
                    0 => -30.0,
                    1 => 30.0,
                    _ => (i as f32 * 0.17).sin() * 3.0,
                })
                .collect::<Vec<_>>();
            let targets = (0..rows)
                .map(|i| if i % 3 == 0 { 0.0 } else { discount })
                .collect::<Vec<_>>();
            let mut unweighted_gradients: Vec<f32> = Vec::new();
            for weight in [1.0, 10.0] {
                let mut graph = Graph::new();
                let z = graph.parameter("logits", &[rows, 1]);
                let t = graph.input("target", &[rows, 1]);
                let value = loss(&mut graph, z, t, rows, discount, weight);
                let scaled = graph.scale(value, 0.3);
                graph.set_outputs(vec![scaled]);
                let mut session = build_session(&graph, &gpu, Mode::Training, false);
                session.set_parameter("logits", &logits);
                session.set_input("target", &targets);
                session.step();
                session.wait();
                let mut actual_loss = [0.0];
                let mut gradients = vec![0.0; rows];
                session.read_output_by_index(0, &mut actual_loss);
                session.read_param_grad("logits", &mut gradients);
                let mut expected_loss = 0.0;
                for (i, ((&z, &target), &actual)) in
                    logits.iter().zip(&targets).zip(&gradients).enumerate()
                {
                    let z = f64::from(z);
                    let t = f64::from(target);
                    let w = if target == 0.0 {
                        f64::from(weight)
                    } else {
                        1.0
                    };
                    expected_loss +=
                        0.3 * w * (z.max(0.0) + (-z.abs()).exp().ln_1p() - t * z) / rows as f64;
                    let expected = 0.3 * w * (1.0 / (1.0 + (-z).exp()) - t) / rows as f64;
                    let error = (f64::from(actual) - expected).abs();
                    maximum_gradient_error = maximum_gradient_error.max(error);
                    assert!(actual.is_finite() && error <= 1e-7 + 2e-6 * expected.abs());
                    if weight == 10.0 && target != 0.0 {
                        let delta = (actual - unweighted_gradients[i]).abs();
                        assert!(delta <= 1e-7 + 2e-6 * actual.abs());
                    }
                }
                let error = (f64::from(actual_loss[0]) - expected_loss).abs();
                maximum_loss_error = maximum_loss_error.max(error);
                assert!(actual_loss[0].is_finite() && error <= 5e-5);
                if weight == 1.0 {
                    unweighted_gradients = gradients;
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({"status":"passed","cases":10,
                "maximum_loss_error":maximum_loss_error,
                "maximum_gradient_error":maximum_gradient_error})
        );
    }
}
