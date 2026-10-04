//! Continuous deterministic prediction: cosine distance to detached CNN tokens.

use super::*;

/// Optax's cosine distance with a floor on each squared norm, not its root.
pub(super) fn cosine_distance(graph: &mut Graph, prediction: NodeId, target: NodeId) -> NodeId {
    assert_eq!(graph.node(prediction).ty.shape, graph.node(target).ty.shape);
    let shape = graph.node(prediction).ty.shape.clone();
    assert_eq!(shape.len(), 2);
    let mut normalize = |value| {
        let squared = graph.mul(value, value);
        let norm_squared = graph.sum_inner(squared);
        let bounded = graph.clamp(norm_squared, 1e-8, f32::MAX);
        let log = graph.log(bounded);
        let log_inverse_root = graph.scale(log, -0.5);
        let inverse_root = graph.exp(log_inverse_root);
        let inverse_root = graph.broadcast_inner(inverse_root, shape[1]);
        graph.mul(value, inverse_root)
    };
    let prediction = normalize(prediction);
    let target = normalize(target);
    let product = graph.mul(prediction, target);
    let similarity = graph.sum_inner(product);
    let negative = graph.neg(similarity);
    let one = graph.constant(vec![1.0; shape[0]], &[shape[0], 1]);
    graph.add(one, negative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdp_graph_has_embedding_predictor_without_pixel_decoder() {
        use meganeura::graph::Op;
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let parameters = graph
            .nodes()
            .iter()
            .filter_map(|node| {
                if let Op::Parameter { name } = &node.op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert!(
            !parameters
                .iter()
                .any(|name| name.starts_with("world.decoder."))
        );
        assert!(parameters.contains(&"world.future_predictor.layer0.weight"));
        assert!(parameters.contains(&"world.representation.encoder.cnn0.weight"));
        assert_eq!(future_head_revision(&config), Some("continuous-cosine-v1"));
    }

    #[test]
    #[ignore = "requires separately guarded GPU; independent F64 CDP cosine values, raw gradients and detached targets"]
    fn cosine_matches_f64_values_and_gradients() {
        use crate::dreamer::runtime::build_session;
        const ROWS: usize = 4;
        const WIDTH: usize = 256;
        let mut graph = Graph::new();
        let prediction = graph.parameter("prediction", &[ROWS, WIDTH]);
        let target = graph.parameter("target", &[ROWS, WIDTH]);
        let detached = graph.stop_gradient(target);
        let distances = cosine_distance(&mut graph, prediction, detached);
        let loss = graph.mean_all(distances);
        graph.set_outputs(vec![loss, distances]);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!gpu.device_information().is_software_emulated);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        let target = (0..ROWS * WIDTH)
            .map(|i| (i % 19) as f32 * 0.07 - 0.5)
            .collect::<Vec<_>>();
        let prediction = (0..ROWS * WIDTH)
            .map(|i| match i / WIDTH {
                0 => 0.0,
                1 => (i % 13) as f32 * 1e-8,
                2 => (i % 23) as f32 * 0.03 - 0.2,
                _ => -target[i],
            })
            .collect::<Vec<_>>();
        session.set_parameter("prediction", &prediction);
        session.set_parameter("target", &target);
        session.step();
        session.wait();
        assert!(
            !session.has_param_grad("target"),
            "target side must be detached"
        );
        let mut actual = vec![0.0; ROWS];
        let mut gradients = vec![0.0; ROWS * WIDTH];
        session.read_output_by_index(1, &mut actual);
        session.read_param_grad("prediction", &mut gradients);
        let mut loss = 0.0;
        for row in 0..ROWS {
            let x = prediction[row * WIDTH..(row + 1) * WIDTH]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>();
            let y = target[row * WIDTH..(row + 1) * WIDTH]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>();
            let square = x.iter().map(|v| v * v).sum::<f64>();
            let norm_x = square.max(f64::from(1e-8_f32)).sqrt();
            let norm_y = y
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .max(f64::from(1e-8_f32))
                .sqrt();
            let cosine = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / (norm_x * norm_y);
            assert!((f64::from(actual[row]) - (1.0 - cosine)).abs() < 3e-6);
            loss += (1.0 - cosine) / ROWS as f64;
            for i in 0..WIDTH {
                let derivative = (-y[i] / (norm_x * norm_y)
                    + if square > f64::from(1e-8_f32) {
                        cosine * x[i] / square
                    } else {
                        0.0
                    })
                    / ROWS as f64;
                let actual = f64::from(gradients[row * WIDTH + i]);
                assert!(
                    actual.is_finite()
                        && (actual - derivative).abs() <= 2e-5 + 2e-5 * derivative.abs(),
                    "gradient {row}/{i}: {actual} != {derivative}"
                );
            }
        }
        assert!((f64::from(session.read_output(1)[0]) - loss).abs() < 3e-6);
        let memory = session.device_memory_stats().unwrap();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        println!(
            "CDP cosine: 4 distances, 1024 raw gradients, detached targets pass F64 reference"
        );
    }
}
