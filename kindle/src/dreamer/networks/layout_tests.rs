use super::*;
use crate::dreamer::runtime::build_session;
use meganeura::Mode;
use std::sync::Arc;

// Independent scalar reference, including analytical derivatives. In particular,
// neither the reference layout nor its gradients use the grouped graph helpers.
#[test]
#[ignore = "requires a GPU"]
fn tiny_grouped_rssm_layout_matches_scalar_outputs_and_gradients() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        let info = gpu.device_information();
        assert_eq!(info.device_name, expected);
        assert!(!info.is_software_emulated);
    }
    for (batch, blocks, lane, shared_width) in [(1, 1, 3, 7), (2, 3, 5, 9), (16, 8, 256, 768)] {
        let width = blocks * lane;
        let mut graph = Graph::new();
        let deter = graph.parameter("deter", &[batch, width]);
        let shared = graph.parameter("shared", &[batch, shared_width]);
        let gates = graph.parameter("gates", &[batch, 3 * width]);
        let grouped = grouped_rssm_input(&mut graph, deter, shared, batch, blocks);
        let updated = grouped_gru_update(&mut graph, deter, gates, batch, blocks);
        let squared = graph.mul(grouped, grouped);
        let input_loss = graph.mean_all(squared);
        let squared = graph.mul(updated, updated);
        let update_loss = graph.mean_all(squared);
        let loss = graph.add(input_loss, update_loss);
        graph.set_outputs(vec![loss, grouped, updated]);
        let mut session = build_session(&graph, &gpu, Mode::Training, false);
        if std::env::var_os("KINDLE_EXPECT_DEVICE_NAME").is_some() {
            let memory = gpu.memory_stats();
            eprintln!("memory usage={} budget={}", memory.usage, memory.budget);
            assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        }
        let data = |count, phase: f32| {
            (0..count)
                .map(|i| ((i as f32 * 0.7 + phase).sin()) * 0.8)
                .collect::<Vec<_>>()
        };
        let deter = data(batch * width, 0.1);
        let shared = data(batch * shared_width, 0.3);
        let gates = data(batch * 3 * width, 0.5);
        for (name, values) in [("deter", &deter), ("shared", &shared), ("gates", &gates)] {
            session.set_parameter(name, values);
        }
        session.set_adam(0.0, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();

        let grouped_len = batch * blocks * (lane + shared_width);
        let mut grouped = Vec::with_capacity(grouped_len);
        let mut updated = vec![0.0; deter.len()];
        let mut grad_deter = vec![0.0; deter.len()];
        let mut grad_shared = vec![0.0; shared.len()];
        let mut grad_gates = vec![0.0; gates.len()];
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        for row in 0..batch {
            for block in 0..blocks {
                let offset = row * width + block * lane;
                grouped.extend_from_slice(&deter[offset..offset + lane]);
                grouped.extend_from_slice(&shared[row * shared_width..(row + 1) * shared_width]);
                for j in 0..lane {
                    let i = offset + j;
                    let g = row * 3 * width + block * 3 * lane + j;
                    let reset = sigmoid(gates[g]);
                    let candidate = (reset * gates[g + lane]).tanh();
                    let update = sigmoid(gates[g + 2 * lane] - 1.0);
                    let value = update * candidate + (1.0 - update) * deter[i];
                    updated[i] = value;
                    let dy = 2.0 * value / deter.len() as f32;
                    grad_deter[i] = 2.0 * deter[i] / grouped_len as f32 + dy * (1.0 - update);
                    let dc = dy * update * (1.0 - candidate * candidate);
                    grad_gates[g] = dc * gates[g + lane] * reset * (1.0 - reset);
                    grad_gates[g + lane] = dc * reset;
                    grad_gates[g + 2 * lane] =
                        dy * (candidate - deter[i]) * update * (1.0 - update);
                }
            }
            for j in 0..shared_width {
                let i = row * shared_width + j;
                grad_shared[i] = 2.0 * shared[i] * blocks as f32 / grouped_len as f32;
            }
        }
        let close = |label, actual: &[f32], expected: &[f32]| {
            let mut error = 0.0f64;
            let mut norm = 0.0f64;
            for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - b).abs() < 2e-6 * (1.0 + b.abs()),
                    "{label}[{i}]: {a} != {b}"
                );
                error += f64::from(a - b).powi(2);
                norm += f64::from(b).powi(2);
            }
            let relative = (error / norm.max(1e-30)).sqrt();
            eprintln!("B{batch}/blocks{blocks}/lane{lane} {label}: relative L2 {relative}");
            assert!(relative < 2e-5);
        };
        for (index, expected) in [(1, &grouped), (2, &updated)] {
            let mut actual = vec![0.0; expected.len()];
            session.read_output_by_index(index, &mut actual);
            close("output", &actual, expected);
        }
        for (name, expected) in [
            ("deter", grad_deter),
            ("shared", grad_shared),
            ("gates", grad_gates),
        ] {
            let mut actual = vec![0.0; expected.len()];
            session.read_param_grad(name, &mut actual);
            close(name, &actual, &expected);
        }
    }
}
