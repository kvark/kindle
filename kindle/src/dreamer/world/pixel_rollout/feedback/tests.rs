use super::*;
use crate::dreamer::runtime;
use meganeura::{
    Mode,
    reference::{Feeds, evaluate},
};
use std::sync::Arc;

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
    gradients: Vec<Vec<f64>>,
    expected: [f64; 3],
}
fn fixture(batch: usize, horizon: usize, bins: usize, zero: bool) -> Fixture {
    let rows = batch * horizon;
    let mut graph = Graph::new();
    let mut feeds = Feeds::new();
    let mut parameters = Vec::new();
    let values = (0..rows * bins)
        .map(|i| ((i * 7 % 29) as f32 - 14.) * 0.25)
        .collect::<Vec<_>>();
    let continuation = (0..rows)
        .map(|i| match i % 11 {
            0 => -30.,
            1 => 30.,
            _ => ((i * 3 % 17) as f32 - 8.) * 0.25,
        })
        .collect::<Vec<_>>();
    let mut reward_target = vec![0.; rows * bins];
    for i in 0..rows {
        reward_target[i * bins + i % bins] = 0.25;
        reward_target[i * bins + (i + 1) % bins] += 0.75;
    }
    let targets = (0..rows)
        .map(|i| if i % 3 == 0 { 0. } else { 1. - 1. / 333. })
        .collect::<Vec<_>>();
    let weights = (0..rows)
        .map(|i| f32::from(!zero && i % 4 != 2))
        .collect::<Vec<_>>();
    let specs = [
        ("reward", values.clone(), vec![rows, bins]),
        ("continuation", continuation.clone(), vec![rows, 1]),
        ("reward_target", reward_target.clone(), vec![rows, bins]),
        ("continuation_target", targets.clone(), vec![rows, 1]),
        ("keep", weights.clone(), vec![rows, 1]),
    ];
    let mut nodes = Vec::new();
    for (name, value, shape) in specs {
        nodes.push(graph.parameter(name, &shape));
        feeds.set(name, &value);
        parameters.push((name.to_owned(), value));
    }
    let loss = event_losses(&mut graph, nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]);
    let total = graph.add(loss[0], loss[1]);
    graph.set_outputs(vec![total, loss[0], loss[1]]);
    let mut gradients = vec![
        vec![0.; rows * bins],
        vec![0.; rows],
        vec![0.; rows * bins],
        vec![0.; rows],
        vec![0.; rows],
    ];
    let mut expected = [0.; 3];
    for row in 0..rows {
        let logits = &values[row * bins..(row + 1) * bins];
        let max = logits
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = logits
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>();
        let scale = f64::from(weights[row]) / rows as f64;
        for j in 0..bins {
            let p = (f64::from(logits[j]) - max).exp() / sum;
            let target = f64::from(reward_target[row * bins + j]);
            expected[1] -= scale * target * (f64::from(logits[j]) - max - sum.ln());
            gradients[0][row * bins + j] = scale * (p - target);
        }
        let x = f64::from(continuation[row]);
        let y = f64::from(targets[row]);
        expected[2] += scale * (x.max(0.) + (-x.abs()).exp().ln_1p() - x * y);
        gradients[1][row] = scale * (1. / (1. + (-x).exp()) - y);
    }
    expected[0] = expected[1] + expected[2];
    Fixture {
        graph,
        feeds,
        parameters,
        gradients,
        expected,
    }
}
#[test]
fn future_event_losses_match_f64_with_terminal_masks_and_stopped_targets() {
    for zero in [false, true] {
        let f = fixture(2, 3, 5, zero);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let actual = evaluate(&backward, &f.feeds).unwrap();
        for (i, e) in f.expected.iter().enumerate() {
            assert!((actual[backward.outputs()[i] as usize].data[0] - e).abs() < 2e-6);
        }
        for (i, e) in f.gradients.iter().enumerate() {
            for (a, b) in actual[backward.outputs()[3 + i] as usize]
                .data
                .iter()
                .zip(e)
            {
                assert!((a - b).abs() < 1e-7, "{a} vs {b}");
            }
        }
        if zero {
            assert_eq!(actual[backward.outputs()[0] as usize].data[0], 0.);
        }
    }
}
#[test]
#[ignore = "requires separately declared guarded future-event values/derivatives check"]
fn image_feedback_events_native_values_and_derivatives() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for zero in [false, true] {
        let f = fixture(8, 16, 255, zero);
        let mut s = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
        for (name, value) in &f.parameters {
            s.set_parameter(name, value);
        }
        s.step();
        s.wait();
        assert_eq!(s.adam_step_count(), 0);
        for (i, e) in f.expected.iter().enumerate() {
            let mut a = [0.];
            s.read_output_by_index(i, &mut a);
            assert!((f64::from(a[0]) - e).abs() < 2e-4, "{} vs {e}", a[0]);
        }
        for ((name, values), expected) in f.parameters.iter().zip(&f.gradients) {
            let mut actual = vec![0.; values.len()];
            if s.has_param_grad(name) {
                s.read_param_grad(name, &mut actual);
            }
            for (a, b) in actual.iter().zip(expected) {
                assert!((f64::from(*a) - b).abs() < 2e-6);
            }
            s.read_param(name, &mut actual);
            assert_eq!(&actual, values);
        }
    }
}
