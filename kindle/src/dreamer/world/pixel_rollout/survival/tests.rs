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
    expected: f64,
}

fn fixture(batch: usize, horizon: usize, gamma: f32, zero: bool) -> Fixture {
    let mut graph = Graph::new();
    let mut feeds = Feeds::new();
    let mut parameters = Vec::new();
    let mut logits = Vec::new();
    let mut targets = Vec::new();
    let mut weights = Vec::new();
    let mut probabilities = Vec::new();
    let mut actuals = Vec::new();
    let mut masks = Vec::new();
    for t in 0..horizon {
        let values = (0..batch)
            .map(|lane| [-3., 0., 1., 3., -30., 30.][(t * 3 + lane) % 6])
            .collect::<Vec<_>>();
        let target = (0..batch)
            .map(|lane| if lane % 3 == 1 && t == 1 { 0. } else { gamma })
            .collect::<Vec<_>>();
        let keep = (0..batch)
            .map(|lane| f32::from(!zero && (lane % 3 == 0 || t < 2)))
            .collect::<Vec<_>>();
        probabilities.push(
            values
                .iter()
                .map(|&x| 1.0 / (1.0 + (-f64::from(x)).exp()))
                .collect::<Vec<_>>(),
        );
        actuals.push(
            target
                .iter()
                .map(|&x| (f64::from(x) / f64::from(gamma)).clamp(0., 1.))
                .collect::<Vec<_>>(),
        );
        masks.push(keep.clone());
        let mut nodes = Vec::new();
        for (prefix, data) in [("logit", values), ("target", target), ("keep", keep)] {
            let name = format!("{prefix}{t}");
            nodes.push(graph.parameter(&name, &[batch, 1]));
            feeds.set(&name, &data);
            parameters.push((name, data));
        }
        logits.push(nodes[0]);
        targets.push(nodes[1]);
        weights.push(nodes[2]);
    }
    let loss = build(&mut graph, &logits, &targets, &weights, gamma, batch);
    graph.set_outputs(vec![loss]);
    let mut predicted = vec![1.; batch];
    let mut actual = vec![1.; batch];
    let mut expected = 0.;
    let mut gradients = vec![vec![0.; batch]; 3 * horizon];
    for t in 0..horizon {
        for lane in 0..batch {
            predicted[lane] *= (probabilities[t][lane] / f64::from(gamma)).clamp(0., 1.);
            actual[lane] *= actuals[t][lane];
            let factor = f64::from(masks[t][lane]) / (batch * horizon) as f64;
            expected += factor * (predicted[lane] - actual[lane]).powi(2);
            for i in 0..=t {
                if probabilities[i][lane] < f64::from(gamma) {
                    gradients[3 * i][lane] += 2.
                        * factor
                        * (predicted[lane] - actual[lane])
                        * predicted[lane]
                        * (1. - probabilities[i][lane]);
                }
            }
        }
    }
    Fixture {
        graph,
        feeds,
        parameters,
        gradients,
        expected,
    }
}

#[test]
fn survival_brier_matches_independent_f64_products_and_all_logit_derivatives() {
    for gamma in [0.75, 1. - 1. / 333.] {
        for zero in [false, true] {
            let f = fixture(3, 4, gamma, zero);
            let backward = meganeura::autodiff::differentiate(&f.graph);
            let values = evaluate(&backward, &f.feeds).unwrap();
            assert!((values[backward.outputs()[0] as usize].data[0] - f.expected).abs() < 1e-7);
            for (index, expected) in f.gradients.iter().enumerate() {
                for (a, b) in values[backward.outputs()[index + 1] as usize]
                    .data
                    .iter()
                    .zip(expected)
                {
                    assert!((a - b).abs() < 1e-7, "parameter{index}: {a} vs {b}");
                }
            }
            if zero {
                assert_eq!(f.expected, 0.);
            }
        }
    }
}

#[test]
#[ignore = "requires separately declared guarded survival value/derivative qualification"]
fn survival_brier_native_values_and_all_derivatives() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for gamma in [0.75, 1. - 1. / 333.] {
        for zero in [false, true] {
            let f = fixture(8, 16, gamma, zero);
            let mut session = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
            for (name, value) in &f.parameters {
                session.set_parameter(name, value);
            }
            session.step();
            session.wait();
            assert_eq!(session.adam_step_count(), 0);
            let mut actual = [0.];
            session.read_output_by_index(0, &mut actual);
            assert!(
                (f64::from(actual[0]) - f.expected).abs() < 2e-5,
                "{} vs {}",
                actual[0],
                f.expected
            );
            for ((name, values), expected) in f.parameters.iter().zip(&f.gradients) {
                let mut actual = vec![0.; expected.len()];
                if session.has_param_grad(name) {
                    session.read_param_grad(name, &mut actual);
                }
                for (a, b) in actual.iter().zip(expected) {
                    assert!((f64::from(*a) - b).abs() < 2e-6, "{name}: {a} vs {b}");
                }
                session.read_param(name, &mut actual);
                assert_eq!(&actual, values);
            }
        }
    }
}
