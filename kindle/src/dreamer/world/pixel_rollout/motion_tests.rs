use super::*;
use crate::dreamer::{networks::pixel_cross_entropy, runtime};
use meganeura::reference::{Feeds, evaluate};
use std::sync::Arc;

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
    gradients: Vec<Vec<f64>>,
    loss: f64,
}

fn fixture(width: usize, zero: bool) -> Fixture {
    let mut graph = Graph::new();
    let mut feeds = Feeds::new();
    let mut parameters = Vec::new();
    let mut gradients = Vec::new();
    let mut losses = Vec::new();
    let mut frames = Vec::new();
    let mut frame_values = Vec::new();
    for time in 0..3 {
        let name = format!("frame_{time}");
        let values = (0..2 * width)
            .map(|i| ((i * 3 + time * 5) % 11) as f32 / 16. - 0.375)
            .collect::<Vec<_>>();
        frames.push(graph.input(&name, &[2, width]));
        feeds.set(&name, &values);
        frame_values.push(values);
    }
    let mut expected_loss = 0.;
    for time in 0..2 {
        let name = format!("logits_{time}");
        let logits = graph.parameter(&name, &[2, width]);
        let values = (0..2 * width)
            .map(|i| ((i * 7 + time * 3) % 17) as f32 / 4. - 2.)
            .collect::<Vec<_>>();
        feeds.set(&name, &values);
        parameters.push((name, values.clone()));
        let mask = if zero {
            [0., 0.]
        } else {
            [1., f32::from(time == 0)]
        };
        let keep = graph.input(&format!("keep_{time}"), &[2, 1]);
        feeds.set(&format!("keep_{time}"), &mask);
        let errors = pixel_cross_entropy(&mut graph, logits, frames[time + 1]);
        losses.push(motion_loss(
            &mut graph,
            errors,
            frames[time + 1],
            frames[time],
            keep,
        ));
        let mut gradient = vec![0.; 2 * width];
        for (row, &mask) in mask.iter().enumerate() {
            let range = row * width..(row + 1) * width;
            let weights = frame_values[time + 1][range.clone()]
                .iter()
                .zip(&frame_values[time][range.clone()])
                .map(|(&a, &b)| 1. + 100. * f64::from(mask) * (f64::from(a) - f64::from(b)).abs())
                .collect::<Vec<_>>();
            let mean = weights.iter().sum::<f64>() / width as f64;
            for (i, weight) in range.zip(weights) {
                let x = f64::from(values[i]);
                let y = f64::from(frame_values[time + 1][i]) + 0.5;
                let scale = f64::from(mask) * weight / mean / 4.;
                expected_loss += (x.max(0.) + (-x.abs()).exp().ln_1p() - x * y) * scale;
                gradient[i] = (1. / (1. + (-x).exp()) - y) * scale;
            }
        }
        gradients.push(gradient);
    }
    let loss = sum(&mut graph, &losses);
    let loss = graph.scale(loss, 0.25);
    graph.set_outputs(vec![loss]);
    Fixture {
        graph,
        feeds,
        parameters,
        gradients,
        loss: expected_loss,
    }
}

#[test]
fn temporal_motion_loss_matches_f64_including_reset_and_zero_suffixes() {
    for zero in [false, true] {
        let f = fixture(8, zero);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let values = evaluate(&backward, &f.feeds).unwrap();
        let loss = values[backward.outputs()[0] as usize].data[0];
        assert!((loss - f.loss).abs() < 1e-10);
        for (index, expected) in f.gradients.iter().enumerate() {
            let actual = &values[backward.outputs()[index + 1] as usize].data;
            for (&a, &b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn temporal_motion_targets_and_masks_are_stopped() {
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[2, 8]);
    let current = graph.parameter("current", &[2, 8]);
    let previous = graph.parameter("previous", &[2, 8]);
    let keep = graph.parameter("keep", &[2, 1]);
    let errors = pixel_cross_entropy(&mut graph, logits, current);
    let loss = motion_loss(&mut graph, errors, current, previous, keep);
    graph.set_outputs(vec![loss]);
    let backward = meganeura::autodiff::differentiate(&graph);
    let mut feeds = Feeds::new();
    feeds
        .set("logits", &[0.1; 16])
        .set("current", &[0.2; 16])
        .set("previous", &[-0.1; 16])
        .set("keep", &[1., 0.]);
    let values = evaluate(&backward, &feeds).unwrap();
    assert!(
        values[backward.outputs()[1] as usize].data[..8]
            .iter()
            .any(|&x| x != 0.)
    );
    for &output in &backward.outputs()[2..] {
        assert!(values[output as usize].data.iter().all(|&x| x == 0.));
    }
}

#[test]
#[ignore = "requires guarded GPU; new composed temporal motion loss values and derivatives"]
fn motion_rollout_native_matches_f64() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for zero in [false, true] {
        let f = fixture(1024, zero);
        let mut session = runtime::build_session(&f.graph, &gpu, meganeura::Mode::Training, false);
        for (name, values) in &f.parameters {
            session.set_parameter(name, values);
        }
        for time in 0..3 {
            let name = format!("frame_{time}");
            session.set_input(&name, &f.feeds.f32(&name).unwrap());
        }
        for time in 0..2 {
            let name = format!("keep_{time}");
            session.set_input(&name, &f.feeds.f32(&name).unwrap());
        }
        session.step();
        session.wait();
        let mut actual = [0.];
        session.read_output_by_index(0, &mut actual);
        assert!((f64::from(actual[0]) - f.loss).abs() < 2e-3);
        for ((name, values), expected) in f.parameters.iter().zip(&f.gradients) {
            let mut actual = vec![0.; values.len()];
            session.read_param_grad(name, &mut actual);
            for (&a, &b) in actual.iter().zip(expected) {
                assert!((f64::from(a) - b).abs() < 2e-6);
            }
        }
        let names = f
            .parameters
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            session.read_params(&names),
            f.parameters
                .iter()
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>()
        );
    }
}
