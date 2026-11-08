use super::*;
use crate::dreamer::runtime;
use meganeura::{
    Mode,
    graph::Op,
    reference::{Feeds, evaluate},
};
use std::{collections::HashSet, sync::Arc};

#[test]
fn cpu_spatial_history_resets_preserve_lane_and_chw_order_without_current_leak() {
    let images = (1..=16).map(|x| x as f32).collect::<Vec<_>>();
    let actions = [-1, -1, 0, 2, 0, -1, 0, 3];
    assert_eq!(before_cpu(&images, &actions, 0, 2, 2), vec![0.; 8]);
    assert_eq!(
        before_cpu(&images, &actions, 1, 2, 2),
        [0., 0., 1., 2., 0., 0., 3., 4.]
    );
    assert_eq!(
        before_cpu(&images, &actions, 2, 2, 2),
        [1., 2., 5., 6., 0., 0., 0., 0.]
    );
    assert_eq!(
        before_cpu(&images, &actions, 3, 2, 2),
        [5., 6., 9., 10., 0., 0., 11., 12.]
    );
    assert_eq!(
        after_cpu(&images, &actions, 2, 2, 2),
        [5., 6., 9., 10., 0., 0., 11., 12.]
    );
    assert_eq!(
        after_cpu(&images, &actions, 0, 2, 2),
        [0., 0., 1., 2., 0., 0., 3., 4.]
    );
    assert_eq!(
        shift_cpu(
            &before_cpu(&images, &actions, 3, 2, 2),
            &[21., 22., 23., 24.],
            2
        ),
        [9., 10., 21., 22., 11., 12., 23., 24.]
    );
    assert_eq!(
        before_cpu(&images[..12], &actions[..6], 2, 2, 2),
        before_cpu(&images, &actions, 2, 2, 2)
    );
}

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
    gradients: Vec<Vec<f64>>,
}
fn fixture(width: usize) -> Fixture {
    let mut graph = Graph::new();
    let mut feeds = Feeds::new();
    let mut parameters = Vec::new();
    let mut gradients = Vec::new();
    let mut images = Vec::new();
    let mut keeps = Vec::new();
    let denominator = (4 * width) as f64;
    for time in 0..4 {
        let name = format!("image{time}");
        let values = (0..2 * width)
            .map(|i| ((time * 3 + i) % 23) as f32 / 64. - 0.25)
            .collect::<Vec<_>>();
        images.push(graph.parameter(&name, &[2, width]));
        feeds.set(&name, &values);
        parameters.push((name, values));
        gradients.push(
            (0..2 * width)
                .map(|i| {
                    if time == 1 && i < width {
                        2. / denominator
                    } else if time == 2 {
                        2.75 / denominator
                    } else {
                        0.
                    }
                })
                .collect(),
        );
    }
    for time in 0..4 {
        let name = format!("keep{time}");
        let values = if time == 0 {
            vec![0., 0.]
        } else if time == 2 {
            vec![1., 0.]
        } else {
            vec![1., 1.]
        };
        keeps.push(graph.parameter(&name, &[2, 1]));
        feeds.set(&name, &values);
        parameters.push((name, values));
        gradients.push(vec![0., 0.]);
    }
    let name = "predicted0".to_string();
    let values = (0..2 * width)
        .map(|i| i as f32 / 64. - 0.125)
        .collect::<Vec<_>>();
    let predicted = graph.parameter(&name, &[2, width]);
    feeds.set(&name, &values);
    parameters.push((name, values));
    gradients.push(vec![1.25 / denominator; 2 * width]);
    let prior = before(&mut graph, &images, &keeps, 3, 2, width);
    let history = after(&mut graph, &images, &keeps, 2, 2, width);
    let first = shift(&mut graph, history, predicted, 2, width);
    let older = slice_columns(&mut graph, first, 2, 2 * width, 0, width);
    let latest = slice_columns(&mut graph, first, 2, 2 * width, width, width);
    let older = graph.scale(older, 0.75);
    let latest = graph.scale(latest, 0.25);
    let next = graph.add(older, latest);
    let second = shift(&mut graph, first, next, 2, width);
    let a = graph.mean_all(prior);
    let b = graph.mean_all(history);
    let c = graph.mean_all(second);
    let total = graph.add(a, b);
    let total = graph.add(total, c);
    let mut outputs = vec![total, prior, history, first, second, images[0], images[3]];
    outputs.extend(keeps);
    graph.set_outputs(outputs);
    Fixture {
        graph,
        feeds,
        parameters,
        gradients,
    }
}

#[test]
fn spatial_history_graph_matches_cpu_and_analytic_derivatives() {
    let f = fixture(12);
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let actual = evaluate(&backward, &f.feeds).unwrap();
    for (i, expected) in f.gradients.iter().enumerate() {
        for (a, b) in actual[backward.outputs()[f.graph.outputs().len() + i] as usize]
            .data
            .iter()
            .zip(expected)
        {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }
    let images = f.parameters[..4]
        .iter()
        .flat_map(|p| p.1.iter().copied())
        .collect::<Vec<_>>();
    let actions = [-1, -1, 0, 2, 0, -1, 0, 3];
    for (index, expected) in [
        (1, before_cpu(&images, &actions, 3, 2, 12)),
        (2, after_cpu(&images, &actions, 2, 2, 12)),
    ] {
        let values = &actual[backward.outputs()[index] as usize].data;
        assert_eq!(
            values,
            &expected.iter().map(|&x| f64::from(x)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn spatial_seed_ancestry_excludes_all_future_images_and_reset_labels() {
    let mut graph = Graph::new();
    let images = (0..4)
        .map(|i| graph.input(&format!("image{i}"), &[2, 12]))
        .collect::<Vec<_>>();
    let keeps = (0..4)
        .map(|i| graph.input(&format!("keep{i}"), &[2, 1]))
        .collect::<Vec<_>>();
    let seed = after(&mut graph, &images, &keeps, 1, 2, 12);
    let mut seen = HashSet::new();
    let mut pending = vec![seed];
    while let Some(n) = pending.pop() {
        if seen.insert(n) {
            pending.extend(graph.node(n).inputs.iter().copied());
        }
    }
    assert!(seen.contains(&images[0]) && seen.contains(&images[1]) && seen.contains(&keeps[1]));
    assert!(
        images[2..]
            .iter()
            .chain(&keeps[2..])
            .all(|n| !seen.contains(n))
    );
    assert!(
        seen.iter()
            .all(|&n| !matches!(&graph.node(n).op,Op::Input{name} if name=="keep2"))
    );
}

#[test]
#[ignore = "requires separately declared guarded spatial-history native values/derivatives check"]
fn spatial_history_native_values_and_derivatives() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let f = fixture(3 * 4 * 4);
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let expected = evaluate(&backward, &f.feeds).unwrap();
    let mut s = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    for (name, value) in &f.parameters {
        s.set_parameter(name, value);
    }
    s.step();
    s.wait();
    assert_eq!(s.adam_step_count(), 0);
    for (i, &node) in f.graph.outputs().iter().enumerate() {
        let mut actual = vec![0.; expected[node as usize].data.len()];
        s.read_output_by_index(i, &mut actual);
        for (a, b) in actual.iter().zip(&expected[node as usize].data) {
            assert!((f64::from(*a) - b).abs() < 1e-6);
        }
    }
    for ((name, values), gradient) in f.parameters.iter().zip(&f.gradients) {
        let mut actual = vec![0.; values.len()];
        if s.has_param_grad(name) {
            s.read_param_grad(name, &mut actual);
        }
        for (a, b) in actual.iter().zip(gradient) {
            assert!((f64::from(*a) - b).abs() < 1e-6);
        }
        s.read_param(name, &mut actual);
        assert_eq!(&actual, values);
    }
}
