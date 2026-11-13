use super::*;
use crate::dreamer::runtime;
use meganeura::{
    Mode,
    reference::{Feeds, evaluate, gradients},
};
use std::sync::Arc;

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
}

fn fixture() -> Fixture {
    let (batch, side) = (2, 3);
    let area = side * side;
    let mut graph = Graph::new();
    let latest = graph.parameter("latest", &[batch, 3 * area]);
    let redraw = graph.parameter("redraw", &[batch, 3 * area]);
    let masks = graph.parameter("masks", &[batch, FIELDS * area]);
    let logits = compose(&mut graph, latest, redraw, masks, batch, side);
    let prediction = crate::dreamer::networks::centered_pixels(&mut graph, logits);
    let loss = gradients::weighted_loss(&mut graph, logits, 71, 1.0);
    graph.set_outputs(vec![loss, logits, prediction]);
    let parameters: Vec<(String, Vec<f32>)> = vec![
        (
            "latest".into(),
            (0..batch * 3 * area)
                .map(|i| (i % 19) as f32 / 32.0 - 0.3)
                .collect(),
        ),
        (
            "redraw".into(),
            (0..batch * 3 * area)
                .map(|i| (i % 7) as f32 / 8.0 - 0.3)
                .collect(),
        ),
        (
            "masks".into(),
            (0..batch * FIELDS * area)
                .map(|i| (i % 13) as f32 / 20.0 - 0.3)
                .collect(),
        ),
    ];
    let mut feeds = Feeds::new();
    for (name, values) in &parameters {
        feeds.set(name, values);
    }
    Fixture {
        graph,
        feeds,
        parameters,
    }
}

fn reference(latest: &[f32], redraw: &[f32], masks: &[f32], batch: usize, side: usize) -> Vec<f64> {
    let area = side * side;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let weights = |b: usize, pixel: usize, axis: usize| {
        let values = (0..OFFSETS)
            .map(|j| f64::from(masks[b * FIELDS * area + (axis * OFFSETS + j) * area + pixel]))
            .collect::<Vec<_>>();
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps = values.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
        let total = exps.iter().sum::<f64>();
        exps.into_iter().map(|v| v / total).collect::<Vec<_>>()
    };
    let mut horizontal = vec![0.0; latest.len()];
    let mut result = horizontal.clone();
    for b in 0..batch {
        for y in 0..side {
            for x in 0..side {
                let w = weights(b, y * side + x, 0);
                for c in 0..3 {
                    for (j, &weight) in w.iter().enumerate() {
                        let source = x as isize - (j as isize - RADIUS);
                        if (0..side as isize).contains(&source) {
                            horizontal[b * 3 * area + c * area + y * side + x] += weight
                                * (f64::from(
                                    latest[b * 3 * area + c * area + y * side + source as usize],
                                ) + 0.5);
                        }
                    }
                }
            }
        }
    }
    for b in 0..batch {
        for y in 0..side {
            for x in 0..side {
                let w = weights(b, y * side + x, 1);
                let alpha = sigmoid(f64::from(
                    masks[b * FIELDS * area + 2 * OFFSETS * area + y * side + x],
                ));
                for c in 0..3 {
                    let mut moved = 0.0;
                    for (j, &weight) in w.iter().enumerate() {
                        let source = y as isize - (j as isize - RADIUS);
                        if (0..side as isize).contains(&source) {
                            moved += weight
                                * horizontal[b * 3 * area + c * area + source as usize * side + x];
                        }
                    }
                    let i = b * 3 * area + c * area + y * side + x;
                    let probability = (alpha * moved
                        + (1.0 - alpha) * sigmoid(f64::from(redraw[i])))
                    .clamp(f64::from(1e-6f32), f64::from(1.0f32 - 1e-6f32));
                    result[i] = probability.ln() - (1.0 - probability).ln();
                }
            }
        }
    }
    result
}

#[test]
fn transport_matches_independent_reference_and_finite_differences() {
    let f = fixture();
    let values = evaluate(&f.graph, &f.feeds).unwrap();
    let expected = reference(
        &f.parameters[0].1,
        &f.parameters[1].1,
        &f.parameters[2].1,
        2,
        3,
    );
    for (a, b) in values[f.graph.outputs()[1] as usize]
        .data
        .iter()
        .zip(expected)
    {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
    gradients::check(
        &f.graph,
        &f.feeds,
        &gradients::Options {
            max_elementwise: 64,
            probes: 3,
            ..Default::default()
        },
    )
    .unwrap()
    .assert_passed("pixel transport derivatives");
}

#[test]
fn transport_offsets_padding_blend_extremes_and_lanes() {
    let mut f = fixture();
    let area = 9;
    for (dx, dy) in [(0, 0), (-1, 1), (2, -2), (8, 0)] {
        let mut masks = vec![-80.0; 2 * FIELDS * area];
        for b in 0..2 {
            for pixel in 0..area {
                masks[b * FIELDS * area + (dx + RADIUS) as usize * area + pixel] = 80.0;
                masks[b * FIELDS * area + (OFFSETS + (dy + RADIUS) as usize) * area + pixel] = 80.0;
                masks[b * FIELDS * area + 2 * OFFSETS * area + pixel] = 80.0;
            }
        }
        f.feeds.set("masks", &masks);
        let v = evaluate(&f.graph, &f.feeds).unwrap();
        let actual = &v[f.graph.outputs()[2] as usize].data;
        for b in 0..2 {
            for c in 0..3 {
                for y in 0..3 {
                    for x in 0..3 {
                        let (sx, sy) = (x as isize - dx, y as isize - dy);
                        let expected = if (0..3).contains(&sx) && (0..3).contains(&sy) {
                            f64::from(
                                f.parameters[0].1[b * 27 + c * 9 + sy as usize * 3 + sx as usize],
                            ) + 0.5
                        } else {
                            0.0
                        };
                        let expected =
                            expected.clamp(f64::from(1e-6f32), f64::from(1.0f32 - 1e-6f32)) - 0.5;
                        assert!((actual[b * 27 + c * 9 + y * 3 + x] - expected).abs() < 1e-10);
                    }
                }
            }
        }
    }
    let mut masks = f.parameters[2].1.clone();
    for b in 0..2 {
        masks[b * FIELDS * area + 2 * OFFSETS * area..(b + 1) * FIELDS * area].fill(-80.0);
    }
    f.feeds.set("masks", &masks);
    let v = evaluate(&f.graph, &f.feeds).unwrap();
    for (&a, &z) in v[f.graph.outputs()[1] as usize]
        .data
        .iter()
        .zip(&f.parameters[1].1)
    {
        assert!((a - f64::from(z)).abs() < 1e-12);
    }
    f.feeds.set("masks", &f.parameters[2].1);
    let a = evaluate(&f.graph, &f.feeds).unwrap();
    let mut latest = f.parameters[0].1.clone();
    latest[27..].fill(0.4);
    f.feeds.set("latest", &latest);
    let b = evaluate(&f.graph, &f.feeds).unwrap();
    assert_eq!(
        a[f.graph.outputs()[1] as usize].data[..27],
        b[f.graph.outputs()[1] as usize].data[..27]
    );
    for white in [false, true] {
        f.feeds.set("masks", &masks);
        f.feeds
            .set("redraw", &vec![if white { 80.0 } else { -80.0 }; 54]);
        let v = evaluate(&meganeura::autodiff::differentiate(&f.graph), &f.feeds).unwrap();
        assert!(v.iter().flat_map(|v| &v.data).all(|v| v.is_finite()));
    }
}

#[test]
fn transport_gradients_cross_generated_frames() {
    let mut g = Graph::new();
    let latest = g.parameter("latest", &[1, 27]);
    let redraw = g.parameter("redraw", &[1, 27]);
    let masks = g.parameter("masks", &[1, FIELDS * 9]);
    let first = compose(&mut g, latest, redraw, masks, 1, 3);
    let image = crate::dreamer::networks::centered_pixels(&mut g, first);
    let second = compose(&mut g, image, redraw, masks, 1, 3);
    let loss = gradients::weighted_loss(&mut g, second, 73, 1.0);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.set("latest", &[0.1; 27]);
    feeds.set("redraw", &[0.2; 27]);
    feeds.set("masks", &[0.0; FIELDS * 9]);
    gradients::check(
        &g,
        &feeds,
        &gradients::Options {
            max_elementwise: 32,
            probes: 2,
            ..Default::default()
        },
    )
    .unwrap()
    .assert_passed("two-step transport gradient");
    let b = meganeura::autodiff::differentiate(&g);
    let v = evaluate(&b, &feeds).unwrap();
    for &id in &b.outputs()[1..] {
        assert!(v[id as usize].data.iter().any(|x| x.abs() > 1e-8));
    }
}

#[test]
#[ignore = "requires separately declared guarded pixel-transport F64/native qualification"]
fn transport_native_values_derivatives_extremes_and_lanes_match_f64() {
    let mut f = fixture();
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    assert_eq!(
        session.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    session.clear_optimizer();
    for kind in 0..4 {
        let mut parameters = f.parameters.clone();
        if kind == 1 {
            parameters[0].1[27..].fill(0.4);
        }
        if kind >= 2 {
            parameters[1].1.fill(if kind == 2 { -80.0 } else { 80.0 });
            for b in 0..2 {
                parameters[2].1[b * FIELDS * 9 + 2 * OFFSETS * 9..(b + 1) * FIELDS * 9].fill(-80.0);
            }
        }
        for (name, data) in &parameters {
            session.set_parameter(name, data);
            f.feeds.set(name, data);
        }
        let expected = evaluate(&backward, &f.feeds).unwrap();
        session.step();
        session.wait();
        let check = |actual: &[f32], expected: &[f64]| {
            for (&a, &b) in actual.iter().zip(expected) {
                assert!(
                    a.is_finite() && (f64::from(a) - b).abs() <= 5e-6 + 5e-4 * b.abs(),
                    "{a} vs {b}"
                );
            }
        };
        for (i, &id) in f.graph.outputs().iter().enumerate() {
            let mut a = vec![0.0; f.graph.node(id).ty.num_elements()];
            session.read_output_by_index(i, &mut a);
            check(&a, &expected[backward.outputs()[i] as usize].data);
        }
        for (i, (name, data)) in parameters.iter().enumerate() {
            let mut a = vec![0.0; data.len()];
            session.read_param_grad(name, &mut a);
            check(
                &a,
                &expected[backward.outputs()[f.graph.outputs().len() + i] as usize].data,
            );
            session.read_param(name, &mut a);
            assert_eq!(&a, data);
        }
    }
    assert_eq!(session.adam_step_count(), 0);
}
