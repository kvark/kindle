use super::*;
use crate::dreamer::runtime;
use meganeura::{
    Mode,
    graph::Op,
    reference::{Feeds, evaluate, gradients},
};
use std::sync::Arc;

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
}

fn fixture() -> Fixture {
    let mut graph = Graph::new();
    let correction = graph.parameter("correction", &[2, 7]);
    let history = graph.parameter("history", &[2, 14]);
    let logits = residual_logits(&mut graph, correction, history, 2, 7);
    let prediction = super::super::centered_pixels(&mut graph, logits);
    let target = graph.constant(vec![0.2; 14], &[2, 7]);
    let errors = super::super::pixel_cross_entropy(&mut graph, logits, target);
    let loss = graph.mean_all(errors);
    graph.set_outputs(vec![loss, logits, prediction]);
    let parameters: Vec<(String, Vec<f32>)> = vec![
        (
            "correction".into(),
            (0..14).map(|i| (i % 5) as f32 / 10. - 0.2).collect(),
        ),
        (
            "history".into(),
            (0..28)
                .map(|i| [-0.5, -0.4, -0.2, 0., 0.2, 0.4, 0.5][i % 7])
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

#[test]
fn spatial_residual_zero_correction_copies_only_latest_image_in_each_lane() {
    let Fixture {
        graph,
        mut feeds,
        parameters,
    } = fixture();
    feeds.set("correction", &[0.; 14]);
    let before = evaluate(&graph, &feeds).unwrap();
    let prediction = &before[graph.outputs()[2] as usize].data;
    let history = &parameters[1].1;
    for b in 0..2 {
        for i in 0..7 {
            let latest = f64::from(history[b * 14 + 7 + i]);
            let expected =
                (latest + 0.5).clamp(f64::from(1. / 255.0f32), f64::from(254. / 255.0f32)) - 0.5;
            assert!((prediction[b * 7 + i] - expected).abs() < 1e-12);
            assert!((prediction[b * 7 + i] - latest).abs() <= 1. / 255. + 1e-7);
        }
    }
    let mut changed = history.clone();
    changed[..7].fill(0.125);
    feeds.set("history", &changed);
    let older = evaluate(&graph, &feeds).unwrap();
    assert_eq!(older[graph.outputs()[2] as usize].data, *prediction);
    changed[8] = 0.3;
    feeds.set("history", &changed);
    let latest = evaluate(&graph, &feeds).unwrap();
    let actual = &latest[graph.outputs()[2] as usize].data;
    assert_ne!(actual[1], prediction[1]);
    for i in (0..14).filter(|&i| i != 1) {
        assert_eq!(actual[i], prediction[i]);
    }
}

#[test]
fn spatial_residual_copy_and_correction_derivatives_match_finite_differences() {
    let Fixture { graph, feeds, .. } = fixture();
    gradients::check(
        &graph,
        &feeds,
        &gradients::Options {
            max_elementwise: 64,
            probes: 2,
            tolerance: meganeura::reference::Tolerance {
                rtol: 1e-5,
                floor: 1e-5,
            },
            ..Default::default()
        },
    )
    .unwrap()
    .assert_passed("copy/correction derivatives");
}

#[test]
fn spatial_residual_generated_history_keeps_the_multistep_gradient() {
    let mut graph = Graph::new();
    let history = graph.parameter("history", &[2, 6]);
    let first = graph.parameter("first", &[2, 3]);
    let second = graph.parameter("second", &[2, 3]);
    let first = residual_logits(&mut graph, first, history, 2, 3);
    let first = super::super::centered_pixels(&mut graph, first);
    let shifted = crate::dreamer::spatial_history::shift(&mut graph, history, first, 2, 3);
    let second = residual_logits(&mut graph, second, shifted, 2, 3);
    let target = graph.constant(vec![0.125; 6], &[2, 3]);
    let errors = super::super::pixel_cross_entropy(&mut graph, second, target);
    let loss = graph.mean_all(errors);
    graph.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.set("history", &[-0.3; 12]);
    feeds.set("first", &[0.1; 6]);
    feeds.set("second", &[-0.2; 6]);
    gradients::check(&graph, &feeds, &gradients::Options::default())
        .unwrap()
        .assert_passed("multistep copy gradient");
    let backward = meganeura::autodiff::differentiate(&graph);
    let values = evaluate(&backward, &feeds).unwrap();
    for &gradient in &backward.outputs()[1..] {
        assert!(
            values[gradient as usize]
                .data
                .iter()
                .any(|x| x.abs() > 1e-6)
        );
    }
}

#[test]
fn spatial_residual_decoder_preserves_parameter_layout_and_adds_causal_history_path() {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    let build = |c: &DreamerConfig| {
        let mut graph = Graph::new();
        let decoder = Decoder::new(&mut graph, c, "world.decoder", c.feature_dim());
        let features = graph.input("features", &[2, c.feature_dim()]);
        let history = graph.input("history", &[2, 2 * c.observation_dim()]);
        let logits = decoder.forward_logits_with_history(&mut graph, features, Some(history), 2);
        graph.set_outputs(vec![logits]);
        graph
    };
    let control = build(&c);
    c.rgb_spatial_residual = true;
    let candidate = build(&c);
    let layout = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(layout(&candidate), layout(&control));
    let output = candidate.node(candidate.outputs()[0]);
    assert!(matches!(output.op, Op::Add));
    assert!(
        candidate
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Log))
            .count()
            >= 2
    );
    assert_eq!(output.ty.shape, [2, c.observation_dim()]);
}

#[test]
#[ignore = "requires separately declared guarded copy/correction F64/native check"]
fn spatial_residual_native_values_and_derivatives_match_f64() {
    let Fixture {
        graph,
        feeds,
        parameters,
    } = fixture();
    let backward = meganeura::autodiff::differentiate(&graph);
    let expected = evaluate(&backward, &feeds).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut s = runtime::build_session(&graph, &gpu, Mode::Training, false);
    assert_eq!(
        s.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for (name, values) in &parameters {
        s.set_parameter(name, values);
    }
    s.clear_optimizer();
    s.step();
    s.wait();
    let compare = |actual: &[f32], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (f64::from(a) - b).abs() <= 5e-5 + 5e-4 * b.abs(),
                "{a} vs {b}"
            );
        }
    };
    for (i, &id) in graph.outputs().iter().enumerate() {
        let mut actual = vec![0.; graph.node(id).ty.num_elements()];
        s.read_output_by_index(i, &mut actual);
        compare(&actual, &expected[backward.outputs()[i] as usize].data);
    }
    for (i, (name, initial)) in parameters.iter().enumerate() {
        let mut actual = vec![0.; initial.len()];
        s.read_param_grad(name, &mut actual);
        compare(
            &actual,
            &expected[backward.outputs()[graph.outputs().len() + i] as usize].data,
        );
        s.read_param(name, &mut actual);
        assert_eq!(&actual, initial);
    }
    assert_eq!(s.adam_step_count(), 0);
}

#[test]
#[ignore = "requires separately declared guarded production decoder copy check"]
fn spatial_residual_native_decoder_zero_correction_copies_latest_and_preserves_lanes() {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_residual = true;
    let batch = 2;
    let width = c.observation_dim();
    let mut graph = Graph::new();
    let decoder = Decoder::new(&mut graph, &c, "world.decoder", c.feature_dim());
    let features = graph.input("features", &[batch, c.feature_dim()]);
    let history = graph.input("history", &[batch, 2 * width]);
    let logits = decoder.forward_logits_with_history(&mut graph, features, Some(history), batch);
    let image = super::super::centered_pixels(&mut graph, logits);
    graph.set_outputs(vec![image]);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut s = runtime::build_session(&graph, &gpu, Mode::Inference, false);
    assert_eq!(
        s.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    runtime::initialize_d3(&mut s, &graph, c.seed);
    for name in ["world.decoder.imgout.weight", "world.decoder.imgout.bias"] {
        s.set_parameter(name, &vec![0.; s.param_size(name).unwrap()]);
    }
    s.set_input("features", &vec![0.; batch * c.feature_dim()]);
    let mut histories = (0..batch * 2 * width)
        .map(|i| (i % 251) as f32 / 255. - 0.5)
        .collect::<Vec<_>>();
    let mut outputs = Vec::new();
    for change in [false, true] {
        if change {
            histories[width..2 * width].fill(0.25);
        }
        s.set_input("history", &histories);
        s.step();
        s.wait();
        let mut image = vec![0.; batch * width];
        s.read_output_by_index(0, &mut image);
        for b in 0..batch {
            for i in 0..width {
                let expected = (histories[b * 2 * width + width + i] + 0.5)
                    .clamp(1. / 255., 254. / 255.)
                    - 0.5;
                assert!((image[b * width + i] - expected).abs() < 3e-6);
            }
        }
        outputs.push(image);
    }
    assert_ne!(outputs[0][..width], outputs[1][..width]);
    assert_eq!(outputs[0][width..], outputs[1][width..]);
    assert_eq!(s.adam_step_count(), 0);
}
