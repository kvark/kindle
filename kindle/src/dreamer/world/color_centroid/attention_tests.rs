use super::*;
use meganeura::reference::{Feeds, evaluate};

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
    weights: Vec<f64>,
    gradient: Vec<f64>,
    loss: f64,
}

fn fixture() -> Fixture {
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[2, 48]);
    let pixels = graph.parameter("pixels", &[2, 48]);
    let motion = graph.parameter("motion", &[2, 48]);
    let keep = graph.parameter("keep", &[2, 1]);
    let settings = ColorCentroidSupervision {
        scale: 7.,
        channel: 0,
        background_scale: 1.5,
        region: [0.25, 0.25, 0.75, 0.75],
    };
    let weights = attention(&mut graph, &settings, pixels, motion, 2, 4);
    let errors = super::super::super::networks::pixel_cross_entropy(&mut graph, logits, pixels);
    let errors = graph.mul(errors, weights);
    let errors = graph.sum_inner(errors);
    let stopped = graph.stop_gradient(keep);
    let errors = graph.mul(errors, stopped);
    let loss = graph.mean_all(errors);
    graph.set_outputs(vec![loss, weights]);
    let mut images = vec![-0.5; 96];
    images[5] = 0.3;
    images[0] = 0.5;
    images[48 + 10] = 0.1;
    images[48 + 16 + 10] = -0.3;
    images[48 + 32 + 10] = -0.4;
    let logits = (0..96)
        .map(|i| (i % 11) as f32 / 8. - 0.6)
        .collect::<Vec<_>>();
    let motion = (0..96)
        .map(|i| 1. + (i % 7) as f32 / 4.)
        .collect::<Vec<_>>();
    let mut expected = vec![0.; 96];
    for row in 0..2 {
        for channel in 0..3 {
            for i in 0..16 {
                let inside = (1..3).contains(&(i % 4)) && (1..3).contains(&(i / 4));
                let mass = if inside {
                    (f64::from(images[row * 48 + i]) + 0.5)
                        - 1.5
                            * ((f64::from(images[row * 48 + 16 + i]) + 0.5)
                                .max(f64::from(images[row * 48 + 32 + i]) + 0.5))
                } else {
                    0.
                };
                expected[row * 48 + channel * 16 + i] =
                    f64::from(motion[row * 48 + channel * 16 + i]) * (1. + 7. * mass.max(0.));
            }
        }
        let mean =
            expected[row * 48..(row + 1) * 48].iter().sum::<f64>() * f64::from(1.0_f32 / 48.0);
        for v in &mut expected[row * 48..(row + 1) * 48] {
            *v /= mean;
        }
    }
    let mut loss = 0.;
    let mut gradient = vec![0.; 96];
    for i in 0..48 {
        let x = f64::from(logits[i]);
        let target = f64::from(images[i]) + 0.5;
        loss += expected[i] * (x.exp().ln_1p() - x * target) / 2.;
        gradient[i] = expected[i] * (1. / (1. + (-x).exp()) - target) / 2.;
    }
    let parameters: Vec<(String, Vec<f32>)> = vec![
        ("logits".into(), logits),
        ("pixels".into(), images),
        ("motion".into(), motion),
        ("keep".into(), vec![1., 0.]),
    ];
    let mut feeds = Feeds::new();
    for (name, values) in &parameters {
        feeds.set(name, values);
    }
    Fixture {
        graph,
        feeds,
        parameters,
        weights: expected,
        gradient,
        loss,
    }
}

#[test]
fn color_attention_values_derivatives_and_detached_replay_match_f64() {
    let f = fixture();
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let values = evaluate(&backward, &f.feeds).unwrap();
    let out = backward.outputs();
    assert!((values[out[0] as usize].data[0] - f.loss).abs() < 1e-10);
    for (&a, &b) in values[out[1] as usize].data.iter().zip(&f.weights) {
        assert!((a - b).abs() < 1e-10);
    }
    for (&a, &b) in values[out[2] as usize].data.iter().zip(&f.gradient) {
        assert!((a - b).abs() < 1e-10);
    }
    for &node in &out[3..] {
        assert!(values[node as usize].data.iter().all(|v| *v == 0.));
    }
    for row in f.weights.chunks_exact(48) {
        assert!((row.iter().sum::<f64>() / 48. - 1.).abs() < 1e-7);
    }
    assert!(f.weights[5] > f.weights[0]);
    let _ = meganeura::compile_training_graph(&f.graph);
}

#[test]
#[ignore = "requires separately declared guarded color-attention qualification"]
fn native_color_attention_values_derivatives_and_reset_zeros() {
    let f = fixture();
    let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session = super::super::super::runtime::build_session(
        &f.graph,
        &gpu,
        meganeura::Mode::Training,
        false,
    );
    for (name, values) in &f.parameters {
        session.set_parameter(name, values);
    }
    session.step();
    session.wait();
    assert!((f64::from(session.read_loss()) - f.loss).abs() < 2e-5);
    let mut weights = vec![0.; 96];
    session.read_output_by_index(1, &mut weights);
    for (&a, &b) in weights.iter().zip(&f.weights) {
        assert!((f64::from(a) - b).abs() < 3e-6);
    }
    let mut gradient = vec![0.; 96];
    session.read_param_grad("logits", &mut gradient);
    for (&a, &b) in gradient.iter().zip(&f.gradient) {
        assert!((f64::from(a) - b).abs() < 3e-6);
    }
    assert_eq!(&gradient[48..], &[0.; 48]);
    for name in ["pixels", "motion", "keep"] {
        assert!(!session.has_param_grad(name));
    }
    for (name, values) in &f.parameters {
        assert_eq!(&session.read_params(&[name])[0], values);
    }
    assert_eq!(session.adam_step_count(), 0);
    eprintln!(
        "Native color-attention F64 values/derivatives/mean-one weights,detached replay andexact reset zeros pass"
    );
}
