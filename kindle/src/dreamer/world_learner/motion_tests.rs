use super::*;
use crate::dreamer::networks::{pixel_cross_entropy, pixel_motion_weights};
use meganeura::reference::{Feeds, evaluate};

fn fixture(width: usize) -> (Graph, Feeds, Vec<f32>, Vec<f64>, Vec<f64>, f64) {
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[2, width]);
    let current = graph.input("current", &[2, width]);
    let previous = graph.input("previous", &[2, width]);
    let keep = graph.input("keep", &[2, 1]);
    let weight = pixel_motion_weights(&mut graph, current, previous, keep);
    let loss = pixel_cross_entropy(&mut graph, logits, current);
    let loss = graph.mul(loss, weight);
    let loss = graph.sum_all(loss);
    let loss = graph.scale(loss, 0.5);
    graph.set_outputs(vec![loss, weight]);
    let values = (0..2 * width)
        .map(|i| ((i * 7 % 17) as f32 - 8.) / 2.)
        .collect::<Vec<_>>();
    let targets = (0..2 * width)
        .map(|i| (i % 5) as f32 / 5. - 0.4)
        .collect::<Vec<_>>();
    let previous = targets
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 3 == 0 { v } else { -v * 0.5 })
        .collect::<Vec<_>>();
    let mut expected_weights = targets
        .iter()
        .zip(&previous)
        .enumerate()
        .map(|(i, (&a, &b))| {
            1. + if i < width {
                0.
            } else {
                100. * (f64::from(a) - f64::from(b)).abs()
            }
        })
        .collect::<Vec<_>>();
    for row in expected_weights.chunks_mut(width) {
        let mean = row.iter().sum::<f64>() / width as f64;
        for value in row {
            *value /= mean;
        }
    }
    let mut loss = 0.;
    let mut gradient = Vec::new();
    for ((&x, &y), &w) in values.iter().zip(&targets).zip(&expected_weights) {
        let x = f64::from(x);
        let y = f64::from(y) + 0.5;
        loss += (x.max(0.) + (-x.abs()).exp().ln_1p() - x * y) * w / 2.;
        gradient.push((1. / (1. + (-x).exp()) - y) * w / 2.);
    }
    let mut feeds = Feeds::new();
    feeds
        .set("logits", &values)
        .set("current", &targets)
        .set("previous", &previous)
        .set("keep", &[0., 1.]);
    (graph, feeds, values, expected_weights, gradient, loss)
}

#[test]
fn motion_attention_matches_f64_and_has_uniform_reset_rows() {
    let (graph, feeds, _, weights, gradient, loss) = fixture(8);
    let backward = meganeura::autodiff::differentiate(&graph);
    let actual = evaluate(&backward, &feeds).unwrap();
    assert!((actual[backward.outputs()[0] as usize].data[0] - loss).abs() < 1e-10);
    for (a, b) in actual[backward.outputs()[1] as usize]
        .data
        .iter()
        .zip(&weights)
    {
        assert!((a - b).abs() < 1e-10);
    }
    for (a, b) in actual[backward.outputs()[2] as usize]
        .data
        .iter()
        .zip(&gradient)
    {
        assert!((a - b).abs() < 1e-10);
    }
    assert!(weights[..8].iter().all(|&x| x == 1.));
}

#[test]
fn motion_attention_is_detached_from_image_targets() {
    let mut graph = Graph::new();
    let current = graph.parameter("current", &[2, 8]);
    let previous = graph.parameter("previous", &[2, 8]);
    let keep = graph.constant(vec![1., 1.], &[2, 1]);
    let weights = pixel_motion_weights(&mut graph, current, previous, keep);
    let loss = graph.sum_all(weights);
    graph.set_outputs(vec![loss]);
    let backward = meganeura::autodiff::differentiate(&graph);
    let mut feeds = Feeds::new();
    feeds.set("current", &[0.2; 16]).set("previous", &[0.0; 16]);
    let values = evaluate(&backward, &feeds).unwrap();
    for &output in &backward.outputs()[1..] {
        assert!(values[output as usize].data.iter().all(|&x| x == 0.));
    }
}

#[test]
fn motion_objective_preserves_parameters_and_checkpoint_objectives_are_versioned() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = super::super::ObservationKind::Rgb64;
    config.actor_critic_gradient = false;
    config.disagreement_bonus = false;
    config.loss_scales.policy = 0.;
    config.loss_scales.value = 0.;
    config.loss_scales.replay_value = 0.;
    let normal = world::build_world_only_pixel_training_graph(&config, 0, PixelLoss::CrossEntropy);
    let motion =
        world::build_world_only_pixel_training_graph(&config, 0, PixelLoss::MotionCrossEntropy);
    let names = |graph: &Graph| {
        graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                meganeura::graph::Op::Parameter { name } => {
                    Some((name.clone(), n.ty.shape.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&normal), names(&motion));
    meganeura::compile_training_graph(&motion);
    let read = |v| checkpoint_pixel_loss(&v);
    assert_eq!(
        read(serde_json::json!({"format":"kindle-world-only-v1"})).unwrap(),
        PixelLoss::SquaredError
    );
    assert_eq!(
        read(serde_json::json!({"format":"kindle-world-only-v1","pixel_loss":"cross_entropy"}))
            .unwrap(),
        PixelLoss::CrossEntropy
    );
    assert_eq!(
        read(
            serde_json::json!({"format":"kindle-world-only-v2","pixel_loss":"motion_cross_entropy"})
        )
        .unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert!(read(serde_json::json!({"format":"kindle-world-only-v2"})).is_err());
    assert!(read(serde_json::json!({"format":"unknown"})).is_err());
}

#[test]
fn predictive_objective_is_explicit_and_legacy_objective_readers_reject_it() {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum LegacyPixelLoss {
        SquaredError,
        CrossEntropy,
        MotionCrossEntropy,
    }
    let encoded = serde_json::to_value(PixelLoss::PredictiveMotionEnergy).unwrap();
    assert_eq!(encoded, "predictive_motion_energy");
    assert!(serde_json::from_value::<LegacyPixelLoss>(encoded.clone()).is_err());
    let metadata = serde_json::json!({"format":"kindle-world-only-v2","pixel_loss":encoded});
    assert_eq!(
        checkpoint_pixel_loss(&metadata).unwrap(),
        PixelLoss::PredictiveMotionEnergy
    );
}

#[test]
#[ignore = "requires guarded GPU; image-wise motion normalization and derivatives"]
fn motion_attention_native_matches_f64() {
    let (graph, feeds, values, weights, gradient, loss) = fixture(1024);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session = runtime::build_session(&graph, &gpu, Mode::Training, false);
    session.set_parameter("logits", &values);
    for name in ["current", "previous", "keep"] {
        session.set_input(name, &feeds.f32(name).unwrap());
    }
    session.step();
    session.wait();
    let mut output = [0.];
    let mut actual_weights = vec![0.; weights.len()];
    let mut actual_gradient = vec![0.; gradient.len()];
    session.read_output_by_index(0, &mut output);
    session.read_output_by_index(1, &mut actual_weights);
    session.read_param_grad("logits", &mut actual_gradient);
    assert!(
        (f64::from(output[0]) - loss).abs() < 2e-3,
        "{} vs {loss}",
        output[0]
    );
    for (a, b) in actual_weights.iter().zip(weights) {
        assert!((f64::from(*a) - b).abs() < 2e-5);
    }
    for (a, b) in actual_gradient.iter().zip(gradient) {
        assert!((f64::from(*a) - b).abs() < 2e-6);
    }
}
