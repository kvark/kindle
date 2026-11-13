use super::*;
use crate::dreamer::ObservationKind;
use meganeura::{
    graph::Op,
    reference::{Feeds, evaluate},
};
use serde_json::json;
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.world_microbatch_size = Some(2);
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.world_mean_latents = true;
    c
}

struct Fixture {
    graph: Graph,
    logits: Vec<f32>,
    weights: Vec<f32>,
    probabilities: Vec<f64>,
    gradient: Vec<f64>,
    loss: f64,
}

fn fixture() -> Fixture {
    let mut graph = Graph::new();
    let logits = graph.parameter("logits", &[8, 4]);
    let probabilities = super::super::networks::mixed_probabilities(&mut graph, logits, 8, 4, 0.01);
    let weights = graph.input("weights", &[8, 4]);
    let product = graph.mul(probabilities, weights);
    let loss = graph.sum_all(product);
    graph.set_outputs(vec![loss, probabilities]);
    let logits = (0..32)
        .map(|i| (i % 11) as f32 / 4. - 1.25)
        .collect::<Vec<_>>();
    let weights = (0..32)
        .map(|i| (i % 7) as f32 / 5. - 0.6)
        .collect::<Vec<_>>();
    let mut expected = vec![0.; 32];
    let mut gradient = vec![0.; 32];
    let mut loss = 0.;
    for row in 0..8 {
        let l = &logits[row * 4..(row + 1) * 4];
        let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = l
            .iter()
            .map(|&x| (f64::from(x) - f64::from(max)).exp())
            .collect::<Vec<_>>();
        let total = exp.iter().sum::<f64>();
        let expectation = (0..4)
            .map(|i| exp[i] / total * f64::from(weights[row * 4 + i]))
            .sum::<f64>();
        for (i, &exponential) in exp.iter().enumerate() {
            let index = row * 4 + i;
            let softmax = exponential / total;
            expected[index] = f64::from(1.0_f32 - 0.01) * softmax + f64::from(0.01_f32 / 4.);
            gradient[index] =
                f64::from(1.0_f32 - 0.01) * softmax * (f64::from(weights[index]) - expectation);
            loss += expected[index] * f64::from(weights[index]);
        }
    }
    Fixture {
        graph,
        logits,
        weights,
        probabilities: expected,
        gradient,
        loss,
    }
}

#[test]
fn mean_latent_probabilities_and_true_derivatives_match_f64() {
    let f = fixture();
    let mut feeds = Feeds::new();
    feeds.set("logits", &f.logits).set("weights", &f.weights);
    let derivative = meganeura::autodiff::differentiate(&f.graph);
    let values = evaluate(&derivative, &feeds).unwrap();
    let out = derivative.outputs();
    assert!((values[out[0] as usize].data[0] - f.loss).abs() < 1e-10);
    for (&a, &b) in values[out[1] as usize].data.iter().zip(&f.probabilities) {
        assert!((a - b).abs() < 1e-10);
    }
    for (&a, &b) in values[out[2] as usize].data.iter().zip(&f.gradient) {
        assert!((a - b).abs() < 1e-10);
    }
    let _ = meganeura::compile_training_graph(&f.graph);
}

#[test]
fn cpu_mean_latents_match_probabilities_without_advancing_rng() {
    let c = config();
    let f = fixture();
    let mut rng = StdRng::seed_from_u64(37);
    let mut untouched = rng.clone();
    let values = super::super::gaussian::sample_cpu(&c, &f.logits, &mut rng);
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    for (&a, &b) in values.iter().zip(&f.probabilities) {
        assert!((f64::from(a) - b).abs() < 2e-7);
    }
    assert!(values.iter().all(|&v| v > 0. && v < 1.));
    for row in values.chunks_exact(4) {
        assert!((row.iter().sum::<f32>() - 1.).abs() < 2e-7);
    }
}

#[test]
fn mean_latent_graph_has_no_samples_and_preserves_parameters_and_inference_graphs() {
    let c = config();
    let mut plain = c.clone();
    plain.world_mean_latents = false;
    assert!(
        serde_json::to_value(&plain)
            .unwrap()
            .get("world_mean_latents")
            .is_none()
    );
    let parameters = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>()
    };
    let enabled =
        world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionCrossEntropy);
    let disabled =
        world::build_world_only_pixel_training_graph(&plain, 2, PixelLoss::MotionCrossEntropy);
    assert_eq!(parameters(&enabled), parameters(&disabled));
    assert_eq!(enabled.outputs().len(), disabled.outputs().len());
    assert!(
        !enabled
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ArgMaxInner))
    );
    assert!(
        !enabled
            .nodes()
            .iter()
            .any(|n| matches!(&n.op,Op::Input{name} if name.contains("uniforms")))
    );
    assert!(
        disabled
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ArgMaxInner))
    );
    for build in [
        world::build_observe_graph,
        world::build_transition_graph,
        world::build_observation_prediction_graph,
        world::build_head_graph,
    ] {
        assert_eq!(
            format!("{:?}", build(&c, 2).nodes()),
            format!("{:?}", build(&plain, 2).nodes())
        );
    }
}

#[test]
fn mean_latent_headers_and_constructor_guards_precede_gpu() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    let m = json!({"format":"kindle-world-only-v24","config":c,"pixel_loss":"motion_cross_entropy","prediction_length":2,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 2, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..24 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["config"]["world_mean_latents"] = json!(false);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    for (objective, length) in [
        (PixelLoss::SquaredError, 2),
        (PixelLoss::MotionCrossEntropy, 0),
    ] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), length, objective).is_err());
    }
    for field in [
        "world_gaussian",
        "rgb_spatial_decoder",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
        let changed: DreamerConfig = serde_json::from_value(bad["config"].clone()).unwrap();
        assert!(
            WorldModelLearner::with_pixel_loss(changed, 2, PixelLoss::MotionCrossEntropy).is_err()
        );
    }
    for field in ["color_centroid_supervision", "rgb_color_attention"] {
        let mut bad = m.clone();
        bad["config"][field] =
            json!({"scale":1000.,"channel":0,"background_scale":1.5,"region":[0.,0.,1.,1.]});
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut old = m;
    old["format"] = json!("kindle-world-only-v2");
    old["config"]["world_mean_latents"] = json!(false);
    assert_eq!(
        checkpoint_pixel_loss(&old).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
}

#[test]
#[ignore = "requires separately declared guarded mean-latent qualification"]
fn native_mean_latent_probabilities_and_derivatives() {
    let f = fixture();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    session.set_parameter("logits", &f.logits);
    session.set_input("weights", &f.weights);
    session.step();
    session.wait();
    assert!((f64::from(session.read_loss()) - f.loss).abs() < 2e-6);
    let mut values = vec![0.; 32];
    session.read_output_by_index(1, &mut values);
    for (&a, &b) in values.iter().zip(&f.probabilities) {
        assert!((f64::from(a) - b).abs() < 2e-6);
    }
    let mut gradient = vec![0.; 32];
    session.read_param_grad("logits", &mut gradient);
    for (&a, &b) in gradient.iter().zip(&f.gradient) {
        assert!((f64::from(a) - b).abs() < 2e-6);
    }
    let c = config();
    let mut rng = StdRng::seed_from_u64(41);
    let cpu = super::super::gaussian::sample_cpu(&c, &f.logits, &mut rng);
    for (&a, &b) in values.iter().zip(&cpu) {
        assert!((a - b).abs() < 2e-6);
    }
    assert_eq!(session.read_params(&["logits"])[0], f.logits);
    assert_eq!(session.adam_step_count(), 0);
    eprintln!("Native mean-latent F64 values/true derivatives andCPU probability parity pass");
}
