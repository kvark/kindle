use super::*;
use crate::dreamer::{ModelSize, ObservationKind, VectorDreamerAgent};
use meganeura::graph::Op;
use serde_json::json;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn spatial_decoder_configuration_is_explicit_standalone_only() {
    let c = config();
    assert!(c.check().is_ok());
    assert!(validate_config(&c).is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_spatial_decoder")
            .is_none()
    );
    assert!(VectorDreamerAgent::learned_rgb(c.clone(), 8).is_err());
    for defect in 0..10 {
        let mut bad = c.clone();
        match defect {
            0 => bad.observation_kind = ObservationKind::Features,
            1 => bad.world_gaussian = true,
            2 => bad.rgb_previous_frame = true,
            3 => bad.rgb_latent_only = true,
            4 => bad.rgb_frame_pair = true,
            5 => bad.loss_scales.policy = 1.,
            6 => bad.loss_scales.value = 1.,
            7 => bad.loss_scales.replay_value = 1.,
            8 => bad.actor_critic_gradient = true,
            9 => bad.video_encoder = Some(crate::dreamer::VideoEncoder::Frozen),
            _ => unreachable!(),
        }
        assert!(bad.check().is_err());
    }
    assert!(WorldModelLearner::new(c).is_err());
}

#[test]
fn spatial_decoder_headers_reject_old_formats_objectives_and_phases_before_gpu() {
    let m = json!({"format":"kindle-world-only-v10","config":config(),"pixel_loss":"motion_cross_entropy","updates":0,"prediction_length":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 0, 0).unwrap(),
        TrainingScope::All
    );
    for v in 1..10 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{v}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for (key, value) in [
        ("rgb_spatial_decoder", json!(false)),
        ("world_gaussian", json!(true)),
        ("rgb_latent_only", json!(true)),
        ("rgb_previous_frame", json!(true)),
    ] {
        let mut bad = m.clone();
        bad["config"][key] = value;
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["pixel_loss"] = json!("squared_error");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionCrossEntropy, 0, 0).is_err());
}

#[test]
fn spatial_decoder_preserves_rgb_rssm_geometry_and_counts_additional_state() {
    let mut c = config();
    c.model_size = ModelSize::Size12M;
    c.value_bins = 255;
    c.observation_kind = ObservationKind::Rgb64;
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    let graph = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
    let parameters = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 77);
    assert_eq!(
        parameters
            .iter()
            .map(|n| n.ty.shape.iter().product::<usize>())
            .sum::<usize>(),
        8_857_723
    );
    assert_eq!(c.feature_dim(), 2560);
    assert_eq!(2 * c.observation_dim(), 24576);
    let names = graph
        .nodes()
        .iter()
        .filter_map(|n| {
            if let Op::Input { name } = &n.op {
                Some(name)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len()
    );
    let inference = world::build_observation_prediction_graph(&c, 8);
    assert!(inference.nodes().iter().any(
        |n| matches!(&n.op,Op::Input{name} if name=="decoder_history") && n.ty.shape == [8, 24576]
    ));
}

#[test]
#[ignore = "requires separately declared guarded spatial decoder native mechanism check"]
fn spatial_decoder_native_history_controls() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut c = config();
    c.batch_size = 2;
    let graph = world::build_observation_prediction_graph(&c, 2);
    let mut s = runtime::build_session(&graph, &gpu, Mode::Inference, false);
    runtime::initialize_d3(&mut s, &graph, c.seed);
    s.set_input("deter", &vec![0.; 2 * c.network().deter]);
    s.set_input(
        "stoch",
        &vec![0.; 2 * c.network().stoch * c.network().classes],
    );
    let names = s
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut initial = Vec::new();
    for name in &names {
        let mut value = vec![0.; s.param_size(name).unwrap()];
        s.read_param(name, &mut value);
        initial.push(value);
    }
    let width = c.observation_dim();
    s.set_input("decoder_history", &vec![0.; 4 * width]);
    s.step();
    s.wait();
    let mut baseline = vec![0.; 2 * width];
    s.read_output_by_index(0, &mut baseline);
    for lag in 0..2 {
        let mut history = vec![0.; 4 * width];
        history[lag * width + 16 * 32 + 16] = 0.5;
        s.set_input("decoder_history", &history);
        s.step();
        s.wait();
        let mut value = vec![0.; 2 * width];
        s.read_output_by_index(0, &mut value);
        assert!(
            value
                .iter()
                .all(|v| v.is_finite() && (-0.5..=0.5).contains(v))
        );
        assert!(
            value[..width]
                .iter()
                .zip(&baseline[..width])
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
        assert_eq!(value[width..], baseline[width..]);
    }
    assert_eq!(s.adam_step_count(), 0);
    for (name, expected) in names.iter().zip(initial) {
        let mut value = vec![0.; expected.len()];
        s.read_param(name, &mut value);
        assert_eq!(value, expected);
    }
}
