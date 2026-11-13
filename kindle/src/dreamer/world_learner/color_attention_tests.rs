use super::*;
use crate::dreamer::{ColorCentroidSupervision, ObservationKind};
use meganeura::graph::Op;
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
    c.rgb_color_attention = Some(ColorCentroidSupervision {
        scale: 1000.,
        channel: 0,
        background_scale: 1.5,
        region: [0.0625, 0.4375, 0.9375, 0.875],
    });
    c
}

#[test]
fn color_attention_headers_and_constructor_guards_precede_gpu() {
    let c = config();
    assert!(c.check().is_ok());
    assert!(validate_config(&c).is_ok());
    let m = json!({"format":"kindle-world-only-v23","config":c,"pixel_loss":"motion_cross_entropy","prediction_length":2,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 2, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..23 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["config"]["rgb_color_attention"] = json!(null);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    for (objective, length) in [
        (PixelLoss::SquaredError, 2),
        (PixelLoss::MotionCrossEntropy, 0),
    ] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), length, objective).is_err());
    }
    let mut mixed = c.clone();
    mixed.color_centroid_supervision = c.rgb_color_attention.clone();
    assert!(WorldModelLearner::with_pixel_loss(mixed, 2, PixelLoss::MotionCrossEntropy).is_err());
    for value in [0., -1., f32::NAN, f32::INFINITY] {
        let mut invalid = c.clone();
        invalid.rgb_color_attention.as_mut().unwrap().scale = value;
        assert!(invalid.check().is_err());
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
    }
    let mut old = m;
    old["format"] = json!("kindle-world-only-v2");
    old["config"]["rgb_color_attention"] = json!(null);
    assert_eq!(
        checkpoint_pixel_loss(&old).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
}

#[test]
fn color_attention_has_no_new_parameters_or_inference_path_and_defaults_off() {
    let c = config();
    let mut plain = c.clone();
    plain.rgb_color_attention = None;
    assert!(
        serde_json::to_value(&plain)
            .unwrap()
            .get("rgb_color_attention")
            .is_none()
    );
    let legacy: DreamerConfig =
        serde_json::from_value(serde_json::to_value(&plain).unwrap()).unwrap();
    assert!(legacy.rgb_color_attention.is_none());
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
        !parameters(&enabled)
            .keys()
            .any(|n| n.starts_with("world.color_centroid."))
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
