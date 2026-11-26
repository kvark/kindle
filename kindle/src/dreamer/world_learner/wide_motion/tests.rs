use super::*;
use crate::dreamer::{ModelSize, ObservationKind, agent::DreamerCore, world, world_learner};
use meganeura::graph::Op;
use serde_json::json;
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb64;
    c.model_size = ModelSize::Size1M;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.world_microbatch_size = Some(2);
    c.value_bins = 255;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_wide_motion = true;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c
}

fn metadata(features: bool) -> Value {
    let mut c = config();
    if features {
        c.world_action_features = Some(vec![vec![0.; 18]; 18]);
    }
    let mut m = json!({"format":"kindle-world-only-v18", "config":c,
        "pixel_loss":"motion_event_rollout", "prediction_length":2, "updates":0});
    if features {
        super::super::action_features::wrap_metadata(&mut m).unwrap();
    }
    wrap_metadata(&mut m).unwrap();
    m
}

#[test]
fn wide_motion_headers_bind_both_base_formats_and_reject_disguises_before_gpu() {
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_wide_motion")
            .is_none()
    );
    for features in [false, true] {
        let m = metadata(features);
        assert_eq!(m["format"], FORMAT);
        assert_eq!(
            world_learner::checkpoint_pixel_loss(&m).unwrap(),
            PixelLoss::MotionEventRollout
        );
        assert_eq!(
            super::super::TrainingScope::from_metadata(&m, PixelLoss::MotionEventRollout, 2, 0)
                .unwrap(),
            super::super::TrainingScope::All
        );
        let base = base_metadata(&m).unwrap().unwrap();
        assert_eq!(
            base["config"]["world_action_features"],
            m["config"]["world_action_features"]
        );
        assert!(base["config"].get("rgb_wide_motion").is_none());
        for version in 1..27 {
            let mut bad = m.clone();
            bad["format"] = json!(format!("kindle-world-only-v{version}"));
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        }
        for (key, value) in [
            ("wide_motion_base_format", json!("kindle-world-only-v2")),
            ("training_scope", json!("all")),
            ("phase_start_updates", json!(1)),
            ("prediction_length", json!(0)),
            ("pixel_loss", json!("motion_posterior_history")),
        ] {
            let mut bad = m.clone();
            bad[key] = value;
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err(), "{key}");
        }
        let mut bad = m.clone();
        bad["config"]["rgb_wide_motion"] = json!(false);
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m.clone();
        bad["wide_motion_base_format"] = json!(if features {
            "kindle-world-only-v18"
        } else {
            "kindle-world-only-v25"
        });
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m;
        bad["config"]["rgb_wide_motion"] = json!(false);
        bad["format"] = json!("kindle-world-only-v18");
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
    }
}

#[test]
fn wide_motion_rejects_unsupported_construction_and_whole_agents_before_gpu() {
    let c = config();
    assert!(validate(&c, 16, PixelLoss::MotionEventRollout).is_ok());
    for loss in [
        PixelLoss::MotionCrossEntropy,
        PixelLoss::MotionPosteriorHistory,
        PixelLoss::MotionCleanPrior,
        PixelLoss::MotionFixedTeacherEvents,
    ] {
        assert!(world_learner::WorldModelLearner::with_pixel_loss(c.clone(), 16, loss).is_err());
    }
    assert!(
        world_learner::WorldModelLearner::with_pixel_loss(
            c.clone(),
            0,
            PixelLoss::MotionEventRollout
        )
        .is_err()
    );
    for field in [
        "rgb_spatial_action",
        "rgb_spatial_fusion",
        "rgb_spatial_decoder",
    ] {
        let mut value = serde_json::to_value(&c).unwrap();
        value[field] = json!(false);
        let bad: DreamerConfig = serde_json::from_value(value).unwrap();
        assert!(bad.check().is_err());
    }
    for field in [
        "rgb_late_action",
        "rgb_spatial_transport",
        "rgb_spatial_residual",
        "world_image_feedback",
        "world_gaussian",
        "world_mean_latents",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut value = serde_json::to_value(&c).unwrap();
        value[field] = json!(true);
        let bad: DreamerConfig = serde_json::from_value(value).unwrap();
        assert!(bad.check().is_err());
    }
    let mut invalid = DreamerConfig::tiny(18);
    invalid.rgb_wide_motion = true;
    assert!(std::panic::catch_unwind(|| DreamerCore::new(invalid)).is_err());
}

#[test]
fn wide_motion_adds_only_the_declared_local_parameter_shapes() {
    let mut c = config();
    let parameters = |c: &DreamerConfig| {
        let g = world::build_world_only_pixel_training_graph(c, 2, PixelLoss::MotionEventRollout);
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>()
    };
    let wide = parameters(&c);
    c.rgb_wide_motion = false;
    let original = parameters(&c);
    assert_eq!(wide.len(), 80);
    assert_eq!(
        wide.keys().collect::<Vec<_>>(),
        original.keys().collect::<Vec<_>>()
    );
    let count = |p: &BTreeMap<String, Vec<usize>>| {
        p.values()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>()
    };
    assert_eq!(count(&original), 603_915);
    assert_eq!(count(&wide), 675_643);
    let changed = wide
        .iter()
        .filter(|(k, s)| original[*k] != **s)
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        changed,
        [
            "world.decoder.fusion.bias",
            "world.decoder.fusion.norm.weight",
            "world.decoder.fusion.weight",
            "world.decoder.imgout.weight",
            "world.decoder.motion0.bias",
            "world.decoder.motion0.norm.weight",
            "world.decoder.motion0.weight",
            "world.decoder.motion1.bias",
            "world.decoder.motion1.norm.weight",
            "world.decoder.motion1.weight",
            "world.decoder.motion2.bias",
            "world.decoder.motion2.norm.weight",
            "world.decoder.motion2.weight"
        ]
    );
    assert_eq!(count(&wide) - count(&original), 71_728);
}
