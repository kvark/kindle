use super::*;
use crate::dreamer::{ModelSize, ObservationKind, agent::DreamerCore, world, world_learner};
use meganeura::graph::Op;
use serde_json::json;
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb64;
    c.model_size = ModelSize::Size1M;
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(8);
    c.value_bins = 255;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_late_action = true;
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
        "pixel_loss":"motion_event_rollout", "prediction_length":16, "updates":0});
    if features {
        super::super::action_features::wrap_metadata(&mut m).unwrap();
    }
    wrap_metadata(&mut m).unwrap();
    m
}

#[test]
fn late_action_headers_bind_both_base_formats_and_reject_disguises_before_gpu() {
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_late_action")
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
            super::super::TrainingScope::from_metadata(&m, PixelLoss::MotionEventRollout, 16, 0)
                .unwrap(),
            super::super::TrainingScope::All
        );
        let base = base_metadata(&m).unwrap().unwrap();
        assert_eq!(
            base["config"]["world_action_features"],
            m["config"]["world_action_features"]
        );
        assert!(base["config"].get("rgb_late_action").is_none());
        for version in 1..26 {
            let mut bad = m.clone();
            bad["format"] = json!(format!("kindle-world-only-v{version}"));
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        }
        for (key, value) in [
            ("late_action_base_format", json!("kindle-world-only-v2")),
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
        bad["config"]["rgb_late_action"] = json!(false);
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m.clone();
        bad["late_action_base_format"] = json!(if features {
            "kindle-world-only-v18"
        } else {
            "kindle-world-only-v25"
        });
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m;
        bad["config"]["rgb_late_action"] = json!(false);
        bad["format"] = json!("kindle-world-only-v18");
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
    }
}

#[test]
fn late_action_rejects_unsupported_construction_and_whole_agents_before_gpu() {
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
    invalid.rgb_late_action = true;
    assert!(std::panic::catch_unwind(|| DreamerCore::new(invalid)).is_err());
}

#[test]
fn late_action_moves_exactly_7200_weights_without_changing_world_size() {
    let mut c = config();
    let parameters = |c: &DreamerConfig| {
        let graph =
            world::build_world_only_pixel_training_graph(c, 16, PixelLoss::MotionEventRollout);
        graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>()
    };
    let late = parameters(&c);
    c.rgb_late_action = false;
    let early = parameters(&c);
    assert_eq!(late.len(), 80);
    assert_eq!(
        late.keys().collect::<Vec<_>>(),
        early.keys().collect::<Vec<_>>()
    );
    let count = |p: &BTreeMap<String, Vec<usize>>| {
        p.values()
            .map(|shape| shape.iter().product::<usize>())
            .sum::<usize>()
    };
    assert_eq!(count(&late), 603_915);
    assert_eq!(count(&early), count(&late));
    let changed = late
        .iter()
        .filter(|(key, shape)| early[*key] != **shape)
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        changed,
        [
            "world.decoder.fusion.weight",
            "world.decoder.motion0.weight"
        ]
    );
    for (name, delta) in [
        ("world.decoder.motion0.weight", -7200i64),
        ("world.decoder.fusion.weight", 7200),
    ] {
        let elements = |p: &BTreeMap<String, Vec<usize>>| p[name].iter().product::<usize>() as i64;
        assert_eq!(elements(&late) - elements(&early), delta);
    }
}
