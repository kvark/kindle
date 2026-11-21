use super::*;
use crate::dreamer::{ModelSize, ObservationKind, world_learner};

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.model_size = ModelSize::Size1M;
    c.observation_kind = ObservationKind::Rgb64;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_event_heads = true;
    c.world_survival_scale = Some(10.);
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c
}

fn metadata() -> Value {
    let c = config();
    let mut m = json!({"format":"kindle-world-only-v18","config":c,"updates":0,
        "pixel_loss":"motion_event_rollout","prediction_length":16});
    rgb_event_heads::wrap_metadata(&mut m).unwrap();
    wrap_metadata(&mut m).unwrap();
    m
}

#[test]
fn cumulative_survival_header_binds_coefficient_base_scope_and_legacy_compatibility() {
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("world_survival_scale")
            .is_none()
    );
    let m = metadata();
    assert_eq!(m["format"], FORMAT);
    assert_eq!(
        world_learner::checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionEventRollout
    );
    let base = base_metadata(&m).unwrap().unwrap();
    assert!(base["config"].get("world_survival_scale").is_none());
    assert_eq!(base["format"], "kindle-world-only-v28");
    assert_eq!(
        world_learner::checkpoint_pixel_loss(&base).unwrap(),
        PixelLoss::MotionEventRollout
    );
    for (key, value) in [
        ("format", json!("kindle-world-only-v28")),
        ("world_survival_base_format", json!("kindle-world-only-v25")),
        ("world_survival_scale", json!(1.)),
        ("training_scope", json!("all")),
        ("prediction_length", json!(32)),
        ("pixel_loss", json!("motion_cross_entropy")),
    ] {
        let mut bad = m.clone();
        bad[key] = value;
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err(), "{key}");
    }
    for key in ["world_survival_base_format", "world_survival_scale"] {
        let mut bad = m.clone();
        bad.as_object_mut().unwrap().remove(key);
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err(), "{key}");
    }
    for (key, value) in [
        ("world_survival_scale", json!(0.)),
        ("world_survival_scale", json!(-1.)),
        ("world_survival_scale", Value::Null),
        ("rgb_event_heads", json!(false)),
    ] {
        let mut bad = m.clone();
        bad["config"][key] = value;
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err(), "{key}");
    }
    let mut bad = base.clone();
    bad["world_survival_scale"] = json!(10.);
    assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
}

#[test]
fn cumulative_survival_invalid_public_construction_rejects_before_gpu() {
    let mut whole_agent = DreamerConfig::tiny(18);
    whole_agent.world_survival_scale = Some(10.);
    assert!(
        std::panic::catch_unwind(|| crate::dreamer::agent::DreamerCore::new(whole_agent)).is_err()
    );
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        let mut c = config();
        c.world_survival_scale = Some(scale);
        assert!(c.check().is_err());
        assert!(
            world_learner::WorldModelLearner::with_pixel_loss(c, 16, PixelLoss::MotionEventRollout)
                .is_err()
        );
    }
    let mut c = config();
    c.rgb_event_heads = false;
    assert!(
        world_learner::WorldModelLearner::with_pixel_loss(c, 16, PixelLoss::MotionEventRollout)
            .is_err()
    );
    assert!(
        world_learner::WorldModelLearner::with_pixel_loss(
            config(),
            32,
            PixelLoss::MotionEventRollout
        )
        .is_err()
    );
    assert!(
        world_learner::WorldModelLearner::with_pixel_loss(
            config(),
            16,
            PixelLoss::MotionCrossEntropy
        )
        .is_err()
    );
}
