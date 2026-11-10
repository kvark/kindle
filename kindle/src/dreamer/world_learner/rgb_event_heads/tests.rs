use super::*;
use crate::dreamer::{ModelSize, ObservationKind, agent::DreamerCore, world, world_learner};
use meganeura::graph::Op;
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.model_size = ModelSize::Size1M;
    c.observation_kind = ObservationKind::Rgb64;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.world_microbatch_size = Some(2);
    c.value_bins = 255;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_event_heads = true;
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
fn rgb_event_headers_bind_flag_geometry_base_and_full_training_scope_before_gpu() {
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_event_heads")
            .is_none()
    );
    for features in [false, true] {
        let m = metadata(features);
        assert_eq!(m["format"], FORMAT);
        assert_eq!(
            m["rgb_event_head_geometry"],
            json!({"latent":640,"visual":256,"inputs":896,"units":64})
        );
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
        assert!(base["config"].get("rgb_event_heads").is_none());
        assert_eq!(
            base["config"]["world_action_features"],
            m["config"]["world_action_features"]
        );
        for version in 1..28 {
            let mut bad = m.clone();
            bad["format"] = json!(format!("kindle-world-only-v{version}"));
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        }
        for (key, value) in [
            ("rgb_event_head_base_format", json!("kindle-world-only-v2")),
            (
                "rgb_event_head_geometry",
                json!({"latent":640,"visual":256,"inputs":640,"units":64}),
            ),
            ("training_scope", json!("all")),
            ("phase_start_updates", json!(1)),
            ("prediction_length", json!(8)),
            ("pixel_loss", json!("motion_posterior_history")),
        ] {
            let mut bad = m.clone();
            bad[key] = value;
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err(), "{key}");
        }
        for field in ["rgb_event_head_base_format", "rgb_event_head_geometry"] {
            let mut bad = m.clone();
            bad.as_object_mut().unwrap().remove(field);
            assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        }
        let mut bad = m.clone();
        bad["config"]
            .as_object_mut()
            .unwrap()
            .remove("rgb_event_heads");
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m.clone();
        bad["config"]["rgb_event_heads"] = json!(false);
        bad["format"] = json!("kindle-world-only-v18");
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
        let mut bad = m;
        bad["rgb_event_head_base_format"] = json!(if features {
            "kindle-world-only-v18"
        } else {
            "kindle-world-only-v25"
        });
        assert!(world_learner::checkpoint_pixel_loss(&bad).is_err());
    }
}

#[test]
fn rgb_event_scope_rejects_unqualified_models_objectives_and_whole_agents_before_gpu() {
    let c = config();
    assert!(validate(&c, 16, PixelLoss::MotionEventRollout).is_ok());
    for (h, loss) in [
        (0, PixelLoss::MotionEventRollout),
        (8, PixelLoss::MotionEventRollout),
        (16, PixelLoss::MotionCrossEntropy),
        (16, PixelLoss::MotionPosteriorHistory),
        (16, PixelLoss::MotionCleanPrior),
        (16, PixelLoss::MotionFixedTeacherEvents),
    ] {
        assert!(world_learner::WorldModelLearner::with_pixel_loss(c.clone(), h, loss).is_err());
    }
    for field in [
        "rgb_spatial_action",
        "rgb_spatial_fusion",
        "rgb_spatial_decoder",
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        v[field] = json!(false);
        let bad: DreamerConfig = serde_json::from_value(v).unwrap();
        assert!(bad.check().is_err());
    }
    for field in [
        "rgb_wide_motion",
        "rgb_late_action",
        "rgb_spatial_transport",
        "rgb_spatial_residual",
        "world_image_feedback",
        "world_gaussian",
        "world_mean_latents",
        "rgb_previous_frame",
        "rgb_frame_pair",
        "rgb_latent_only",
        "rgb_coordinates",
        "rgb_stride2",
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        v[field] = json!(true);
        let bad: DreamerConfig = serde_json::from_value(v).unwrap();
        assert!(bad.check().is_err(), "{field}");
    }
    for (field, value) in [
        ("observation_kind", json!("rgb32")),
        ("model_size", json!("size12_m")),
        ("prior_reward_scale", json!(1.)),
        ("prior_continuation_scale", json!(1.)),
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        v[field] = value;
        let bad = serde_json::from_value::<DreamerConfig>(v).unwrap();
        assert!(
            validate(&bad, 16, PixelLoss::MotionEventRollout).is_err(),
            "{field}"
        );
    }
    let mut bad = DreamerConfig::tiny(18);
    bad.rgb_event_heads = true;
    assert!(std::panic::catch_unwind(|| DreamerCore::new(bad)).is_err());
}

#[test]
fn rgb_event_heads_share_encoder_and_change_only_two_input_matrices() {
    let mut c = config();
    let parameters = |c: &DreamerConfig| {
        let g = world::build_world_only_pixel_training_graph(c, 2, PixelLoss::MotionEventRollout);
        let mut p = BTreeMap::new();
        for n in g.nodes() {
            if let Op::Parameter { name } = &n.op {
                assert!(
                    p.insert(name.clone(), n.ty.shape.clone()).is_none(),
                    "duplicate {name}"
                );
            }
        }
        p
    };
    let visual = parameters(&c);
    c.rgb_event_heads = false;
    let original = parameters(&c);
    assert_eq!(visual.len(), 80);
    assert_eq!(
        visual.keys().collect::<Vec<_>>(),
        original.keys().collect::<Vec<_>>()
    );
    let count = |p: &BTreeMap<String, Vec<usize>>| {
        p.values()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>()
    };
    assert_eq!(count(&original), 603_915);
    assert_eq!(count(&visual), 636_683);
    let changed = visual
        .iter()
        .filter(|(k, s)| original[*k] != **s)
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        changed,
        [
            "world.continuation.layer0.weight",
            "world.reward.layer0.weight"
        ]
    );
    for name in changed {
        assert_eq!(original[name].iter().product::<usize>(), 640 * 64);
        assert_eq!(visual[name].iter().product::<usize>(), 896 * 64);
    }
}
