use super::*;
use crate::dreamer::ObservationKind;
use meganeura::graph::Op;
use serde_json::json;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.world_backprop_length = c.batch_length;
    c.world_microbatch_size = Some(c.batch_size);
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn suffix_events_require_explicit_v13_objective_and_fusion_without_a_phase() {
    let m = json!({"format":"kindle-world-only-v13", "config":config(),
        "pixel_loss":"motion_event_rollout", "prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionEventRollout
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionEventRollout, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..13 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in ["rgb_spatial_decoder", "rgb_spatial_fusion"] {
        let mut bad = m.clone();
        bad["config"][field] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in [
        "world_image_feedback",
        "world_gaussian",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for objective in [
        "squared_error",
        "motion_cross_entropy",
        "motion_image_feedback",
    ] {
        let mut bad = m.clone();
        bad["pixel_loss"] = json!(objective);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["prediction_length"] = json!(0);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionEventRollout, 16, 0).is_err());
    assert!(
        WorldModelLearner::with_pixel_loss(config(), 0, PixelLoss::MotionEventRollout).is_err()
    );
    let mut bad = config();
    bad.rgb_spatial_fusion = false;
    assert!(WorldModelLearner::with_pixel_loss(bad, 16, PixelLoss::MotionEventRollout).is_err());
}

#[test]
fn suffix_events_add_two_reports_without_parameters_or_feedback_inputs() {
    let mut c = config();
    c.model_size = super::super::ModelSize::Size12M;
    c.observation_kind = ObservationKind::Rgb64;
    c.value_bins = 255;
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(8);
    let g = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionEventRollout);
    assert_eq!(g.outputs().len(), world::LOSS_IMAGE_FEEDBACK + 4);
    let parameters = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 80);
    assert_eq!(
        parameters
            .iter()
            .map(|n| n.ty.num_elements())
            .sum::<usize>(),
        8_871_955
    );
    let inputs = g
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
        inputs.len(),
        inputs
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    assert!(
        inputs
            .iter()
            .all(|name| !name.starts_with("pixel_feedback_uniforms_"))
    );
    assert_eq!(
        inputs
            .iter()
            .filter(|name| name.starts_with("pixel_rollout_weight_"))
            .count(),
        16
    );
    assert_eq!(
        inputs
            .iter()
            .filter(|name| name.starts_with("reward_target_"))
            .count(),
        32
    );
    assert_eq!(
        inputs
            .iter()
            .filter(|name| name.starts_with("continuation_target_"))
            .count(),
        32
    );
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.reward.",
        "world.continuation.",
        "world.dynamics.core.",
        "world.dynamics.prior",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(name, buffer)| name.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
        );
    }
}
