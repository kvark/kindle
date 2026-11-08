use super::*;
use crate::dreamer::ObservationKind;
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
fn clean_prior_headers_require_v14_fusion_positive_suffix_and_no_phase() {
    let m = json!({"format":"kindle-world-only-v14","config":config(),"pixel_loss":"motion_clean_prior","prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCleanPrior
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCleanPrior, 16, 0).unwrap(),
        TrainingScope::All
    );
    for v in 1..14 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{v}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for key in ["rgb_spatial_decoder", "rgb_spatial_fusion"] {
        let mut bad = m.clone();
        bad["config"][key] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for key in [
        "world_gaussian",
        "world_image_feedback",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][key] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for loss in [
        "motion_cross_entropy",
        "motion_event_rollout",
        "motion_image_feedback",
    ] {
        let mut bad = m.clone();
        bad["pixel_loss"] = json!(loss);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["prediction_length"] = json!(0);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionCleanPrior, 16, 0).is_err());
    assert!(WorldModelLearner::with_pixel_loss(config(), 0, PixelLoss::MotionCleanPrior).is_err());
}
