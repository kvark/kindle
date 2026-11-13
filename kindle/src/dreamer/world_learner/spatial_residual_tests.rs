use super::*;
use crate::dreamer::ObservationKind;
use meganeura::graph::Op;
use serde_json::json;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_residual = true;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(c.batch_size);
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn spatial_residual_headers_bind_v15_flag_objective_geometry_and_scope() {
    let m = json!({"format":"kindle-world-only-v15","config":config(),"pixel_loss":"motion_event_rollout","prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionEventRollout
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionEventRollout, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..15 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in [
        "rgb_spatial_residual",
        "rgb_spatial_decoder",
        "rgb_spatial_fusion",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for objective in [
        "motion_clean_prior",
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
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_spatial_residual")
            .is_none()
    );
}

#[test]
fn spatial_residual_rejects_incompatible_constructor_before_gpu() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    let mut bad = c.clone();
    bad.rgb_spatial_fusion = false;
    assert!(bad.check().is_err());
    for objective in [PixelLoss::MotionCrossEntropy, PixelLoss::MotionCleanPrior] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), 16, objective).is_err());
    }
    assert!(WorldModelLearner::with_pixel_loss(c, 0, PixelLoss::MotionEventRollout).is_err());
}

#[test]
fn spatial_residual_training_preserves_size_and_all_parameter_gradients() {
    let mut c = config();
    c.model_size = super::super::ModelSize::Size12M;
    c.observation_kind = ObservationKind::Rgb64;
    c.batch_size = 8;
    c.world_microbatch_size = Some(8);
    c.value_bins = 255;
    let graph = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionEventRollout);
    let parameters = graph
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
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert_eq!(plan.param_buffers.len(), 80);
    assert_eq!(plan.param_grad_pairs.len(), 80);
    let names = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some(name),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len()
    );
}
