use super::*;
use crate::dreamer::{ModelSize, ObservationKind, VectorDreamerAgent};
use meganeura::graph::Op;
use serde_json::json;
use std::collections::HashSet;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(c.batch_size);
    c.loss_scales.policy = 0.0;
    c.loss_scales.value = 0.0;
    c.loss_scales.replay_value = 0.0;
    c.disagreement_bonus = false;
    c
}

#[test]
fn local_action_headers_and_constructors_enforce_explicit_v18_scope_before_gpu() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    let m = json!({"format":"kindle-world-only-v18","config":c,"pixel_loss":"motion_event_rollout","prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionEventRollout
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionEventRollout, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..18 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in [
        "rgb_spatial_action",
        "rgb_spatial_fusion",
        "rgb_spatial_decoder",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in [
        "rgb_spatial_residual",
        "rgb_spatial_transport",
        "world_image_feedback",
        "world_gaussian",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["prediction_length"] = json!(0);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    for objective in [
        PixelLoss::MotionCrossEntropy,
        PixelLoss::MotionCleanPrior,
        PixelLoss::MotionFixedTeacherEvents,
    ] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), 16, objective).is_err());
    }
    assert!(
        WorldModelLearner::with_pixel_loss(c.clone(), 0, PixelLoss::MotionEventRollout).is_err()
    );
    assert!(VectorDreamerAgent::learned_rgb(c, 8).is_err());
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_spatial_action")
            .is_none()
    );
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionEventRollout, 16, 0).is_err());
}

#[test]
fn local_action_preserves_causal_inputs_and_adds_only_21248_weights_at_12m() {
    let mut c = config();
    c.model_size = ModelSize::Size12M;
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
        8_893_203
    );
    let inputs = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some(name),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(inputs.len(), inputs.iter().collect::<HashSet<_>>().len());
    let mut original = c.clone();
    original.rgb_spatial_action = false;
    let baseline =
        world::build_world_only_pixel_training_graph(&original, 16, PixelLoss::MotionEventRollout);
    let common = parameters
        .iter()
        .filter(|node| {
            baseline.nodes().iter().any(|other| {
                matches!((&other.op,&node.op),(Op::Parameter{name:a},Op::Parameter{name:b}) if a==b)
                    && other.ty.shape == node.ty.shape
            })
        })
        .count();
    assert_eq!(common, 70);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    let local = plan
        .param_buffers
        .iter()
        .filter(|(n, _)| {
            n.starts_with("world.decoder.motion") || n.starts_with("world.decoder.fusion")
        })
        .collect::<Vec<_>>();
    assert_eq!(local.len(), 12);
    assert!(
        local
            .iter()
            .all(|(_, buffer)| plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
    );
    let inference = world::build_observation_prediction_graph(&c, 8);
    assert!(inference.nodes().iter().any(
        |n| matches!(&n.op,Op::Input{name} if name=="decoder_action") && n.ty.shape == [8, 18]
    ));
}
