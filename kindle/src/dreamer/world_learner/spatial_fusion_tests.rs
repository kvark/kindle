use super::*;
use crate::dreamer::{ModelSize, ObservationKind, VectorDreamerAgent};
use meganeura::graph::Op;
use serde_json::json;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn spatial_fusion_is_optional_and_requires_the_selected_world_architecture() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("rgb_spatial_fusion")
            .is_none()
    );
    let mut bad = c.clone();
    bad.rgb_spatial_decoder = false;
    assert!(bad.check().is_err());
    assert!(std::panic::catch_unwind(|| crate::dreamer::DreamerCore::new(bad)).is_err());
    let mut bad = c.clone();
    bad.world_image_feedback = true;
    assert!(bad.check().is_err());
    assert!(VectorDreamerAgent::learned_rgb(c, 8).is_err());
}

#[test]
fn spatial_fusion_headers_bind_v12_architecture_objective_and_scope() {
    let m = json!({"format":"kindle-world-only-v12","config":config(),"pixel_loss":"motion_cross_entropy","prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..12 {
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
        "world_gaussian",
        "rgb_latent_only",
        "rgb_previous_frame",
        "world_image_feedback",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["pixel_loss"] = json!("squared_error");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionCrossEntropy, 16, 0).is_err());
}

#[test]
fn spatial_fusion_keeps_world_geometry_and_all_new_weights_receive_gradients() {
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
    assert_eq!(parameters.len(), 80);
    assert_eq!(
        parameters
            .iter()
            .map(|n| n.ty.num_elements())
            .sum::<usize>(),
        8_871_955
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
    let (plan, _) = meganeura::compile_training_graph(&graph);
    let fusion = plan
        .param_buffers
        .iter()
        .filter(|(name, _)| name.starts_with("world.decoder.fusion."))
        .collect::<Vec<_>>();
    assert_eq!(fusion.len(), 3);
    assert!(
        fusion
            .iter()
            .all(|(_, buffer)| plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
    );
}
