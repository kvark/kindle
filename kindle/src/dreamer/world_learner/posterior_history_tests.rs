use super::*;
use crate::dreamer::{ModelSize, ObservationKind};
use meganeura::graph::Op;
use serde_json::json;
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.model_size = ModelSize::Size1M;
    c.observation_kind = ObservationKind::Rgb64;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.value_bins = 255;
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(8);
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c
}

#[test]
fn posterior_history_requires_its_own_header_and_positive_suffix() {
    let c = config();
    let m = json!({"format":"kindle-world-only-v21","config":c,"updates":0,"prediction_length":16,"pixel_loss":"motion_posterior_history"});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionPosteriorHistory
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionPosteriorHistory, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..21 {
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
        "rgb_spatial_transport",
        "rgb_spatial_residual",
        "world_gaussian",
        "world_image_feedback",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for length in [json!(0), json!(-1), json!(null), json!(0.5)] {
        let mut bad = m.clone();
        bad["prediction_length"] = length;
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["pixel_loss"] = json!("motion_event_rollout");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    assert!(
        WorldModelLearner::with_pixel_loss(c.clone(), 0, PixelLoss::MotionPosteriorHistory)
            .is_err()
    );
    let mut bad = c;
    bad.rgb_spatial_action = false;
    assert!(
        WorldModelLearner::with_pixel_loss(bad, 16, PixelLoss::MotionPosteriorHistory).is_err()
    );
}

#[test]
fn posterior_history_reuses_geometry_and_posterior_noise_without_prior_rollout() {
    let mut c = config();
    c.batch_size = 1;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.world_microbatch_size = Some(1);
    let g = world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionPosteriorHistory);
    let old = world::build_world_only_pixel_training_graph(&c, 0, PixelLoss::MotionCrossEntropy);
    let parameters = |g: &Graph| -> BTreeMap<String, Vec<usize>> {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect()
    };
    let p = parameters(&g);
    assert_eq!(p, parameters(&old));
    assert_eq!(p.len(), 80);
    assert_eq!(
        p.values()
            .map(|v| v.iter().product::<usize>())
            .sum::<usize>(),
        603915
    );
    assert_eq!(g.outputs().len(), world::LOSS_WORLD_ROLLOUT + 3);
    assert!(g.nodes().iter().all(|n|!matches!(&n.op,Op::Input{name} if name.starts_with("pixel_rollout_uniforms_")||name.starts_with("pixel_feedback_"))));
    let (plan, _) = meganeura::compile_training_graph(&g);
    assert_eq!(plan.param_buffers.len(), 80);
    assert!(
        plan.param_buffers
            .iter()
            .all(|(_, b)| plan.param_grad_pairs.iter().any(|(p, _)| p == b))
    );
}
