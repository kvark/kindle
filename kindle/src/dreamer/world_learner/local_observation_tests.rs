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
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(8);
    c.value_bins = 255;
    c.loss_scales.policy = 0.0;
    c.loss_scales.value = 0.0;
    c.loss_scales.replay_value = 0.0;
    c.disagreement_bonus = false;
    c
}

#[test]
fn local_observation_headers_require_explicit_h0_v20_without_changing_v18() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    let m = json!({"format":"kindle-world-only-v20","config":c,
        "pixel_loss":"motion_cross_entropy","prediction_length":0,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 0, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..20 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err(), "v{version}");
    }
    for objective in [
        "motion_event_rollout",
        "motion_clean_prior",
        "squared_error",
        "cross_entropy",
    ] {
        let mut bad = m.clone();
        bad["pixel_loss"] = json!(objective);
        assert!(checkpoint_pixel_loss(&bad).is_err(), "{objective}");
    }
    for length in [json!(16), json!(-1), json!(0.5), json!(null)] {
        let mut bad = m.clone();
        bad["prediction_length"] = length;
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad.as_object_mut().unwrap().remove("prediction_length");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    for field in [
        "rgb_spatial_action",
        "rgb_spatial_fusion",
        "rgb_spatial_decoder",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err(), "{field}");
    }
    for field in [
        "rgb_spatial_transport",
        "rgb_spatial_residual",
        "world_image_feedback",
        "world_gaussian",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err(), "{field}");
    }
    let mut bad = m.clone();
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionCrossEntropy, 0, 0).is_err());
    let mut old = m;
    old["format"] = json!("kindle-world-only-v18");
    old["pixel_loss"] = json!("motion_event_rollout");
    old["prediction_length"] = json!(16);
    assert_eq!(
        checkpoint_pixel_loss(&old).unwrap(),
        PixelLoss::MotionEventRollout
    );
}

#[test]
fn local_observation_graph_keeps_parameters_and_has_no_sampled_future_losses() {
    let c = config();
    let graph = world::build_world_only_pixel_training_graph(&c, 0, PixelLoss::MotionCrossEntropy);
    let prior = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionEventRollout);
    let parameters = |graph: &Graph| -> BTreeMap<String, Vec<usize>> {
        graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                _ => None,
            })
            .collect()
    };
    let actual = parameters(&graph);
    assert_eq!(actual, parameters(&prior));
    assert_eq!(actual.len(), 80);
    assert_eq!(
        actual
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum::<usize>(),
        603_915
    );
    assert_eq!(graph.outputs().len(), 11);
    assert!(
        graph.outputs()[..9]
            .iter()
            .all(|&output| graph.node(output).ty.num_elements() == 1)
    );
    assert_eq!(
        graph.node(graph.outputs()[9]).ty.num_elements(),
        32 * 8 * c.network().deter
    );
    assert_eq!(
        graph.node(graph.outputs()[10]).ty.num_elements(),
        32 * 8 * c.network().stoch * c.network().classes
    );
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            assert!(!name.starts_with("pixel_rollout_") && !name.starts_with("pixel_feedback_"));
        }
    }
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert_eq!(plan.param_buffers.len(), 80);
    assert!(plan.param_buffers.iter().all(|(_, buffer)| {
        plan.param_grad_pairs
            .iter()
            .any(|(parameter, _)| parameter == buffer)
    }));
    let inference = world::build_observation_prediction_graph(&c, 8);
    assert!(inference.nodes().iter().any(
        |node| matches!(&node.op, Op::Input { name } if name == "decoder_action")
            && node.ty.shape == [8, 18]
    ));
}
