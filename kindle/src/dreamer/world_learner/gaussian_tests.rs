use super::*;
use crate::dreamer::{ObservationKind, networks};
use meganeura::graph::Op;
use serde_json::json;
fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.world_gaussian = true;
    c.batch_size = 2;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.0;
    c.loss_scales.value = 0.0;
    c.loss_scales.replay_value = 0.0;
    c.loss_scales.reconstruction = 100.0;
    c
}
#[test]
fn gaussian_headers_reject_old_versions_and_phase_scope_before_gpu() {
    let c = config();
    let m = json!({"format":"kindle-world-only-v8","config":c,"pixel_loss":"motion_cross_entropy","prediction_length":16,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 16, 0).unwrap(),
        TrainingScope::All
    );
    for version in 1..8 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for (key, value) in [
        ("world_gaussian", json!(false)),
        ("rgb_previous_frame", json!(true)),
        ("rgb_latent_only", json!(true)),
    ] {
        let mut bad = m.clone();
        bad["config"][key] = value;
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
fn gaussian_graph_preserves_state_width_and_has_unique_inputs_without_argmax() {
    let c = config();
    let g = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
    let names = g
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
    assert!(!g.nodes().iter().any(|n| matches!(n.op, Op::ArgMaxOneHot)));
    assert_eq!(g.outputs().len(), world::LOSS_WORLD_ROLLOUT + 3);
    let size = c.network();
    let mut graph = Graph::new();
    let prior = networks::Prior::new(&mut graph, &c);
    let h = graph.input("h", &[2, size.deter]);
    let p = prior.forward(&mut graph, h, 2);
    assert_eq!(graph.node(p).ty.shape, [2 * size.stoch, 2 * size.classes]);
    let (plan, _) = meganeura::compile_training_graph(&g);
    assert_eq!(plan.param_buffers.len(), 68);
    let mut c = c;
    c.world_gaussian = false;
    let old = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
    let params = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let (a, b) = (params(&g), params(&old));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    let changed = a
        .iter()
        .filter(|(k, v)| b[*k] != **v)
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>();
    assert_eq!(changed.len(), 4);
    assert!(
        changed
            .iter()
            .all(|n| n.starts_with("world.dynamics.priorlogit.")
                || n.starts_with("world.representation.posterior.obslogit."))
    );
}
#[test]
fn gaussian_full_shape_geometry_is_explicit() {
    let mut c = config();
    c.model_size = crate::dreamer::ModelSize::Size12M;
    c.value_bins = 255;
    c.observation_kind = ObservationKind::Rgb64;
    c.batch_size = 8;
    let g = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
    let weights = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .collect::<Vec<_>>();
    assert_eq!(weights.len(), 68);
    assert_eq!(
        weights
            .iter()
            .map(|n| n.ty.shape.iter().product::<usize>())
            .sum::<usize>(),
        9_115_843
    );
    assert_eq!(c.feature_dim(), 2560);
    assert_eq!(c.encoded_observation_dim(), 1024);
}
