use super::*;
use crate::dreamer::{ColorCentroidSupervision, ModelSize, ObservationKind};
use meganeura::graph::Op;
use serde_json::json;
use std::collections::{BTreeMap, HashSet};

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.world_microbatch_size = Some(2);
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.color_centroid_supervision = Some(ColorCentroidSupervision {
        scale: 1000.,
        channel: 0,
        background_scale: 1.5,
        region: [0.0625, 0.4375, 0.9375, 0.875],
    });
    c
}
fn parameters(g: &Graph) -> BTreeMap<String, Vec<usize>> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
            _ => None,
        })
        .collect()
}
#[test]
fn centroid_headers_and_constructors_reject_incompatible_paths_before_gpu() {
    let c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    let m = json!({"format":"kindle-world-only-v22","config":c,"pixel_loss":"motion_cross_entropy","prediction_length":2,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 2, 0).unwrap(),
        TrainingScope::All
    );
    for v in 1..22 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{v}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for value in [json!(0), json!(-1), json!(0.5), json!(null)] {
        let mut bad = m.clone();
        bad["prediction_length"] = value;
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["config"]["color_centroid_supervision"] = json!(null);
    assert!(checkpoint_pixel_loss(&bad).is_err());
    for field in [
        "world_gaussian",
        "rgb_spatial_decoder",
        "rgb_previous_frame",
        "rgb_latent_only",
    ] {
        let mut bad = m.clone();
        bad["config"][field] = json!(true);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for objective in [
        PixelLoss::SquaredError,
        PixelLoss::CrossEntropy,
        PixelLoss::MotionEventRollout,
    ] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), 2, objective).is_err());
    }
    assert!(WorldModelLearner::with_pixel_loss(c, 0, PixelLoss::MotionCrossEntropy).is_err());
    let mut old = m;
    old["format"] = json!("kindle-world-only-v2");
    old["config"]
        .as_object_mut()
        .unwrap()
        .remove("color_centroid_supervision");
    assert_eq!(
        checkpoint_pixel_loss(&old).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
}
#[test]
fn centroid_adds_one_shared_linear_head_and_preserves_inference_graphs() {
    let mut c = config();
    c.model_size = ModelSize::Size12M;
    c.observation_kind = ObservationKind::Rgb64;
    c.value_bins = 255;
    let mut plain = c.clone();
    plain.color_centroid_supervision = None;
    let g = world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionCrossEntropy);
    let control =
        world::build_world_only_pixel_training_graph(&plain, 2, PixelLoss::MotionCrossEntropy);
    let all = parameters(&g);
    let shared = all
        .iter()
        .filter(|(name, _)| !name.starts_with("world.color_centroid."))
        .map(|(a, b)| (a.clone(), b.clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(shared, parameters(&control));
    assert_eq!(all.len(), 70);
    assert_eq!(
        all.values()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>(),
        8_857_797
    );
    assert_eq!(all["world.color_centroid.out.weight"], [2560, 2]);
    assert_eq!(g.outputs().len(), 41);
    for build in [
        world::build_observe_graph,
        world::build_transition_graph,
        world::build_observation_prediction_graph,
        world::build_head_graph,
    ] {
        assert_eq!(
            format!("{:?}", build(&c, 2).nodes()),
            format!("{:?}", build(&plain, 2).nodes())
        );
    }
    let names = g
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
    assert_eq!(names.len(), names.iter().collect::<HashSet<_>>().len());
    meganeura::compile_training_graph(&g);
}
#[test]
fn centroid_future_loss_reaches_encoder_prior_core_and_head_without_future_state_inputs() {
    let c = config();
    let mut g = world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionCrossEntropy);
    let future = g.outputs()[world::LOSS_WORLD_CENTROID];
    g.set_outputs(vec![future]);
    let backward = meganeura::autodiff::differentiate(&g);
    let mut connected = Vec::new();
    for (n, &grad) in g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .zip(&backward.outputs()[1..])
    {
        if backward.node(grad).ty == n.ty {
            let Op::Parameter { name } = &n.op else {
                unreachable!()
            };
            connected.push(name.as_str());
        }
    }
    for prefix in [
        "world.representation.encoder.",
        "world.representation.posterior.",
        "world.dynamics.core.",
        "world.dynamics.prior0.",
        "world.dynamics.prior1.",
        "world.dynamics.priorlogit.",
        "world.color_centroid.",
    ] {
        assert!(connected.iter().any(|n| n.starts_with(prefix)), "{prefix}");
    }
    assert!(
        connected
            .iter()
            .all(|n| !n.starts_with("behavior.") && !n.starts_with("world.decoder."))
    );
    // The only learned head reads causal prior features; future targets and
    // reset weights enter detached target/mask branches after state creation.
    assert!(g.nodes().iter().any(|n| matches!(n.op, Op::StopGradient)));
}
