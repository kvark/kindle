use super::*;
use crate::dreamer::{ModelSize, ObservationKind, VectorDreamerAgent};
use meganeura::graph::Op;
use serde_json::json;
fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.world_image_feedback = true;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c
}
#[test]
fn image_feedback_requires_explicit_spatial_architecture_objective_and_suffix() {
    let c = config();
    assert!(c.check().is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    assert!(
        serde_json::to_value(DreamerConfig::tiny(18))
            .unwrap()
            .get("world_image_feedback")
            .is_none()
    );
    let mut bad = c.clone();
    bad.rgb_spatial_decoder = false;
    assert!(bad.check().is_err());
    assert!(
        WorldModelLearner::with_pixel_loss(c.clone(), 2, PixelLoss::MotionCrossEntropy).is_err()
    );
    assert!(
        WorldModelLearner::with_pixel_loss(c.clone(), 0, PixelLoss::MotionImageFeedback).is_err()
    );
    let mut bad = c.clone();
    bad.world_image_feedback = false;
    assert!(WorldModelLearner::with_pixel_loss(bad, 2, PixelLoss::MotionImageFeedback).is_err());
    assert!(VectorDreamerAgent::learned_rgb(c, 8).is_err());
}
#[test]
fn image_feedback_headers_bind_v11_architecture_objective_and_scope() {
    let m = json!({"format":"kindle-world-only-v11","config":config(),"pixel_loss":"motion_image_feedback","prediction_length":2,"updates":0});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionImageFeedback
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionImageFeedback, 2, 0).unwrap(),
        TrainingScope::All
    );
    for v in 1..11 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{v}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    for field in ["world_image_feedback", "rgb_spatial_decoder"] {
        let mut bad = m.clone();
        bad["config"][field] = json!(false);
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["pixel_loss"] = json!("motion_cross_entropy");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m;
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, PixelLoss::MotionImageFeedback, 2, 0).is_err());
}
#[test]
fn image_feedback_adds_no_parameters_and_has_unique_shared_target_names() {
    let mut c = config();
    c.model_size = ModelSize::Size12M;
    c.value_bins = 255;
    c.observation_kind = ObservationKind::Rgb64;
    c.batch_size = 8;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    let graph =
        world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionImageFeedback);
    let parameters = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 77);
    assert_eq!(
        parameters
            .iter()
            .map(|n| n.ty.shape.iter().product::<usize>())
            .sum::<usize>(),
        8857723
    );
    assert_eq!(graph.outputs().len(), world::LOSS_IMAGE_FEEDBACK + 4);
    let mut seen = std::collections::HashSet::new();
    let mut pending = vec![graph.outputs()[0]];
    while let Some(n) = pending.pop() {
        if seen.insert(n) {
            pending.extend(graph.node(n).inputs.iter().copied());
        }
    }
    assert!(seen.contains(&graph.outputs()[world::LOSS_IMAGE_FEEDBACK]));
    assert!(seen.contains(&graph.outputs()[world::LOSS_IMAGE_FEEDBACK + 1]));
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
    for t in 0..32 {
        assert_eq!(
            names
                .iter()
                .filter(|n| n.as_str() == format!("reward_target_{t}"))
                .count(),
            1
        );
        assert_eq!(
            names
                .iter()
                .filter(|n| n.as_str() == format!("continuation_target_{t}"))
                .count(),
            1
        );
    }
    let f = world::build_image_feedback_graph(&c, 8);
    assert!(
        f.nodes()
            .iter()
            .all(|n| !matches!(&n.op,Op::Parameter{name} if name.starts_with("world.dynamics.")))
    );
    assert_eq!(
        f.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .count(),
        2
    );
}
#[test]
#[ignore = "requires separately declared guarded generated-image belief mechanism check"]
fn image_feedback_native_generated_image_controls_belief_at_fixed_h() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let c = config();
    let graph = world::build_image_feedback_graph(&c, 2);
    let mut s = runtime::build_session(&graph, &gpu, Mode::Inference, false);
    runtime::initialize_d3(&mut s, &graph, c.seed);
    s.set_input("deter", &vec![0.; 2 * c.network().deter]);
    let width = c.observation_dim();
    s.set_input("generated_image", &vec![0.; 2 * width]);
    s.step();
    s.wait();
    let mut a = vec![0.; 2 * c.network().stoch * c.network().classes];
    s.read_output_by_index(0, &mut a);
    let mut image = vec![0.; 2 * width];
    image[16 * 32 + 16] = 0.5;
    s.set_input("generated_image", &image);
    s.step();
    s.wait();
    let mut b = vec![0.; a.len()];
    s.read_output_by_index(0, &mut b);
    let lane = c.network().stoch * c.network().classes;
    assert!(
        a[..lane]
            .iter()
            .zip(&b[..lane])
            .any(|(a, b)| (a - b).abs() > 1e-6)
    );
    assert_eq!(a[lane..], b[lane..]);
    assert!(b.iter().all(|v| v.is_finite()));
    assert_eq!(s.adam_step_count(), 0);
}
