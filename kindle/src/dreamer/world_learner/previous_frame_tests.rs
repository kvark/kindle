use super::*;
use crate::dreamer::ObservationKind;
use meganeura::{
    graph::Op,
    reference::{Feeds, evaluate},
};
use serde_json::json;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_previous_frame = true;
    c.actor_critic_gradient = false;
    c.loss_scales.reconstruction = 100.;
    c.loss_scales.future_prediction = 0.;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn previous_frame_is_optional_world_only_and_preserves_public_rgb_geometry() {
    let mut c = config();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    assert_eq!(c.observation_channels(), 3);
    assert_eq!(c.observation_dim(), 3 * 32 * 32);
    let encoded = serde_json::to_value(&c).unwrap();
    assert_eq!(encoded["rgb_previous_frame"], true);
    assert_eq!(serde_json::from_value::<DreamerConfig>(encoded).unwrap(), c);
    c.rgb_previous_frame = false;
    assert!(
        serde_json::to_value(&c)
            .unwrap()
            .get("rgb_previous_frame")
            .is_none()
    );
    for defect in 0..10 {
        let mut bad = config();
        match defect {
            0 => bad.observation_kind = ObservationKind::Features,
            1 => bad.rgb_frame_pair = true,
            2 => bad.rgb_latent_only = true,
            3 => bad.loss_scales.policy = 1.,
            4 => bad.loss_scales.value = 1.,
            5 => bad.loss_scales.replay_value = 1.,
            6 => bad.actor_critic_gradient = true,
            7 => bad.video_encoder = Some(crate::dreamer::VideoEncoder::Frozen),
            8 => bad.loss_scales.future_prediction = 1.,
            9 => {
                bad.observation_crop = Some(crate::vision::preprocess_gpu::FrameCrop {
                    x: 0,
                    y: 0,
                    width: 160,
                    height: 160,
                })
            }
            _ => unreachable!(),
        }
        assert!(bad.check().unwrap_err().contains("previous-frame RGB"));
    }
}

#[test]
fn previous_images_preserve_lane_order_and_duplicate_only_starts_and_resets() {
    let images = (0..12).map(|i| i as f32).collect::<Vec<_>>();
    assert_eq!(previous_images(&images, &[-1, -1], 0, 2), [0., 1., 2., 3.]);
    assert_eq!(previous_images(&images, &[0, -1], 1, 2), [0., 1., 6., 7.]);
    assert_eq!(previous_images(&images, &[-1, 3], 2, 2), [8., 9., 6., 7.]);
    // A normal incoming action keeps history even on a terminal arrival.
    assert_eq!(previous_images(&images, &[0, 3], 1, 2), [0., 1., 2., 3.]);
    assert_eq!(
        previous_images(&images[..8], &[0, -1], 1, 2),
        previous_images(&images, &[0, -1], 1, 2)
    );
}

#[test]
fn previous_frame_headers_require_v5_and_all_weight_scope() {
    let mut m = json!({"format":"kindle-world-only-v5","config":{"rgb_previous_frame":true},"pixel_loss":"motion_cross_entropy"});
    assert_eq!(
        checkpoint_pixel_loss(&m).unwrap(),
        PixelLoss::MotionCrossEntropy
    );
    assert_eq!(
        TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 0, 0).unwrap(),
        TrainingScope::All
    );
    for version in [
        "kindle-world-only-v1",
        "kindle-world-only-v2",
        "kindle-world-only-v3",
        "kindle-world-only-v4",
    ] {
        m["format"] = json!(version);
        assert!(checkpoint_pixel_loss(&m).is_err());
    }
    m["format"] = json!("kindle-world-only-v5");
    for flag in [json!(false), json!(null)] {
        m["config"]["rgb_previous_frame"] = flag;
        assert!(checkpoint_pixel_loss(&m).is_err());
    }
    m["config"]["rgb_previous_frame"] = json!(true);
    m["config"]["rgb_latent_only"] = json!(true);
    assert!(checkpoint_pixel_loss(&m).is_err());
    m["config"]["rgb_latent_only"] = json!(false);
    m["training_scope"] = json!("frozen_visual");
    assert!(TrainingScope::from_metadata(&m, PixelLoss::MotionCrossEntropy, 0, 0).is_err());
}

fn input_names(g: &Graph, output: u32) -> std::collections::BTreeSet<String> {
    let mut seen = std::collections::HashSet::new();
    let mut pending = vec![output];
    let mut names = std::collections::BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Op::Input { name } = &g.node(id).op {
            names.insert(name.clone());
        }
        pending.extend(g.node(id).inputs.iter().copied());
    }
    names
}

#[test]
fn previous_frame_encoder_ancestry_is_two_images_and_prior_transition_has_no_image() {
    let c = config();
    let g = world::build_observe_graph(&c, 2);
    assert_eq!(
        input_names(&g, g.outputs()[3]),
        ["observation".to_owned(), "previous_observation".to_owned()].into()
    );
    let prior = world::build_transition_graph(&c, 2);
    assert_eq!(
        input_names(&prior, prior.outputs()[1]),
        ["deter".to_owned(), "stoch".to_owned(), "action".to_owned()].into()
    );
    let mut legacy = c.clone();
    legacy.rgb_previous_frame = false;
    let g = world::build_observe_graph(&legacy, 2);
    assert_eq!(
        input_names(&g, g.outputs()[3]),
        ["observation".to_owned()].into()
    );
}

#[test]
fn previous_frame_only_widens_first_convolution_and_keeps_world_gradients() {
    let mut c = config();
    c.batch_size = 2;
    c.batch_length = 3;
    c.world_backprop_length = 3;
    let g = world::build_world_only_pixel_training_graph(&c, 0, PixelLoss::MotionCrossEntropy);
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.dynamics.",
        "world.representation.encoder.",
        "world.representation.posterior.",
        "world.decoder.",
        "world.reward.",
        "world.continuation.",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(name, buffer)| name.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(p, _)| p == buffer)),
            "{prefix}"
        );
    }
    let shapes = |c: &DreamerConfig| {
        world::build_world_only_pixel_training_graph(c, 0, PixelLoss::MotionCrossEntropy)
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let mut large = DreamerConfig::new(18);
    large.observation_kind = ObservationKind::Rgb64;
    large.rgb_previous_frame = true;
    large.actor_critic_gradient = false;
    large.loss_scales.policy = 0.;
    large.loss_scales.value = 0.;
    large.loss_scales.replay_value = 0.;
    large.loss_scales.reconstruction = 100.;
    large.loss_scales.future_prediction = 0.;
    large.disagreement_bonus = false;
    large.batch_size = 8;
    large.batch_length = 32;
    large.world_backprop_length = 32;
    let current = shapes(&large);
    large.rgb_previous_frame = false;
    let old = shapes(&large);
    assert_eq!(current.len(), 68);
    assert_eq!(
        current.keys().collect::<Vec<_>>(),
        old.keys().collect::<Vec<_>>()
    );
    let changed = current
        .iter()
        .filter(|(k, v)| old[*k] != **v)
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>();
    assert_eq!(changed, ["world.representation.encoder.cnn0.weight"]);
    assert_eq!(current[changed[0]], [32 * 6 * 5 * 5]);
    assert_eq!(old[changed[0]], [32 * 3 * 5 * 5]);
    assert_eq!(
        current
            .values()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>(),
        8_855_075
    );
}

#[test]
fn previous_frame_concat_matches_chw_rows_and_mean_derivative() {
    let c = config();
    let b = 2;
    let width = c.observation_dim();
    let mut g = Graph::new();
    let current = g.parameter("current", &[b, width]);
    let pair = world::encoder_observation(&mut g, &c, current, "previous", b);
    let a = (0..b * width)
        .map(|i| (i % 17) as f32 / 32. - 0.25)
        .collect::<Vec<_>>();
    let p = (0..b * width)
        .map(|i| (i % 23) as f32 / 64. - 0.25)
        .collect::<Vec<_>>();
    let mut feeds = Feeds::new();
    feeds.set("current", &a).set("previous", &p);
    g.set_outputs(vec![pair]);
    let values = evaluate(&g, &feeds).unwrap();
    let expected = p
        .chunks_exact(width)
        .zip(a.chunks_exact(width))
        .flat_map(|(p, a)| p.iter().chain(a).copied().map(f64::from))
        .collect::<Vec<_>>();
    assert_eq!(values[pair as usize].data, expected);
    let loss = g.mean_all(pair);
    g.set_outputs(vec![loss]);
    let backward = meganeura::autodiff::differentiate(&g);
    let values = evaluate(&backward, &feeds).unwrap();
    let expected_loss = expected.iter().sum::<f64>() / expected.len() as f64;
    assert!((values[backward.outputs()[0] as usize].data[0] - expected_loss).abs() < 1e-9);
    for &v in &values[backward.outputs()[1] as usize].data {
        assert!((v - 1. / expected.len() as f64).abs() < 1e-10);
    }
    let mut legacy = c;
    legacy.rgb_previous_frame = false;
    assert_eq!(
        world::encoder_observation(&mut g, &legacy, current, "unused", b),
        current
    );
}
