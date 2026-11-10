use super::*;
use crate::dreamer::spatial_history::{context, context_cpu, fixed, fixed_cpu};
use crate::dreamer::{DreamerCore, ObservationKind};
use meganeura::graph::{NodeId, Op};
use meganeura::reference::{Feeds, evaluate};

fn spatial() -> DreamerConfig {
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

fn with_context(lag: usize) -> DreamerConfig {
    let mut c = spatial();
    c.rgb_context_lag = Some(lag);
    c
}

#[test]
fn context_lag_requires_the_spatial_decoder_and_excludes_feedback_variants() {
    let c = with_context(8);
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    assert!(
        serde_json::to_value(spatial())
            .unwrap()
            .get("rgb_context_lag")
            .is_none()
    );
    let mut bad = c.clone();
    bad.rgb_spatial_decoder = false;
    bad.rgb_spatial_fusion = false;
    assert!(bad.check().is_err());
    for lag in [0, 33] {
        let mut bad = c.clone();
        bad.rgb_context_lag = Some(lag);
        assert!(bad.check().is_err());
    }
    let mut bad = c.clone();
    bad.rgb_history_palette = Some(vec![[0.0; 3], [0.5; 3]]);
    assert!(bad.check().is_err());
    assert!(std::panic::catch_unwind(|| DreamerCore::new(c)).is_err());
}

#[test]
fn cpu_context_clamps_to_the_window_start_and_masks_resets() {
    // batch 2, width 2, four arrivals; lane 1 resets at arrival 2.
    let images: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let actions = [-1, -1, 0, 0, 0, -1, 0, 0];
    assert_eq!(context_cpu(&images, &actions, 0, 8, 2, 2), vec![0.0; 8]);
    // time 3, lag 8 -> clamped to arrival 0 for lane 0; lane 1 crosses its reset.
    assert_eq!(
        context_cpu(&images, &actions, 3, 8, 2, 2),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
    // time 3, lag 1 -> arrival 2 for both lanes (lane 1's reset is at arrival 2, which is its own frame).
    assert_eq!(
        context_cpu(&images, &actions, 3, 1, 2, 2),
        vec![8.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 11.0]
    );
    assert_eq!(
        fixed_cpu(&images, 3, 2, 2),
        vec![12.0, 13.0, 12.0, 13.0, 14.0, 15.0, 14.0, 15.0]
    );
}

#[test]
fn graph_context_matches_the_cpu_context() {
    let (batch, width, length, lag) = (2, 3, 5, 3);
    let mut g = Graph::new();
    let images: Vec<NodeId> = (0..length)
        .map(|t| g.parameter(&format!("image_{t}"), &[batch, width]))
        .collect();
    let keeps: Vec<NodeId> = (0..length)
        .map(|t| g.parameter(&format!("keep_{t}"), &[batch, 1]))
        .collect();
    let outputs: Vec<NodeId> = (0..length)
        .map(|t| context(&mut g, &images, &keeps, t, lag, batch, width))
        .collect();
    let last = fixed(&mut g, images[length - 1], batch, width);
    g.set_outputs(outputs.iter().copied().chain([last]).collect());
    let actions = [-1, -1, 0, 0, 0, -1, 0, 0, 0, 0];
    let pixels: Vec<f32> = (0..length * batch * width)
        .map(|i| i as f32 / 10.0)
        .collect();
    let mut feeds = Feeds::new();
    for t in 0..length {
        feeds.set(
            &format!("image_{t}"),
            &pixels[t * batch * width..(t + 1) * batch * width],
        );
        let keep: Vec<f32> = (0..batch)
            .map(|row| f32::from(actions[t * batch + row] >= 0))
            .collect();
        feeds.set(&format!("keep_{t}"), &keep);
    }
    let values = evaluate(&g, &feeds).unwrap();
    for t in 0..length {
        let expected = context_cpu(&pixels, &actions, t, lag, batch, width);
        let actual: Vec<f32> = values[outputs[t] as usize]
            .data
            .iter()
            .map(|&v| v as f32)
            .collect();
        assert_eq!(actual, expected, "time {t}");
    }
    let actual: Vec<f32> = values[last as usize]
        .data
        .iter()
        .map(|&v| v as f32)
        .collect();
    assert_eq!(actual, fixed_cpu(&pixels, length - 1, batch, width));
}

#[test]
fn context_graphs_keep_the_spatial_parameters_and_drop_the_rollout_shift() {
    let parameters = |g: &Graph| {
        let mut names = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    let plain =
        world::build_world_only_pixel_training_graph(&spatial(), 2, PixelLoss::MotionCrossEntropy);
    let fixed_context = world::build_world_only_pixel_training_graph(
        &with_context(2),
        2,
        PixelLoss::MotionCrossEntropy,
    );
    assert_eq!(parameters(&plain), parameters(&fixed_context));
    assert_eq!(plain.outputs().len(), fixed_context.outputs().len());
    assert_ne!(plain.nodes().len(), fixed_context.nodes().len());
    for build in [
        world::build_observe_graph,
        world::build_transition_graph,
        world::build_head_graph,
    ] {
        assert_eq!(
            build(&spatial(), 2).nodes().len(),
            build(&with_context(2), 2).nodes().len()
        );
    }
}
