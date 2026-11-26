use super::*;
use crate::dreamer::networks::{snap_to_palette, snap_to_palette_cpu};
use crate::dreamer::{DreamerCore, ObservationKind};
use meganeura::graph::Op;
use meganeura::reference::{Feeds, evaluate, gradients};

const PALETTE: [[f32; 3]; 3] = [[-0.5, -0.5, -0.5], [0.28, -0.22, -0.22], [0.06, 0.06, 0.06]];

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

fn with_palette() -> DreamerConfig {
    let mut c = config();
    c.rgb_history_palette = Some(PALETTE.to_vec());
    c
}

#[test]
fn history_palette_requires_the_spatial_decoder_and_bounded_centered_colours() {
    let c = with_palette();
    assert!(c.check().is_ok() && validate_config(&c).is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    assert!(
        serde_json::to_value(config())
            .unwrap()
            .get("rgb_history_palette")
            .is_none()
    );
    let mut bad = c.clone();
    bad.rgb_spatial_decoder = false;
    bad.rgb_spatial_fusion = false;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.rgb_history_palette = Some(vec![[0.0; 3]; 33]);
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.rgb_history_palette = Some(vec![[0.0; 3], [0.6, 0.0, 0.0]]);
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.rgb_history_palette = Some(vec![[0.0; 3]]);
    assert!(bad.check().is_err());
    assert!(std::panic::catch_unwind(|| DreamerCore::new(c)).is_err());
}

#[test]
fn cpu_snap_picks_the_nearest_colour_and_the_first_of_a_tie() {
    // Two pixels per row, channel-major: pixel0 = (0.3, -0.2, -0.2) is red-ish,
    // pixel1 = (-0.22, -0.22, -0.22) ties between black and grey in distance? No:
    // black is 0.28 away, grey 0.28 away -> first (black) wins.
    let frame = [0.3, -0.22, -0.2, -0.22, -0.2, -0.22];
    let snapped = snap_to_palette_cpu(&frame, &PALETTE, 6);
    assert_eq!(snapped, [0.28, -0.5, -0.22, -0.5, -0.22, -0.5]);
}

#[test]
fn graph_snap_matches_the_cpu_snap_and_passes_gradients_straight_through() {
    let (batch, side) = (2, 4);
    let width = 3 * side * side;
    let frame = (0..batch * width)
        .map(|i| ((i * 7919) % 101) as f32 / 100.0 - 0.5)
        .collect::<Vec<_>>();
    let mut snapped_graph = Graph::new();
    let input = snapped_graph.parameter("frame", &[batch, width]);
    let snapped = snap_to_palette(&mut snapped_graph, input, &PALETTE, batch, side);
    let loss = gradients::weighted_loss(&mut snapped_graph, snapped, 4101, 1.0);
    snapped_graph.set_outputs(vec![loss, snapped]);
    let mut plain_graph = Graph::new();
    let plain_input = plain_graph.parameter("frame", &[batch, width]);
    let plain_loss = gradients::weighted_loss(&mut plain_graph, plain_input, 4101, 1.0);
    plain_graph.set_outputs(vec![plain_loss]);
    let mut feeds = Feeds::new();
    feeds.set("frame", &frame);
    let values = evaluate(&snapped_graph, &feeds).unwrap();
    let expected = snap_to_palette_cpu(&frame, &PALETTE, width);
    let actual = &values[snapped as usize].data;
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - f64::from(*e)).abs() < 1e-6, "{a} != {e}");
    }
    assert!(
        expected
            .iter()
            .all(|v| PALETTE.iter().flatten().any(|c| c == v))
    );
    assert_ne!(expected, frame);
    let snapped_backward = meganeura::autodiff::differentiate(&snapped_graph);
    let plain_backward = meganeura::autodiff::differentiate(&plain_graph);
    let snapped_values = evaluate(&snapped_backward, &feeds).unwrap();
    let plain_values = evaluate(&plain_backward, &feeds).unwrap();
    let snapped_gradient = &snapped_values[snapped_backward.outputs()[2] as usize].data;
    let plain_gradient = &plain_values[plain_backward.outputs()[1] as usize].data;
    assert_eq!(snapped_gradient.len(), batch * width);
    assert!(plain_gradient.iter().any(|g| g.abs() > 1e-3));
    for (s, p) in snapped_gradient.iter().zip(plain_gradient) {
        assert!((s - p).abs() < 1e-9, "straight-through gradient {s} != {p}");
    }
}

#[test]
fn rollout_graphs_snap_only_the_fed_back_history_when_a_palette_is_set() {
    let count = |c: &DreamerConfig| {
        let g = world::build_world_only_pixel_training_graph(c, 2, PixelLoss::MotionCrossEntropy);
        (
            g.nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::ArgMaxInner))
                .count(),
            g.outputs().len(),
            g.nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Parameter { .. }))
                .count(),
        )
    };
    let (plain_argmax, plain_outputs, plain_parameters) = count(&config());
    let (palette_argmax, palette_outputs, palette_parameters) = count(&with_palette());
    assert_eq!(palette_argmax, plain_argmax + 2);
    assert_eq!(palette_outputs, plain_outputs);
    assert_eq!(palette_parameters, plain_parameters);
    for build in [
        world::build_observe_graph,
        world::build_transition_graph,
        world::build_head_graph,
    ] {
        let plain = build(&config(), 2);
        let palette = build(&with_palette(), 2);
        assert_eq!(plain.nodes().len(), palette.nodes().len());
    }
}
