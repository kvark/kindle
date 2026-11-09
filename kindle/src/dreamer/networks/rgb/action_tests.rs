use super::*;
use crate::dreamer::{ObservationKind, runtime};
use meganeura::{
    Mode,
    graph::Op,
    reference::{Feeds, evaluate, gradients},
};
use std::sync::Arc;

#[test]
fn action_planes_are_channel_major_lane_local_and_differentiable() {
    let mut g = Graph::new();
    let h = g.parameter("history", &[2, 24]);
    let a = g.parameter("action", &[2, 3]);
    let joined = action_history(&mut g, h, a, 2, 3, 2);
    let loss = gradients::weighted_loss(&mut g, joined, 91, 1.0);
    g.set_outputs(vec![loss, joined]);
    let history = (0..48).map(|i| i as f32 / 64.0 - 0.5).collect::<Vec<_>>();
    let action = [0., 1., 0., 1., 0., 0.];
    let mut feeds = Feeds::new();
    feeds.set("history", &history);
    feeds.set("action", &action);
    let values = evaluate(&g, &feeds).unwrap();
    let expected = (0..2)
        .flat_map(|b| {
            history[b * 24..(b + 1) * 24]
                .iter()
                .copied()
                .chain(action[b * 3..(b + 1) * 3].iter().flat_map(|&a| [a; 4]))
        })
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_eq!(values[joined as usize].data, expected);
    gradients::check(
        &g,
        &feeds,
        &gradients::Options {
            max_elementwise: 64,
            probes: 3,
            ..Default::default()
        },
    )
    .unwrap()
    .assert_passed("action planes");
    feeds.set("action", &[0., 1., 0., 0., 0., 0.]);
    let reset = evaluate(&g, &feeds).unwrap();
    assert_eq!(reset[joined as usize].data[..36], expected[..36]);
    assert_eq!(reset[joined as usize].data[60..], [0.; 12]);
}

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
}
fn fixture_with_options(transport: bool, late: bool) -> Fixture {
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_spatial_transport = transport;
    c.rgb_late_action = late;
    let mut g = Graph::new();
    let decoder = Decoder::new(&mut g, &c, "world.decoder", c.feature_dim());
    let state = g.parameter("fixture.state", &[2, c.feature_dim()]);
    let history = g.parameter("fixture.history", &[2, 2 * c.observation_dim()]);
    let action = g.parameter("fixture.action", &[2, 3]);
    let logits =
        decoder.forward_logits_with_action_history(&mut g, state, Some(history), Some(action), 2);
    let pixels = crate::dreamer::networks::centered_pixels(&mut g, logits);
    let loss = gradients::weighted_loss(&mut g, pixels, 93, 1.0);
    g.set_outputs(vec![loss, pixels]);
    let parameters = g
        .nodes()
        .iter()
        .filter_map(|n| {
            let Op::Parameter { name } = &n.op else {
                return None;
            };
            let values = if name == "fixture.action" {
                vec![0., 1., 0., 0., 0., 1.]
            } else if name.ends_with("norm.weight") {
                vec![1.; n.ty.num_elements()]
            } else if name.ends_with("bias") {
                vec![0.; n.ty.num_elements()]
            } else {
                (0..n.ty.num_elements())
                    .map(|i| ((i * 13 + name.len() * 7) % 37) as f32 / 370.0 - 0.05)
                    .collect()
            };
            Some((name.clone(), values))
        })
        .collect::<Vec<_>>();
    let mut feeds = Feeds::new();
    for (n, v) in &parameters {
        feeds.set(n, v);
    }
    Fixture {
        graph: g,
        feeds,
        parameters,
    }
}

#[test]
fn local_decoder_responds_to_actions_at_fixed_state_and_history() {
    reference_actions_and_history(false);
}

#[test]
fn action_transport_decoder_has_causal_action_history_derivatives_and_lane_isolation() {
    reference_actions_and_history(true);
}

fn reference_actions_and_history(transport: bool) {
    reference_options(transport, false);
}

fn reference_options(transport: bool, late: bool) {
    let mut f = fixture_with_options(transport, late);
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let original = evaluate(&backward, &f.feeds).unwrap();
    assert!(original.iter().flat_map(|v| &v.data).all(|x| x.is_finite()));
    for name in ["fixture.action", "fixture.history"] {
        let index = f.parameters.iter().position(|(n, _)| n == name).unwrap();
        assert!(
            original[backward.outputs()[2 + index] as usize]
                .data
                .iter()
                .any(|x| x.abs() > 1e-10),
            "{name}"
        );
    }
    if transport || late {
        for name in ["fixture.action", "fixture.history"] {
            let index = f.parameters.iter().position(|(n, _)| n == name).unwrap();
            let values = f.parameters[index].1.clone();
            let direction = (0..values.len())
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect::<Vec<f64>>();
            let norm = (values.len() as f64).sqrt();
            let expected = original[backward.outputs()[2 + index] as usize]
                .data
                .iter()
                .zip(&direction)
                .map(|(g, d)| g * d / norm)
                .sum::<f64>();
            let losses = [-1.0, 1.0].map(|sign| {
                let changed = values
                    .iter()
                    .zip(&direction)
                    .map(|(&v, d)| (f64::from(v) + sign * 0.01 * d / norm) as f32)
                    .collect::<Vec<_>>();
                f.feeds.set(name, &changed);
                let reference = evaluate(&f.graph, &f.feeds).unwrap();
                reference[f.graph.outputs()[0] as usize].data[0]
            });
            f.feeds.set(name, &values);
            let observed = (losses[1] - losses[0]) / 0.02;
            assert!(
                (observed - expected).abs() <= 3e-6 + 3e-3 * expected.abs(),
                "{name}: {observed} vs {expected}"
            );
        }
    }
    let pixels = f.graph.outputs()[1];
    f.feeds.set("fixture.action", &[0., 1., 0., 1., 0., 0.]);
    let changed = evaluate(&f.graph, &f.feeds).unwrap();
    let width = 3 * 32 * 32;
    assert_eq!(
        original[pixels as usize].data[..width],
        changed[pixels as usize].data[..width]
    );
    assert!(
        original[pixels as usize].data[width..]
            .iter()
            .zip(&changed[pixels as usize].data[width..])
            .any(|(a, b)| (a - b).abs() > 1e-7)
    );
}

#[test]
#[ignore = "requires separately declared guarded local-action composition qualification"]
fn local_action_native_values_gradients_and_lane_controls_match_f64() {
    native_values_gradients_and_lane_controls(false);
}

#[test]
#[ignore = "requires separately declared guarded action-transport composition qualification"]
fn action_transport_native_values_gradients_and_lane_controls_match_f64() {
    native_values_gradients_and_lane_controls(true);
}

fn native_values_gradients_and_lane_controls(transport: bool) {
    native_options(transport, false);
}

fn native_options(transport: bool, late: bool) {
    let mut f = fixture_with_options(transport, late);
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    assert_eq!(
        session.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    session.clear_optimizer();
    let mut first_lane = None;
    for action in [
        [0., 1., 0., 0., 0., 1.],
        [0., 1., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
    ] {
        f.feeds.set("fixture.action", &action);
        for (name, values) in &f.parameters {
            session.set_parameter(
                name,
                if name == "fixture.action" {
                    &action
                } else {
                    values
                },
            );
        }
        let expected = evaluate(&backward, &f.feeds).unwrap();
        session.step();
        session.wait();
        if late {
            let mut pixels = vec![0.; 2 * 3 * 32 * 32];
            session.read_output_by_index(1, &mut pixels);
            let lane = pixels[..3 * 32 * 32].to_vec();
            if let Some(expected) = &first_lane {
                assert_eq!(&lane, expected);
            } else {
                first_lane = Some(lane);
            }
        }
        let check = |actual: &[f32], expected: &[f64]| {
            assert_eq!(actual.len(), expected.len());
            for (&a, &b) in actual.iter().zip(expected) {
                assert!(
                    a.is_finite() && (f64::from(a) - b).abs() <= 2e-4 + 2e-3 * b.abs(),
                    "{a} vs {b}"
                );
            }
        };
        for (i, &id) in f.graph.outputs().iter().enumerate() {
            let mut values = vec![0.; f.graph.node(id).ty.num_elements()];
            session.read_output_by_index(i, &mut values);
            check(&values, &expected[backward.outputs()[i] as usize].data);
        }
        for (i, (name, values)) in f.parameters.iter().enumerate() {
            let mut actual = vec![0.; values.len()];
            session.read_param_grad(name, &mut actual);
            check(&actual, &expected[backward.outputs()[2 + i] as usize].data);
            session.read_param(name, &mut actual);
            assert_eq!(
                actual,
                if name == "fixture.action" {
                    action.to_vec()
                } else {
                    values.clone()
                }
            );
        }
    }
    assert_eq!(session.adam_step_count(), 0);
}

#[test]
fn late_action_cpu_derivatives_and_lane_isolation() {
    reference_options(false, true);
}

#[test]
#[ignore = "requires separately declared guarded late-action qualification"]
fn late_action_native_values_gradients_and_lane_controls_match_f64() {
    native_options(false, true);
}

#[test]
fn late_action_motion_ancestors_exclude_commands_until_fusion() {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_late_action = true;
    let mut graph = Graph::new();
    let decoder = Decoder::new(&mut graph, &c, "world.decoder", c.feature_dim());
    let state = graph.input("state", &[2, c.feature_dim()]);
    let history = graph.input("history", &[2, 2 * c.observation_dim()]);
    let action = graph.input("action", &[2, 18]);
    decoder.forward_logits_with_action_history(&mut graph, state, Some(history), Some(action), 2);
    for (index, layer) in decoder
        .motion
        .iter()
        .chain(decoder.fusion.as_deref())
        .enumerate()
    {
        let weight = layer.convolution.layer.weight;
        let conv = graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Conv2d { .. }) && n.inputs.contains(&weight))
            .unwrap();
        let seen = super::fusion_tests::ancestors(&graph, conv.id);
        assert!(seen.contains(&history));
        assert_eq!(seen.contains(&action), index == 3);
        assert_eq!(seen.contains(&state), index == 3);
    }
}
