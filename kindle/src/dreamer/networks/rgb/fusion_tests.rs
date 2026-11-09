use super::*;
use crate::dreamer::runtime;
use meganeura::{
    Mode,
    graph::Op,
    reference::{Feeds, evaluate, gradients},
};
use std::{collections::HashSet, sync::Arc};

pub(super) fn ancestors(graph: &Graph, node: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![node];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(graph.node(n).inputs.iter().copied());
        }
    }
    seen
}

#[test]
fn fusion_hidden_convolution_has_both_global_and_history_ancestors() {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    let mut graph = Graph::new();
    let decoder = Decoder::new(&mut graph, &c, "world.decoder", c.feature_dim());
    let features = graph.input("features", &[2, c.feature_dim()]);
    let history = graph.input("history", &[2, 2 * c.observation_dim()]);
    let output = decoder.forward_logits_with_history(&mut graph, features, Some(history), 2);
    let fusion_weight = decoder.fusion.as_ref().unwrap().convolution.layer.weight;
    let conv = graph
        .nodes()
        .iter()
        .find(|node| matches!(node.op, Op::Conv2d { .. }) && node.inputs.contains(&fusion_weight))
        .unwrap();
    let seen = ancestors(&graph, conv.id);
    assert!(seen.contains(&features) && seen.contains(&history));
    assert!(ancestors(&graph, output).contains(&conv.id));
    assert_eq!(decoder.output.layer.in_channels, 16);
    assert_eq!(graph.node(output).ty.shape, [2, c.observation_dim()]);
}

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    parameters: Vec<(String, Vec<f32>)>,
}

fn fixture() -> Fixture {
    let mut graph = Graph::new();
    let global = graph.parameter("global", &[2, 32 * 3 * 3]);
    let history = graph.parameter("history", &[2, 8 * 3 * 3]);
    let joined = concat_columns(&mut graph, global, history, 2, 32 * 3 * 3, 8 * 3 * 3);
    let joined = graph.reshape(joined, &[2 * 40 * 3 * 3]);
    let fusion = ConvNorm::new(&mut graph, "fusion", 40, 16, 3);
    let output = Convolution::new(&mut graph, "output", 16, 3, 3);
    let fused = fusion.forward(&mut graph, joined, 2, false);
    let decoded = output.forward(&mut graph, fused, 2);
    let loss = gradients::weighted_loss(&mut graph, decoded, 991, 0.125);
    graph.set_outputs(vec![loss, fused, decoded]);
    let mut feeds = Feeds::new();
    let parameters = graph
        .nodes()
        .iter()
        .filter_map(|node| {
            let Op::Parameter { name } = &node.op else {
                return None;
            };
            let values = (0..node.ty.num_elements())
                .map(|i| {
                    if name.ends_with("norm.weight") {
                        1. + (i % 5) as f32 / 32.
                    } else if name.ends_with("bias") {
                        (i % 3) as f32 / 64. - 1. / 64.
                    } else {
                        ((i * 13 + 7) % 23) as f32 / 64. - 11. / 64.
                    }
                })
                .collect::<Vec<_>>();
            feeds.set(name, &values);
            Some((name.clone(), values))
        })
        .collect();
    Fixture {
        graph,
        feeds,
        parameters,
    }
}

#[test]
fn fusion_f64_derivatives_match_elementwise_and_directional_finite_differences() {
    let f = fixture();
    let options = gradients::Options {
        max_elementwise: 36,
        probes: 2,
        tolerance: meganeura::reference::Tolerance {
            rtol: 1e-5,
            floor: 1e-4,
        },
        ..Default::default()
    };
    gradients::check(&f.graph, &f.feeds, &options)
        .unwrap()
        .assert_passed("fusion finite differences");
}

#[test]
#[ignore = "requires separately declared guarded nonlinear fusion F64/native check"]
fn fusion_native_values_and_derivatives_match_f64() {
    let f = fixture();
    let backward = meganeura::autodiff::differentiate(&f.graph);
    let expected = evaluate(&backward, &f.feeds).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
    assert_eq!(
        session.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for (name, values) in &f.parameters {
        session.set_parameter(name, values);
    }
    session.clear_optimizer();
    session.step();
    session.wait();
    let compare = |actual: &[f32], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (f64::from(a) - b).abs() <= 3e-5 + 3e-4 * b.abs(),
                "{a} vs {b}"
            );
        }
    };
    for (i, &id) in f.graph.outputs().iter().enumerate() {
        let mut actual = vec![0.; f.graph.node(id).ty.num_elements()];
        session.read_output_by_index(i, &mut actual);
        compare(&actual, &expected[backward.outputs()[i] as usize].data);
    }
    for (i, (name, initial)) in f.parameters.iter().enumerate() {
        let mut gradient = vec![0.; initial.len()];
        session.read_param_grad(name, &mut gradient);
        compare(
            &gradient,
            &expected[backward.outputs()[f.graph.outputs().len() + i] as usize].data,
        );
        let mut value = vec![0.; initial.len()];
        session.read_param(name, &mut value);
        assert_eq!(&value, initial);
    }
    assert_eq!(session.adam_step_count(), 0);
}

#[test]
#[ignore = "requires separately declared guarded joint global/history decoder check"]
fn fusion_native_decoder_has_joint_logit_interaction_and_lane_isolation() {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    let mut graph = Graph::new();
    let decoder = Decoder::new(&mut graph, &c, "world.decoder", c.feature_dim());
    let feature = graph.input("features", &[2, c.feature_dim()]);
    let history = graph.input("history", &[2, 2 * c.observation_dim()]);
    let output = decoder.forward_logits_with_history(&mut graph, feature, Some(history), 2);
    graph.set_outputs(vec![output]);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut s = runtime::build_session(&graph, &gpu, Mode::Inference, false);
    assert_eq!(
        s.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    runtime::initialize_d3(&mut s, &graph, c.seed);
    let names = s
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let initial = names
        .iter()
        .map(|name| {
            let mut v = vec![0.; s.param_size(name).unwrap()];
            s.read_param(name, &mut v);
            v
        })
        .collect::<Vec<_>>();
    let mut outputs = Vec::new();
    for (change_global, change_history) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut features = vec![0.; 2 * c.feature_dim()];
        for b in 0..2 {
            for slot in 0..c.network().stoch {
                features[b * c.feature_dim() + c.network().deter + slot * c.network().classes] = 1.;
            }
        }
        if change_global {
            features[0] = 0.5;
        }
        let mut history = vec![0.; 4 * c.observation_dim()];
        if change_history {
            history[16 * 32 + 16] = 0.5;
        }
        s.set_input("features", &features);
        s.set_input("history", &history);
        s.step();
        s.wait();
        let mut values = vec![0.; 2 * c.observation_dim()];
        s.read_output_by_index(0, &mut values);
        assert!(values.iter().all(|v| v.is_finite()));
        outputs.push(values);
    }
    let width = c.observation_dim();
    for values in &outputs[1..] {
        assert_eq!(values[width..], outputs[0][width..]);
    }
    let interaction = (0..width)
        .map(|i| (outputs[3][i] - outputs[1][i] - outputs[2][i] + outputs[0][i]).abs())
        .fold(0., f32::max);
    assert!(interaction > 1e-5, "joint logit interaction {interaction}");
    assert_eq!(s.adam_step_count(), 0);
    for (name, expected) in names.iter().zip(initial) {
        let mut actual = vec![0.; expected.len()];
        s.read_param(name, &mut actual);
        assert_eq!(actual, expected);
    }
}
