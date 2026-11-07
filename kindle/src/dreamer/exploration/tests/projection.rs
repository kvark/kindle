use super::*;
use crate::dreamer::{ModelSize, ObservationKind};
use meganeura::{CoopPolicy, SessionConfig, SessionOptions, graph::Op};

fn graph(config: &DreamerConfig, rows: usize, factored: bool) -> Graph {
    let mut graph = Graph::new();
    let ensemble = Disagreement::new(&mut graph, config);
    let state = graph.input("state", &[rows, config.feature_dim()]);
    let predictions = if factored {
        ensemble
            .heads
            .iter()
            .map(|head| head.forward_all_actions(&mut graph, state, config.action_count))
            .collect::<Vec<_>>()
    } else {
        let state = repeat_rows(&mut graph, state, config.action_count);
        let actions = config.action_count;
        let action = graph.constant(
            (0..rows * actions * actions)
                .map(|i| f32::from(i / actions % actions == i % actions))
                .collect(),
            &[rows * actions, actions],
        );
        ensemble.predict(&mut graph, state, action)
    };
    let bonus = action_effect_disagreement(&mut graph, &predictions, config.action_count);
    graph.set_outputs(predictions.into_iter().chain([bonus]).collect());
    graph
}

#[test]
fn projection_reuse_preserves_parameters_and_reduces_first_layer_rows() {
    for model_size in [ModelSize::Tiny, ModelSize::Size1M, ModelSize::Size12M] {
        let config = DreamerConfig {
            model_size,
            observation_kind: ObservationKind::Rgb64,
            ..DreamerConfig::tiny(18)
        };
        let control = graph(&config, 128, false);
        let candidate = graph(&config, 128, true);
        let parameters = |graph: &Graph| {
            graph
                .nodes()
                .iter()
                .filter_map(|node| {
                    if let Op::Parameter { name } = &node.op {
                        Some((name.clone(), node.ty.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(parameters(&control), parameters(&candidate));
        for (graph, rows) in [(&control, 128 * 18), (&candidate, 128)] {
            let products = graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::MatMul))
                .collect::<Vec<_>>();
            assert_eq!(products.len(), MEMBERS * 3);
            for head in products.as_chunks::<3>().0 {
                assert_eq!(graph.node(head[0].inputs[0]).ty.shape[0], rows);
                for product in &head[1..] {
                    assert_eq!(graph.node(product.inputs[0]).ty.shape[0], 128 * 18);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; same-precision projection algebra and reduced-input error check"]
fn projection_reuse_matches_expanded_all_action_predictions() {
    let gpu = gpu();
    for model_size in [ModelSize::Tiny, ModelSize::Size1M, ModelSize::Size12M] {
        let config = DreamerConfig {
            model_size,
            observation_kind: ObservationKind::Rgb64,
            ..DreamerConfig::tiny(18)
        };
        let rows = 128;
        let state = (0..rows * config.feature_dim())
            .map(|i| ((i as f32 * 0.071).sin() + (i as f32 * 0.013).cos()) * 0.5)
            .collect::<Vec<_>>();
        let run = |factored, coop| {
            let graph = graph(&config, rows, factored);
            let mut session = meganeura::build(
                &graph,
                SessionConfig {
                    gpu: Some(Arc::clone(&gpu)),
                    mode: Mode::Inference,
                    runtime: SessionOptions {
                        coop,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .0;
            initialize_d3(&mut session, &graph, 701);
            session.set_input("state", &state);
            session.step();
            session.wait();
            (0..=MEMBERS)
                .map(|index| {
                    let width = if index == MEMBERS {
                        1
                    } else {
                        config.encoded_observation_dim()
                    };
                    let mut data = vec![0.0; rows * 18 * width];
                    session.read_output_by_index(index, &mut data);
                    assert!(data.iter().all(|value| value.is_finite()));
                    data
                })
                .collect::<Vec<_>>()
        };
        let reference = run(false, CoopPolicy::NativeF32);
        for (coop, limit) in [(CoopPolicy::NativeF32, 2e-4), (CoopPolicy::Auto, 0.02)] {
            let actual = run(true, coop);
            for (index, (expected, values)) in reference.iter().zip(&actual).enumerate() {
                let square = |value: f32| f64::from(value).powi(2);
                let error = expected
                    .iter()
                    .zip(values)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>();
                let relative_l2 =
                    (error / expected.iter().map(|v| square(*v)).sum::<f64>().max(1e-20)).sqrt();
                eprintln!(
                    "projection {model_size:?} {coop:?} output{index}: relative_l2={relative_l2}"
                );
                assert!(
                    relative_l2 <= limit,
                    "output{index}: {relative_l2} > {limit}"
                );
            }
        }
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
}

#[test]
#[ignore = "requires GPU; selected centered projections versus all-action reference, including resets"]
fn selected_action_effect_matches_expanded_bonus() {
    let gpu = gpu();
    let rows = 128;
    for model_size in [ModelSize::Tiny, ModelSize::Size1M, ModelSize::Size12M] {
        let config = DreamerConfig {
            model_size,
            observation_kind: ObservationKind::Rgb64,
            ..DreamerConfig::tiny(18)
        };
        let states = (0..rows * config.feature_dim())
            .map(|i| (i as f32 * 0.071).sin())
            .collect::<Vec<_>>();
        let actions = (0..rows * 18)
            .map(|i| f32::from(i / 18 % 7 != 0 && i % 18 == i / 18 % 18))
            .collect::<Vec<_>>();
        let run = |selected_only, coop| {
            let graph = if selected_only {
                let mut graph = Graph::new();
                let ensemble = Disagreement::new(&mut graph, &config);
                let state = graph.input("state", &[rows, config.feature_dim()]);
                let action = graph.input("action", &[rows, 18]);
                let bonus = ensemble.bonus(&mut graph, state, action);
                graph.set_outputs(vec![bonus]);
                graph
            } else {
                let mut graph = graph(&config, rows, false);
                let all = graph.outputs()[MEMBERS];
                let all = graph.reshape(all, &[rows, 18]);
                let action = graph.input("action", &[rows, 18]);
                let selected = graph.mul(all, action);
                let bonus = graph.sum_inner(selected);
                graph.set_outputs(vec![bonus]);
                graph
            };
            let mut session = meganeura::build(
                &graph,
                SessionConfig {
                    gpu: Some(Arc::clone(&gpu)),
                    mode: Mode::Inference,
                    runtime: SessionOptions {
                        coop,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .0;
            initialize_d3(&mut session, &graph, 701);
            session.set_input("state", &states);
            session.set_input("action", &actions);
            session.step();
            session.wait();
            let actual = session.read_output(rows);
            assert!(actual.iter().all(|v| v.is_finite()));
            for row in (0..rows).step_by(7) {
                assert!(
                    actual[row].abs() <= 1e-9,
                    "reset row bonus: {}",
                    actual[row]
                );
            }
            actual
        };
        let reference = run(false, CoopPolicy::NativeF32);
        for (coop, limit) in [(CoopPolicy::NativeF32, 2e-4), (CoopPolicy::Auto, 0.02)] {
            let actual = run(true, coop);
            let error = reference
                .iter()
                .zip(&actual)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>();
            let norm = reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            let relative = (error / norm.max(1e-20)).sqrt();
            eprintln!("selected bonus {model_size:?}/{coop:?}: relative_l2={relative}");
            assert!(relative <= limit, "{relative} > {limit}");
        }
    }
}
