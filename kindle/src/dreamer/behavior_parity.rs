//! Row-independent head gradients: full B16/T64/H15 versus sixteen row slices.

use std::ops::Range;
use std::sync::Arc;

use meganeura::{Mode, Session};

use super::build_training_graph;
use crate::dreamer::DreamerConfig;
use crate::dreamer::runtime::{build_session, initialize_d3};

fn weight(row: usize) -> f32 {
    if row.is_multiple_of(11) {
        0.0
    } else {
        (row % 7 + 1) as f32 / 7.0
    }
}

fn features(config: &DreamerConfig, rows: Range<usize>, offset: usize) -> Vec<f32> {
    let network = config.network();
    let mut values = vec![0.0; rows.len() * config.feature_dim()];
    for (local, row) in rows.enumerate() {
        let target = &mut values[local * config.feature_dim()..][..config.feature_dim()];
        for (column, value) in target[..network.deter].iter_mut().enumerate() {
            *value = ((row * 17 + column * 13 + offset) as f32 * 0.017).sin() * 0.3;
        }
        for variable in 0..network.stoch {
            let class = (row * 7 + variable * 3 + offset) % network.classes;
            target[network.deter + variable * network.classes + class] = 1.0;
        }
    }
    values
}

fn targets(rows: Range<usize>, width: usize, offset: usize) -> Vec<f32> {
    rows.flat_map(|row| {
        (0..width).map(move |column| f32::from(column == (row * 7 + offset) % width))
    })
    .collect()
}

fn inputs(
    config: &DreamerConfig,
    imagined: Range<usize>,
    replay: Range<usize>,
) -> Vec<(&'static str, Vec<f32>)> {
    let mut actions = targets(imagined.clone(), config.action_count, 1);
    for (row, target) in imagined
        .clone()
        .zip(actions.chunks_exact_mut(config.action_count))
    {
        let advantage = (row % 9) as f32 / 4.0 - 1.0;
        for value in target {
            *value *= advantage * weight(row);
        }
    }
    vec![
        ("imagined_feature", features(config, imagined.clone(), 0)),
        ("action_target", actions),
        ("imagined_weight", imagined.clone().map(weight).collect()),
        (
            "imagined_value_target",
            targets(imagined.clone(), config.value_bins, 2),
        ),
        (
            "imagined_slow_target",
            targets(imagined, config.value_bins, 3),
        ),
        ("replay_feature", features(config, replay.clone(), 71)),
        ("replay_weight", replay.clone().map(weight).collect()),
        (
            "replay_value_target",
            targets(replay.clone(), config.value_bins, 5),
        ),
        ("replay_slow_target", targets(replay, config.value_bins, 7)),
    ]
}

fn fill(session: &mut Session, values: &[(&str, Vec<f32>)]) {
    for (name, value) in values {
        session.set_input(name, value);
    }
}

#[test]
fn production_rows_include_the_whole_imagination_horizon() {
    for action_count in [4, 18] {
        let config = DreamerConfig::new(action_count);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.batch_length * config.imagination_length, 960);
        assert_eq!(
            config.batch_size * config.batch_length * config.imagination_length,
            15360
        );
        assert_eq!(config.batch_size * (config.batch_length - 1), 1008);
    }
}

#[test]
fn partitioned_inputs_match_full_inputs_without_regenerating_row_rngs() {
    for actions in [4, 18] {
        let config = DreamerConfig::tiny(actions);
        let full = inputs(&config, 0..12, 0..8);
        let left = inputs(&config, 0..6, 0..4);
        let right = inputs(&config, 6..12, 4..8);
        assert_eq!(full.len(), 9);
        for ((name, expected), ((left_name, mut a), (right_name, b))) in
            full.into_iter().zip(left.into_iter().zip(right))
        {
            assert_eq!(name, left_name);
            assert_eq!(name, right_name);
            a.extend(b);
            assert_eq!(expected, a, "partitioned {name}");
        }
        let targets = inputs(&config, 0..12, 0..8).swap_remove(1).1;
        assert!(targets.iter().any(|value| *value > 0.0));
        assert!(targets.iter().any(|value| *value < 0.0));
        assert!(targets.contains(&0.0));
    }
}

#[test]
#[ignore = "compares all production actor/value gradients and losses on GPU"]
fn published_actions_full_behavior_gradients_match_row_partitions() {
    check_behavior_gradients(18);
}

#[test]
#[ignore = "compares all four-action production actor/value gradients and losses on GPU"]
fn minimal_actions_full_behavior_gradients_match_row_partitions() {
    check_behavior_gradients(4);
}

fn check_behavior_gradients(action_count: usize) {
    let config = DreamerConfig::new(action_count);
    let partitions = config.batch_size;
    let imagined_per_part = config.batch_length * config.imagination_length;
    let replay_per_part = config.batch_length - 1;
    let imagined_rows = partitions * imagined_per_part;
    let replay_rows = partitions * replay_per_part;
    assert_eq!((partitions, imagined_rows, replay_rows), (16, 15360, 1008));
    let gpu = Arc::new(crate::init_gpu_context().expect("Vulkan compute device"));
    let make_session = |imagined, replay| {
        let graph = build_training_graph(&config, imagined, replay);
        let mut session = build_session(&graph, &gpu, Mode::Training, false);
        initialize_d3(&mut session, &graph, config.seed ^ 0x5eed_0000_0000_0001);
        // Exercise the value trunk as well as its normally zero-initialized head.
        let name = "behavior.value.out.weight";
        let values = (0..session.param_size(name).unwrap())
            .map(|index| (index as f32 * 0.17).sin() * 0.02)
            .collect::<Vec<_>>();
        session.set_parameter(name, &values);
        session.clear_optimizer();
        session
    };
    let mut full = make_session(imagined_rows, replay_rows);
    let mut part = make_session(imagined_per_part, replay_per_part);
    let mut names = full
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    names.sort_unstable();
    let mut part_names = part
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    part_names.sort_unstable();
    assert!(!names.is_empty());
    assert_eq!(names, part_names);
    let mut expected_gradients = Vec::new();
    for name in &names {
        assert!(
            full.has_param_grad(name) && part.has_param_grad(name),
            "missing gradient: {name}"
        );
        assert_eq!(full.param_size(name), part.param_size(name));
        assert_eq!(
            full.read_params(&[name]),
            part.read_params(&[name]),
            "changed initialization: {name}"
        );
        expected_gradients.push(vec![0.0_f64; full.param_size(name).unwrap()]);
    }
    fill(
        &mut full,
        &inputs(&config, 0..imagined_rows, 0..replay_rows),
    );
    full.step();
    full.wait();
    let mut expected_losses = [0.0_f64; 6];
    for index in 0..partitions {
        fill(
            &mut part,
            &inputs(
                &config,
                index * imagined_per_part..(index + 1) * imagined_per_part,
                index * replay_per_part..(index + 1) * replay_per_part,
            ),
        );
        part.step();
        part.wait();
        for (metric, total) in expected_losses.iter_mut().enumerate() {
            let mut value = [0.0];
            part.read_output_by_index(metric, &mut value);
            assert!(value[0].is_finite());
            *total += f64::from(value[0]) / partitions as f64;
        }
        for (name, total) in names.iter().zip(&mut expected_gradients) {
            let mut gradient = vec![0.0; total.len()];
            part.read_param_grad(name, &mut gradient);
            assert!(
                gradient.iter().all(|value| value.is_finite()),
                "non-finite {name}"
            );
            for (expected, value) in total.iter_mut().zip(gradient) {
                *expected += f64::from(value) / partitions as f64;
            }
        }
    }
    for (metric, expected) in expected_losses.into_iter().enumerate() {
        let mut actual = [0.0];
        full.read_output_by_index(metric, &mut actual);
        let actual = f64::from(actual[0]);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 3e-4 * expected.abs().max(1.0),
            "actions={action_count}, metric={metric}: full={actual}, partitioned={expected}"
        );
    }
    let mut worst_relative = 0.0_f64;
    for (name, expected) in names.iter().zip(expected_gradients) {
        let mut actual = vec![0.0; expected.len()];
        full.read_param_grad(name, &mut actual);
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "non-finite {name}"
        );
        let difference = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (f64::from(*a) - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let actual_norm = actual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected_norm = expected
            .iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = actual_norm.max(expected_norm);
        assert!(norm > 0.0, "fixture did not exercise {name}");
        let relative = difference / norm;
        worst_relative = worst_relative.max(relative);
        assert!(
            difference <= 3e-3 * norm + 1e-7,
            "actions={action_count}, {name}: relative L2={relative}, norm={norm}"
        );
    }
    eprintln!(
        "actions={action_count}, imagined_rows={imagined_rows}, replay_rows={replay_rows}, \
        all_parameter_gradients={}, worst_relative={worst_relative}",
        names.len()
    );
}
