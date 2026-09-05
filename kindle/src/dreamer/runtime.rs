//! Meganeura session construction, D3 initialization, and model syncing.

use std::collections::HashMap;
use std::sync::Arc;

use meganeura::graph::Op;
use meganeura::{Graph, Mode, Session, SessionConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::config::DreamerConfig;

pub(crate) fn build_session(
    graph: &Graph,
    gpu: &Arc<blade_graphics::Context>,
    mode: Mode,
    skip_full_optimize: bool,
) -> Session {
    meganeura::build(
        graph,
        SessionConfig {
            mode,
            gpu: Some(Arc::clone(gpu)),
            skip_full_optimize: mode == Mode::Training && skip_full_optimize,
            ..SessionConfig::default()
        },
    )
    .0
}

pub(crate) fn initialize_d3(session: &mut Session, graph: &Graph, seed: u64) {
    let shapes = graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.as_str(), node.ty.shape.as_slice())),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let parameters = session
        .param_names()
        .into_iter()
        .map(|name| {
            let size = session.param_size(name).expect("known parameter size");
            (name.to_owned(), size)
        })
        .collect::<Vec<_>>();
    for (name, size) in parameters {
        let shape = shapes
            .get(name.as_str())
            .copied()
            .unwrap_or_else(|| panic!("parameter {name} missing from source graph"));
        let values = if name.ends_with(".norm.weight") {
            vec![1.0; size]
        } else if name.ends_with(".bias") {
            vec![0.0; size]
        } else if name == "world.reward.out.weight" || name == "behavior.value.out.weight" {
            // D3 initializes reward and value predictions to exactly zero.
            vec![0.0; size]
        } else {
            let output_scale = if name == "behavior.actor.out.weight" {
                0.01
            } else {
                1.0
            };
            let fan_in = d3_fan_in(shape);
            // Match reconstruction's spatial-head initialization in the causal
            // ablation. Only the trunk's smaller deterministic input differs.
            let initialization_name = match name.strip_prefix("world.future_predictor.") {
                Some(suffix) => format!("world.decoder.{suffix}"),
                None => name.clone(),
            };
            truncated_normal(&initialization_name, size, fan_in, output_scale, seed)
        };
        session.set_parameter(&name, &values);
    }
}

fn d3_fan_in(shape: &[usize]) -> usize {
    match shape {
        [] | [_] => 1,
        [input, _] => *input,
        _ => shape[..shape.len() - 1].iter().product(),
    }
}

fn truncated_normal(
    name: &str,
    size: usize,
    fan_in: usize,
    output_scale: f32,
    seed: u64,
) -> Vec<f32> {
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut rng = StdRng::seed_from_u64(hash);
    let scale = 1.1368 / (fan_in as f32).sqrt() * output_scale;
    let mut output = Vec::with_capacity(size);
    while output.len() < size {
        let u1 = rng.random_range(f32::EPSILON..1.0);
        let u2 = rng.random::<f32>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        for normal in [radius * angle.cos(), radius * angle.sin()] {
            if normal.abs() <= 2.0 {
                output.push(normal * scale);
                if output.len() == size {
                    break;
                }
            }
        }
    }
    output
}

pub(crate) fn sync_matching(source: &Session, target: &mut Session, prefix: &str) {
    let names = source
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .filter(|name| source.param_size(name) == target.param_size(name))
        .collect::<Vec<_>>();
    let values = source.read_params(&names);
    for (name, values) in names.into_iter().zip(values) {
        target.set_parameter(name, &values);
    }
}

pub(crate) fn ema_matching(source: &Session, target: &mut Session, prefix: &str, rate: f32) {
    assert!((0.0..0.5).contains(&rate));
    let names = source
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .filter(|name| source.param_size(name) == target.param_size(name))
        .collect::<Vec<_>>();
    let source_values = source.read_params(&names);
    let target_values = target.read_params(&names);
    for ((name, source_values), mut target_values) in
        names.into_iter().zip(source_values).zip(target_values)
    {
        for (target, source) in target_values.iter_mut().zip(source_values) {
            *target = rate * source + (1.0 - rate) * *target;
        }
        target.set_parameter(name, &target_values);
    }
}

pub(crate) fn configure_d3_optimizer(
    session: &mut Session,
    config: &DreamerConfig,
    learner_step: u64,
    learning_rate: f32,
) {
    session.set_laprop(
        d3_learning_rate(learning_rate, config.learning_rate_warmup, learner_step),
        config.optimizer_beta1,
        config.optimizer_beta2,
        config.optimizer_epsilon,
    );
    session.set_adaptive_grad_clip(config.agc, config.agc_pmin);
}

fn d3_learning_rate(learning_rate: f32, warmup: u64, learner_step: u64) -> f32 {
    let multiplier = if warmup == 0 {
        1.0
    } else {
        (learner_step as f32 / warmup as f32).min(1.0)
    };
    learning_rate * multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_is_deterministic_and_bounded() {
        let left = truncated_normal("weight", 10_000, 100, 1.0, 7);
        let right = truncated_normal("weight", 10_000, 100, 1.0, 7);
        assert_eq!(left, right);
        let bound = 2.0 * 1.1368 / 10.0;
        assert!(left.iter().all(|value| value.abs() <= bound));
    }

    #[test]
    fn truncation_changes_with_experiment_seed() {
        let seed_zero = truncated_normal("weight", 10_000, 100, 1.0, 0);
        let seed_one = truncated_normal("weight", 10_000, 100, 1.0, 1);
        assert_ne!(seed_zero, seed_one);
    }

    #[test]
    fn grouped_block_linear_uses_upstream_fan_in() {
        assert_eq!(d3_fan_in(&[8, 96, 64]), 768);
        assert_eq!(d3_fan_in(&[96, 64]), 96);
    }

    #[test]
    fn learning_rate_warmup_matches_optax_step_indexing() {
        let mut config = DreamerConfig::tiny(2);
        config.learning_rate = 4e-5;
        config.learning_rate_warmup = 1_000;
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 0), 0.0);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 500), 2e-5);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 1_000), 4e-5);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 2_000), 4e-5);
    }
}
