use super::*;
use crate::dreamer::ModelSize;
use meganeura::graph::Op;
use std::collections::BTreeMap;

fn parameters(graph: &meganeura::Graph) -> BTreeMap<String, Vec<usize>> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn posterior_average_graph_preserves_actor_parameters() {
    for model in [ModelSize::Tiny, ModelSize::Size12M] {
        let mut config = super::super::latent_shape_tests::config();
        config.model_size = model;
        let ordinary = behavior::build_actor_inference_graph(&config, 2);
        for samples in [2, 8, 16] {
            let average = behavior::build_actor_posterior_average_graph(&config, 2, samples);
            assert_eq!(parameters(&ordinary), parameters(&average));
            assert_eq!(average.node(average.outputs()[0]).ty.shape, [2, 3]);
            let uniforms = average
                .nodes()
                .iter()
                .find(|node| {
                    matches!(&node.op,
                Op::Input { name } if name == "posterior_uniforms")
                })
                .unwrap();
            assert_eq!(
                uniforms.ty.shape,
                [
                    2 * samples * config.network().stoch,
                    config.network().classes
                ]
            );
        }
    }
    for samples in [0, 1, 17, usize::MAX] {
        assert!(check_actor_posterior_samples(samples).is_err());
    }
    for samples in [2, 8, 16] {
        assert!(check_actor_posterior_samples(samples).is_ok());
    }
}

fn probabilities(logits: &[f32], mix: f32) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut values: Vec<_> = logits.iter().map(|&x| (f64::from(x) - max).exp()).collect();
    let total: f64 = values.iter().sum();
    for value in &mut values {
        *value = *value / total * (1.0 - f64::from(mix)) + f64::from(mix) / logits.len() as f64;
    }
    values
}

fn sample(logits: &[f32], draws: &[f32], mix: f32) -> Vec<f32> {
    let values = probabilities(logits, mix);
    let index = values
        .iter()
        .zip(draws)
        .enumerate()
        .max_by(|a, b| {
            let score = |value: f64, draw: f32| {
                value.ln()
                    - (-f64::from(draw.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON)).ln()).ln()
            };
            score(*a.1.0, *a.1.1)
                .total_cmp(&score(*b.1.0, *b.1.1))
                .then_with(|| b.0.cmp(&a.0))
        })
        .unwrap()
        .0;
    (0..logits.len())
        .map(|i| if i == index { 1.0 } else { 0.0 })
        .collect()
}

#[test]
fn posterior_average_reference_handles_mixing_and_draws() {
    assert_eq!(sample(&[0.0, 0.0], &[0.9, 0.1], 0.01), [1.0, 0.0]);
    assert_eq!(sample(&[0.0, 0.0], &[0.1, 0.9], 0.01), [0.0, 1.0]);
    assert_eq!(sample(&[0.0, 0.0], &[0.5, 0.5], 0.01), [1.0, 0.0]);
    assert_eq!(sample(&[0.0, -1000.0], &[0.1, 0.999], 0.2), [0.0, 1.0]);
    let a = probabilities(&[2.0, 0.0], 0.0);
    let b = probabilities(&[0.0, 0.0], 0.0);
    let mean = (a[0] + b[0]) / 2.0;
    assert!((mean - 0.6903985389889412).abs() < 1e-12);
    assert!((mean - probabilities(&[1.0, 0.0], 0.0)[0]).abs() > 0.04);
}

fn snapshot(core: &DreamerCore) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for session in [&core.world_train, &core.behavior_train, &core.behavior_slow] {
        assert_eq!(session.adam_step_count(), 0);
        let names = session.param_names();
        result.extend(session.read_params(&names));
        for name in names
            .into_iter()
            .filter(|name| session.has_param_grad(name))
        {
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_adam_m(name, &mut values);
            result.push(values.clone());
            session.read_adam_v(name, &mut values);
            result.push(values);
        }
    }
    result
}

#[test]
#[ignore = "requires separately declared guarded actor posterior average qualification"]
fn posterior_average_native_values_state_rng_and_frozen_guards() {
    let mut config = super::super::latent_shape_tests::config();
    config.actor_unimix = 0.1;
    let make = || {
        let mut core = DreamerCore::new(config.clone()).unwrap();
        assert_eq!(
            core.gpu_device().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!core.gpu_device().is_software_emulated);
        let weights: Vec<_> = (0..config.network().units * 3)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.09)
            .collect();
        core.behavior_train
            .set_parameter("behavior.actor.out.weight", &weights);
        VectorCore::new(core, 2)
    };
    let mut ordinary = make();
    let mut average = make();
    let initial = snapshot(&average.learner);
    assert_eq!(snapshot(&ordinary.learner), initial);
    average
        .set_actor_posterior_samples_for_evaluation(8)
        .unwrap();
    assert!(average.use_greedy_posterior_for_evaluation().is_err());
    assert!(average.set_action_temperature_for_evaluation(1.0).is_err());
    for scheduled in [false, true] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if scheduled {
                    average.learn_scheduled(1);
                } else {
                    average.learn();
                }
            }))
            .is_err()
        );
    }
    let mut reference = build_session(
        &behavior::build_actor_inference_graph(&config, 16),
        &average.learner.gpu,
        Mode::Inference,
        false,
    );
    share_matching(
        &mut average.learner.behavior_train,
        &mut reference,
        "behavior.actor.",
    );
    let names = average.policy.param_names();
    assert_eq!(names.len(), 11);
    assert_eq!(
        average.policy.read_params(&names),
        average.learner.behavior_train.read_params(&names)
    );
    let size = config.network();
    let mut varied = false;
    for time in 0..4 {
        for core in [&mut ordinary, &mut average] {
            core.ingest(
                (0..2)
                    .map(|id| {
                        (
                            id,
                            Observation::rgb(
                                (0..config.observation_dim())
                                    .map(|i| ((i + id * 11 + time * 7) % 31) as f32 / 31.0)
                                    .collect(),
                                32,
                            ),
                            FrameFlags {
                                is_first: time == 0,
                                ..Default::default()
                            },
                            Reward::default(),
                        )
                    })
                    .collect(),
            );
        }
        assert!(
            average
                .set_actor_posterior_samples_for_evaluation(8)
                .is_err()
        );
        for id in 0..2 {
            assert_eq!(ordinary.feature(id), average.feature(id));
        }
        let mask = [true, false, true, false, true, true];
        let mask = (time % 2 == 1).then_some(mask.as_slice());
        let mode = if time % 2 == 0 {
            ActionMode::Greedy
        } else {
            ActionMode::Sample
        };
        let policy_draws: Vec<_> = average
            .streams
            .iter()
            .map(|stream| {
                if mode == ActionMode::Sample {
                    stream.policy_rng.clone().random::<f32>()
                } else {
                    0.0
                }
            })
            .collect();
        let forced = [Some(time % 3), Some((time + 1) % 3)];
        assert_eq!(
            ordinary.act_inner(mode, Some(&forced), mask),
            average.act_inner(mode, Some(&forced), mask)
        );
        for (a, b) in ordinary.streams.iter().zip(&average.streams) {
            assert_eq!(
                a.posterior_rng.clone().random::<u64>(),
                b.posterior_rng.clone().random::<u64>()
            );
            assert_eq!(
                a.policy_rng.clone().random::<u64>(),
                b.policy_rng.clone().random::<u64>()
            );
        }
        let mut posterior = vec![0.0; 2 * size.stoch * size.classes];
        average.observe.read_output_by_index(1, &mut posterior);
        let draws = average
            .actor_posterior_average
            .as_ref()
            .unwrap()
            .draws
            .clone();
        let mut states = Vec::new();
        for id in 0..2 {
            let state = average.feature(id);
            for particle in 0..8 {
                states.extend_from_slice(&state[..size.deter]);
                for variable in 0..size.stoch {
                    let q = (id * size.stoch + variable) * size.classes;
                    let u = ((id * 8 + particle) * size.stoch + variable) * size.classes;
                    states.extend(sample(
                        &posterior[q..q + size.classes],
                        &draws[u..u + size.classes],
                        config.unimix,
                    ));
                }
            }
        }
        reference.set_input("feature", &states);
        reference.step();
        reference.wait();
        let mut individual = vec![0.0; 16 * 3];
        reference.read_output_by_index(0, &mut individual);
        let mut actual = vec![0.0; 6];
        average.policy.read_output_by_index(0, &mut actual);
        let mut expected_actions = Vec::new();
        for id in 0..2 {
            let distributions: Vec<_> = individual[id * 24..(id + 1) * 24]
                .chunks(3)
                .map(|row| probabilities(row, 0.0))
                .collect();
            varied |= distributions[1..]
                .iter()
                .any(|p| (p[0] - distributions[0][0]).abs() > 1e-4);
            let mean: Vec<_> = (0..3)
                .map(|a| distributions.iter().map(|p| p[a]).sum::<f64>() / 8.0)
                .collect();
            for (&logit, &probability) in actual[id * 3..(id + 1) * 3].iter().zip(&mean) {
                assert!((f64::from(logit).exp() - probability).abs() < 2e-4);
            }
            let allowed = |a: usize| mask.is_none_or(|mask| mask[id * 3 + a]);
            let total: f64 = (0..3).filter(|&a| allowed(a)).map(|a| mean[a]).sum();
            let count = (0..3).filter(|&a| allowed(a)).count();
            let mix = f64::from(config.actor_unimix);
            let mixed: Vec<_> = (0..3)
                .map(|a| {
                    if allowed(a) {
                        mean[a] / total * (1.0 - mix) + mix / count as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let chosen = if mode == ActionMode::Greedy {
                mixed
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .unwrap()
                    .0
            } else {
                let mut cumulative = 0.0;
                mixed
                    .iter()
                    .position(|p| {
                        cumulative += p;
                        f64::from(policy_draws[id]) < cumulative
                    })
                    .unwrap_or(2)
            };
            expected_actions.push(chosen as f32);
        }
        let buffer = average
            .acting
            .actions(&average.policy, &policy_draws, mode, mask, 1.0);
        let mut selected = vec![0.0; 2];
        average
            .learner
            .readback
            .read_regions(&mut [(buffer, &mut selected)]);
        assert_eq!(selected, expected_actions);
        assert_eq!(
            ordinary.learner.environment_step,
            average.learner.environment_step
        );
        assert_eq!(average.learner.learner_step, 0);
    }
    assert!(
        varied,
        "fixture must exercise nonuniform particle-dependent actor outputs"
    );
    assert_eq!(snapshot(&average.learner), initial);
    assert_eq!(snapshot(&ordinary.learner), initial);
    eprintln!(
        "Actor posterior average: 8 samples, 2 streams, 4 synthetic decisions, independent sampling/means/actions, identical forced-action belief and original RNGs, frozen weights/moments; zero learner updates/Atari actions"
    );
}
