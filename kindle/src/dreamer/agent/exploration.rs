use super::*;
use crate::dreamer::exploration::Disagreement;

fn config(scale: f32) -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = crate::ObservationKind::Rgb64;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = scale;
    config
}

fn fixture(config: &DreamerConfig) -> SequenceBatch {
    let size = config.network();
    let mut batch = SequenceBatch::default();
    batch.initial_deter = (0..config.batch_size * size.deter)
        .map(|i| (i % 7) as f32 * 0.03)
        .collect();
    batch.initial_stoch = (0..config.batch_size * size.stoch * size.classes)
        .map(|i| f32::from(i % size.classes == 0))
        .collect();
    for time in 0..config.batch_length {
        batch.observations.push(
            (0..config.batch_size * config.observation_dim())
                .map(|i| ((i * 7 + time * 13) % 256) as f32 / 255.0 - 0.5)
                .collect(),
        );
        batch.previous_actions.push(
            (0..config.batch_size * config.action_count)
                .map(|i| {
                    f32::from(
                        i % config.action_count
                            == (time + i / config.action_count) % config.action_count,
                    )
                })
                .collect(),
        );
        batch.rewards.push(vec![0.0; config.batch_size]);
        batch.flags.push(
            (0..config.batch_size)
                .map(|row| FrameFlags {
                    is_first: matches!((time, row), (0, 0) | (2, 1)),
                    is_last: matches!((time, row), (1, 1) | (3, 0)),
                    is_terminal: (time, row) == (1, 1),
                })
                .collect(),
        );
    }
    batch
}

fn check_device(core: &DreamerCore) {
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        assert_eq!(core.gpu_device().device_name, expected);
        assert!(!core.gpu_device().is_software_emulated);
        let memory = core.gpu_memory_budget();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
    }
}

fn direct_bonus(core: &mut DreamerCore, states: &[f32], actions: &[f32]) -> Vec<f32> {
    direct_bonus_precision(core, states, actions, meganeura::CoopPolicy::Auto).0
}

fn direct_bonus_precision(
    core: &mut DreamerCore,
    states: &[f32],
    actions: &[f32],
    coop: meganeura::CoopPolicy,
) -> (Vec<f32>, Vec<String>) {
    let rows = actions.len() / core.config.action_count;
    let mut graph = meganeura::Graph::new();
    let head = Disagreement::new(&mut graph, &core.config);
    let state = graph.input("state", &[rows, core.config.feature_dim()]);
    let action = graph.input("action", &[rows, core.config.action_count]);
    let bonus = head.bonus(&mut graph, state, action);
    graph.set_outputs(vec![bonus]);
    let mut session = meganeura::build(
        &graph,
        meganeura::SessionConfig {
            mode: Mode::Inference,
            gpu: Some(Arc::clone(&core.gpu)),
            runtime: meganeura::SessionOptions {
                coop,
                gpu_timing: meganeura::GpuOptions::from_env().timing,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0;
    share_matching(&mut core.world_train, &mut session, "world.exploration.");
    session.set_input("state", states);
    session.set_input("action", actions);
    session.step();
    session.wait();
    let mut result = vec![0.0; rows];
    session.read_output_by_index(0, &mut result);
    (result, session.dispatch_pipeline_keys())
}

#[test]
#[ignore = "read-only GPU diagnostic; all-action bonus on saved real posterior features"]
fn probe_disagreement_checkpoint() {
    use meganeura::data::safetensors::SafeTensorsModel;
    let source = Path::new(&std::env::var("KINDLE_DISAG_PROBE_SOURCE").unwrap()).to_owned();
    let states =
        SafeTensorsModel::load(std::env::var("KINDLE_DISAG_PROBE_STATES").unwrap().into()).unwrap();
    let output = Path::new(&std::env::var("KINDLE_DISAG_PROBE_OUTPUT").unwrap()).to_owned();
    assert!(!output.exists());
    let mut core = DreamerCore::restore(&source).unwrap();
    check_device(&core);
    assert!(core.config.uses_disagreement());
    let shape = &states.tensor_info()["states"].shape;
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[1], core.config.feature_dim());
    let samples = states.tensor_f32("states").unwrap();
    let mut expanded = Vec::new();
    let mut actions = Vec::new();
    for state in samples.chunks_exact(shape[1]) {
        for action in 0..core.config.action_count {
            expanded.extend_from_slice(state);
            actions.extend((0..core.config.action_count).map(|index| f32::from(index == action)));
        }
    }
    let (bonus, pipelines) =
        direct_bonus_precision(&mut core, &expanded, &actions, meganeura::CoopPolicy::Auto);
    let (bonus_native_f32, pipelines_native_f32) = direct_bonus_precision(
        &mut core,
        &expanded,
        &actions,
        meganeura::CoopPolicy::NativeF32,
    );
    assert!(
        bonus
            .iter()
            .chain(&bonus_native_f32)
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    let after = output.with_extension("checkpoint");
    core.save_checkpoint(&after).unwrap();
    let mut tensors = 0;
    for component in [CHECKPOINT_WORLD, CHECKPOINT_BEHAVIOR, CHECKPOINT_SLOW_VALUE] {
        let before = SafeTensorsModel::load(source.join(component)).unwrap();
        let after = SafeTensorsModel::load(after.join(component)).unwrap();
        assert_eq!(before.tensor_info().len(), after.tensor_info().len());
        for name in before.tensor_info().keys() {
            assert_eq!(
                before.tensor_f32(name).unwrap(),
                after.tensor_f32(name).unwrap(),
                "{name}"
            );
            tensors += 1;
        }
    }
    check_device(&core);
    let result = serde_json::json!({"status":"complete", "samples":shape[0],
        "actions":core.config.action_count, "bonus":bonus, "unchanged_tensors":tensors,
        "learner_updates":0, "bonus_native_f32":bonus_native_f32,
        "pipelines":pipelines, "pipelines_native_f32":pipelines_native_f32});
    fs::write(output, serde_json::to_vec(&result).unwrap()).unwrap();
}

#[test]
#[ignore = "requires GPU; replay/imagination action alignment, resets and intrinsic advantages"]
fn tiny_disagreement_targets_are_aligned_and_create_advantages() {
    let config = config(1.0);
    let batch = fixture(&config);
    let mut core = DreamerCore::new(config.clone()).unwrap();
    check_device(&core);
    let posterior = core.sample_posterior_batch(&batch);
    let rows = config.batch_size * config.batch_length;
    let width = config.network().stoch * config.network().classes;
    let mut previous_states = Vec::new();
    let mut actions = Vec::new();
    for time in 0..config.batch_length {
        let mut deter = if time == 0 {
            batch.initial_deter.clone()
        } else {
            posterior.deter[time - 1].clone()
        };
        let mut stoch = if time == 0 {
            batch.initial_stoch.clone()
        } else {
            posterior.stoch[time - 1].clone()
        };
        let mut action = batch.previous_actions[time].clone();
        for row in 0..config.batch_size {
            if batch.flags[time][row].is_first {
                deter[row * config.network().deter..(row + 1) * config.network().deter].fill(0.0);
                stoch[row * width..(row + 1) * width].fill(0.0);
                action[row * config.action_count..(row + 1) * config.action_count].fill(0.0);
            }
        }
        previous_states.extend(join_features(&deter, &stoch, config.batch_size, &config));
        actions.extend(action);
    }
    let mut expected = direct_bonus(&mut core, &previous_states, &actions);
    for (bonus, flags) in expected.iter_mut().zip(batch.flags.iter().flatten()) {
        if flags.is_first {
            *bonus = 0.0;
        }
    }
    let mut actual = vec![0.0; rows];
    core.world_posterior
        .read_output_by_index(world::POSTERIOR_BONUS, &mut actual);
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 3e-6)
    );
    let targets = core.imagine_and_target(&batch, &posterior);
    assert!((targets.replay_intrinsic_reward_mean - mean(&expected)).abs() < 3e-6);
    let imagined_rows = rows * config.imagination_length;
    let mut states = vec![0.0; imagined_rows * config.feature_dim()];
    let mut actions = vec![0.0; imagined_rows * config.action_count];
    core.imagination
        .read_output_by_index(world::IMAGINATION_FEATURE, &mut states);
    core.imagination
        .read_output_by_index(world::IMAGINATION_ACTION, &mut actions);
    let imagined = direct_bonus(&mut core, &states, &actions);
    assert!((targets.imagined_intrinsic_reward_mean - mean(&imagined)).abs() < 3e-6);
    // Initially the observed-reward head predicts exactly zero. Every arrival's
    // imagined reward is therefore the matching state/action bonus, not r_t.
    assert!((targets.imagined_reward_mean - mean(&imagined)).abs() < 3e-6);
    let horizon = config.imagination_length;
    let mut continuation = vec![0.0; rows * (horizon + 1)];
    core.imagination
        .read_output_by_index(world::IMAGINATION_CONTINUATION, &mut continuation);
    let mut returns = vec![0.0; imagined_rows];
    let mut encoded = vec![0.0; config.value_bins];
    for start in 0..rows {
        // Reward/value heads are zero at initialization. Independently expand
        // the return recurrence to catch a one-step shift of the action bonus.
        let mut next = 0.0;
        for time in (0..horizon).rev() {
            let row = time * rows + start;
            next = imagined[row] + continuation[(time + 1) * rows + start] * config.lambda * next;
            returns[row] = next;
            core.bins.encode(next, &mut encoded);
            assert!(
                encoded
                    .iter()
                    .zip(
                        &targets.imagined_value_target
                            [row * config.value_bins..(row + 1) * config.value_bins]
                    )
                    .all(|(a, b)| (a - b).abs() < 3e-6)
            );
        }
        let mut weight = 1.0;
        for time in 0..horizon {
            let row = time * rows + start;
            weight *= continuation[row];
            for action in 0..config.action_count {
                let index = row * config.action_count + action;
                let expected = actions[index] * weight * returns[row] / targets.return_scale;
                assert!((targets.action_target[index] - expected).abs() < 3e-6);
            }
        }
    }
    for stream in 0..config.batch_size {
        let mut next = returns[(config.batch_length - 1) * config.batch_size + stream];
        for time in (0..config.batch_length - 1).rev() {
            let arrival = (time + 1) * config.batch_size + stream;
            let flags = batch.flags[time + 1][stream];
            let live = if flags.is_terminal {
                0.0
            } else {
                config.continuation_discount()
            };
            let trace = if flags.is_last { 0.0 } else { config.lambda };
            next =
                expected[arrival] + (1.0 - trace) * live * returns[arrival] + live * trace * next;
            let row = time * config.batch_size + stream;
            core.bins.encode(next, &mut encoded);
            assert!(
                encoded
                    .iter()
                    .zip(
                        &targets.replay_value_target
                            [row * config.value_bins..(row + 1) * config.value_bins]
                    )
                    .all(|(a, b)| (a - b).abs() < 3e-6)
            );
            assert_eq!(
                targets.replay_weight[row],
                f32::from(!batch.flags[time][stream].is_last)
            );
        }
    }
    assert_eq!(targets.replay_reward_target_mean, 0.0);
    assert_eq!(targets.replay_reward_prediction_mean, 0.0);
    assert!(targets.advantage_abs_mean > 0.0);
    let world = core.train_world(&batch, &posterior, &targets);
    let behavior = core.train_behavior(&targets);
    assert!(world.exploration_loss > 0.0 && world.exploration_loss.is_finite());
    assert_eq!(world.positive_reward_count, 0);
    assert!(behavior.imagined_intrinsic_reward_mean > 0.0);
    check_device(&core);
}

#[test]
#[ignore = "requires GPU; exact zero-coefficient baseline updates and RNG equivalence"]
fn tiny_disagreement_zero_scale_preserves_baseline_updates() {
    let mut plain = config(0.0);
    plain.disagreement_bonus = false;
    let mut baseline = DreamerCore::new(plain).unwrap();
    let mut zero = DreamerCore::with_gpu(config(0.0), Arc::clone(&baseline.gpu));
    let batch = fixture(&baseline.config);
    for _ in 0..4 {
        let update = |core: &mut DreamerCore| {
            let posterior = core.sample_posterior_batch(&batch);
            let targets = core.imagine_and_target(&batch, &posterior);
            let world = core.train_world(&batch, &posterior, &targets);
            core.sync_world_inference();
            let behavior = core.train_behavior(&targets);
            core.sync_behavior_inference();
            core.learner_step += 1;
            serde_json::to_value((world, behavior)).unwrap()
        };
        assert_eq!(update(&mut baseline), update(&mut zero));
        for (left, right) in [
            (&baseline.world_train, &zero.world_train),
            (&baseline.behavior_train, &zero.behavior_train),
            (&baseline.behavior_slow, &zero.behavior_slow),
        ] {
            assert_eq!(left.param_names(), right.param_names());
            let names = left.param_names();
            assert_eq!(left.read_params(&names), right.read_params(&names));
            let trained = names
                .into_iter()
                .filter(|name| left.has_param_grad(name))
                .collect::<Vec<_>>();
            if !trained.is_empty() {
                assert_eq!(
                    left.read_adam_states(&trained),
                    right.read_adam_states(&trained)
                );
                assert_eq!(left.adam_step_count(), right.adam_step_count());
            }
        }
        assert_eq!(
            baseline.rngs.train_posterior.clone().random::<u64>(),
            zero.rngs.train_posterior.clone().random::<u64>()
        );
        assert_eq!(
            baseline.rngs.imagination.clone().random::<u64>(),
            zero.rngs.imagination.clone().random::<u64>()
        );
    }
    check_device(&baseline);
}
