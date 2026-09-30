//! Export actual native learner steps for comparison with upstream Agent.loss.
//! Only fixed-batch construction bypasses the production path, not its targets.

use super::*;
use std::collections::BTreeMap;

fn save_json(path: &Path, value: &impl serde::Serialize) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(std::io::BufWriter::new(file), value).unwrap();
}

fn checkpoint(core: &mut DreamerCore, path: &Path) {
    std::fs::create_dir(path).unwrap();
    core.world_train
        .save_checkpoint(&path.join("world.safetensors"))
        .unwrap();
    core.behavior_train
        .save_checkpoint(&path.join("behavior.safetensors"))
        .unwrap();
    core.behavior_slow
        .save_checkpoint(&path.join("slow.safetensors"))
        .unwrap();
}

fn gradients(session: &Session, prefix: &str) -> BTreeMap<String, Vec<f32>> {
    session
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .map(|name| {
            assert!(session.has_param_grad(name), "{name}");
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut values);
            assert!(values.iter().all(|x| x.is_finite()), "{name}");
            (name.to_owned(), values)
        })
        .collect()
}

fn weights(session: &Session) -> BTreeMap<String, Vec<f32>> {
    let names = session
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with("world."))
        .collect::<Vec<_>>();
    names
        .iter()
        .map(|name| (*name).to_owned())
        .zip(session.read_params(&names))
        .collect()
}

fn raw_world_gradients(
    core: &mut DreamerCore,
    before: BTreeMap<String, Vec<f32>>,
) -> BTreeMap<String, Vec<f32>> {
    // A single full-BPTT batch leaves its complete inputs resident. Replay its
    // backward pass at pre-update weights, without touching optimizer state.
    let after = weights(&core.world_train);
    for (name, values) in &before {
        core.world_train.set_parameter(name, values);
    }
    core.world_train.clear_optimizer();
    core.world_train.step();
    core.world_train.wait();
    let result = gradients(&core.world_train, "world.");
    for (name, values) in &after {
        core.world_train.set_parameter(name, values);
    }
    result
}

fn raw_behavior_gradients(
    core: &mut DreamerCore,
    batch: &BehaviorTrainingBatch,
) -> BTreeMap<String, Vec<f32>> {
    // The production optimizer clips in place. An extra diagnostic backward
    // pass reads raw gradients without changing parameters or optimizer state.
    // Imagined features are already resident from imagine_and_target().
    for (name, values) in [
        ("action_target", &batch.action_target),
        ("imagined_weight", &batch.imagined_weight),
        ("imagined_value_target", &batch.imagined_value_target),
        ("imagined_slow_target", &batch.imagined_slow_target),
        ("replay_feature", &batch.replay_feature),
        ("replay_weight", &batch.replay_weight),
        ("replay_value_target", &batch.replay_value_target),
        ("replay_slow_target", &batch.replay_slow_target),
    ] {
        core.behavior_train.set_input(name, values);
    }
    core.behavior_train.clear_optimizer();
    core.behavior_train.step();
    core.behavior_train.wait();
    gradients(&core.behavior_train, "behavior.")
}

fn fixture(config: &DreamerConfig, rng: &mut StdRng) -> SequenceBatch {
    let (rows, length, network) = (config.batch_size, config.batch_length, config.network());
    let mut batch = SequenceBatch::default();
    batch.initial_deter = (0..rows * network.deter)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();
    batch.initial_stoch = vec![0.0; rows * network.stoch * network.classes];
    for state in batch.initial_stoch.chunks_exact_mut(network.classes) {
        state[rng.random_range(0..network.classes)] = 1.0;
    }
    for time in 0..length {
        batch.observations.push(
            (0..rows * config.observation_dim())
                .map(|_| f32::from(rng.random::<u8>()) / 255.0 - 0.5)
                .collect(),
        );
        let flags = (0..rows)
            .map(|row| FrameFlags {
                is_first: matches!((time, row), (0, 0) | (2, 0) | (3, 1)),
                is_last: matches!((time, row), (1, 0) | (2, 1)),
                is_terminal: (time, row) == (2, 1),
            })
            .collect::<Vec<_>>();
        let mut actions = vec![0.0; rows * config.action_count];
        for (row, flag) in flags.iter().enumerate() {
            if !flag.is_first {
                actions[row * config.action_count + rng.random_range(0..config.action_count)] = 1.0;
            }
        }
        batch.previous_actions.push(actions);
        batch.rewards.push(
            (0..rows)
                .map(|row| {
                    if flags[row].is_first {
                        0.0
                    } else {
                        ((time + row) % 3) as f32 - 1.0
                    }
                })
                .collect(),
        );
        batch.flags.push(flags);
    }
    batch
}

#[test]
#[ignore = "requires GPU; exports four fixed-batch steps to KINDLE_DREAMER_STEP_REFERENCE"]
fn export_upstream_fixed_batch_reference() {
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_STEP_REFERENCE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let config = DreamerConfig {
        model_size: crate::ModelSize::Size1M,
        observation_kind: crate::ObservationKind::Rgb64,
        batch_size: 2,
        batch_length: 4,
        world_backprop_length: 4,
        imagination_length: 3,
        learning_rate_warmup: 2,
        seed: 103,
        ..DreamerConfig::new(18)
    };
    save_json(&root.join("config.json"), &config);
    let mut core = DreamerCore::new(config.clone()).unwrap();
    let check_device = |core: &DreamerCore| {
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            let device = core.world_train.device_information();
            assert_eq!(device.device_name, expected);
            assert!(!device.is_software_emulated);
            let memory = core.gpu.memory_stats();
            assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        }
    };
    check_device(&core);
    checkpoint(&mut core, &root.join("initial"));
    let mut rng = StdRng::seed_from_u64(701);
    let size = config.network();
    let starts = config.batch_size * config.batch_length;
    for step in 0..4 {
        let batch = fixture(&config, &mut rng);
        // Record draws using cloned streams. Production methods consume the
        // originals; no injected samples or altered RNG are used by the native.
        let mut posterior_rng = core.rngs.train_posterior.clone();
        let posterior_uniforms = (0..config.batch_length)
            .map(|_| {
                (0..config.batch_size * size.stoch * size.classes)
                    .map(|_| posterior_rng.random::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut imagination_rng = core.rngs.imagination.clone();
        let mut action_uniforms = Vec::new();
        let mut latent_uniforms = Vec::new();
        for _ in 0..config.imagination_length {
            action_uniforms.push(
                (0..starts * config.action_count)
                    .map(|_| imagination_rng.random::<f32>())
                    .collect::<Vec<_>>(),
            );
            latent_uniforms.push(
                (0..starts * size.stoch * size.classes)
                    .map(|_| imagination_rng.random::<f32>())
                    .collect::<Vec<_>>(),
            );
        }
        let posterior = core.sample_posterior_batch(&batch);
        let targets = core.imagine_and_target(&batch, &posterior);
        let before = weights(&core.world_train);
        let world = core.train_world(&batch, &posterior, &targets);
        let mut grads = raw_world_gradients(&mut core, before);
        core.sync_world_inference();
        grads.extend(raw_behavior_gradients(&mut core, &targets));
        let behavior = core.train_behavior(&targets);
        core.sync_behavior_inference();
        core.learner_step += 1;
        let path = root.join(format!("step{step}"));
        checkpoint(&mut core, &path);
        save_json(
            &path.join("batch.json"),
            &serde_json::json!({
                "initial_deter": batch.initial_deter, "initial_stoch": batch.initial_stoch,
                "observations": batch.observations, "actions": batch.previous_actions,
                "rewards": batch.rewards, "flags": batch.flags,
                "posterior_uniforms": posterior_uniforms, "action_uniforms": action_uniforms,
                "latent_uniforms": latent_uniforms,
            }),
        );
        save_json(
            &path.join("outputs.json"),
            &serde_json::json!({
                "deter": posterior.deter, "stoch": posterior.stoch,
                "world": world, "behavior": behavior,
                "return_normalizer": core.return_normalizer.state(),
                "action_target": targets.action_target,
                "imagined_weight": targets.imagined_weight,
                "imagined_value_target": targets.imagined_value_target,
                "imagined_slow_target": targets.imagined_slow_target,
                "replay_value_target": targets.replay_value_target,
                "replay_slow_target": targets.replay_slow_target,
                "replay_weight": targets.replay_weight,
            }),
        );
        save_json(&path.join("gradients.json"), &grads);
        check_device(&core);
    }
}

#[test]
#[ignore = "requires GPU timestamps; synthetic batch only, no environment or replay resume"]
fn profile_fixed_batch_checkpoint() {
    use meganeura::data::safetensors::SafeTensorsModel;
    let source = std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap();
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut core = DreamerCore::restore(source).unwrap();
    let config = core.config.clone();
    assert_eq!(config.observation_kind, crate::ObservationKind::Rgb64);
    let memory = core.gpu_memory_budget();
    assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        assert_eq!(core.gpu_device().device_name, expected);
        assert!(!core.gpu_device().is_software_emulated);
    }
    let mut rng = StdRng::seed_from_u64(701);
    let batch = fixture(&config, &mut rng);
    let posterior = core.sample_posterior_batch(&batch);
    let targets = core.imagine_and_target(&batch, &posterior);
    core.train_world(&batch, &posterior, &targets);
    core.sync_world_inference();
    core.train_behavior(&targets);
    core.sync_behavior_inference();
    core.learner_step += 1;
    checkpoint(&mut core, &root.join("before"));
    core.profile_sessions(root.join("profiles")).unwrap();
    checkpoint(&mut core, &root.join("after"));
    for component in ["world", "behavior", "slow"] {
        let before =
            SafeTensorsModel::load(root.join("before").join(format!("{component}.safetensors")))
                .unwrap();
        let after =
            SafeTensorsModel::load(root.join("after").join(format!("{component}.safetensors")))
                .unwrap();
        assert_eq!(before.tensor_info().len(), after.tensor_info().len());
        for name in before.tensor_info().keys() {
            assert_eq!(
                before.tensor_f32(name).unwrap(),
                after.tensor_f32(name).unwrap(),
                "{name}"
            );
        }
    }
    let memory = core.gpu_memory_budget();
    assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
    save_json(&root.join("config.json"), &config);
}

#[test]
#[ignore = "paired synthetic full-update timing and split-reduction parity; no gameplay"]
fn compare_split_world_updates() {
    let source =
        std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap());
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut control = DreamerCore::restore(&source).unwrap();
    let mut candidate = DreamerCore::restore(&source).unwrap();
    let graph = world::build_training_graph(
        &world_training_config(&control.config),
        control.config.world_backprop_length,
    );
    let mut unsplit = meganeura::build(
        &graph,
        meganeura::SessionConfig {
            gpu: Some(Arc::clone(&control.gpu)),
            skip_full_optimize: control.config.skip_full_optimize,
            ..Default::default()
        },
    )
    .0;
    checkpoint::load_session(&mut unsplit, &source.join(CHECKPOINT_WORLD)).unwrap();
    for target in [
        &mut control.world_posterior,
        &mut control.world_observe_live,
        &mut control.imagination,
        &mut control.world_transition_live,
        &mut control.world_heads_live,
    ] {
        share_matching(&mut unsplit, target, "world.");
    }
    share_matching(&mut control.behavior_train, &mut unsplit, "behavior.value.");
    unsplit.set_submission_chunks(4);
    control.world_train = unsplit;
    let extra_dispatches =
        candidate.world_train.plan().dispatches.len() - control.world_train.plan().dispatches.len();
    assert!(extra_dispatches >= 2);
    let check_device = |core: &DreamerCore| {
        assert_eq!(
            core.gpu_device().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!core.gpu_device().is_software_emulated);
        let memory = core.gpu_memory_budget();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
    };
    check_device(&control);
    check_device(&candidate);
    let mut rng = StdRng::seed_from_u64(701);
    let mut timing = [Vec::new(), Vec::new()];
    let mut numerical = Vec::new();
    for step in 0..10 {
        let batch = fixture(&control.config, &mut rng);
        let mut reports = [serde_json::Value::Null, serde_json::Value::Null];
        let mut raw = Vec::new();
        let mut cores = [(0, &mut control), (1, &mut candidate)];
        if step % 2 != 0 {
            cores.swap(0, 1);
        }
        for (index, core) in cores {
            let before = (step == 0).then(|| weights(&core.world_train));
            let start = std::time::Instant::now();
            let posterior = core.sample_posterior_batch(&batch);
            let targets = core.imagine_and_target(&batch, &posterior);
            let world = core.train_world(&batch, &posterior, &targets);
            core.sync_world_inference();
            let behavior = core.train_behavior(&targets);
            core.sync_behavior_inference();
            core.learner_step += 1;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if step >= 2 {
                timing[index].push(elapsed_ms);
            }
            reports[index] = serde_json::json!({"world": world, "behavior": behavior});
            if let Some(before) = before {
                raw.push(raw_world_gradients(core, before));
            }
        }
        if step == 0 {
            assert_eq!(
                reports[0], reports[1],
                "forward losses/targets before the first update"
            );
            assert_eq!(
                raw[0].keys().collect::<Vec<_>>(),
                raw[1].keys().collect::<Vec<_>>()
            );
            for (name, a) in &raw[0] {
                let b = &raw[1][name];
                assert_eq!(a.len(), b.len(), "{name}");
                let max_abs = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let error = a
                    .iter()
                    .zip(b)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
                assert!(b.iter().all(|x| x.is_finite()));
                assert!(
                    max_abs <= 1e-6 || error <= 2e-4 * norm,
                    "{name}: max_abs={max_abs:e}, relative_l2={:e}",
                    error / norm
                );
                numerical.push(serde_json::json!({"name": name, "max_abs": max_abs, "relative_l2": error / norm.max(1e-30)}));
            }
            checkpoint(&mut control, &root.join("control-first"));
            checkpoint(&mut candidate, &root.join("candidate-first"));
        }
        save_json(&root.join(format!("step{step}.json")), &reports);
    }
    checkpoint(&mut control, &root.join("control-final"));
    checkpoint(&mut candidate, &root.join("candidate-final"));
    let mut state_checks = 0;
    for component in ["world", "behavior", "slow"] {
        use meganeura::data::safetensors::SafeTensorsModel;
        let load = |arm: &str, stage: &str| {
            SafeTensorsModel::load(root.join(format!("{arm}-{stage}/{component}.safetensors")))
                .unwrap()
        };
        let a = load("control", "first");
        let b = load("candidate", "first");
        assert_eq!(
            a.tensor_info()
                .keys()
                .collect::<std::collections::BTreeSet<_>>(),
            b.tensor_info()
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
        );
        for name in a.tensor_info().keys() {
            let a = a.tensor_f32(name).unwrap();
            let b = b.tensor_f32(name).unwrap();
            assert_eq!(a.len(), b.len(), "{component}.{name}");
            let max_abs = a
                .iter()
                .zip(b.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let error = a
                .iter()
                .zip(b.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            assert!(
                max_abs <= 1e-6 || error <= 2e-4 * norm,
                "{component}.{name}: {max_abs:e}"
            );
            assert!(b.iter().all(|x| x.is_finite()));
            state_checks += 1;
        }
        for arm in ["control", "candidate"] {
            let state = load(arm, "final");
            for name in state.tensor_info().keys() {
                assert!(
                    state
                        .tensor_f32(name)
                        .unwrap()
                        .iter()
                        .all(|x| x.is_finite())
                );
            }
        }
    }
    check_device(&control);
    check_device(&candidate);
    save_json(
        &root.join("comparison.json"),
        &serde_json::json!({
            "config": control.config, "extra_dispatches": extra_dispatches, "whole_update_wall_ms": timing,
        "raw_gradients": numerical, "first_step_state_tensors_checked": state_checks,
        "control_optimizer_step": control.world_train.adam_step_count(),
            "candidate_optimizer_step": candidate.world_train.adam_step_count(),
            "limits": "resident synthetic batches; replay sampling excluded; first-step numerical check, not bitwise subsequent trajectories or learning parity"
        }),
    );
}
