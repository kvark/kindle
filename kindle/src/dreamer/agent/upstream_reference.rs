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
