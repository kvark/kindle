use super::*;

fn export_gradients(path: &std::path::Path, values: &BTreeMap<String, Vec<f32>>) {
    std::fs::create_dir(path).unwrap();
    let mut entries = Vec::new();
    for (i, (name, values)) in values.iter().enumerate() {
        assert!(values.iter().all(|v| v.is_finite()));
        let file = format!("{i}.bin");
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path.join(&file), bytes).unwrap();
        entries.push(serde_json::json!({"name": name, "file": file, "elements": values.len()}));
    }
    save_json(&path.join("index.json"), &entries);
}

#[test]
#[ignore = "Declared actual-world copy-composition raw-gradient qualification"]
fn actual_world() {
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    let source =
        std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut core = restore_profile_core(&source);
    assert_eq!(core.gpu_device().device_name, "NVIDIA GeForce RTX 5080");
    assert!(!core.gpu_device().is_software_emulated);
    let headroom = |core: &DreamerCore| {
        let m = core.gpu_memory_budget();
        let n = m.budget_bytes.saturating_sub(m.usage_bytes);
        assert!(n >= 2 << 30);
        n
    };
    let mut minimum = headroom(&core);
    assert_eq!(core.config.batch_size, 32);
    assert_eq!(core.config.batch_length, 16);
    assert_eq!(core.config.world_backprop_length, 16);
    assert!(core.world_posterior.is_none());
    let mut rng = StdRng::seed_from_u64(701);
    let batch = fixture(&core.config, &mut rng);
    checkpoint(&mut core, &root.join("initial"));
    let before = weights(&core.world_train);
    let posterior = core.sample_posterior_batch(&batch);
    let mut targets = core.imagine_and_target(&batch, &posterior);
    let world = core.train_world(&batch, &posterior, &mut targets);
    core.sync_world_inference();
    let behavior = core.train_behavior(&targets);
    core.sync_behavior_inference();
    core.learner_step += 1;
    checkpoint(&mut core, &root.join("first"));
    save_json(
        &root.join("report.json"),
        &serde_json::json!({"world":world,"behavior":behavior}),
    );
    save_json(&root.join("world-plan.json"), core.world_train.plan());
    save_json(
        &root.join("world-pipelines.json"),
        &core.world_train.dispatch_pipeline_keys(),
    );
    let after = weights(&core.world_train);
    for (name, values) in &before {
        core.world_train.set_parameter(name, values);
    }
    core.world_train.clear_optimizer();
    checkpoint(&mut core, &root.join("before_raw"));
    let rows = core.config.batch_size;
    let bins = core.config.value_bins;
    let mut counts = Vec::new();
    for variant in 0..2 {
        let reward_targets: Vec<_> = (0..16)
            .map(|time| {
                let mut values = vec![0.0; rows * bins];
                for row in 0..rows {
                    let value = batch.rewards[time][row] + if variant == 0 { 0.0 } else { 0.25 };
                    core.bins
                        .encode(value, &mut values[row * bins..(row + 1) * bins]);
                }
                values
            })
            .collect();
        for (time, values) in reward_targets.iter().enumerate() {
            core.world_train
                .set_input(&format!("reward_target_{time}"), values);
        }
        core.world_train.step();
        core.world_train.wait();
        let whole = gradients(&core.world_train, "world.");
        export_gradients(&root.join(format!("whole-{variant}")), &whole);
        for time in 0..16 {
            core.world_train.set_input(
                &format!("reward_target_{time}"),
                &vec![f32::NAN; rows * bins],
            );
        }
        core.world_train.prepare_step();
        core.world_train.wait();
        for (time, values) in reward_targets.iter().enumerate() {
            core.world_train
                .set_input(&format!("reward_target_{time}"), values);
        }
        core.world_train.finish_prepared_step();
        core.world_train.wait();
        let staged = gradients(&core.world_train, "world.");
        assert!(whole.keys().eq(staged.keys()));
        for (name, a) in &whole {
            let b = &staged[name];
            assert!(
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{name}"
            );
        }
        counts.push(serde_json::json!({"tensors":whole.len(),"elements":whole.values().map(Vec::len).sum::<usize>()}));
        export_gradients(&root.join(format!("staged-{variant}")), &staged);
        minimum = minimum.min(headroom(&core));
    }
    checkpoint(&mut core, &root.join("after_raw"));
    for (name, values) in after {
        core.world_train.set_parameter(&name, &values);
    }
    checkpoint(&mut core, &root.join("restored"));
    save_json(
        &root.join("result.json"),
        &serde_json::json!({"status":"complete","raw_backward_passes":4,"initializer_updates":1,"game_actions":0,"variants":counts,"minimum_sampled_headroom_bytes":minimum}),
    );
}
