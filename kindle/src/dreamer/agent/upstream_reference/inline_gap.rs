use super::*;
// Diagnostic-only fixtures; full groups/work are restored before output checks.
use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub(super) fn output_fingerprint(session: &meganeura::Session, path: &Path) -> serde_json::Value {
    let outputs = session.read_buffers(&session.plan().output_buffers);
    let mut data = Vec::new();
    let mut offsets = vec![0];
    for output in outputs {
        assert!(output.iter().all(|x| x.is_finite()));
        for value in output {
            data.extend_from_slice(&value.to_le_bytes());
        }
        offsets.push(data.len());
    }
    let hash = format!("{:x}", Sha256::digest(&data));
    std::fs::write(path, &data).unwrap();
    serde_json::json!({"sha256": hash, "offsets": offsets, "bytes": data.len()})
}

pub(super) fn execute(session: &mut meganeura::Session) -> serde_json::Value {
    let cache_before = session.command_cache_counts();
    let started = std::time::Instant::now();
    session.step();
    session.wait();
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let timings: Vec<_> = session
        .gpu_timings()
        .into_iter()
        .map(|(label, duration)| (label, duration.as_secs_f64() * 1000.0))
        .collect();
    assert!(timings.iter().all(|(_, ms)| ms.is_finite() && *ms >= 0.0));
    let total_ms: f64 = timings.iter().map(|(_, ms)| ms).sum();
    serde_json::json!({"wall_ms": wall_ms, "total_ms": total_ms, "timings": timings,
        "cache_before": cache_before, "cache_after": session.command_cache_counts()})
}

#[test]
#[ignore = "counterfactual family attribution; masked outputs are not learner results"]
fn fixed_session_family_masks() {
    use meganeura::data::safetensors::SafeTensorsModel;
    let source = std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap();
    let root = PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    assert!(meganeura::GpuOptions::from_env().timing);
    assert_eq!(std::env::var("MEGANEURA_COMMAND_CACHE").unwrap(), "1");
    let mut core = restore_profile_core(Path::new(&source));
    assert_eq!(core.gpu_device().device_name, "NVIDIA GeForce RTX 5080");
    assert!(!core.gpu_device().is_software_emulated);
    let memory = core.gpu_memory_budget();
    let initial_headroom = memory.budget_bytes.saturating_sub(memory.usage_bytes);
    assert!(initial_headroom >= 2 << 30);
    core.world_train.set_submission_chunks(1);
    assert!(core.world_posterior.is_none(), "staged world required");
    let config = core.config.clone();
    save_json(&root.join("config.json"), &config);
    for (name, session) in [
        ("posterior", core.posterior_session()),
        ("imagination", &core.imagination),
        ("world_gradient", &core.world_train),
        ("behavior_gradient", &core.behavior_train),
        ("slow_value", &core.behavior_slow),
    ] {
        save_json(&root.join(format!("{name}-plan.json")), session.plan());
        save_json(
            &root.join(format!("{name}-pipelines.json")),
            &session.dispatch_pipeline_keys(),
        );
    }
    checkpoint(&mut core, &root.join("initial"));
    let mut rng = StdRng::seed_from_u64(701);
    let batch = fixture(&config, &mut rng);
    let posterior = core.sample_posterior_batch(&batch);
    let mut targets = core.imagine_and_target(&batch, &posterior);
    let world = core.train_world(&batch, &posterior, &mut targets);
    core.sync_world_inference();
    let behavior = core.train_behavior(&targets);
    core.sync_behavior_inference();
    core.learner_step += 1;
    save_json(
        &root.join("first-outputs.json"),
        &serde_json::json!({
            "deter": posterior.deter, "stoch": posterior.stoch,
            "action_target": targets.action_target,
            "imagined_weight": targets.imagined_weight,
            "imagined_value_target": targets.imagined_value_target,
            "imagined_slow_target": targets.imagined_slow_target,
            "replay_value_target": targets.replay_value_target,
            "replay_slow_target": targets.replay_slow_target,
            "replay_weight": targets.replay_weight,
        }),
    );
    save_json(
        &root.join("report.json"),
        &serde_json::json!({"world": world, "behavior": behavior}),
    );
    checkpoint(&mut core, &root.join("before"));
    for session in [&mut core.world_train, &mut core.behavior_train] {
        session.clear_optimizer();
        session.clear_grad_accumulate();
        session.disable_grad_clip();
    }
    let mut session_counts = BTreeMap::new();
    for (name, session) in [
        ("imagination", &mut core.imagination),
        ("slow_value", &mut core.behavior_slow),
        ("world_gradient", &mut core.world_train),
        ("behavior_gradient", &mut core.behavior_train),
    ] {
        session.set_submission_chunks(1);
        let path = root.join(name);
        std::fs::create_dir(&path).unwrap();
        let dispatches = session.plan().dispatches.len();
        let metadata: Vec<_> = session
            .plan()
            .dispatches
            .iter()
            .enumerate()
            .map(|(i, d)| {
                serde_json::json!({"index": i, "shader": format!("{:?}", d.shader),
                "label": d.label, "workgroups": d.workgroups, "origin": d.origin})
            })
            .collect();
        save_json(
            &path.join("metadata.json"),
            &serde_json::json!({
                "dispatches": metadata, "groups": session.diagnostic_groups(),
                "pipelines": session.dispatch_pipeline_keys(),
            }),
        );
        let mut baseline_before = Vec::new();
        for i in 0..72 {
            let sample = execute(session);
            assert_eq!(sample["timings"].as_array().unwrap().len(), 1);
            if i >= 8 {
                baseline_before.push(sample);
            }
        }
        let before = output_fingerprint(session, &path.join("before.f32"));
        let mut samples = Vec::with_capacity(3 * 8 * 16);
        for sweep in 0..3 {
            for position in 0..8 {
                let mask = (position + 3 * sweep) % 8;
                session.set_diagnostic_work_mask(7);
                execute(session);
                session.set_diagnostic_work_mask(mask);
                for repeat in 0..18 {
                    let mut sample = execute(session);
                    let timings = sample["timings"].as_array().unwrap();
                    assert_eq!(timings.len(), 1);
                    assert_eq!(timings[0][0], "cached_graph");
                    sample["mask"] = serde_json::json!(mask);
                    sample["sweep"] = serde_json::json!(sweep);
                    sample["repeat"] = serde_json::json!(repeat);
                    if repeat >= 2 {
                        assert_eq!(sample["cache_before"][0], sample["cache_after"][0]);
                        assert_eq!(
                            sample["cache_before"][1].as_u64().unwrap() + 1,
                            sample["cache_after"][1].as_u64().unwrap()
                        );
                        samples.push(sample);
                    }
                }
            }
            session.set_diagnostic_work_mask(7);
            execute(session);
            let after = output_fingerprint(session, &path.join(format!("sweep-{sweep}.f32")));
            assert_eq!(before, after, "{name} sweep {sweep} changed outputs");
            eprintln!("{name}: family sweep {sweep} complete, full outputs restored exactly");
        }
        session.set_diagnostic_work_mask(7);
        let baseline_after: Vec<_> = (0..64).map(|_| execute(session)).collect();
        save_json(
            &path.join("result.json"),
            &serde_json::json!({
                "status": "complete", "dispatches": dispatches, "samples": samples,
                "baseline_before": baseline_before, "baseline_after": baseline_after,
                "outputs": before,
            }),
        );
        session_counts.insert(name, dispatches);
    }
    checkpoint(&mut core, &root.join("after"));
    let mut exact_tensors = 0;
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
            exact_tensors += 1;
        }
    }
    assert_eq!(exact_tensors, 292);
    let memory = core.gpu_memory_budget();
    let final_headroom = memory.budget_bytes.saturating_sub(memory.usage_bytes);
    assert!(final_headroom >= 2 << 30);
    save_json(
        &root.join("result.json"),
        &serde_json::json!({
            "status": "complete", "synthetic_updates": 1, "game_actions": 0,
            "exact_tensors": exact_tensors, "sessions": session_counts, "masks": 8,
            "sweeps": 3, "measured_per_mask_per_sweep": 16,
            "minimum_sampled_headroom_bytes": initial_headroom.min(final_headroom),
        }),
    );
}
