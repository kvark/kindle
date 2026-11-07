use super::{captured_group::*, *};
use meganeura::{Session, compile::BufferRef, data::safetensors::SafeTensorsModel};
use std::{collections::BTreeSet, ops::Range};

fn measure(
    session: &mut Session,
    saved: &mut Session,
    inputs: &[BufferRef],
    spans: &[Range<usize>],
    count: usize,
) -> serde_json::Value {
    let gpu = session.context();
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "captured convolution cut",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    let cache_before = session.command_cache_counts();
    let start = std::time::Instant::now();
    for _ in 0..count {
        copy_inputs(session, saved, inputs, &mut encoder, true);
        for span in spans {
            session.record_diagnostic_groups(&mut encoder, span.clone());
        }
    }
    let record_ms = start.elapsed().as_secs_f64() * 1000.0;
    submit(session, Some(saved), &mut encoder);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let timings: Vec<_> = encoder
        .last_timing()
        .pass_durations()
        .map(|(name, duration)| (name.to_owned(), duration.as_secs_f64() * 1000.0))
        .collect();
    let scopes = spans.len() + 1;
    assert_eq!(timings.len(), count * scopes);
    assert!(timings.iter().all(|(_, ms)| ms.is_finite() && *ms >= 0.0));
    gpu.destroy_command_encoder(&mut encoder);
    serde_json::json!({"calls":count,"spans":spans,"scopes_per_call":scopes,
        "record_ms":record_ms,"wall_ms":wall_ms,"timings":timings,
        "cache_before":cache_before,"cache_after":session.command_cache_counts()})
}

#[test]
#[ignore = "Bounded saved-input replay of original learner allocation; no updates"]
fn replay_convolution_boundary() {
    let source = std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap();
    let captured = std::path::PathBuf::from(std::env::var_os("KINDLE_CAPTURED_GROUP").unwrap());
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    assert!(meganeura::GpuOptions::from_env().timing);
    assert_eq!(std::env::var("MEGANEURA_COMMAND_CACHE").unwrap(), "1");
    let mut core = restore_profile_core(Path::new(&source));
    for (session, name) in [
        (&mut core.world_train, "world"),
        (&mut core.behavior_train, "behavior"),
        (&mut core.behavior_slow, "slow"),
    ] {
        crate::dreamer::checkpoint::load_session(
            session,
            &captured.join("before").join(format!("{name}.safetensors")),
        )
        .unwrap();
    }
    assert_eq!(core.gpu_device().device_name, "NVIDIA GeForce RTX 5080");
    let gpu = core.world_train.context();
    let headroom = || {
        let memory = gpu.memory_stats();
        let bytes = memory.budget.saturating_sub(memory.usage);
        assert!(bytes >= 2 << 30);
        bytes
    };
    let mut minimum = headroom();
    core.world_train.set_submission_chunks(1);
    assert!(core.world_posterior.is_none());
    save_json(&root.join("config.json"), &core.config);
    checkpoint(&mut core, &root.join("before"));
    for session in [&mut core.world_train, &mut core.behavior_train] {
        session.clear_optimizer();
        session.clear_grad_accumulate();
        session.disable_grad_clip();
    }
    let session = &mut core.world_train;
    let metadata: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(captured.join("metadata.json")).unwrap())
            .unwrap();
    assert_eq!(
        serde_json::to_value(&session.plan().dispatches).unwrap(),
        metadata["dispatches"]
    );
    assert_eq!(
        serde_json::to_value(session.dispatch_pipeline_keys()).unwrap(),
        metadata["pipelines"]
    );
    assert_eq!(
        serde_json::to_value(session.diagnostic_groups()).unwrap(),
        metadata["groups"]
    );
    assert_eq!(
        serde_json::to_value(&session.plan().buffers).unwrap(),
        metadata["buffers"]
    );
    let inputs: Vec<BufferRef> = serde_json::from_value(metadata["inputs"].clone()).unwrap();
    let outputs: Vec<BufferRef> = serde_json::from_value(metadata["outputs"].clone()).unwrap();
    for (ids, name) in [(&inputs, "input_regions"), (&outputs, "output_regions")] {
        let regions: Vec<_> = ids
            .iter()
            .map(|&id| (id, session.diagnostic_buffer_region(id)))
            .collect();
        assert_eq!(serde_json::to_value(regions).unwrap(), metadata[name]);
    }
    let input_set: BTreeSet<_> = inputs.iter().map(|id| id.0).collect();
    assert_eq!(input_set.len(), 7);
    assert!(outputs.iter().all(|id| !input_set.contains(&id.0)));
    let mut saved = backup(session, &inputs);
    for &id in &inputs {
        let bytes = std::fs::read(captured.join(format!("input-{}.bin", id.0))).unwrap();
        assert_eq!(bytes.len(), session.plan().buffers[id.0 as usize]);
        let words: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        saved.set_input_u32(&format!("saved_{}", id.0), &words);
    }
    minimum = minimum.min(headroom());
    let expected: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(captured.join("target-outputs.json")).unwrap())
            .unwrap();
    let group = metadata["group"].as_u64().unwrap() as usize;
    assert_eq!(group, 1249);
    let modes = [("intact", group..group + 1), ("boundary", group..group + 2)];
    let mut measurements = BTreeMap::new();
    let mut exact_outputs = 0;
    for (name, span) in modes {
        if name == "boundary" {
            session.split_diagnostic_group(3977);
        }
        let spans = std::slice::from_ref(&span);
        measure(session, &mut saved, &inputs, spans, 16);
        measurements.insert(name, measure(session, &mut saved, &inputs, spans, 64));
        let actual = dump(session, &outputs, &root, name, true);
        for output in actual.as_array().unwrap() {
            assert_eq!(
                Some(output),
                expected
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|row| row["buffer"] == output["buffer"])
            );
            exact_outputs += 1;
        }
        minimum = minimum.min(headroom());
        println!("Completed {name}: {} exact outputs", outputs.len());
    }
    save_json(
        &root.join("world-groups.json"),
        &session.diagnostic_groups(),
    );
    save_json(&root.join("measurements.json"), &measurements);
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
    assert_eq!(exact_outputs, 10);
    save_json(
        &root.join("result.json"),
        &serde_json::json!({"status":"complete",
        "synthetic_updates":0,"game_actions":0,"exact_tensors":exact_tensors,
        "group_replays":160,"minimum_sampled_headroom_bytes":minimum,
        "exact_outputs":exact_outputs,"exact_plans_pipelines_regions":true}),
    );
}
