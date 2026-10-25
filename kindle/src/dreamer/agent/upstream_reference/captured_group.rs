use super::*;
use meganeura::{Session, compile::BufferRef};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, ops::Range};

pub(super) fn backup(session: &Session, inputs: &[BufferRef]) -> Session {
    let mut graph = meganeura::Graph::new();
    let outputs = inputs
        .iter()
        .map(|&id| {
            let bytes = session.plan().buffers[id.0 as usize];
            assert!(bytes.is_multiple_of(4));
            graph.input(&format!("saved_{}", id.0), &[bytes / 4])
        })
        .collect();
    graph.set_outputs(outputs);
    Session::with_context_opts(
        meganeura::compile::compile(&graph),
        session.context(),
        meganeura::SessionOptions {
            no_alias: true,
            ..Default::default()
        },
    )
}

pub(super) fn copy_inputs(
    session: &Session,
    saved: &Session,
    inputs: &[BufferRef],
    encoder: &mut blade_graphics::CommandEncoder,
    restore: bool,
) {
    let mut transfer = encoder.transfer(if restore {
        "restore_inputs"
    } else {
        "capture_inputs"
    });
    for &id in inputs {
        let original = session.diagnostic_buffer_piece(id);
        let copy = saved.input_buffer(&format!("saved_{}", id.0)).unwrap();
        let (src, dst) = if restore {
            (copy, original)
        } else {
            (original, copy)
        };
        transfer.copy_buffer_to_buffer(src, dst, session.plan().buffers[id.0 as usize] as u64);
    }
}

pub(super) fn dump(
    session: &Session,
    ids: &[BufferRef],
    root: &Path,
    name: &str,
    finite: bool,
) -> serde_json::Value {
    let rows: Vec<_> = ids.iter().zip(session.read_buffers(ids)).map(|(id, values)| {
        assert!(!finite || values.iter().all(|x| x.is_finite()), "{name}/{}", id.0);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(root.join(format!("{name}-{}.bin", id.0)), &bytes).unwrap();
        serde_json::json!({"buffer":id.0,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))})
    }).collect();
    serde_json::json!(rows)
}

pub(super) fn submit(
    session: &mut Session,
    saved: Option<&mut Session>,
    encoder: &mut blade_graphics::CommandEncoder,
) {
    let gpu = session.context();
    let sync = gpu.submit(encoder);
    session.track_submission(sync.clone());
    assert!(gpu.wait_for(&sync, !0).unwrap());
    session.wait();
    if let Some(saved) = saved {
        saved.track_submission(sync);
        saved.wait();
    }
}

fn measure(
    session: &mut Session,
    mut saved: Option<&mut Session>,
    inputs: &[BufferRef],
    spans: &[Range<usize>],
    count: usize,
) -> serde_json::Value {
    let gpu = session.context();
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "captured convolution group",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    let cache_before = session.command_cache_counts();
    let start = std::time::Instant::now();
    for _ in 0..count {
        if let Some(saved) = saved.as_deref() {
            copy_inputs(session, saved, inputs, &mut encoder, true);
        }
        if spans.is_empty() {
            session.record(&mut encoder).unwrap();
        } else {
            for span in spans {
                session.record_diagnostic_groups(&mut encoder, span.clone());
            }
        }
    }
    let record_ms = start.elapsed().as_secs_f64() * 1000.0;
    submit(session, saved.as_deref_mut(), &mut encoder);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let timings: Vec<_> = encoder
        .last_timing()
        .pass_durations()
        .map(|(name, duration)| (name.to_owned(), duration.as_secs_f64() * 1000.0))
        .collect();
    let scopes = spans.len().max(1) + usize::from(saved.is_some());
    assert_eq!(timings.len(), count * scopes);
    assert!(timings.iter().all(|(_, ms)| ms.is_finite() && *ms >= 0.0));
    gpu.destroy_command_encoder(&mut encoder);
    serde_json::json!({"calls":count,"scopes_per_call":scopes,"record_ms":record_ms,
        "wall_ms":wall_ms,"timings":timings,"cache_before":cache_before,"cache_after":session.command_cache_counts()})
}

fn value(index: usize, salt: u32) -> f32 {
    let x = (index as u32).wrapping_mul(1_664_525).wrapping_add(salt);
    ((x ^ (x >> 16)) % 61) as f32 / 64.0 - 30.0 / 64.0
}

#[test]
#[ignore = "Bounded actual-buffer context diagnosis; one preparatory update"]
fn captured_convolution_group() {
    use meganeura::data::safetensors::SafeTensorsModel;
    let source = std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap();
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    assert!(meganeura::GpuOptions::from_env().timing);
    assert_eq!(std::env::var("MEGANEURA_COMMAND_CACHE").unwrap(), "1");
    let mut core = restore_profile_core(Path::new(&source));
    assert_eq!(core.gpu_device().device_name, "NVIDIA GeForce RTX 5080");
    let headroom = |gpu: &blade_graphics::Context| {
        let memory = gpu.memory_stats();
        let bytes = memory.budget.saturating_sub(memory.usage);
        assert!(bytes >= 2 << 30);
        bytes
    };
    let gpu = core.world_train.context();
    let mut minimum = headroom(&gpu);
    core.world_train.set_submission_chunks(1);
    assert!(core.world_posterior.is_none());
    let config = core.config.clone();
    save_json(&root.join("config.json"), &config);
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
        "deter":posterior.deter,"stoch":posterior.stoch,"action_target":targets.action_target,
        "imagined_weight":targets.imagined_weight,"imagined_value_target":targets.imagined_value_target,
        "imagined_slow_target":targets.imagined_slow_target,"replay_value_target":targets.replay_value_target,
        "replay_slow_target":targets.replay_slow_target,"replay_weight":targets.replay_weight}),
    );
    save_json(
        &root.join("report.json"),
        &serde_json::json!({"world":world,"behavior":behavior}),
    );
    checkpoint(&mut core, &root.join("before"));
    for session in [&mut core.world_train, &mut core.behavior_train] {
        session.clear_optimizer();
        session.clear_grad_accumulate();
        session.disable_grad_clip();
    }
    let session = &mut core.world_train;
    let group = session
        .diagnostic_groups()
        .iter()
        .position(|r| r == &(3974..3979))
        .unwrap();
    let total = session.diagnostic_groups().len();
    let selected = &session.plan().dispatches[3974..3979];
    let inputs: Vec<_> = selected
        .iter()
        .flat_map(|d| d.input_buffers.iter().map(|b| b.0))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(BufferRef)
        .collect();
    let outputs: Vec<_> = selected.iter().map(|d| d.output_buffer).collect();
    assert_eq!(inputs.len(), 7);
    let dy = selected[3].input_buffers[0];
    assert_eq!(selected[4].input_buffers[0], dy);
    let reduced = [
        dy,
        selected[3].input_buffers[1],
        selected[4].input_buffers[1],
    ];
    let regions: Vec<_> = inputs
        .iter()
        .map(|&id| (id, session.diagnostic_buffer_region(id)))
        .collect();
    for (i, (_, a)) in regions.iter().enumerate() {
        for (_, b) in &regions[i + 1..] {
            assert!(
                a.0 != b.0 || a.1 + a.2 <= b.1 || b.1 + b.2 <= a.1,
                "overlapping input ranges"
            );
        }
    }
    let memory = session.memory_summary();
    save_json(
        &root.join("metadata.json"),
        &serde_json::json!({
        "dispatches":session.plan().dispatches,"pipelines":session.dispatch_pipeline_keys(),
        "groups":session.diagnostic_groups(),"buffers":session.plan().buffers,
        "group":group,"inputs":inputs,"outputs":outputs,"input_regions":regions,"bf16_inputs":reduced,
        "output_regions":outputs.iter().map(|&id| (id,session.diagnostic_buffer_region(id))).collect::<Vec<_>>(),
        "memory":{"allocated_buffer_bytes":memory.allocated_buffer_bytes,
            "device_local_requested_bytes":memory.device_local_bytes,"num_allocations":memory.num_allocations}}),
    );
    let mut measurements = BTreeMap::new();
    let segmented = [0..group, group..group + 1, group + 1..total];
    for (name, spans) in [("whole", &[][..]), ("segmented", &segmented[..])] {
        measure(session, None, &inputs, spans, 16);
        measurements.insert(name, measure(session, None, &inputs, spans, 64));
        let state = inline_gap::output_fingerprint(session, &root.join(format!("{name}.f32")));
        save_json(&root.join(format!("{name}-outputs.json")), &state);
    }
    let baseline: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("whole-outputs.json")).unwrap())
            .unwrap();
    let segmented_outputs: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("segmented-outputs.json")).unwrap())
            .unwrap();
    assert_eq!(baseline, segmented_outputs);
    let mut captured = backup(session, &inputs);
    let mut synthetic = backup(session, &inputs);
    minimum = minimum.min(headroom(&gpu));
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "capture graph cut",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    session.record_diagnostic_groups(&mut encoder, 0..group);
    copy_inputs(session, &captured, &inputs, &mut encoder, false);
    submit(session, Some(&mut captured), &mut encoder);
    let captured_inputs = dump(session, &inputs, &root, "input", false);
    let captured_backup = captured.read_buffers(&captured.plan().output_buffers);
    for ((&id, words), actual) in inputs
        .iter()
        .zip(captured_backup)
        .zip(session.read_buffers(&inputs))
    {
        assert_eq!(words.len(), actual.len());
        assert!(
            words
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "capture {}",
            id.0
        );
    }
    encoder.start();
    session.record_diagnostic_groups(&mut encoder, group..group + 1);
    submit(session, None, &mut encoder);
    let expected = dump(session, &outputs, &root, "expected", true);
    encoder.start();
    session.record_diagnostic_groups(&mut encoder, group + 1..total);
    submit(session, None, &mut encoder);
    assert_eq!(
        baseline,
        inline_gap::output_fingerprint(session, &root.join("captured-full.f32"))
    );
    for &id in &inputs {
        let len = session.plan().buffers[id.0 as usize] / 4;
        let words: Vec<u32> = if reduced.contains(&id) {
            let salt = if id == dy {
                1_013_904_223
            } else {
                1_987_654_321
            };
            (0..len)
                .map(|i| {
                    (value(2 * i, salt).to_bits() >> 16)
                        | (value(2 * i + 1, salt).to_bits() & 0xffff_0000)
                })
                .collect()
        } else {
            (0..len).map(|i| value(i, id.0).to_bits()).collect()
        };
        synthetic.set_input_u32(&format!("saved_{}", id.0), &words);
    }
    for (name, saved) in [("captured", &mut captured), ("synthetic", &mut synthetic)] {
        let span = group..group + 1;
        let spans = std::slice::from_ref(&span);
        measure(session, Some(saved), &inputs, spans, 16);
        measurements.insert(name, measure(session, Some(saved), &inputs, spans, 64));
        let actual = dump(session, &outputs, &root, name, true);
        if name == "captured" {
            assert_eq!(expected, actual);
        }
        minimum = minimum.min(headroom(&gpu));
    }
    encoder.start();
    copy_inputs(session, &captured, &inputs, &mut encoder, true);
    session.record(&mut encoder).unwrap();
    submit(session, Some(&mut captured), &mut encoder);
    assert_eq!(
        baseline,
        inline_gap::output_fingerprint(session, &root.join("restored-full.f32"))
    );
    gpu.destroy_command_encoder(&mut encoder);
    save_json(&root.join("measurements.json"), &measurements);
    save_json(&root.join("inputs.json"), &captured_inputs);
    save_json(&root.join("target-outputs.json"), &expected);
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
            assert!(
                before.tensor_f32(name).unwrap() == after.tensor_f32(name).unwrap(),
                "changed {name}"
            );
            exact_tensors += 1;
        }
    }
    assert_eq!(exact_tensors, 292);
    save_json(
        &root.join("result.json"),
        &serde_json::json!({"status":"complete",
        "synthetic_updates":1,"game_actions":0,"exact_tensors":exact_tensors,
        "full_optimizer_free_replays":162,"target_only_replays":160,
        "minimum_sampled_headroom_bytes":minimum,"exact_captured_inputs":7,
        "exact_target_outputs":5,"exact_full_restores":3}),
    );
}
