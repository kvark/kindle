//! Capture a fixed component cohort for an independent, CPU-only F64 oracle.
use super::*;
use crate::dreamer::{
    device_copy::DeviceCopies, readback::Readback, replay::FrameFlags, runtime::build_session,
};
use std::{collections::BTreeMap, io::Write, path::Path};

fn dummy(gpu: &Arc<gpu::Context>, inputs: &[(String, usize, usize)]) -> Session {
    let mut graph = meganeura::Graph::new();
    let outputs = inputs
        .iter()
        .map(|(name, rows, width)| {
            let input = graph.input(name, &[*rows, *width]);
            graph.neg(input)
        })
        .collect();
    graph.set_outputs(outputs);
    build_session(&graph, gpu, meganeura::Mode::Inference, false)
}

fn fill_outputs(session: &mut Session, inputs: &[(String, usize, usize)], values: &[Vec<f32>]) {
    for ((name, rows, width), value) in inputs.iter().zip(values) {
        assert_eq!(value.len(), rows * width);
        session.set_input(name, &value.iter().map(|x| -*x).collect::<Vec<_>>());
    }
    session.step();
}

fn save(root: &Path, name: &str, values: &[f32]) {
    let mut file = std::io::BufWriter::new(
        std::fs::File::create_new(root.join(format!("{name}.f32"))).unwrap(),
    );
    for value in values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}

fn pattern(count: usize, shift: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i + shift) % 197) as f32 - 98.0) / 64.0)
        .collect()
}

#[test]
#[ignore = "requires guarded GPU; capture fixed targets for independent F64 review"]
fn capture_fixed_components() {
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_TARGET_QUAL_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let gpu = meganeura::reference::gpu::shared_context();
    let info = gpu.device_information();
    assert_eq!(
        info.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!info.is_software_emulated);
    let mut readback = Readback::new(Arc::clone(&gpu));
    let mut copies = DeviceCopies::new(Arc::clone(&gpu));
    let mut cases = Vec::new();
    let mut minimum_headroom = u64::MAX;
    for (index, (batch_size, length, horizon, count, actions, intrinsic)) in [
        (1, 2, 1, 3, 2, false),
        (3, 7, 5, 15, 4, true),
        (8, 16, 15, 255, 18, false),
        (32, 16, 15, 255, 18, true),
    ]
    .into_iter()
    .enumerate()
    {
        let mut config = DreamerConfig::tiny(actions);
        config.batch_size = batch_size;
        config.batch_length = length;
        config.world_backprop_length = length;
        config.imagination_length = horizon;
        config.value_bins = count;
        config.disagreement_bonus = intrinsic;
        config.intrinsic_reward_scale = if intrinsic { 0.05 } else { 0.0 };
        let bins = TwoHotBins::new(count);
        let mut targets = TargetsGpu::new(Arc::clone(&gpu), &config, &bins);
        let starts = batch_size * length;
        let rows = starts * horizon;
        let all = rows + starts;
        let replay_rows = starts - batch_size;
        let feature = config.feature_dim();
        let imagined_inputs = vec![
            ("feature".into(), rows, feature),
            ("all_feature".into(), all, feature),
            ("action".into(), rows, actions),
            ("reward".into(), all, 1),
            ("continuation".into(), all, 1),
            ("value".into(), all, 1),
            ("bonus".into(), rows, 1),
        ];
        let mut imagination = dummy(&gpu, &imagined_inputs);
        let posterior_inputs = vec![
            ("deter".into(), starts, 2),
            ("stoch".into(), starts, 3),
            ("bonus".into(), starts, 1),
        ];
        let mut posterior = dummy(&gpu, &posterior_inputs);
        let behavior_inputs = vec![
            ("imagined_feature".into(), rows, feature),
            ("action_target".into(), rows, actions),
            ("imagined_weight".into(), rows, 1),
            ("imagined_value_target".into(), rows, count),
            ("imagined_slow_target".into(), rows, count),
            ("replay_feature".into(), replay_rows, feature),
            ("replay_weight".into(), replay_rows, 1),
            ("replay_value_target".into(), replay_rows, count),
            ("replay_slow_target".into(), replay_rows, count),
        ];
        let mut behavior = dummy(&gpu, &behavior_inputs);
        let world_inputs: Vec<_> = (0..length)
            .flat_map(|t| {
                [
                    (format!("reward_target_{t}"), batch_size, count),
                    (format!("continuation_target_{t}"), batch_size, 1),
                    (format!("replay_value_target_{t}"), batch_size, count),
                    (format!("replay_slow_target_{t}"), batch_size, count),
                    (format!("replay_value_weight_{t}"), batch_size, 1),
                ]
            })
            .collect();
        let mut world = dummy(&gpu, &world_inputs);
        let mean_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "target_qualification_means",
            size: ((3 * all + 128) * 4) as u64,
            memory: gpu::Memory::Shared,
        });
        let mean_pieces =
            std::array::from_fn(|head| mean_buffer.at(((64 + head * all) * 4) as u64));
        let initial_state = [-0.25, 0.75];
        targets.restore_normalizer(initial_state);
        let mut first = BTreeMap::new();
        for round in 0..3 {
            let variant = usize::from(round == 1);
            if round == 2 {
                targets.restore_normalizer(initial_state);
            }
            let before = targets.normalizer_state();
            let directory = root.join(format!("case-{index}-round-{round}"));
            std::fs::create_dir(&directory).unwrap();
            let reward = (0..all)
                .map(|i| match (i + variant) % 11 {
                    0 => 0.0,
                    1 => -1.0,
                    2 => 1.0,
                    3 => -1000.0,
                    4 => 1000.0,
                    _ => ((i % 43) as f32 - 21.0) * 0.125,
                })
                .collect::<Vec<_>>();
            let value = pattern(all, 17 + variant);
            let slow = (0..all)
                .map(|i| {
                    let center = bins.values()[(i + variant) % count];
                    match i % 5 {
                        0 => center,
                        1 => center.next_up(),
                        2 => center.next_down(),
                        3 => 1e9,
                        _ => -1e9,
                    }
                })
                .collect::<Vec<_>>();
            let continuation = (0..all)
                .map(|i| [0.0, 1.0, 0.99, 0.3, -0.5, 1.5][(i + variant) % 6])
                .collect::<Vec<_>>();
            let bonus = (0..rows)
                .map(|i| ((i + variant) % 17) as f32 * 0.125)
                .collect::<Vec<_>>();
            let replay_bonus = (0..starts)
                .map(|i| ((i + variant) % 13) as f32 * 0.25)
                .collect::<Vec<_>>();
            let mut action = vec![0.0; rows * actions];
            for i in 0..rows {
                action[i * actions + (i + variant) % actions] = 1.0;
            }
            let features = pattern(rows * feature, variant);
            fill_outputs(
                &mut imagination,
                &imagined_inputs,
                &[
                    features.clone(),
                    pattern(all * feature, 31 + variant),
                    action.clone(),
                    reward.clone(),
                    continuation.clone(),
                    value.clone(),
                    bonus.clone(),
                ],
            );
            fill_outputs(
                &mut posterior,
                &posterior_inputs,
                &[
                    pattern(starts * 2, variant),
                    pattern(starts * 3, variant),
                    replay_bonus.clone(),
                ],
            );
            let mut mean_storage = vec![12345.25; 64];
            mean_storage.extend(&reward);
            mean_storage.extend(&value);
            mean_storage.extend(&slow);
            mean_storage.extend([12345.25; 64]);
            write_shared(mean_buffer, &mean_storage);
            let mut batch = SequenceBatch::default();
            batch.rewards = (0..length)
                .map(|t| {
                    (0..batch_size)
                        .map(|b| ((t + b + variant) % 5) as f32 - 2.0)
                        .collect()
                })
                .collect();
            batch.flags = (0..length)
                .map(|t| {
                    (0..batch_size)
                        .map(|b| {
                            let last = (t + b + variant) % 3 == 1;
                            FrameFlags {
                                is_first: t == 0,
                                is_last: last,
                                is_terminal: last && (t + b) % 2 == 0,
                            }
                        })
                        .collect()
                })
                .collect();
            targets.generate(
                &batch,
                &imagination,
                (&posterior, 2),
                &behavior,
                mean_pieces,
            );
            let mut regions = targets.handoffs(&world, &behavior, &imagination);
            regions.push((
                imagination.output_buffer(0).unwrap(),
                behavior.input_buffer("imagined_feature").unwrap(),
                rows * feature * 4,
            ));
            copies.copy_regions(&regions);
            world.step();
            world.wait();
            let report = targets.report();
            behavior.step();
            behavior.wait();
            targets.finish_consumers();
            let mut outputs = BTreeMap::new();
            for (prefix, session, inputs) in [
                ("behavior", &behavior, &behavior_inputs),
                ("world", &world, &world_inputs),
            ] {
                for (slot, (name, rows, width)) in inputs.iter().enumerate() {
                    let mut input = vec![0.0; rows * width];
                    let mut consumed = vec![0.0; rows * width];
                    readback.read_regions(&mut [
                        (session.input_buffer(name).unwrap(), &mut input),
                        (session.output_buffer(slot).unwrap(), &mut consumed),
                    ]);
                    assert!(
                        input
                            .iter()
                            .zip(&consumed)
                            .all(|(x, y)| x.is_finite() && x.to_bits() == (-*y).to_bits()),
                        "{name}"
                    );
                    outputs.insert(format!("{prefix}-{name}"), input);
                }
            }
            for (name, source, count) in [
                ("returns", targets.returns.at(0), rows),
                ("replay_returns", targets.replay_returns.at(0), replay_rows),
            ] {
                let mut values = vec![0.0; count];
                readback.read_regions(&mut [(source, &mut values)]);
                assert!(values.iter().all(|x| x.is_finite()));
                outputs.insert(name.into(), values);
            }
            outputs.insert("report".into(), report.to_vec());
            let after = targets.normalizer_state();
            assert_eq!(after, [report[23], report[24]]);
            let means_after = unsafe {
                std::slice::from_raw_parts(mean_buffer.data().cast::<f32>(), mean_storage.len())
            };
            assert!(
                means_after
                    .iter()
                    .zip(&mean_storage)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            if round == 0 {
                first = outputs.clone();
            }
            if round == 2 {
                assert_eq!(outputs.len(), first.len());
                for (name, values) in &outputs {
                    assert!(
                        values
                            .iter()
                            .zip(&first[name])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "restored {name}"
                    );
                }
            }
            for (name, values) in &outputs {
                save(&directory, name, values);
            }
            for (name, values) in [
                ("reward", &reward),
                ("value", &value),
                ("slow", &slow),
                ("continuation", &continuation),
                ("bonus", &bonus),
                ("replay_bonus", &replay_bonus),
                ("action", &action),
                ("features", &features),
            ] {
                save(&directory, name, values);
            }
            save(&directory, "bins", bins.values());
            let observed: Vec<_> = batch.rewards.iter().flatten().copied().collect();
            let last: Vec<_> = batch
                .flags
                .iter()
                .flatten()
                .map(|f| f32::from(f.is_last))
                .collect();
            let terminal: Vec<_> = batch
                .flags
                .iter()
                .flatten()
                .map(|f| f32::from(f.is_terminal))
                .collect();
            save(&directory, "observed", &observed);
            save(&directory, "last", &last);
            save(&directory, "terminal", &terminal);
            let memory = gpu.memory_stats();
            minimum_headroom = minimum_headroom.min(memory.budget.saturating_sub(memory.usage));
            assert!(minimum_headroom >= 2 << 30);
            cases.push(serde_json::json!({"index": index, "round": round, "directory": directory,
                "config": config, "state_before": before, "state_after": after,
                "outputs": outputs.iter().map(|(name, values)| (name, values.len())).collect::<BTreeMap<_, _>>(),
                "device_copy_regions": regions.len(), "source_canary_and_values_exact": true,
                "consumer_negations_exact": true, "restored_outputs_exact": round == 2}));
        }
        gpu.destroy_buffer(mean_buffer);
    }
    let result = serde_json::json!({"status": "captured", "cases": cases,
        "minimum_sampled_headroom_bytes": minimum_headroom, "target_generations": 12,
        "learner_updates": 0, "game_actions": 0, "independent_numerical_review_pending": true});
    let file = std::fs::File::create_new(root.join("result.json")).unwrap();
    serde_json::to_writer_pretty(file, &result).unwrap();
}
