use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{io::Write, path::Path};

pub(crate) fn save_floats(path: &Path, values: &[f32]) {
    let mut stream = std::io::BufWriter::new(std::fs::File::create_new(path).unwrap());
    for value in values {
        stream.write_all(&value.to_le_bytes()).unwrap();
    }
    stream.flush().unwrap();
}

pub(crate) fn compare(bins: &TwoHotBins, logits: &[f32], actual: &[f32]) -> serde_json::Value {
    assert_eq!(logits.len(), actual.len() * bins.len());
    let mut failures = Vec::new();
    let mut references = Vec::new();
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    let mut max_abs_error = 0.0f64;
    let mut max_direct_reference_error = 0.0f64;
    let mut symmetric_rows = 0;
    for (row_index, (row, &got)) in logits.chunks_exact(bins.len()).zip(actual).enumerate() {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<_> = row.iter().map(|x| ((*x - maximum) as f64).exp()).collect();
        let denominator: f64 = exp.iter().sum();
        let reference: f64 = (0..bins.len() / 2)
            .map(|i| {
                let j = bins.len() - 1 - i;
                (exp[i] * bins.values()[i] as f64 + exp[j] * bins.values()[j] as f64) / denominator
            })
            .sum();
        let direct = exp
            .iter()
            .zip(bins.values())
            .map(|(p, b)| p / denominator * *b as f64)
            .sum::<f64>();
        max_direct_reference_error = max_direct_reference_error.max((direct - reference).abs());
        let error = (got as f64 - reference).abs();
        let symmetric = row.iter().eq(row.iter().rev());
        if symmetric {
            symmetric_rows += 1;
        }
        let bound = 1e-4 + 2e-5 * reference.abs();
        if !got.is_finite() || error > bound || (symmetric && got != 0.0) {
            failures.push(
                serde_json::json!({"row":row_index,"actual":got,"reference":reference,
                "absolute_error":error,"bound":bound,"symmetric":symmetric}),
            );
        }
        max_abs_error = max_abs_error.max(error);
        squared_error += error * error;
        squared_reference += reference * reference;
        references.push(reference);
    }
    let relative_l2 = (squared_error / squared_reference.max(1e-30)).sqrt();
    serde_json::json!({"status":if failures.is_empty() && relative_l2 <= 2e-5 {"complete"} else {"failed"},
        "rows":actual.len(),"bins":bins.len(),"symmetric_rows":symmetric_rows,
        "relative_l2":relative_l2,"max_absolute_error":max_abs_error,
        "max_direct_reference_error":max_direct_reference_error,"failures":failures,"references":references})
}

#[test]
fn shader_validates() {
    let module = naga::front::wgsl::parse_str(include_str!("../categorical_gpu.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires guarded GPU; independent F64 categorical means and lifetimes"]
fn qualify_synthetic() {
    const GUARD: usize = 64;
    const CANARY: f32 = 12345.25;
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_MEANS_ROOT").unwrap());
    std::fs::create_dir(&root).unwrap();
    let gpu = meganeura::reference::gpu::shared_context();
    let device = gpu.device_information();
    assert_eq!(
        device.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let mut readback = super::super::readback::Readback::new(Arc::clone(&gpu));
    let mut cases = Vec::new();
    let mut minimum_headroom = u64::MAX;
    for (count, rows) in [(3, 17), (7, 129), (255, 1025), (511, 257)] {
        let bins = TwoHotBins::new(count);
        let mut decoder = CategoricalGpu::new(Arc::clone(&gpu), &bins, rows);
        let memory = gpu.memory_stats();
        minimum_headroom = minimum_headroom.min(memory.budget.saturating_sub(memory.usage));
        assert!(minimum_headroom >= 2 << 30);
        let mut rng = StdRng::seed_from_u64(701 + count as u64);
        let mut inputs = vec![0.0; count * rows];
        for (r, row) in inputs.chunks_exact_mut(count).enumerate() {
            match r % 7 {
                0 => row.fill(0.0),
                1 => {
                    for i in 0..=count / 2 {
                        row[i] = rng.random_range(-20.0..20.0);
                        row[count - 1 - i] = row[i];
                    }
                }
                2 => {
                    row.fill(-100.0);
                    row[r % count] = 0.0;
                }
                3 => {
                    row.fill(-1.0);
                    row[r % count] += 1e-5;
                }
                4 => {
                    for x in row {
                        *x = rng.random_range(-4.0..4.0);
                    }
                }
                5 => {
                    for x in row {
                        *x = rng.random_range(-80.0..20.0);
                    }
                }
                _ => {
                    row.fill(-40.0);
                    row[count / 2] = 0.0;
                    row[r % count] = -39.9999;
                }
            }
        }
        let source = gpu.create_buffer(gpu::BufferDesc {
            name: "mean_test_input",
            size: ((inputs.len() + 2 * GUARD) * 4) as u64,
            memory: gpu::Memory::Shared,
        });
        let target = gpu.create_buffer(gpu::BufferDesc {
            name: "mean_test_output",
            size: ((rows + 2 * GUARD) * 4) as u64,
            memory: gpu::Memory::Shared,
        });
        let mut saved_first = Vec::new();
        for change in 0..3 {
            let mut changed = inputs.clone();
            if change == 1 {
                for (r, row) in changed.chunks_exact_mut(count).enumerate() {
                    row[(r + 1) % count] += 0.125;
                }
            }
            decoder.wait();
            unsafe {
                let src = std::slice::from_raw_parts_mut(
                    source.data().cast::<f32>(),
                    inputs.len() + 2 * GUARD,
                );
                src.fill(CANARY);
                src[GUARD..GUARD + inputs.len()].copy_from_slice(&changed);
                std::slice::from_raw_parts_mut(target.data().cast::<f32>(), rows + 2 * GUARD)
                    .fill(CANARY);
            }
            decoder.decode_into(&[source.at((GUARD * 4) as u64)], target.at(0), GUARD);
            let mut full_output = vec![0.0; rows + 2 * GUARD];
            readback.read_regions(&mut [(target.at(0), &mut full_output)]);
            assert!(
                full_output[..GUARD]
                    .iter()
                    .chain(&full_output[GUARD + rows..])
                    .all(|&v| v == CANARY)
            );
            let actual = &full_output[GUARD..GUARD + rows];
            let full_input = unsafe {
                std::slice::from_raw_parts(source.data().cast::<f32>(), inputs.len() + 2 * GUARD)
            };
            assert!(
                full_input[..GUARD]
                    .iter()
                    .chain(&full_input[GUARD + inputs.len()..])
                    .all(|&v| v == CANARY)
            );
            assert_eq!(&full_input[GUARD..GUARD + inputs.len()], &changed);
            if change == 0 {
                saved_first = actual.to_vec();
            }
            if change == 1 {
                assert_ne!(actual, &saved_first);
            }
            if change == 2 {
                assert_eq!(actual, &saved_first);
            }
            save_floats(
                &root.join(format!("bins{count}-change{change}-inputs.f32")),
                &changed,
            );
            save_floats(
                &root.join(format!("bins{count}-change{change}-means.f32")),
                actual,
            );
            let mut comparison = compare(&bins, &changed, actual);
            comparison["change"] = change.into();
            cases.push(comparison);
        }
        decoder.wait();
        gpu.destroy_buffer(source);
        gpu.destroy_buffer(target);
    }
    let pass = cases.iter().all(|c| c["status"] == "complete");
    let result = serde_json::json!({"status":if pass {"complete"} else {"failed"},"cases":cases,
        "minimum_sampled_headroom_bytes":minimum_headroom,"game_actions":0,"synthetic_updates":0,
        "canaries":true,"changed_and_restored":true});
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert!(
        pass,
        "categorical numerical qualification failed; full results retained"
    );
}
