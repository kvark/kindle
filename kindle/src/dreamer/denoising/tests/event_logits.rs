//! Frozen event-head inputs and logits; no production prediction or training changes.
use super::super::*;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    checkpoint: PathBuf,
    images: PathBuf,
    actions: PathBuf,
    teacher: PathBuf,
    ordinary: PathBuf,
    expected_events: PathBuf,
    output: PathBuf,
}
fn floats(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn write_floats(file: &mut impl Write, values: &[f32]) {
    file.write_all(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn head(
    model: &mut DenoisingWorldLearner,
    history: &History,
    frame: &[f32],
    incoming: &[i32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = &model.config;
    let b = c.batch_size;
    let w = 3 * c.side * c.side;
    assert_eq!(frame.len(), b * w);
    assert_eq!(incoming.len(), b);
    let mut input = Vec::new();
    for (lane, &action) in incoming.iter().enumerate() {
        input.extend(history.condition(lane, action, c.action_count, false));
        input.extend_from_slice(&frame[lane * w..(lane + 1) * w]);
    }
    model.events.set_input("event_input", &input);
    model.events.step();
    model.events.wait();
    let mut logits = vec![0.; b * c.value_bins];
    let mut cont = vec![0.; b];
    model.events.read_output_by_index(0, &mut logits);
    model.events.read_output_by_index(1, &mut cont);
    let bins = TwoHotBins::new(c.value_bins);
    let mut scratch = vec![0.; c.value_bins];
    let mut reward = Vec::new();
    let mut positive = Vec::new();
    for row in logits.chunks_exact(c.value_bins) {
        reward.push(bins.decode_logits_with_scratch(row, &mut scratch));
        positive.push(
            (scratch[c.value_bins / 2 + 1..].iter().sum::<f64>() / scratch.iter().sum::<f64>())
                as f32,
        );
    }
    let gamma = 1. - 1. / c.horizon as f32;
    let continuation = cont
        .iter()
        .map(|&v| gamma / (1. + (-v).exp()))
        .collect::<Vec<_>>();
    let mut outputs = reward;
    outputs.extend(positive);
    outputs.extend(continuation);
    assert!(
        logits
            .iter()
            .chain(&cont)
            .chain(&outputs)
            .all(|v| v.is_finite())
    );
    (logits, cont, outputs)
}

#[test]
#[ignore = "requires declared retained checkpoint/cached RGB and guarded GPU; head-only diagnosis"]
fn frozen_event_logits() {
    let started = Instant::now();
    let case: Case = serde_json::from_slice(
        &fs::read(std::env::var("KINDLE_EVENT_LOGITS_CASE").unwrap()).unwrap(),
    )
    .unwrap();
    assert!(!case.output.exists());
    let images = floats(&case.images);
    let teacher = floats(&case.teacher);
    let ordinary = floats(&case.ordinary);
    let expected = floats(&case.expected_events);
    let bytes = fs::read(&case.actions).unwrap();
    assert_eq!(bytes.len(), 32 * 8 * 4);
    let actions = bytes
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut model = DenoisingWorldLearner::restore(&case.checkpoint).unwrap();
    assert_eq!(
        model.gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let c = model.config.clone();
    let b = c.batch_size;
    let w = 3 * c.side * c.side;
    assert_eq!((b, c.side, c.value_bins, model.updates), (8, 64, 255, 2048));
    assert_eq!(c.sigma_data, 0.05);
    assert_eq!(c.log_noise_mean, -1.2);
    assert_eq!(images.len(), 32 * b * w);
    assert_eq!(teacher.len(), 4 * 27 * b * w);
    assert_eq!(ordinary.len(), 4 * 16 * b * w);
    assert_eq!(expected.len(), 4 * 3 * 16 * b);
    model.sync();
    fs::create_dir(&case.output).unwrap();
    let mut logits_out = fs::File::create(case.output.join("logits.f32")).unwrap();
    let mut cont_out = fs::File::create(case.output.join("continuation-logits.f32")).unwrap();
    let mut outputs = fs::File::create(case.output.join("outputs.f32")).unwrap();
    let mut rows = Vec::new();
    let mut batches = 0;
    // Baseline replay must match before new input comparisons are interpreted.
    for mode in ["ordinary", "actual", "forecast"] {
        let draws = if mode == "actual" { 1 } else { 4 };
        for draw in 0..draws {
            let mut history = History::empty(b, w);
            let start = if mode == "ordinary" {
                16
            } else if mode == "actual" {
                4
            } else {
                5
            };
            for t in 0..start {
                history.push(
                    &images[t * b * w..(t + 1) * b * w]
                        .iter()
                        .map(|v| 2. * v)
                        .collect::<Vec<_>>(),
                    &actions[t * b..(t + 1) * b],
                );
            }
            let mut baseline = [Vec::new(), Vec::new(), Vec::new()];
            for target in start..32 {
                let current = if mode == "ordinary" {
                    &ordinary[(draw * 16 + target - 16) * b * w..(draw * 16 + target - 15) * b * w]
                } else if mode == "forecast" {
                    &teacher[(draw * 27 + target - 5) * b * w..(draw * 27 + target - 4) * b * w]
                } else {
                    &images[target * b * w..(target + 1) * b * w]
                };
                let frame = current.iter().map(|v| 2. * v).collect::<Vec<_>>();
                let incoming = &actions[target * b..(target + 1) * b];
                let (logits, cont, values) = head(&mut model, &history, &frame, incoming);
                write_floats(&mut logits_out, &logits);
                write_floats(&mut cont_out, &cont);
                write_floats(&mut outputs, &values);
                rows.push(serde_json::json!({"mode":mode,"draw":draw,"target":target}));
                batches += 1;
                for k in 0..3 {
                    baseline[k].extend_from_slice(&values[k * b..(k + 1) * b]);
                }
                let next = if mode == "ordinary" {
                    frame
                } else {
                    images[target * b * w..(target + 1) * b * w]
                        .iter()
                        .map(|v| 2. * v)
                        .collect()
                };
                history.push(&next, incoming);
            }
            if mode == "ordinary" {
                for (k, values) in baseline.iter().enumerate() {
                    let offset = (draw * 3 + k) * 16 * b;
                    assert!(
                        values
                            .iter()
                            .zip(&expected[offset..offset + 16 * b])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "cached event baseline must be bit-exact"
                    );
                }
            }
        }
    }
    assert_eq!(batches, 200);
    assert_eq!(model.updates, 2048);
    logits_out.sync_all().unwrap();
    cont_out.sync_all().unwrap();
    outputs.sync_all().unwrap();
    write_floats(
        &mut fs::File::create(case.output.join("support.f32")).unwrap(),
        TwoHotBins::new(255).values(),
    );
    fs::write(
        case.output.join("rows.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
    model.save(&case.output.join("frozen")).unwrap();
    let result = serde_json::json!({"status":"complete","updates":2048,"no_learning":true,"head_batches":200,"rows":1600,"exact_baseline_arrays":12,"new_rgb_generation":false,"new_emulator_actions":0,"new_emulator_frames":0,"seconds":started.elapsed().as_secs_f64()});
    fs::write(
        case.output.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("{result}");
}
