//! Frozen sampling intervention, with exact cached and production baseline checks.
use super::super::*;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    checkpoint: PathBuf,
    images: PathBuf,
    actions: PathBuf,
    expected_teacher: PathBuf,
    expected_ordinary: PathBuf,
    expected_production: PathBuf,
    output: PathBuf,
    draws: Vec<u64>,
}

fn six_steps() -> [f32; 7] {
    let lo = SIGMA_MIN.powf(1. / 7.);
    let hi = SIGMA_MAX.powf(1. / 7.);
    let mut values = [0.; 7];
    for (i, v) in values[..6].iter_mut().enumerate() {
        *v = (hi + i as f32 / 5. * (lo - hi)).powi(7);
    }
    values[0] = SIGMA_MAX;
    values[5] = SIGMA_MIN;
    values
}

fn floats(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn write_floats(output: &mut impl Write, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    output.write_all(&bytes).unwrap();
}
fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "baseline must be bit-exact"
    );
}

/// Only observed prefix images and submitted future commands enter generation.
fn generate(
    model: &mut DenoisingWorldLearner,
    images: &[f32],
    actions: &[i32],
    future: &[i32],
    seed: u64,
    sigmas: &[f32],
) -> Vec<f32> {
    let c = model.config.clone();
    let b = c.batch_size;
    let w = 3 * c.side * c.side;
    assert!(actions.len() >= 5 * b && actions.len().is_multiple_of(b));
    assert_eq!(images.len(), actions.len() * w);
    assert!(valid_images(images));
    assert!(!future.is_empty() && future.len().is_multiple_of(b));
    assert!(future.iter().all(|&a| a >= 0 && a < c.action_count as i32));
    let mut history = History::empty(b, w);
    for (frame, a) in images.chunks_exact(b * w).zip(actions.chunks_exact(b)) {
        history.push(&frame.iter().map(|v| 2. * v).collect::<Vec<_>>(), a);
    }
    model.sync();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::new();
    for incoming in future.chunks_exact(b) {
        let base = history.base(incoming, c.residual_frame);
        let mut x = gaussian::normals(&mut rng, b * w)
            .iter()
            .map(|v| v * SIGMA_MAX)
            .collect::<Vec<_>>();
        for pair in sigmas.windows(2) {
            let d = model.denoise(&history, incoming, &x, pair[0]).unwrap();
            for (i, (v, pred)) in x.iter_mut().zip(d).enumerate() {
                let pred = coordinate(pred, base[i], c.residual_frame);
                *v += (*v - pred) / pair[0] * (pair[1] - pair[0]);
            }
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = absolute(*v, base[i], c.residual_frame);
        }
        result.extend(x.iter().map(|v| v * 0.5));
        history.push(&x, incoming);
    }
    result
}

#[test]
fn six_step_grid_matches_f64_and_brackets_data_scale() {
    let before = schedule();
    let values = six_steps();
    assert_eq!(values.len(), 7);
    assert_eq!((values[0], values[5], values[6]), (5., 0.002, 0.));
    assert!(values.windows(2).all(|v| v[0].is_finite() && v[0] > v[1]));
    let expected = [
        5.,
        1.817629234727216,
        0.5566730245378364,
        0.13390231634575755,
        0.0223335895321752,
        0.002,
        0.,
    ];
    for (value, expected) in values.iter().zip(expected) {
        assert!((*value as f64 - expected).abs() <= expected * 2e-6);
    }
    assert!(values[3] > 0.05 && values[4] < 0.05);
    assert!(values[3] - values[4] < before[1] - before[2]);
    assert_eq!(schedule(), before);
}

#[test]
#[ignore = "requires declared current checkpoint, cached controls and guarded GPU"]
fn frozen_six_step_intervention() {
    let started = Instant::now();
    let case: Case = serde_json::from_slice(
        &fs::read(std::env::var("KINDLE_SAMPLER_STEPS_CASE").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(case.draws, [83001, 83003, 83009, 83023]);
    assert!(!case.output.exists());
    let images = floats(&case.images);
    let bytes = fs::read(&case.actions).unwrap();
    assert_eq!(bytes.len(), 32 * 8 * 4);
    let actions = bytes
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let teacher = floats(&case.expected_teacher);
    let ordinary = floats(&case.expected_ordinary);
    let production = floats(&case.expected_production);
    let mut model = DenoisingWorldLearner::restore(&case.checkpoint).unwrap();
    assert_eq!(
        model.gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let b = model.config.batch_size;
    let w = 3 * model.config.side * model.config.side;
    assert_eq!((b, model.config.side, model.updates), (8, 64, 2048));
    assert_eq!(model.config.sigma_data, 0.05);
    assert!(model.config.residual_frame && model.config.half_max_noise);
    assert_eq!(images.len(), 32 * b * w);
    assert_eq!(teacher.len(), 4 * 27 * b * w);
    assert_eq!(ordinary.len(), 4 * 16 * b * w);
    assert_eq!(production.len(), 4 * (17 * b * w + 3 * 16 * b));
    fs::create_dir(&case.output).unwrap();
    for (steps, sigmas) in [(3, schedule().to_vec()), (6, six_steps().to_vec())] {
        let mut teacher_out =
            fs::File::create(case.output.join(format!("teacher{steps}.f32"))).unwrap();
        let mut ordinary_out =
            fs::File::create(case.output.join(format!("ordinary{steps}.f32"))).unwrap();
        for (draw, &seed) in case.draws.iter().enumerate() {
            for target in 5..32 {
                let result = generate(
                    &mut model,
                    &images[..target * b * w],
                    &actions[..target * b],
                    &actions[target * b..(target + 1) * b],
                    seed,
                    &sigmas,
                );
                if steps == 3 {
                    let offset = (draw * 27 + target - 5) * b * w;
                    exact(&result, &teacher[offset..offset + b * w]);
                }
                write_floats(&mut teacher_out, &result);
            }
            let result = generate(
                &mut model,
                &images[..16 * b * w],
                &actions[..16 * b],
                &actions[16 * b..],
                seed,
                &sigmas,
            );
            if steps == 3 {
                exact(
                    &result,
                    &ordinary[draw * 16 * b * w..(draw + 1) * 16 * b * w],
                );
            }
            write_floats(&mut ordinary_out, &result);
        }
        teacher_out.sync_all().unwrap();
        ordinary_out.sync_all().unwrap();
        if steps == 3 {
            let mut original = Vec::new();
            let future = actions[16 * b..]
                .iter()
                .map(|&v| v as usize)
                .collect::<Vec<_>>();
            for &seed in &case.draws {
                let result = model
                    .predict(&images[..16 * b * w], &actions[..16 * b], &future, seed)
                    .unwrap();
                original.extend(result.observations);
                original.extend(result.posterior);
                original.extend(result.rewards.into_iter().flatten());
                original.extend(result.reward_positive_probability.into_iter().flatten());
                original.extend(result.continuations.into_iter().flatten());
            }
            exact(&original, &production);
            write_floats(
                &mut fs::File::create(case.output.join("production3.f32")).unwrap(),
                &original,
            );
        }
    }
    assert_eq!(model.updates, 2048);
    model.save(&case.output.join("frozen")).unwrap();
    let result = serde_json::json!({"status":"complete","no_learning":true,"updates":2048,"teacher_queries_per_mode":108,"ordinary_queries_per_mode":4,"ordinary_steps_per_mode":64,"exact_teacher_batch_arrays":108,"exact_ordinary_batch_arrays":64,"production_parity_arrays":20,"sigmas6":six_steps(),"new_emulator_actions":0,"new_emulator_frames":0,"seconds":started.elapsed().as_secs_f64()});
    fs::write(
        case.output.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("{result}");
}
