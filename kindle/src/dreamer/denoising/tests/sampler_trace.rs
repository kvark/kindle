//! Frozen stage inspection; no production API, network, loss or sampler changes.
use super::super::*;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    checkpoint: PathBuf,
    images: PathBuf,
    actions: PathBuf,
    expected_final: PathBuf,
    expected_origin16: PathBuf,
    output: PathBuf,
    draws: Vec<u64>,
    targets: Vec<usize>,
}
fn floats(path: &Path) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    assert!(data.len().is_multiple_of(4));
    data.chunks_exact(4)
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
fn exact(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let different = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(different, 0, "{label}: unequal float32 values");
}

/// Receives only observed prefix pixels/actions and one incoming command per lane.
/// Expected predictions and target pixels are not arguments to this function.
fn trace_one(
    model: &mut DenoisingWorldLearner,
    images: &[f32],
    actions: &[i32],
    incoming: &[i32],
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let c = model.config.clone();
    let b = c.batch_size;
    let w = 3 * c.side * c.side;
    assert!(actions.len() >= 5 * b && actions.len().is_multiple_of(b));
    assert_eq!(images.len(), actions.len() * w);
    assert_eq!(incoming.len(), b);
    assert!(valid_images(images));
    assert!(
        incoming
            .iter()
            .all(|&a| a >= 0 && a < c.action_count as i32)
    );
    let mut history = History::empty(b, w);
    for (frame, actions) in images.chunks_exact(b * w).zip(actions.chunks_exact(b)) {
        history.push(&frame.iter().map(|v| 2. * v).collect::<Vec<_>>(), actions);
    }
    model.sync();
    let base = history.base(incoming, c.residual_frame);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = gaussian::normals(&mut rng, b * w)
        .iter()
        .map(|v| v * SIGMA_MAX)
        .collect::<Vec<_>>();
    let mut stages = Vec::new();
    for pair in schedule().windows(2) {
        let d = model.denoise(&history, incoming, &x, pair[0]).unwrap();
        stages.push(d.iter().map(|v| v * 0.5).collect());
        for (i, (v, pred)) in x.iter_mut().zip(d).enumerate() {
            let pred = coordinate(pred, base[i], c.residual_frame);
            *v += (*v - pred) / pair[0] * (pair[1] - pair[0]);
        }
    }
    let final_rgb = x
        .iter()
        .enumerate()
        .map(|(i, &v)| absolute(v, base[i], c.residual_frame) * 0.5)
        .collect();
    (stages, final_rgb)
}

#[test]
#[ignore = "requires declared input, retained checkpoint and guarded GPU; frozen sampler traces"]
fn frozen_sampler_stages() {
    let started = Instant::now();
    let case: Case = serde_json::from_slice(
        &fs::read(std::env::var("KINDLE_SAMPLER_TRACE_CASE").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(case.draws, [83001, 83003, 83009, 83023]);
    assert_eq!(case.targets, (5..32).collect::<Vec<_>>());
    assert!(
        !case.output.exists(),
        "do not overwrite an earlier diagnostic"
    );
    let images = floats(&case.images);
    let raw_actions = fs::read(&case.actions).unwrap();
    assert_eq!(raw_actions.len(), 32 * 8 * 4);
    let actions = raw_actions
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = floats(&case.expected_final);
    let expected_origin = floats(&case.expected_origin16);
    let mut model = DenoisingWorldLearner::restore(&case.checkpoint).unwrap();
    assert_eq!(
        model.gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let b = model.config.batch_size;
    let w = 3 * model.config.side * model.config.side;
    assert_eq!((b, model.config.side, model.updates), (8, 64, 2048));
    assert!(model.config.residual_frame && model.config.half_max_noise);
    assert_eq!(model.parameter_names().len(), 45);
    assert_eq!(images.len(), 32 * b * w);
    assert_eq!(expected.len(), 4 * 27 * b * w);
    assert_eq!(expected_origin.len(), 4 * (2 * b * w + 3 * b));
    fs::create_dir(&case.output).unwrap();
    let mut output = fs::File::create(case.output.join("denoised.f32")).unwrap();
    let mut finals = fs::File::create(case.output.join("final.f32")).unwrap();
    let mut origins = Vec::new();
    let mut count = 0;
    for &seed in &case.draws {
        for &target in &case.targets {
            let (stages, final_rgb) = trace_one(
                &mut model,
                &images[..target * b * w],
                &actions[..target * b],
                &actions[target * b..(target + 1) * b],
                seed,
            );
            assert_eq!(stages.len(), 3);
            exact(
                &final_rgb,
                &expected[count * b * w..(count + 1) * b * w],
                "saved teacher final",
            );
            for stage in &stages {
                write_floats(&mut output, stage);
            }
            write_floats(&mut finals, &final_rgb);
            count += 1;
        }
        let future = actions[16 * b..17 * b]
            .iter()
            .map(|&a| a as usize)
            .collect::<Vec<_>>();
        let baseline = model
            .predict(&images[..16 * b * w], &actions[..16 * b], &future, seed)
            .unwrap();
        origins.extend(baseline.observations);
        origins.extend(baseline.posterior);
        origins.extend(&baseline.rewards[0]);
        origins.extend(&baseline.reward_positive_probability[0]);
        origins.extend(&baseline.continuations[0]);
    }
    output.sync_all().unwrap();
    finals.sync_all().unwrap();
    exact(&origins, &expected_origin, "production origin16 outputs");
    write_floats(
        &mut fs::File::create(case.output.join("origin16.f32")).unwrap(),
        &origins,
    );
    assert_eq!((count, model.updates), (108, 2048));
    model.save(&case.output.join("frozen")).unwrap();
    let receipt = serde_json::json!({"status":"complete","updates":model.updates,"no_learning":true,"traces":count,"stages":3,"trace_denoiser_calls":324,"production_predict_queries":4,"exact_final_batch_arrays":108,"exact_final_images":864,"exact_origin16_model_arrays":20,"new_emulator_actions":0,"new_emulator_frames":0,"sigmas":&schedule()[..3],"seconds":started.elapsed().as_secs_f64()});
    fs::write(
        case.output.join("result.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    println!("{receipt}");
}
