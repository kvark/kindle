//! Construct a production-size Dreamer core and optionally execute one update.
//!
//! Example:
//! `cargo run --release -p kindle --example dreamer_canary -- 12m 64 4 --learn`
//! Set `MEGANEURA_DEVICE_ID` when the host has multiple adapters.

use std::{env, fs, path::PathBuf, sync::Arc, time::Instant};

use kindle::vision::Observation;
use kindle::{ActionMode, DreamerConfig, DreamerCore, FrameFlags, ModelSize, Reward};

fn parse_model_size(value: &str) -> ModelSize {
    match value {
        "1m" => ModelSize::Size1M,
        "12m" => ModelSize::Size12M,
        other => panic!("unsupported model size {other:?}; use 1m or 12m"),
    }
}

fn parse_usize(name: &str, value: Option<String>) -> usize {
    value
        .unwrap_or_else(|| panic!("missing {name}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {name}"))
}

fn main() {
    let mut args = env::args().skip(1);
    let model_size = parse_model_size(&args.next().expect("missing model size"));
    let world_backprop_length = parse_usize("world backprop length", args.next());
    let world_microbatch_size = parse_usize("world microbatch size", args.next());
    let mut run_learner = false;
    let mut repetitions = 1usize;
    let mut updates = 1usize;
    let mut profile_directory = None;
    let mut trace_directory = None;
    let mut checkpoint = None;
    let mut prediction_only = false;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--learn" => run_learner = true,
            "--prediction-only" => prediction_only = true,
            "--repeat" => repetitions = parse_usize("repetition count", args.next()),
            "--updates" => updates = parse_usize("learner update count", args.next()),
            "--profile-dir" => {
                profile_directory = Some(args.next().expect("missing profile directory"))
            }
            "--trace-dir" => trace_directory = Some(args.next().expect("missing trace directory")),
            "--checkpoint" => checkpoint = Some(args.next().expect("missing checkpoint path")),
            other => panic!(
                "unknown option {other:?}; use --learn, --prediction-only, --updates N, --repeat N, --profile-dir PATH, --trace-dir PATH or --checkpoint PATH"
            ),
        }
    }

    let mut config = DreamerConfig::new(18);
    config.model_size = model_size;
    config.batch_size = 16;
    config.batch_length = 64;
    config.world_backprop_length = world_backprop_length;
    config.world_microbatch_size = Some(world_microbatch_size);
    config.replay_capacity = 2_048;
    if prediction_only {
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        config.train_ratio = 256.0;
    }
    config.validate();
    assert!(repetitions > 0 && updates > 0);
    assert!(
        checkpoint.is_none() || (run_learner && repetitions == 1),
        "saving requires --learn and one repetition"
    );
    let trace = prepare_trace(
        trace_directory.as_deref(),
        run_learner,
        repetitions,
        profile_directory.is_some(),
        &config,
    );
    eprintln!("config={}", serde_json::to_string(&config).unwrap());

    eprintln!(
        "constructing model={model_size:?} bptt={world_backprop_length} microbatch={world_microbatch_size}"
    );
    let gpu = Arc::new(kindle::init_gpu_context().expect("GPU initialization failed"));
    report_memory("before construction", &gpu);

    for iteration in 0..repetitions {
        let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let device = core.gpu_device();
        eprintln!(
            "iteration {iteration} constructed on {} ({}, {})",
            device.device_name, device.driver_name, device.driver_info
        );
        report_memory("after construction", &gpu);

        if run_learner {
            fill_synthetic_replay(&mut core, &config);
            for _ in 0..updates {
                let started = Instant::now();
                let report = core
                    .learn()
                    .expect("synthetic replay should contain one batch");
                println!("{}", serde_json::to_string(&report).unwrap());
                eprintln!(
                    "iteration {iteration} learned step={} elapsed={:?}",
                    report.learner_step,
                    started.elapsed(),
                );
            }
            report_memory("after learner update", &gpu);
            if let Some(directory) = &profile_directory {
                core.profile_sessions(directory)
                    .expect("profile model sessions");
            }
            if let Some(path) = &checkpoint {
                core.save_checkpoint(path).expect("save canary state");
            }
        }

        drop(core);
        report_memory("after core drop", &gpu);
    }

    drop(gpu);
    eprintln!("GPU context dropped");
    if let Some(directory) = trace {
        // Session and transfer drops harvest their final pending submissions.
        meganeura::profiler::record_instant("kindle_capture_complete");
        meganeura::profiler::save(directory.join("trace.pftrace")).expect("save learner trace");
    }
}

fn validate_trace(
    enabled: bool,
    feature: bool,
    timing: bool,
    learning: bool,
    repetitions: usize,
    session_profiles: bool,
) -> Result<(), &'static str> {
    if !enabled {
        return Ok(());
    }
    if !feature {
        return Err("--trace-dir requires building with --features profiler");
    }
    if !timing {
        return Err("--trace-dir requires MEGANEURA_GPU_TIMING=1 before GPU construction");
    }
    if !learning || repetitions != 1 {
        return Err("--trace-dir requires --learn and one repetition");
    }
    if session_profiles {
        return Err("collect --profile-dir separately; it changes dispatch grouping");
    }
    Ok(())
}

fn prepare_trace(
    directory: Option<&str>,
    learning: bool,
    repetitions: usize,
    session_profiles: bool,
    config: &DreamerConfig,
) -> Option<PathBuf> {
    let directory = PathBuf::from(directory?);
    validate_trace(
        true,
        cfg!(feature = "profiler"),
        meganeura::GpuOptions::from_env().timing,
        learning,
        repetitions,
        session_profiles,
    )
    .expect("invalid trace configuration");
    fs::create_dir(&directory).expect("trace directory must be fresh");
    let file = fs::File::create_new(directory.join("contract.json")).expect("fresh trace contract");
    serde_json::to_writer_pretty(
        file,
        &serde_json::json!({
            "schema_version": 1,
            "scope": "synthetic_core_only",
            "gpu_timing": true,
            "per_dispatch_profiling": false,
            "config": config,
            "timing_contract": "calibrated pass-start to next-start or submission completion; not instruction-level kernel time",
            "speedup_assessed": false,
            "coverage_qualified": false,
        }),
    )
    .expect("write trace contract");
    meganeura::profiler::init_with_targets(&["kindle"]);
    meganeura::profiler::record_instant("kindle_capture_begin");
    Some(directory)
}

#[cfg(test)]
mod tests {
    use super::validate_trace;

    #[test]
    fn absent_trace_keeps_the_original_modes() {
        assert_eq!(validate_trace(false, false, false, false, 7, true), Ok(()));
    }

    #[test]
    fn trace_requires_explicit_feature_and_gpu_timing() {
        assert!(validate_trace(true, false, true, true, 1, false).is_err());
        assert!(validate_trace(true, true, false, true, 1, false).is_err());
        assert_eq!(validate_trace(true, true, true, true, 1, false), Ok(()));
    }

    #[test]
    fn trace_does_not_mix_repeated_or_per_dispatch_profiles() {
        assert!(validate_trace(true, true, true, false, 1, false).is_err());
        assert!(validate_trace(true, true, true, true, 2, false).is_err());
        assert!(validate_trace(true, true, true, true, 1, true).is_err());
    }
}

fn fill_synthetic_replay(core: &mut DreamerCore, config: &DreamerConfig) {
    let observation =
        |step: usize| Observation::from_vec(vec![step as f32 / 1_000.0; Observation::LEN]);
    core.begin_episode(observation(0));
    for step in 1..=config.batch_length {
        core.act(ActionMode::Greedy, None);
        core.observe(
            observation(step),
            Reward {
                extrinsic: if step.is_multiple_of(17) { 1.0 } else { 0.0 },
                intrinsic: 0.0,
            },
            FrameFlags::default(),
        );
    }
}

fn report_memory(stage: &str, gpu: &blade_graphics::Context) {
    let stats = gpu.memory_stats();
    eprintln!(
        "device memory {stage}: usage={} budget={}",
        stats.usage, stats.budget
    );
}
