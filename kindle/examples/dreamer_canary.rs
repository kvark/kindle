//! Construct a production-size Dreamer core and optionally execute one update.
//!
//! Example:
//! `cargo run --release -p kindle --example dreamer_canary -- 12m 64 4 --learn`
//! Set `MEGANEURA_DEVICE_ID` when the host has multiple adapters.

use std::{env, sync::Arc, time::Instant};

use kindle::vision::DinoObservation;
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
    while let Some(option) = args.next() {
        match option.as_str() {
            "--learn" => run_learner = true,
            "--repeat" => repetitions = parse_usize("repetition count", args.next()),
            other => panic!("unknown option {other:?}; use --learn or --repeat N"),
        }
    }

    let mut config = DreamerConfig::new(18);
    config.model_size = model_size;
    config.batch_size = 16;
    config.batch_length = 64;
    config.world_backprop_length = world_backprop_length;
    config.world_microbatch_size = Some(world_microbatch_size);
    config.replay_capacity = 2_048;
    config.validate();

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
            let started = Instant::now();
            let report = core
                .learn()
                .expect("synthetic replay should contain one batch");
            eprintln!(
                "iteration {iteration} learned step={} elapsed={:?} world_loss={} behavior_loss={}",
                report.learner_step,
                started.elapsed(),
                report.world.total_loss,
                report.behavior.total_loss,
            );
            report_memory("after learner update", &gpu);
        }

        drop(core);
        report_memory("after core drop", &gpu);
    }

    drop(gpu);
    eprintln!("GPU context dropped");
}

fn fill_synthetic_replay(core: &mut DreamerCore, config: &DreamerConfig) {
    let observation =
        |step: usize| DinoObservation::from_vec(vec![step as f32 / 1_000.0; DinoObservation::LEN]);
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
