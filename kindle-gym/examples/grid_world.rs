use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use kindle::{ActionMode, DreamerAgent, DreamerConfig, Environment, ModelSize};
use kindle_gym::grid_world::{ACTION_COUNT, GridWorld};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;

const ACTION_NAMES: [&str; ACTION_COUNT] = ["up", "down", "left", "right"];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Size {
    Tiny,
    #[value(name = "1m")]
    OneMillion,
    #[value(name = "12m")]
    TwelveMillion,
    #[value(name = "25m")]
    TwentyFiveMillion,
    #[value(name = "50m")]
    FiftyMillion,
    #[value(name = "100m")]
    OneHundredMillion,
    #[value(name = "200m")]
    TwoHundredMillion,
}

impl From<Size> for ModelSize {
    fn from(size: Size) -> Self {
        match size {
            Size::Tiny => Self::Tiny,
            Size::OneMillion => Self::Size1M,
            Size::TwelveMillion => Self::Size12M,
            Size::TwentyFiveMillion => Self::Size25M,
            Size::FiftyMillion => Self::Size50M,
            Size::OneHundredMillion => Self::Size100M,
            Size::TwoHundredMillion => Self::Size200M,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run a measured Dreamer or random-policy GridWorld baseline")]
struct Arguments {
    /// DINOv3 ViT-S/16 model.safetensors; omitted with --random.
    #[arg(value_name = "DINO_CHECKPOINT")]
    dino_checkpoint: Option<PathBuf>,

    /// Run a uniform random valid-action control without constructing Dreamer.
    #[arg(long)]
    random: bool,

    /// Number of environment interactions in this run.
    #[arg(long, default_value_t = 100_000)]
    steps: usize,

    /// Agent or random-policy seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Dreamer network preset. The first baseline uses 1m.
    #[arg(long, value_enum, default_value_t = Size::OneMillion)]
    model_size: Size,

    /// Override D3's 1,000-update warmup for short diagnostic runs.
    #[arg(long)]
    learning_rate_warmup: Option<u64>,

    /// Override replayed samples per environment step (Atari-100k uses 256).
    #[arg(long)]
    train_ratio: Option<f32>,

    /// Emit an aggregate interval event every N environment steps.
    #[arg(long, default_value_t = 1_000)]
    report_every: usize,

    /// Restore model/optimizer state from this checkpoint. Replay refills fresh.
    #[arg(long, value_name = "DIRECTORY")]
    restore: Option<PathBuf>,

    /// Run a frozen greedy policy evaluation from --restore without learning.
    #[arg(long)]
    evaluate: bool,

    /// Save model/optimizer state to this checkpoint after the run.
    #[arg(long, value_name = "DIRECTORY")]
    checkpoint: Option<PathBuf>,
}

#[derive(Default)]
struct RunMetrics {
    total_reward: f32,
    interval_reward: f32,
    completed_return_sum: f32,
    episode_return: f32,
    episode_length: usize,
    episodes: usize,
    interval_episodes: usize,
    learner_updates: usize,
    action_counts: [usize; ACTION_COUNT],
    interval_action_counts: [usize; ACTION_COUNT],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arguments = Arguments::parse();
    if arguments.steps == 0 {
        return Err("--steps must be positive".into());
    }
    if arguments.report_every == 0 {
        return Err("--report-every must be positive".into());
    }
    if arguments.random {
        if arguments.dino_checkpoint.is_some()
            || arguments.restore.is_some()
            || arguments.checkpoint.is_some()
            || arguments.learning_rate_warmup.is_some()
            || arguments.train_ratio.is_some()
            || arguments.evaluate
        {
            return Err(
                "--random does not accept Dreamer checkpoints or configuration overrides".into(),
            );
        }
        run_random(&arguments)
    } else {
        let dino_checkpoint = arguments
            .dino_checkpoint
            .as_ref()
            .ok_or("DINO_CHECKPOINT is required unless --random is used")?;
        if arguments.restore.is_some()
            && (arguments.learning_rate_warmup.is_some() || arguments.train_ratio.is_some())
        {
            return Err("training overrides cannot change restored configuration".into());
        }
        if arguments.evaluate && arguments.restore.is_none() {
            return Err("--evaluate requires --restore".into());
        }
        if arguments.evaluate && arguments.checkpoint.is_some() {
            return Err("--evaluate does not write a training checkpoint".into());
        }
        run_dreamer(&arguments, dino_checkpoint)
    }
}

fn run_random(arguments: &Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    emit(
        &mut output,
        &json!({
            "event": "run_start",
            "agent": "random",
            "steps": arguments.steps,
            "seed": arguments.seed,
            "actions": ACTION_NAMES,
        }),
    )?;

    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let mut environment = GridWorld::new();
    environment.reset();
    let mut metrics = RunMetrics::default();

    for run_step in 1..=arguments.steps {
        let mask = environment.action_mask().expect("GridWorld action mask");
        let valid = mask
            .iter()
            .enumerate()
            .filter_map(|(action, enabled)| enabled.then_some(action))
            .collect::<Vec<_>>();
        let action = valid[rng.random_range(0..valid.len())];
        let transition = environment.step(action);
        record_transition(
            &mut output,
            &mut metrics,
            run_step,
            action,
            &transition,
            arguments.report_every,
            started,
        )?;
        if transition.terminated || transition.truncated {
            environment.reset();
        }
    }
    emit_summary(
        &mut output,
        "random",
        arguments.steps,
        &metrics,
        started,
        arguments.steps as u64,
        0,
    )
}

fn run_dreamer(
    arguments: &Arguments,
    dino_checkpoint: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = DreamerConfig::new(ACTION_COUNT);
    config.model_size = arguments.model_size.into();
    config.seed = arguments.seed;
    if let Some(warmup) = arguments.learning_rate_warmup {
        config.learning_rate_warmup = warmup;
    }
    if let Some(train_ratio) = arguments.train_ratio {
        config.train_ratio = train_ratio;
    }
    let mut agent = if let Some(checkpoint) = &arguments.restore {
        DreamerAgent::restore(checkpoint, dino_checkpoint, None)?
    } else {
        DreamerAgent::new(config, dino_checkpoint, None)?
    };

    let mut output = io::BufWriter::new(io::stdout().lock());
    let starting_environment_step = agent.core().environment_step();
    let starting_learner_step = agent.core().learner_step();
    let agent_name = if arguments.evaluate {
        "dreamer_eval"
    } else {
        "dreamer"
    };
    emit(
        &mut output,
        &json!({
            "event": "run_start",
            "agent": agent_name,
            "steps": arguments.steps,
            "restored": arguments.restore.is_some(),
            "starting_environment_step": starting_environment_step,
            "starting_learner_step": starting_learner_step,
            "config": agent.core().config(),
            "actions": ACTION_NAMES,
        }),
    )?;

    let started = Instant::now();
    let mut environment = GridWorld::new();
    let first = environment.reset();
    agent.begin_episode(&first);
    let mut metrics = RunMetrics::default();

    for run_step in 1..=arguments.steps {
        let mask = environment.action_mask();
        let mode = if arguments.evaluate {
            ActionMode::Greedy
        } else {
            ActionMode::Sample
        };
        let action = agent.act(mode, mask.as_deref());
        let transition = environment.step(action);
        let ended = transition.terminated || transition.truncated;
        agent.observe(&transition);
        if !arguments.evaluate {
            for report in agent.learn_scheduled(1) {
                metrics.learner_updates += 1;
                emit(
                    &mut output,
                    &json!({
                        "event": "learner",
                        "run_step": run_step,
                        "environment_step": agent.core().environment_step(),
                        "elapsed_seconds": started.elapsed().as_secs_f64(),
                        "report": report,
                    }),
                )?;
            }
        }
        record_transition(
            &mut output,
            &mut metrics,
            run_step,
            action,
            &transition,
            arguments.report_every,
            started,
        )?;
        if ended && run_step < arguments.steps {
            agent.begin_episode(&environment.reset());
        }
    }

    if let Some(checkpoint) = &arguments.checkpoint {
        agent.save_checkpoint(checkpoint)?;
        emit(
            &mut output,
            &json!({
                "event": "checkpoint",
                "path": checkpoint,
                "environment_step": agent.core().environment_step(),
                "learner_step": agent.core().learner_step(),
            }),
        )?;
    }
    emit_summary(
        &mut output,
        agent_name,
        arguments.steps,
        &metrics,
        started,
        agent.core().environment_step() - starting_environment_step,
        agent.core().learner_step() - starting_learner_step,
    )
}

fn record_transition(
    output: &mut impl Write,
    metrics: &mut RunMetrics,
    run_step: usize,
    action: usize,
    transition: &kindle::Transition,
    report_every: usize,
    started: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    metrics.action_counts[action] += 1;
    metrics.interval_action_counts[action] += 1;
    let reward = transition.reward.extrinsic;
    metrics.total_reward += reward;
    metrics.interval_reward += reward;
    metrics.episode_return += reward;
    metrics.episode_length += 1;

    if transition.terminated || transition.truncated {
        metrics.episodes += 1;
        metrics.interval_episodes += 1;
        metrics.completed_return_sum += metrics.episode_return;
        emit(
            output,
            &json!({
                "event": "episode",
                "run_step": run_step,
                "episode": metrics.episodes,
                "length": metrics.episode_length,
                "return": metrics.episode_return,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
            }),
        )?;
        metrics.episode_return = 0.0;
        metrics.episode_length = 0;
    }

    if run_step.is_multiple_of(report_every) {
        emit(
            output,
            &json!({
                "event": "interval",
                "run_step": run_step,
                "interval_steps": report_every,
                "reward": metrics.interval_reward,
                "completed_episodes": metrics.interval_episodes,
                "learner_updates": metrics.learner_updates,
                "action_counts": metrics.interval_action_counts,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
            }),
        )?;
        metrics.interval_reward = 0.0;
        metrics.interval_episodes = 0;
        metrics.interval_action_counts.fill(0);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_summary(
    output: &mut impl Write,
    agent: &str,
    steps: usize,
    metrics: &RunMetrics,
    started: Instant,
    environment_step_delta: u64,
    learner_step_delta: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let seconds = started.elapsed().as_secs_f64();
    emit(
        output,
        &json!({
            "event": "run_end",
            "agent": agent,
            "steps": steps,
            "total_reward": metrics.total_reward,
            "completed_episodes": metrics.episodes,
            "mean_completed_return": (metrics.episodes > 0)
                .then(|| metrics.completed_return_sum / metrics.episodes as f32),
            "partial_episode_return": metrics.episode_return,
            "partial_episode_length": metrics.episode_length,
            "learner_updates": metrics.learner_updates,
            "action_counts": metrics.action_counts,
            "environment_step_delta": environment_step_delta,
            "learner_step_delta": learner_step_delta,
            "elapsed_seconds": seconds,
            "environment_steps_per_second": steps as f64 / seconds,
            "learner_updates_per_second": metrics.learner_updates as f64 / seconds,
        }),
    )?;
    Ok(())
}

fn emit(output: &mut impl Write, value: &serde_json::Value) -> io::Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)
}
