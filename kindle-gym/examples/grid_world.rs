use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use kindle::{ActionMode, DreamerAgent, DreamerConfig, Environment, ModelSize};
use kindle_gym::grid_world::{ACTION_COUNT, GridWorld, POSITION_RANDOMIZATION_SEED_XOR, WIDTH};
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

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum RewardMode {
    /// The environment's sparse food reward used by the baseline.
    #[default]
    SparseFood,
    /// Diagnostic: reward every frame in either of the two rightmost columns.
    DenseRight,
}

impl RewardMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::SparseFood => "sparse_food",
            Self::DenseRight => "dense_right",
        }
    }

    fn apply(self, environment: &GridWorld, transition: &mut kindle::Transition) {
        if matches!(self, Self::DenseRight) {
            transition.reward.extrinsic = f32::from(environment.position().0 >= WIDTH - 2);
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

    /// Extrinsic reward definition. Dense-right is a visual learning probe,
    /// not a replacement for the sparse-food baseline.
    #[arg(long, value_enum, default_value_t = RewardMode::SparseFood)]
    reward_mode: RewardMode,

    /// Teleport to a uniformly random cell after each action. This makes the
    /// rendered position unpredictable from action history for visual probes.
    #[arg(long)]
    randomize_position: bool,

    /// Expose all four actions, including border no-ops. Dreamer imagination
    /// has no environment action mask, so this removes a real/prior mismatch.
    #[arg(long)]
    all_actions: bool,

    /// Override the baseline's D3 12m network preset.
    #[arg(long, value_enum)]
    model_size: Option<Size>,

    /// Override D3's 1,000-update warmup for short diagnostic runs.
    #[arg(long)]
    learning_rate_warmup: Option<u64>,

    /// Override D3's 4e-5 learning rate for short wiring diagnostics.
    #[arg(long)]
    learning_rate: Option<f32>,

    /// Override only the actor/critic rate. By default it follows
    /// --learning-rate, preserving D3's shared optimizer rate.
    #[arg(long)]
    behavior_learning_rate: Option<f32>,

    /// Override replayed samples per environment step (Atari-100k uses 256).
    #[arg(long)]
    train_ratio: Option<f32>,

    /// Override the number of prior transitions in each imagined rollout.
    #[arg(long)]
    imagination_length: Option<usize>,

    /// Override the posterior/prior free-nat information budget.
    #[arg(long)]
    free_nats: Option<f32>,

    /// Override only the dynamics/prior KL floor. By default it follows
    /// --free-nats, preserving D3's balanced-KL setting.
    #[arg(long)]
    dynamics_free_nats: Option<f32>,

    /// Override D3's dynamics/prior KL loss coefficient.
    #[arg(long)]
    dynamics_loss_scale: Option<f32>,

    /// Override the DINO-feature reconstruction loss weight.
    #[arg(long)]
    reconstruction_loss_scale: Option<f32>,

    /// Stop replay-value gradients at the RSSM boundary while retaining the
    /// behavior critic's replay-value training objective.
    #[arg(long)]
    stop_replay_value_gradient: bool,

    /// Train only the reward-prediction path. This is a wiring diagnostic,
    /// not a control-learning baseline.
    #[arg(long)]
    reward_only: bool,

    /// Force uniformly random valid actions for the first N interactions
    /// while still training the complete agent. This isolates replay coverage
    /// from representation and behavior learning.
    #[arg(long, default_value_t = 0)]
    random_action_steps: usize,

    /// Emit an aggregate interval event every N environment steps.
    #[arg(long, default_value_t = 1_000)]
    report_every: usize,

    /// Write JSONL directly to this file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

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

impl Arguments {
    fn has_training_config_override(&self) -> bool {
        self.model_size.is_some()
            || self.learning_rate.is_some()
            || self.behavior_learning_rate.is_some()
            || self.learning_rate_warmup.is_some()
            || self.train_ratio.is_some()
            || self.imagination_length.is_some()
            || self.free_nats.is_some()
            || self.dynamics_free_nats.is_some()
            || self.dynamics_loss_scale.is_some()
            || self.reconstruction_loss_scale.is_some()
            || self.stop_replay_value_gradient
            || self.reward_only
    }
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
    interval_learner_updates: usize,
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
    if arguments.randomize_position && !matches!(arguments.reward_mode, RewardMode::DenseRight) {
        return Err("--randomize-position requires --reward-mode dense-right".into());
    }
    if arguments.random {
        if arguments.dino_checkpoint.is_some()
            || arguments.restore.is_some()
            || arguments.checkpoint.is_some()
            || arguments.has_training_config_override()
            || arguments.random_action_steps != 0
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
        if arguments.restore.is_some() && arguments.has_training_config_override() {
            return Err("training overrides cannot change restored configuration".into());
        }
        if arguments.evaluate && arguments.restore.is_none() {
            return Err("--evaluate requires --restore".into());
        }
        if arguments.evaluate && arguments.checkpoint.is_some() {
            return Err("--evaluate does not write a training checkpoint".into());
        }
        if arguments.evaluate && arguments.random_action_steps != 0 {
            return Err("--evaluate does not accept --random-action-steps".into());
        }
        if arguments.reward_only
            && (arguments.reconstruction_loss_scale.is_some()
                || arguments.dynamics_loss_scale.is_some())
        {
            return Err("--reward-only already fixes all loss weights".into());
        }
        run_dreamer(&arguments, dino_checkpoint)
    }
}

fn run_random(arguments: &Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = open_output(arguments)?;
    emit(
        &mut output,
        &json!({
            "event": "run_start",
            "agent": "random",
            "steps": arguments.steps,
            "seed": arguments.seed,
            "reward_mode": arguments.reward_mode.as_str(),
            "randomize_position": arguments.randomize_position,
            "all_actions": arguments.all_actions,
            "random_action_steps": arguments.steps,
            "actions": ACTION_NAMES,
        }),
    )?;

    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let mut position_rng = StdRng::seed_from_u64(arguments.seed ^ POSITION_RANDOMIZATION_SEED_XOR);
    let mut environment = GridWorld::new();
    environment.reset();
    let mut metrics = RunMetrics::default();

    for run_step in 1..=arguments.steps {
        let mask = if arguments.all_actions {
            None
        } else {
            environment.action_mask()
        };
        let action = random_valid_action(mask.as_deref(), &mut rng);
        let mut transition = environment.step(action);
        if arguments.randomize_position {
            environment.randomize_position(&mut transition, &mut position_rng);
        }
        arguments.reward_mode.apply(&environment, &mut transition);
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
    if let Some(model_size) = arguments.model_size {
        config.model_size = model_size.into();
    }
    config.seed = arguments.seed;
    if let Some(learning_rate) = arguments.learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(learning_rate) = arguments.behavior_learning_rate {
        config.behavior_learning_rate = Some(learning_rate);
    }
    if let Some(warmup) = arguments.learning_rate_warmup {
        config.learning_rate_warmup = warmup;
    }
    if let Some(train_ratio) = arguments.train_ratio {
        config.train_ratio = train_ratio;
    }
    if let Some(imagination_length) = arguments.imagination_length {
        config.imagination_length = imagination_length;
    }
    if let Some(free_nats) = arguments.free_nats {
        config.free_nats = free_nats;
    }
    if let Some(free_nats) = arguments.dynamics_free_nats {
        config.dynamics_free_nats = Some(free_nats);
    }
    if let Some(scale) = arguments.dynamics_loss_scale {
        config.loss_scales.dynamics = scale;
    }
    if let Some(scale) = arguments.reconstruction_loss_scale {
        config.loss_scales.reconstruction = scale;
    }
    if arguments.stop_replay_value_gradient {
        config.replay_value_gradient = false;
    }
    if arguments.reward_only {
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.continuation = 0.0;
        config.loss_scales.dynamics = 0.0;
        config.loss_scales.representation = 0.0;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
    }
    let mut agent = if let Some(checkpoint) = &arguments.restore {
        DreamerAgent::restore(checkpoint, dino_checkpoint, None)?
    } else {
        DreamerAgent::new(config, dino_checkpoint, None)?
    };

    let mut output = open_output(arguments)?;
    let starting_environment_step = agent.core().environment_step();
    let starting_learner_step = agent.core().learner_step();
    let (world_parameters, behavior_parameters) = agent.core().trainable_parameter_counts();
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
            "reward_mode": arguments.reward_mode.as_str(),
            "randomize_position": arguments.randomize_position,
            "all_actions": arguments.all_actions,
            "random_action_steps": arguments.random_action_steps,
            "trainable_parameters": {
                "world": world_parameters,
                "behavior": behavior_parameters,
                "total": world_parameters + behavior_parameters,
            },
            "config": agent.core().config(),
            "actions": ACTION_NAMES,
        }),
    )?;

    let started = Instant::now();
    let mut environment = GridWorld::new();
    let first = environment.reset();
    agent.begin_episode(&first);
    let mut metrics = RunMetrics::default();
    // A separate RNG keeps the agent's categorical samples unchanged while
    // matching the standalone random-policy trajectory for the same seed.
    let mut exploration_rng = StdRng::seed_from_u64(arguments.seed);
    let mut position_rng = StdRng::seed_from_u64(arguments.seed ^ POSITION_RANDOMIZATION_SEED_XOR);

    for run_step in 1..=arguments.steps {
        let mask = if arguments.all_actions {
            None
        } else {
            environment.action_mask()
        };
        let action = if run_step <= arguments.random_action_steps {
            let action = random_valid_action(mask.as_deref(), &mut exploration_rng);
            let mut forced_mask = [false; ACTION_COUNT];
            forced_mask[action] = true;
            agent.act(ActionMode::Sample, Some(&forced_mask))
        } else {
            let mode = if arguments.evaluate {
                ActionMode::Greedy
            } else {
                ActionMode::Sample
            };
            agent.act(mode, mask.as_deref())
        };
        let mut transition = environment.step(action);
        if arguments.randomize_position {
            environment.randomize_position(&mut transition, &mut position_rng);
        }
        arguments.reward_mode.apply(&environment, &mut transition);
        let ended = transition.terminated || transition.truncated;
        agent.observe(&transition);
        if !arguments.evaluate {
            for report in agent.learn_scheduled(1) {
                metrics.learner_updates += 1;
                metrics.interval_learner_updates += 1;
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

fn random_valid_action(mask: Option<&[bool]>, rng: &mut StdRng) -> usize {
    let valid = (0..ACTION_COUNT)
        .filter(|&action| mask.is_none_or(|mask| mask[action]))
        .collect::<Vec<_>>();
    valid[rng.random_range(0..valid.len())]
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
                "learner_updates": metrics.interval_learner_updates,
                "action_counts": metrics.interval_action_counts,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
            }),
        )?;
        metrics.interval_reward = 0.0;
        metrics.interval_episodes = 0;
        metrics.interval_learner_updates = 0;
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
    writeln!(output)?;
    output.flush()
}

fn open_output(arguments: &Arguments) -> io::Result<Box<dyn Write>> {
    if let Some(path) = &arguments.output {
        Ok(Box::new(BufWriter::new(File::create(path)?)))
    } else {
        Ok(Box::new(BufWriter::new(io::stdout())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_right_reward_is_determined_by_rendered_position() {
        let mut environment = GridWorld::new();
        environment.reset();
        for expected in [0.0, 0.0, 1.0] {
            let mut transition = environment.step(3);
            RewardMode::DenseRight.apply(&environment, &mut transition);
            assert_eq!(transition.reward.extrinsic, expected);
        }
    }

    #[test]
    fn random_action_respects_the_environment_mask() {
        let mut rng = StdRng::seed_from_u64(7);
        let mask = [false, true, false, true];
        for _ in 0..100 {
            assert!(matches!(random_valid_action(Some(&mask), &mut rng), 1 | 3));
        }
    }
}
