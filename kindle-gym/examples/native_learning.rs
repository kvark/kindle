//! Seeded extrinsic- and intrinsic-reward gates for native environments.
//!
//! The default invocation mirrors the Python CartPole gate and exits with
//! status 2 unless the recent mean return reaches 195:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning
//! ```
//!
//! The intrinsic-only CartPole gate uses 50k training decisions followed by a
//! frozen 160k-decision greedy evaluation on each of three exploration seeds:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env cartpole --steps 26250 --evaluation-steps 20000 \
//!   --seeds 42,43,44 --recent-window 200 --log-every 6250 \
//!   --extrinsic-alpha 0 --reward-homeostatic 2 \
//!   --per-step-grpo --no-sil
//! ```
//!
//! The same harness can exercise the other episodic discrete native games:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env taxi --steps 100000 --require-recent-return 0
//! ```
//!
//! A shaped, legal-action-masked Taxi completion gate is:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env taxi --steps 100000 --recent-window 25 \
//!   --reward-homeostatic 0.5 --per-step-grpo --action-masks \
//!   --sil-event-filter --sil-loss-coef 2 \
//!   --require-recent-success-rate 0.9
//! ```
//!
//! Acrobot benefits from collecting several coherent repeat-four successes
//! before reducing persistence to two for finer control:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env acrobot --steps 30000 --recent-window 25 \
//!   --extrinsic-alpha 0.001 --no-grpo --no-advantage-normalize \
//!   --action-repeat 4 --action-repeat-after-success 2 \
//!   --action-repeat-switch-successes 50 \
//!   --sil-event-filter --sil-loss-coef 16 --sil-buffer-capacity 100000 \
//!   --require-recent-success-rate 0.9
//! ```
//!
//! MountainCar uses explicit goal events because its successful terminal step
//! still returns -1. Coherent repeated actions discover momentum-building
//! trajectories; after the first success, a shorter repeat improves control:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env mountain-car --steps 50000 --recent-window 25 \
//!   --extrinsic-alpha 0.001 --no-grpo --no-advantage-normalize \
//!   --action-repeat 8 --action-repeat-after-success 4 \
//!   --sil-event-filter --sil-loss-coef 4 --sil-buffer-capacity 50000 \
//!   --require-recent-success-rate 0.9
//! ```
//!
//! Pendulum uses continuous actions. A deterministic expert baseline and an
//! online expert-correction (DAgger) gate are available separately:
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_learning -- \
//!   --env pendulum --steps 102000 --evaluation-steps 2000 \
//!   --pendulum-dagger --require-recent-success-rate 0.9
//! ```

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter, Observation, RewardWeights};
use kindle_gym::{
    acrobot::Acrobot, cart_pole::CartPole, mountain_car::MountainCar, pendulum::Pendulum,
    taxi::Taxi,
};
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Game {
    CartPole,
    MountainCar,
    Acrobot,
    Taxi,
    Pendulum,
}

impl Game {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cartpole" | "cart-pole" => Some(Self::CartPole),
            "mountaincar" | "mountain-car" => Some(Self::MountainCar),
            "acrobot" => Some(Self::Acrobot),
            "taxi" => Some(Self::Taxi),
            "pendulum" => Some(Self::Pendulum),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::CartPole => "CartPole",
            Self::MountainCar => "MountainCar",
            Self::Acrobot => "Acrobot",
            Self::Taxi => "Taxi",
            Self::Pendulum => "Pendulum",
        }
    }

    fn observation_dim(self) -> usize {
        match self {
            Self::CartPole => 4,
            Self::MountainCar => 2,
            Self::Acrobot => 6,
            Self::Taxi => kindle_gym::taxi::OBS_DIM,
            Self::Pendulum => 3,
        }
    }

    fn num_actions(self) -> usize {
        match self {
            Self::CartPole => 2,
            Self::MountainCar | Self::Acrobot => 3,
            Self::Taxi => kindle_gym::taxi::NUM_ACTIONS,
            Self::Pendulum => 1,
        }
    }

    fn new_environment(self) -> Box<dyn Environment> {
        match self {
            Self::CartPole => Box::new(CartPole::new()),
            Self::MountainCar => Box::new(MountainCar::new()),
            Self::Acrobot => Box::new(Acrobot::new()),
            Self::Taxi => Box::new(Taxi::new()),
            Self::Pendulum => Box::new(Pendulum::new()),
        }
    }
}

fn pendulum_expert_action(observation: &Observation) -> Action {
    assert!(observation.data.len() >= 3);
    let theta = observation.data[1].atan2(observation.data[0]);
    let theta_dot = observation.data[2] * 8.0;
    let physical_torque = if theta.abs() < 0.5 {
        (-8.0 * theta - 2.0 * theta_dot).clamp(-2.0, 2.0)
    } else if theta_dot.abs() < 1e-5 || theta_dot * theta.cos() < 0.0 {
        2.0
    } else {
        -2.0
    };
    Action::Continuous(vec![physical_torque / 2.0])
}

#[derive(Clone, Debug)]
struct Args {
    game: Game,
    lanes: usize,
    steps: usize,
    seed: u64,
    seeds: Option<Vec<u64>>,
    log_every: usize,
    recent_window: usize,
    action_repeat: usize,
    action_repeat_after_success: Option<usize>,
    action_repeat_switch_successes: usize,
    required_recent_return: Option<f32>,
    required_recent_success_rate: Option<f32>,
    extrinsic_alpha: f32,
    reward_surprise: f32,
    reward_novelty: f32,
    reward_homeostatic: f32,
    reward_order: f32,
    entropy_beta: f32,
    advantage_normalize: bool,
    use_grpo: bool,
    episode_grpo: bool,
    use_sil: bool,
    sil_loss_coef: f32,
    sil_buffer_capacity: usize,
    sil_event_filter: bool,
    use_action_masks: bool,
    lr_drop_on_success_rate: Option<f32>,
    use_adam: bool,
    random_policy: bool,
    pendulum_expert: bool,
    pendulum_dagger: bool,
    pendulum_success_return: f32,
    deterministic_eval: bool,
    evaluation_steps: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            game: Game::CartPole,
            lanes: 8,
            steps: 100_000,
            seed: 42,
            seeds: None,
            log_every: 5_000,
            recent_window: 5,
            action_repeat: 1,
            action_repeat_after_success: None,
            action_repeat_switch_successes: 1,
            required_recent_return: Some(195.0),
            required_recent_success_rate: None,
            extrinsic_alpha: 1.0,
            reward_surprise: 0.0,
            reward_novelty: 0.0,
            reward_homeostatic: 0.0,
            reward_order: 0.0,
            entropy_beta: 0.01,
            advantage_normalize: true,
            use_grpo: true,
            episode_grpo: true,
            use_sil: true,
            sil_loss_coef: 1.0,
            sil_buffer_capacity: 10_000,
            sil_event_filter: false,
            use_action_masks: false,
            lr_drop_on_success_rate: None,
            use_adam: false,
            random_policy: false,
            pendulum_expert: false,
            pendulum_dagger: false,
            pendulum_success_return: -400.0,
            deterministic_eval: false,
            evaluation_steps: 0,
        }
    }
}

fn usage() {
    println!(
        "native_learning [--env cartpole|mountain-car|acrobot|taxi|pendulum] \
         [--lanes N] [--steps N] [--seed N|--seeds N,N,...] [--log-every N] \
         [--recent-window N] \
         [--action-repeat N] \
         [--action-repeat-after-success N] \
         [--action-repeat-switch-successes N] \
         [--require-recent-return R] [--require-recent-success-rate R] \
         [--no-gate] [--extrinsic-alpha W] \
         [--reward-surprise W] [--reward-novelty W] \
         [--reward-homeostatic W] [--reward-order W] \
         [--entropy-beta W] \
         [--no-advantage-normalize] [--no-grpo] \
         [--per-step-grpo] [--no-sil] [--sil-loss-coef W] \
         [--sil-buffer-capacity N] \
         [--sil-event-filter] [--action-masks] \
         [--lr-drop-on-success-rate R] [--use-adam] \
         [--random-policy] [--pendulum-expert] [--pendulum-dagger] \
         [--pendulum-success-return R] [--deterministic-eval] \
         [--evaluation-steps N]"
    );
}

fn parse_value<T: std::str::FromStr>(flag: &str, value: Option<String>) -> Result<T, String> {
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_seeds(value: Option<String>) -> Result<Vec<u64>, String> {
    let value = value.ok_or_else(|| "--seeds requires a value".to_owned())?;
    let seeds = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u64>()
                .map_err(|_| format!("invalid seed in --seeds: {part}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if seeds.is_empty() {
        return Err("--seeds requires at least one seed".into());
    }
    let mut unique = seeds.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != seeds.len() {
        return Err("--seeds values must be unique".into());
    }
    Ok(seeds)
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                usage();
                return Ok(None);
            }
            "--env" => {
                let value = args.next().ok_or("--env requires a value")?;
                parsed.game =
                    Game::parse(&value).ok_or_else(|| format!("unknown game: {value}"))?;
                // Only CartPole has a checked-in solve threshold. Other games
                // report learning metrics unless the caller supplies a gate.
                parsed.required_recent_return = (parsed.game == Game::CartPole).then_some(195.0);
            }
            "--lanes" => parsed.lanes = parse_value("--lanes", args.next())?,
            "--steps" => parsed.steps = parse_value("--steps", args.next())?,
            "--seed" => {
                parsed.seed = parse_value("--seed", args.next())?;
                parsed.seeds = None;
            }
            "--seeds" => parsed.seeds = Some(parse_seeds(args.next())?),
            "--log-every" => parsed.log_every = parse_value("--log-every", args.next())?,
            "--recent-window" => {
                parsed.recent_window = parse_value("--recent-window", args.next())?;
            }
            "--action-repeat" => {
                parsed.action_repeat = parse_value("--action-repeat", args.next())?;
            }
            "--action-repeat-after-success" => {
                parsed.action_repeat_after_success =
                    Some(parse_value("--action-repeat-after-success", args.next())?);
            }
            "--action-repeat-switch-successes" => {
                parsed.action_repeat_switch_successes =
                    parse_value("--action-repeat-switch-successes", args.next())?;
            }
            "--require-recent-return" => {
                parsed.required_recent_return =
                    Some(parse_value("--require-recent-return", args.next())?);
            }
            "--require-recent-success-rate" => {
                parsed.required_recent_success_rate =
                    Some(parse_value("--require-recent-success-rate", args.next())?);
            }
            "--no-gate" => {
                parsed.required_recent_return = None;
                parsed.required_recent_success_rate = None;
            }
            "--extrinsic-alpha" => {
                parsed.extrinsic_alpha = parse_value("--extrinsic-alpha", args.next())?;
            }
            "--reward-surprise" => {
                parsed.reward_surprise = parse_value("--reward-surprise", args.next())?;
            }
            "--reward-novelty" => {
                parsed.reward_novelty = parse_value("--reward-novelty", args.next())?;
            }
            "--reward-homeostatic" => {
                parsed.reward_homeostatic = parse_value("--reward-homeostatic", args.next())?;
            }
            "--reward-order" => {
                parsed.reward_order = parse_value("--reward-order", args.next())?;
            }
            "--entropy-beta" => {
                parsed.entropy_beta = parse_value("--entropy-beta", args.next())?;
            }
            "--no-advantage-normalize" => parsed.advantage_normalize = false,
            "--no-grpo" => parsed.use_grpo = false,
            "--per-step-grpo" => parsed.episode_grpo = false,
            "--no-sil" => parsed.use_sil = false,
            "--sil-loss-coef" => {
                parsed.sil_loss_coef = parse_value("--sil-loss-coef", args.next())?;
            }
            "--sil-buffer-capacity" => {
                parsed.sil_buffer_capacity = parse_value("--sil-buffer-capacity", args.next())?;
            }
            "--sil-event-filter" => parsed.sil_event_filter = true,
            "--action-masks" => parsed.use_action_masks = true,
            "--lr-drop-on-success-rate" => {
                parsed.lr_drop_on_success_rate =
                    Some(parse_value("--lr-drop-on-success-rate", args.next())?);
            }
            "--use-adam" => parsed.use_adam = true,
            "--random-policy" => parsed.random_policy = true,
            "--pendulum-expert" => parsed.pendulum_expert = true,
            "--pendulum-dagger" => parsed.pendulum_dagger = true,
            "--pendulum-success-return" => {
                parsed.pendulum_success_return =
                    parse_value("--pendulum-success-return", args.next())?;
            }
            "--deterministic-eval" => parsed.deterministic_eval = true,
            "--evaluation-steps" => {
                parsed.evaluation_steps = parse_value("--evaluation-steps", args.next())?;
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    if parsed.lanes == 0 {
        return Err("--lanes must be greater than zero".into());
    }
    if parsed.steps == 0 {
        return Err("--steps must be greater than zero".into());
    }
    if parsed.recent_window == 0 {
        return Err("--recent-window must be greater than zero".into());
    }
    if parsed.action_repeat == 0 {
        return Err("--action-repeat must be greater than zero".into());
    }
    if parsed.action_repeat_after_success == Some(0) {
        return Err("--action-repeat-after-success must be greater than zero".into());
    }
    if parsed.action_repeat_switch_successes == 0 {
        return Err("--action-repeat-switch-successes must be greater than zero".into());
    }
    if parsed
        .required_recent_success_rate
        .is_some_and(|rate| !(0.0..=1.0).contains(&rate))
    {
        return Err("--require-recent-success-rate must be in [0, 1]".into());
    }
    if parsed
        .lr_drop_on_success_rate
        .is_some_and(|rate| !(0.0..=1.0).contains(&rate))
    {
        return Err("--lr-drop-on-success-rate must be in [0, 1]".into());
    }
    if !parsed.sil_loss_coef.is_finite() || parsed.sil_loss_coef < 0.0 {
        return Err("--sil-loss-coef must be finite and non-negative".into());
    }
    if !parsed.entropy_beta.is_finite() || parsed.entropy_beta < 0.0 {
        return Err("--entropy-beta must be finite and non-negative".into());
    }
    if parsed.sil_buffer_capacity == 0 {
        return Err("--sil-buffer-capacity must be greater than zero".into());
    }
    if (parsed.pendulum_expert || parsed.pendulum_dagger) && parsed.game != Game::Pendulum {
        return Err("--pendulum-expert/--pendulum-dagger require --env pendulum".into());
    }
    if usize::from(parsed.pendulum_expert)
        + usize::from(parsed.pendulum_dagger)
        + usize::from(parsed.random_policy)
        > 1
    {
        return Err(
            "--pendulum-expert, --pendulum-dagger, and --random-policy are mutually exclusive"
                .into(),
        );
    }
    if parsed.pendulum_dagger && parsed.evaluation_steps == 0 {
        return Err("--pendulum-dagger requires --evaluation-steps".into());
    }
    if parsed.pendulum_expert && parsed.evaluation_steps > 0 {
        return Err("--pendulum-expert cannot be combined with --evaluation-steps".into());
    }
    if !parsed.pendulum_success_return.is_finite() {
        return Err("--pendulum-success-return must be finite".into());
    }
    if parsed.evaluation_steps >= parsed.steps && parsed.evaluation_steps > 0 {
        return Err("--evaluation-steps must be less than --steps".into());
    }
    Ok(Some(parsed))
}

fn recent_mean(returns: &[VecDeque<f32>]) -> Option<f32> {
    let (sum, count) = returns
        .iter()
        .flat_map(|lane| lane.iter())
        .fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    (count > 0).then_some(sum / count as f32)
}

fn recent_success_rate(successes: &[VecDeque<bool>]) -> Option<f32> {
    let (successful, count) = successes
        .iter()
        .flat_map(|lane| lane.iter())
        .fold((0usize, 0usize), |(successful, count), &success| {
            (successful + usize::from(success), count + 1)
        });
    (count > 0).then_some(successful as f32 / count as f32)
}

fn cartpole_return_lr_trigger(game: Game, extrinsic_alpha: f32, mean: Option<f32>) -> bool {
    extrinsic_alpha > 0.0 && game == Game::CartPole && mean.is_some_and(|value| value >= 80.0)
}

fn run(args: &Args) -> bool {
    println!(
        "kindle native learning: {} | lanes={} steps={} seed={} recent={} repeat={} | ext={} intrinsic=[{},{},{},{}] grpo={} sil={}:{} event_filter={} masks={} adam={} policy={} deterministic={}",
        args.game.name(),
        args.lanes,
        args.steps,
        args.seed,
        args.recent_window,
        args.action_repeat,
        args.extrinsic_alpha,
        args.reward_surprise,
        args.reward_novelty,
        args.reward_homeostatic,
        args.reward_order,
        if !args.use_grpo || args.pendulum_dagger {
            "off"
        } else if args.episode_grpo {
            "episode"
        } else {
            "step"
        },
        args.use_sil && args.game != Game::Pendulum,
        args.sil_loss_coef,
        args.sil_event_filter,
        args.use_action_masks,
        args.use_adam,
        if args.pendulum_expert {
            "expert"
        } else if args.pendulum_dagger {
            "dagger"
        } else if args.random_policy {
            "random"
        } else {
            "kindle"
        },
        args.deterministic_eval,
    );

    let adapters = (0..args.lanes)
        .map(|_| {
            if args.game == Game::Pendulum {
                Box::new(GenericAdapter::continuous(
                    0,
                    args.game.observation_dim(),
                    1,
                    0.5,
                )) as Box<dyn kindle::EnvAdapter>
            } else {
                Box::new(GenericAdapter::discrete(
                    0,
                    args.game.observation_dim(),
                    args.game.num_actions(),
                )) as Box<dyn kindle::EnvAdapter>
            }
        })
        .collect::<Vec<_>>();
    let config = AgentConfig {
        latent_dim: 16,
        hidden_dim: 32,
        batch_size: args.lanes,
        learning_rate: if args.pendulum_dagger { 0.0 } else { 3e-4 },
        lr_policy: 5e-4,
        reward_weights: RewardWeights {
            surprise: args.reward_surprise,
            novelty: args.reward_novelty,
            homeostatic: args.reward_homeostatic,
            order: args.reward_order,
        },
        extrinsic_reward_alpha: args.extrinsic_alpha,
        gamma: 0.99,
        advantage_clamp: 2.0,
        entropy_beta: args.entropy_beta,
        entropy_floor: 0.0,
        advantage_normalize: args.advantage_normalize && !args.pendulum_dagger,
        use_grpo: args.use_grpo && !args.pendulum_dagger,
        use_grpo_episode: args.episode_grpo && !args.pendulum_dagger,
        use_sil: args.use_sil && args.game != Game::Pendulum,
        sil_loss_coef: args.sil_loss_coef,
        sil_buffer_capacity: args.sil_buffer_capacity,
        sil_event_filter: args.sil_event_filter,
        value_loss_coef: 0.0,
        end_to_end_encoder: true,
        recon_loss_coef: if args.game == Game::Pendulum {
            0.0
        } else {
            1.0
        },
        n_step: 8,
        rollout_length: 8,
        action_repeat: args.action_repeat,
        use_adam: args.use_adam,
        ..AgentConfig::default()
    };

    println!("building agent...");
    let mut agent = Agent::new(config, adapters);
    agent.set_policy_lr_multiplier("policy.", 5.0);
    let mut environments = (0..args.lanes)
        .map(|_| args.game.new_environment())
        .collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut episode_returns = vec![0.0f32; args.lanes];
    let mut recent_returns = vec![VecDeque::with_capacity(args.recent_window); args.lanes];
    let mut recent_successes = vec![VecDeque::with_capacity(args.recent_window); args.lanes];
    let mut completed_episodes = 0usize;
    let mut successful_episodes = 0usize;
    let mut natural_terminals = 0usize;
    let mut time_limits = 0usize;
    let mut learning_rates_dropped = false;
    let mut action_repeat_switched = false;
    let start = Instant::now();

    for step in 0..args.steps {
        let evaluation_start = args.steps.saturating_sub(args.evaluation_steps);
        if args.evaluation_steps > 0 && step == evaluation_start {
            let policy = if args.random_policy {
                "random"
            } else if args.pendulum_expert {
                "expert"
            } else {
                "greedy"
            };
            println!(
                "[evaluation] freezing learning and resetting metrics for {} {policy} steps",
                args.evaluation_steps,
            );
            agent.set_learning_rate(0.0);
            agent.set_lr_policy(0.0);
            for (lane, environment) in environments.iter_mut().enumerate() {
                environment.reset();
                agent.mark_boundary(lane);
                episode_returns[lane] = 0.0;
                recent_returns[lane].clear();
                recent_successes[lane].clear();
            }
            completed_episodes = 0;
            successful_episodes = 0;
            natural_terminals = 0;
            time_limits = 0;
        }
        if args.use_action_masks {
            let env_refs = environments
                .iter()
                .map(|environment| &**environment as &dyn Environment)
                .collect::<Vec<_>>();
            agent.set_action_masks_from_envs(&env_refs);
        }
        let observations = environments
            .iter()
            .map(|environment| environment.observe())
            .collect::<Vec<_>>();
        let proposed_actions =
            if args.deterministic_eval || (args.evaluation_steps > 0 && step >= evaluation_start) {
                agent.act_greedy(&observations, &mut rng)
            } else {
                agent.act(&observations, &mut rng)
            };
        let evaluating = args.evaluation_steps > 0 && step >= evaluation_start;
        let expert_actions = (args.pendulum_dagger && !evaluating).then(|| {
            observations
                .iter()
                .map(pendulum_expert_action)
                .collect::<Vec<_>>()
        });
        let actions = if args.pendulum_expert {
            observations
                .iter()
                .map(pendulum_expert_action)
                .collect::<Vec<_>>()
        } else if args.random_policy {
            (0..args.lanes)
                .map(|_| {
                    if args.game == Game::Pendulum {
                        Action::Continuous(vec![rng.random_range(-1.0..=1.0)])
                    } else {
                        Action::Discrete(rng.random_range(0..args.game.num_actions()))
                    }
                })
                .collect::<Vec<_>>()
        } else {
            proposed_actions.clone()
        };
        let mut results = environments
            .iter_mut()
            .zip(&actions)
            .map(|(environment, action)| environment.step(action))
            .collect::<Vec<_>>();

        for (lane, result) in results.iter_mut().enumerate() {
            let episode_return = episode_returns[lane] + result.reward;
            if args.game == Game::Pendulum && result.done() {
                result.task_event = episode_return >= args.pendulum_success_return;
            }
            episode_returns[lane] = episode_return;
            if result.done() {
                if recent_returns[lane].len() == args.recent_window {
                    recent_returns[lane].pop_front();
                    recent_successes[lane].pop_front();
                }
                recent_returns[lane].push_back(episode_returns[lane]);
                recent_successes[lane].push_back(result.task_event);
                episode_returns[lane] = 0.0;
                completed_episodes += 1;
                successful_episodes += usize::from(result.task_event);
                natural_terminals += usize::from(result.terminated);
                time_limits += usize::from(result.truncated);
            }
            // During DAgger collection, a
            // constant positive advantage turns the continuous Gaussian loss
            // into behavior cloning of the externally executed action. The
            // raw environment reward above remains the reported metric.
            if args.pendulum_dagger && !evaluating {
                result.reward = 1.0;
                result.task_event = false;
            }
        }
        // DAgger executes the current policy to visit its own states, but
        // trains its e2e head on the expert's corrective action for each
        // acted-from observation.
        let training_actions = expert_actions.as_deref().unwrap_or(&actions);
        agent.observe_step_results(&results, training_actions, &mut rng);

        if !action_repeat_switched
            && successful_episodes >= args.action_repeat_switch_successes
            && let Some(repeat) = args.action_repeat_after_success
        {
            agent.set_action_repeat(repeat);
            action_repeat_switched = true;
            println!(
                "[action-repeat] {} task successes reached; repeat {} -> {repeat}",
                args.action_repeat_switch_successes, args.action_repeat,
            );
        }

        if args.log_every > 0 && (step + 1) % args.log_every == 0 {
            let mean = recent_mean(&recent_returns);
            let success_rate = recent_success_rate(&recent_successes);
            // The CartPole return is an external task metric. It may schedule
            // an extrinsic run, but must not influence an intrinsic-only one.
            let return_trigger = cartpole_return_lr_trigger(args.game, args.extrinsic_alpha, mean);
            let success_trigger = args
                .lr_drop_on_success_rate
                .is_some_and(|threshold| success_rate.is_some_and(|value| value >= threshold));
            if (return_trigger || success_trigger) && !learning_rates_dropped {
                agent.set_learning_rate(3e-5);
                agent.set_lr_policy(5e-5);
                learning_rates_dropped = true;
                println!(
                    "[lr-drop] solve trigger reached; learning rates divided by 10 (recent return={}, success={})",
                    mean.map_or_else(|| "n/a".to_owned(), |value| format!("{value:+.2}")),
                    success_rate.map_or_else(
                        || "n/a".to_owned(),
                        |value| format!("{:.1}%", value * 100.0)
                    ),
                );
            }
            let diagnostics = &agent.diagnostics()[0];
            let elapsed = start.elapsed().as_secs_f32().max(1e-3);
            println!(
                "step={:>6} episodes={:>5} recent={:>8} success={:>6} | wm={:.3} pi={:+.3} ent={:.2} sil={}/{} | {:.0} env-steps/s",
                step + 1,
                completed_episodes,
                mean.map_or_else(|| "n/a".to_owned(), |value| format!("{value:+.2}")),
                success_rate.map_or_else(
                    || "n/a".to_owned(),
                    |value| format!("{:.0}%", value * 100.0)
                ),
                diagnostics.loss_world_model,
                diagnostics.loss_policy,
                diagnostics.policy_entropy,
                agent.sil_buffer_size(),
                agent.sil_updates_fired_count(),
                ((step + 1) * args.lanes) as f32 / elapsed,
            );
        }
    }

    let recent = recent_mean(&recent_returns);
    let success_rate = recent_success_rate(&recent_successes);
    let elapsed = start.elapsed().as_secs_f32().max(1e-3);
    println!("\n--- {} summary ---", args.game.name());
    let evaluation_decisions = args.evaluation_steps * args.lanes;
    let total_decisions = args.steps * args.lanes;
    println!("total environment decisions: {total_decisions}");
    println!(
        "training/evaluation decisions: {}/{}",
        total_decisions - evaluation_decisions,
        evaluation_decisions,
    );
    println!(
        "episodes: {completed_episodes} ({successful_episodes} successes, {natural_terminals} natural terminals, {time_limits} time limits)",
    );
    println!(
        "recent mean return: {}",
        recent.map_or_else(|| "n/a".to_owned(), |value| format!("{value:+.2}"))
    );
    println!(
        "recent task-success rate: {}",
        success_rate.map_or_else(
            || "n/a".to_owned(),
            |value| format!("{:.1}%", value * 100.0)
        )
    );
    println!(
        "wall: {elapsed:.1}s, throughput: {:.0} env-steps/s",
        (args.steps * args.lanes) as f32 / elapsed
    );

    let mut gates_passed = true;
    if let Some(required) = args.required_recent_return {
        let passed = recent.is_some_and(|value| value >= required);
        gates_passed &= passed;
        println!(
            "return gate: {} (recent {} vs required {required:+.2})",
            if passed { "PASS" } else { "FAIL" },
            recent.map_or_else(|| "n/a".to_owned(), |value| format!("{value:+.2}")),
        );
    }
    if let Some(required) = args.required_recent_success_rate {
        let passed = success_rate.is_some_and(|value| value >= required);
        gates_passed &= passed;
        println!(
            "success gate: {} (recent {} vs required {:.1}%)",
            if passed { "PASS" } else { "FAIL" },
            success_rate.map_or_else(
                || "n/a".to_owned(),
                |value| format!("{:.1}%", value * 100.0)
            ),
            required * 100.0,
        );
    }
    gates_passed
}

fn main() -> ExitCode {
    env_logger::init();
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => return ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            usage();
            return ExitCode::from(2);
        }
    };
    let seeds = args.seeds.clone().unwrap_or_else(|| vec![args.seed]);
    let multi_seed = seeds.len() > 1;
    let gates_configured =
        args.required_recent_return.is_some() || args.required_recent_success_rate.is_some();
    let mut passed = true;
    for seed in &seeds {
        let mut run_args = args.clone();
        run_args.seed = *seed;
        run_args.seeds = None;
        passed &= run(&run_args);
    }
    if multi_seed {
        let seed_list = seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if gates_configured {
            println!(
                "\nmulti-seed gate: {} ({} seeds: {seed_list})",
                if passed { "PASS" } else { "FAIL" },
                seeds.len(),
            );
        } else {
            println!(
                "\nmulti-seed runs complete ({} seeds: {seed_list}; no gates configured)",
                seeds.len(),
            );
        }
    }
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_list_parsing_is_ordered_and_strict() {
        assert_eq!(
            parse_seeds(Some("42, 7,123".into())).unwrap(),
            vec![42, 7, 123]
        );
        assert!(parse_seeds(Some("42,42".into())).is_err());
        assert!(parse_seeds(Some("42,nope".into())).is_err());
        assert!(parse_seeds(Some(" , ".into())).is_err());
        assert!(parse_seeds(None).is_err());
    }

    #[test]
    fn cartpole_score_cannot_schedule_an_intrinsic_only_run() {
        assert!(!cartpole_return_lr_trigger(
            Game::CartPole,
            0.0,
            Some(500.0)
        ));
        assert!(cartpole_return_lr_trigger(Game::CartPole, 1.0, Some(80.0)));
        assert!(!cartpole_return_lr_trigger(Game::Taxi, 1.0, Some(500.0)));
    }
}
