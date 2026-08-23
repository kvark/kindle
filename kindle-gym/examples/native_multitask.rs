//! Train and freeze one task-conditioned Kindle policy on CartPole and Taxi.
//!
//! This is an outcome gate for simultaneous multi-game competence, not merely
//! a graph-stability probe. Eight lanes of each task share one policy while
//! fixed task embeddings, same-task GRPO groups, and collection-time action
//! masks keep their semantics separate.
//!
//! ```text
//! cargo run --release -p kindle-gym --example native_multitask -- \
//!   --seeds 42,43,44
//!
//! cargo run --release -p kindle-gym --example native_multitask -- \
//!   --sequential --seeds 42,43,44
//! ```

use kindle::{Agent, AgentConfig, Environment, GenericAdapter, RewardWeights};
use kindle_gym::{cart_pole::CartPole, taxi::Taxi};
use rand::SeedableRng;
use std::process::ExitCode;
use std::time::Instant;

const LANES_PER_TASK: usize = 8;
const TASKS: usize = 2;
const LANES: usize = LANES_PER_TASK * TASKS;
const CART_ENV_ID: u32 = 0;
const TAXI_ENV_ID: u32 = 1;

#[derive(Clone, Debug)]
struct Args {
    train_steps: usize,
    eval_steps: usize,
    seed: u64,
    seeds: Option<Vec<u64>>,
    log_every: usize,
    stochastic_eval: bool,
    sequential: bool,
    taxi_first: bool,
    latent_dim: usize,
    hidden_dim: usize,
    require_cart_return: f32,
    require_taxi_success: f32,
    sil_loss_coef: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            train_steps: 100_000,
            eval_steps: 20_000,
            seed: 42,
            seeds: None,
            log_every: 20_000,
            stochastic_eval: true,
            sequential: false,
            taxi_first: false,
            latent_dim: 32,
            hidden_dim: 64,
            require_cart_return: 195.0,
            require_taxi_success: 0.9,
            sil_loss_coef: 2.0,
        }
    }
}

fn usage() {
    println!(
        "native_multitask [--train-steps N] [--eval-steps N] \
         [--seed N|--seeds N,N,...] \
         [--log-every N] [--stochastic-eval|--greedy-eval] \
         [--joint|--sequential] [--cart-first|--taxi-first] \
         [--latent-dim N] [--hidden-dim N] \
         [--require-cart-return R] \
         [--require-taxi-success R] [--sil-loss-coef R]"
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
            "--train-steps" => {
                parsed.train_steps = parse_value("--train-steps", args.next())?;
            }
            "--eval-steps" => {
                parsed.eval_steps = parse_value("--eval-steps", args.next())?;
            }
            "--seed" => {
                parsed.seed = parse_value("--seed", args.next())?;
                parsed.seeds = None;
            }
            "--seeds" => parsed.seeds = Some(parse_seeds(args.next())?),
            "--log-every" => parsed.log_every = parse_value("--log-every", args.next())?,
            "--stochastic-eval" => parsed.stochastic_eval = true,
            "--greedy-eval" => parsed.stochastic_eval = false,
            "--joint" => parsed.sequential = false,
            "--sequential" => parsed.sequential = true,
            "--cart-first" => {
                parsed.sequential = true;
                parsed.taxi_first = false;
            }
            "--taxi-first" => {
                parsed.sequential = true;
                parsed.taxi_first = true;
            }
            "--latent-dim" => parsed.latent_dim = parse_value("--latent-dim", args.next())?,
            "--hidden-dim" => parsed.hidden_dim = parse_value("--hidden-dim", args.next())?,
            "--require-cart-return" => {
                parsed.require_cart_return = parse_value("--require-cart-return", args.next())?;
            }
            "--require-taxi-success" => {
                parsed.require_taxi_success = parse_value("--require-taxi-success", args.next())?;
            }
            "--sil-loss-coef" => {
                parsed.sil_loss_coef = parse_value("--sil-loss-coef", args.next())?;
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    if parsed.train_steps == 0 || parsed.eval_steps == 0 {
        return Err("--train-steps and --eval-steps must be positive".into());
    }
    if parsed.latent_dim == 0 || parsed.hidden_dim == 0 {
        return Err("--latent-dim and --hidden-dim must be positive".into());
    }
    if !parsed.require_cart_return.is_finite() {
        return Err("--require-cart-return must be finite".into());
    }
    if !parsed.require_taxi_success.is_finite()
        || !(0.0..=1.0).contains(&parsed.require_taxi_success)
    {
        return Err("--require-taxi-success must be in [0, 1]".into());
    }
    if !parsed.sil_loss_coef.is_finite() || parsed.sil_loss_coef < 0.0 {
        return Err("--sil-loss-coef must be finite and non-negative".into());
    }
    Ok(Some(parsed))
}

#[derive(Default)]
struct EvalStats {
    current_returns: Vec<f32>,
    cart_returns: Vec<f32>,
    taxi_returns: Vec<f32>,
    taxi_successes: usize,
}

impl EvalStats {
    fn new() -> Self {
        Self {
            current_returns: vec![0.0; LANES],
            ..Self::default()
        }
    }

    fn observe(&mut self, lane: usize, result: &kindle::StepResult) {
        self.current_returns[lane] += result.reward;
        if !result.done() {
            return;
        }
        let episode_return = self.current_returns[lane];
        self.current_returns[lane] = 0.0;
        if lane < LANES_PER_TASK {
            self.cart_returns.push(episode_return);
        } else {
            self.taxi_returns.push(episode_return);
            self.taxi_successes += usize::from(result.task_event);
        }
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn agent_config(args: &Args, batch_size: usize) -> AgentConfig {
    AgentConfig {
        latent_dim: args.latent_dim,
        hidden_dim: args.hidden_dim,
        batch_size,
        learning_rate: 3e-4,
        lr_policy: 5e-4,
        reward_weights: RewardWeights {
            surprise: 0.0,
            novelty: 0.0,
            homeostatic: 0.5,
            order: 0.0,
        },
        extrinsic_reward_alpha: 1.0,
        gamma: 0.99,
        advantage_clamp: 2.0,
        entropy_beta: 0.01,
        entropy_floor: 0.0,
        advantage_normalize: true,
        use_grpo: true,
        use_grpo_episode: false,
        use_sil: true,
        sil_loss_coef: args.sil_loss_coef,
        sil_buffer_capacity: 20_000,
        sil_event_filter: true,
        value_loss_coef: 0.0,
        end_to_end_encoder: true,
        orthogonal_task_codes: true,
        recon_loss_coef: 1.0,
        n_step: 8,
        rollout_length: 8,
        use_adam: false,
        ..AgentConfig::default()
    }
}

fn cart_adapter() -> Box<dyn kindle::EnvAdapter> {
    Box::new(GenericAdapter::discrete(CART_ENV_ID, 4, 2))
}

fn taxi_adapter() -> Box<dyn kindle::EnvAdapter> {
    Box::new(GenericAdapter::discrete(
        TAXI_ENV_ID,
        kindle_gym::taxi::OBS_DIM,
        kindle_gym::taxi::NUM_ACTIONS,
    ))
}

fn run_joint(args: &Args) -> bool {
    let adapters = (0..LANES_PER_TASK)
        .map(|_| cart_adapter())
        .chain((0..LANES_PER_TASK).map(|_| taxi_adapter()))
        .collect::<Vec<_>>();
    let config = agent_config(args, LANES);
    println!(
        "building one CartPole+Taxi agent: lanes={LANES} train={} eval={} seed={} model={}/{}",
        args.train_steps, args.eval_steps, args.seed, args.latent_dim, args.hidden_dim,
    );
    let mut agent = Agent::new(config, adapters);
    agent.set_policy_lr_multiplier("policy.", 5.0);
    let mut environments = (0..LANES_PER_TASK)
        .map(|_| Box::new(CartPole::new()) as Box<dyn Environment>)
        .chain((0..LANES_PER_TASK).map(|_| Box::new(Taxi::new()) as Box<dyn Environment>))
        .collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut training_events = [0usize; TASKS];
    let mut evaluation = EvalStats::new();
    let total_steps = args.train_steps + args.eval_steps;
    let start = Instant::now();

    for step in 0..total_steps {
        let evaluating = step >= args.train_steps;
        if step == args.train_steps {
            println!("[evaluation] freezing learning and resetting all task metrics");
            agent.set_learning_rate(0.0);
            agent.set_lr_policy(0.0);
            for (lane, environment) in environments.iter_mut().enumerate() {
                environment.reset();
                agent.mark_boundary(lane);
            }
        }
        let env_refs = environments
            .iter()
            .map(|environment| &**environment as &dyn Environment)
            .collect::<Vec<_>>();
        agent.set_action_masks_from_envs(&env_refs);
        let observations = environments
            .iter()
            .map(|environment| environment.observe())
            .collect::<Vec<_>>();
        let actions = if evaluating && !args.stochastic_eval {
            agent.act_greedy(&observations, &mut rng)
        } else {
            agent.act(&observations, &mut rng)
        };
        let results = environments
            .iter_mut()
            .zip(&actions)
            .map(|(environment, action)| environment.step(action))
            .collect::<Vec<_>>();
        for (lane, result) in results.iter().enumerate() {
            if evaluating {
                evaluation.observe(lane, result);
            } else if result.task_event {
                training_events[usize::from(lane >= LANES_PER_TASK)] += 1;
            }
        }
        agent.observe_step_results(&results, &actions, &mut rng);

        if args.log_every > 0 && (step + 1) % args.log_every == 0 {
            println!(
                "step={:>6} phase={} events=cart:{}/taxi:{} sil={}/{} | {:.0} decisions/s",
                step + 1,
                if evaluating { "eval" } else { "train" },
                training_events[0],
                training_events[1],
                agent.sil_buffer_size(),
                agent.sil_updates_fired_count(),
                ((step + 1) * LANES) as f32 / start.elapsed().as_secs_f32().max(1e-3),
            );
        }
    }

    let cart_return = mean(&evaluation.cart_returns);
    let taxi_return = mean(&evaluation.taxi_returns);
    let taxi_success =
        evaluation.taxi_successes as f32 / evaluation.taxi_returns.len().max(1) as f32;
    println!(
        "\n--- frozen {} joint-policy evaluation ---",
        if args.stochastic_eval {
            "stochastic"
        } else {
            "greedy"
        }
    );
    println!(
        "CartPole: episodes={} mean_return={cart_return:+.2}",
        evaluation.cart_returns.len()
    );
    println!(
        "Taxi: episodes={} mean_return={taxi_return:+.2} completion={:.1}%",
        evaluation.taxi_returns.len(),
        taxi_success * 100.0,
    );
    let cart_passed =
        evaluation.cart_returns.len() >= LANES_PER_TASK && cart_return >= args.require_cart_return;
    let taxi_passed = evaluation.taxi_returns.len() >= LANES_PER_TASK
        && taxi_success >= args.require_taxi_success;
    println!(
        "CartPole gate: {} ({cart_return:+.2} >= {:+.2})",
        if cart_passed { "PASS" } else { "FAIL" },
        args.require_cart_return,
    );
    println!(
        "Taxi gate: {} ({:.1}% >= {:.1}%)",
        if taxi_passed { "PASS" } else { "FAIL" },
        taxi_success * 100.0,
        args.require_taxi_success * 100.0,
    );
    if cart_passed && taxi_passed {
        println!("native multitask gate: PASS");
        true
    } else {
        eprintln!("native multitask gate: FAIL");
        false
    }
}

#[derive(Default)]
struct PhaseStats {
    current_returns: Vec<f32>,
    returns: Vec<f32>,
    successes: usize,
}

struct PhaseConfig<'a> {
    steps: usize,
    learning: bool,
    stochastic_eval: bool,
    log_every: usize,
    label: &'a str,
}

fn run_phase(
    agent: &mut Agent,
    environments: &mut [Box<dyn Environment>],
    rng: &mut rand::rngs::StdRng,
    config: PhaseConfig<'_>,
) -> PhaseStats {
    for (lane, environment) in environments.iter_mut().enumerate() {
        environment.reset();
        agent.mark_boundary(lane);
    }
    let mut stats = PhaseStats {
        current_returns: vec![0.0; environments.len()],
        ..PhaseStats::default()
    };
    let start = Instant::now();
    for step in 0..config.steps {
        let env_refs = environments
            .iter()
            .map(|environment| &**environment as &dyn Environment)
            .collect::<Vec<_>>();
        agent.set_action_masks_from_envs(&env_refs);
        let observations = environments
            .iter()
            .map(|environment| environment.observe())
            .collect::<Vec<_>>();
        let actions = if config.learning || config.stochastic_eval {
            agent.act(&observations, rng)
        } else {
            agent.act_greedy(&observations, rng)
        };
        let results = environments
            .iter_mut()
            .zip(&actions)
            .map(|(environment, action)| environment.step(action))
            .collect::<Vec<_>>();
        for (lane, result) in results.iter().enumerate() {
            stats.current_returns[lane] += result.reward;
            if result.done() {
                stats.returns.push(stats.current_returns[lane]);
                stats.current_returns[lane] = 0.0;
                stats.successes += usize::from(result.task_event);
            }
        }
        agent.observe_step_results(&results, &actions, rng);

        if config.log_every > 0 && (step + 1) % config.log_every == 0 {
            println!(
                "{}: step={:>6}/{} episodes={} events={} sil={}/{} | {:.0} decisions/s",
                config.label,
                step + 1,
                config.steps,
                stats.returns.len(),
                stats.successes,
                agent.sil_buffer_size(),
                agent.sil_updates_fired_count(),
                ((step + 1) * environments.len()) as f32 / start.elapsed().as_secs_f32().max(1e-3),
            );
        }
    }
    stats
}

#[derive(Clone, Copy)]
enum NativeTask {
    CartPole,
    Taxi,
}

impl NativeTask {
    fn name(self) -> &'static str {
        match self {
            Self::CartPole => "CartPole",
            Self::Taxi => "Taxi",
        }
    }

    fn adapter(self) -> Box<dyn kindle::EnvAdapter> {
        match self {
            Self::CartPole => cart_adapter(),
            Self::Taxi => taxi_adapter(),
        }
    }

    fn environments(self) -> Vec<Box<dyn Environment>> {
        (0..LANES_PER_TASK)
            .map(|_| match self {
                Self::CartPole => Box::new(CartPole::new()) as Box<dyn Environment>,
                Self::Taxi => Box::new(Taxi::new()) as Box<dyn Environment>,
            })
            .collect()
    }

    fn passed(self, stats: &PhaseStats, args: &Args) -> bool {
        if stats.returns.len() < LANES_PER_TASK {
            return false;
        }
        match self {
            Self::CartPole => mean(&stats.returns) >= args.require_cart_return,
            Self::Taxi => {
                stats.successes as f32 / stats.returns.len() as f32 >= args.require_taxi_success
            }
        }
    }
}

fn activate_task(agent: &mut Agent, task: NativeTask) -> Vec<Box<dyn Environment>> {
    for lane in 0..LANES_PER_TASK {
        agent.switch_lane(lane, task.adapter());
    }
    task.environments()
}

fn print_phase_result(task: NativeTask, label: &str, stats: &PhaseStats) {
    let task_return = mean(&stats.returns);
    let success = stats.successes as f32 / stats.returns.len().max(1) as f32;
    match task {
        NativeTask::CartPole => println!(
            "{label}: episodes={} mean_return={task_return:+.2}",
            stats.returns.len()
        ),
        NativeTask::Taxi => println!(
            "{label}: episodes={} mean_return={task_return:+.2} completion={:.1}%",
            stats.returns.len(),
            success * 100.0,
        ),
    }
}

fn run_sequential(args: &Args) -> bool {
    let (first, second) = if args.taxi_first {
        (NativeTask::Taxi, NativeTask::CartPole)
    } else {
        (NativeTask::CartPole, NativeTask::Taxi)
    };
    let config = agent_config(args, LANES_PER_TASK);
    println!(
        "building one sequential {}→{} agent: lanes={LANES_PER_TASK} train/task={} eval/phase={} seed={} model={}/{}",
        first.name(),
        second.name(),
        args.train_steps,
        args.eval_steps,
        args.seed,
        args.latent_dim,
        args.hidden_dim,
    );
    let mut agent = Agent::new(
        config,
        (0..LANES_PER_TASK)
            .map(|_| first.adapter())
            .collect::<Vec<_>>(),
    );
    agent.set_policy_lr_multiplier("policy.", 5.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut environments = first.environments();

    let first_training_label = format!("{} training", first.name());
    let _first_training = run_phase(
        &mut agent,
        &mut environments,
        &mut rng,
        PhaseConfig {
            steps: args.train_steps,
            learning: true,
            stochastic_eval: args.stochastic_eval,
            log_every: args.log_every,
            label: &first_training_label,
        },
    );
    agent.set_learning_rate(0.0);
    agent.set_lr_policy(0.0);
    let first_before_label = format!("{} before switch", first.name());
    let first_before = run_phase(
        &mut agent,
        &mut environments,
        &mut rng,
        PhaseConfig {
            steps: args.eval_steps,
            learning: false,
            stochastic_eval: args.stochastic_eval,
            log_every: args.log_every,
            label: &first_before_label,
        },
    );

    environments = activate_task(&mut agent, second);
    agent.set_learning_rate(3e-4);
    agent.set_lr_policy(5e-4);
    let second_training_label = format!("{} training", second.name());
    let _second_training = run_phase(
        &mut agent,
        &mut environments,
        &mut rng,
        PhaseConfig {
            steps: args.train_steps,
            learning: true,
            stochastic_eval: args.stochastic_eval,
            log_every: args.log_every,
            label: &second_training_label,
        },
    );
    agent.set_learning_rate(0.0);
    agent.set_lr_policy(0.0);
    let second_after_label = format!("{} after switch", second.name());
    let second_after = run_phase(
        &mut agent,
        &mut environments,
        &mut rng,
        PhaseConfig {
            steps: args.eval_steps,
            learning: false,
            stochastic_eval: args.stochastic_eval,
            log_every: args.log_every,
            label: &second_after_label,
        },
    );

    environments = activate_task(&mut agent, first);
    let first_retained_label = format!("{} retained", first.name());
    let first_retained = run_phase(
        &mut agent,
        &mut environments,
        &mut rng,
        PhaseConfig {
            steps: args.eval_steps,
            learning: false,
            stochastic_eval: args.stochastic_eval,
            log_every: args.log_every,
            label: &first_retained_label,
        },
    );

    println!(
        "\n--- frozen {} sequential-policy evaluation ---",
        if args.stochastic_eval {
            "stochastic"
        } else {
            "greedy"
        }
    );
    print_phase_result(first, &first_before_label, &first_before);
    print_phase_result(second, &second_after_label, &second_after);
    print_phase_result(first, &first_retained_label, &first_retained);
    let first_before_passed = first.passed(&first_before, args);
    let second_passed = second.passed(&second_after, args);
    let first_retained_passed = first.passed(&first_retained, args);
    println!(
        "sequential gates: {}-before={} {}={} {}-retained={}",
        first.name(),
        if first_before_passed { "PASS" } else { "FAIL" },
        second.name(),
        if second_passed { "PASS" } else { "FAIL" },
        first.name(),
        if first_retained_passed {
            "PASS"
        } else {
            "FAIL"
        },
    );
    first_before_passed && second_passed && first_retained_passed
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
    let mut passed = true;
    for seed in &seeds {
        let mut run_args = args.clone();
        run_args.seed = *seed;
        run_args.seeds = None;
        passed &= if run_args.sequential {
            run_sequential(&run_args)
        } else {
            run_joint(&run_args)
        };
    }
    if seeds.len() > 1 {
        println!(
            "\nnative multitask {} multi-seed gate: {} ({} seeds: {})",
            if args.sequential {
                "sequential"
            } else {
                "joint"
            },
            if passed { "PASS" } else { "FAIL" },
            seeds.len(),
            seeds
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
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
    fn evaluation_stats_keep_tasks_separate() {
        let mut stats = EvalStats::new();
        let mut cart = CartPole::new();
        let mut taxi = Taxi::new();
        let cart_result = cart.step(&kindle::Action::Discrete(0));
        let taxi_result = taxi.step(&kindle::Action::Discrete(0));
        stats.observe(0, &cart_result);
        stats.observe(LANES_PER_TASK, &taxi_result);
        assert!(stats.cart_returns.is_empty());
        assert!(stats.taxi_returns.is_empty());
        assert_eq!(stats.current_returns[0], 1.0);
        assert_eq!(stats.current_returns[LANES_PER_TASK], -1.0);
    }

    #[test]
    fn seed_list_is_strict() {
        assert_eq!(
            parse_seeds(Some("42, 43,44".into())).unwrap(),
            vec![42, 43, 44]
        );
        assert!(parse_seeds(Some("42,42".into())).is_err());
        assert!(parse_seeds(Some("nope".into())).is_err());
        assert!(parse_seeds(Some(" , ".into())).is_err());
    }

    #[test]
    fn defaults_select_the_validated_joint_gate() {
        let args = Args::default();
        assert!(args.stochastic_eval);
        assert!(!args.sequential);
        assert!(!args.taxi_first);
        assert_eq!((args.latent_dim, args.hidden_dim), (32, 64));
        assert_eq!(
            (args.require_cart_return, args.require_taxi_success),
            (195.0, 0.9)
        );
        assert_eq!(args.sil_loss_coef, 2.0);
    }
}
