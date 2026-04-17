//! Multi-env benchmark: runs all 7 kindle-gym environments with both
//! L0-only (num_options=1) and L1 (num_options=4, option_horizon=10)
//! at N=1, reporting convergence and L1 diagnostics side by side.
//!
//! Run: `cargo run --release --example bench_envs`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use rand::SeedableRng;

struct EnvSpec {
    name: &'static str,
    env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
}

fn make_envs() -> Vec<EnvSpec> {
    use kindle_gym::*;
    vec![
        EnvSpec {
            name: "GridWorld",
            env: Box::new(grid_world::GridWorld::new()),
            adapter: Box::new(GenericAdapter::discrete(
                0,
                grid_world::OBS_DIM,
                grid_world::NUM_ACTIONS,
            )),
        },
        EnvSpec {
            name: "CartPole",
            env: Box::new(cart_pole::CartPole::new()),
            adapter: Box::new(GenericAdapter::discrete(1, 4, 2)),
        },
        EnvSpec {
            name: "MountainCar",
            env: Box::new(mountain_car::MountainCar::new()),
            adapter: Box::new(GenericAdapter::discrete(2, 2, 3)),
        },
        EnvSpec {
            name: "Acrobot",
            env: Box::new(acrobot::Acrobot::new()),
            adapter: Box::new(GenericAdapter::discrete(3, 6, 3)),
        },
        EnvSpec {
            name: "Taxi",
            env: Box::new(taxi::Taxi::new()),
            adapter: Box::new(GenericAdapter::discrete(
                4,
                taxi::OBS_DIM,
                taxi::NUM_ACTIONS,
            )),
        },
        EnvSpec {
            name: "RandomWalk",
            env: Box::new(random_walk::RandomWalk::new(10)),
            adapter: Box::new(GenericAdapter::discrete(5, 10, 2)),
        },
        EnvSpec {
            name: "Pendulum",
            env: Box::new(pendulum::Pendulum::new()),
            adapter: Box::new(GenericAdapter::continuous(6, 3, 1, 0.5)),
        },
    ]
}

fn run_one(
    name: &str,
    mut env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
    l1: bool,
) {
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options: if l1 { 4 } else { 1 },
        option_horizon: 10,
        ..AgentConfig::default()
    };
    let mode = if l1 { "L1" } else { "L0" };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let steps = 1000;
    let mut early_wm = f32::NAN;
    let mut late_wm = f32::NAN;
    let mut early_r = f32::NAN;
    let mut late_r = f32::NAN;
    let mut late_ent = f32::NAN;
    let mut late_opt = 0u32;
    let mut late_gdist = 0.0f32;

    for step in 0..steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        let d = &agent.diagnostics()[0];
        if step == 100 {
            early_wm = d.loss_world_model;
            early_r = d.reward_mean;
        }
        if step == steps - 1 {
            late_wm = d.loss_world_model;
            late_r = d.reward_mean;
            late_ent = d.policy_entropy;
            late_opt = d.current_option;
            late_gdist = d.goal_distance;
        }

        assert!(
            d.loss_world_model.is_finite(),
            "{name} {mode}: NaN wm_loss at step {step}"
        );
    }

    let wm_ok = late_wm < early_wm * 10.0 || late_wm < 0.1;
    let wm_icon = if wm_ok { "✓" } else { "✗" };

    print!(
        "  {mode:>2} | wm: {:.4} → {:.4} {wm_icon} | r: {:+.2} → {:+.2} | ent={:.2}",
        early_wm, late_wm, early_r, late_r, late_ent
    );
    if l1 {
        print!(" | opt={} gdist={:.1}", late_opt, late_gdist);
    }
    println!();
}

fn main() {
    println!("kindle multi-env benchmark (N=1, 1000 steps each)");
    println!("===================================================\n");

    for spec in make_envs() {
        println!(
            "{} (obs={}, actions={}):",
            spec.name,
            spec.env.observation_dim(),
            spec.env.num_actions()
        );

        // L0 run
        let envs2 = make_envs();
        let spec2 = envs2.into_iter().find(|s| s.name == spec.name).unwrap();
        run_one(spec.name, spec.env, spec.adapter, false);

        // L1 run
        run_one(spec2.name, spec2.env, spec2.adapter, true);

        println!();
    }
}
