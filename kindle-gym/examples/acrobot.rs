//! Run the kindle agent on Acrobot.
//!
//! Run: `cargo run --example acrobot`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::acrobot::Acrobot;
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("kindle Acrobot");
    println!("=============");

    let mut env = Acrobot::new();
    let adapter = Box::new(GenericAdapter::discrete(3, 6, 3));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-4,
        lr_policy: 5e-5,
        extrinsic_reward_alpha: 1.0,
        warmup_steps: 200,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs)...");
    let mut agent = Agent::new(config, vec![adapter]);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 5000;

    for step in 0..num_steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let result = env.step(&action);
        agent.observe_step_results(
            std::slice::from_ref(&result),
            std::slice::from_ref(&action),
            &mut rng,
        );

        if (step + 1) % 1000 == 0 {
            let diags = agent.diagnostics();
            let d = &diags[0];
            println!(
                "step {:>5} | wm={:.4} pi={:.4} | r={:.3} ent={:.2} | buf={}",
                d.step,
                d.loss_world_model,
                d.loss_policy,
                d.reward_mean,
                d.policy_entropy,
                d.buffer_len,
            );
        }
    }

    let d = &agent.diagnostics()[0];
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(d).unwrap());
}
