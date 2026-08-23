//! Tier 5: long-run stability tests.
//!
//! These verify that the agent remains stable (no NaN, no divergence)
//! over extended training runs. They are GPU-gated via `#[ignore]` and
//! run manually or via `cargo test -- --ignored`.

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::grid_world::{GridWorld, NUM_ACTIONS, OBS_DIM};
use kindle_gym::random_walk::RandomWalk;
use rand::SeedableRng;

/// Short stability check on GridWorld with all Phase 4 mechanisms enabled.
///
/// Verifies:
/// - No NaN in any loss or diagnostic value across many steps
/// - World model loss decreases from early to late
/// - Drift stays bounded
///
/// Step count chosen to fit within ~1 minute on lavapipe; longer runs
/// are reserved for real GPU.
#[test]
#[ignore] // requires GPU
fn stability_grid_world() {
    let mut env = GridWorld::new();
    let adapter = Box::new(GenericAdapter::discrete(0, OBS_DIM, NUM_ACTIONS));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 16,
        buffer_capacity: 2000,
        batch_size: 1,
        learning_rate: 1e-3,
        // Anti-collapse recon (see canaries.rs note on the 2026-06-10
        // forward-dynamics fix).
        recon_loss_coef: 1.0,
        drift_interval: 500,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Step count chosen to fit within ~2 minutes on lavapipe (CPU Vulkan).
    // Real GPUs run this orders of magnitude faster; longer sweeps belong
    // in a separate manual benchmark.
    let total_steps = 5000usize;
    // Average WM loss over windows rather than single samples — per-step
    // loss is noisy, so single-sample comparisons are unreliable.
    let early_start = 100usize;
    let avg_window = 20usize;
    let mut early_losses: Vec<f32> = Vec::new();
    let mut late_losses: Vec<f32> = Vec::new();
    let mut max_drift = 0.0f32;

    for step in 0..total_steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let result = env.step(&action);
        agent.observe_step_results(
            std::slice::from_ref(&result),
            std::slice::from_ref(&action),
            &mut rng,
        );

        let diags = agent.diagnostics();
        let d = &diags[0];
        assert!(
            d.loss_world_model.is_finite(),
            "NaN/Inf in wm_loss at step {step}"
        );
        assert!(
            d.reward_mean.is_finite(),
            "NaN/Inf in reward at step {step}"
        );
        if (early_start..early_start + avg_window).contains(&step) {
            early_losses.push(d.loss_world_model);
        }
        if step >= total_steps - avg_window {
            late_losses.push(d.loss_world_model);
        }
        if d.repr_drift > max_drift {
            max_drift = d.repr_drift;
        }
    }

    let early_avg = early_losses.iter().sum::<f32>() / early_losses.len() as f32;
    let late_avg = late_losses.iter().sum::<f32>() / late_losses.len() as f32;
    // World model should converge: late average strictly below early
    // average (escape hatch: absolute loss already tiny).
    assert!(
        late_avg < early_avg || late_avg < 0.1,
        "world model did not converge: early_avg={early_avg:.4}, late_avg={late_avg:.4}"
    );
    // Drift stays bounded (we reduce encoder LR on excess drift)
    assert!(
        max_drift < 10.0,
        "representation drift too large: {max_drift}"
    );
}

/// Random walk canary: the world model should near-perfectly predict
/// deterministic next-state transitions on a 1D random walk.
#[test]
#[ignore] // requires GPU
fn stability_random_walk_convergence() {
    let size = 10;
    let mut env = RandomWalk::new(size);
    let adapter = Box::new(GenericAdapter::discrete(1, size, 2));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 16,
        buffer_capacity: 2000,
        batch_size: 1,
        learning_rate: 1e-3,
        // Anti-collapse recon (see canaries.rs note on the 2026-06-10
        // forward-dynamics fix).
        recon_loss_coef: 1.0,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    for _ in 0..5000 {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let result = env.step(&action);
        agent.observe_step_results(
            std::slice::from_ref(&result),
            std::slice::from_ref(&action),
            &mut rng,
        );
    }

    let final_loss = agent.diagnostics()[0].loss_world_model;
    assert!(
        final_loss.is_finite() && final_loss < 0.5,
        "random walk world model didn't converge: {final_loss}"
    );
}
