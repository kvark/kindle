//! Reward signal verification.
//!
//! Verify that the reward circuit produces non-zero, varying signals
//! across different states and transitions.

use kindle::adapter::OBS_TOKEN_DIM;
use kindle::buffer::{ExperienceBuffer, Transition};
use kindle::env::HomeostaticVariable;
use kindle::reward::{RewardCircuit, RewardWeights};

/// Reward signals should be non-zero and vary across different states.
#[test]
fn reward_varies_across_states() {
    let rc = RewardCircuit::new(RewardWeights::default());

    // Different surprise levels
    let r1 = rc.compute(0.1, 1.0, 0.0, 0.0);
    let r2 = rc.compute(0.9, 1.0, 0.0, 0.0);
    assert!(
        (r1 - r2).abs() > 0.1,
        "reward should vary with surprise: r1={r1}, r2={r2}"
    );

    // Different novelty levels
    let r3 = rc.compute(0.5, 1.0, 0.0, 0.0);
    let r4 = rc.compute(0.5, 0.1, 0.0, 0.0);
    assert!(
        (r3 - r4).abs() > 0.1,
        "reward should vary with novelty: r3={r3}, r4={r4}"
    );

    // Different homeostatic levels
    let r5 = rc.compute(0.5, 0.5, 0.0, 0.0);
    let r6 = rc.compute(0.5, 0.5, -1.0, 0.0);
    assert!(
        (r5 - r6).abs() > 0.1,
        "reward should vary with homeostatic: r5={r5}, r6={r6}"
    );
}

/// Novelty should decay as the same state is revisited.
#[test]
fn novelty_decays_with_visits() {
    let mut buf = ExperienceBuffer::new(100, 0.1);
    let latent = vec![0.5, 0.5];

    let n0 = RewardCircuit::novelty(buf.visit_count(&latent));
    assert_eq!(n0, 1.0, "first visit should have max novelty");

    // Visit the same state repeatedly
    for _ in 0..10 {
        buf.push(Transition {
            latent: latent.clone(),
            ..Default::default()
        });
    }

    let n10 = RewardCircuit::novelty(buf.visit_count(&latent));
    assert!(n10 < n0, "novelty should decay: n0={n0}, n10={n10}");
    // 1/sqrt(10) ≈ 0.316
    assert!(
        (n10 - 1.0 / (10.0f32).sqrt()).abs() < 1e-5,
        "novelty should be 1/sqrt(N): n10={n10}"
    );
}

/// Homeostatic reward should be zero when in range, negative when out.
#[test]
fn homeostatic_penalty_gradient() {
    let in_range = vec![HomeostaticVariable {
        value: 0.5,
        target: 0.5,
        tolerance: 0.1,
    }];
    assert_eq!(RewardCircuit::homeostatic(&in_range), 0.0);

    // Gradually move out of range
    let mut penalties = Vec::new();
    for offset in [0.0, 0.2, 0.5, 1.0] {
        let vars = vec![HomeostaticVariable {
            value: 0.5 + offset,
            target: 0.5,
            tolerance: 0.1,
        }];
        penalties.push(RewardCircuit::homeostatic(&vars));
    }

    // Should be monotonically decreasing
    for i in 1..penalties.len() {
        assert!(
            penalties[i] <= penalties[i - 1],
            "homeostatic penalty should increase with deviation: {:?}",
            penalties
        );
    }
}

/// Combined reward should be non-zero for a typical transition.
#[test]
fn combined_reward_nonzero() {
    let rc = RewardCircuit::new(RewardWeights::default());
    let r = rc.compute(0.5, 0.8, -0.1, 0.0);
    assert!(r.abs() > 0.01, "combined reward should be non-zero: {r}");
    assert!(r.is_finite(), "reward must be finite");
}

/// Order reward should be positive when the agent transitions from a
/// diverse historical obs distribution into a concentrated recent one.
#[test]
fn order_rewards_agent_concentration() {
    let mut rc = RewardCircuit::new(RewardWeights::default());

    // Build a bank of 16 distinct obs patterns that collectively hit many
    // digest buckets.
    let patterns: Vec<Vec<f32>> = (0..16)
        .map(|k| {
            (0..OBS_TOKEN_DIM)
                .map(|j| ((k as f32 + j as f32 * 0.37).sin()) * 2.0)
                .collect()
        })
        .collect();

    // Fill the reference window past warmup with the diverse mix.
    for step in 0..256 {
        rc.observe_order(&patterns[step % patterns.len()]);
    }

    // Now concentrate onto a single pattern for long enough to flush the
    // recent window.
    let mut order = 0.0;
    for _ in 0..256 {
        order = rc.observe_order(&patterns[0]);
    }

    assert!(
        order > 0.0,
        "order reward should be positive after concentration: {order}"
    );
    let (h_ref, h_recent) = rc.last_order_entropies();
    assert!(
        h_ref > h_recent,
        "H_reference ({h_ref}) must exceed H_recent ({h_recent}) for positive order"
    );
}

/// Stationary behaviour should produce order ≈ 0 (no entropy differential).
#[test]
fn order_is_zero_for_stationary_behaviour() {
    let mut rc = RewardCircuit::new(RewardWeights::default());
    let patterns: Vec<Vec<f32>> = (0..8)
        .map(|k| {
            (0..OBS_TOKEN_DIM)
                .map(|j| ((k as f32 * 1.1 + j as f32 * 0.19).cos()) * 1.5)
                .collect()
        })
        .collect();

    // Cycle patterns for long enough that reference and recent converge
    // to the same distribution.
    let mut order = 0.0;
    for step in 0..1024 {
        order = rc.observe_order(&patterns[step % patterns.len()]);
    }

    assert!(
        order.abs() < 0.2,
        "stationary behaviour should yield near-zero order, got {order}"
    );
}
