//! Frozen Reward Circuit.
//!
//! Computes `r_t` as a weighted sum of four primitive signals:
//! - **Surprise**: world model prediction error
//! - **Novelty**: inverse-sqrt visit count
//! - **Homeostatic balance**: deviation from target ranges
//! - **Order**: causal entropy reduction of digested observations
//!
//! This circuit is intentionally frozen — it receives no gradient updates.
//! All computation happens on CPU.
//!
//! ## Order reward
//!
//! The old implementation (`-H(|o_i| / Σ|o_j|)`) was a component-wise entropy
//! of the raw observation vector. It was ungrounded (an encoding artefact —
//! any one-hot observation was trivially maxed), non-causal (depended on the
//! env's observation shape rather than the agent's actions), and therefore
//! gameable.
//!
//! The new formulation measures **the agent's contribution to reducing
//! observation-space entropy, relative to its own historical baseline**:
//!
//! 1. At construction, sample a frozen random digest `φ: R^{obs_token_dim}
//!    → R^{d}` (small random linear map) and fixed per-dim bucket edges.
//!    φ never trains.
//! 2. Each step, the agent pushes `bucket_id(φ(obs_token_t))` onto two
//!    ring buffers: `recent` (short window, length W) and `reference`
//!    (long window, length W_ref ≫ W).
//! 3. `H_recent` = bucket-histogram entropy over `recent`;
//!    `H_reference` = entropy over `reference`.
//! 4. `r_order = H_reference − H_recent`.
//!
//! Interpretation:
//! - Positive when the agent has concentrated its recent trajectory into a
//!   smaller subset of the digested phase space than its historical
//!   average — it is *building structure*.
//! - Negative when the agent is currently dispersing mass more than its
//!   long-run average — it is *breaking symmetry*.
//! - Zero once the reference window matches the recent window (stationary
//!   behaviour).
//!
//! Properties:
//! - **Grounded**: φ and the bucket grid are fixed at init and never train,
//!   preserving the frozen-reward-circuit invariant.
//! - **Causal**: the signal is always agent-relative — the reference window
//!   is the agent's own past behaviour, not an external datum.
//! - **Environment-agnostic**: operates on `OBS_TOKEN_DIM` tokens produced
//!   by the adapter, not on env-specific features.
//! - **Ungameable by latent collapse**: the digest acts on the observation
//!   token, not the encoder latent, so the encoder cannot cheat the signal.

use crate::adapter::OBS_TOKEN_DIM;
use crate::env::HomeostaticVariable;
use std::collections::VecDeque;

/// Digest output dimensionality. `BUCKETS_PER_DIM ^ DIGEST_DIM` is the total
/// bucket count; we want this well below `RECENT_WINDOW` so buckets get
/// populated enough to distinguish entropies.
const DIGEST_DIM: usize = 4;
/// Number of bins per digest dim. 4 bins × 4 dims = 256 total buckets.
const BUCKETS_PER_DIM: u32 = 4;
/// Recent-window length (short-horizon observation distribution).
const RECENT_WINDOW: usize = 64;
/// Reference-window length (long-horizon baseline). Must exceed
/// `RECENT_WINDOW` for the differential to carry signal.
const REFERENCE_WINDOW: usize = 512;
/// Minimum reference-window fill before a non-zero order signal is emitted.
/// Below this we return 0 — the baseline hasn't stabilized yet.
const REFERENCE_WARMUP: usize = 128;

/// Weights for the four reward components.
#[derive(Clone, Debug)]
pub struct RewardWeights {
    pub surprise: f32,
    pub novelty: f32,
    pub homeostatic: f32,
    /// Weight for the order primitive. Defaults to `0.5` — on by default
    /// now that the signal is grounded (agent-relative entropy reduction
    /// rather than the old observation-component entropy).
    pub order: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        // Homeostatic dominates, surprise secondary, order and novelty as
        // equal-weight exploration/consolidation duals.
        Self {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
            order: 0.5,
        }
    }
}

/// The frozen reward circuit.
pub struct RewardCircuit {
    pub weights: RewardWeights,
    /// Random linear digest `φ: R^{obs_token_dim} → R^{DIGEST_DIM}`, frozen
    /// at construction. Stored row-major: `DIGEST_DIM` rows, each of length
    /// `OBS_TOKEN_DIM`.
    phi: Vec<f32>,
    recent: VecDeque<u32>,
    reference: VecDeque<u32>,
    /// Last computed `(H_reference, H_recent)` for diagnostics.
    last_h_reference: f32,
    last_h_recent: f32,
}

impl RewardCircuit {
    /// Build the frozen circuit. Uses a stable default seed for the random
    /// digest so reward semantics are deterministic across runs.
    pub fn new(weights: RewardWeights) -> Self {
        Self::with_seed(weights, 0xA11CE)
    }

    pub fn with_seed(weights: RewardWeights, seed: u64) -> Self {
        let phi = random_linear(DIGEST_DIM, OBS_TOKEN_DIM, seed);
        Self {
            weights,
            phi,
            recent: VecDeque::with_capacity(RECENT_WINDOW),
            reference: VecDeque::with_capacity(REFERENCE_WINDOW),
            last_h_reference: 0.0,
            last_h_recent: 0.0,
        }
    }

    /// Surprise: the L2 norm of world model prediction error.
    /// `pred_error` is `||W(z_{t-1}, a_{t-1}) - z_t||_2`, already computed.
    pub fn surprise(pred_error: f32) -> f32 {
        pred_error
    }

    /// Novelty: `1 / sqrt(N(z_t))` where N is the visit count.
    pub fn novelty(visit_count: u32) -> f32 {
        if visit_count == 0 {
            1.0
        } else {
            1.0 / (visit_count as f32).sqrt()
        }
    }

    /// Homeostatic balance: negative penalty for deviation from target ranges.
    /// `r_homeo = -sum_i max(0, |h_i - target_i| - tolerance_i)`
    pub fn homeostatic(variables: &[HomeostaticVariable]) -> f32 {
        let mut penalty = 0.0f32;
        for var in variables {
            let deviation = (var.value - var.target).abs() - var.tolerance;
            penalty += deviation.max(0.0);
        }
        -penalty
    }

    /// Push one observation token onto the rolling windows and return the
    /// current order signal (`H_reference − H_recent`, or 0 during warmup).
    ///
    /// `obs_token` must have length `OBS_TOKEN_DIM` — the universal-adapter
    /// output shape. Shorter slices return 0 without mutating state.
    pub fn observe_order(&mut self, obs_token: &[f32]) -> f32 {
        if obs_token.len() != OBS_TOKEN_DIM {
            return 0.0;
        }
        let bucket = self.digest_bucket(obs_token);
        push_bounded(&mut self.recent, bucket, RECENT_WINDOW);
        push_bounded(&mut self.reference, bucket, REFERENCE_WINDOW);
        self.recompute_order()
    }

    /// Most recent `(H_reference, H_recent)` pair, for diagnostics. Both
    /// zero until the first `observe_order` call.
    pub fn last_order_entropies(&self) -> (f32, f32) {
        (self.last_h_reference, self.last_h_recent)
    }

    /// Combined reward signal.
    pub fn compute(&self, surprise: f32, novelty: f32, homeostatic: f32, order: f32) -> f32 {
        self.weights.surprise * surprise
            + self.weights.novelty * novelty
            + self.weights.homeostatic * homeostatic
            + self.weights.order * order
    }

    fn digest_bucket(&self, obs_token: &[f32]) -> u32 {
        // φ · obs_token  —> DIGEST_DIM floats
        let mut digest = [0.0f32; DIGEST_DIM];
        for (d, dest) in digest.iter_mut().enumerate() {
            let row = &self.phi[d * OBS_TOKEN_DIM..(d + 1) * OBS_TOKEN_DIM];
            let mut acc = 0.0f32;
            for (w, x) in row.iter().zip(obs_token.iter()) {
                acc += w * x;
            }
            *dest = acc;
        }
        // Squash into (-1, 1) so bucket edges don't depend on input scale.
        // tanh gives a stable, monotonic mapping.
        let mut bucket_id: u32 = 0;
        for &d in digest.iter() {
            let squashed = d.tanh(); // in (-1, 1)
            // Map (-1, 1) to [0, BUCKETS_PER_DIM - 1].
            let bin =
                (((squashed * 0.5 + 0.5) * BUCKETS_PER_DIM as f32) as u32).min(BUCKETS_PER_DIM - 1);
            bucket_id = bucket_id * BUCKETS_PER_DIM + bin;
        }
        bucket_id
    }

    fn recompute_order(&mut self) -> f32 {
        let h_recent = window_entropy(&self.recent);
        let h_reference = window_entropy(&self.reference);
        self.last_h_recent = h_recent;
        self.last_h_reference = h_reference;
        if self.reference.len() < REFERENCE_WARMUP {
            return 0.0;
        }
        h_reference - h_recent
    }
}

fn push_bounded(buf: &mut VecDeque<u32>, value: u32, cap: usize) {
    if buf.len() == cap {
        buf.pop_front();
    }
    buf.push_back(value);
}

fn window_entropy(window: &VecDeque<u32>) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    let total_buckets = (BUCKETS_PER_DIM as usize).pow(DIGEST_DIM as u32);
    let mut counts = vec![0u32; total_buckets];
    for &b in window.iter() {
        counts[b as usize] += 1;
    }
    let n = window.len() as f32;
    let mut h = 0.0f32;
    for c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f32 / n;
        h -= p * p.ln();
    }
    h
}

/// Deterministic random `out_dim × in_dim` matrix, scaled by `1/sqrt(in_dim)`
/// so digest outputs stay roughly unit-scale for unit-scale inputs. Uses a
/// golden-ratio hash so the same seed always produces the same matrix — the
/// digest is frozen across runs.
fn random_linear(out_dim: usize, in_dim: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = 1.0 / (in_dim as f32).sqrt();
    let n = out_dim * in_dim;
    (0..n)
        .map(|i| {
            let h = ((seed as f64 + i as f64 * 1.234_567) * 0.618_033_988_749_895).fract() as f32;
            ((h * PI * 2.0).sin()) * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_passes_through() {
        assert_eq!(RewardCircuit::surprise(0.5), 0.5);
    }

    #[test]
    fn novelty_first_visit() {
        assert_eq!(RewardCircuit::novelty(0), 1.0);
        assert_eq!(RewardCircuit::novelty(1), 1.0);
        assert!((RewardCircuit::novelty(4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn homeostatic_in_range() {
        let vars = vec![HomeostaticVariable {
            value: 0.5,
            target: 0.5,
            tolerance: 0.1,
        }];
        assert_eq!(RewardCircuit::homeostatic(&vars), 0.0);
    }

    #[test]
    fn homeostatic_out_of_range() {
        let vars = vec![HomeostaticVariable {
            value: 1.0,
            target: 0.5,
            tolerance: 0.1,
        }];
        // deviation = |1.0 - 0.5| - 0.1 = 0.4
        assert!((RewardCircuit::homeostatic(&vars) - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn order_zero_during_warmup() {
        let mut rc = RewardCircuit::new(RewardWeights::default());
        let obs = vec![0.1f32; OBS_TOKEN_DIM];
        // First observation: reference still empty — must return 0.
        let r = rc.observe_order(&obs);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn order_rewards_concentration_relative_to_reference() {
        let mut rc = RewardCircuit::new(RewardWeights::default());

        // Seed reference window with a diverse distribution: different
        // obs patterns that map to different digest buckets.
        let mut patterns: Vec<Vec<f32>> = Vec::new();
        for k in 0..16 {
            let mut v = vec![0.0f32; OBS_TOKEN_DIM];
            for (j, slot) in v.iter_mut().enumerate() {
                *slot = ((k as f32 + j as f32 * 0.37).sin()) * 2.0;
            }
            patterns.push(v);
        }
        // Fill reference beyond warmup with the diverse patterns.
        for step in 0..REFERENCE_WARMUP + 32 {
            let p = &patterns[step % patterns.len()];
            rc.observe_order(p);
        }
        let (h_ref_before, _) = rc.last_order_entropies();

        // Now drive the agent into a single pattern for >= RECENT_WINDOW
        // steps, flushing the recent buffer to one bucket.
        let concentrated = patterns[0].clone();
        let mut last = 0.0;
        for _ in 0..RECENT_WINDOW * 2 {
            last = rc.observe_order(&concentrated);
        }
        let (_h_ref_after, h_recent_after) = rc.last_order_entropies();

        assert!(
            h_recent_after < h_ref_before,
            "recent entropy {h_recent_after} should drop below the diverse reference {h_ref_before}"
        );
        assert!(
            last > 0.0,
            "order reward should be positive once agent concentrates: got {last}"
        );
    }

    #[test]
    fn order_ignores_wrong_length_tokens() {
        let mut rc = RewardCircuit::new(RewardWeights::default());
        let r = rc.observe_order(&[0.0f32; 3]); // too short
        assert_eq!(r, 0.0);
        // Internal windows should not have been mutated.
        assert_eq!(rc.recent.len(), 0);
        assert_eq!(rc.reference.len(), 0);
    }

    #[test]
    fn combined_reward() {
        let rc = RewardCircuit::new(RewardWeights {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
            order: 0.0,
        });
        let r = rc.compute(0.2, 0.8, -0.1, 0.0);
        // 1.0*0.2 + 0.5*0.8 + 2.0*(-0.1) + 0.0*0.0 = 0.4
        assert!((r - 0.4).abs() < 1e-6);
    }
}
