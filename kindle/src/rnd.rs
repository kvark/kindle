//! Random Network Distillation curiosity primitive (Burda et al.
//! 2018).
//!
//! The curiosity-sweep on ARC-AGI-3 (commit 780c9a2) showed
//! kindle's two existing curiosity signals degenerate on visual
//! tasks: **surprise** dies once the world-model's prediction
//! error saturates, and **latent-space novelty** collapses when a
//! CNN encoder compresses all game states into a tight region
//! that shares visit-count buckets.
//!
//! RND sidesteps both failure modes:
//!
//!   - A **frozen random target** network `f_T: z → R^k` is built
//!     once at init. It never trains. Its output on any particular
//!     `z` is a fixed deterministic-random feature vector.
//!
//!   - A **trainable predictor** `f_P: z → R^k` is trained every
//!     step by MSE against `f_T(z)`. It learns to predict the
//!     target's output on frequently-visited `z` and fails to
//!     predict on rare `z`.
//!
//!   - The intrinsic reward is `||f_P(z) − f_T(z)||² / k` —
//!     proportional to how unfamiliar `z` is *to the predictor
//!     specifically*. Unlike WM surprise, this doesn't decay as
//!     kindle's WM converges, because the target is independent of
//!     world-model quality.
//!
//! CPU implementation mirrors the M6 / M7 pattern: at kindle's
//! latent dims (~16), the 2-layer MLP is trivially fast in Rust.
//! Stop-grad is automatic: the encoder never sees RND's gradient
//! (we call forward on a detached latent snapshot, not a graph
//! node).

/// 2-layer MLP `latent_dim → hidden_dim → feature_dim` with ReLU
/// between. No biases (RND papers use unbiased linear layers).
struct TwoLayerMlp {
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    w1: Vec<f32>, // [hidden_dim, in_dim] row-major
    w2: Vec<f32>, // [out_dim, hidden_dim] row-major
}

impl TwoLayerMlp {
    fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            in_dim,
            hidden_dim,
            out_dim,
            w1: xavier(hidden_dim, in_dim, seed),
            w2: xavier(out_dim, hidden_dim, seed.wrapping_add(1)),
        }
    }

    /// Forward. Returns the `out_dim`-long feature vector.
    fn forward(&self, z: &[f32]) -> Vec<f32> {
        debug_assert_eq!(z.len(), self.in_dim);
        let mut h = vec![0.0f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = 0.0f32;
            let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for k in 0..self.in_dim {
                acc += row[k] * z[k];
            }
            h[j] = acc.max(0.0); // ReLU
        }
        let mut out = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let mut acc = 0.0f32;
            let row = &self.w2[o * self.hidden_dim..(o + 1) * self.hidden_dim];
            for j in 0..self.hidden_dim {
                acc += row[j] * h[j];
            }
            out[o] = acc;
        }
        out
    }

    /// One SGD step against a target feature vector. Returns the
    /// mean squared error pre-update (the same scalar we use for
    /// the curiosity reward, so the caller doesn't need a separate
    /// forward).
    #[allow(clippy::needless_range_loop)]
    fn train_step(&mut self, z: &[f32], target: &[f32], lr: f32) -> f32 {
        debug_assert_eq!(z.len(), self.in_dim);
        debug_assert_eq!(target.len(), self.out_dim);
        // Forward (recompute — cheap for small nets, simpler code).
        let mut h = vec![0.0f32; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = 0.0f32;
            let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for k in 0..self.in_dim {
                acc += row[k] * z[k];
            }
            mask[j] = acc > 0.0;
            h[j] = if mask[j] { acc } else { 0.0 };
        }
        let mut y = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let mut acc = 0.0f32;
            let row = &self.w2[o * self.hidden_dim..(o + 1) * self.hidden_dim];
            for j in 0..self.hidden_dim {
                acc += row[j] * h[j];
            }
            y[o] = acc;
        }

        // Per-dim residual and MSE for the reward.
        let mut sq = 0.0f32;
        let mut d_y = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let d = y[o] - target[o];
            d_y[o] = d;
            sq += d * d;
        }
        let mse = sq / self.out_dim as f32;

        // Backprop. w2[o, j]: grad = d_y[o] * h[j]; h[j] grad = Σ_o d_y[o] w2[o, j] (only if mask[j]).
        let mut d_h = vec![0.0f32; self.hidden_dim];
        for o in 0..self.out_dim {
            let row_off = o * self.hidden_dim;
            for j in 0..self.hidden_dim {
                self.w2[row_off + j] -= lr * d_y[o] * h[j];
                if mask[j] {
                    d_h[j] += d_y[o] * self.w2[row_off + j];
                }
            }
        }
        // w1[j, k]: grad = d_h[j] * z[k] (only if mask[j]).
        for j in 0..self.hidden_dim {
            if !mask[j] {
                continue;
            }
            let row_off = j * self.in_dim;
            for k in 0..self.in_dim {
                self.w1[row_off + k] -= lr * d_h[j] * z[k];
            }
        }
        mse
    }
}

/// RND curiosity state. Holds a frozen random target and a
/// trainable predictor, both MLPs on the agent's latent `z`.
pub struct RndState {
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    pub last_mse: f32,
    target: TwoLayerMlp,
    predictor: TwoLayerMlp,
}

impl RndState {
    pub fn new(
        latent_dim: usize,
        feature_dim: usize,
        hidden_dim: usize,
        lr: f32,
        seed: u64,
    ) -> Self {
        // Different seeds for target vs predictor so they start
        // disagreeing (otherwise MSE is 0 at init and the reward
        // signal is dead on the first step before training moves
        // anything).
        Self {
            feature_dim,
            hidden_dim,
            lr,
            last_mse: 0.0,
            target: TwoLayerMlp::new(
                latent_dim,
                hidden_dim,
                feature_dim,
                seed ^ 0xDEAD_BEEF_DEAD_BEEF,
            ),
            predictor: TwoLayerMlp::new(
                latent_dim,
                hidden_dim,
                feature_dim,
                seed ^ 0xCAFE_F00D_CAFE_F00D,
            ),
        }
    }

    /// Step the predictor against the frozen target on `z`, and
    /// return the squared-prediction-error (per-dim-averaged) that
    /// should be emitted as the curiosity reward on this step.
    ///
    /// Returning the pre-update MSE matches Burda et al.'s
    /// formulation: the reward is the predictor's CURRENT
    /// confusion, not the lower post-update value.
    pub fn step(&mut self, z: &[f32]) -> f32 {
        debug_assert_eq!(z.len(), self.target.in_dim);
        let target_out = self.target.forward(z);
        let mse = self.predictor.train_step(z, &target_out, self.lr);
        self.last_mse = mse;
        mse
    }

    /// Pure forward-only reward read; does not train the predictor.
    /// Useful for diagnostics.
    pub fn reward(&self, z: &[f32]) -> f32 {
        let t = self.target.forward(z);
        let p = self.predictor.forward(z);
        let mut sq = 0.0f32;
        for (ti, pi) in t.iter().zip(p.iter()) {
            let d = pi - ti;
            sq += d * d;
        }
        sq / self.feature_dim as f32
    }
}

/// Xavier-uniform init using splitmix64-style hashing. Unlike the
/// golden-ratio-fract approach used in other CPU heads, this one
/// handles full-range u64 seeds correctly (those lose precision
/// when cast to f64 and fed through `.fract()`). RND specifically
/// needs that: its target/predictor seeds are XOR'd with 64-bit
/// magic constants to guarantee they differ from each other, which
/// pushes the raw u64 outside f64's precision band.
fn xavier(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    (0..n as u64)
        .map(|i| {
            // splitmix64 on (seed, i).
            let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            x ^= x >> 33;
            // Top 24 bits → [0, 1).
            let u = (x >> 40) as f32 / (1u64 << 24) as f32;
            // Same sin-based shape as the other heads, so the init
            // distribution width matches.
            (u * PI * 2.0).sin() * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rnd_init_emits_positive_reward() {
        // At init, target and predictor disagree by construction
        // (different seeds), so the very first `step` should return
        // a strictly positive MSE.
        let mut s = RndState::new(8, 16, 32, 1e-3, 42);
        let z = vec![0.1f32; 8];
        let mse = s.step(&z);
        assert!(mse > 0.0, "init MSE should be > 0, got {mse}");
    }

    #[test]
    fn rnd_predictor_fits_target_on_repeated_input() {
        // On the SAME z repeatedly, the predictor should converge
        // onto the target's output, driving MSE toward zero.
        let mut s = RndState::new(8, 16, 32, 5e-2, 7);
        let z = vec![0.3f32; 8];
        let initial = s.step(&z);
        for _ in 0..2000 {
            s.step(&z);
        }
        let final_mse = s.step(&z);
        assert!(
            final_mse < initial * 0.1,
            "predictor should fit target on repeated z: {initial} → {final_mse}"
        );
        assert!(final_mse < 1e-2, "final MSE should be near zero: {final_mse}");
    }

    #[test]
    fn rnd_rewards_novel_states_higher() {
        // After training heavily on one z, a DIFFERENT z (that the
        // predictor hasn't seen) should yield higher MSE.
        let mut s = RndState::new(8, 16, 32, 2e-2, 13);
        let z_familiar = vec![0.2f32; 8];
        for _ in 0..1500 {
            s.step(&z_familiar);
        }
        let r_familiar = s.reward(&z_familiar);
        let z_novel: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let r_novel = s.reward(&z_novel);
        assert!(
            r_novel > r_familiar * 3.0,
            "novel z should yield higher RND reward than familiar z: familiar={r_familiar}, novel={r_novel}"
        );
    }

    #[test]
    fn rnd_target_does_not_drift() {
        // The target weights should be identical before and after a
        // train_step — only the predictor updates.
        let mut s = RndState::new(4, 8, 16, 1e-1, 99);
        let w_before = s.target.w1.clone();
        let z = vec![0.5f32; 4];
        for _ in 0..100 {
            s.step(&z);
        }
        assert_eq!(w_before, s.target.w1, "target net should be frozen");
    }
}
