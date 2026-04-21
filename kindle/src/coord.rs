//! Continuous coordinate action head (CPU).
//!
//! Purpose: on tasks with spatial-click actions (ARC-AGI-3
//! complex actions), kindle's discrete policy only picks the
//! action ID; the `(x, y)` coordinates were previously random
//! baseline noise. This head lets the policy *control* the
//! coordinates.
//!
//! Architecture: a 2-layer MLP maps the agent's latent `z` to two
//! means `(μ_x, μ_y) ∈ [−1, 1]` (tanh-squashed). At act-time, the
//! harness samples `x, y ~ Normal(μ, σ²)` with `σ` fixed for
//! exploration, clamps to `[−1, 1]`, then maps to the game's
//! integer coordinate range via `round((v + 1)/2 · max_coord)`.
//!
//! Training: REINFORCE with a scalar advantage per step.
//! Gradient w.r.t. μ is `(sample − μ) / σ² · advantage` (natural
//! Gaussian policy gradient). The advantage is whatever signal
//! the harness supplies — typically the same per-step reward
//! kindle emits to its main policy, so the coord head learns to
//! pick `(x, y)` that correlate with high-reward steps.
//!
//! Kept CPU-side (mirrors outcome / RND / approach modules).
//! Stop-grad into the encoder is automatic at the CPU boundary.

/// 2-layer MLP `latent_dim → hidden_dim → 2` with tanh on the
/// output so μ stays in `[−1, 1]`.
pub struct CoordHead {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    pub sigma: f32,
    w1: Vec<f32>, // [hidden_dim, latent_dim]
    b1: Vec<f32>, // [hidden_dim]
    w2: Vec<f32>, // [2, hidden_dim]
    b2: [f32; 2],
    /// Last (μ_x, μ_y) emitted per lane — cached so `train_step`
    /// doesn't need the caller to pass μ back.
    per_lane_mu: Vec<[f32; 2]>,
    /// Last sampled (x, y) per lane in `[-1, 1]` space (pre-game
    /// rescale). Needed for REINFORCE's `(sample − μ)` term.
    per_lane_sample: Vec<[f32; 2]>,
    pub last_loss: f32,
}

impl CoordHead {
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        batch_size: usize,
        lr: f32,
        sigma: f32,
        seed: u64,
    ) -> Self {
        Self {
            latent_dim,
            hidden_dim,
            lr,
            sigma: sigma.max(1e-3),
            w1: xavier(hidden_dim, latent_dim, seed),
            b1: vec![0.0; hidden_dim],
            w2: xavier(2, hidden_dim, seed.wrapping_add(1)),
            b2: [0.0; 2],
            per_lane_mu: vec![[0.0, 0.0]; batch_size.max(1)],
            per_lane_sample: vec![[0.0, 0.0]; batch_size.max(1)],
            last_loss: 0.0,
        }
    }

    /// Forward on `z` → `(μ_x, μ_y)` in `[−1, 1]`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, z: &[f32]) -> [f32; 2] {
        debug_assert_eq!(z.len(), self.latent_dim);
        let mut h = vec![0.0f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.latent_dim..(j + 1) * self.latent_dim];
            for k in 0..self.latent_dim {
                acc += row[k] * z[k];
            }
            h[j] = acc.max(0.0); // ReLU
        }
        let mut out = [0.0f32; 2];
        for o in 0..2 {
            let mut acc = self.b2[o];
            let row = &self.w2[o * self.hidden_dim..(o + 1) * self.hidden_dim];
            for j in 0..self.hidden_dim {
                acc += row[j] * h[j];
            }
            out[o] = acc.tanh();
        }
        out
    }

    /// Sample `(x, y)` for one lane. `rand_pair` is a function
    /// that returns two standard-normal samples (so callers can
    /// keep their RNG state). The sample is cached per-lane for
    /// the next `train_step` call.
    pub fn sample(
        &mut self,
        lane_idx: usize,
        z: &[f32],
        mut rand_pair: impl FnMut() -> (f32, f32),
    ) -> [f32; 2] {
        let mu = self.forward(z);
        let (nx, ny) = rand_pair();
        let sx = (mu[0] + self.sigma * nx).clamp(-1.0, 1.0);
        let sy = (mu[1] + self.sigma * ny).clamp(-1.0, 1.0);
        if lane_idx < self.per_lane_mu.len() {
            self.per_lane_mu[lane_idx] = mu;
            self.per_lane_sample[lane_idx] = [sx, sy];
        }
        [sx, sy]
    }

    /// REINFORCE update for one lane. Applies
    /// `grad(μ) = (sample − μ) / σ² · advantage` through the
    /// MLP. `lane_idx` must match an earlier `sample` call with
    /// the same lane.
    ///
    /// Returns the (signed) reward-weighted log-likelihood of the
    /// sample (for diagnostics); lower = worse; the head is
    /// minimizing negative-advantage-weighted log-likelihood.
    #[allow(clippy::needless_range_loop)]
    pub fn train_step(&mut self, lane_idx: usize, z: &[f32], advantage: f32) -> f32 {
        if lane_idx >= self.per_lane_mu.len() {
            return 0.0;
        }
        if advantage.abs() < 1e-8 {
            return 0.0;
        }
        let mu = self.per_lane_mu[lane_idx];
        let sample = self.per_lane_sample[lane_idx];
        // Forward, saving activations (mirrors `forward` but with
        // intermediate state retained for backprop).
        let mut h = vec![0.0f32; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.latent_dim..(j + 1) * self.latent_dim];
            for k in 0..self.latent_dim {
                acc += row[k] * z[k];
            }
            mask[j] = acc > 0.0;
            h[j] = if mask[j] { acc } else { 0.0 };
        }
        // For each coord: d_loss/d_mu = −adv · (sample − μ) / σ²
        // (negative because we gradient-descend to minimize NLL
        // weighted by −advantage; sign flipped so positive
        // advantage pushes μ TOWARD the sample).
        let inv_sigma_sq = 1.0 / (self.sigma * self.sigma);
        let mut d_mu_pre_tanh = [0.0f32; 2];
        for o in 0..2 {
            let residual = sample[o] - mu[o];
            let d_mu = advantage * residual * inv_sigma_sq;
            // Chain through tanh: d_pre = d_mu * (1 − μ²).
            let d_pre = d_mu * (1.0 - mu[o] * mu[o]);
            d_mu_pre_tanh[o] = d_pre;
        }

        // Backprop into the hidden layer.
        let mut d_h = vec![0.0f32; self.hidden_dim];
        let lr = self.lr;
        // w2 grads: dL/dw2[o, j] = d_pre[o] * h[j]. Update then
        // accumulate d_h[j] = Σ_o d_pre[o] * w2[o, j].
        for o in 0..2 {
            let row_off = o * self.hidden_dim;
            for j in 0..self.hidden_dim {
                self.w2[row_off + j] += lr * d_mu_pre_tanh[o] * h[j];
                if mask[j] {
                    d_h[j] += d_mu_pre_tanh[o] * self.w2[row_off + j];
                }
            }
            self.b2[o] += lr * d_mu_pre_tanh[o];
        }
        // w1, b1 grads through relu.
        for j in 0..self.hidden_dim {
            if !mask[j] {
                continue;
            }
            self.b1[j] += lr * d_h[j];
            let row_off = j * self.latent_dim;
            for k in 0..self.latent_dim {
                self.w1[row_off + k] += lr * d_h[j] * z[k];
            }
        }
        // Diagnostic "loss" (NLL of sample under current μ, up to
        // constants). Positive means sample is unlikely under μ.
        let mut nll = 0.0f32;
        for o in 0..2 {
            let d = sample[o] - mu[o];
            nll += d * d * inv_sigma_sq * 0.5;
        }
        self.last_loss = nll;
        nll
    }
}

fn xavier(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    (0..n as u64)
        .map(|i| {
            // splitmix64 — same pattern as rnd::xavier (works for
            // full-range u64 seeds, unlike the fract-of-huge-f64
            // hash used in the older CPU heads).
            let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            x ^= x >> 33;
            let u = (x >> 40) as f32 / (1u64 << 24) as f32;
            (u * PI * 2.0).sin() * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counter-style standard-normal "generator" that just cycles
    /// through a fixed list of values. Enough for deterministic
    /// tests; real callers pass a proper RNG.
    fn fixed_normals(values: Vec<(f32, f32)>) -> impl FnMut() -> (f32, f32) {
        use std::cell::RefCell;
        let state = RefCell::new(values.into_iter().cycle());
        move || state.borrow_mut().next().unwrap()
    }

    #[test]
    fn forward_returns_values_in_minus_one_to_one() {
        let h = CoordHead::new(4, 8, 1, 1e-2, 0.2, 42);
        let z = vec![0.5f32; 4];
        let [mx, my] = h.forward(&z);
        assert!(mx >= -1.0 && mx <= 1.0, "mu_x: {mx}");
        assert!(my >= -1.0 && my <= 1.0, "mu_y: {my}");
    }

    #[test]
    fn sample_stays_in_range_and_caches_per_lane() {
        let mut h = CoordHead::new(4, 8, 2, 1e-2, 0.2, 13);
        let z = vec![0.2f32; 4];
        let s0 = h.sample(0, &z, fixed_normals(vec![(0.5, -0.3)]));
        let s1 = h.sample(1, &z, fixed_normals(vec![(1.0, 1.0)]));
        assert!(s0[0].abs() <= 1.0 && s0[1].abs() <= 1.0);
        assert!(s1[0].abs() <= 1.0 && s1[1].abs() <= 1.0);
        // Per-lane cache retained.
        assert_eq!(h.per_lane_sample[0], s0);
        assert_eq!(h.per_lane_sample[1], s1);
    }

    #[test]
    fn positive_advantage_pulls_mu_toward_sample() {
        // If a sample got high advantage, repeated training on
        // the same z should move μ toward the sample value.
        let mut h = CoordHead::new(4, 16, 1, 5e-2, 0.3, 77);
        let z = vec![0.4f32; 4];
        // Force a sample far from the initial mu in the +x direction.
        let init_mu = h.forward(&z);
        // Sample with a fixed very-positive noise so sample[0] is
        // near +1 regardless of init mu.
        let rng_high_x = fixed_normals(vec![(3.0, 0.0)]);
        let _s = h.sample(0, &z, rng_high_x);
        for _ in 0..200 {
            h.train_step(0, &z, 1.0); // positive advantage
            // Resample to keep the stored sample valid (the head
            // stores the last sample; in a real loop the caller
            // re-samples each step).
            let _ = h.sample(0, &z, fixed_normals(vec![(3.0, 0.0)]));
        }
        let final_mu = h.forward(&z);
        assert!(
            final_mu[0] > init_mu[0] + 0.05,
            "positive advantage on +x sample should raise mu_x: {} → {}",
            init_mu[0],
            final_mu[0]
        );
    }

    #[test]
    fn zero_advantage_noops() {
        // If advantage is zero, train_step should neither update
        // weights nor move mu.
        let mut h = CoordHead::new(4, 8, 1, 5e-2, 0.2, 31);
        let z = vec![0.1f32; 4];
        let _s = h.sample(0, &z, fixed_normals(vec![(0.5, 0.5)]));
        let w1_before = h.w1.clone();
        h.train_step(0, &z, 0.0);
        assert_eq!(h.w1, w1_before, "zero-advantage update must be a no-op");
    }
}
