//! Continuous coordinate action head (CPU).
//!
//! Purpose: on tasks with spatial-click actions (ARC-AGI-3
//! complex actions), kindle's discrete policy only picks the
//! action ID; the `(x, y)` coordinates were previously random
//! baseline noise. This head lets the policy *control* the
//! coordinates.
//!
//! Architecture: a 2-layer MLP maps the agent's coordinate features to two
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
//! The agent supplies the observation token plus its fixed task embedding.
//! An earlier `[latent z, token]` input was larger, required an otherwise
//! unnecessary GPU encoder pass, and made distilled checkpoints depend on the
//! encoder's batch topology. Deterministic task context prevents cross-game
//! coordinate interference while keeping checkpoints portable across training
//! and evaluation batch sizes.

/// 2-layer MLP `input_dim → hidden_dim → 2` with tanh on the
/// output so μ stays in `[−1, 1]`.
pub struct CoordHead {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    pub sigma: f32,
    w1: Vec<f32>, // [hidden_dim, input_dim]
    b1: Vec<f32>, // [hidden_dim]
    w2: Vec<f32>, // [2, hidden_dim]
    b2: [f32; 2],
    // Adam state used only by offline supervised imitation. Online
    // REINFORCE deliberately keeps its existing lightweight SGD update.
    supervised_m_w1: Vec<f32>,
    supervised_v_w1: Vec<f32>,
    supervised_m_b1: Vec<f32>,
    supervised_v_b1: Vec<f32>,
    supervised_m_w2: Vec<f32>,
    supervised_v_w2: Vec<f32>,
    supervised_m_b2: [f32; 2],
    supervised_v_b2: [f32; 2],
    supervised_step: u64,
    /// Last (μ_x, μ_y) emitted per lane — cached so `train_step`
    /// doesn't need the caller to pass μ back.
    per_lane_mu: Vec<[f32; 2]>,
    /// Last sampled (x, y) per lane in `[-1, 1]` space (pre-game
    /// rescale). Needed for REINFORCE's `(sample − μ)` term.
    per_lane_sample: Vec<[f32; 2]>,
    /// The features each lane's last sample was drawn from — cached so
    /// `train_step` backprops through the activations that actually
    /// produced μ. (Callers run train_step post-observe, when the
    /// lane's current state is already the next state.)
    per_lane_input: Vec<Vec<f32>>,
    pub last_loss: f32,
}

impl CoordHead {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        batch_size: usize,
        lr: f32,
        sigma: f32,
        seed: u64,
    ) -> Self {
        let w1 = xavier(hidden_dim, input_dim, seed);
        let w2 = xavier(2, hidden_dim, seed.wrapping_add(1));
        Self {
            input_dim,
            hidden_dim,
            lr,
            sigma: sigma.max(1e-3),
            supervised_m_w1: vec![0.0; w1.len()],
            supervised_v_w1: vec![0.0; w1.len()],
            supervised_m_b1: vec![0.0; hidden_dim],
            supervised_v_b1: vec![0.0; hidden_dim],
            supervised_m_w2: vec![0.0; w2.len()],
            supervised_v_w2: vec![0.0; w2.len()],
            supervised_m_b2: [0.0; 2],
            supervised_v_b2: [0.0; 2],
            supervised_step: 0,
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: [0.0; 2],
            per_lane_mu: vec![[0.0, 0.0]; batch_size.max(1)],
            per_lane_sample: vec![[0.0, 0.0]; batch_size.max(1)],
            per_lane_input: vec![Vec::new(); batch_size.max(1)],
            last_loss: 0.0,
        }
    }

    /// Forward on coordinate features → `(μ_x, μ_y)` in `[−1, 1]`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, z: &[f32]) -> [f32; 2] {
        debug_assert_eq!(z.len(), self.input_dim);
        let mut h = vec![0.0f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.input_dim..(j + 1) * self.input_dim];
            for k in 0..self.input_dim {
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

    /// One full-batch Adam update for behavior cloning. Closely spaced ARC
    /// states produce similarly encoded latents; accumulating their gradients
    /// at a common parameter snapshot avoids the sample-order bias of repeated
    /// SGD updates and Adam supplies the per-parameter scaling needed to
    /// distinguish those small state changes.
    pub fn train_supervised_batch(
        &mut self,
        input_stack: &[f32],
        targets: &[[f32; 2]],
        active: &[bool],
        weight: f32,
    ) -> f32 {
        if targets.len() != active.len()
            || input_stack.len() != targets.len() * self.input_dim
            || !weight.is_finite()
            || weight <= 0.0
        {
            return 0.0;
        }
        let sample_weights = active
            .iter()
            .map(|&enabled| if enabled { weight } else { 0.0 })
            .collect::<Vec<_>>();
        self.train_supervised_batch_weighted(input_stack, targets, &sample_weights)
    }

    /// Full-batch Adam behavior cloning with a non-negative weight per row.
    /// Multi-task callers use this to make each task contribute equal total
    /// gradient even when successful trajectories have very different lengths.
    pub fn train_supervised_batch_weighted(
        &mut self,
        input_stack: &[f32],
        targets: &[[f32; 2]],
        sample_weights: &[f32],
    ) -> f32 {
        if targets.len() != sample_weights.len()
            || input_stack.len() != targets.len() * self.input_dim
            || sample_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return 0.0;
        }
        let mut g_w1 = vec![0.0f32; self.w1.len()];
        let mut g_b1 = vec![0.0f32; self.b1.len()];
        let mut g_w2 = vec![0.0f32; self.w2.len()];
        let mut g_b2 = [0.0f32; 2];
        let mut loss = 0.0f32;
        let mut count = 0usize;

        for (i, (&target, &sample_weight)) in targets.iter().zip(sample_weights).enumerate() {
            if sample_weight == 0.0 {
                continue;
            }
            let z = &input_stack[i * self.input_dim..(i + 1) * self.input_dim];
            let target = target.map(|v| {
                if v.is_finite() {
                    v.clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            });
            let (h, mask, mu) = self.forward_with_activations(z);
            let mut output_signal = [0.0f32; 2];
            for o in 0..2 {
                let residual = target[o] - mu[o];
                loss += 0.5 * sample_weight * residual * residual;
                output_signal[o] = sample_weight * residual * (1.0 - mu[o] * mu[o]);
            }
            let mut d_h = vec![0.0f32; self.hidden_dim];
            for o in 0..2 {
                let row_off = o * self.hidden_dim;
                for j in 0..self.hidden_dim {
                    g_w2[row_off + j] += output_signal[o] * h[j];
                    if mask[j] {
                        d_h[j] += output_signal[o] * self.w2[row_off + j];
                    }
                }
                g_b2[o] += output_signal[o];
            }
            for j in 0..self.hidden_dim {
                if !mask[j] {
                    continue;
                }
                g_b1[j] += d_h[j];
                let row_off = j * self.input_dim;
                for k in 0..self.input_dim {
                    g_w1[row_off + k] += d_h[j] * z[k];
                }
            }
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }

        let scale = 1.0 / count as f32;
        self.supervised_step = self.supervised_step.saturating_add(1);
        let correction1 = 1.0 - 0.9f32.powf(self.supervised_step as f32);
        let correction2 = 1.0 - 0.999f32.powf(self.supervised_step as f32);
        let lr = self.lr;
        for i in 0..self.w1.len() {
            adam_ascent(
                &mut self.w1[i],
                g_w1[i] * scale,
                &mut self.supervised_m_w1[i],
                &mut self.supervised_v_w1[i],
                lr,
                correction1,
                correction2,
            );
        }
        for i in 0..self.b1.len() {
            adam_ascent(
                &mut self.b1[i],
                g_b1[i] * scale,
                &mut self.supervised_m_b1[i],
                &mut self.supervised_v_b1[i],
                lr,
                correction1,
                correction2,
            );
        }
        for i in 0..self.w2.len() {
            adam_ascent(
                &mut self.w2[i],
                g_w2[i] * scale,
                &mut self.supervised_m_w2[i],
                &mut self.supervised_v_w2[i],
                lr,
                correction1,
                correction2,
            );
        }
        for o in 0..2 {
            adam_ascent(
                &mut self.b2[o],
                g_b2[o] * scale,
                &mut self.supervised_m_b2[o],
                &mut self.supervised_v_b2[o],
                lr,
                correction1,
                correction2,
            );
        }
        self.last_loss = loss / count as f32;
        self.last_loss
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
            self.per_lane_input[lane_idx].clear();
            self.per_lane_input[lane_idx].extend_from_slice(z);
        }
        [sx, sy]
    }

    /// REINFORCE update for one lane. Applies
    /// `grad(μ) = (sample − μ) / σ² · advantage` through the
    /// MLP. `lane_idx` must match an earlier `sample` call with
    /// the same lane; the update backprops through the features
    /// cached by that `sample` call (the state μ was computed
    /// from), not whatever the lane's current state is.
    ///
    /// Returns the (signed) reward-weighted log-likelihood of the
    /// sample (for diagnostics); lower = worse; the head is
    /// minimizing negative-advantage-weighted log-likelihood.
    #[allow(clippy::needless_range_loop)]
    pub fn train_step(&mut self, lane_idx: usize, advantage: f32) -> f32 {
        if lane_idx >= self.per_lane_mu.len() {
            return 0.0;
        }
        if advantage.abs() < 1e-8 {
            return 0.0;
        }
        if self.per_lane_input[lane_idx].len() != self.input_dim {
            // No matching sample() recorded for this lane yet.
            return 0.0;
        }
        let z = std::mem::take(&mut self.per_lane_input[lane_idx]);
        let mu = self.per_lane_mu[lane_idx];
        let sample = self.per_lane_sample[lane_idx];
        // Forward, saving activations (mirrors `forward` but with
        // intermediate state retained for backprop).
        let (h, mask, _) = self.forward_with_activations(&z);
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

        self.apply_output_signal(&z, &h, &mask, d_mu_pre_tanh);
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

    fn forward_with_activations(&self, z: &[f32]) -> (Vec<f32>, Vec<bool>, [f32; 2]) {
        let mut h = vec![0.0f32; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.input_dim..(j + 1) * self.input_dim];
            for k in 0..self.input_dim {
                acc += row[k] * z[k];
            }
            mask[j] = acc > 0.0;
            h[j] = if mask[j] { acc } else { 0.0 };
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
        (h, mask, out)
    }

    /// Apply an ascent-direction signal at the pre-tanh output.
    fn apply_output_signal(
        &mut self,
        z: &[f32],
        h: &[f32],
        mask: &[bool],
        output_signal: [f32; 2],
    ) {
        let mut d_h = vec![0.0f32; self.hidden_dim];
        let lr = self.lr;
        // Accumulate the hidden signal from the forward-pass weights before
        // mutating w2, avoiding an O(lr*d²*h) update-order bias.
        for o in 0..2 {
            let row_off = o * self.hidden_dim;
            for j in 0..self.hidden_dim {
                let w_old = self.w2[row_off + j];
                self.w2[row_off + j] += lr * output_signal[o] * h[j];
                if mask[j] {
                    d_h[j] += output_signal[o] * w_old;
                }
            }
            self.b2[o] += lr * output_signal[o];
        }
        for j in 0..self.hidden_dim {
            if !mask[j] {
                continue;
            }
            self.b1[j] += lr * d_h[j];
            let row_off = j * self.input_dim;
            for k in 0..self.input_dim {
                self.w1[row_off + k] += lr * d_h[j] * z[k];
            }
        }
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(24 + self.parameter_count() * 4);
        bytes.extend_from_slice(b"KCOORD1\0");
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.hidden_dim as u64).to_le_bytes());
        for value in self.parameters() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)
    }

    pub fn load(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        let bytes = std::fs::read(path)?;
        if bytes.len() < 24 || &bytes[..8] != b"KCOORD1\0" {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid coord checkpoint",
            ));
        }
        let input_dim = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let hidden_dim = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        if input_dim != self.input_dim || hidden_dim != self.hidden_dim {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "coord checkpoint shape ({input_dim}, {hidden_dim}) != ({}, {})",
                    self.input_dim, self.hidden_dim
                ),
            ));
        }
        let expected = 24 + self.parameter_count() * 4;
        if bytes.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "coord checkpoint has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        let values = bytes[24..]
            // `bytes.len() == expected` above proves that there is no
            // partial float at the end.
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        self.load_parameters(&values)
            .map_err(|message| Error::new(ErrorKind::InvalidData, message))
    }

    fn parameter_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn parameters(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.parameter_count());
        values.extend_from_slice(&self.w1);
        values.extend_from_slice(&self.b1);
        values.extend_from_slice(&self.w2);
        values.extend_from_slice(&self.b2);
        values
    }

    fn load_parameters(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() != self.parameter_count() {
            return Err(format!(
                "coord checkpoint has {} parameters, expected {}",
                values.len(),
                self.parameter_count()
            ));
        }
        let mut offset = 0;
        for destination in [&mut self.w1, &mut self.b1, &mut self.w2] {
            let end = offset + destination.len();
            destination.copy_from_slice(&values[offset..end]);
            offset = end;
        }
        self.b2.copy_from_slice(&values[offset..offset + 2]);
        Ok(())
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

fn adam_ascent(
    parameter: &mut f32,
    gradient: f32,
    first_moment: &mut f32,
    second_moment: &mut f32,
    lr: f32,
    correction1: f32,
    correction2: f32,
) {
    *first_moment = 0.9 * *first_moment + 0.1 * gradient;
    *second_moment = 0.999 * *second_moment + 0.001 * gradient * gradient;
    let m_hat = *first_moment / correction1.max(f32::EPSILON);
    let v_hat = *second_moment / correction2.max(f32::EPSILON);
    *parameter += lr * m_hat / (v_hat.sqrt() + 1e-8);
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
        assert!((-1.0..=1.0).contains(&mx), "mu_x: {mx}");
        assert!((-1.0..=1.0).contains(&my), "mu_y: {my}");
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
            h.train_step(0, 1.0); // positive advantage
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
        h.train_step(0, 0.0);
        assert_eq!(h.w1, w1_before, "zero-advantage update must be a no-op");
    }

    #[test]
    fn supervised_batch_separates_nearby_states() {
        let mut h = CoordHead::new(4, 32, 4, 2e-3, 0.2, 117);
        let inputs = [
            0.50, 0.10, 0.20, 0.30, 0.50, 0.11, 0.20, 0.30, 0.50, 0.12, 0.20, 0.30, 0.50, 0.13,
            0.20, 0.30,
        ];
        let targets = [[-0.75, -0.5], [-0.25, -0.16], [0.25, 0.16], [0.75, 0.5]];
        let active = [true; 4];
        for _ in 0..3000 {
            h.train_supervised_batch(&inputs, &targets, &active, 1.0);
        }
        let mse = inputs
            .chunks(4)
            .zip(targets)
            .map(|(input, target)| {
                let prediction = h.forward(input);
                (prediction[0] - target[0]).powi(2) + (prediction[1] - target[1]).powi(2)
            })
            .sum::<f32>()
            / 4.0;
        assert!(mse < 2e-3, "batch imitation mse={mse}");
    }

    #[test]
    fn coordinate_checkpoint_round_trips() {
        let mut source = CoordHead::new(4, 8, 1, 1e-2, 0.2, 12);
        let z = [0.1, 0.2, 0.3, 0.4];
        for _ in 0..20 {
            source.train_supervised_batch(&z, &[[0.7, -0.4]], &[true], 1.0);
        }
        let path = std::env::temp_dir().join(format!(
            "kindle-coord-{}-{}.bin",
            std::process::id(),
            source.parameters()[0].to_bits()
        ));
        source.save(&path).unwrap();
        let mut loaded = CoordHead::new(4, 8, 1, 1e-2, 0.2, 99);
        loaded.load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(source.forward(&z), loaded.forward(&z));
    }
}
