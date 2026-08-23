//! Circular experience buffer for continual learning.
//!
//! All modules draw from and write to a shared `ExperienceBuffer`.
//! There is no episodic boundary — experience accumulates continuously.

use hashbrown::HashMap;
use rand::Rng;

/// Fixed-capacity circular buffer.
pub struct RingBuffer<T> {
    data: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            head: 0,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a new item, overwriting the oldest when full.
    pub fn push(&mut self, item: T) {
        let cap = self.data.len();
        self.data[self.head] = item;
        self.head = (self.head + 1) % cap;
        if self.len < cap {
            self.len += 1;
        }
    }

    /// Get item by logical index (0 = oldest still in buffer).
    pub fn get(&self, index: usize) -> &T {
        assert!(index < self.len, "index out of bounds");
        let cap = self.data.len();
        let start = if self.len < cap { 0 } else { self.head };
        &self.data[(start + index) % cap]
    }

    /// Get mutable item by logical index (0 = oldest still in buffer).
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len, "index out of bounds");
        let cap = self.data.len();
        let start = if self.len < cap { 0 } else { self.head };
        &mut self.data[(start + index) % cap]
    }

    /// Get the most recent item.
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let cap = self.data.len();
        let idx = (self.head + cap - 1) % cap;
        Some(&self.data[idx])
    }

    /// Return the N most recent items, oldest first.
    pub fn recent(&self, n: usize) -> Vec<&T> {
        let n = n.min(self.len);
        let start = self.len.saturating_sub(n);
        (start..self.len).map(|i| self.get(i)).collect()
    }

    /// Sample `n` random indices (with replacement) from the buffer.
    /// Returns an empty vec when the buffer is empty (an empty range
    /// would otherwise panic in `random_range`).
    pub fn sample_indices<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<usize> {
        if self.len == 0 {
            return Vec::new();
        }
        (0..n).map(|_| rng.random_range(0..self.len)).collect()
    }
}

/// A single timestep stored in the experience buffer.
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub observation: Vec<f32>,
    pub latent: Vec<f32>,
    pub action: Vec<f32>,
    /// Optional normalized parameters consumed by the action (currently ARC
    /// click x/y). Length ACTION_PARAMETER_DIM for agent-collected
    /// transitions; zeros mean no parameters or a centered parameter value.
    /// Kept separate so discrete policy labels remain valid one-hot vectors.
    pub action_parameters: Vec<f32>,
    /// Discrete-action availability mask in effect when `action` was
    /// sampled. Length MAX_ACTION_DIM for agent-collected transitions.
    /// Delayed PPO updates need the acted-from state's mask, not the mask
    /// of whichever environment state is live at update time.
    pub action_mask: Vec<f32>,
    pub reward: f32,
    pub pred_error: f32,
    /// Value baseline V(s_t) at this step, cached from the act-time
    /// policy forward. Used by the n-step advantage path (see
    /// `AgentConfig::n_step`) to compute `R − V(s_old)` without a
    /// second forward on old state.
    pub value: f32,
    /// Probability π_old(a_t | s_t) of the action actually taken, cached
    /// at act-time under the policy that collected this transition. Used
    /// as the denominator in PPO's importance ratio; unused by the plain
    /// policy-gradient path. Always in (0, 1].
    pub prob_taken: f32,
    /// Full pre-softmax logits at action time (length MAX_ACTION_DIM).
    /// Populated only when AgentConfig::use_kl_ppo is on; otherwise empty.
    /// Used by the KL-penalty PPO path to compute KL(π_new ‖ π_old)
    /// exactly via softmax(old_logits) at training time.
    pub logits_at_action: Vec<f32>,
    /// L1 option index active at this step. Mirrors `lane.current_option`
    /// at push time; needed so the n-step training forward can feed the
    /// option_onehot that matches the old state (options can change
    /// within the n-step lookahead window at short horizons).
    pub option_idx: u32,
    /// Environment id (from the adapter active when this step was recorded).
    /// Used to stratify sampling and mark env-boundary resets.
    pub env_id: u32,
    /// Boundary flag: true on the first transition after `switch_env`, so
    /// the world model and return estimators do not cross environments.
    pub env_boundary: bool,
    /// Total episode return at the time this episode COMPLETED. Set
    /// retroactively when the episode-end boundary is detected
    /// (see `Agent::backfill_episode_returns`); zero until then.
    /// Used by `use_grpo_episode` advantage to apply the same
    /// per-episode score to every transition in the episode. A
    /// transition with `episode_return == 0` is either pre-first-
    /// episode-completion or comes from an episode whose return was
    /// genuinely zero — the GRPO update path uses an explicit
    /// "episode complete" flag rather than relying on this value.
    pub episode_return: f32,
    /// Set when the transition's containing episode has been
    /// retroactively annotated (i.e. the episode has ended). Used by
    /// `use_grpo_episode` advantage to gate "skip transitions whose
    /// episode hasn't ended yet" from "this transition's episode
    /// genuinely returned zero".
    pub episode_complete: bool,
}

/// Key for the novelty visit-count map.
/// Discretizes a latent vector into a grid cell.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateKey(Vec<i32>);

impl StateKey {
    /// Quantize a latent vector into grid cells with the given resolution.
    pub fn from_latent(latent: &[f32], resolution: f32) -> Self {
        Self(
            latent
                .iter()
                .map(|&x| (x / resolution).round() as i32)
                .collect(),
        )
    }
}

/// The agent's sole persistent memory.
pub struct ExperienceBuffer {
    transitions: RingBuffer<Transition>,
    pub visit_counts: HashMap<StateKey, u32>,
    grid_resolution: f32,
    /// When `visit_counts.len()` would exceed this on insert, the entire
    /// HashMap is cleared. 0 = unbounded (legacy behavior). Keeps the
    /// novelty-bonus working memory bounded under long runs / large
    /// batch_size where the latent grid space is effectively unbounded
    /// (rank-256 latents at grid_resolution 0.5 → every step is a new
    /// key, HashMap grows ~1 KB/step × n_lanes indefinitely).
    visit_counts_max: usize,
    /// When > 0, only the first `visit_count_dims` of each latent are
    /// quantized into the visit-count `StateKey`. The remaining dims
    /// are ignored. 0 = use all dims (legacy behavior).
    ///
    /// Motivation: at `latent_dim = 256` and `grid_resolution = 0.5`,
    /// the StateKey grid has ~3^256 cells — astronomically unbounded.
    /// Every observation maps to a unique cell, so `visit_count` is
    /// effectively constant 1 for any state the policy/planner ever
    /// reaches. The "novelty bonus" `1/sqrt(visit_count + 1)` is then
    /// uniform across candidate trajectories and the planner's
    /// trajectory-scoring degenerates to "pick a random one."
    ///
    /// Truncating to e.g. 8 dims makes the grid manageable (~3^8 = 6.6k
    /// cells), so revisits are common and the count is informative.
    /// This is a coarse trick — proper fix is contrastive or
    /// random-projection encoding — but it makes the existing
    /// novelty signal usable.
    ///
    /// Mutually exclusive with `visit_count_proj_dim` (when both are >0,
    /// projection takes priority since it's strictly more general).
    visit_count_dims: usize,
    /// When > 0, project the latent through a fixed random matrix
    /// before quantizing. Output dim = this value. Compared to
    /// `visit_count_dims` truncation, random projection preserves L2
    /// distance approximately (Johnson-Lindenstrauss) across the FULL
    /// 256-dim latent, so nearby latents in any subspace get nearby
    /// projections — not just nearby latents whose differences fall
    /// in the first N dims. Default 0 = disabled.
    visit_count_proj_dim: usize,
    /// Random projection matrix, shape `[latent_dim_observed, proj_dim]`,
    /// lazily initialized on first `push` once we know the latent_dim.
    /// Seeded deterministically (proj_seed) so behavior reproduces
    /// across runs of the same agent config.
    visit_count_proj_matrix: Option<Vec<f32>>,
    visit_count_proj_seed: u64,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize, grid_resolution: f32) -> Self {
        Self::with_visit_counts_max(capacity, grid_resolution, 0)
    }

    pub fn with_visit_counts_max(
        capacity: usize,
        grid_resolution: f32,
        visit_counts_max: usize,
    ) -> Self {
        Self::with_visit_count_dims(capacity, grid_resolution, visit_counts_max, 0)
    }

    pub fn with_visit_count_dims(
        capacity: usize,
        grid_resolution: f32,
        visit_counts_max: usize,
        visit_count_dims: usize,
    ) -> Self {
        Self::with_visit_count_config(
            capacity,
            grid_resolution,
            visit_counts_max,
            visit_count_dims,
            0,
            0x9E37_79B9_7F4A_7C15,
        )
    }

    /// Full constructor with random-projection config. `proj_dim = 0` =
    /// disabled (use truncation or full latent based on `visit_count_dims`).
    pub fn with_visit_count_config(
        capacity: usize,
        grid_resolution: f32,
        visit_counts_max: usize,
        visit_count_dims: usize,
        visit_count_proj_dim: usize,
        visit_count_proj_seed: u64,
    ) -> Self {
        Self {
            transitions: RingBuffer::new(capacity),
            visit_counts: HashMap::new(),
            grid_resolution,
            visit_counts_max,
            visit_count_dims,
            visit_count_proj_dim,
            visit_count_proj_matrix: None,
            visit_count_proj_seed,
        }
    }

    /// Lazy-initialize the random projection matrix on first use. Uses
    /// a deterministic seed + a Box-Muller-style normal sample so the
    /// matrix is reproducible across runs of the same agent config.
    fn ensure_proj_matrix(&mut self, latent_dim: usize) {
        if self.visit_count_proj_matrix.is_some() {
            return;
        }
        let proj_dim = self.visit_count_proj_dim;
        if proj_dim == 0 {
            return;
        }
        // Standard random projection: each entry ~ N(0, 1) / sqrt(latent_dim).
        // The 1/sqrt(d) scaling preserves expected L2 distance.
        let scale = 1.0 / (latent_dim as f32).sqrt();
        let mut state = self.visit_count_proj_seed.max(1);
        // Simple xorshift64 generator for determinism. Box-Muller pair.
        let mut next_u64 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut next_normal = || {
            let u1 = (next_u64() & 0x7FFFFFFF) as f32 / (i32::MAX as f32);
            let u2 = (next_u64() & 0x7FFFFFFF) as f32 / (i32::MAX as f32);
            // Box-Muller: r * cos(theta)
            let r = (-2.0 * (u1.max(1e-20)).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            r * theta.cos()
        };
        let mut mat = vec![0.0f32; latent_dim * proj_dim];
        for entry in mat.iter_mut() {
            *entry = next_normal() * scale;
        }
        self.visit_count_proj_matrix = Some(mat);
    }

    /// Compute the projection-or-truncation reduction of a latent into
    /// the buffer used for `StateKey` hashing.
    ///
    /// Priority: projection > truncation > pass-through. If
    /// `visit_count_proj_dim > 0`, runs latent @ proj_matrix (`proj_dim`
    /// outputs). Else if `visit_count_dims > 0`, truncates to first
    /// `visit_count_dims` dims. Else returns the latent as-is.
    fn count_latent_owned(&self, latent: &[f32]) -> Vec<f32> {
        if self.visit_count_proj_dim > 0 {
            // Project: out[j] = sum_i latent[i] * proj[i*proj_dim + j].
            let proj = self
                .visit_count_proj_matrix
                .as_ref()
                .expect("proj matrix must be initialized before count_latent_owned");
            let pd = self.visit_count_proj_dim;
            let ld = latent.len();
            let mut out = vec![0.0f32; pd];
            for j in 0..pd {
                let mut acc = 0.0f32;
                for i in 0..ld {
                    acc += latent[i] * proj[i * pd + j];
                }
                out[j] = acc;
            }
            out
        } else if self.visit_count_dims > 0 && self.visit_count_dims < latent.len() {
            latent[..self.visit_count_dims].to_vec()
        } else {
            latent.to_vec()
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Record a new transition and update the novelty visit count.
    pub fn push(&mut self, transition: Transition) {
        // Lazy-init the projection matrix once we know latent_dim.
        if self.visit_count_proj_dim > 0 && self.visit_count_proj_matrix.is_none() {
            self.ensure_proj_matrix(transition.latent.len());
        }
        let reduced = self.count_latent_owned(&transition.latent);
        let key = StateKey::from_latent(&reduced, self.grid_resolution);
        // Cap memory: clear before inserting if the bound is set and
        // adding a NEW key would exceed it. Existing-key updates always
        // proceed (just an integer ++). The full-clear strategy is
        // simpler than LRU/random eviction and adequate for
        // novelty-bonus, which is most useful on recently-seen states.
        if self.visit_counts_max > 0
            && !self.visit_counts.contains_key(&key)
            && self.visit_counts.len() >= self.visit_counts_max
        {
            self.visit_counts.clear();
        }
        *self.visit_counts.entry(key).or_insert(0) += 1;
        self.transitions.push(transition);
    }

    /// Get transition by logical index.
    pub fn get(&self, index: usize) -> &Transition {
        self.transitions.get(index)
    }

    /// Return the most recent transition.
    pub fn last(&self) -> Option<&Transition> {
        self.transitions.last()
    }

    /// Return the N most recent transitions.
    pub fn recent_window(&self, n: usize) -> Vec<&Transition> {
        self.transitions.recent(n)
    }

    /// Look up the visit count for a latent vector.
    pub fn visit_count(&self, latent: &[f32]) -> u32 {
        if self.visit_count_proj_dim > 0 && self.visit_count_proj_matrix.is_none() {
            // Matrix not yet initialized (no push() yet). Treat as
            // unvisited; the first push() will initialize and start
            // accumulating counts.
            return 0;
        }
        let reduced = self.count_latent_owned(latent);
        let key = StateKey::from_latent(&reduced, self.grid_resolution);
        self.visit_counts.get(&key).copied().unwrap_or(0)
    }

    /// Get a mutable reference to transition by logical index.
    pub fn get_mut(&mut self, index: usize) -> &mut Transition {
        self.transitions.get_mut(index)
    }

    /// Sample a batch for training: `replay_ratio` fraction from full history,
    /// the rest from the most recent window.
    pub fn sample_batch<R: Rng>(
        &self,
        rng: &mut R,
        batch_size: usize,
        replay_ratio: f32,
    ) -> Vec<&Transition> {
        if self.is_empty() {
            return Vec::new();
        }

        let replay_count = (batch_size as f32 * replay_ratio).round() as usize;
        let recent_count = batch_size - replay_count;

        let mut batch = Vec::with_capacity(batch_size);

        // Replay samples from full buffer history
        for _ in 0..replay_count.min(self.len()) {
            let idx = rng.random_range(0..self.len());
            batch.push(self.transitions.get(idx));
        }

        // Recent samples from the tail of the buffer
        let recent_start = self.len().saturating_sub(recent_count * 4);
        let recent_range = recent_start..self.len();
        if !recent_range.is_empty() {
            for _ in 0..recent_count {
                let idx = rng.random_range(recent_range.clone());
                batch.push(self.transitions.get(idx));
            }
        }

        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_and_get() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.get(0), 10);
        assert_eq!(*rb.get(2), 30);

        // Overwrite oldest
        rb.push(40);
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.get(0), 20);
        assert_eq!(*rb.get(2), 40);
    }

    #[test]
    fn ring_buffer_recent() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(5);
        for i in 0..5 {
            rb.push(i);
        }
        let recent: Vec<_> = rb.recent(3).into_iter().copied().collect();
        assert_eq!(recent, vec![2, 3, 4]);
    }

    #[test]
    fn ring_buffer_last() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        assert!(rb.last().is_none());
        rb.push(1);
        assert_eq!(*rb.last().unwrap(), 1);
        rb.push(2);
        assert_eq!(*rb.last().unwrap(), 2);
    }

    #[test]
    fn experience_buffer_visit_counts() {
        let mut buf = ExperienceBuffer::new(100, 0.1);
        let t = Transition {
            latent: vec![0.15, 0.25],
            ..Default::default()
        };
        buf.push(t.clone());
        buf.push(t.clone());

        assert_eq!(buf.visit_count(&[0.15, 0.25]), 2);
        assert_eq!(buf.visit_count(&[0.99, 0.99]), 0);
    }

    #[test]
    fn experience_buffer_sample_batch() {
        let mut buf = ExperienceBuffer::new(100, 0.1);
        for i in 0..50 {
            buf.push(Transition {
                reward: i as f32,
                ..Default::default()
            });
        }

        let mut rng = rand::rng();
        let batch = buf.sample_batch(&mut rng, 10, 0.2);
        assert_eq!(batch.len(), 10);
    }
}
