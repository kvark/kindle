//! Value head: predicts expected discounted return from a latent state.
//!
//! Trained on Monte-Carlo return-to-go computed CPU-side after each
//! episode completes. Used by the planner to score rollouts by predicted
//! V(z') at each step — the mechanism by which "many losses + few wins"
//! become a directional gradient in latent space:
//!
//! - Loss-trajectory latents see `R_t ≈ 0` across the episode, so V
//!   learns ≈ 0 across their region of latent space.
//! - Win-trajectory latents see `R_t = γ^(T-t) · reward_end`, with V
//!   learning the **distance-to-win** along the trajectory.
//! - The value head's MLP smoothly interpolates between these two
//!   regions, so the planner gets a gradient toward win-states from
//!   anywhere in the latent space — not just exact replay of past wins.

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

pub struct ValueHead {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
}

impl ValueHead {
    pub fn new(g: &mut Graph, latent_dim: usize, hidden_dim: usize) -> Self {
        Self {
            fc1: nn::Linear::new(g, "value_head.fc1", latent_dim, hidden_dim),
            fc2: nn::Linear::no_bias(g, "value_head.fc2", hidden_dim, 1),
        }
    }

    pub fn forward(&self, g: &mut Graph, z: NodeId) -> NodeId {
        let h = self.fc1.forward(g, z);
        let h = g.relu(h);
        self.fc2.forward(g, h)
    }
}
