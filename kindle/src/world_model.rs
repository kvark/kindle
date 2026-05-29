//! World Model: forward dynamics predictor.
//!
//! Predicts the next latent state given the current latent and action:
//! `W(z_t, a_t) -> z_hat_{t+1}`
//!
//! Loss: `MSE(z_hat_{t+1}, stop_grad(z_{t+1}))`.
//! The target `z_{t+1}` is fed as a graph input (not parameter), which
//! prevents gradients from flowing through it — achieving stop-gradient.
//!
//! The prediction error also serves as the "surprise" component of the
//! reward circuit.

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// Forward dynamics world model.
///
/// The first layer is split into two projections (`z_proj` + `a_proj`)
/// whose outputs are summed. This is equivalent to `W @ [z; a]` but
/// avoids needing a concat op and — critically — keeps the encoder's
/// z_t on the loss path so it receives gradients.
///
/// `sigma_proj` is the optional GRAM-style heteroscedastic σ head: when
/// `Some`, the WM additionally predicts per-dim Gaussian σ at each
/// (z, a). σ is trained against `|z_target − z_hat|` (stop-grad on the
/// residual so the mean prediction is not biased) and consumed by the
/// planner to scale per-element rollout noise.
pub struct WorldModel {
    pub z_proj: nn::Linear,
    pub a_proj: nn::Linear,
    pub fc2: nn::Linear,
    pub fc_out: nn::Linear,
    pub sigma_proj: Option<nn::Linear>,
}

impl WorldModel {
    /// Build the world model parameters. Use [`Self::new_stochastic`] to
    /// also build the σ-prediction head (GRAM heteroscedastic variant).
    pub fn new(g: &mut Graph, latent_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self::new_inner(g, latent_dim, action_dim, hidden_dim, false)
    }

    /// Build with σ head wired up for GRAM-style stochastic rollouts.
    /// Adds one Linear `world_model.sigma_proj` of shape
    /// `[hidden_dim → latent_dim]` (~latent×hidden params).
    pub fn new_stochastic(g: &mut Graph, latent_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self::new_inner(g, latent_dim, action_dim, hidden_dim, true)
    }

    fn new_inner(
        g: &mut Graph,
        latent_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        stochastic: bool,
    ) -> Self {
        let sigma_proj = if stochastic {
            Some(nn::Linear::no_bias(
                g,
                "world_model.sigma_proj",
                hidden_dim,
                latent_dim,
            ))
        } else {
            None
        };
        Self {
            z_proj: nn::Linear::new(g, "world_model.z_proj", latent_dim, hidden_dim),
            a_proj: nn::Linear::no_bias(g, "world_model.a_proj", action_dim, hidden_dim),
            fc2: nn::Linear::new(g, "world_model.fc2", hidden_dim, hidden_dim),
            fc_out: nn::Linear::no_bias(g, "world_model.fc_out", hidden_dim, latent_dim),
            sigma_proj,
        }
    }

    /// Forward pass: `(z_t, action) -> z_hat_{t+1}`.
    ///
    /// `z_t`: `[batch, latent_dim]` — encoder output (on the gradient path).
    /// `action`: `[batch, action_dim]` — one-hot or continuous action vector.
    pub fn forward(&self, g: &mut Graph, z_t: NodeId, action: NodeId) -> NodeId {
        let (mu, _maybe_sigma) = self.forward_with_optional_sigma(g, z_t, action, false);
        mu
    }

    /// Forward + σ-prediction. When `sigma_proj` is None this is identical
    /// to [`Self::forward`] and the second slot in the returned tuple is
    /// a no-op zero scalar (callers should branch on `stochastic` rather
    /// than rely on the σ output being meaningful).
    pub fn forward_with_sigma(
        &self,
        g: &mut Graph,
        z_t: NodeId,
        action: NodeId,
    ) -> (NodeId, Option<NodeId>) {
        let (mu, sigma) = self.forward_with_optional_sigma(g, z_t, action, true);
        (mu, sigma)
    }

    fn forward_with_optional_sigma(
        &self,
        g: &mut Graph,
        z_t: NodeId,
        action: NodeId,
        want_sigma: bool,
    ) -> (NodeId, Option<NodeId>) {
        let h_z = self.z_proj.forward(g, z_t);
        let h_a = self.a_proj.forward(g, action);
        let h = g.add(h_z, h_a);
        let h = g.relu(h);
        let h2 = self.fc2.forward(g, h);
        let h2 = g.relu(h2);
        let mu = self.fc_out.forward(g, h2);
        let sigma = match (want_sigma, &self.sigma_proj) {
            (true, Some(sp)) => {
                // σ ∈ (0, 1) per element via sigmoid. Stop-grad on h2
                // before sigma_proj: σ-head training updates ONLY
                // sigma_proj.weight, never the shared WM trunk. Without
                // this, σ-loss gradients propagate back through fc2/
                // z_proj/a_proj and reshape the latent representation
                // toward "good for predicting σ" — observed to inflate
                // μ-loss by 20× in early-training LL smoke test.
                let h2_det = g.stop_gradient(h2);
                let raw = sp.forward(g, h2_det);
                Some(g.sigmoid(raw))
            }
            _ => None,
        };
        (mu, sigma)
    }

    /// Build the MSE loss against the target latent.
    ///
    /// `z_target` must be a graph input node (not derived from parameters)
    /// to achieve stop-gradient semantics.
    pub fn loss(g: &mut Graph, z_pred: NodeId, z_target: NodeId) -> NodeId {
        g.mse_loss(z_pred, z_target)
    }

    /// Heteroscedastic σ regression loss: σ-head trained to predict
    /// `|z_target − μ|` per element. Stop-grad on the residual so the
    /// σ-training signal does not perturb the mean head's optimum.
    /// Returns the σ regression term (already a scalar). Callers should
    /// scale + add to the main WM loss.
    pub fn sigma_loss(
        g: &mut Graph,
        mu: NodeId,
        sigma: NodeId,
        z_target: NodeId,
    ) -> NodeId {
        let neg_mu = g.neg(mu);
        let resid = g.add(z_target, neg_mu);
        let abs_resid = g.abs(resid);
        let target = g.stop_gradient(abs_resid);
        g.mse_loss(sigma, target)
    }

    /// k-step rollout: apply WM k times iteratively given a sequence
    /// of actions. Returns z_hat_k.
    ///
    /// `z_t`: [batch, latent_dim] — starting latent (graph input)
    /// `actions`: slice of k NodeIds, each [batch, action_dim]
    ///
    /// All NodeIds in `actions` must be stop_gradient'd inputs.
    pub fn rollout_k(&self, g: &mut Graph, z_t: NodeId, actions: &[NodeId]) -> NodeId {
        let mut z = z_t;
        for &a in actions {
            z = self.forward(g, z, a);
        }
        z
    }
}
