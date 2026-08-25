use crate::vision::DinoObservation;

/// DreamerV3 scaling presets from the pinned upstream configuration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSize {
    /// CI and wiring preset only; not an upstream benchmark size.
    Tiny,
    Size1M,
    #[default]
    Size12M,
    Size25M,
    Size50M,
    Size100M,
    Size200M,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NetworkSize {
    pub deter: usize,
    pub hidden: usize,
    pub stoch: usize,
    pub classes: usize,
    pub blocks: usize,
    pub units: usize,
    pub vision_depth: usize,
}

impl ModelSize {
    pub const fn network(self) -> NetworkSize {
        match self {
            Self::Tiny => NetworkSize {
                deter: 32,
                hidden: 16,
                stoch: 4,
                classes: 4,
                blocks: 4,
                units: 32,
                vision_depth: 4,
            },
            Self::Size1M => NetworkSize {
                deter: 512,
                hidden: 64,
                stoch: 32,
                classes: 4,
                blocks: 8,
                units: 64,
                vision_depth: 4,
            },
            Self::Size12M => NetworkSize {
                deter: 2_048,
                hidden: 256,
                stoch: 32,
                classes: 16,
                blocks: 8,
                units: 256,
                vision_depth: 16,
            },
            Self::Size25M => NetworkSize {
                deter: 3_072,
                hidden: 384,
                stoch: 32,
                classes: 24,
                blocks: 8,
                units: 384,
                vision_depth: 24,
            },
            Self::Size50M => NetworkSize {
                deter: 4_096,
                hidden: 512,
                stoch: 32,
                classes: 32,
                blocks: 8,
                units: 512,
                vision_depth: 32,
            },
            Self::Size100M => NetworkSize {
                deter: 6_144,
                hidden: 768,
                stoch: 32,
                classes: 48,
                blocks: 8,
                units: 768,
                vision_depth: 48,
            },
            Self::Size200M => NetworkSize {
                deter: 8_192,
                hidden: 1_024,
                stoch: 32,
                classes: 64,
                blocks: 8,
                units: 1_024,
                vision_depth: 64,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LossScales {
    pub reconstruction: f32,
    pub reward: f32,
    pub continuation: f32,
    pub dynamics: f32,
    pub representation: f32,
    pub policy: f32,
    pub value: f32,
    pub replay_value: f32,
}

impl Default for LossScales {
    fn default() -> Self {
        Self {
            reconstruction: 1.0,
            reward: 1.0,
            continuation: 1.0,
            dynamics: 1.0,
            representation: 0.1,
            policy: 1.0,
            value: 1.0,
            replay_value: 0.3,
        }
    }
}

/// Configuration of the D3 baseline.
///
/// Defaults mirror the pinned DreamerV3 configuration where the stack can
/// express it directly. The default network is the upstream 12M preset: it
/// is large enough to be representative while keeping rapid local
/// experiments practical. Larger upstream presets remain one enum change.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DreamerConfig {
    pub action_count: usize,
    pub model_size: ModelSize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub batch_length: usize,
    /// Static truncated-BPTT chunk used by the Meganeura world graph. Replay
    /// and imagination still use the full D3 batch length.
    pub world_backprop_length: usize,
    pub replay_context: usize,
    pub train_ratio: f32,
    pub imagination_length: usize,
    pub horizon: usize,
    pub lambda: f32,
    /// Free-nat floor for the representation KL and, by default, dynamics KL.
    pub free_nats: f32,
    /// Optional dynamics-only floor. `None` preserves D3's shared floor;
    /// `Some(0.0)` continuously trains the prior while leaving the posterior's
    /// representation free-nat budget unchanged.
    #[serde(default)]
    pub dynamics_free_nats: Option<f32>,
    pub unimix: f32,
    pub value_bins: usize,
    pub actor_unimix: f32,
    pub actor_entropy: f32,
    pub slow_value_rate: f32,
    pub return_norm_rate: f32,
    pub learning_rate: f32,
    /// Optional actor/critic rate for optimization diagnostics. `None` keeps
    /// D3's shared world/behavior learning rate.
    #[serde(default)]
    pub behavior_learning_rate: Option<f32>,
    /// Number of completed learner updates before actor parameter updates are
    /// enabled. The behavior critic and actor optimizer moments continue
    /// training while the actor parameters are gated.
    #[serde(default)]
    pub actor_learning_starts: u64,
    pub learning_rate_warmup: u64,
    pub optimizer_beta1: f32,
    pub optimizer_beta2: f32,
    pub optimizer_epsilon: f32,
    /// DreamerV3's per-parameter adaptive-gradient clipping ratio.
    pub agc: f32,
    /// Lower bound for the parameter norm used by adaptive clipping.
    pub agc_pmin: f32,
    pub loss_scales: LossScales,
    /// Route D3's replay-value auxiliary loss into the world representation.
    /// The behavior critic is trained regardless of this switch.
    #[serde(default = "default_replay_value_gradient")]
    pub replay_value_gradient: bool,
    pub extrinsic_reward_scale: f32,
    pub intrinsic_reward_scale: f32,
    pub seed: u64,
    pub skip_full_optimize: bool,
}

impl DreamerConfig {
    pub fn new(action_count: usize) -> Self {
        let config = Self {
            action_count,
            model_size: ModelSize::Size12M,
            // Full DINO replay entries are intentionally compressed to a
            // fixed 7x7x64 map. 100k entries are ~1.25 GB before RSSM context.
            replay_capacity: 100_000,
            batch_size: 16,
            batch_length: 64,
            world_backprop_length: 8,
            replay_context: 1,
            train_ratio: 32.0,
            imagination_length: 15,
            horizon: 333,
            lambda: 0.95,
            free_nats: 1.0,
            dynamics_free_nats: None,
            unimix: 0.01,
            value_bins: 255,
            actor_unimix: 0.01,
            actor_entropy: 3e-4,
            slow_value_rate: 0.02,
            return_norm_rate: 0.01,
            learning_rate: 4e-5,
            behavior_learning_rate: None,
            actor_learning_starts: 0,
            learning_rate_warmup: 1_000,
            optimizer_beta1: 0.9,
            optimizer_beta2: 0.999,
            optimizer_epsilon: 1e-20,
            agc: 0.3,
            agc_pmin: 1e-3,
            loss_scales: LossScales::default(),
            replay_value_gradient: true,
            extrinsic_reward_scale: 1.0,
            intrinsic_reward_scale: 0.0,
            seed: 0,
            skip_full_optimize: false,
        };
        config.validate();
        config
    }

    pub fn tiny(action_count: usize) -> Self {
        Self {
            model_size: ModelSize::Tiny,
            replay_capacity: 1_024,
            batch_size: 2,
            batch_length: 4,
            world_backprop_length: 4,
            imagination_length: 3,
            value_bins: 15,
            learning_rate_warmup: 0,
            ..Self::new(action_count)
        }
    }

    pub fn network(&self) -> NetworkSize {
        self.model_size.network()
    }

    pub fn feature_dim(&self) -> usize {
        let size = self.network();
        size.deter + size.stoch * size.classes
    }

    pub const fn observation_dim(&self) -> usize {
        DinoObservation::LEN
    }

    pub fn behavior_learning_rate(&self) -> f32 {
        self.behavior_learning_rate.unwrap_or(self.learning_rate)
    }

    pub const fn actor_update_scale(&self, learner_step: u64) -> f32 {
        if learner_step >= self.actor_learning_starts {
            1.0
        } else {
            0.0
        }
    }

    pub fn dynamics_free_nats(&self) -> f32 {
        self.dynamics_free_nats.unwrap_or(self.free_nats)
    }

    /// Eligible replay items required before scheduled learning begins.
    pub fn replay_warmup_sequences(&self) -> usize {
        self.batch_size * self.batch_length
    }

    /// Frames needed to expose [`Self::replay_warmup_sequences`] complete
    /// context-plus-training sequences.
    pub fn replay_warmup_frames(&self) -> usize {
        self.replay_warmup_sequences() + self.replay_context + self.batch_length - 1
    }

    pub fn continuation_discount(&self) -> f32 {
        1.0 - 1.0 / self.horizon as f32
    }

    pub fn validate(&self) {
        if let Err(error) = self.check() {
            panic!("invalid Dreamer configuration: {error}");
        }
    }

    pub fn check(&self) -> Result<(), String> {
        let size = self.network();
        if self.action_count <= 1 {
            return Err("action_count must be greater than one".into());
        }
        if size.deter == 0 || !size.deter.is_multiple_of(size.blocks) {
            return Err("deter must be non-zero and divisible by blocks".into());
        }
        if size.hidden == 0 || size.stoch == 0 || size.classes <= 1 {
            return Err("hidden/stoch/classes network dimensions are invalid".into());
        }
        if size.units == 0 || size.vision_depth == 0 {
            return Err("head and vision dimensions must be non-zero".into());
        }
        if self.replay_context != 1 {
            return Err("the baseline pins D3's one-step replay context".into());
        }
        if self.batch_size == 0 || self.batch_length <= 1 {
            return Err("batch_size must be positive and batch_length must exceed one".into());
        }
        if self.world_backprop_length == 0
            || self.world_backprop_length > self.batch_length
            || !self.batch_length.is_multiple_of(self.world_backprop_length)
        {
            return Err("world_backprop_length must be a positive divisor of batch_length".into());
        }
        if self.replay_capacity < self.replay_warmup_frames() {
            return Err("replay_capacity cannot satisfy the D3 learner warmup gate".into());
        }
        if !self.train_ratio.is_finite() || self.train_ratio <= 0.0 {
            return Err("train_ratio must be finite and positive".into());
        }
        if self.imagination_length == 0 || self.horizon <= 1 {
            return Err("imagination_length must be positive and horizon must exceed one".into());
        }
        if !(0.0..=1.0).contains(&self.lambda) {
            return Err("lambda must be in [0, 1]".into());
        }
        if !self.free_nats.is_finite() || self.free_nats < 0.0 {
            return Err("free_nats must be finite and non-negative".into());
        }
        if self
            .dynamics_free_nats
            .is_some_and(|free_nats| !free_nats.is_finite() || free_nats < 0.0)
        {
            return Err("dynamics_free_nats must be finite and non-negative".into());
        }
        if !(0.0..1.0).contains(&self.unimix) {
            return Err("RSSM unimix must be in [0, 1)".into());
        }
        if self.value_bins < 3 || self.value_bins.is_multiple_of(2) {
            return Err("value_bins must be odd and at least three".into());
        }
        if !(0.0..1.0).contains(&self.actor_unimix) {
            return Err("actor_unimix must be in [0, 1)".into());
        }
        if !self.actor_entropy.is_finite() || self.actor_entropy < 0.0 {
            return Err("actor_entropy must be finite and non-negative".into());
        }
        if !(0.0..0.5).contains(&self.slow_value_rate) {
            return Err("slow_value_rate must be in [0, 0.5)".into());
        }
        if !(0.0..=1.0).contains(&self.return_norm_rate) {
            return Err("return_norm_rate must be in [0, 1]".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive".into());
        }
        if self
            .behavior_learning_rate
            .is_some_and(|rate| !rate.is_finite() || rate <= 0.0)
        {
            return Err("behavior_learning_rate must be finite and positive".into());
        }
        if !self.optimizer_beta1.is_finite()
            || !self.optimizer_beta2.is_finite()
            || !(0.0..1.0).contains(&self.optimizer_beta1)
            || !(0.0..1.0).contains(&self.optimizer_beta2)
        {
            return Err("optimizer beta values must be finite and in [0, 1)".into());
        }
        if !self.optimizer_epsilon.is_finite() || self.optimizer_epsilon <= 0.0 {
            return Err("optimizer_epsilon must be finite and positive".into());
        }
        if !self.agc.is_finite() || self.agc < 0.0 {
            return Err("agc must be finite and non-negative".into());
        }
        if !self.agc_pmin.is_finite() || self.agc_pmin < 0.0 {
            return Err("agc_pmin must be finite and non-negative".into());
        }
        let loss_scales = [
            self.loss_scales.reconstruction,
            self.loss_scales.reward,
            self.loss_scales.continuation,
            self.loss_scales.dynamics,
            self.loss_scales.representation,
            self.loss_scales.policy,
            self.loss_scales.value,
            self.loss_scales.replay_value,
        ];
        if loss_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err("loss scales must be finite and non-negative".into());
        }
        if !self.extrinsic_reward_scale.is_finite() || !self.intrinsic_reward_scale.is_finite() {
            return Err("reward scales must be finite".into());
        }
        Ok(())
    }
}

const fn default_replay_value_gradient() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_size_presets_are_pinned() {
        let one = ModelSize::Size1M.network();
        assert_eq!(
            (one.deter, one.hidden, one.stoch, one.classes),
            (512, 64, 32, 4)
        );
        let twelve = ModelSize::Size12M.network();
        assert_eq!(
            (twelve.deter, twelve.hidden, twelve.classes),
            (2_048, 256, 16)
        );
        let two_hundred = ModelSize::Size200M.network();
        assert_eq!(
            (two_hundred.deter, two_hundred.hidden, two_hundred.classes),
            (8_192, 1_024, 64)
        );
    }

    #[test]
    fn d3_data_defaults_are_explicit() {
        let config = DreamerConfig::new(18);
        assert_eq!((config.batch_size, config.batch_length), (16, 64));
        assert_eq!(config.world_backprop_length, 8);
        assert_eq!(config.imagination_length, 15);
        assert_eq!(config.value_bins, 255);
        assert_eq!(config.replay_context, 1);
        assert_eq!(config.replay_warmup_sequences(), 1_024);
        assert_eq!(config.replay_warmup_frames(), 1_088);
        assert_eq!(config.train_ratio, 32.0);
        assert_eq!(config.actor_unimix, 0.01);
        assert_eq!(config.dynamics_free_nats, None);
        assert_eq!(config.dynamics_free_nats(), config.free_nats);
        assert_eq!(config.behavior_learning_rate, None);
        assert_eq!(config.behavior_learning_rate(), config.learning_rate);
        assert_eq!(config.actor_learning_starts, 0);
        assert_eq!(config.actor_update_scale(0), 1.0);
        assert!(config.replay_value_gradient);
    }

    #[test]
    fn invalid_loss_scales_are_rejected() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.reconstruction = f32::NAN;
        assert_eq!(
            config.check().unwrap_err(),
            "loss scales must be finite and non-negative"
        );
        config.loss_scales.reconstruction = -1.0;
        assert!(config.check().is_err());
    }

    #[test]
    fn dynamics_floor_can_override_the_d3_shared_default() {
        let mut config = DreamerConfig::tiny(3);
        config.free_nats = 1.0;
        assert_eq!(config.dynamics_free_nats(), 1.0);
        config.dynamics_free_nats = Some(0.0);
        assert_eq!(config.dynamics_free_nats(), 0.0);
        assert!(config.check().is_ok());
        config.dynamics_free_nats = Some(-1.0);
        assert_eq!(
            config.check().unwrap_err(),
            "dynamics_free_nats must be finite and non-negative"
        );
    }

    #[test]
    fn actor_learning_gate_has_an_explicit_update_boundary() {
        let mut config = DreamerConfig::tiny(3);
        config.actor_learning_starts = 1_300;
        assert_eq!(config.actor_update_scale(0), 0.0);
        assert_eq!(config.actor_update_scale(1_299), 0.0);
        assert_eq!(config.actor_update_scale(1_300), 1.0);
    }

    #[test]
    fn older_configs_default_new_diagnostic_switches() {
        let mut value = serde_json::to_value(DreamerConfig::tiny(3)).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("replay_value_gradient");
        object.remove("behavior_learning_rate");
        object.remove("dynamics_free_nats");
        object.remove("actor_learning_starts");
        let restored: DreamerConfig = serde_json::from_value(value).unwrap();
        assert!(restored.replay_value_gradient);
        assert_eq!(restored.behavior_learning_rate, None);
        assert_eq!(restored.behavior_learning_rate(), restored.learning_rate);
        assert_eq!(restored.dynamics_free_nats, None);
        assert_eq!(restored.dynamics_free_nats(), restored.free_nats);
        assert_eq!(restored.actor_learning_starts, 0);
        assert_eq!(restored.actor_update_scale(0), 1.0);
    }
}
