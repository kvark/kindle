use crate::vision::{OBSERVATION_CHANNELS, OBSERVATION_GRID};

/// Values retained in replay and re-encoded by the current world model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    #[default]
    Features,
    /// One GPU resize to 64x64, channel-major RGB in [-0.5, 0.5].
    Rgb64,
    /// One GPU resize to 48x48, channel-major RGB in [-0.5, 0.5].
    Rgb48,
    /// One GPU resize to 32x32, channel-major RGB in [-0.5, 0.5].
    Rgb32,
    /// One GPU resize to 96x96, channel-major RGB in [-0.5, 0.5].
    Rgb96,
}

impl ObservationKind {
    pub const fn is_rgb(self) -> bool {
        !matches!(self, Self::Features)
    }
}

/// Native-detail, phase-aligned causal replay. Both modes re-encode pixels;
/// only Joint allows world/task gradients into the pretrained Tiny weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoEncoder {
    Frozen,
    Joint,
}

/// Bootstrap used by the real-replay value target; imagined learning is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayValueBootstrap {
    #[default]
    Imagined,
    Posterior,
}

impl ReplayValueBootstrap {
    pub(crate) fn values<'a>(self, posterior: &'a [f32], imagined: &'a [f32]) -> &'a [f32] {
        assert_eq!(posterior.len(), imagined.len());
        match self {
            Self::Imagined => imagined,
            Self::Posterior => posterior,
        }
    }
}

/// Matrix-product policy for training graphs. Parameters, optimizer state,
/// reductions and accumulators remain f32; F16 permits reduced input precision
/// in both forward and backward products, not half-storage checkpoints.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingPrecision {
    #[default]
    Auto,
    F32,
    F16,
    /// BF16 dense regions plus convolution operands and stored forward results.
    /// Convolution derivatives and grouped products retain F32 arithmetic.
    Bf16,
    /// BF16 learner inference/behavior; F16-input, F32-storage world gradients.
    Mixed,
}

impl TrainingPrecision {
    pub(crate) fn coop_policy(self) -> meganeura::CoopPolicy {
        match self {
            Self::Auto => meganeura::CoopPolicy::Auto,
            Self::F32 => meganeura::CoopPolicy::NativeF32,
            Self::F16 => meganeura::CoopPolicy::AllowF16,
            Self::Bf16 | Self::Mixed => meganeura::CoopPolicy::NativeF32,
        }
    }

    pub(crate) fn uses_bf16(self) -> bool {
        matches!(self, Self::Bf16 | Self::Mixed)
    }
}

/// Model scaling presets, including the pinned DreamerV3 configurations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSize {
    /// CI and wiring preset only; not an upstream benchmark size.
    Tiny,
    Size1M,
    /// Intermediate Kindle development preset; not an upstream benchmark size.
    Size4M,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LatentShape {
    pub variables: usize,
    pub classes: usize,
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
            Self::Size4M => NetworkSize {
                deter: 1_024,
                hidden: 128,
                stoch: 32,
                classes: 8,
                blocks: 8,
                units: 128,
                vision_depth: 8,
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
    /// Predict features from the deterministic prior. Feature observations use
    /// squared error; RGB uses CDP cosine distance to detached CNN embeddings.
    #[serde(default)]
    pub future_prediction: f32,
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
            future_prediction: 0.0,
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

/// Fixed statistics for predicting each visual feature in standardized units.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FeatureStandardization {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

/// Training-only position targets derived from a color within a fixed image
/// region. The region is [x0, y0, x1, y1] in full-image normalized coordinates.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ColorCentroidSupervision {
    pub scale: f32,
    pub channel: usize,
    pub background_scale: f32,
    pub region: [f32; 4],
}

/// Optional supervised actor targets from explicitly trusted executed actions.
/// Weights are applied before averaging over all replay rows, not renormalized
/// within each minibatch; rare actions therefore retain their relative weight.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExpertImitation {
    pub scale: f32,
    pub action_weights: Vec<f32>,
    /// Route the same trusted targets through the frozen actor into world state.
    #[serde(default, skip_serializing_if = "is_false")]
    pub world_gradient: bool,
    /// Hold the initial scale through start_update, then linearly reach zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay: Option<ExpertImitationDecay>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExpertImitationDecay {
    pub start_update: u64,
    pub end_update: u64,
    /// Fade actor imitation while retaining the base world-imitation scale.
    #[serde(default, skip_serializing_if = "is_false")]
    pub actor_only: bool,
}

impl ExpertImitation {
    /// `update` is the one-based LearnReport counter of the update being run.
    pub fn scale_at_update(&self, update: u64) -> f32 {
        match &self.decay {
            None => self.scale,
            Some(decay) if update <= decay.start_update => self.scale,
            Some(decay) if update >= decay.end_update => 0.0,
            Some(decay) => {
                let remaining = (decay.end_update - update) as f64;
                let duration = (decay.end_update - decay.start_update) as f64;
                self.scale * (remaining / duration) as f32
            }
        }
    }

    pub fn world_scale_at_update(&self, update: u64) -> f32 {
        if self.decay.as_ref().is_some_and(|decay| decay.actor_only) {
            self.scale
        } else {
            self.scale_at_update(update)
        }
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

/// Prior-only predictions from the first posterior state of each training chunk.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct OpenLoopSupervision {
    pub horizon: usize,
    pub scale: f32,
}

/// Configuration of the D3 baseline. Defaults use the upstream 12M preset;
/// the pinned upstream default is 200M. Larger presets remain one enum change.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DreamerConfig {
    pub action_count: usize,
    /// Standalone world: fixed submitted-action features, with unchanged input width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_action_features: Option<Vec<Vec<f32>>>,
    pub model_size: ModelSize,
    /// Regroup categorical state without changing the preset's flattened width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent_shape: Option<LatentShape>,
    #[serde(default)]
    pub training_precision: TrainingPrecision,
    #[serde(default)]
    pub observation_kind: ObservationKind,
    /// CDP input is the previous and current resized RGB image, oldest first.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_frame_pair: bool,
    /// Standalone RGB world: encode previous/current images; replay and targets stay RGB.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_previous_frame: bool,
    /// Standalone RGB world: decode from RSSM state and two causal RGB images.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_spatial_decoder: bool,
    /// Standalone spatial world: jointly transform recurrent and image-history features.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_spatial_fusion: bool,
    /// Standalone fused world: add a learned correction to the latest image's log odds.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_spatial_residual: bool,
    /// Standalone fused world: mix learned horizontal/vertical RGB transport with generation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_spatial_transport: bool,
    /// Standalone fused world: wider local image history with incoming action planes.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_spatial_action: bool,
    /// Standalone local-action world: append commands after image motion extraction.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_late_action: bool,
    /// Standalone local-action world: widen the motion and fusion path to32 channels.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_wide_motion: bool,
    /// Standalone world: read shared-CNN features of decoded RGB in event heads.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_event_heads: bool,
    /// Standalone RGB-event world: Brier loss on cumulative imagined survival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_survival_scale: Option<f32>,
    /// Standalone spatial world: encode generated images into the imagined belief.
    #[serde(default, skip_serializing_if = "is_false")]
    pub world_image_feedback: bool,
    /// Standalone spatial world: snap generated frames to the nearest of these
    /// centered RGB colours before they re-enter the image history; the forward
    /// value is discrete, the gradient passes straight through.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb_history_palette: Option<Vec<[f32; 3]>>,
    /// Append fixed x/y pixel-center coordinates inside the learned RGB encoder.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_coordinates: bool,
    /// Standalone RGB world: image-only posterior and stochastic-only decoder.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_latent_only: bool,
    /// Standalone RGB world: diagonal Gaussian state with reparameterized samples.
    #[serde(default, skip_serializing_if = "is_false")]
    pub world_gaussian: bool,
    /// Standalone world: feed categorical probabilities directly as latent state.
    #[serde(default, skip_serializing_if = "is_false")]
    pub world_mean_latents: bool,
    /// Optional compact standalone Gaussian state; otherwise retain preset width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_gaussian_dim: Option<usize>,
    /// Downsample the learned RGB encoder with stride-two convolutions, without max-pooling.
    #[serde(default, skip_serializing_if = "is_false")]
    pub rgb_stride2: bool,
    /// Optional native-frame rectangle for the learned-RGB GPU preprocessor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_crop: Option<crate::vision::preprocess_gpu::FrameCrop>,
    #[serde(default)]
    pub video_encoder: Option<VideoEncoder>,
    /// Fixed, training-only statistics for future-prediction targets. Does not
    /// transform encoder/RSSM inputs. Diagnostics decode back to raw features.
    #[serde(default)]
    pub future_target_standardization: Option<FeatureStandardization>,
    /// CDP ablation: center predictions and detached targets using the current
    /// full replay batch's target mean. Acting/forecast outputs stay raw.
    #[serde(default)]
    pub cdp_centered: bool,
    /// Optional feature targets for the full-posterior reconstruction head.
    /// Separate from future targets; RGB reconstruction never uses these.
    #[serde(default)]
    pub reconstruction_target_standardization: Option<FeatureStandardization>,
    /// Per-patch hidden width immediately before the 64-channel feature decoder
    /// output. Fresh configs use 64 to avoid a hard affine rank bottleneck.
    /// Zero preserves the preset vision depth for legacy checkpoints.
    #[serde(default)]
    pub observation_decoder_depth: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub batch_length: usize,
    /// Static truncated-BPTT chunk used by the Meganeura world graph. Replay
    /// and imagination still use the full D3 batch length.
    pub world_backprop_length: usize,
    /// Optional row microbatch for the world-model graph. Gradients are
    /// averaged across equal microbatches before one optimizer step, so this
    /// changes peak activation memory without changing the effective D3 batch.
    #[serde(default)]
    pub world_microbatch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_loop_supervision: Option<OpenLoopSupervision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_centroid_supervision: Option<ColorCentroidSupervision>,
    /// Standalone-world loss attention derived from replay RGB. Reuses the
    /// color/region settings; scale controls attention, without a position head.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb_color_attention: Option<ColorCentroidSupervision>,
    /// One cached state plus optional forward-only observations before the
    /// training sequence. The default of 1 performs no state recomputation.
    pub replay_context: usize,
    /// After fresh sequences, draw this fraction from sequences containing a
    /// true terminal in the training rows. This deliberately changes the replay
    /// distribution for all objectives, without importance correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_replay_fraction: Option<f32>,
    /// Multiply only true-terminal posterior continuation losses, keeping the
    /// ordinary row denominator. `None` and `Some(1.0)` keep the legacy graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_continuation_weight: Option<f32>,
    /// Auxiliary continuation BCE on sampled one-step priors; reset rows are masked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_continuation_scale: Option<f32>,
    /// Auxiliary reward CE on sampled one-step priors; reset rows are masked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_reward_scale: Option<f32>,
    /// Replay samples per action. Zero disables scheduled updates; explicit
    /// `DreamerCore::learn` remains available for isolated diagnostics.
    pub train_ratio: f32,
    pub imagination_length: usize,
    pub horizon: usize,
    pub lambda: f32,
    #[serde(default)]
    pub replay_value_bootstrap: ReplayValueBootstrap,
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
    /// Override only the actor MLP hidden width; world and critic keep the preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_hidden_units: Option<usize>,
    /// Learn an independent linear logit correction from imagined policy losses;
    /// the base actor learns only from expert imitation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub actor_residual: bool,
    /// Auxiliary normalized lambda-return gradient through frozen imagined dynamics.
    /// Categorical draws use straight-through derivatives; disagreement and
    /// trajectory weights remain detached. Added to the existing actor objective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_dynamics_scale: Option<f32>,
    /// Replace sampled-action policy gradients with a detached one-step
    /// action-value expectation at replay posterior starts.
    #[serde(default, skip_serializing_if = "is_false")]
    pub actor_expected_value: bool,
    pub slow_value_rate: f32,
    pub return_norm_rate: f32,
    pub learning_rate: f32,
    /// Optional CNN/feature-adapter rate; defaults to the world learning rate.
    #[serde(default)]
    pub encoder_learning_rate: Option<f32>,
    /// Optional RSSM (including posterior) and future-predictor learning rate.
    #[serde(default)]
    pub dynamics_learning_rate: Option<f32>,
    /// Optional actor/critic rate for optimization diagnostics. `None` keeps
    /// D3's shared world/behavior learning rate.
    #[serde(default)]
    pub behavior_learning_rate: Option<f32>,
    /// Number of completed learner updates before actor parameter updates are
    /// enabled. The behavior critic and actor optimizer moments continue
    /// training while the actor parameters are gated.
    #[serde(default)]
    pub actor_learning_starts: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_imitation: Option<ExpertImitation>,
    /// Positive-advantage actor imitation of completed observed episode returns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_actor_scale: Option<f32>,
    /// Prefer completed observed game returns to bootstrapped replay-value targets.
    #[serde(default, skip_serializing_if = "is_false")]
    pub replay_value_completed: bool,
    /// Use observed completed returns as a conservative lower bound when the
    /// critic also includes nonnegative model-derived disagreement rewards.
    #[serde(default, skip_serializing_if = "is_false")]
    pub replay_actor_observed_lower_bound: bool,
    /// Allow caller-supplied boundaries for the auxiliary observed-return actor.
    /// These do not end game episodes or change model/critic continuation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub replay_actor_auxiliary_boundaries: bool,
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
    /// Dreamer's optional `ac_grads`: actor and imagined-value losses shape
    /// their initial posterior states. Future imagined states remain detached.
    #[serde(default)]
    pub actor_critic_gradient: bool,
    pub extrinsic_reward_scale: f32,
    pub intrinsic_reward_scale: f32,
    /// Recompute disagreement about action effects on observation encodings in
    /// imagination and replay. Zero intrinsic scale removes the ensemble.
    #[serde(default)]
    pub disagreement_bonus: bool,
    /// Add bounded fixed-feature visitation novelty to the intrinsic channel.
    /// Counts persist across episode boundaries and model checkpoints.
    #[serde(default)]
    pub visitation_bonus: bool,
    pub seed: u64,
    pub skip_full_optimize: bool,
}

impl DreamerConfig {
    pub fn new(action_count: usize) -> Self {
        let config = Self {
            action_count,
            world_action_features: None,
            model_size: ModelSize::Size12M,
            latent_shape: None,
            training_precision: TrainingPrecision::Auto,
            observation_kind: ObservationKind::Features,
            rgb_frame_pair: false,
            rgb_previous_frame: false,
            rgb_spatial_decoder: false,
            rgb_spatial_fusion: false,
            rgb_spatial_residual: false,
            rgb_spatial_transport: false,
            rgb_spatial_action: false,
            rgb_late_action: false,
            rgb_wide_motion: false,
            rgb_event_heads: false,
            world_survival_scale: None,
            world_image_feedback: false,
            rgb_history_palette: None,
            rgb_coordinates: false,
            rgb_latent_only: false,
            world_gaussian: false,
            world_mean_latents: false,
            world_gaussian_dim: None,
            rgb_stride2: false,
            observation_crop: None,
            video_encoder: None,
            future_target_standardization: None,
            cdp_centered: false,
            reconstruction_target_standardization: None,
            observation_decoder_depth: OBSERVATION_CHANNELS,
            // Full visual replay entries are intentionally compressed to a
            // fixed 7x7x64 map. 100k entries are ~1.25 GB before RSSM context.
            replay_capacity: 100_000,
            batch_size: 16,
            batch_length: 64,
            world_backprop_length: 8,
            world_microbatch_size: None,
            open_loop_supervision: None,
            color_centroid_supervision: None,
            rgb_color_attention: None,
            replay_context: 1,
            terminal_replay_fraction: None,
            terminal_continuation_weight: None,
            prior_continuation_scale: None,
            prior_reward_scale: None,
            train_ratio: 32.0,
            imagination_length: 15,
            horizon: 333,
            lambda: 0.95,
            replay_value_bootstrap: ReplayValueBootstrap::Imagined,
            free_nats: 1.0,
            dynamics_free_nats: None,
            unimix: 0.01,
            value_bins: 255,
            // Upstream's discrete Head.categorical constructs Categorical
            // without unimix. Only the RSSM's OneHot uses the 1% mixture.
            actor_unimix: 0.0,
            actor_entropy: 3e-4,
            actor_dynamics_scale: None,
            actor_expected_value: false,
            actor_hidden_units: None,
            actor_residual: false,
            slow_value_rate: 0.02,
            return_norm_rate: 0.01,
            learning_rate: 4e-5,
            encoder_learning_rate: None,
            dynamics_learning_rate: None,
            behavior_learning_rate: None,
            actor_learning_starts: 0,
            expert_imitation: None,
            replay_actor_scale: None,
            replay_value_completed: false,
            replay_actor_observed_lower_bound: false,
            replay_actor_auxiliary_boundaries: false,
            learning_rate_warmup: 1_000,
            optimizer_beta1: 0.9,
            optimizer_beta2: 0.999,
            optimizer_epsilon: 1e-20,
            agc: 0.3,
            agc_pmin: 1e-3,
            loss_scales: LossScales::default(),
            replay_value_gradient: true,
            actor_critic_gradient: false,
            extrinsic_reward_scale: 1.0,
            intrinsic_reward_scale: 0.0,
            disagreement_bonus: false,
            visitation_bonus: false,
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
        let mut size = self.model_size.network();
        if let Some(shape) = self.latent_shape {
            size.stoch = shape.variables;
            size.classes = shape.classes;
        }
        if let Some(dim) = self.world_gaussian_dim {
            size.stoch = dim;
            size.classes = 1;
        }
        size
    }

    pub fn uses_disagreement(&self) -> bool {
        self.disagreement_bonus && self.intrinsic_reward_scale > 0.0
    }

    pub fn feature_dim(&self) -> usize {
        let size = self.network();
        size.deter + size.stoch * size.classes
    }

    pub(crate) fn latent_parameter_classes(&self) -> usize {
        self.network().classes * if self.world_gaussian { 2 } else { 1 }
    }

    pub const fn observation_dim(&self) -> usize {
        self.observation_grid() * self.observation_grid() * self.observation_channels()
    }

    pub const fn observation_grid(&self) -> usize {
        match self.observation_kind {
            ObservationKind::Features => OBSERVATION_GRID,
            ObservationKind::Rgb64 => 64,
            ObservationKind::Rgb48 => 48,
            ObservationKind::Rgb32 => 32,
            ObservationKind::Rgb96 => 96,
        }
    }

    pub const fn observation_channels(&self) -> usize {
        match self.observation_kind {
            ObservationKind::Features => OBSERVATION_CHANNELS,
            ObservationKind::Rgb64
            | ObservationKind::Rgb48
            | ObservationKind::Rgb32
            | ObservationKind::Rgb96 => {
                if self.rgb_frame_pair {
                    6
                } else {
                    3
                }
            }
        }
    }

    pub(crate) const fn observation_shape(&self, batch: usize) -> [usize; 2] {
        match self.observation_kind {
            ObservationKind::Features => [
                batch * OBSERVATION_GRID * OBSERVATION_GRID,
                OBSERVATION_CHANNELS,
            ],
            ObservationKind::Rgb64
            | ObservationKind::Rgb48
            | ObservationKind::Rgb32
            | ObservationKind::Rgb96 => [batch, self.observation_dim()],
        }
    }

    pub(crate) fn encoded_observation_dim(&self) -> usize {
        match self.observation_kind {
            ObservationKind::Features => {
                OBSERVATION_GRID * OBSERVATION_GRID * self.network().vision_depth
            }
            ObservationKind::Rgb64
            | ObservationKind::Rgb48
            | ObservationKind::Rgb32
            | ObservationKind::Rgb96 => {
                let side = self.observation_grid() / 16;
                side * side * self.network().vision_depth * 4
            }
        }
    }

    pub fn observation_decoder_depth(&self) -> usize {
        if self.observation_decoder_depth == 0 {
            self.network().vision_depth
        } else {
            self.observation_decoder_depth
        }
    }

    pub fn behavior_learning_rate(&self) -> f32 {
        self.behavior_learning_rate.unwrap_or(self.learning_rate)
    }

    pub fn is_cdp(&self) -> bool {
        self.observation_kind.is_rgb() && self.loss_scales.future_prediction > 0.0
    }

    /// CDP forecasts CNN embeddings, not RGB images or reconstructed features.
    pub fn prediction_dim(&self) -> usize {
        if self.is_cdp() {
            self.encoded_observation_dim()
        } else {
            self.observation_dim()
        }
    }

    pub fn world_microbatch_size(&self) -> usize {
        self.world_microbatch_size.unwrap_or(self.batch_size)
    }

    pub fn actor_hidden_units(&self) -> usize {
        self.actor_hidden_units.unwrap_or(self.network().units)
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
        if self.video_encoder.is_some() {
            self.batch_size
        } else {
            self.batch_size * self.batch_length
        }
    }

    /// Frames needed to expose [`Self::replay_warmup_sequences`] complete
    /// context-plus-training sequences.
    pub fn replay_warmup_frames(&self) -> usize {
        let stride = if self.video_encoder.is_some() {
            self.batch_length
        } else {
            1
        };
        self.replay_warmup_sequences() * stride + self.replay_context + self.batch_length - 1
    }

    pub fn continuation_discount(&self) -> f32 {
        1.0 - 1.0 / self.horizon as f32
    }

    pub fn expert_world_gradient(&self) -> bool {
        self.expert_imitation
            .as_ref()
            .is_some_and(|expert| expert.world_gradient)
    }

    pub fn validate(&self) {
        if let Err(error) = self.check() {
            panic!("invalid Dreamer configuration: {error}");
        }
    }

    pub fn check(&self) -> Result<(), String> {
        if let Some(scale) = self.world_survival_scale
            && (!scale.is_finite() || scale <= 0.0 || !self.rgb_event_heads)
        {
            return Err(
                "world survival scale requires positive finite scale and RGB event heads".into(),
            );
        }
        if self.rgb_event_heads
            && (self.model_size != ModelSize::Size1M
                || self.observation_kind != ObservationKind::Rgb64
                || !self.rgb_spatial_action
                || !self.rgb_spatial_fusion
                || !self.rgb_spatial_decoder
                || self.rgb_wide_motion
                || self.rgb_late_action
                || self.rgb_spatial_transport
                || self.rgb_spatial_residual
                || self.world_image_feedback
                || self.world_gaussian
                || self.world_mean_latents
                || self.rgb_previous_frame
                || self.rgb_frame_pair
                || self.rgb_latent_only
                || self.rgb_coordinates
                || self.rgb_stride2
                || self.rgb_color_attention.is_some()
                || self.color_centroid_supervision.is_some())
        {
            return Err(
                "RGB event heads require the plain 1M RGB64 local-action categorical fused world"
                    .into(),
            );
        }
        if self.rgb_wide_motion
            && (!self.rgb_spatial_action
                || !self.rgb_spatial_fusion
                || !self.rgb_spatial_decoder
                || self.rgb_late_action
                || self.rgb_spatial_transport
                || self.rgb_spatial_residual
                || self.world_image_feedback
                || self.world_gaussian
                || self.world_mean_latents
                || self.rgb_previous_frame
                || self.rgb_latent_only
                || self.rgb_color_attention.is_some()
                || self.color_centroid_supervision.is_some())
        {
            return Err(
                "wide motion requires a plain local-action categorical fused decoder".into(),
            );
        }
        if self.rgb_late_action
            && (!self.rgb_spatial_action
                || self.rgb_spatial_transport
                || self.rgb_spatial_residual
                || self.world_image_feedback
                || self.world_gaussian
                || self.world_mean_latents
                || self.rgb_previous_frame
                || self.rgb_latent_only
                || self.rgb_color_attention.is_some()
                || self.color_centroid_supervision.is_some())
        {
            return Err(
                "late action requires a plain local-action categorical fused decoder".into(),
            );
        }
        if self.rgb_spatial_action && (!self.rgb_spatial_fusion || self.rgb_spatial_residual) {
            return Err("spatial action requires fusion without residual".into());
        }
        if self.rgb_spatial_transport && (!self.rgb_spatial_fusion || self.rgb_spatial_residual) {
            return Err("spatial transport requires fusion without spatial residual".into());
        }
        if self.rgb_spatial_residual && !self.rgb_spatial_fusion {
            return Err("spatial residual requires the standalone fused RGB decoder".into());
        }
        if self.rgb_spatial_fusion && (!self.rgb_spatial_decoder || self.world_image_feedback) {
            return Err(
                "spatial fusion requires the standalone spatial RGB decoder without image feedback"
                    .into(),
            );
        }
        if self.world_image_feedback && !self.rgb_spatial_decoder {
            return Err("image feedback requires the standalone spatial RGB decoder".into());
        }
        if let Some(palette) = &self.rgb_history_palette
            && (!self.rgb_spatial_decoder
                || !(2..=32).contains(&palette.len())
                || palette
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite() || !(-0.5..=0.5).contains(value)))
        {
            return Err("history palette requires the standalone spatial RGB decoder and 2..=32 finite centered colours".into());
        }
        if let Some(dim) = self.world_gaussian_dim {
            let preset = self.model_size.network();
            if !self.world_gaussian
                || self.latent_shape.is_some()
                || dim == 0
                || dim > preset.stoch * preset.classes
            {
                return Err("world_gaussian_dim requires Gaussian latents, no latent_shape and a positive dimension within the preset width".into());
            }
        }
        if let Some(shape) = self.latent_shape {
            let preset = self.model_size.network();
            if shape.variables == 0
                || shape.classes < 2
                || shape.variables.checked_mul(shape.classes) != Some(preset.stoch * preset.classes)
            {
                return Err("latent_shape requires positive variables, classes > 1 and the preset's flattened width".into());
            }
            if !self.is_cdp() || !self.cdp_centered {
                return Err("latent_shape requires centered RGB CDP".into());
            }
        }
        if self.actor_residual
            && (!self.is_cdp()
                || !self.cdp_centered
                || !self.loss_scales.policy.is_finite()
                || self.loss_scales.policy <= 0.0
                || !self.expert_imitation.as_ref().is_some_and(|expert| {
                    expert.world_gradient
                        && expert.scale.is_finite()
                        && expert.scale > 0.0
                        && expert.decay.is_none()
                })
                || self.actor_critic_gradient
                || self.actor_dynamics_scale.is_some()
                || self.replay_actor_scale.is_some()
                || self.actor_learning_starts != 0)
        {
            return Err("residual actor requires centered RGB CDP, positive policy and constant world expert imitation, without other actor losses or delayed actor updates".into());
        }
        if self.actor_hidden_units == Some(0) {
            return Err("actor_hidden_units must be positive".into());
        }
        if self.actor_expected_value
            && (self.actor_residual
                || self.actor_dynamics_scale.is_some()
                || self.replay_actor_scale.is_some()
                || self.actor_critic_gradient
                || !self.loss_scales.policy.is_finite()
                || self.loss_scales.policy <= 0.0)
        {
            return Err("expected-value actor requires positive policy scale without other actor experiments".into());
        }
        if let Some(scale) = self.actor_dynamics_scale
            && (!scale.is_finite()
                || scale <= 0.0
                || !(scale * self.loss_scales.policy).is_finite())
        {
            return Err(
                "actor dynamics requires a positive finite scale and finite weighted scale".into(),
            );
        }
        if self.replay_value_completed
            && (self.uses_disagreement()
                || self.replay_actor_auxiliary_boundaries
                || !self.loss_scales.replay_value.is_finite()
                || self.loss_scales.replay_value <= 0.0)
        {
            return Err("completed replay values require observed rewards, game-level returns and a positive replay-value scale".into());
        }
        if let Some(scale) = self.replay_actor_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err("completed-return actor requires a positive finite scale".into());
        }
        if self.replay_actor_scale.is_some()
            && self.uses_disagreement()
            && !self.replay_actor_observed_lower_bound
        {
            return Err("completed-return actor with model-derived disagreement requires explicit observed lower-bound mode".into());
        }
        if self.replay_actor_observed_lower_bound
            && (self.replay_actor_scale.is_none() || !self.uses_disagreement())
        {
            return Err("observed lower-bound mode requires completed-return actor learning and active disagreement".into());
        }
        if self.replay_actor_auxiliary_boundaries
            && (!self.replay_actor_observed_lower_bound
                || self.extrinsic_reward_scale < 0.0
                || self.intrinsic_reward_scale < 0.0)
        {
            return Err("auxiliary actor boundaries require observed lower-bound mode and nonnegative reward scales".into());
        }
        let size = self.network();
        if self.rgb_spatial_decoder
            && (!self.observation_kind.is_rgb()
                || self.video_encoder.is_some()
                || self.rgb_frame_pair
                || self.rgb_previous_frame
                || self.rgb_latent_only
                || self.world_gaussian
                || self.observation_crop.is_some()
                || self.is_cdp()
                || self.loss_scales.policy != 0.0
                || self.loss_scales.value != 0.0
                || self.loss_scales.replay_value != 0.0
                || self.actor_critic_gradient)
        {
            return Err("spatial decoder requires a standalone plain categorical RGB world".into());
        }
        if self.world_gaussian
            && (!self.observation_kind.is_rgb()
                || self.video_encoder.is_some()
                || self.rgb_frame_pair
                || self.rgb_previous_frame
                || self.rgb_latent_only
                || self.observation_crop.is_some()
                || self.is_cdp()
                || self.loss_scales.policy != 0.0
                || self.loss_scales.value != 0.0
                || self.loss_scales.replay_value != 0.0
                || self.actor_critic_gradient)
        {
            return Err("Gaussian latents require a standalone plain RGB world".into());
        }
        if self.rgb_previous_frame
            && (!self.observation_kind.is_rgb()
                || self.video_encoder.is_some()
                || self.rgb_frame_pair
                || self.rgb_latent_only
                || self.observation_crop.is_some()
                || self.is_cdp()
                || self.loss_scales.policy != 0.0
                || self.loss_scales.value != 0.0
                || self.loss_scales.replay_value != 0.0
                || self.actor_critic_gradient)
        {
            return Err(
                "previous-frame RGB requires a plain recurrent RGB world without behavior learning"
                    .into(),
            );
        }
        if self.rgb_latent_only
            && (!self.observation_kind.is_rgb()
                || self.video_encoder.is_some()
                || self.rgb_frame_pair
                || self.observation_crop.is_some()
                || self.is_cdp()
                || self.loss_scales.policy != 0.0
                || self.loss_scales.value != 0.0
                || self.loss_scales.replay_value != 0.0
                || self.actor_critic_gradient)
        {
            return Err(
                "latent-only RGB requires a plain learned RGB world without behavior learning"
                    .into(),
            );
        }
        if self.rgb_stride2 && (!self.observation_kind.is_rgb() || self.video_encoder.is_some()) {
            return Err("RGB stride-two convolutions require the learned RGB encoder".into());
        }
        if self.rgb_coordinates && (!self.observation_kind.is_rgb() || self.video_encoder.is_some())
        {
            return Err("RGB coordinates require the learned RGB encoder".into());
        }
        if self.rgb_frame_pair && (!self.is_cdp() || self.color_centroid_supervision.is_some()) {
            return Err("RGB frame pairs require CDP without color-centroid supervision".into());
        }
        if let Some(crop) = self.observation_crop {
            crop.check().map_err(str::to_owned)?;
            if !self.observation_kind.is_rgb() || self.color_centroid_supervision.is_some() {
                return Err(
                    "image crop requires learned RGB without color-centroid supervision".into(),
                );
            }
        }
        if let Some(rollout) = &self.open_loop_supervision
            && (!self.is_cdp()
                || rollout.horizon < 2
                || rollout.horizon >= self.world_backprop_length
                || !rollout.scale.is_finite()
                || rollout.scale <= 0.0
                || [
                    self.loss_scales.future_prediction,
                    self.loss_scales.reward,
                    self.loss_scales.continuation,
                ]
                .iter()
                .any(|weight| !(weight * rollout.scale).is_finite()))
        {
            return Err("open-loop supervision requires CDP, 2 <= horizon < world BPTT, and a positive finite scale".into());
        }
        if let Some(expert) = &self.expert_imitation
            && (!expert.scale.is_finite()
                || expert.scale <= 0.0
                || expert.action_weights.len() != self.action_count
                || !expert.action_weights.iter().any(|&weight| weight > 0.0)
                || expert.action_weights.iter().any(|&weight| {
                    !weight.is_finite() || weight < 0.0 || !(weight * expert.scale).is_finite()
                }))
        {
            return Err("expert imitation requires positive finite scale and nonnegative finite action weights".into());
        }
        if self
            .expert_imitation
            .as_ref()
            .and_then(|expert| expert.decay.as_ref())
            .is_some_and(|decay| decay.end_update <= decay.start_update)
        {
            return Err("expert imitation decay requires end_update > start_update".into());
        }
        if self.expert_imitation.as_ref().is_some_and(|expert| {
            !expert.world_gradient && expert.decay.as_ref().is_some_and(|decay| decay.actor_only)
        }) {
            return Err("actor-only expert imitation decay requires world_gradient".into());
        }
        for centroid in self
            .color_centroid_supervision
            .iter()
            .chain(self.rgb_color_attention.iter())
        {
            let [x0, y0, x1, y1] = centroid.region;
            let side = self.observation_grid();
            let contains_center = |low, high| {
                (0..side).any(|i| {
                    let center = (i as f32 + 0.5) / side as f32;
                    center >= low && center < high
                })
            };
            if !self.observation_kind.is_rgb()
                || !centroid.scale.is_finite()
                || centroid.scale <= 0.0
                || centroid.channel >= 3
                || !centroid.background_scale.is_finite()
                || centroid.background_scale <= 0.0
                || centroid
                    .region
                    .iter()
                    .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
                || x0 >= x1
                || y0 >= y1
                || !contains_center(x0, x1)
                || !contains_center(y0, y1)
            {
                return Err("color centroid requires RGB, positive finite scales, channel0..2 and a nonempty normalized region".into());
            }
        }
        if self.cdp_centered
            && (!self.is_cdp()
                || self.world_backprop_length != self.batch_length
                || self.world_microbatch_size() != self.batch_size
                || self.batch_size.saturating_mul(self.batch_length) < 2)
        {
            return Err(
                "centered CDP requires CDP with full-batch BPTT and at least two targets".into(),
            );
        }
        for (statistics, enabled) in [
            (
                &self.future_target_standardization,
                self.loss_scales.future_prediction > 0.0,
            ),
            (
                &self.reconstruction_target_standardization,
                self.loss_scales.reconstruction > 0.0,
            ),
        ] {
            if let Some(stats) = statistics
                && (self.observation_kind != ObservationKind::Features
                    || !enabled
                    || stats.mean.len() != self.observation_dim()
                    || stats.scale.len() != self.observation_dim()
                    || stats.mean.iter().any(|x| !x.is_finite())
                    || stats
                        .scale
                        .iter()
                        .any(|x| !x.is_finite() || *x <= 0.0 || !x.recip().is_finite()))
            {
                return Err("target standardization requires an enabled feature head, finite means and positive scales".into());
            }
        }
        if self.video_encoder.is_some()
            && (self.observation_kind != ObservationKind::Features
                || self.batch_length != crate::vision::levjepa::FRAMES
                || self.world_backprop_length != self.batch_length
                || self.replay_context != 1
                || self.loss_scales.reconstruction != 0.0
                || self.loss_scales.future_prediction <= 0.0)
        {
            return Err(
                "causal pixel replay requires feature prediction, full T16 BPTT and context1"
                    .into(),
            );
        }
        if self.observation_kind.is_rgb()
            && (self.visitation_bonus
                || (self.loss_scales.future_prediction > 0.0)
                    == (self.loss_scales.reconstruction > 0.0))
        {
            return Err("RGB learning requires exactly one of reconstruction or CDP, without host visitation".into());
        }
        if self.action_count <= 1 {
            return Err("action_count must be greater than one".into());
        }
        if let Some(table) = &self.world_action_features
            && (table.len() != self.action_count
                || table.iter().any(|row| {
                    row.len() != self.action_count
                        || row.iter().any(|x| !x.is_finite() || x.abs() > 1.0)
                }))
        {
            return Err(
                "world action features require an action_count square finite table in [-1, 1]"
                    .into(),
            );
        }
        if size.deter == 0 || !size.deter.is_multiple_of(size.blocks) {
            return Err("deter must be non-zero and divisible by blocks".into());
        }
        if size.hidden == 0
            || size.stoch == 0
            || size.classes == 0
            || (size.classes == 1 && !self.world_gaussian)
        {
            return Err("hidden/stoch/classes network dimensions are invalid".into());
        }
        if size.units == 0 || size.vision_depth == 0 {
            return Err("head and vision dimensions must be non-zero".into());
        }
        if self.replay_context == 0 {
            return Err("replay_context must include a cached state".into());
        }
        if let Some(fraction) = self.terminal_replay_fraction {
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err("terminal_replay_fraction must be finite and in [0, 1]".into());
            }
            if self.video_encoder.is_some() {
                return Err("terminal replay does not support causal video sequences".into());
            }
        }
        if let Some(weight) = self.terminal_continuation_weight
            && (!self.is_cdp()
                || self.open_loop_supervision.is_some()
                || self.horizon <= 1
                || !weight.is_finite()
                || weight <= 0.0
                || !(weight * self.loss_scales.continuation).is_finite()
                || !((weight - 1.0) / self.continuation_discount()).is_finite())
        {
            return Err("terminal continuation weight requires CDP without open-loop supervision, horizon > 1, and positive finite effective weight".into());
        }
        if let Some(scale) = self.prior_continuation_scale
            && (!self.is_cdp()
                || self.open_loop_supervision.is_some()
                || self.terminal_continuation_weight.is_some()
                || self.horizon <= 1
                || !scale.is_finite()
                || scale <= 0.0)
        {
            return Err("prior continuation requires CDP without open-loop or terminal weighting, horizon > 1, and positive finite scale".into());
        }
        if let Some(scale) = self.prior_reward_scale
            && (!self.is_cdp()
                || self.open_loop_supervision.is_some()
                || self.prior_continuation_scale.is_some()
                || !scale.is_finite()
                || scale <= 0.0)
        {
            return Err("prior reward requires CDP without open-loop or prior continuation, and positive finite scale".into());
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
        let world_microbatch_size = self.world_microbatch_size();
        if world_microbatch_size == 0
            || world_microbatch_size > self.batch_size
            || !self.batch_size.is_multiple_of(world_microbatch_size)
        {
            return Err("world_microbatch_size must be a positive divisor of batch_size".into());
        }
        let stride = if self.video_encoder.is_some() {
            self.batch_length
        } else {
            1
        };
        let warmup_frames = self
            .replay_warmup_sequences()
            .checked_mul(stride)
            .and_then(|frames| frames.checked_add(self.replay_context))
            .and_then(|frames| frames.checked_add(self.batch_length - 1));
        if warmup_frames.is_none_or(|frames| frames > self.replay_capacity) {
            return Err("replay_capacity cannot satisfy the D3 learner warmup gate".into());
        }
        if !self.train_ratio.is_finite() || self.train_ratio < 0.0 {
            return Err("train_ratio must be finite and non-negative".into());
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
        for (name, rate) in [
            ("behavior_learning_rate", self.behavior_learning_rate),
            ("encoder_learning_rate", self.encoder_learning_rate),
            ("dynamics_learning_rate", self.dynamics_learning_rate),
        ] {
            if rate.is_some_and(|rate| {
                !rate.is_finite() || rate <= 0.0 || !(rate / self.learning_rate).is_finite()
            }) {
                return Err(format!(
                    "{name} must be finite and positive with a finite rate ratio"
                ));
            }
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
            self.loss_scales.future_prediction,
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
        if self.disagreement_bonus && (self.intrinsic_reward_scale < 0.0 || self.visitation_bonus) {
            return Err(
                "disagreement requires a non-negative intrinsic scale and no host visitation"
                    .into(),
            );
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
    fn terminal_continuation_weight_preserves_defaults_and_checks_scope() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("terminal_continuation_weight").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config.terminal_continuation_weight = Some(10.0);
        assert!(config.check().is_err());
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        for weight in [0.5, 1.0, 10.0] {
            config.terminal_continuation_weight = Some(weight);
            config.check().unwrap();
            let serialized = serde_json::to_value(&config).unwrap();
            assert_eq!(serialized["terminal_continuation_weight"], weight);
            assert_eq!(
                serde_json::from_value::<DreamerConfig>(serialized).unwrap(),
                config
            );
        }
        for weight in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            config.terminal_continuation_weight = Some(weight);
            assert!(config.check().is_err());
        }
        config.terminal_continuation_weight = Some(10.0);
        config.open_loop_supervision = Some(OpenLoopSupervision {
            horizon: 3,
            scale: 1.0,
        });
        assert!(config.check().is_err());
        config.open_loop_supervision = None;
        config.horizon = 1;
        assert!(config.check().is_err());
        config.horizon = 333;
        config.loss_scales.continuation = f32::MAX;
        assert!(config.check().is_err());
    }

    #[test]
    fn prior_continuation_is_optional_and_validated() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("prior_continuation_scale").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config.prior_continuation_scale = Some(1.0);
        assert!(config.check().is_err());
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.check().unwrap();
        let saved = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(saved).unwrap(),
            config
        );
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            config.prior_continuation_scale = Some(scale);
            assert!(config.check().is_err());
        }
        config.prior_continuation_scale = Some(1.0);
        config.terminal_continuation_weight = Some(1.0);
        assert!(config.check().is_err());
        config.terminal_continuation_weight = None;
        config.open_loop_supervision = Some(OpenLoopSupervision {
            horizon: 3,
            scale: 1.0,
        });
        assert!(config.check().is_err());
        config.open_loop_supervision = None;
        config.horizon = 1;
        assert!(config.check().is_err());
    }

    #[test]
    fn prior_reward_is_optional_and_validated() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("prior_reward_scale").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config.prior_reward_scale = Some(1.0);
        assert!(config.check().is_err());
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.check().unwrap();
        let saved = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(saved).unwrap(),
            config
        );
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            config.prior_reward_scale = Some(scale);
            assert!(config.check().is_err());
        }
        config.prior_reward_scale = Some(1.0);
        config.prior_continuation_scale = Some(1.0);
        assert!(config.check().is_err());
        config.prior_continuation_scale = None;
        config.open_loop_supervision = Some(OpenLoopSupervision {
            horizon: 3,
            scale: 1.0,
        });
        assert!(config.check().is_err());
    }

    #[test]
    fn rgb_coordinates_preserve_observations_and_legacy_config() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("rgb_coordinates").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config.rgb_coordinates = true;
        assert!(config.check().unwrap_err().contains("coordinates require"));
        for (kind, side) in [
            (ObservationKind::Rgb32, 32),
            (ObservationKind::Rgb48, 48),
            (ObservationKind::Rgb64, 64),
        ] {
            config.observation_kind = kind;
            assert!(config.check().is_ok());
            assert_eq!(config.observation_dim(), 3 * side * side);
            let encoded = serde_json::to_value(&config).unwrap();
            assert_eq!(encoded["rgb_coordinates"], true);
            assert_eq!(
                serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
                config
            );
        }
        config.video_encoder = Some(VideoEncoder::Frozen);
        assert!(config.check().unwrap_err().contains("coordinates require"));
    }

    #[test]
    fn rgb_frame_pairs_preserve_legacy_config_and_require_cdp() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("rgb_frame_pair").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config.rgb_frame_pair = true;
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("frame pairs require CDP")
        );
        config.observation_kind = ObservationKind::Rgb64;
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("frame pairs require CDP")
        );
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        for (kind, side) in [
            (ObservationKind::Rgb32, 32),
            (ObservationKind::Rgb48, 48),
            (ObservationKind::Rgb64, 64),
        ] {
            config.observation_kind = kind;
            assert!(config.check().is_ok());
            assert_eq!(config.observation_channels(), 6);
            assert_eq!(config.observation_dim(), 6 * side * side);
            let encoded = serde_json::to_value(&config).unwrap();
            assert_eq!(encoded["rgb_frame_pair"], true);
            assert_eq!(
                serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
                config
            );
        }
        config.color_centroid_supervision = Some(ColorCentroidSupervision {
            scale: 1.0,
            channel: 0,
            background_scale: 1.0,
            region: [0.0, 0.0, 1.0, 1.0],
        });
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("without color-centroid")
        );
    }

    #[test]
    fn completed_replay_values_preserve_defaults_and_require_observed_game_returns() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("replay_value_completed").is_none());
        assert!(
            !serde_json::from_value::<DreamerConfig>(legacy)
                .unwrap()
                .replay_value_completed
        );
        config.replay_value_completed = true;
        config.check().unwrap();
        assert!(config.replay_actor_scale.is_none());
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = config.clone();
            invalid.loss_scales.replay_value = scale;
            assert!(invalid.check().is_err());
        }
        config.disagreement_bonus = true;
        config.intrinsic_reward_scale = 1.0;
        assert!(config.check().is_err());
        config.intrinsic_reward_scale = 0.0;
        config.check().unwrap();
        config.disagreement_bonus = false;
        config.visitation_bonus = true;
        config.intrinsic_reward_scale = 1.0;
        config.check().unwrap();
        config.replay_actor_auxiliary_boundaries = true;
        assert!(config.check().is_err());
    }

    #[test]
    fn completed_return_actor_config_is_optional_and_requires_observed_rewards() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("replay_actor_scale").is_none());
        assert!(legacy.get("replay_actor_observed_lower_bound").is_none());
        assert!(
            serde_json::from_value::<DreamerConfig>(legacy)
                .unwrap()
                .replay_actor_scale
                .is_none()
        );
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            config.replay_actor_scale = Some(scale);
            assert!(config.check().is_err());
        }
        config.replay_actor_scale = Some(100.0);
        config.check().unwrap();
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored.replay_actor_scale, Some(100.0));
        config.disagreement_bonus = true;
        config.intrinsic_reward_scale = 1.0;
        assert!(config.check().unwrap_err().contains("model-derived"));
        config.disagreement_bonus = false;
        config.visitation_bonus = true;
        config.check().unwrap();
    }

    #[test]
    fn completed_return_lower_bound_requires_both_actor_and_disagreement() {
        let mut config = DreamerConfig::tiny(3);
        config.replay_actor_observed_lower_bound = true;
        assert!(config.check().is_err());
        config.replay_actor_scale = Some(100.0);
        assert!(config.check().is_err());
        config.disagreement_bonus = true;
        config.intrinsic_reward_scale = 1.0;
        config.check().unwrap();
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        config.intrinsic_reward_scale = 0.0;
        assert!(config.check().is_err());
        config.intrinsic_reward_scale = 1.0;
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            config.replay_actor_scale = Some(scale);
            assert!(config.check().is_err());
        }
    }

    #[test]
    fn auxiliary_actor_boundaries_require_lower_bound_and_preserve_defaults() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("replay_actor_auxiliary_boundaries").is_none());
        assert!(
            !serde_json::from_value::<DreamerConfig>(legacy)
                .unwrap()
                .replay_actor_auxiliary_boundaries
        );
        config.replay_actor_auxiliary_boundaries = true;
        assert!(config.check().is_err());
        config.replay_actor_scale = Some(100.0);
        config.replay_actor_observed_lower_bound = true;
        config.disagreement_bonus = true;
        config.intrinsic_reward_scale = 1.0;
        config.check().unwrap();
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        config.extrinsic_reward_scale = -1.0;
        assert!(config.check().is_err());
    }

    #[test]
    fn replay_context_validates_prefix_capacity_and_preserves_default() {
        let mut config = DreamerConfig::tiny(3);
        assert_eq!(config.replay_context, 1);
        let default = serde_json::to_value(&config).unwrap();
        assert_eq!(default["replay_context"], 1);
        let original_warmup = config.replay_warmup_frames();
        for context in [2, 9] {
            config.replay_context = context;
            config.replay_capacity = original_warmup + context - 1;
            config.check().unwrap();
            assert_eq!(config.replay_warmup_frames(), config.replay_capacity);
            let encoded = serde_json::to_value(&config).unwrap();
            assert_eq!(
                serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
                config
            );
            config.replay_capacity -= 1;
            assert!(config.check().unwrap_err().contains("warmup"));
        }
        config.replay_context = 0;
        assert!(config.check().unwrap_err().contains("cached state"));
        config.replay_context = usize::MAX;
        assert!(config.check().unwrap_err().contains("warmup"));
        config.replay_context = 9;
        config.video_encoder = Some(VideoEncoder::Joint);
        config.batch_length = crate::vision::levjepa::FRAMES;
        config.world_backprop_length = config.batch_length;
        assert!(config.check().is_err());
    }

    #[test]
    fn expert_imitation_defaults_preserve_legacy_config_and_validate_weights() {
        let mut config = DreamerConfig::tiny(3);
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("expert_imitation").is_none());
        assert!(legacy.get("replay_actor_scale").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        for (scale, weights) in [
            (0.0, vec![1.0; 3]),
            (f32::NAN, vec![1.0; 3]),
            (f32::INFINITY, vec![1.0; 3]),
            (1.0, vec![1.0; 2]),
            (1.0, vec![0.0; 3]),
            (1.0, vec![-1.0, 1.0, 1.0]),
            (1.0, vec![f32::NAN, 1.0, 1.0]),
            (f32::MAX, vec![2.0; 3]),
        ] {
            config.expert_imitation = Some(ExpertImitation {
                scale,
                action_weights: weights,
                world_gradient: false,
                decay: None,
            });
            assert!(config.check().unwrap_err().contains("expert imitation"));
        }
        config.expert_imitation = Some(ExpertImitation {
            scale: 2.0,
            action_weights: vec![0.05, 0.0, 1.0],
            world_gradient: false,
            decay: None,
        });
        config.check().unwrap();
        let encoded = serde_json::to_value(&config).unwrap();
        assert!(encoded.get("expert_imitation").is_some());
        assert!(encoded["expert_imitation"].get("world_gradient").is_none());
        assert!(encoded["expert_imitation"].get("decay").is_none());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
            config
        );
        config.expert_imitation.as_mut().unwrap().world_gradient = true;
        assert!(config.expert_world_gradient());
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(serde_json::to_value(&config).unwrap())
                .unwrap(),
            config
        );
    }

    #[test]
    fn expert_decay_uses_report_update_boundaries_and_restored_config() {
        let mut config = DreamerConfig::tiny(3);
        config.expert_imitation = Some(ExpertImitation {
            scale: 100.0,
            action_weights: vec![1.0; 3],
            world_gradient: true,
            decay: Some(ExpertImitationDecay {
                actor_only: false,
                start_update: 10,
                end_update: 14,
            }),
        });
        config.check().unwrap();
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        let expert = restored.expert_imitation.unwrap();
        for (update, scale) in [
            (0, 100.0),
            (1, 100.0),
            (10, 100.0),
            (11, 75.0),
            (12, 50.0),
            (13, 25.0),
            (14, 0.0),
            (15, 0.0),
            (u64::MAX, 0.0),
        ] {
            assert_eq!(expert.scale_at_update(update), scale);
            assert_eq!(expert.world_scale_at_update(update), scale);
        }
        let expert = config.expert_imitation.as_mut().unwrap();
        expert.decay = Some(ExpertImitationDecay {
            actor_only: false,
            start_update: u64::MAX - 4,
            end_update: u64::MAX,
        });
        assert_eq!(expert.scale_at_update(u64::MAX - 2), 50.0);
        expert.decay = Some(ExpertImitationDecay {
            actor_only: false,
            start_update: 0,
            end_update: 1,
        });
        assert_eq!(expert.scale_at_update(0), 100.0);
        assert_eq!(expert.scale_at_update(1), 0.0);
        for (start_update, end_update) in [(0, 0), (10, 10), (11, 10)] {
            config.expert_imitation.as_mut().unwrap().decay = Some(ExpertImitationDecay {
                actor_only: false,
                start_update,
                end_update,
            });
            assert!(
                config
                    .check()
                    .unwrap_err()
                    .contains("expert imitation decay")
            );
        }
    }

    #[test]
    fn actor_only_decay_preserves_world_scale_and_legacy_serialization() {
        let mut config = DreamerConfig::tiny(3);
        config.expert_imitation = Some(ExpertImitation {
            scale: 100.0,
            action_weights: vec![1.0; 3],
            world_gradient: true,
            decay: Some(ExpertImitationDecay {
                start_update: 10,
                end_update: 14,
                actor_only: false,
            }),
        });
        let legacy = serde_json::to_value(&config).unwrap();
        assert_eq!(
            legacy["expert_imitation"]["decay"],
            serde_json::json!({
                "start_update": 10, "end_update": 14
            })
        );
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
            config
        );
        config
            .expert_imitation
            .as_mut()
            .unwrap()
            .decay
            .as_mut()
            .unwrap()
            .actor_only = true;
        config.check().unwrap();
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        let expert = restored.expert_imitation.unwrap();
        for (update, actor) in [
            (0, 100.0),
            (10, 100.0),
            (12, 50.0),
            (14, 0.0),
            (u64::MAX, 0.0),
        ] {
            assert_eq!(expert.scale_at_update(update), actor);
            assert_eq!(expert.world_scale_at_update(update), 100.0);
        }
        config.expert_imitation.as_mut().unwrap().world_gradient = false;
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("requires world_gradient")
        );
    }

    #[test]
    fn cdp_configuration_and_split_rates_roundtrip() {
        let mut config = DreamerConfig::new(18);
        config.model_size = ModelSize::Size1M;
        config.observation_kind = ObservationKind::Rgb64;
        assert!(!config.is_cdp());
        assert_eq!(config.prediction_dim(), 12288);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.encoder_learning_rate = Some(6e-6);
        config.dynamics_learning_rate = Some(4e-4);
        assert!(config.check().is_ok());
        assert!(config.is_cdp());
        assert_eq!(config.prediction_dim(), 256);
        let json = serde_json::to_vec(&config).unwrap();
        assert_eq!(
            config,
            serde_json::from_slice::<DreamerConfig>(&json).unwrap()
        );
        for rate in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let mut invalid = config.clone();
            invalid.encoder_learning_rate = Some(rate);
            assert!(invalid.check().is_err());
            invalid = config.clone();
            invalid.dynamics_learning_rate = Some(rate);
            assert!(invalid.check().is_err());
        }
        config.loss_scales.reconstruction = 1.0;
        assert!(
            config.check().is_err(),
            "no undeclared hybrid RGB/CDP objective"
        );
    }

    #[test]
    fn centered_cdp_requires_one_full_target_batch() {
        let mut config = DreamerConfig::tiny(3);
        assert!(!config.cdp_centered);
        config.cdp_centered = true;
        assert!(config.check().is_err());
        config.observation_kind = ObservationKind::Rgb64;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        assert!(config.check().is_ok());
        let json = serde_json::to_vec(&config).unwrap();
        assert_eq!(
            config,
            serde_json::from_slice::<DreamerConfig>(&json).unwrap()
        );
        config.world_microbatch_size = Some(1);
        assert!(config.check().is_err());
        config.world_microbatch_size = None;
        config.world_backprop_length = 1;
        assert!(config.check().is_err());
    }

    #[test]
    fn feature_standardization_is_optional_validated_and_serialized() {
        let mut config = DreamerConfig::tiny(3);
        assert!(config.future_target_standardization.is_none());
        config.loss_scales.future_prediction = 0.25;
        config.future_target_standardization = Some(FeatureStandardization {
            mean: vec![2.0; config.observation_dim()],
            scale: vec![0.1; config.observation_dim()],
        });
        assert!(config.check().is_ok());
        let roundtrip: DreamerConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(roundtrip, config);
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            let mut bad = config.clone();
            bad.future_target_standardization.as_mut().unwrap().scale[0] = invalid;
            assert!(bad.check().is_err());
        }
        let mut bad = config.clone();
        bad.future_target_standardization
            .as_mut()
            .unwrap()
            .mean
            .pop();
        assert!(bad.check().is_err());
        config.loss_scales.future_prediction = 0.0;
        assert!(config.check().is_err());
    }

    #[test]
    fn posterior_target_statistics_require_feature_reconstruction() {
        let mut config = DreamerConfig::tiny(3);
        assert!(config.reconstruction_target_standardization.is_none());
        config.reconstruction_target_standardization = Some(FeatureStandardization {
            mean: vec![2.0; config.observation_dim()],
            scale: vec![0.1; config.observation_dim()],
        });
        assert!(config.check().is_ok());
        let restored: DreamerConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(config, restored);
        config
            .reconstruction_target_standardization
            .as_mut()
            .unwrap()
            .scale[0] = 0.0;
        assert!(config.check().is_err());
        config
            .reconstruction_target_standardization
            .as_mut()
            .unwrap()
            .scale[0] = 0.1;
        config.loss_scales.reconstruction = 0.0;
        assert!(config.check().is_err());
        config.loss_scales.reconstruction = 1.0;
        config.observation_kind = ObservationKind::Rgb64;
        assert!(config.check().is_err());
    }

    #[test]
    fn upstream_size_presets_are_pinned() {
        let one = ModelSize::Size1M.network();
        assert_eq!(
            (
                one.deter,
                one.hidden,
                one.stoch,
                one.classes,
                one.blocks,
                one.units,
                one.vision_depth,
            ),
            (512, 64, 32, 4, 8, 64, 4)
        );
        let twelve = ModelSize::Size12M.network();
        assert_eq!(
            (
                twelve.deter,
                twelve.hidden,
                twelve.stoch,
                twelve.classes,
                twelve.blocks,
                twelve.units,
                twelve.vision_depth,
            ),
            (2_048, 256, 32, 16, 8, 256, 16)
        );
        let two_hundred = ModelSize::Size200M.network();
        assert_eq!(
            (
                two_hundred.deter,
                two_hundred.hidden,
                two_hundred.stoch,
                two_hundred.classes,
                two_hundred.blocks,
                two_hundred.units,
                two_hundred.vision_depth,
            ),
            (8_192, 1_024, 32, 64, 8, 1_024, 64)
        );
    }

    #[test]
    fn intermediate_cdp_configuration_roundtrips() {
        let mut config = DreamerConfig::new(18);
        config.model_size = ModelSize::Size4M;
        config.observation_kind = ObservationKind::Rgb64;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.cdp_centered = true;
        config.batch_size = 8;
        config.batch_length = 16;
        config.world_backprop_length = 16;
        config.world_microbatch_size = Some(8);
        config.train_ratio = 16.0;
        assert!(config.check().is_ok());
        assert_eq!(config.feature_dim(), 1_280);
        assert_eq!(config.prediction_dim(), 512);
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["model_size"], "size4_m");
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
            config
        );
    }

    #[test]
    fn d3_data_defaults_are_explicit() {
        let config = DreamerConfig::new(18);
        assert_eq!(config.action_count, 18);
        assert_eq!(config.model_size, ModelSize::Size12M);
        assert_eq!(config.replay_capacity, 100_000);
        assert_eq!((config.batch_size, config.batch_length), (16, 64));
        assert_eq!(config.world_backprop_length, 8);
        assert_eq!(config.world_microbatch_size, None);
        assert_eq!(config.world_microbatch_size(), 16);
        assert_eq!(config.imagination_length, 15);
        assert_eq!(config.value_bins, 255);
        assert_eq!(config.replay_context, 1);
        assert_eq!(config.replay_warmup_sequences(), 1_024);
        assert_eq!(config.replay_warmup_frames(), 1_088);
        assert_eq!(config.train_ratio, 32.0);
        assert_eq!(config.horizon, 333);
        assert_eq!(config.continuation_discount(), 1.0 - 1.0 / 333.0);
        assert_eq!(config.lambda, 0.95);
        assert_eq!(config.free_nats, 1.0);
        assert_eq!(config.unimix, 0.01);
        assert_eq!(config.actor_unimix, 0.0);
        assert_eq!(config.actor_entropy, 3e-4);
        assert_eq!(config.slow_value_rate, 0.02);
        assert_eq!(config.return_norm_rate, 0.01);
        assert_eq!(config.learning_rate, 4e-5);
        assert_eq!(config.learning_rate_warmup, 1_000);
        assert_eq!(config.optimizer_beta1, 0.9);
        assert_eq!(config.optimizer_beta2, 0.999);
        assert_eq!(config.optimizer_epsilon, 1e-20);
        assert_eq!(config.agc, 0.3);
        assert_eq!(config.agc_pmin, 1e-3);
        assert_eq!(config.loss_scales, LossScales::default());
        assert_eq!(config.dynamics_free_nats, None);
        assert_eq!(config.dynamics_free_nats(), config.free_nats);
        assert_eq!(config.behavior_learning_rate, None);
        assert_eq!(config.behavior_learning_rate(), config.learning_rate);
        assert_eq!(config.actor_learning_starts, 0);
        assert_eq!(config.actor_update_scale(0), 1.0);
        assert_eq!(config.observation_decoder_depth, OBSERVATION_CHANNELS);
        assert_eq!(config.observation_decoder_depth(), OBSERVATION_CHANNELS);
        assert!(config.replay_value_gradient);
        assert_eq!(config.extrinsic_reward_scale, 1.0);
        assert_eq!(config.intrinsic_reward_scale, 0.0);
        assert_eq!(config.seed, 0);
        assert!(!config.skip_full_optimize);
    }

    #[test]
    fn world_microbatch_must_partition_the_effective_batch() {
        let mut config = DreamerConfig::new(18);
        config.world_microbatch_size = Some(4);
        assert_eq!(config.world_microbatch_size(), 4);
        assert!(config.check().is_ok());

        config.world_microbatch_size = Some(3);
        assert_eq!(
            config.check().unwrap_err(),
            "world_microbatch_size must be a positive divisor of batch_size"
        );
        config.world_microbatch_size = Some(0);
        assert!(config.check().is_err());
        config.world_microbatch_size = Some(32);
        assert!(config.check().is_err());
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
    fn older_configs_default_new_fields() {
        let mut value = serde_json::to_value(DreamerConfig::tiny(3)).unwrap();
        assert!(value.get("open_loop_supervision").is_none());
        assert!(value.get("observation_crop").is_none());
        let object = value.as_object_mut().unwrap();
        object.remove("replay_value_gradient");
        object.remove("replay_value_bootstrap");
        object.remove("actor_critic_gradient");
        object.remove("behavior_learning_rate");
        object.remove("dynamics_free_nats");
        object.remove("actor_learning_starts");
        object.remove("observation_decoder_depth");
        object.remove("world_microbatch_size");
        object.remove("training_precision");
        let restored: DreamerConfig = serde_json::from_value(value).unwrap();
        assert!(restored.replay_value_gradient);
        assert_eq!(
            restored.replay_value_bootstrap,
            ReplayValueBootstrap::Imagined
        );
        assert!(!restored.actor_critic_gradient);
        assert_eq!(restored.behavior_learning_rate, None);
        assert_eq!(restored.behavior_learning_rate(), restored.learning_rate);
        assert_eq!(restored.dynamics_free_nats, None);
        assert_eq!(restored.dynamics_free_nats(), restored.free_nats);
        assert_eq!(restored.actor_learning_starts, 0);
        assert_eq!(restored.actor_update_scale(0), 1.0);
        assert_eq!(restored.observation_decoder_depth, 0);
        assert_eq!(
            restored.observation_decoder_depth(),
            restored.network().vision_depth
        );
        assert_eq!(restored.world_microbatch_size, None);
        assert_eq!(restored.world_microbatch_size(), restored.batch_size);
        assert_eq!(restored.training_precision, TrainingPrecision::Auto);
        assert_eq!(restored.open_loop_supervision, None);
        assert_eq!(restored.observation_crop, None);
    }

    #[test]
    fn image_crop_roundtrips_and_requires_compatible_pixels() {
        use crate::vision::preprocess_gpu::FrameCrop;
        let mut config = DreamerConfig::tiny(3);
        config.observation_crop = Some(FrameCrop {
            x: 0,
            y: 32,
            width: 160,
            height: 168,
        });
        assert!(config.check().is_err());
        config.observation_kind = ObservationKind::Rgb64;
        assert!(config.check().is_ok());
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["observation_crop"]["y"], 32);
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(value).unwrap(),
            config
        );
        config.color_centroid_supervision = Some(ColorCentroidSupervision {
            scale: 1.0,
            channel: 0,
            background_scale: 1.5,
            region: [0.0, 0.0, 1.0, 1.0],
        });
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("without color-centroid")
        );
        config.color_centroid_supervision = None;
        config.observation_crop.as_mut().unwrap().width = 0;
        assert!(config.check().is_err());
    }

    #[test]
    fn replay_bootstrap_roundtrips_and_rejects_unknown_modes() {
        let mut config = DreamerConfig::tiny(3);
        assert_eq!(
            config.replay_value_bootstrap,
            ReplayValueBootstrap::Imagined
        );
        config.replay_value_bootstrap = ReplayValueBootstrap::Posterior;
        let mut value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["replay_value_bootstrap"], "posterior");
        let restored: DreamerConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            restored.replay_value_bootstrap,
            ReplayValueBootstrap::Posterior
        );
        value["replay_value_bootstrap"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<DreamerConfig>(value).is_err());
    }

    #[test]
    fn open_loop_requires_cdp_valid_horizon_and_finite_weights() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.batch_length = 4;
        config.world_backprop_length = 4;
        config.open_loop_supervision = Some(OpenLoopSupervision {
            horizon: 3,
            scale: 1.0,
        });
        assert!(config.check().is_ok());
        let restored: DreamerConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(config, restored);
        for horizon in [0, 1, 4, usize::MAX] {
            config.open_loop_supervision.as_mut().unwrap().horizon = horizon;
            assert!(config.check().is_err());
        }
        config.open_loop_supervision.as_mut().unwrap().horizon = 3;
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            config.open_loop_supervision.as_mut().unwrap().scale = scale;
            assert!(config.check().is_err());
        }
        config.open_loop_supervision.as_mut().unwrap().scale = 1.0;
        config.observation_kind = ObservationKind::Features;
        assert!(config.check().is_err());
    }

    #[test]
    fn training_precision_round_trips_and_selects_the_backend_policy() {
        for (precision, policy) in [
            (TrainingPrecision::Auto, meganeura::CoopPolicy::Auto),
            (TrainingPrecision::F32, meganeura::CoopPolicy::NativeF32),
            (TrainingPrecision::F16, meganeura::CoopPolicy::AllowF16),
            (TrainingPrecision::Bf16, meganeura::CoopPolicy::NativeF32),
            (TrainingPrecision::Mixed, meganeura::CoopPolicy::NativeF32),
        ] {
            let mut config = DreamerConfig::tiny(3);
            config.training_precision = precision;
            let restored: DreamerConfig =
                serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
            assert_eq!(restored, config);
            assert_eq!(restored.training_precision.coop_policy(), policy);
        }
    }
}
