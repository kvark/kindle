//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation → (adapter) → encoder → world model → reward/return → policy → (adapter) → action.
//!
//! The agent's GPU graphs are built once with universal token sizes
//! (`OBS_TOKEN_DIM`, `MAX_ACTION_DIM`, `WM_ACTION_DIM`); a per-env `EnvAdapter` translates
//! between the env's native shapes and these token sizes. `switch_lane`
//! swaps one lane's adapter without touching any compiled graph.
//!
//! ## Batched lanes (Phase E)
//!
//! The agent is multi-lane: `N = config.batch_size` concurrent lanes share
//! the three compiled GPU sessions. Each lane owns its own adapter,
//! experience buffer, reward circuit and boundary flag; every `observe()`
//! call advances all N lanes in lockstep, stacking per-lane obs/action/
//! z_target/task rows into a single batched dispatch for the world model
//! and policy.
//!
//! For `N = 1` the runtime behaviour matches the pre-Phase-E single-lane
//! agent — construction takes a one-element `vec![adapter]` and every
//! step fed a one-element slice.
//!
//! Two core GPU sessions with independent learning rates:
//! 1. World model (encoder + world model): base LR
//! 2. Policy + value: 0.5× base, gated on warmup
//!
//! Phase 4 continual learning mechanisms:
//! - Replay mixing
//! - Representation drift monitor
//! - Entropy floor

use crate::OptLevel;
use crate::adapter::{
    ACTION_PARAMETER_DIM, EnvAdapter, MAX_ACTION_DIM, OBS_TOKEN_DIM, TASK_DIM, WM_ACTION_DIM,
};
use crate::approach;
use crate::buffer::{ExperienceBuffer, Transition};
use crate::coord;
use crate::delta_goals;
use crate::diayn;
use crate::encoder::{CnnEncoder, Encoder};
use crate::env::{Action, ActionKind, Environment, Observation, StepResult};
use crate::option;
use crate::outcome;
use crate::planner;
use crate::policy;
use crate::reward::{RewardCircuit, RewardWeights};
use crate::rnd;
use crate::world_model::WorldModel;
use crate::xeps_memory;
use hashbrown::HashMap;
use meganeura::Session;
use meganeura::graph::{Graph, NodeId};
use meganeura::nn;
use std::sync::Arc;

/// Error returned by [`Agent::bind_v2s_image_external`].
#[derive(Debug)]
pub enum BindV2sImageError {
    /// `Agent` was constructed without `encoder_kind == EfficientNetV2S`,
    /// so there is no V2-S session whose `image` input we can rebind.
    NoV2sSession,
    /// meganeura rejected the bind — see [`meganeura::ExternalBindError`].
    BindFailed(meganeura::ExternalBindError),
}

impl std::fmt::Display for BindV2sImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoV2sSession => write!(
                f,
                "bind_v2s_image_external requires encoder_kind=EfficientNetV2S"
            ),
            Self::BindFailed(e) => write!(f, "bind_external_buffer failed: {e:?}"),
        }
    }
}

impl std::error::Error for BindV2sImageError {}
use rand::Rng;

/// What encoder kindle builds as the WM graph's backbone.
///
/// `Mlp` (default) is kindle's original dense encoder — flat
/// obs-token vector → hidden → latent. Suits structured
/// low-dim obs (CartPole, LunarLander, Taxi, etc.).
///
/// `Cnn { channels, height, width }` builds a small conv-net
/// encoder on raw NCHW pixel/grid input. Intended for visual
/// tasks (ARC-AGI-3's 64×64 colour grid, Atari-like frames).
/// When this variant is selected, `Agent::set_visual_obs` must
/// be called every step before `observe` with a flat
/// `[batch_size · channels · height · width]` input; the obs
/// token path still flows to the reward circuit in parallel.
///
/// `EfficientNetV2S` runs a frozen, ImageNet-pretrained
/// EfficientNetV2-S features[0:6] (via meganeura) on raw RGB
/// `[batch · 3 · 192 · 192]` input as a sibling session, then
/// feeds its `(160, 12, 12)` feature map into the same internal
/// CNN encoder as `Cnn`. The harness writes raw RGB into
/// `Agent::image_input_host_ptr` each step; kindle does the V2-S
/// forward + feature upload internally during `observe`. The V2-S
/// weights file path is supplied via
/// `AgentConfig::efficientnet_weights_path`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    Mlp,
    /// Tiny CNN: conv(8, 3×3, s=2) → conv(16, 3×3, s=2) → global_avg_pool → fc.
    /// ~5k params, suitable only for tiny visual tasks (synthetic/grid/test).
    /// The global pool destroys spatial information — DON'T use for Atari.
    Cnn {
        channels: u32,
        height: u32,
        width: u32,
    },
    /// Nature-DQN-scale CNN (Mnih et al. 2015):
    /// conv(32, 8×8, s=4) → conv(64, 4×4, s=2) → conv(64, 3×3, s=1)
    /// → flatten → fc(512) → fc(latent_dim). ~1.7M params.
    /// Standard for 84×84×4 Atari preprocessed frames; preserves spatial
    /// structure via flatten (no global pooling).
    CnnDqn {
        channels: u32,
        height: u32,
        width: u32,
    },
    EfficientNetV2S,
}

/// Output channel count of EfficientNetV2-S features[0:6].
pub const EFFICIENTNET_V2S_OUT_CHANNELS: u32 = 160;
/// Output spatial side length (12 × 12) of V2-S features[0:6].
pub const EFFICIENTNET_V2S_OUT_HW: u32 = 12;
/// Input spatial side length (192 × 192) accepted by V2-S.
pub const EFFICIENTNET_V2S_IN_HW: u32 = 192;
/// RGB channel count.
pub const EFFICIENTNET_V2S_IN_CHANNELS: u32 = 3;

impl EncoderKind {
    /// Flat element count for the visual_obs slot: `0` for Mlp,
    /// `channels · height · width` for `Cnn` / `CnnDqn`, `160 · 12 · 12`
    /// for V2-S (the V2-S feature map dim — V2-S's raw image input lives
    /// in a separate buffer reached via `Agent::image_input_host_ptr`).
    pub fn visual_dim(&self) -> usize {
        match *self {
            EncoderKind::Mlp => 0,
            EncoderKind::Cnn {
                channels,
                height,
                width,
            }
            | EncoderKind::CnnDqn {
                channels,
                height,
                width,
            } => (channels as usize) * (height as usize) * (width as usize),
            EncoderKind::EfficientNetV2S => {
                (EFFICIENTNET_V2S_OUT_CHANNELS as usize)
                    * (EFFICIENTNET_V2S_OUT_HW as usize)
                    * (EFFICIENTNET_V2S_OUT_HW as usize)
            }
        }
    }

    /// Internal CNN encoder shape (channels, height, width) used
    /// to build the latent encoder. For `Cnn` / `CnnDqn` this is the
    /// user-supplied shape; for `V2-S` it's the fixed (160, 12, 12)
    /// V2-S feature shape.
    pub fn cnn_shape(&self) -> Option<(u32, u32, u32)> {
        match *self {
            EncoderKind::Mlp => None,
            EncoderKind::Cnn {
                channels,
                height,
                width,
            }
            | EncoderKind::CnnDqn {
                channels,
                height,
                width,
            } => Some((channels, height, width)),
            EncoderKind::EfficientNetV2S => Some((
                EFFICIENTNET_V2S_OUT_CHANNELS,
                EFFICIENTNET_V2S_OUT_HW,
                EFFICIENTNET_V2S_OUT_HW,
            )),
        }
    }
}

/// Criterion for ranking terminal entries into the M7 prototype
/// buffer's top-P% fraction. See `AgentConfig::approach_rank_by`.
///
/// The M7 confidence-weighting run (commit ac27d5c) showed that
/// ranking by `Return` converges on the wrong prototype on
/// LunarLander, because the highest-return episodes under v3
/// homeo are the shortest crashes. `Novelty` ranks by terminal
/// rarity instead, which on LunarLander promotes the rare
/// soft-landing terminal basin. Trade-off: on envs where success
/// is a *common* terminal state (e.g. CartPole timeouts), `Novelty`
/// promotes rare-crash terminals and is counter-productive;
/// `Return` is the right choice there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApproachRankBy {
    /// Cumulative intrinsic episode return (default; M7 v1 behaviour).
    Return,
    /// `1 / sqrt(visit_count(z_end))` — standard kindle novelty at
    /// the terminal latent, computed on the lane's grid-discretized
    /// experience buffer. Rare terminals rank high.
    Novelty,
}

/// What target the M6 outcome-value head trains against at
/// episode completion. See `AgentConfig::outcome_target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeTarget {
    /// Train `R̂` against the centered *sum* of per-step `r_base`
    /// over the just-ended episode. Every step of the episode
    /// shares one target. The original design; M6 v2 showed this
    /// converges on an episode-wide mean (intra-episode-flat `R̂`).
    EpisodeSum,
    /// Train `R̂` against the *last step's* `r_base` before episode
    /// boundary. Same shared-target-per-episode shape as
    /// `EpisodeSum` but aimed at the terminal state's homeo
    /// profile.
    TerminalReward,
    /// Train `R̂` against a *per-step* reward-to-go target:
    /// `target(t) = Σ_{k=t}^{T} r_base_k`. Each step in a
    /// completed episode now carries its own supervision signal,
    /// which defeats the M6 v2 "intra-episode-flat" failure mode —
    /// early windows of a soft-landing episode get a higher
    /// expected-future-reward target than early windows of a crash
    /// episode, so `R̂` learns per-step differentiation instead of
    /// a single per-episode bias.
    RewardToGo,
}

/// How M6 injects its output into the agent-facing reward.
///
/// `Raw` is the default — simply add `α · clamp(R̂)` to the step
/// reward. `PotentialDelta` adds `α · (R̂_t − R̂_{t-1})` instead,
/// i.e. the per-step *change* in the state-value estimate rather
/// than its absolute value. Classical potential-based shaping (Ng
/// et al. 1999) — guaranteed policy-invariant up to a constant,
/// converts a state-value head into a per-step signal
/// automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeBonus {
    Raw,
    PotentialDelta,
}

/// Agent configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub lr_policy: f32,
    /// Use the Adam optimizer instead of SGD for all sessions
    /// (policy, wm, option). Standard for vision-based RL
    /// (Atari, etc) where SGD's lack of per-parameter scaling makes
    /// CNN training extremely slow under sparse rewards. Adam betas
    /// fixed at 0.9 / 0.999, epsilon 1e-8 (PyTorch defaults). Default
    /// `false` (preserves the historical SGD behavior on classic
    /// control tasks where it worked well).
    pub use_adam: bool,
    /// Adam epsilon. Default 1e-8 matches PyTorch. Raise to 1e-4 for
    /// sparse-reward visual tasks (Atari) where late-training NaN
    /// collapse stems from `v_t -> 0` on idle parameters. The larger
    /// eps bounds the per-parameter update by ~lr/eps when v is tiny.
    pub adam_eps: f32,
    /// Global L2 gradient-norm clip applied to every policy/wm/option
    /// session step. 0.0 (default) disables it on the wm/option
    /// sessions; standard PPO uses 0.5. Bounds the per-step parameter
    /// update — required for sustained long-horizon training on
    /// sparse-reward visual tasks where Adam's variance estimate
    /// eventually drives unbounded updates.
    ///
    /// Note: the POLICY session always carries a built-in 10.0 safety
    /// clip (value-head NaN guard, Task #259); a non-zero
    /// `grad_clip_norm` overrides that, but 0.0 does not remove it.
    pub grad_clip_norm: f32,
    /// Cadence for the grad clip when enabled. 1 (default) clips every
    /// step. >1 amortizes the CPU-readback cost: clip every N steps.
    /// 5-10 typical for Atari long-horizon training (still bounds the
    /// runaway, ~5x faster than every-step clipping).
    pub grad_clip_every: u32,
    pub reward_weights: RewardWeights,
    pub warmup_steps: usize,
    /// Probability of running an additional replay training step per observe().
    pub replay_ratio: f32,
    pub grid_resolution: f32,
    /// Maximum size of the per-lane `visit_counts` HashMap that backs
    /// the `1/sqrt(visit_count)` novelty bonus. When inserting a new
    /// key would exceed this, the whole HashMap is cleared. 0 =
    /// unbounded (legacy behavior). At `latent_dim=256` and
    /// `grid_resolution=0.5` the grid is effectively unbounded — most
    /// steps produce a unique key, so memory grows ~1 KB/step ×
    /// n_lanes indefinitely. Cap suggested for joint multi-game runs:
    /// 10000–100000 (10 MB–100 MB per lane).
    pub visit_counts_max: usize,
    /// Number of latent dimensions used for visit-count quantization.
    /// 0 = all dims (legacy). For `latent_dim=256`, all dims gives an
    /// astronomically sparse grid → `visit_count` ≈ 1 always, making
    /// the novelty bonus uninformative for the planner's trajectory
    /// scoring. Setting `visit_count_dims=8` truncates to the first 8
    /// dims (~3^8 ≈ 6.6k cells) so revisits are common and the count
    /// genuinely measures novelty. Default 0 = unchanged behavior.
    pub visit_count_dims: usize,
    /// Optional random-projection dim for visit-count hashing. When >0,
    /// projects the latent through a fixed (deterministically seeded)
    /// random matrix of shape `[latent_dim, visit_count_proj_dim]`
    /// before quantizing. Approximately preserves L2 distance across
    /// the full latent (Johnson-Lindenstrauss); strictly more
    /// informative than `visit_count_dims` truncation which only sees
    /// the first N dims. Default 0 = use truncation or full latent.
    pub visit_count_proj_dim: usize,
    /// Seed for the random-projection matrix. Default 0x9E37_79B9... so
    /// behavior reproduces across runs of the same agent config.
    pub visit_count_proj_seed: u64,
    pub entropy_beta: f32,
    /// Floor for policy entropy — updates suppressed when entropy falls below this.
    pub entropy_floor: f32,
    /// Step interval between representation drift measurements.
    pub drift_interval: usize,
    /// Drift threshold beyond which encoder LR is reduced.
    pub drift_threshold: f32,
    /// Action persistence: the sampled action is held for this many
    /// consecutive `act()` calls per lane before the policy is resampled.
    /// `1` (default) is the classic per-step reactive policy. `K > 1`
    /// stretches the effective credit-assigner horizon by K× with no
    /// graph change — a cheap precursor to the Phase G option layer.
    /// See `docs/phase-g-l1-options.md`.
    pub action_repeat: usize,
    /// Label-smoothing coefficient for the per-row advantage-weighted
    /// action target sent to the policy graph. At `ε > 0`, the pure
    /// one-hot target `[0, 0, 1, 0]` becomes `[(1−ε)·one_hot + ε/K]`
    /// before advantage scaling, preventing the softmax from collapsing
    /// to deterministic (gradient is nonzero on every logit even at a
    /// one-hot softmax, so the policy can always recover from bad local
    /// minima). At `ε = 0` (default), the target is exact one-hot —
    /// backward compatible. `ε = 0.1` is a standard starting point.
    pub label_smoothing: f32,
    /// Number of discrete options for the L1 option-policy (Phase G).
    /// `1` (default) skips L1 entirely — no option_session compiled, no
    /// goal conditioning, byte-parity with the pre-Phase-G agent.
    /// `≥ 2` activates the full L1 path: option_session, per-lane goal
    /// conditioning on L0's z input, option-return training.
    pub num_options: usize,
    /// Goal-latent width decoded per option. Defaults to `latent_dim`.
    pub option_dim: usize,
    /// Fixed number of env steps per option (v1: no learned termination).
    pub option_horizon: usize,
    /// Entropy bonus weight on the option_session's option-choice
    /// distribution. > 0 prevents L1 from collapsing to picking only
    /// one option (the early-best one); each option then gets exercised
    /// regularly, which matters when per_option_heads is on (each option
    /// owns its own policy head). Default 0.0 (no bonus, current
    /// behavior preserved).
    pub option_entropy_beta: f32,
    /// DIAYN intrinsic reward weight. > 0 enables a discriminator
    /// `q(option | z)` and adds the per-step bonus
    /// `α · (log q(option_t | z_t) - log(1/num_options))` to the
    /// reward stream. Trains options to produce z trajectories that
    /// are mutually distinguishable. Requires `num_options >= 2`.
    /// Standard DIAYN uses α = 1.0; smaller (0.1-0.3) reduces the
    /// intrinsic vs extrinsic balance for envs with strong extrinsic
    /// signal. Default 0.0 (no DIAYN, parity preserved).
    pub diayn_reward_alpha: f32,
    /// DIAYN discriminator hidden width. Default 32.
    pub diayn_hidden_dim: usize,
    /// DIAYN discriminator learning rate. None = `lr_option`.
    pub diayn_lr: Option<f32>,
    /// L1 option-policy learning rate.
    pub lr_option: f32,
    /// Goal-achievement bonus coefficient. When L1 is active, each step's
    /// L0 reward is augmented with `−α · ‖z_t − goal‖`, giving L0 a
    /// self-supervised signal to drive the latent toward the option's goal
    /// regardless of the frozen reward circuit's output. `α = 0` disables.
    pub goal_bonus_alpha: f32,
    /// Phase G v4: enable the learned-termination head. When `true`, the
    /// agent forwards the option session every step, samples a Bernoulli
    /// from the predicted `β(z_t)`, and terminates the current option if
    /// either the sample or the fixed-horizon cap fires. The termination
    /// head is trained via BCE against a target derived from whether
    /// switching options would have been beneficial at that state.
    /// When `false`, termination is purely horizon-based (v1/v2/v3
    /// behaviour) and the head receives no gradient; the option-session
    /// graph shape is identical either way to keep parameter layouts
    /// stable across config changes.
    pub learned_termination: bool,
    /// Phase G Tier-3: replace the shared L0 `fc2` head with N
    /// per-option `[hidden_dim → action_dim]` heads gated by
    /// `option_onehot`. Each option's head only receives gradient
    /// when that option is active, silo'ing per-option capacity so
    /// different options can commit to genuinely different action
    /// distributions without fighting in one shared parameter space.
    /// Only effective when `num_options >= 2`; ignored otherwise.
    /// Default `true` — the additive-bias-only behaviour (Phase G v2
    /// through Tier-1) is kept as the `false` path for A/B.
    pub per_option_heads: bool,
    /// Sequence-level credit: discount factor for the n-step return
    /// used as the policy advantage baseline target. `0.0 ≤ γ < 1`.
    /// Only consulted when `n_step >= 2`. Default `0.95`.
    pub gamma: f32,
    /// Sequence-level credit: lookahead horizon for the policy
    /// advantage. `1` (default) preserves the pre-GAE single-step
    /// advantage (`r_t − V(s_t)`). `n ≥ 2` delays the policy update
    /// by `n−1` steps so it can fold in the next `n` rewards as a
    /// Monte-Carlo return:
    ///   R_t = Σ_{k=0}^{n-1} γ^k · r_{t+k}   (truncated at
    ///         `env_boundary`), normalized by Σ γ^k so the magnitude
    ///         stays comparable to a single-step reward.
    ///   advantage_t = clamp(R_t − V(s_t), -1, 1).
    /// The value head still trains on the single-step reward target
    /// (`lane.last_base_reward`) to avoid the bootstrap-instability
    /// failure mode that killed the earlier GAE attempt (commit
    /// 8f291e5). 4–8 is a reasonable range for LunarLander-scale
    /// episodes (100–300 env steps).
    pub n_step: usize,
    /// M6 learnable reward (see `docs/phase-m6-learnable-reward.md`):
    /// weight α on the outcome-value head's prediction when it's added
    /// as a fifth reward primitive — `r_t = r_base + α · R̂(z_t)`.
    /// `0.0` (default) disables M6; behaviour is byte-identical to the
    /// pre-M6 agent.
    pub outcome_reward_alpha: f32,
    /// M6: learning rate for the outcome-value head. `None` (default)
    /// resolves to `learning_rate × 0.3` — same scale as the credit head.
    pub lr_outcome: Option<f32>,
    /// M6: EMA rate for the running baseline used to center episode-
    /// return targets fed to the outcome head. Lower = smoother
    /// baseline, higher variance reduction but slower drift. Default
    /// `0.05`.
    pub outcome_baseline_ema: f32,
    /// M6: what target the outcome head trains against on episode
    /// completion. The M6 mechanism check (2026-04-19) showed that
    /// `EpisodeSum` — the original design — inherits the silence of
    /// kindle's four primitives on event-ordering: soft and crash
    /// LunarLander episodes produce nearly-identical cumulative
    /// returns, so the head learns an episode-length correlate
    /// instead of a quality signal. `TerminalReward` trains R̂ on
    /// just the *last* step's `r_base` instead, a different self-
    /// observable that can (and does, for v5 shaping) differ
    /// strongly between soft and crash if the reward circuit's
    /// terminal homeo profile already differs.
    pub outcome_target: OutcomeTarget,
    /// M6: how the head's output enters the per-step reward. See
    /// `OutcomeBonus`. `Raw` is the default; `PotentialDelta` is
    /// the Ng-et-al potential-based-shaping variant.
    pub outcome_bonus: OutcomeBonus,
    /// Symmetric clamp applied to the per-row policy advantage
    /// `(reward − value)` before it scales the action target. The
    /// historical default `1.0` bounds the cross-entropy gradient
    /// but also *destroys* advantage magnitude on steps where the
    /// reward actually differentiates actions (typical LunarLander
    /// mid-flight decisions, where r−V can be ±5 but gets clipped
    /// to ±1). Raise this when diagnosing a policy that fails to
    /// commit under default-regularized kindle.
    pub advantage_clamp: f32,
    /// Magnitude threshold above which the policy-loss watchdog
    /// re-initializes policy parameters (alongside the non-finite
    /// check). Default `1000.0` catches the "finite but runaway"
    /// regime that can follow a brief performance peak. Raise this
    /// to a very large value to effectively disable the watchdog
    /// when ablating whether its resets are preventing convergence.
    pub policy_loss_watchdog_threshold: f32,
    /// M7 approach-reward weight. When `> 0`, kindle maintains a
    /// single prototype centroid in latent space (the mean of the
    /// top-`approach_top_frac` fraction of recent terminal latents
    /// by episode return) and adds `−α · ‖z_t − centroid‖` to the
    /// per-step reward. Gives the reward a continuous
    /// approach-shaping signal that the homeo primitives cannot
    /// express. Zero (default) disables M7; byte-identical pre-M7.
    /// See `docs/phase-m7-approach-reward.md`.
    pub approach_reward_alpha: f32,
    /// M7 rolling buffer of (terminal_latent, episode_return) pairs.
    pub approach_buffer_size: usize,
    /// M7 fraction of highest-return terminals to average into the
    /// prototype. 0.2 = the top-20%.
    pub approach_top_frac: f32,
    /// M7 episodes between centroid recomputes. Lower = prototype
    /// tracks recent experience faster but jitters more; higher =
    /// stable but stale.
    pub approach_update_interval: usize,
    /// M7 warmup: no approach reward until this many completed
    /// episodes have been observed (so the prototype is built from
    /// a non-trivial distribution).
    pub approach_warmup_episodes: usize,
    /// M7 symmetric distance clamp (pre-α multiplication) to bound
    /// worst-case per-step bonus magnitude.
    pub approach_distance_clamp: f32,
    /// M7 confidence saturation: number of completed episodes
    /// required for the prototype to reach full confidence
    /// (`c = 1`). Confidence ramps linearly from `0` at
    /// `approach_warmup_episodes` up to `1` at
    /// `warmup + saturation` episodes. When ramping is active, the
    /// per-step approach bonus is scaled by `c` and the homeo
    /// reward is tapered by `homeo_confidence_taper · c`. `0`
    /// (default) disables ramping — `c = 1` always, behaviour
    /// identical to the non-confidence-aware M7 v1.
    pub approach_confidence_saturation: usize,
    /// M7↔homeo integration: fraction of the homeo reward to
    /// remove once `c = 1`. `homeo_effective = homeo_raw · (1 − τ · c)`.
    /// `0` (default) keeps homeo at full weight even when M7 is
    /// fully confident (M7 is purely additive). `0.5` halves the
    /// homeo contribution at full confidence so M7 can lead
    /// without homeo's misaligned basin dominating. `1.0` turns
    /// homeo off entirely at full confidence — M7 becomes the sole
    /// reward signal.
    pub homeo_confidence_taper: f32,
    /// M7 prototype-selection criterion. `Return` (default) ranks
    /// terminal entries by cumulative kindle intrinsic reward —
    /// fine when the env's high-return terminals are also task-
    /// success terminals (the 6 kindle-friendly envs, to varying
    /// strengths). `Novelty` ranks by `1 / sqrt(visit_count)` at
    /// the terminal latent, promoting rare terminals over common
    /// ones. On LunarLander this is expected to help (soft
    /// landings are rare, crashes are common); on CartPole etc.
    /// it would hurt (timeout-success is common, so novelty
    /// demotes it).
    pub approach_rank_by: ApproachRankBy,
    /// RND curiosity weight. When `> 0`, kindle builds a
    /// Random-Network-Distillation pair (frozen random target +
    /// trainable predictor, both MLPs on the agent's latent
    /// `z`) and adds `α · mse(predictor(z), target(z))` to the
    /// per-step reward. Zero (default) disables RND and skips
    /// both net construction and per-step training overhead.
    /// Unlike the surprise primitive, RND's prediction error
    /// doesn't decay as the world model converges, so it remains
    /// a live exploration signal throughout training — designed
    /// for the curiosity-death failure mode the ARC-AGI-3 sweep
    /// exposed (see commit 780c9a2).
    pub rnd_reward_alpha: f32,
    /// RND feature dim (target + predictor output size). Default 16.
    pub rnd_feature_dim: usize,
    /// RND hidden-layer width. Default 64.
    pub rnd_hidden_dim: usize,
    /// RND predictor learning rate. `None` → `learning_rate × 0.3`,
    /// matching the credit / outcome-head scale.
    pub rnd_lr: Option<f32>,
    /// Continuous coordinate-action head. `0.0` (default)
    /// disables it — kindle doesn't build the head and
    /// `sample_coords()` returns zeros. `α > 0` constructs a
    /// CPU MLP `z → (μ_x, μ_y) ∈ [−1, 1]` that the harness can
    /// sample from to pick spatial-click coordinates on envs
    /// that support them (ARC-AGI-3 complex actions). The head
    /// trains via REINFORCE on the last observed reward minus an EMA
    /// baseline; `α` scales the reinforcement magnitude. Harnesses should
    /// mask out lanes whose executed action did not consume coordinates.
    pub coord_action_alpha: f32,
    /// Hidden width for the coord head. Default 32.
    pub coord_hidden_dim: usize,
    /// Gaussian exploration noise stddev in the `[−1, 1]` action
    /// space. Default 0.3 — enough diversity to cover the coord
    /// space, narrow enough to leave signal in the mean.
    pub coord_sigma: f32,
    /// LR for the coord head's REINFORCE update. `None` →
    /// `learning_rate × 0.3`.
    pub coord_lr: Option<f32>,
    /// M8 delta-goal reward. When `> 0`, kindle maintains a bank of
    /// latent positions where a significant state change was just
    /// observed (`‖z_cur − z_prev‖ ≥ delta_goal_threshold`) and
    /// rewards the policy with `−α · min_i ‖z − g_i‖` (clamped by
    /// `delta_goal_distance_clamp`). Self-supervised: no task labels.
    /// Goal-bank stays diverse because candidates within
    /// `delta_goal_merge_radius` of an existing goal are dropped.
    /// Zero (default) disables M8 and skips construction.
    pub delta_goal_alpha: f32,
    /// M8 minimum per-step latent-delta to trigger a new goal entry.
    /// Below this, a step's `z_cur` is not recorded even if it lands
    /// in a novel region. Calibrate alongside `latent_dim`.
    pub delta_goal_threshold: f32,
    /// M8 merge radius: a candidate goal within this L2 distance of
    /// any existing bank entry is considered a duplicate and dropped.
    pub delta_goal_merge_radius: f32,
    /// M8 maximum bank size. Oldest entries are evicted first.
    pub delta_goal_bank_size: usize,
    /// M8 symmetric distance clamp (pre-α) to bound worst-case
    /// per-step reward magnitude.
    pub delta_goal_distance_clamp: f32,
    /// M8 v2: minimum world-model prediction error required to
    /// consider recording a new goal. Only transitions where
    /// `pred_error >= delta_goal_surprise_threshold` AND
    /// `‖Δobs‖ >= delta_goal_threshold` enter the bank. Banking on
    /// raw obs-deltas alone (v1) collects the agent's routine
    /// trajectory — no gradient toward new behaviour. Gating on
    /// WM surprise biases the bank toward transitions the WM
    /// didn't predict, i.e. genuinely unexpected events. `0.0`
    /// (treat as "no surprise gate") preserves v1 semantics.
    pub delta_goal_surprise_threshold: f32,
    /// Cross-episode state-action novelty weight. When `> 0`,
    /// kindle maintains a persistent `(quantized_state, action)`
    /// visit counter (shared across lanes, spanning episode
    /// boundaries) and emits `α / sqrt(1 + count)` as an intrinsic
    /// per-step reward keyed on the PREVIOUS step's (state, action)
    /// pair. Unlike the existing state-only novelty primitive,
    /// this discriminates between actions at the same state —
    /// targeting the "reach L1 every episode and retry the same
    /// actions there" failure mode observed on ARC-AGI-3. Zero
    /// (default) disables the memory and skips construction.
    pub xeps_reward_alpha: f32,
    /// Grid resolution for the cross-episode memory's state key.
    /// `None` → reuse `grid_resolution` (the state-novelty bucket
    /// size), keeping the two quantization schemes consistent.
    pub xeps_grid_resolution: Option<f32>,
    /// Extrinsic-reward weight. When `> 0`, the harness can supply
    /// a per-lane scalar via `Agent::set_extrinsic_reward(&[f32])`
    /// before each `observe()` call; kindle adds
    /// `α · extrinsic[i]` to the per-step reward used by credit
    /// and policy training. Unlike the homeo primitive — which
    /// always subtracts a positive deviation from a target and
    /// thus produces a one-sided signal — extrinsic reward is
    /// signed and passes through unchanged, so sparse ±1 signals
    /// (Atari, Gym classic control reward) reach policy gradient
    /// with the correct sign. Kindle's value head absorbs the DC
    /// offset automatically; the advantage sees the per-step
    /// variance. Zero (default) preserves the pre-primitive
    /// behaviour byte-for-byte.
    ///
    /// Note: using the extrinsic primitive couples kindle's policy
    /// to a task-specific reward signal and thus violates the
    /// cold-start self-training thesis. Prefer expressing goals
    /// via homeostatic variables when possible; this channel is
    /// primarily diagnostic (validating kindle's policy-gradient
    /// machinery on envs where the reward signal is known).
    pub extrinsic_reward_alpha: f32,
    /// Global advantage-norm clip applied across the batch before
    /// the policy update is built. When `> 0`, the L2 norm of the
    /// per-lane advantage vector is bounded: if
    /// `‖adv‖_2 > policy_adv_global_clip`, every lane's advantage
    /// is rescaled by `clip / ‖adv‖_2`. This is the policy-gradient
    /// analogue of global-grad-norm clipping — since each lane's
    /// contribution to the policy gradient is linear in its
    /// advantage, clipping advantages bounds the gradient norm.
    /// Complements per-lane `advantage_clamp` (which is an L∞
    /// bound). Zero (default) disables.
    pub policy_adv_global_clip: f32,
    /// Adaptive LR target for policy updates. When `> 0`, kindle
    /// maintains an EMA of `|pi_loss|` (smoothed by
    /// `policy_lr_adaptive_ema`) and scales the per-step policy
    /// learning rate by `target / max(ema, target)`. When the
    /// policy is taking huge update steps (loss magnitude far
    /// above the target), effective LR drops proportionally;
    /// when the policy has settled into a range below the target,
    /// full LR is used. Classic adaptive-step-size damping — the
    /// simpler cousin of KL-constrained TRPO. Zero (default)
    /// disables.
    pub policy_lr_adaptive_target: f32,
    /// EMA rate for the `|pi_loss|` tracking in adaptive LR.
    /// Default `0.05` (20-step effective window). Higher = more
    /// responsive but noisier.
    pub policy_lr_adaptive_ema: f32,
    /// TD-bootstrap the value-head target (only affects
    /// `policy_step_n_step`, i.e. when `n_step >= 2`).
    ///
    /// Default `false` — the value head trains on single-step
    /// reward at the ripe state, which on sparse-reward envs
    /// leaves V≈0 everywhere and the advantage signal is
    /// sparse-but-correctly-signed.
    ///
    /// When `true`: value target becomes the bootstrapped n-step
    /// return
    /// ```text
    /// V_target(s_ripe) = Σ_{k=0}^{n-1} γ^k r_{ripe+k}
    ///                  + γ^n · V(s_{ripe+n})
    /// ```
    /// (with bootstrap suppressed when the episode terminates
    /// inside the window or at the bootstrap point). This is the
    /// classical n-step TD target — rewards propagate backward
    /// through V via Bellman recursion, so every state with any
    /// causal connection to a future reward gets a dense
    /// non-zero V estimate and the advantage `ret - V(s_ripe)`
    /// carries gradient at every step, not just at reward events.
    ///
    /// Requires `buf_len >= n_step + 1` (one extra transition for
    /// the bootstrap). Lanes with shorter buffers fall back to
    /// bootstrap=0 (equivalent to treating them as episode-
    /// terminating, which is the safe conservative default).
    ///
    /// The bootstrap uses the STORED `τ_{ripe+n_step}.value` — a
    /// value prediction from the policy session at the time that
    /// transition was observed. This is a "target-network" style
    /// stale estimate, which is stable by construction (the
    /// bootstrap target doesn't change under the current update,
    /// avoiding the DQN-style divergence mode).
    pub value_bootstrap: bool,
    /// GAE (Generalized Advantage Estimation) λ parameter.
    /// `0.0` = disabled (advantage = `R_n − V(s_ripe)` as before).
    /// `(0, 1]` = enable:
    /// ```text
    /// Â_t = Σ_{k=0}^{n-1} (γλ)^k · δ_{t+k}
    /// δ_{t+k} = r_{t+k} + γ·V(s_{t+k+1}) - V(s_{t+k})
    /// ```
    /// Exponentially-weighted average over all n-step TD targets —
    /// λ=0 → pure 1-step TD (low variance, high bias),
    /// λ=1 → Monte-Carlo return − V (high variance, unbiased),
    /// λ≈0.95 is the PPO/A2C default.
    ///
    /// Why this matters over plain `value_bootstrap`: GAE decouples
    /// the *advantage* estimator from the *value target*. With only
    /// `value_bootstrap`, advantage = `R_n − V(s_ripe)` and V is
    /// trained to fit `R_n` — so as V gets accurate, advantage → 0
    /// and the policy-gradient signal dies. GAE advantages are
    /// TD-error-based, so they stay non-trivial (E[Â_t]=0 but
    /// Var[Â_t] > 0) even when V has converged.
    ///
    /// Enabling GAE also enables the bootstrap headroom (needs
    /// `V(s_{t+1})` for each per-step δ). The value target itself
    /// stays on the `value_bootstrap` path — set `value_bootstrap =
    /// true` alongside `gae_lambda > 0` for the standard A2C setup.
    pub gae_lambda: f32,
    /// Coefficient on the value-head MSE before it's summed with
    /// the policy loss. Standard PPO/A2C use `0.5`; defaults to
    /// `1.0` here for backward compatibility.
    ///
    /// With a shared optimizer, the combined-loss gradient is
    /// dominated by whichever head produces the larger loss
    /// magnitude. On dense-reward envs the value MSE is on the
    /// reward scale (potentially tens or hundreds), while the
    /// policy cross-entropy is on a log-scale (O(log K) ≈ 0.7 for
    /// CartPole). Leaving the value at 1.0 makes the policy
    /// effectively learn ~50× slower than the value. Setting this
    /// to 0.1–0.5 rebalances without separate LRs or sessions.
    pub value_loss_coef: f32,
    /// Soft-clamp range for the value head's output: V ∈ [-scale, +scale]
    /// via `scaled_tanh`. Default 200.0 fits CartPole / Acrobot returns.
    /// Pendulum returns are -1500..0; bump to 2000+ there. Returns are
    /// not normalized inside kindle, so per-env value-range tuning is
    /// the user's responsibility — the soft clamp is just a safety net
    /// against runaway value-MSE gradients destabilizing the encoder.
    pub value_clip_scale: f32,
    /// Symmetric clamp applied to V(s_{t+n}) inside the n-step bootstrap
    /// (`R_n + γ^n·clamp(V_next)`) and to V used as the GAE baseline.
    /// Independent of `value_clip_scale` — this is a *training-stability*
    /// clamp on stale stored V values, not the V-head's output range.
    /// Default 100.0 (kindle's pre-config-knob hardcoded value). Lower
    /// than `value_clip_scale` because stored V values from earlier
    /// updates can lag the current V-head significantly. For envs whose
    /// returns exceed ±100 (Pendulum), bump along with `value_clip_scale`
    /// — but expect post-solve crashes to grow more violent on dense-
    /// reward envs (CartPole regresses from peak +329 to peak +64 if
    /// you raise this to 200 unilaterally).
    pub bootstrap_value_clamp: f32,
    /// Symmetric clamp applied to the value-head TRAINING target (the
    /// n-step return / single-step reward fed into MSE against V(s_t)).
    /// Default 100.0 — the kindle pre-config-knob hardcoded value. Caps
    /// the magnitude of the V regression target so a runaway reward
    /// (e.g. a sparse +200 landing bonus or a very negative homeo
    /// deviation) cannot drive the value-MSE gradient into a regime
    /// that explodes the value-head weights.
    ///
    /// Per-env tuning: must be ≥ the absolute magnitude of any single-
    /// episode return the V-head should be able to predict end-to-end.
    /// LunarLander has terminal events of -100 (crash) and +200..+300
    /// (soft landing); leaving this at 100 clips the *positive* tail
    /// asymmetrically, biasing V to predict crashes well and landings
    /// poorly. Bump to 300 (or scale rewards Python-side) for that env.
    /// Pendulum has -1500..0 returns; bump to 1500.
    ///
    /// Independent of `value_clip_scale` (V-head output range) and
    /// `bootstrap_value_clamp` (stale V-readings clamp); those exist
    /// to bound *forward-pass* stability, this clamps the *backward
    /// target*.
    pub value_target_clamp: f32,
    /// Number of transitions before a positive-reward terminal that
    /// receive a retroactive curiosity bonus. Default 0 (disabled).
    /// When `terminal_proximity_k > 0`, the agent adds
    /// `terminal_proximity_bonus` to the `reward` field of each of the
    /// last `K` transitions in a lane's buffer at the moment
    /// `mark_boundary` is called — *if* the just-ended episode's terminal
    /// transition reward exceeds `terminal_proximity_threshold`.
    ///
    /// Mechanism: focuses kindle's exploration toward the boundary states
    /// that *immediately precede* successful terminals, without coupling
    /// to env-specific shaping. On LunarLander the policy learns to seek
    /// the few-step approach to a safe landing rather than treating all
    /// novel states as equally interesting.
    ///
    /// Asymmetric design: the bonus only fires for *positive*-reward
    /// terminals (above `terminal_proximity_threshold`). Crashes and
    /// other negative-reward terminals receive no bonus. This makes the
    /// reward an attractor toward the goal region rather than toward
    /// any-termination.
    ///
    /// Bonuses do not propagate across episode boundaries: the lookback
    /// stops at the previous `env_boundary` flag.
    pub terminal_proximity_k: usize,
    /// Per-step bonus added to each of the K transitions before a
    /// successful terminal. Default 0.0 (disabled). See
    /// `terminal_proximity_k`.
    pub terminal_proximity_bonus: f32,
    /// Reward threshold the just-ended episode's terminal transition
    /// must exceed for the proximity bonus to fire. Default 0.0
    /// (any positive-reward terminal). Set higher to require a clearly
    /// positive landing-style outcome (e.g. >5.0 for LunarLander where
    /// landing yields a +30 spike vs intrinsic rewards typically <5).
    pub terminal_proximity_threshold: f32,
    /// Reconstruction decoder anti-collapse loss coefficient. When > 0,
    /// the world-model graph builds a decoder `z → obs'` and adds
    /// MSE(obs', stop_grad(obs)) to the total loss. Forces the
    /// encoder to retain enough information about the raw observation
    /// that an inverse exists — the classical auto-encoder anti-
    /// collapse term. Default 1.0 because forward-dynamics loss targets a
    /// stopped latent and therefore cannot train the encoder by itself.
    pub recon_loss_coef: f32,
    /// When true (and the encoder is CNN/CnnDqn), the WM-session recon
    /// branch targets the raw `visual_obs` (reshaped to
    /// `[batch, c·h·w]`) instead of the OBS_TOKEN_DIM pooled `obs`.
    /// The pooled obs is rank ~8 globally / 1–2 per-game on static
    /// ARC frames — too low-rank to force higher-rank z. The raw
    /// frame is rank ~30+ globally, providing real anti-collapse
    /// pressure. Cost: decoder grows from `latent_dim → hidden_dim
    /// → OBS_TOKEN_DIM (64)` to `latent_dim → hidden_dim → c·h·w
    /// (4096 at 64×64×1)` — roughly +1M params at typical sizes.
    /// Ignored for MLP encoders. Default false.
    pub recon_visual_target: bool,
    /// When true, apply LayerNorm to z immediately before the policy
    /// (and value) head's first layer. Amplifies within-game state
    /// variation relative to between-game centroid offset — the
    /// failure mode diagnosed under 25-game multi-training where the
    /// encoder's between-game variance was 15× the within-game
    /// variance, drowning state info in the policy gradient. Default
    /// false. Currently wired in `build_ppo_policy_graph` only;
    /// other builders ignore the flag.
    pub policy_z_layer_norm: bool,
    /// Constant multiplier applied AFTER the policy-z LayerNorm,
    /// when `policy_z_layer_norm = true`. Default 1.0 leaves the
    /// raw normalized output (mean 0, std 1 per row). LayerNorm
    /// alone shrank z magnitude from ~40 to ~1, killing the
    /// policy's ability to produce committed logits under
    /// PPO clip + entropy_beta (verified 2026-05-06). Setting
    /// this to e.g. 10–40 recovers signal magnitude while keeping
    /// the per-dim equalization. Ignored when
    /// `policy_z_layer_norm = false`.
    pub policy_z_layer_norm_scale: f32,
    /// Reward-prediction-from-z anti-collapse loss coefficient. When
    /// non-zero (e2e only), the policy graph builds a head `z → r̂`
    /// and adds MSE(r̂, stop_grad(reward)) to the total loss. Adds
    /// input `reward_target`. Forces z to retain reward-predictive
    /// features (per-row single-step reward). Default 0.0.
    pub reward_pred_loss_coef: f32,
    /// Update the policy only every N env-steps, then do N
    /// gradient steps in a row on the accumulated rollout.
    /// Default `1` = per-env-step update (the existing behavior).
    ///
    /// Why this matters: kindle's per-step update means by the
    /// time a transition becomes "ripe" (n_step in the past), the
    /// policy has already taken n_step + 1 gradient steps since
    /// that transition was collected. The data is effectively
    /// off-policy from the very first update, which biases the
    /// policy-gradient estimator and produces the commit/uncommit
    /// oscillation seen on CartPole. Setting `policy_update_interval
    /// = n_step + 1` keeps the rollout fully on-policy with respect
    /// to the policy that collected it — the standard A2C/PPO
    /// setup. On the fire step, the agent does `interval` sequential
    /// gradient steps, each trained on the ripe transition at a
    /// different offset back through the rollout buffer.
    pub policy_update_interval: usize,
    /// Normalize advantages per-batch to zero mean / unit std before
    /// feeding them into the policy update. Standard PPO/A2C trick.
    /// Default `false` (backward compat).
    ///
    /// On many envs V lags the reward early on, so advantages are all
    /// same-sign (e.g. +1/+2 every step when reward is +1). Same-sign
    /// advantages push the policy toward whichever action a majority of
    /// lanes happened to take, regardless of whether that action is
    /// actually better — the advantage signal carries the *bias* from
    /// "V doesn't predict reward yet" rather than the *differential*
    /// information we actually need. Mean-centering strips the bias;
    /// variance-normalization fixes the gradient-magnitude scale.
    pub advantage_normalize: bool,
    /// Enable the PPO clipped-surrogate policy loss. When `true`, the
    /// agent uses `build_ppo_policy_graph` instead of the plain
    /// advantage-weighted CE path — see that function's docstring for
    /// the formula. The clipped ratio provides a mathematical trust
    /// region: once the policy has moved ε away from the data-
    /// collection policy on a given transition, the gradient through
    /// that transition drops to zero. This is what stabilizes committed
    /// policies and closes the commit/uncommit oscillation that
    /// advantage-normalized on-policy PG cannot escape on its own.
    /// Default `false`. Requires `policy_update_interval > 1` (at least
    /// some rollout window) for the ratio to be meaningfully ≠ 1 across
    /// inner-loop steps.
    pub use_ppo: bool,
    /// PPO clip radius ε. Standard value `0.2`. Ratio is clipped to
    /// `[1 − ε, 1 + ε]`.
    pub ppo_clip_eps: f32,
    /// Use the KL-penalty PPO graph variant (e2e only — non-e2e KL
    /// path not implemented). Mutually exclusive with `use_ppo`
    /// (which selects the clipped surrogate that has been documented
    /// as failing — see docs/failed_experiments.md). The KL variant
    /// uses plain-PG cross-entropy + β·KL(π_new ‖ π_old) for
    /// trust-region behavior with well-defined gradient everywhere
    /// (no dead clip zones). Requires `end_to_end_encoder = true`.
    pub use_kl_ppo: bool,
    /// KL penalty weight β for the use_kl_ppo path. Standard initial
    /// value 0.01-0.1. Adaptive scheduling left to the harness for
    /// now (set higher for stronger trust region / smaller updates).
    pub kl_beta: f32,
    /// When `use_kl_ppo` is on, switch from per-transition logits_at_action
    /// (kindle's default — keeps π_old very close to π_new because the
    /// per-step training cadence keeps drift tiny) to a frozen snapshot
    /// captured ONCE per K-epoch cycle. Standard PPO uses the latter.
    /// Default `false` (per-transition mode preserved).
    pub kl_use_snapshot: bool,
    /// Number of epochs to replay each rollout through the PPO
    /// update. On epoch 1 the ratio is ≈ 1 everywhere (policy
    /// hasn't moved since collection), so the clip does nothing —
    /// the update is identical to plain advantage-weighted PG. On
    /// epochs 2+, the policy has drifted from π_old, ratios diverge
    /// from 1, and the clip actually caps updates that would move
    /// the policy past the trust region. Standard PPO uses 3–10.
    /// Only has effect when `use_ppo = true`. Default 1 (equivalent
    /// to the no-clip baseline for single-epoch runs).
    pub ppo_n_epochs: usize,
    /// Enable Group-Relative Policy Optimization (GRPO). When `true`,
    /// drops the V-baseline and uses batch-relative statistics for
    /// advantage estimation. Two sub-modes controlled by
    /// `use_grpo_episode`:
    ///
    /// - per-step (default, `use_grpo_episode = false`): n-step return
    ///   without V bootstrap, normalized among lanes with the same env id.
    /// - per-episode (`use_grpo_episode = true`): each lane's most
    ///   recently completed episode return is used as the per-transition
    ///   advantage; within-env fork normalization gives the GRPO advantage.
    ///   Closer to the canonical DeepSeek-R1 formulation but introduces
    ///   inter-update lag (lane only contributes after first episode
    ///   completes).
    ///
    /// Combinations:
    /// - `use_grpo` alone: plain-PG e2e graph + GRPO advantage. No clip.
    /// - `use_grpo + use_ppo`: PPO clipped surrogate + GRPO advantage.
    ///   Standard GRPO from DeepSeek-R1.
    /// - `use_grpo + use_kl_ppo`: KL-penalty PPO + GRPO advantage.
    ///
    /// Requires `advantage_normalize = true` since the within-task
    /// normalization IS GRPO's core. `value_loss_coef = 0.0` is
    /// natural (V unused) but harmless if non-zero.
    pub use_grpo: bool,
    /// Per-episode variant of GRPO. When set with `use_grpo`, advantage
    /// is computed from each lane's most-recent completed-episode
    /// return G_i (sum of in-episode rewards), instead of n-step
    /// return. Normalization uses recent G's from the same env id.
    /// Default `false` (per-step GRPO).
    pub use_grpo_episode: bool,
    /// Enable Self-Imitation Learning (Oh et al. 2018). When `true`,
    /// the agent maintains a buffer of `(obs, action)` pairs from
    /// "successful" episodes (return > running EMA baseline). After
    /// each policy update, an additional supervised CE loss step
    /// runs on a random batch from this buffer, pulling the policy
    /// toward "actions you have taken that worked".
    ///
    /// Mechanism: addresses the on-policy A2C limitation that gradient
    /// updates only propagate through actually-sampled (s, a) pairs.
    /// A policy locked at a local optimum in some state-distribution
    /// (e.g., "always fire engine when on ground") never samples the
    /// alternative action there, so it never gets the gradient signal
    /// to escape. SIL forces the policy to imitate its own past
    /// successes, even at states it currently visits with different
    /// actions — provided the encoder generalizes between the two.
    ///
    /// Requires successful episodes to actually occur (sparse-reward
    /// problems with zero successes won't benefit). Compatible with
    /// plain PG, end-to-end, PPO, and GRPO policy graphs; KL-PPO is skipped
    /// because SIL samples do not retain its full old-logit snapshot.
    pub use_sil: bool,
    /// Multiplier on the clipped positive SIL advantage. Default 0.5;
    /// larger values pull the policy more aggressively toward replayed
    /// successes.
    pub sil_loss_coef: f32,
    /// When true, the SIL push gate is restricted to episodes that
    /// contained at least one task event (an explicit
    /// `set_extrinsic_events` mark or a positive extrinsic reward).
    /// Combined with `extrinsic_reward_alpha > 0` this turns the
    /// SIL buffer into a "winning trajectories" replay rather than
    /// a generic above-baseline-return replay. Its quality baseline compares
    /// event episodes only, and every retained event transition has at least
    /// unit pre-coefficient imitation advantage so negative step costs cannot
    /// remove the successful prefix. Default false.
    pub sil_event_filter: bool,
    /// When greater than zero, event-filtered SIL immediately retains at most
    /// this many transitions ending at each positive task event instead of
    /// waiting for and replaying the whole environment episode. This is useful
    /// for continuing games whose meaningful credit boundary is a point or
    /// subgoal rather than game-over (for example Pong). The window never
    /// crosses a real environment boundary. Default 0 keeps episode-level SIL.
    pub sil_event_horizon: usize,
    /// Capacity of the SIL replay buffer. Older samples are evicted
    /// FIFO when full. Default 10000.
    pub sil_buffer_capacity: usize,
    /// EMA decay rate for the "successful episode" baseline. The
    /// baseline tracks recent eligible episode returns; an episode qualifies
    /// for SIL push if its return strictly exceeds the current
    /// baseline. Default 0.99 (≈ 100-episode horizon). Lower values
    /// make the baseline more reactive (fewer episodes qualify but
    /// they are more recently-best).
    pub sil_baseline_decay: f32,
    /// Zero all advantages (and therefore the policy-gradient
    /// signal) for the first `policy_warmup_steps` env-steps.
    /// The value head still trains — only the policy-gradient
    /// portion of the combined loss is silenced. Gives V a head
    /// start so that when the policy starts training the advantages
    /// are grounded in a reasonably accurate baseline rather than
    /// random-init V ≈ 0 (which on dense-reward envs produces
    /// uniformly positive advantages for every state, and the
    /// policy commits to random-majority actions).
    ///
    /// Default 0 (disabled). Try 2000–10000 on dense-reward envs.
    pub policy_warmup_steps: usize,
    /// Use an end-to-end policy graph: encoder + policy + value all
    /// in one session, trained on the combined loss. The encoder
    /// receives gradient from BOTH the value MSE and the policy CE.
    /// Default `false` (use the original z-as-input path where the
    /// encoder is trained ONLY by the WM loss in `wm_session`).
    ///
    /// Empirically isolated as the dominant cause of kindle's
    /// CartPole +24 ceiling: the standard "encoder trained by WM
    /// only" structure produces features optimized for next-state
    /// prediction, not for control discrimination. Pure-numpy A2C
    /// with end-to-end encoder gradient hits +76 on the same env;
    /// kindle's A2C-mimic without this flag stays at +11. The
    /// difference is the encoder gradient flow.
    ///
    /// When enabled, the policy session takes raw `obs` + `task`
    /// inputs (instead of `z`) and computes its own latent via a
    /// dedicated encoder copy (`policy_encoder.*` weights). The
    /// wm_session encoder continues to exist and train independently
    /// — its z is still stored in `Transition.latent` for use by
    /// other modules (credit, etc.) that depend on the WM-shaped
    /// representation. Discrete-policy non-options non-PPO only;
    /// other graph variants need their own e2e implementation.
    pub end_to_end_encoder: bool,
    /// Use orthogonal one-hot task codes for env ids below `TASK_DIM`.
    /// This gives common tasks isolated columns in the trainable task
    /// projection and improves sequential adaptation. Default `false` keeps
    /// the historical dense-hash codes so existing checkpoints retain their
    /// input semantics. Ids beyond `TASK_DIM` use normalized dense hashes.
    pub orthogonal_task_codes: bool,
    /// Recompute V on `ripe.latent` at policy-training time rather
    /// than using the `ripe.value` stored when the action was taken.
    /// Default `false` (use stored).
    ///
    /// Why this matters: kindle's encoder is updated every env-step
    /// by the WM loss, so by the time a transition becomes "ripe"
    /// (n_step env-steps in the past), the latent representation has
    /// shifted. The stored `ripe.value` was computed from the act-time
    /// latent under the act-time value head, both of which differ
    /// from now. Recomputing V on the stored ripe.latent (with current
    /// value head) gives an advantage signal that's consistent with
    /// what the current policy "sees" — closer to what standard A2C
    /// does (it computes V freshly for every rollout sample).
    ///
    /// Costs an extra forward-only `policy_session.step()` per
    /// training call. Only affects the non-GAE n-step path; GAE's
    /// advantage uses multiple stored V's that aren't all worth
    /// recomputing without restructuring.
    pub recompute_base_v: bool,
    /// A2C/PPO rollout buffer length. When `> 1`:
    ///   - The policy graph is built with
    ///     `batch_size = lanes × rollout_length`, so each policy
    ///     `session.step()` sees `rollout_length` temporal offsets
    ///     of every lane at once.
    ///   - The policy update fires every `rollout_length` env-steps
    ///     (supersedes `policy_update_interval`). One gradient step
    ///     per fire, or `ppo_n_epochs` steps under PPO.
    ///
    /// The low-variance mean gradient estimate that a big-batch
    /// update produces is precisely what lets the PPO clip fire on
    /// genuine trust-region excursions rather than on per-lane
    /// noise. Standard A2C: 5–128. Standard PPO: 128–2048 (with
    /// minibatching). On CartPole-scale envs, 16–64 typically
    /// suffice. Default 1 (keeps per-env-step update path identical
    /// to the pre-refactor code).
    pub rollout_length: usize,
    /// Model-based planner. When `> 0`, kindle maintains a CPU
    /// copy of the world-model weights and can simulate candidate
    /// action sequences via `plan_and_queue()` — sampling
    /// `planner_samples` random sequences of length `planner_horizon`,
    /// rolling each out through the frozen WM, scoring by sum of
    /// `1/sqrt(1+visit_count)` over the predicted latents, and
    /// queueing the best sequence for subsequent `act()` calls to consume.
    /// Slots declared by `set_action_parameter_masks()` are searched jointly
    /// with normalized `(x, y)` values. Disabled (0) by default. Applies to
    /// discrete and discrete-plus-parameter spaces; fully continuous policies
    /// are skipped.
    pub planner_horizon: usize,
    /// Number of random action sequences sampled per plan call.
    /// Cost scales O(samples × horizon × hidden²). Default 32.
    pub planner_samples: usize,
    /// Phase-1 option-aware planner: at each of the `planner_horizon`
    /// outer steps, the sampled action is REPEATED this many times in
    /// the WM rollout (and committed `repeat` times to planner_queue).
    /// Effective atomic-action reach per planner call = horizon ×
    /// repeat. Default 1 (no repeat, original behavior). ARC-AGI-3
    /// puzzles often require repeated moves (up-up-up-...), so this
    /// is a cheap reach multiplier.
    pub planner_action_repeat: usize,
    /// k-step WM training: when > 1, builds a separate
    /// `wm_kstep_session` that samples (t, t+k) tuples from the lane
    /// buffer, applies WM k times iteratively, and trains MSE against
    /// the actual z_{t+k}. Forces WM accuracy at depth k, not just
    /// 1-step. Direct fix for planner-rollout compounding error.
    /// Default 1 (disabled, only 1-step training as before).
    pub wm_kstep_k: usize,
    /// k-step batch size per training step. Default 32.
    pub wm_kstep_batch: usize,
    /// k-step training rate: probability per observe() of running a
    /// k-step training step. 0.0 disables; 1.0 = every observe.
    pub wm_kstep_train_prob: f32,
    /// Loss coefficient for the k-step WM loss in wm_kstep_session.
    pub wm_kstep_loss_coef: f32,
    /// Steps between WM-weight refreshes into the planner's CPU
    /// cache. Too frequent wastes cycles; too infrequent uses
    /// stale weights. Default 200.
    pub planner_refresh_interval: usize,
    /// Mix coefficient for the planner's action sampling. 0.0 = pure
    /// uniform-random (the default and what plain shooting MPC uses).
    /// 1.0 = pure policy-guided (sample sequences from the current
    /// policy's logits at each rollout step). Values in between mix:
    /// the planner samples uniformly with probability `1 - x` and from
    /// policy with probability `x`, per rollout step.
    ///
    /// Pure policy-guided (1.0) concentrates exploration on what the
    /// policy already prefers — better for fine-tuning a converged
    /// policy, worse for cold-start rare-event discovery (kindle's
    /// puzzle-game regime). Pure random (0.0) is better when the
    /// policy hasn't yet found any events. Mixing in the middle gives
    /// a smooth transition. Default 0.0 = pure random shooting.
    pub planner_policy_mix: f32,
    /// Temperature applied to policy logits in the planner's
    /// per-step sampling when `planner_policy_mix > 0`. T = 1.0 (default)
    /// samples from the policy's native distribution. T > 1 flattens
    /// (more exploratory); T < 1 sharpens. Useful counterweight to a
    /// peaked policy: T = 2-3 keeps planner exploration broad even
    /// when the policy has committed.
    pub planner_policy_temperature: f32,
    /// Use MCTS tree search instead of CEM/random-shooting MPC for ordinary
    /// discrete action spaces. When `true`, `plan_and_queue` builds a per-lane Monte
    /// Carlo tree, running `mcts_simulations` expansions before reading
    /// off the most-visited action sequence from the root.
    ///
    /// Tradeoff vs random shooting: MCTS shares ancestor paths across
    /// rollouts and uses UCB1 to focus on promising subtrees. Better
    /// for cases where SOME signal can be propagated up the tree (i.e.,
    /// when the WM and novelty score actually discriminate trajectories).
    /// In the kindle setting where `visit_count ≈ 1` makes novelty
    /// near-uniform, MCTS degenerates to depth-first exploration —
    /// similar to random shooting in expectation but with lower effective
    /// branching. Parameterized spaces automatically use shooting because the
    /// current tree cannot compare multiple coordinates for one identity.
    pub planner_use_mcts: bool,
    /// Number of MCTS simulations per planning call (per lane). Each
    /// simulation walks the tree from root to a leaf, expands one new
    /// child via WM forward, scores by novelty, and backs up. Cost
    /// scales O(mcts_simulations × n_lanes × hidden²) so keep it
    /// modest (default 64).
    pub mcts_simulations: usize,
    /// UCB1 exploration constant. `Q + c*sqrt(ln N_parent / n_child)`.
    /// Default sqrt(2) ≈ 1.414 (textbook).
    pub mcts_c_puct: f32,
    /// GRAM-style stochastic WM rollout: σ for Gaussian noise added to
    /// every `wm_planner` step's `z_next` output. Tests the GRAM
    /// (Generative Recursive Reasoning Models, 2026-05) hypothesis that
    /// multi-trajectory rollouts find narrow win paths a deterministic
    /// rollout misses. Each of the `planner_samples` rollouts per lane
    /// gets independent N(0, σ²I) noise per step, so trajectories
    /// diverge by both action choice AND latent perturbation. Default
    /// 0.0 (deterministic, byte-parity with pre-GRAM behavior).
    /// Reasonable range: 0.01–0.2 in z-space. Higher σ = more
    /// exploration; very high σ degenerates to random trajectories.
    /// When combined with `wm_stochastic`, this acts as a multiplier
    /// on the learned per-state σ: noise = planner_noise_sigma ×
    /// σ_learned × ε.
    pub planner_noise_sigma: f32,
    /// Surprise-triggered replanning (P4, 2026-06-10): when > 0 and a
    /// lane has queued planner actions, a step whose WM prediction
    /// error exceeds `replan_surprise_mult x` the lane's running EMA
    /// clears the lane's planner queue — the world diverged from the
    /// plan's assumptions, so stop executing it blind and replan at
    /// the next planner tick. 0 disables (default). Typical 2.0-3.0.
    pub replan_surprise_mult: f32,
    /// (P3) Information-seeking planner bonus: add `alpha · mean σ(z, a)`
    /// per rollout step to the trajectory score. With the learned
    /// heteroscedastic σ head (`wm_stochastic`), high σ marks
    /// transitions the WM cannot yet predict — typically the NEW
    /// game elements at an unsolved level. Steering rollouts toward
    /// them is directed epistemic exploration ("go poke the thing
    /// you don't understand"). 0 disables (default). Requires
    /// `wm_stochastic`.
    /// (P2) Surprise frame-replay ring capacity for CNN encoder modes.
    /// Each observe() admits rows whose WM prediction error exceeds
    /// `surprise_replay_mult x` the lane's error EMA, storing the
    /// post-action FRAME (visual slot row), the action, the acted-from
    /// latent, the obs token, and the env id. A replay dispatch then
    /// re-trains encoder+WM on a priority-sampled batch of these
    /// surprising moments with probability `replay_ratio` per step —
    /// the CNN-mode counterpart of `replay_step` (which skips CNN
    /// because transitions don't store frames), and the mechanism
    /// that concentrates representation learning on novel elements
    /// the moment they appear. ~33 KB per entry at 2x64x64. 0
    /// disables (default).
    /// (P5) Plasticity maintenance interval (steps). Every interval,
    /// the encoder / world-model / recon parameters of the wm session
    /// get a shrink-and-perturb update: p <- shrink·p + noise·std(p)·ε.
    /// Counters the loss of plasticity that hits long-running nets
    /// right when novelty arrives late in training (new elements at a
    /// fresh level) — see the 2026-05-05 rank-collapse episode. The
    /// perturbation is scale-aware (per-tensor std) so it nudges
    /// without erasing. 0 disables (default). Typical 25_000-100_000.
    pub plasticity_interval: usize,
    /// (P5) Multiplicative shrink applied at each plasticity event.
    /// Default 0.999 (gentle).
    pub plasticity_shrink: f32,
    /// (P5) Noise scale as a fraction of each tensor's std.
    /// Default 0.01.
    pub plasticity_noise: f32,
    pub surprise_ring_capacity: usize,
    /// (P2) Admission threshold: a row enters the surprise ring when
    /// its pred_error > mult x lane EMA. 1.0 admits anything above
    /// average; 2.0 only clear spikes. Default 1.5.
    pub surprise_replay_mult: f32,
    pub planner_sigma_alpha: f32,
    /// (P3) σ-budgeted adaptive horizon: per rollout, accumulate the
    /// per-step mean σ; once the running sum exceeds this budget,
    /// stop trusting (scoring) the remainder of the trajectory and
    /// queue only the trusted prefix. Converts the fixed-horizon
    /// commitment problem (L3 wall: WM accuracy at depth > 10) into
    /// an uncertainty-adaptive one: confident rollouts commit deep,
    /// uncertain ones replan early. 0 disables (default). σ is the
    /// sigmoid-bounded head output, so each step contributes (0, 1);
    /// a budget of ~2-3 trusts a handful of uncertain steps or many
    /// confident ones. Requires `wm_stochastic`.
    pub planner_sigma_horizon: f32,
    /// GRAM-style heteroscedastic WM: when true, the WM additionally
    /// learns a per-state per-dim σ head (`world_model.sigma_proj`)
    /// trained to predict `|z_target − z_hat|`. The planner reads σ at
    /// rollout time and scales the noise per element (high-σ states get
    /// more exploration; well-modeled states stay near-deterministic).
    /// Requires `planner_noise_sigma > 0` to have any planner-side
    /// effect (the σ head is still trained but unused). Default false
    /// (v1 fixed-σ behavior). 2026-05-29.
    pub wm_stochastic: bool,
    /// Loss coefficient on the σ-regression term added to the WM loss.
    /// Only applied when `wm_stochastic = true`. Default 0.5.
    pub wm_sigma_loss_coef: f32,
    /// Goal-conditioned planning. When > 0, the planner adds a goal-
    /// similarity term to its trajectory scoring:
    ///
    ///   score(z̃) = novelty(z̃) + α · max_k cos_sim(z̃, win_state_k)
    ///
    /// where `win_state_k` are the latent states the agent visited at
    /// extrinsic-reward events (level completions). The agent emergently
    /// discovers its own goal region in latent space from successful
    /// experience, and the planner navigates toward it using WM rollouts.
    ///
    /// Without this term, kindle's planner only seeks novelty — it has no
    /// concept of "the goal." Adding it turns the planner into something
    /// closer to what humans do: build a mental model of where the goal
    /// is, then plan paths there using the world model.
    ///
    /// 0 (default) = pure novelty (previous behavior). 1.0 puts goal
    /// similarity on roughly equal footing with novelty. Higher values
    /// commit more aggressively to the goal region.
    pub planner_goal_alpha: f32,
    /// Per-env capacity of the win-state archive used for goal similarity.
    /// When full, FIFO eviction. Default 100 (small footprint: 100 × 256
    /// f32 ≈ 100 KB per env).
    pub goal_states_cap: usize,
    /// HER-style relabeling probability: when an episode ends with zero
    /// extrinsic events (a "loss"), push its terminal latent into the
    /// env's `goal_states` queue with this probability. 0 (default) =
    /// disabled. Useful early in training when no real wins exist yet
    /// — the planner gets a structural prior toward "places we've
    /// reached." Real wins evict synthetics from the FIFO over time.
    pub goal_states_her_prob: f32,
    /// BC-from-planner: synthetic R_to_go pushed into `sil_buffer` for
    /// each planner-committed first action. When > 0, the policy is
    /// pulled toward cloning the planner's choices via SIL's existing
    /// update mechanism. Closes the policy-planner gap so the agent
    /// can execute discovered paths without re-planning each time.
    ///
    /// Reasonable values: 0.3-1.0. Higher = stronger imitation. 0
    /// (default) = disabled. Pushes obs from `lane.buffer.last()` so
    /// the (s, a) pair is consistent with what the planner saw.
    pub bc_planner_synthetic_r: f32,
    /// Value head training: scales the value-head MSE loss. When > 0,
    /// kindle builds a separate `value_session` that learns `V(z) →
    /// expected discounted return-to-go`, trained on every completed
    /// episode's transitions (losses contribute `R ≈ 0` and shape the
    /// baseline; wins contribute `R = γ^(T-t)·reward_end` and shape the
    /// peaks). The value head is consumed by the planner via
    /// `planner_value_alpha`. 0 (default) = disabled.
    pub value_head_train_coef: f32,
    /// Planner integration: when > 0, the planner adds `α · V(z_step)`
    /// to each trajectory's score, biasing rollouts toward latents the
    /// value head has learned are close to past wins. Independent of
    /// `value_head_train_coef` (you can train V without consuming it,
    /// or skip training and use a pre-loaded V). 0 (default) = disabled.
    pub planner_value_alpha: f32,
    /// Affordance / state-change planner bonus: when > 0, each trajectory
    /// step contributes `α · ||z_step − z_prev||` (L2 norm of predicted
    /// state change). Prefers planner sequences whose WM rollouts cause
    /// LARGE per-step latent change.
    ///
    /// Rationale (vertical/horizontal transfer): the WM is trained on
    /// transitions from ALL games and ALL levels, so its prediction of
    /// "what change does this action cause from this z" implicitly encodes
    /// affordances that are SHARED across levels and games. At a novel
    /// level (e.g. tu93 L3) where state-specific signals (visit count,
    /// goal cos-sim, win classifier) have no data, the affordance signal
    /// still picks out actions that DO SOMETHING — driving exploration
    /// toward mechanically-active regions instead of staring at walls.
    ///
    /// 0 (default) = disabled. Reasonable values: 0.1–1.0. Computed
    /// per-step in `plan_and_queue_gpu` using the existing planner
    /// trajectory; no new graph or buffer required.
    pub planner_change_alpha: f32,
    /// Discount factor for value-head return-to-go targets. 0.99 default.
    /// Lower (0.95) makes V more local; higher (0.995) propagates win
    /// signal farther backward along the trajectory.
    pub value_head_gamma: f32,
    /// Replay-buffer capacity for value-head training samples (latent,
    /// R_to_go). FIFO eviction. Default 10_000.
    pub value_head_buffer_capacity: usize,
    /// Hidden dim for the value head's MLP. 0 = use `hidden_dim`.
    pub value_head_hidden_dim: usize,
    /// LR scale applied to the value-head optimizer step relative to
    /// the agent's base `learning_rate`. 1.0 = same; lower stabilizes
    /// V early in training when targets are noisy.
    pub value_head_lr_scale: f32,
    /// Cross-game value classifier mode. When `false` (default),
    /// `value_replay_step` samples round-robin across env_id buckets
    /// (game-stratified) — each game contributes equally regardless
    /// of how many wins it produced. Prevents the "sp80 monopolizes
    /// the gradient" failure mode from joint training (2026-05-15).
    ///
    /// When `true`, the sampler ignores the per-env stratification:
    /// every push pools into env_id 0 buckets and sampling draws
    /// uniformly from the pooled queue. The classifier is trained
    /// on the UNION of all games' wins/losses → learns features that
    /// are common to "winning" across games. Useful for horizontal
    /// transfer (game→game): a never-seen game still gets a non-zero
    /// classifier signal from latents that look like generic wins.
    ///
    /// Trade-off: cross-game mode forgoes the bias fix. Best paired
    /// with games that produce roughly comparable win rates, OR with
    /// a fresh game where stratified mode would have zero data.
    pub value_buffer_cross_game: bool,
    /// Cross-game goal-state pool. When `false` (default), each env_id
    /// has its own goal_states queue and the planner only matches
    /// trajectories against its OWN env_id's wins. When `true`, all
    /// wins from all envs pool into a single shared queue (key=0) and
    /// every lane's planner matches against the shared pool. Lets a
    /// novel game get planner-goal pull from other games' wins —
    /// useful when shared encoder produces analogous latents for
    /// analogous winning configurations across games.
    ///
    /// Pairs with `value_buffer_cross_game`: same idea, applied to
    /// the planner_goal_alpha cos-sim instead of the classifier.
    pub goal_states_cross_game: bool,
    /// Confidence-weighted planner mode. When `true`, the planner
    /// scoring loop weights exploit terms (goal_alpha, value_alpha)
    /// by per-lane confidence C, and explore terms (change and model
    /// uncertainty) by (1 − C). visit_count remains as an always-on
    /// base novelty signal.
    ///
    /// C is per-lane, starts at 0.5, and updates each observe() step:
    /// - rises by `confidence_win_increment` on each extrinsic event
    ///   (sil_ep_event_count incrementing by 1)
    /// - falls by the constant `confidence_novelty_drop_rate` each step
    ///
    /// The asymmetry ("rises only on winning") keeps the planner
    /// in explore-mode at frontier states (newly reached levels)
    /// where the WM/value head haven't validated their predictions
    /// with real wins. This is the proposed unlock for L2→L3:
    /// when restored to L2 archive states, encountering novel L2
    /// dynamics drops C → planner shifts to exploration → may
    /// stumble on L3 transition (the same mechanism that found L2
    /// from L1 via vanilla change_alpha).
    pub confidence_mode: bool,
    /// Per-event confidence increment. Default 0.02 (50 events to
    /// reach saturation from 0).
    pub confidence_win_increment: f32,
    /// Per-step confidence decay (constant subtraction from C each
    /// observe step). Models "competence atrophies without validation":
    /// if no wins occur, C drifts toward 0 → planner shifts to
    /// exploration. Pairs with win_increment to set steady-state C.
    ///
    /// Steady state: C_eq ≈ win_inc * events_per_step / drop_rate.
    /// Default 0.0001/step (a slow decay biased toward exploit; raise
    /// to ~0.001 to target C ≈ 0.4 at tu93's typical event rate).
    /// Lower drop_rate → more exploit; higher → more explore.
    pub confidence_novelty_drop_rate: f32,
    /// Sub-goal centroid count for online k-means clustering of the
    /// `goal_states` queue. 0 (default) = disabled. When > 0, the
    /// agent maintains K centroid vectors per env (or pooled when
    /// `goal_states_cross_game=true`); each new win latent is pulled
    /// toward its nearest centroid by `subgoal_lr`.
    ///
    /// Used by the planner via `planner_subgoal_alpha`. The centroids
    /// are an ABSTRACTION over raw win states — they represent
    /// regions ("a-winning-state-looks-like-this") rather than
    /// specific past wins. Better vertical transfer (L1→L2 within a
    /// game) and horizontal transfer (game→game) because the
    /// abstraction averages out instance-specific details.
    pub subgoal_k: usize,
    /// Online k-means learning rate for sub-goal centroids. Each new
    /// win latent: centroid = (1-lr) * centroid + lr * z. Default
    /// 0.05 (slow) — high lr causes centroids to chase the most
    /// recent wins; low lr averages over many.
    pub subgoal_lr: f32,
    /// Planner integration for sub-goal centroids: when > 0, the
    /// planner adds `α · max_k cos_sim(z_step, centroid_k)` to its
    /// trajectory score. 0 (default) = disabled.
    ///
    /// Use ALONGSIDE `planner_goal_alpha` (which uses raw win
    /// states), not as a replacement: centroids are abstract but
    /// noisy; raw points are specific. Both signals add.
    pub planner_subgoal_alpha: f32,
    /// WM encoder backbone. `Mlp` (default) = kindle's original
    /// obs-token encoder; `Cnn { channels, height, width }` =
    /// conv-net encoder for visual/grid inputs (ARC-AGI-3 etc.);
    /// `EfficientNetV2S` = frozen ImageNet-pretrained V2-S features
    /// followed by the same internal CNN encoder.
    /// See `EncoderKind` doc for the protocol.
    pub encoder_kind: EncoderKind,
    /// Required when `encoder_kind == EfficientNetV2S`. Path to the
    /// BN-folded V2-S safetensors produced by meganeura's
    /// `bench/dump_efficientnet_v2_reference.py`. Ignored otherwise.
    pub efficientnet_weights_path: Option<std::path::PathBuf>,
    /// M6 v2: window size for the outcome head's input. `1`
    /// (default) reduces to single-frame `R̂(z_t)` — the back-compat
    /// M6 v1 path. `k ≥ 2` concatenates the last `k` encoder
    /// latents `[z_{t-k+1}, ..., z_t]` so the head can read
    /// trajectory momentum, not just present-state. The M6 v1
    /// mechanism check (2026-04-19) showed that single-frame inputs
    /// can't discriminate soft from crash mid-flight because two
    /// identical mid-flight z's can precede different outcomes — a
    /// windowed input gets a richer condition.
    pub outcome_window: usize,
    /// M6: symmetric clamp applied to the raw outcome-head output
    /// before multiplication by `α`. Caps the worst-case per-step
    /// bonus at `α · outcome_clamp`. Default `5.0`; raise alongside
    /// `α` when probing whether the learned signal is correct-but-
    /// too-quiet relative to the ~3–15/step homeostatic penalty.
    pub outcome_clamp: f32,
    /// M6: hard cap on the number of trajectory latents kept per
    /// episode for the batched end-of-episode backward pass. Must be
    /// large enough to contain a typical episode for the target env;
    /// episodes longer than this get their tail silently truncated
    /// (RewardToGo targets then under-count returns for the truncated
    /// steps). Default `256` — covers LunarLander episodes (100–300
    /// steps). Also used as the outcome-head's compiled batch size, so
    /// changing this recompiles the graph.
    pub outcome_max_episode_len: usize,
    /// Phase G Tier-3: EMA rate for the continuous goal-latent
    /// update. At every option termination, the terminated option's
    /// goal vector is pulled toward the observed end-state latent:
    ///   `goal[o] ← (1 − β) · goal[o] + β · z_end`.
    /// This turns the fixed-table orthogonal anchors (from
    /// `option::build_goal_table`) into learned prototypes that
    /// track where L0 actually ends up under each option, making the
    /// goal-alignment bonus a self-consistency signal rather than an
    /// arbitrary pull toward a latent axis. `0.0` (the default here
    /// is `0.02`) disables the update, giving back the pre-Tier-3
    /// fixed-table behaviour.
    pub goal_ema_rate: f32,
    pub opt_level: OptLevel,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            hidden_dim: 32,
            buffer_capacity: 10_000,
            batch_size: 1,
            learning_rate: 1e-3,
            lr_policy: 5e-4, // 0.5× base
            reward_weights: RewardWeights::default(),
            warmup_steps: 100,
            replay_ratio: 0.2,
            grid_resolution: 0.5,
            visit_counts_max: 0,
            visit_count_dims: 0,
            visit_count_proj_dim: 0,
            visit_count_proj_seed: 0x9E37_79B9_7F4A_7C15,
            entropy_beta: 0.01,
            entropy_floor: 0.1,
            drift_interval: 500,
            drift_threshold: 1.0,
            action_repeat: 1,
            label_smoothing: 0.0,
            num_options: 1,
            option_dim: 0, // 0 = use latent_dim
            option_horizon: 10,
            option_entropy_beta: 0.0,
            diayn_reward_alpha: 0.0,
            diayn_hidden_dim: 32,
            diayn_lr: None,
            lr_option: 2.5e-4,
            goal_bonus_alpha: 0.1,
            learned_termination: false,
            per_option_heads: true,
            gamma: 0.95,
            n_step: 1,
            outcome_reward_alpha: 0.0,
            lr_outcome: None,
            outcome_baseline_ema: 0.05,
            outcome_target: OutcomeTarget::EpisodeSum,
            outcome_bonus: OutcomeBonus::Raw,
            advantage_clamp: 1.0,
            policy_loss_watchdog_threshold: 1000.0,
            approach_reward_alpha: 0.0,
            approach_buffer_size: 100,
            approach_top_frac: 0.2,
            approach_update_interval: 10,
            approach_warmup_episodes: 20,
            approach_distance_clamp: 10.0,
            approach_confidence_saturation: 0,
            homeo_confidence_taper: 0.0,
            approach_rank_by: ApproachRankBy::Return,
            rnd_reward_alpha: 0.0,
            rnd_feature_dim: 16,
            rnd_hidden_dim: 64,
            rnd_lr: None,
            coord_action_alpha: 0.0,
            coord_hidden_dim: 32,
            coord_sigma: 0.3,
            coord_lr: None,
            delta_goal_alpha: 0.0,
            delta_goal_threshold: 0.5,
            delta_goal_merge_radius: 0.1,
            delta_goal_bank_size: 64,
            delta_goal_distance_clamp: 5.0,
            delta_goal_surprise_threshold: 0.5,
            xeps_reward_alpha: 0.0,
            xeps_grid_resolution: None,
            extrinsic_reward_alpha: 0.0,
            policy_adv_global_clip: 0.0,
            policy_lr_adaptive_target: 0.0,
            policy_lr_adaptive_ema: 0.05,
            value_bootstrap: false,
            gae_lambda: 0.0,
            use_adam: false,
            adam_eps: 1e-8,
            grad_clip_norm: 0.0,
            grad_clip_every: 1,
            value_loss_coef: 1.0,
            value_clip_scale: 200.0,
            bootstrap_value_clamp: 100.0,
            value_target_clamp: 100.0,
            terminal_proximity_k: 0,
            terminal_proximity_bonus: 0.0,
            terminal_proximity_threshold: 0.0,
            recon_loss_coef: 1.0,
            recon_visual_target: false,
            policy_z_layer_norm: false,
            policy_z_layer_norm_scale: 1.0,
            reward_pred_loss_coef: 0.0,
            policy_update_interval: 1,
            advantage_normalize: false,
            use_ppo: false,
            ppo_clip_eps: 0.2,
            use_grpo: false,
            use_grpo_episode: false,
            use_sil: false,
            sil_loss_coef: 0.5,
            sil_event_filter: false,
            sil_event_horizon: 0,
            sil_buffer_capacity: 10_000,
            sil_baseline_decay: 0.99,
            use_kl_ppo: false,
            kl_beta: 0.05,
            kl_use_snapshot: false,
            ppo_n_epochs: 1,
            policy_warmup_steps: 0,
            recompute_base_v: false,
            end_to_end_encoder: false,
            orthogonal_task_codes: false,
            rollout_length: 1,
            planner_horizon: 0,
            planner_samples: 32,
            planner_action_repeat: 1,
            wm_kstep_k: 1,
            wm_kstep_batch: 32,
            wm_kstep_train_prob: 0.0,
            wm_kstep_loss_coef: 1.0,
            planner_refresh_interval: 200,
            planner_policy_mix: 0.0,
            planner_policy_temperature: 1.0,
            planner_use_mcts: false,
            mcts_simulations: 64,
            mcts_c_puct: std::f32::consts::SQRT_2,
            planner_noise_sigma: 0.0,
            replan_surprise_mult: 0.0,
            plasticity_interval: 0,
            plasticity_shrink: 0.999,
            plasticity_noise: 0.01,
            surprise_ring_capacity: 0,
            surprise_replay_mult: 1.5,
            planner_sigma_alpha: 0.0,
            planner_sigma_horizon: 0.0,
            wm_stochastic: false,
            wm_sigma_loss_coef: 0.5,
            planner_goal_alpha: 0.0,
            goal_states_cap: 100,
            goal_states_her_prob: 0.0,
            bc_planner_synthetic_r: 0.0,
            value_head_train_coef: 0.0,
            planner_value_alpha: 0.0,
            planner_change_alpha: 0.0,
            value_head_gamma: 0.99,
            value_head_buffer_capacity: 10_000,
            value_head_hidden_dim: 0,
            value_head_lr_scale: 1.0,
            value_buffer_cross_game: false,
            goal_states_cross_game: false,
            confidence_mode: false,
            confidence_win_increment: 0.02,
            confidence_novelty_drop_rate: 0.0001,
            subgoal_k: 0,
            subgoal_lr: 0.05,
            planner_subgoal_alpha: 0.0,
            encoder_kind: EncoderKind::Mlp,
            efficientnet_weights_path: None,
            outcome_window: 1,
            outcome_clamp: 5.0,
            outcome_max_episode_len: 256,
            goal_ema_rate: 0.02,
            opt_level: OptLevel::Full,
        }
    }
}

/// Diagnostics snapshot for observability.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Diagnostics {
    pub step: usize,
    pub env_id: u32,
    /// Forward-dynamics prediction MSE only. This intentionally excludes
    /// the reconstruction regularizer used to train the encoder.
    pub loss_world_model: f32,
    /// Observation/frame reconstruction MSE before its configured weight.
    /// Zero when reconstruction is disabled.
    pub loss_reconstruction: f32,
    pub loss_policy: f32,
    pub loss_replay: f32,
    pub reward_mean: f32,
    pub reward_surprise: f32,
    pub reward_novelty: f32,
    pub reward_homeo: f32,
    pub reward_order: f32,
    pub policy_entropy: f32,
    pub repr_drift: f32,
    pub buffer_len: usize,
    /// L1: which option this lane is currently executing.
    pub current_option: u32,
    /// L1: accumulated return for the current option so far.
    pub option_return: f32,
    /// L1: ‖z_t − goal‖ — how close the lane's latent is to its goal.
    pub goal_distance: f32,
    /// L1 continuous goal prototypes: mean pairwise Euclidean distance
    /// across the `num_options` goal vectors. Zero until the table
    /// diverges from init; drops toward zero if prototypes collapse to
    /// a single point (mode collapse — a sign the EMA rate is too
    /// aggressive relative to L0's per-option differentiation).
    pub goal_diversity: f32,
    /// M6 outcome-value head prediction at this lane's latest latent.
    /// Signed (centered by the lane's baseline EMA), clamped to
    /// `[-5, +5]`. Zero when `outcome_reward_alpha == 0`. Rising magnitude
    /// over training means the head is discriminating between
    /// high-return and low-return trajectories.
    pub r_hat: f32,
    /// M6 running EMA baseline — mean episode return across all
    /// completed episodes observed so far (per-lane baselines; this
    /// diagnostic reports the last updated value across all lanes).
    /// Drifts toward the agent's asymptotic per-episode return.
    pub outcome_baseline: f32,
    /// M6 most recent training loss on a completed episode. Zero
    /// until the first episode boundary fires.
    pub outcome_loss: f32,
    /// M7 L2 distance from this lane's current latent to the
    /// prototype centroid. Zero when M7 is disabled or the
    /// prototype hasn't seeded yet.
    pub approach_distance: f32,
    /// M7 number of completed episodes in the prototype buffer.
    /// Rises until it caps at `approach_buffer_size`.
    pub approach_buffer_fill: usize,
    /// M7 centroid drift at the last recompute — L2 distance from
    /// the previous centroid. High = prototype unstable; zero =
    /// not yet recomputed twice.
    pub approach_centroid_drift: f32,
    /// M7 centroid age in episodes since last recompute. Caps at
    /// `approach_update_interval`.
    pub approach_centroid_age: usize,
    /// M7 current confidence `c ∈ [0, 1]`. Zero before warmup;
    /// ramps linearly from 0 → 1 over
    /// `approach_confidence_saturation` episodes once warmup is
    /// met. Used to scale the approach bonus and (optionally)
    /// taper the homeo reward. Always 0 when M7 is disabled.
    pub approach_confidence: f32,
    /// RND predictor MSE averaged across lanes on the most recent
    /// step. Proportional to the per-step curiosity reward
    /// (ignoring the α weight). Tracks how unfamiliar the current
    /// latent cluster is to the predictor; should be positive when
    /// curiosity is driving exploration, and drift toward zero on
    /// over-visited state regions. Always 0 when RND is disabled.
    pub rnd_mse: f32,
}

/// Per-lane state. One per concurrent batch slot. Every slot owns its own
/// adapter, buffer, reward circuit and boundary flag; GPU graphs are shared
/// across lanes and feed the stacked per-lane inputs in a single dispatch.
/// One sampled transition from a lane's buffer, staged for a batched
/// replay forward+backward. We materialize this as a struct (rather than
/// a 4-tuple) so the replay-step code reads cleanly.
#[derive(Clone)]
struct ReplaySample {
    obs: Vec<f32>,
    action: Vec<f32>,
    action_parameters: Vec<f32>,
    z_target: Vec<f32>,
    env_id: u32,
}

/// One committed planner decision. Parameters are separate from the discrete
/// identity so queued plans cannot corrupt policy one-hot labels. `None` means
/// the caller should obtain parameters from its normal action head/fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PlannedAction {
    action: u32,
    parameters: Option<[f32; ACTION_PARAMETER_DIM]>,
}

fn compose_wm_action_token(base_action: &[f32], parameters: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), WM_ACTION_DIM);
    out.fill(0.0);
    let base_len = base_action.len().min(MAX_ACTION_DIM);
    out[..base_len].copy_from_slice(&base_action[..base_len]);
    let parameter_len = parameters.len().min(ACTION_PARAMETER_DIM);
    out[MAX_ACTION_DIM..MAX_ACTION_DIM + parameter_len]
        .copy_from_slice(&parameters[..parameter_len]);
}

fn compose_coord_features(observation: &[f32], task_embedding: &[f32]) -> Vec<f32> {
    (0..OBS_TOKEN_DIM + TASK_DIM)
        .map(|i| {
            if i < OBS_TOKEN_DIM {
                observation.get(i).copied().unwrap_or(0.0)
            } else {
                task_embedding
                    .get(i - OBS_TOKEN_DIM)
                    .copied()
                    .unwrap_or(0.0)
            }
        })
        .map(|value| {
            if value.is_finite() {
                value.clamp(-10.0, 10.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn is_task_event(explicit_event: Option<bool>, extrinsic_reward: f32) -> bool {
    explicit_event.unwrap_or(extrinsic_reward > 0.0)
}

fn greedy_policy_action(kind: ActionKind, head: &[f32]) -> Action {
    match kind {
        ActionKind::Discrete { n } => {
            let live = n.min(head.len());
            assert!(live > 0, "discrete action space must not be empty");
            let index = (0..live)
                .max_by(|&left, &right| head[left].total_cmp(&head[right]))
                .unwrap_or(0);
            Action::Discrete(index)
        }
        ActionKind::Continuous { dim, .. } => {
            Action::Continuous(head[..dim.min(head.len())].to_vec())
        }
    }
}

struct Lane {
    adapter: Box<dyn EnvAdapter>,
    buffer: ExperienceBuffer,
    reward_circuit: RewardCircuit,
    pending_boundary: bool,

    // Action-persistence state (AgentConfig::action_repeat). When `repeats_left`
    // is > 0 we hand back `cached_action` instead of resampling, but the batched
    // policy forward still runs every step so the other lanes' fresh samples
    // come through in the same dispatch.
    cached_action: Option<Action>,
    repeats_left: usize,

    // --- L1 option state (Phase G) ---
    current_option: u32,
    option_goal: Vec<f32>,
    option_steps_left: usize,
    /// Number of env steps since the current option was last (re)sampled.
    option_elapsed: u32,
    option_return: f32,
    /// Value prediction cached at option-start for advantage computation.
    option_start_value: f32,
    /// Encoder latent captured at option-start. The option policy must train
    /// the choice against the state where it was made, not its terminal state.
    option_start_z: Vec<f32>,

    // Cached last-step values for diagnostics & policy advantage.
    last_value: f32,
    /// π_old(a | s) for the action just sampled in `act()`. Stored
    /// into `Transition.prob_taken` in `observe()` so the PPO path can
    /// compute importance ratios. In (0, 1]; default 1.0 before first
    /// `act()`.
    last_prob_taken: f32,
    /// EMA of this lane's per-step WM prediction error. Lazily seeded
    /// with the first observed value; decay 0.99. Drives the
    /// surprise-triggered replanning gate (`replan_surprise_mult`).
    pred_error_ema: f32,
    /// Full logits (pre-softmax, length MAX_ACTION_DIM) at action time —
    /// used by the KL-penalty PPO path to compute KL(π_new ‖ π_old)
    /// exactly. Stored in `Transition.logits_at_action` on observe.
    /// Only populated when `use_kl_ppo` is on; empty otherwise.
    last_logits: Vec<f32>,
    last_entropy: f32,
    last_surprise: f32,
    last_novelty: f32,
    /// Per-lane confidence in [0, 1]. Rises ONLY on extrinsic events;
    /// falls when per-step novelty (visit_count or WM surprise) is
    /// high. Used by the confidence-weighted planner to dynamically
    /// balance exploit vs explore: high C → emphasize value/goal;
    /// low C → emphasize change/RND/novelty. Default 0.5 at start.
    ///
    /// The design is intentionally asymmetric: exploration must
    /// PROVE itself with extrinsic events before C increases. This
    /// keeps the planner conservative about exploitation at frontier
    /// states (newly reached levels) where WM and value-head haven't
    /// validated their predictions with actual wins.
    confidence: f32,
    last_homeo: f32,
    last_order: f32,
    last_reward: f32,
    /// Reward excluding the L1 goal-alignment bonus. Used as the value
    /// head's TD target so that the value baseline does NOT absorb the
    /// option-conditioned bonus; otherwise advantage = reward − value
    /// would cancel the bonus out and the policy-gradient signal that
    /// distinguishes options would collapse. When L1 is inactive this
    /// equals `last_reward`.
    last_base_reward: f32,

    // --- M6 learnable reward (outcome-value head) ---
    /// Sequence of encoder latents since this lane's last episode
    /// boundary. Capped at `outcome_max_episode_len`; the tail is
    /// silently truncated if the episode overflows. Cleared on
    /// episode reset.
    outcome_ep_trajectory: Vec<Vec<f32>>,
    /// Running sum of `r_base` over the current episode.
    outcome_ep_return: f32,
    /// Most recently completed episode's full policy return (snapshotted
    /// from `sil_ep_return` at env_boundary, before reset). Used by GRPO
    /// per-episode advantage mode (`use_grpo_episode`), so this must include
    /// extrinsic reward rather than the M6-specific intrinsic accumulator.
    last_episode_return: f32,
    /// SIL episode return: accumulates the FULL per-step reward
    /// (including extrinsic) over the current episode. Used by SIL's
    /// "successful episode" predicate. Distinct from `outcome_ep_return`
    /// which only tracks `reward_pre_m6` (intrinsic, excludes extrinsic).
    /// Without this separate tracker, kindle-native runs that disable
    /// intrinsics (surprise=0, novelty=0) would have outcome_ep_return
    /// stuck at 0 and SIL would never push.
    sil_ep_return: f32,
    /// Number of positive task-achievement events in this episode.
    /// Used by SIL's optional event-filter mode (`sil_event_filter`)
    /// to push ONLY trajectories that contained at least one
    /// task event — turning the SIL buffer into a
    /// "winning trajectories" replay rather than a generic
    /// above-baseline-return replay.
    sil_ep_event_count: u32,
    /// Single-step `r_base` of the just-finished step. Becomes the
    /// previous episode's terminal reward when the *next* step
    /// carries `env_boundary=true`. Used by
    /// `OutcomeTarget::TerminalReward`.
    outcome_last_step_reward: f32,
    /// Per-step `r_base` history within the current episode — the
    /// raw material for `OutcomeTarget::RewardToGo`, which
    /// back-accumulates these into per-step targets at episode end.
    /// Cleared alongside `outcome_ep_trajectory` on boundary. Same
    /// cap so we match it row-by-row.
    outcome_ep_step_rewards: Vec<f32>,
    /// Previous step's `R̂` value (post-clamp). Used by
    /// `OutcomeBonus::PotentialDelta` to emit `α · (R̂_t − R̂_{t-1})`
    /// as the per-step bonus instead of `α · R̂_t`. Reset to 0 at
    /// `env_boundary`.
    prev_r_hat: f32,
    /// EMA of completed-episode returns for variance reduction.
    outcome_baseline: f32,
    /// Last `R̂(z_t)` forward read; used for the per-step reward bonus
    /// and diagnostics.
    last_r_hat: f32,
    /// True once this lane has seen at least one completed episode,
    /// so the baseline has meaningful value (before that, `baseline = 0`
    /// and we treat the first episode's return as the baseline seed).
    outcome_baseline_seeded: bool,
}

/// The kindle agent.
pub struct Agent {
    pub config: AgentConfig,
    /// Shared blade GPU context: every meganeura `Session` this agent
    /// owns is built via `Session::with_context` against this same
    /// instance, and any sibling compute pipeline (e.g. the V2-S
    /// preprocess pass added in Track B.2) is constructed against it
    /// too.  Held as `Arc` so `Session`s and the preprocess can each
    /// keep their own clones without ownership conflicts.
    gpu: Arc<blade_graphics::Context>,
    /// N lanes, N = config.batch_size. Fixed at construction.
    lanes: Vec<Lane>,
    /// Per-env task code (key = env_id, value length = TASK_DIM). Codes use the
    /// checkpoint-compatible dense hash by default or orthogonal common ids
    /// when configured. We tile active lane codes into the encoder input each
    /// step. The codes are not trained; their projection is.
    task_embeddings: HashMap<u32, Vec<f32>>,
    wm_session: Session,
    /// Optional sibling session running EfficientNetV2-S features[0:6]
    /// when `encoder_kind == EfficientNetV2S`. Inputs raw RGB
    /// `[batch · 3 · 192 · 192]` via the `image` input slot; outputs
    /// `[batch · 160 · 12 · 12]` features that we copy into
    /// `wm_session`'s `visual_obs` input each step.
    efficientnet_session: Option<Session>,
    /// Byte size of the V2-S session's `image` input buffer. Zero when
    /// V2-S is not in use. Reported by `image_input_host_size`.
    efficientnet_input_size_bytes: usize,
    /// Reusable scratch for the V2-S forward output before it gets
    /// uploaded into `wm_session`'s `visual_obs`. Sized
    /// `[batch · 160 · 12 · 12]`. Empty when V2-S isn't in use.
    efficientnet_output_buf: Vec<f32>,
    /// GPU-side preprocess pipeline that pulls per-lane Dullahan-imported
    /// frames through bilinear resize + BGRA→RGB + uint8→f32 directly
    /// into `efficientnet_session`'s `image` input buffer.  `None` when
    /// V2-S isn't in use, in which case callers stage frames via the
    /// legacy host-pointer path (`image_input_host_ptr`) instead.
    v2s_preprocess: Option<crate::v2s_preprocess::PreprocessPipeline>,
    /// Policy graph selected at construction: categorical cross-entropy for
    /// discrete adapters or advantage-weighted fixed-variance Gaussian NLL
    /// for continuous adapters.
    policy_session: Session,
    /// L1 option-policy session. `None` when `num_options <= 1` (L0-only).
    option_session: Option<Session>,
    /// Per-lane latent dim (the WM graph is [N, latent_dim]).
    latent_dim: usize,
    step_count: usize,
    probe_obs: Option<Vec<Vec<f32>>>,
    probe_reference: Option<Vec<Vec<f32>>>,
    last_wm_loss: f32,
    last_recon_loss: f32,
    last_policy_loss: f32,
    /// EMA of `|last_policy_loss|`. Updated inside the policy
    /// training paths after each `step()`. Drives
    /// `policy_lr_adaptive_target` scaling when enabled. Zero
    /// at init — the first few updates run at full LR until the
    /// EMA warms up.
    policy_loss_ema: f32,
    /// Env-step counter modulo `policy_update_interval`. When it
    /// reaches the configured interval, the policy fires
    /// `interval` consecutive gradient steps on a sliding window
    /// of ripe transitions and resets to 0.
    policy_update_ticks: usize,
    last_replay_loss: f32,
    last_drift: f32,
    encoder_lr_scale: f32,
    /// Batch LR compensation: user's `learning_rate` is per-sample, but
    /// every WM/policy loss is averaged over N rows, so per-sample
    /// gradient magnitude shrinks linearly with N. We multiply every
    /// learning rate by √N at the use sites so the effective per-sample
    /// update matches the N = 1 reference. √N (not N) is the standard
    /// large-batch rule of thumb — linear scaling tends to destabilize
    /// at larger N.
    batch_lr_scale: f32,
    /// Scratch buffers, sized [N × per-lane-dim], reused each step.
    obs_token_scratch: Vec<f32>,
    /// Cached expected byte-size of the WM session's `visual_obs`
    /// input slot. Non-zero only when `encoder_kind` is `Cnn`.
    /// Used to sanity-check caller-supplied slices. The actual
    /// data lives in the meganeura-owned, device-local, host-
    /// visible graph buffer — accessed via
    /// `wm_session.input_host_ptr("visual_obs")` — so kindle keeps
    /// no CPU-side scratch for the CNN input.
    visual_obs_size_bytes: usize,
    /// Base action identity/value used by the policy. Always
    /// `[lanes × MAX_ACTION_DIM]`; never contains auxiliary parameters.
    action_token_scratch: Vec<f32>,
    /// World-model action rows: the base action token followed by optional
    /// normalized parameters such as ARC click `(x, y)`.
    wm_action_token_scratch: Vec<f32>,
    /// One-shot parameters staged by the host for the next `observe()`.
    action_parameter_scratch: Vec<f32>,
    action_parameter_active: Vec<bool>,
    /// Per-lane/per-discrete-action declaration that the action consumes the
    /// two auxiliary parameters. The planner samples parameter tails only for
    /// these entries; ordinary actions must keep a zero tail to avoid OOD WM
    /// exploitation.
    action_parameter_masks: Vec<bool>,
    /// Parameters attached to the most recent planner-queued action popped by
    /// `act()`. The host consumes them once via
    /// `take_planned_action_parameters()` before stepping the environment.
    last_planned_action_parameters: Vec<Option<[f32; ACTION_PARAMETER_DIM]>>,
    z_target_scratch: Vec<f32>,
    task_scratch: Vec<f32>,
    value_target_scratch: Vec<f32>,
    /// Per-row policy action targets for the policy dispatch. Discrete
    /// plain-PG folds advantage into these labels; PPO and continuous PG keep
    /// the raw action here and stage advantage separately.
    policy_action_scratch: Vec<f32>,
    /// Mask scratch buffer fed to the policy graph's `action_mask`
    /// input each policy_step. Same `[policy_batch × MAX_ACTION_DIM]`
    /// layout as `policy_action_scratch`. Populated from
    /// `self.action_masks` (lane × MAX_ACTION_DIM). Padded action heads are
    /// invalid by default for discrete adapters.
    policy_action_mask_scratch: Vec<f32>,
    /// True for discrete policy graphs, all of which consume an
    /// `action_mask` input. Continuous policy graphs do not expose it.
    policy_action_mask_input_present: bool,
    /// Pre-allocated `[policy_batch × latent_dim]` buffer for the
    /// policy session's `z` input. At rollout_length=1 this is the
    /// same size as the encoder's per-lane z output; at >1 the
    /// rows beyond `lanes` are zero-padded on every act() call.
    policy_z_scratch: Vec<f32>,
    /// PPO mode: per-row advantage `[N, 1]` (separate input, not
    /// baked into `policy_action_scratch` like the plain path).
    ppo_advantage_scratch: Vec<f32>,
    /// PPO mode: per-row `π_old(a | s)` for the taken action
    /// `[N, 1]`. Positive, non-zero.
    ppo_old_prob_scratch: Vec<f32>,
    /// KL-PPO mode: per-row stored old_logits `[batch, action_dim]`.
    /// Filled from `ripe.logits_at_action` in policy_step_rollout_batch
    /// when `use_kl_ppo` is on. Empty otherwise.
    kl_old_logits_scratch: Vec<f32>,
    /// True iff the policy graph has the runtime-mutable
    /// "entropy_beta" input (built only when construction-time
    /// entropy_beta > 0 in the e2e graph). Used by
    /// `feed_entropy_beta_input` to gate set_input safely.
    entropy_beta_input_present: bool,
    /// True iff the policy graph has the "old_logits" input
    /// (built only when use_kl_ppo + kl_beta > 0). Otherwise the
    /// KL branch is fully elided and set_input("old_logits", …)
    /// would error.
    old_logits_input_present: bool,
    /// Mirror of `reward_pred_loss_coef > 0 && end_to_end_encoder`. When
    /// true the policy graph has a `reward_target` input that must be
    /// fed every training step.
    reward_pred_input_present: bool,
    /// Scratch for the policy graph's `reward_target` input, sized
    /// `policy_batch × 1`. Filled from `transitions[ripe_idx].reward`.
    /// Empty when `reward_pred_input_present` is false.
    reward_target_scratch: Vec<f32>,
    /// Most recent observed KL(π_new ‖ π_old) from the policy
    /// session, read from output index 3 after each training step
    /// when KL-PPO is on. Used by the harness for adaptive β.
    last_kl: f32,
    /// True when policy_step_rollout_batch is being called purely
    /// to capture the KL π_old snapshot (forward only, LR=0). Set
    /// by `capture_kl_snapshot_logits` for the duration of one call.
    /// Re-introduced for snapshot-bug debug per docs/failed_experiments.md.
    kl_snapshot_capture_pending: bool,
    /// Effective option_dim (resolved from config: 0 → latent_dim). The
    /// goal vector width used by the goal-alignment reward bonus.
    option_dim: usize,
    /// Stacked per-lane one-hot option encodings fed to the policy
    /// graph's `option_onehot` input when L1 is active. Each row's
    /// active-option slot is `1.0`, others `0.0`. Empty (`n * 0`) when
    /// `num_options = 1`.
    option_onehot_scratch: Vec<f32>,
    /// L1 scratch buffers.
    option_taken_scratch: Vec<f32>,
    option_return_scratch: Vec<f32>,
    /// Termination BCE target, `[N, 1]`. Zero unless the agent decided
    /// (this step) that the current option should have ended; then 1.
    termination_target_scratch: Vec<f32>,
    /// Fixed goal lookup table [num_options × option_dim]. Each option
    /// maps to a pre-set orthogonal direction in latent space. L1 learns
    /// which option to pick; the goal vectors themselves are constants.
    goal_table: Vec<f32>,

    /// M6 outcome-value head (CPU). `None` when
    /// `outcome_reward_alpha == 0.0` (default) — no compute cost.
    outcome_head: Option<outcome::OutcomeHead>,
    /// M6: last episode return observed across any lane, smoothed by
    /// the baseline EMA. Diagnostic only — each lane keeps its own
    /// baseline for the actual centering.
    last_outcome_baseline: f32,

    /// M7 approach-reward state. `None` when
    /// `approach_reward_alpha == 0.0` (default) — zero CPU cost.
    approach_state: Option<approach::ApproachState>,
    /// RND curiosity state. `None` when `rnd_reward_alpha == 0.0`
    /// (default) — zero compute cost.
    rnd_state: Option<rnd::RndState>,
    /// Most recent RND per-step MSE across lanes, averaged.
    /// Diagnostic only.
    last_rnd_mse: f32,
    /// DIAYN discriminator state. `None` when `diayn_reward_alpha == 0.0`
    /// or `num_options < 2`. CPU implementation per `kindle::diayn`.
    diayn_state: Option<diayn::DiaynState>,
    /// Most recent DIAYN per-step intrinsic reward across lanes,
    /// averaged. Diagnostic only.
    last_diayn_reward: f32,
    /// Continuous coord-action head. `None` when
    /// `coord_action_alpha == 0.0`.
    coord_head: Option<coord::CoordHead>,
    /// Per-lane last reward, cached so the coord head's
    /// REINFORCE update at the NEXT step can use the advantage
    /// `reward − running_baseline`. We recompute on observe.
    coord_last_reward: Vec<f32>,
    /// Per-lane EMA baselines for coord-head advantage centering. A single
    /// shared baseline mixes unrelated reward scales in heterogeneous batches.
    coord_reward_baseline: Vec<f32>,
    /// M7 per-lane episode-return accumulator (in kindle's
    /// intrinsic reward). Used so the prototype-updater sees the
    /// same `r_ep` that trained the policy, not a separate
    /// quantity.
    approach_ep_returns: Vec<f32>,
    /// M7 latest approach-distance per lane, for diagnostics.
    approach_distances: Vec<f32>,
    /// M8 delta-goal bank (shared across lanes). `None` when
    /// `delta_goal_alpha == 0.0`.
    delta_goal_bank: Option<delta_goals::DeltaGoalBank>,
    /// M8 per-lane previous latent, cleared at episode boundaries
    /// so the cross-episode jump never triggers a spurious goal.
    delta_goal_prev_latent: Vec<Option<Vec<f32>>>,
    /// M8 number of goal-events recorded during the most recent
    /// `observe()` call, summed across lanes. Diagnostic only.
    last_delta_goal_events: usize,
    /// Cross-episode state-action memory (shared across lanes).
    /// `None` when `xeps_reward_alpha == 0.0`.
    xeps_memory: Option<xeps_memory::StateActionMemory>,
    /// Last action sampled for each lane (discrete id), cached so
    /// the next `observe()` can credit the preceding (state,
    /// action) pair. `None` before any action has been sampled on
    /// that lane, or after an episode boundary reset.
    xeps_prev_action: Vec<Option<u32>>,
    /// Track 3 model-based planner. `None` when
    /// `planner_horizon == 0`. Weights are refreshed every
    /// `planner_refresh_interval` calls to `plan_and_queue()`.
    planner: Option<planner::WmRollout>,
    /// GPU-side WM rollout session, batch_size = `n_lanes × planner_samples`.
    /// Used by the model-based planner to roll forward all candidate
    /// sequences in batched GPU dispatches instead of the per-sample
    /// CPU `WmRollout` matmul loop. Weights are synced from `wm_session`
    /// at the same cadence as `planner.refresh_from_session`.
    wm_planner_session: Option<Session>,
    /// Forward-only policy session at the same batch_size as
    /// `wm_planner_session`. When `planner_policy_guided > 0` is on
    /// (currently always on when `planner_horizon > 0`), the planner
    /// samples action sequences from this policy's logits (per-row,
    /// masked) instead of from uniform random. Weights synced from
    /// the main `policy_session` at `planner_refresh_interval` cadence.
    /// `None` when policy-guided planning is off.
    policy_planner_session: Option<Session>,
    /// k-step WM training session. Built when `wm_kstep_k > 1`. Applies
    /// WM k times iteratively on sampled (t, t+k) tuples from the lane
    /// buffer and computes MSE against the actual z_{t+k}. WM params
    /// are synced back to `wm_session` after each step so the canonical
    /// WM benefits from the deeper accuracy training. Direct fix for
    /// the rollout-depth compounding error that blocked L3.
    wm_kstep_session: Option<Session>,
    /// Scratch buffers for wm_kstep_session inputs.
    /// - kstep_z_scratch: [N, ld]
    /// - kstep_z_target_scratch: [N, ld]
    /// - kstep_action_scratch_per_step: Vec<Vec<f32>>, k entries each [N, WM_ACTION_DIM]
    kstep_z_scratch: Vec<f32>,
    kstep_z_target_scratch: Vec<f32>,
    kstep_action_scratch_per_step: Vec<Vec<f32>>,
    /// Most recent k-step WM loss (diagnostic).
    last_wm_kstep_loss: f32,
    /// Forward-only option-policy session. Built when
    /// `planner_horizon > 0` AND `num_options >= 2`. Phase 2 of
    /// option-aware planning:
    /// at each planner outer step, this maps z → option_logits per
    /// row; the planner samples an option and then uses
    /// `policy_planner_session` (which now also takes option_onehot)
    /// to generate atomic actions for the option's window.
    option_planner_session: Option<Session>,
    /// Dedicated WM forward session sized at batch=n_lanes for MCTS
    /// expansion (one row per lane per simulation step). Smaller batch
    /// than `wm_planner_session` so each MCTS dispatch isn't wasting
    /// 31/32 of the parallel slots. Built only when
    /// `planner_use_mcts = true`.
    wm_mcts_session: Option<Session>,
    /// `planner_samples` cached for the GPU planner; mirrors
    /// `config.planner_samples` so the runtime path can avoid borrowing
    /// the config when computing batch offsets.
    planner_samples_cached: usize,
    /// Scratch buffers for the GPU planner. Sized once at construction
    /// to `planner_batch * WM_ACTION_DIM` / latent_dim respectively.
    /// All three are `Vec<f32>` so the planner can avoid per-call
    /// allocation in the hot path.
    planner_z_scratch: Vec<f32>,
    planner_action_scratch: Vec<f32>,
    /// Per-row option_onehot for option-aware planning. Sized
    /// `planner_batch * num_options`. Empty when `num_options < 2`.
    /// Each outer planner step assigns each row an option (currently
    /// uniform-random) and writes the one-hot here; policy_planner_session
    /// then produces option-conditional action logits.
    planner_option_onehot_scratch: Vec<f32>,
    planner_traj_scratch: Vec<f32>,
    /// Per-step learned-σ buffer from the planner's σ-head output (only
    /// populated when `wm_stochastic` is enabled). Layout matches
    /// `planner_traj_scratch` per outer step: `[planner_batch × latent_dim]`.
    /// Lazily resized in the rollout loop (allocated on first use).
    planner_sigma_scratch: Vec<f32>,
    /// (P2) Surprise frame-replay ring (CNN modes). Parallel arrays,
    /// FIFO by `surprise_ring_next`; `surprise_ring_prio` drives
    /// priority-proportional sampling at replay time.
    surprise_ring_frames: Vec<Vec<f32>>,
    surprise_ring_actions: Vec<Vec<f32>>,
    surprise_ring_zprev: Vec<Vec<f32>>,
    surprise_ring_obs: Vec<Vec<f32>>,
    surprise_ring_env: Vec<u32>,
    surprise_ring_prio: Vec<f32>,
    surprise_ring_next: usize,
    /// (P3) Per-(outer-step, row) mean learned σ along each rollout:
    /// `[k * batch]`, row-major by step. Feeds the information-seeking
    /// score bonus and the σ-budgeted adaptive horizon.
    planner_sigma_traj_scratch: Vec<f32>,
    /// Per-step value predictions from the planner's value-head branch.
    /// Layout: `[planner_horizon × planner_batch]`. Only populated when
    /// `wm_planner_has_value_head` is true.
    planner_v_traj_scratch: Vec<f32>,
    /// Pre-allocated planner WM params buffer (sized once for the
    /// largest WM tensor) for the periodic weight refresh.
    planner_param_buf: Vec<f32>,
    /// Per-lane action queue populated by the planner. `act()`
    /// pops from the front of this queue before policy sampling,
    /// so a queued sequence commits the next-K actions.
    planner_queue: Vec<std::collections::VecDeque<PlannedAction>>,
    /// Number of `plan_and_queue` calls since the last WM-weight
    /// refresh. Triggers a refresh when it hits
    /// `planner_refresh_interval`.
    planner_calls_since_refresh: usize,
    /// Per-lane extrinsic reward for the NEXT `observe()` call.
    /// Populated by the harness via `set_extrinsic_reward`. Zero-
    /// initialized; cleared back to 0 after each `observe()` so a
    /// missed `set_extrinsic_reward` doesn't silently repeat the
    /// previous step's value. Only consumed when
    /// `extrinsic_reward_alpha > 0`.
    extrinsic_reward: Vec<f32>,
    /// Explicit per-lane task-event override for the next `observe()` call.
    /// Native `terminated` results populate this even when the environment's
    /// reward is zero or negative (for example MountainCar's terminal -1).
    /// `None` retains the legacy positive-reward inference; `Some(false)` is
    /// important for environments such as CartPole whose positive reward does
    /// not mean the current transition was successful.
    extrinsic_event: Vec<Option<bool>>,
    /// Per-lane intrinsic progress reward, set via
    /// `set_intrinsic_progress`. Adds to the per-step reward sum
    /// but does NOT increment `sil_ep_event_count` — the
    /// win-classifier's is_win label still requires a real
    /// extrinsic event. Designed for dense progress signals like
    /// "persistent configurational change" and "per-episode
    /// coarse-state entropy growth." Scaled by
    /// `extrinsic_reward_alpha` so harness can use [0, 1]
    /// magnitudes. Cleared each observe.
    intrinsic_progress_reward: Vec<f32>,
    /// Per-lane empowerment estimate from the most recent
    /// `plan_and_queue_gpu` call. Defined as the mean across latent
    /// dims of the cross-sample variance of `z_next` at step 0 of
    /// the planner rollout: if different first actions lead to
    /// widely different latents, the current state has high
    /// optionality. Stuck states (every action ends up in the same
    /// place) have ~0 empowerment.
    /// Updated at planner cadence (every macro), not every step.
    /// Exposed via `empowerment()` for the harness to fold into
    /// the intrinsic-progress reward.
    last_empowerment: Vec<f32>,
    /// Per-env win-state archive. Each entry is the lane's latent at the
    /// moment an extrinsic-reward event fired (level completion). FIFO
    /// bounded by `config.goal_states_cap`. Consumed by `plan_and_queue_*`
    /// when `config.planner_goal_alpha > 0` to bias trajectory scoring
    /// toward predicted-similarity to past win-states. The agent thus
    /// emergently discovers its own goal region in latent space.
    ///
    /// Keyed by env_id (`lane.adapter.id()`) so each game has its own
    /// goal archive — winning level 1 of tu93 doesn't bias planning in
    /// g50t. Empty for envs that haven't yet had any extrinsic event.
    goal_states: hashbrown::HashMap<u32, std::collections::VecDeque<Vec<f32>>>,
    /// Per-env (or pooled when `goal_states_cross_game`) sub-goal
    /// centroids learned via online k-means over the `goal_states`
    /// queue. Capacity = `subgoal_k`. Centroids are pulled toward
    /// every newly pushed win latent. The planner adds
    /// `subgoal_alpha * max cos_sim(z_step, centroid)` to its
    /// trajectory score when `subgoal_k > 0`.
    ///
    /// Centroids are an ABSTRACTION over raw goal_states — they
    /// generalize "regions of winning states" instead of pointing
    /// at specific past wins. This is the cleanest knob we have
    /// for vertical transfer: L1 wins create centroids; the
    /// planner navigates toward centroid regions at L2 even with
    /// no L2 wins yet, because the centroid captures the COMMON
    /// shape of winning states in the encoder's latent space.
    subgoal_centroids: hashbrown::HashMap<u32, Vec<Vec<f32>>>,
    /// Per-lane action availability mask for `act()` sampling. Flat
    /// `[batch_size × MAX_ACTION_DIM]` layout; 1.0 = valid, 0.0 = invalid.
    /// Defaults to each adapter's static action range (padded heads invalid).
    /// Set via `Agent::set_action_masks`. Persists across `act()` calls until
    /// re-set (unlike `extrinsic_reward`, which auto-clears) — masks
    /// usually stay stable for many steps within an episode.
    /// Invalid logits are forced to large-negative before sampling so
    /// the categorical never picks them; `last_prob_taken` is computed
    /// from the masked softmax so PPO's π_old denominator matches the
    /// distribution actually sampled from.
    action_masks: Vec<f32>,
    /// SIL replay buffer: (obs, action_idx, value_target, env_id)
    /// from successful episodes. Capacity is balanced across represented task
    /// ids, including inactive ones, while updates balance only ids active in
    /// the current lanes. Long/fast tasks therefore cannot erase another
    /// task, and stale action semantics cannot block adaptation after a switch.
    /// See `AgentConfig::use_sil`.
    sil_buffer: std::collections::VecDeque<SilSample>,
    /// Per-task EMA of eligible episode returns, used as the threshold for
    /// "successful episode" → SIL push. Comparing unrelated reward scales
    /// made positive-return games suppress successful negative-return games.
    sil_baselines: hashbrown::HashMap<u32, f32>,
    /// Counts of SIL update outcomes for diagnostics.
    sil_updates_attempted: u64,
    sil_updates_fired: u64,
    /// (P4) Total planner-queue clears triggered by surprise spikes.
    replan_clears: u64,
    /// (P2) Total surprise-replay dispatches fired.
    surprise_replays: u64,
    sil_last_active_rows: u32,
    /// L1 change in policy.fc1.weight (first 32 values) caused by the
    /// most recent SIL update. >0 means SIL is moving params; ≈0 means
    /// SIL fired but had no effect on params.
    sil_last_param_change: f32,
    /// True iff `wm_planner_session` was built with the value-head
    /// branch (output index 1 = V_next per row). When false the
    /// planner only reads z_next (output index 0). Mirrors
    /// `planner_value_alpha > 0 && planner_horizon > 0`.
    wm_planner_has_value_head: bool,
    /// Win-trajectory replay buffer, GAME-STRATIFIED by env_id.
    /// Per-env_id queue. `(obs_token, env_id, label)`, label =
    /// γ^(T-t) from terminal. Per-env cap is
    /// `value_head_buffer_capacity / 2` (each env can hold up to
    /// half the total class budget — keeps individual fast games
    /// from monopolizing the buffer in joint training).
    ///
    /// Sampled by round-robin across env_ids in `value_replay_step`
    /// so each game contributes equally to the classifier's
    /// gradient, regardless of which game produces most wins.
    /// Entries are `(latent, env_id, label)` — the stored LATENT of
    /// the state (fed to the value head via the `value_z` graph
    /// input), not the obs token (pre-2026-06-10 layout).
    win_buffer: hashbrown::HashMap<u32, std::collections::VecDeque<(Vec<f32>, u32, f32)>>,
    /// Loss-trajectory replay buffer, GAME-STRATIFIED by env_id.
    /// Same structure as `win_buffer`; label always 0.
    loss_buffer: hashbrown::HashMap<u32, std::collections::VecDeque<(Vec<f32>, u32, f32)>>,
    /// Scratch for wm_session's `value_target` input `[batch_size, 1]`.
    /// Set to R_to_go for value-replay calls, zeros otherwise (gated
    /// off by `value_gate=0`).
    value_target_scratch_vh: Vec<f32>,
    /// Scratch for wm_session's `value_z` input `[batch_size, latent_dim]`
    /// — the stored win/loss-sample latents fed to the value head.
    value_z_scratch: Vec<f32>,
    /// Scratch buffers for wm_session's loss-gate inputs (each size 1).
    /// `wm_gate=1, value_gate=0` for normal WM forward;
    /// `wm_gate=0, value_gate=1` for value-replay forward.
    wm_gate_scalar: Vec<f32>,
    value_gate_scalar: Vec<f32>,
    /// Most recent value-head training loss, exposed as a diagnostic.
    last_value_head_loss: f32,
    /// Number of value-head update calls fired so far (diagnostic).
    value_head_updates: u64,
}

/// One sample in the SIL replay buffer.
///
/// Stores (s, a, R_to_go, V_at_collect). The SIL update uses
/// `max(0, R_to_go - V_at_collect)` as the per-sample advantage —
/// the "positive advantage filter" from Oh et al. 2018. This focuses
/// gradient on transitions where the actual outcome exceeded V's
/// prediction at collection time, naturally weighting the late-
/// episode "surprising success" moments (e.g. the touchdown noop
/// that yields +100) over routine descent steps. With explicit event
/// filtering, event-success samples instead have a minimum unit advantage so
/// negative step costs cannot erase the successful trajectory prefix.
#[derive(Clone)]
pub struct SilSample {
    pub obs: Vec<f32>,
    /// Latent of the state the action was taken FROM (the previous
    /// record's latent at collection time). Used as the policy input
    /// by the non-e2e SIL path; e2e re-encodes `obs` instead.
    pub z: Vec<f32>,
    pub action_idx: usize,
    /// Action availability at collection time (length MAX_ACTION_DIM).
    pub action_mask: Vec<f32>,
    /// Undiscounted return-to-go from this step to episode end.
    pub r_to_go: f32,
    /// V(s) baseline cached at collection time.
    pub v_at_collect: f32,
    /// True when this transition came from an episode containing a real task
    /// event. Event-filtered SIL uses this provenance to
    /// imitate successful trajectories even when step costs make their early
    /// return-to-go negative.
    pub event_success: bool,
    pub env_id: u32,
}

fn sil_imitation_advantage(
    r_to_go: f32,
    v_at_collect: f32,
    event_success: bool,
    event_filter: bool,
    cap: f32,
) -> f32 {
    let positive_advantage = (r_to_go - v_at_collect).max(0.0);
    let advantage = if event_filter && event_success {
        positive_advantage.max(1.0)
    } else {
        positive_advantage
    };
    advantage.min(cap.max(0.1))
}

fn admit_sil_episode(
    baseline: &mut f32,
    initialized: &mut bool,
    episode_return: f32,
    event_success: bool,
    event_filter: bool,
    decay: f32,
) -> bool {
    if event_filter && !event_success {
        return false;
    }
    let first_eligible = !*initialized;
    let admit = if first_eligible {
        *baseline = episode_return;
        *initialized = true;
        // A sparse event may be rare enough that throwing the first one away
        // leaves nothing to learn from. Standard SIL retains its historical
        // first-episode warmup behavior.
        event_filter
    } else {
        episode_return > *baseline
    };
    if !first_eligible {
        let decay = decay.clamp(0.0, 1.0);
        *baseline = decay * *baseline + (1.0 - decay) * episode_return;
    }
    admit
}

fn admit_sil_episode_for_env(
    baselines: &mut hashbrown::HashMap<u32, f32>,
    env_id: u32,
    episode_return: f32,
    event_success: bool,
    event_filter: bool,
    decay: f32,
) -> bool {
    if let Some(baseline) = baselines.get_mut(&env_id) {
        let mut initialized = true;
        return admit_sil_episode(
            baseline,
            &mut initialized,
            episode_return,
            event_success,
            event_filter,
            decay,
        );
    }
    let mut baseline = 0.0;
    let mut initialized = false;
    let admitted = admit_sil_episode(
        &mut baseline,
        &mut initialized,
        episode_return,
        event_success,
        event_filter,
        decay,
    );
    if initialized {
        baselines.insert(env_id, baseline);
    }
    admitted
}

fn balanced_sil_sample_indices(
    buffer: &std::collections::VecDeque<SilSample>,
    batch_size: usize,
    seed: u64,
    active_env_ids: &std::collections::BTreeSet<u32>,
) -> Vec<usize> {
    if buffer.is_empty() || batch_size == 0 {
        return Vec::new();
    }
    let mut indices_by_env = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for (index, sample) in buffer.iter().enumerate() {
        if active_env_ids.is_empty() || active_env_ids.contains(&sample.env_id) {
            indices_by_env.entry(sample.env_id).or_default().push(index);
        }
    }
    if indices_by_env.is_empty() {
        return Vec::new();
    }
    let groups = indices_by_env.values().collect::<Vec<_>>();
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..batch_size)
        .map(|row| {
            let group = groups[row % groups.len()];
            group[rng.random_range(0..group.len())]
        })
        .collect()
}

fn sil_samples_from_recent_trajectory(
    buffer: &ExperienceBuffer,
    max_samples: usize,
    event_success: bool,
    value_target_clamp: f32,
) -> Vec<SilSample> {
    if max_samples == 0 || buffer.len() < 2 {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(max_samples.min(buffer.len() - 1));
    let mut r_to_go = 0.0f32;
    let mut idx = buffer.len() - 1;
    loop {
        let transition = buffer.get(idx);
        if transition.reward.is_finite() {
            r_to_go += transition.reward;
        }

        // A transition stores the observation reached by its action. Pair that
        // action with the preceding record, which is the acted-from state.
        // Boundary records have no in-episode predecessor and are not samples.
        if !transition.env_boundary && idx > 0 {
            let from = buffer.get(idx - 1);
            samples.push(SilSample {
                obs: from.observation.clone(),
                z: from.latent.clone(),
                action_idx: transition
                    .action
                    .iter()
                    .position(|&value| value > 0.5)
                    .unwrap_or(0),
                action_mask: transition.action_mask.clone(),
                r_to_go: r_to_go.clamp(-value_target_clamp, value_target_clamp),
                v_at_collect: if transition.value.is_finite() {
                    transition.value
                } else {
                    0.0
                },
                event_success,
                env_id: transition.env_id,
            });
            if samples.len() >= max_samples {
                break;
            }
        }

        if transition.env_boundary || idx == 0 {
            break;
        }
        idx -= 1;
    }
    samples
}

fn append_sil_samples(
    buffer: &mut std::collections::VecDeque<SilSample>,
    samples: impl IntoIterator<Item = SilSample>,
    capacity: usize,
) {
    if capacity == 0 {
        return;
    }
    buffer.extend(samples);
    if buffer.len() <= capacity {
        return;
    }

    // Bound each represented task fairly instead of evicting from the global
    // front. Global FIFO eventually erased every replay sample for an inactive
    // task during a sequential curriculum. Water-filling lets small/new tasks
    // keep all samples they currently have and divides the remaining capacity
    // across larger tasks. With one task this reduces exactly to ordinary FIFO.
    let mut counts = std::collections::BTreeMap::<u32, usize>::new();
    for sample in buffer.iter() {
        *counts.entry(sample.env_id).or_default() += 1;
    }
    let mut targets = counts
        .keys()
        .map(|&env_id| (env_id, 0usize))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut remaining = capacity.min(buffer.len());
    while remaining > 0 {
        let active = counts
            .iter()
            .filter_map(|(&env_id, &count)| (targets[&env_id] < count).then_some(env_id))
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let share = remaining / active.len();
        if share == 0 {
            for env_id in active.into_iter().take(remaining) {
                *targets.get_mut(&env_id).expect("target exists") += 1;
            }
            break;
        }
        for env_id in active {
            let target = targets.get_mut(&env_id).expect("target exists");
            let available = counts[&env_id] - *target;
            let added = available.min(share);
            *target += added;
            remaining -= added;
        }
    }

    let mut drops = counts
        .into_iter()
        .map(|(env_id, count)| (env_id, count - targets[&env_id]))
        .collect::<hashbrown::HashMap<_, _>>();
    buffer.retain(|sample| {
        let drop = drops.get_mut(&sample.env_id).expect("drop count exists");
        if *drop == 0 {
            true
        } else {
            *drop -= 1;
            false
        }
    });
}

/// Apply the terminal-proximity bonus retroactively to the K
/// transitions PRECEDING the terminal one in a lane's buffer when the
/// just-ended episode crossed `threshold` reward (the terminal step
/// itself already carries the big event reward — the bonus credits
/// the approach). Stops (without applying) at the episode's own first
/// transition — its `env_boundary` record — so the bonus never spans
/// episode boundaries. Caller controls when this is invoked — kindle
/// calls it in `mark_boundary`, guarded so a double `mark_boundary`
/// on the same terminal doesn't double-apply. No-op if `k == 0` or
/// `bonus <= 0.0`.
fn apply_terminal_proximity_bonus(
    buffer: &mut crate::buffer::ExperienceBuffer,
    k: usize,
    bonus: f32,
    threshold: f32,
) {
    if k == 0 || bonus <= 0.0 {
        return;
    }
    let buf_len = buffer.len();
    if buf_len == 0 {
        return;
    }
    let last_idx = buf_len - 1;
    if buffer.get(last_idx).reward <= threshold {
        return;
    }
    let lookback = k.min(last_idx);
    for i in 1..=lookback {
        let idx = last_idx - i;
        if buffer.get(idx).env_boundary {
            break;
        }
        buffer.get_mut(idx).reward += bonus;
    }
}

/// Cosine similarity between two equal-length vectors. Returns 0 when
/// either side has zero norm (no direction defined).
/// Compute the n-step discounted return starting at `ripe_idx` in
/// `buffer`, with optional TD-bootstrap from the stored value at
/// `ripe_idx + n_step`.
///
/// Returns `(ret, gk_at_end, terminated)`:
/// - `ret`: the accumulated `Σ γ^k r_{ripe+k}` (0..end) plus the
///   bootstrap `γ^n · V(s_{ripe+n_step})` when `bootstrap=true` and
///   the trajectory didn't terminate within the window.
/// - `gk_at_end`: `γ^k_end` — the discount factor at the point where
///   accumulation stopped. Equals `γ^n_step` on normal completion.
/// - `terminated`: true iff the window hit an `env_boundary` (i.e.
///   the episode ended strictly inside the window). A bootstrap is
///   NOT added in that case — terminal states have zero future value
///   by definition.
///
/// `bootstrap_value_clamp` is the symmetric L∞ bound applied to the
/// stored `V(s_{ripe+n_step})` before discounting — a safeguard
/// against a drifted value head poisoning the TD target.
///
/// Caller guarantees `ripe_idx + n_step ≤ buffer.len()` when
/// `bootstrap=true`; `ripe_idx + n_step - 1 ≤ buffer.len() - 1` when
/// `bootstrap=false` (i.e. n_step rewards must all be in-buffer).
fn compute_td_n_step_return(
    buffer: &crate::buffer::ExperienceBuffer,
    ripe_idx: usize,
    n_step: usize,
    gamma: f32,
    bootstrap: bool,
    bootstrap_value_clamp: f32,
) -> (f32, f32, bool) {
    let mut ret = 0.0f32;
    let mut gk = 1.0f32;
    let mut terminated = false;
    for k in 0..n_step {
        let idx = ripe_idx + k;
        let tr = buffer.get(idx);
        if k > 0 && tr.env_boundary {
            terminated = true;
            break;
        }
        ret += gk * tr.reward;
        gk *= gamma;
    }
    if bootstrap && !terminated {
        let boot_idx = ripe_idx + n_step;
        let boot_tr = buffer.get(boot_idx);
        if !boot_tr.env_boundary && boot_tr.value.is_finite() {
            let boot_v = boot_tr
                .value
                .clamp(-bootstrap_value_clamp, bootstrap_value_clamp);
            ret += gk * boot_v;
        }
    }
    (ret, gk, terminated)
}

/// Compute the GAE (Generalized Advantage Estimation) advantage at
/// `ripe_idx` using `n_step` one-step TD errors folded back with
/// factor `γλ`.
///
/// Formula:
/// ```text
/// δ_t       = r_t + γ·V(s_{t+1}) - V(s_t)         per-step TD error
/// Â_t       = δ_t + γλ·Â_{t+1}·(1 - done_{t+1})   GAE recursion
/// ```
/// where `done_{t+1}` is true if the episode terminated between t
/// and t+1 (i.e. `tr_{t+1}.env_boundary` is set — a fresh episode
/// started, so V(s_{t+1}) is semantically zero for the old episode).
///
/// Requires `ripe_idx + n_step ≤ buffer.len() - 1` (one extra slot
/// past the n-step window to read `V(s_{ripe+n_step})` for the last
/// δ's bootstrap). The caller enforces this via
/// `bootstrap_headroom = 1` when `gae_lambda > 0`.
///
/// `bootstrap_value_clamp` is applied symmetrically to both
/// `V(s_t)` and `V(s_{t+1})` readings before computing δ — the
/// same safeguard as `compute_td_n_step_return`.
fn compute_gae_advantage(
    buffer: &crate::buffer::ExperienceBuffer,
    ripe_idx: usize,
    n_step: usize,
    gamma: f32,
    lambda: f32,
    bootstrap_value_clamp: f32,
) -> f32 {
    // First pass: accumulate δ's forward, stopping if the episode
    // ended strictly inside the window (after k=0).
    let mut deltas: Vec<f32> = Vec::with_capacity(n_step);
    for k in 0..n_step {
        let t = ripe_idx + k;
        let tr = buffer.get(t);
        if k > 0 && tr.env_boundary {
            break;
        }
        let next = buffer.get(t + 1);
        let next_v = if next.env_boundary || !next.value.is_finite() {
            0.0
        } else {
            next.value
                .clamp(-bootstrap_value_clamp, bootstrap_value_clamp)
        };
        let v_t = if tr.value.is_finite() {
            tr.value
                .clamp(-bootstrap_value_clamp, bootstrap_value_clamp)
        } else {
            0.0
        };
        let r = if tr.reward.is_finite() {
            tr.reward
        } else {
            0.0
        };
        let delta = r + gamma * next_v - v_t;
        deltas.push(delta);
    }
    // Second pass: fold back with γλ discount.
    // Â_t = δ_t + γλ · Â_{t+1}. done_{t+1} handling is baked in
    // via the loop cutoff above — once we stopped at a boundary,
    // all later δ's are implicitly zero (no accumulation past
    // termination).
    let mut adv = 0.0f32;
    for &delta in deltas.iter().rev() {
        adv = delta + gamma * lambda * adv;
    }
    adv
}

/// Normalize active advantages either as one batch (`group_ids = None`) or
/// independently within each task/group. Groups with fewer than two active
/// samples are left unchanged: there is no relative baseline to estimate.
///
/// GRPO must compare forks of the same task. Centering returns from unrelated
/// games together leaks reward-scale and difficulty differences into the
/// gradient, so callers pass the lane's environment/task id for GRPO batches.
fn normalize_active_advantages(
    advantages: &mut [f32],
    active: &[bool],
    group_ids: Option<&[u32]>,
    divide_by_std: bool,
) {
    debug_assert_eq!(advantages.len(), active.len());
    if let Some(ids) = group_ids {
        debug_assert_eq!(advantages.len(), ids.len());
    }

    #[derive(Clone, Copy, Default)]
    struct Stats {
        count: usize,
        sum: f32,
        sq_sum: f32,
    }

    let group_for = |i: usize| group_ids.map_or(0, |ids| ids[i]);
    let mut stats: HashMap<u32, Stats> = HashMap::new();
    for (i, &value) in advantages.iter().enumerate() {
        if active[i] {
            let entry = stats.entry(group_for(i)).or_default();
            entry.count += 1;
            entry.sum += value;
        }
    }
    for (i, &value) in advantages.iter().enumerate() {
        if !active[i] {
            continue;
        }
        let entry = stats.get_mut(&group_for(i)).expect("group was collected");
        if entry.count >= 2 {
            let mean = entry.sum / entry.count as f32;
            let d = value - mean;
            entry.sq_sum += d * d;
        }
    }
    for (i, value) in advantages.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }
        let entry = stats.get(&group_for(i)).expect("group was collected");
        if entry.count < 2 {
            continue;
        }
        let mean = entry.sum / entry.count as f32;
        let std = if divide_by_std {
            (entry.sq_sum / entry.count as f32).sqrt().max(1e-3)
        } else {
            1.0
        };
        *value = (*value - mean) / std;
    }
}

fn restore_action_mask(dst: &mut [f32], stored: &[f32]) {
    dst.fill(1.0);
    if stored.len() == dst.len() {
        dst.copy_from_slice(stored);
    }
}

fn unit_cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let a_norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let b_norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if a_norm < 1e-6 || b_norm < 1e-6 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_norm * b_norm)
}

/// Xavier (Glorot uniform) initialization.
/// Apply learning rate to a meganeura `Session`, dispatching on
/// the AgentConfig::use_adam flag. SGD path uses set_learning_rate;
/// Adam path uses set_adam(lr, beta1, beta2, eps).
///
/// Adam epsilon defaults to 1e-8 (PyTorch standard) but can be raised
/// to 1e-4 or 1e-3 for sparse-reward visual tasks where v_t becomes
/// near-zero on idle parameters and then a sudden gradient causes
/// `update = lr · m / (sqrt(v) + eps)` to explode. A larger eps
/// effectively bounds the update magnitude when v is tiny.
fn apply_lr(session: &mut Session, lr: f32, use_adam: bool, adam_eps: f32) {
    if use_adam {
        session.set_adam(lr, 0.9, 0.999, adam_eps);
    } else {
        session.set_learning_rate(lr);
    }
}

fn xavier_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let n = fan_in * fan_out;
    (0..n)
        .map(|i| {
            let x = ((i as f64 + seed as f64) * 0.618_033_988_749_895).fract() as f32;
            (x * PI * 2.0).sin() * scale
        })
        .collect()
}

/// Build a meganeura training session sharing the agent's blade context.
///
/// Threading one `Arc<blade_graphics::Context>` through every `Session`
/// the `Agent` owns lets a sibling compute pipeline (the V2-S preprocess
/// in Track B.2) write directly into a meganeura input slot's underlying
/// blade buffer — no fd/dmabuf interop needed.  Without this all sessions
/// would each spin up an independent context and we'd be back to
/// cross-context fd marshaling.
fn build_session(g: &Graph, opt_level: OptLevel, gpu: &Arc<blade_graphics::Context>) -> Session {
    let cfg = meganeura::SessionConfig {
        gpu: Some(Arc::clone(gpu)),
        skip_full_optimize: matches!(opt_level, OptLevel::None),
        ..meganeura::SessionConfig::default()
    };
    meganeura::build(g, cfg).0
}

/// Build a meganeura inference-only session sharing the agent's blade
/// context.  Used for the V2-S sibling session.
fn build_inference_session(g: &Graph, gpu: &Arc<blade_graphics::Context>) -> Session {
    let cfg = meganeura::SessionConfig {
        mode: meganeura::Mode::Inference,
        gpu: Some(Arc::clone(gpu)),
        ..meganeura::SessionConfig::default()
    };
    meganeura::build(g, cfg).0
}

/// Cosine similarity between two equal-length f32 vectors. Returns 0
/// when either has zero norm (degenerate / zero-init case). Used by
/// the goal-conditioned planner scoring path.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    dot / denom
}

/// Maximum cosine similarity between `z` and any of the saved goal
/// latents for this env. Returns 0.0 when the goal queue is empty
/// (no wins yet for this env). The planner scoring loop calls this
/// per (rollout step × candidate) — kept tight because at K=4 lanes,
/// 32 samples, 5 steps, 100 goals it's 64k cos-sims per planning call.
fn max_goal_similarity(z: &[f32], goals: &std::collections::VecDeque<Vec<f32>>) -> f32 {
    let mut best = 0.0f32;
    for g in goals.iter() {
        let s = cosine_similarity(z, g);
        if s > best {
            best = s;
        }
    }
    best
}

/// Max cos-sim between `z` and any centroid in `centroids`.
/// Returns 0 if list is empty.
fn max_centroid_similarity(z: &[f32], centroids: &[Vec<f32>]) -> f32 {
    let mut best = 0.0f32;
    for c in centroids {
        let s = cosine_similarity(z, c);
        if s > best {
            best = s;
        }
    }
    best
}

/// Online k-means update: nudge nearest centroid toward `z`. If the
/// list has fewer than `k` entries, push `z` as a new centroid.
/// `lr` is the convex-combination weight (high = chase recent wins).
fn online_kmeans_update(centroids: &mut Vec<Vec<f32>>, z: &[f32], k: usize, lr: f32) {
    if k == 0 {
        return;
    }
    if centroids.len() < k {
        centroids.push(z.to_vec());
        return;
    }
    // Find nearest by cos-sim.
    let mut best_i = 0usize;
    let mut best_s = f32::NEG_INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let s = cosine_similarity(z, c);
        if s > best_s {
            best_s = s;
            best_i = i;
        }
    }
    // Pull centroid toward z.
    let c = &mut centroids[best_i];
    for d in 0..c.len().min(z.len()) {
        c[d] = (1.0 - lr) * c[d] + lr * z[d];
    }
}

/// Build a forward-only WM rollout graph: `(z, action) -> z_next`.
/// Used by the GPU-side model-based planner. `batch_size` here equals
/// `n_lanes × planner_samples` so the planner can score `planner_samples`
/// candidate sequences per lane in a single batched forward.
/// k-step WM training graph. Standalone session that does:
///   z_input → WM(z, a_0) → WM(z_1, a_1) → ... → z_hat_k
///   MSE(z_hat_k, z_target_k) → loss
/// Trains the WM params to be accurate at depth k. Synced to
/// wm_session after each step so the canonical wm_session WM
/// benefits from the deeper training signal.
fn build_wm_kstep_graph(
    batch_size: usize,
    latent_dim: usize,
    hidden_dim: usize,
    k: usize,
    wm_stochastic: bool,
) -> Graph {
    assert!(k >= 1, "wm_kstep k must be >= 1");
    let mut g = Graph::new();
    let z_input = g.input("z_input", &[batch_size, latent_dim]);
    let z_target = g.input("z_target_kstep", &[batch_size, latent_dim]);
    let z = g.stop_gradient(z_input);
    let z_target = g.stop_gradient(z_target);

    // k action inputs.
    let mut action_nodes = Vec::with_capacity(k);
    for i in 0..k {
        let name = format!("action_step_{}", i);
        let a = g.input(&name, &[batch_size, WM_ACTION_DIM]);
        let a = g.stop_gradient(a);
        action_nodes.push(a);
    }

    // k-step graph mirrors the main WM struct layout (including
    // sigma_proj when wm_stochastic is on) so parameter sync between
    // sessions stays valid. The k-step loss itself is still MSE on
    // the rolled-out μ — σ training only happens in the main
    // wm_session (single-step heteroscedastic regression).
    let wm = if wm_stochastic {
        WorldModel::new_stochastic(&mut g, latent_dim, WM_ACTION_DIM, hidden_dim)
    } else {
        WorldModel::new(&mut g, latent_dim, WM_ACTION_DIM, hidden_dim)
    };
    let z_hat_k = wm.rollout_k(&mut g, z, &action_nodes);
    let loss = g.mse_loss(z_hat_k, z_target);
    g.set_outputs(vec![loss, z_hat_k]);
    g
}

fn build_wm_planner_graph(
    batch_size: usize,
    latent_dim: usize,
    hidden_dim: usize,
    value_hidden_dim: Option<usize>,
    wm_stochastic: bool,
) -> Graph {
    let mut g = Graph::new();
    let z_raw = g.input("z", &[batch_size, latent_dim]);
    let action_raw = g.input("action", &[batch_size, WM_ACTION_DIM]);
    // Stop gradients on both inputs so set_outputs([z_next]) at the end
    // doesn't drive backward through the WM params — the planner is
    // forward-only.
    let z = g.stop_gradient(z_raw);
    let action = g.stop_gradient(action_raw);
    let wm = if wm_stochastic {
        WorldModel::new_stochastic(&mut g, latent_dim, WM_ACTION_DIM, hidden_dim)
    } else {
        WorldModel::new(&mut g, latent_dim, WM_ACTION_DIM, hidden_dim)
    };
    let (z_next, sigma_opt) = wm.forward_with_sigma(&mut g, z, action);
    let z_next = g.stop_gradient(z_next);
    // Output order: [z_next, optional sigma, optional value]
    // - z_next  always at index 0
    // - σ      at index 1 when wm_stochastic
    // - value  at the next index when value_hidden_dim is Some
    let mut outputs = vec![z_next];
    if let Some(sigma) = sigma_opt {
        let sigma_det = g.stop_gradient(sigma);
        outputs.push(sigma_det);
    }
    if let Some(vh) = value_hidden_dim {
        let vh_module = crate::value_head::ValueHead::new(&mut g, latent_dim, vh);
        let logit = vh_module.forward(&mut g, z_next);
        let prob = g.sigmoid(logit);
        let prob = g.stop_gradient(prob);
        outputs.push(prob);
    }
    g.set_outputs(outputs);
    g
}

/// Build a forward-only policy rollout graph: `z -> logits`.
/// Used alongside `wm_planner_session` for policy-guided planning —
/// at each WM rollout step the planner samples actions from this
/// policy's logits (per-lane, masked, optionally temperature-scaled)
/// instead of from uniform random. The trajectories still get
/// scored by latent-visit-count novelty as before; combining
/// "imagine policy executing" with novelty endpoint scoring is the
/// Dreamer-style approach to driving exploration toward novel
/// regions the policy can actually reach. Builds only the
/// `Policy` MLP (fc1+relu+fc2); no encoder, no value head, no
/// option machinery — those don't matter for k-step latent rollout.
fn build_policy_planner_graph(
    batch_size: usize,
    latent_dim: usize,
    hidden_dim: usize,
    num_options: usize,
) -> Graph {
    let mut g = Graph::new();
    let z_raw = g.input("z", &[batch_size, latent_dim]);
    let z = g.stop_gradient(z_raw);
    // Mirror the training policy graph: when num_options > 1, the
    // logits are shared-trunk + per-option additive bias. This is the
    // "simple" option path (matching build_policy_graph's else-branch
    // when per_option_heads=false). Required for option-aware planner
    // rollouts where each option produces distinct action distributions.
    let logits = if num_options > 1 {
        let policy = policy::Policy::new(&mut g, latent_dim, MAX_ACTION_DIM, hidden_dim);
        let trunk_logits = policy.forward(&mut g, z);
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let option_onehot = g.stop_gradient(option_onehot);
        let option_bias =
            nn::Linear::no_bias(&mut g, "policy.option_bias", num_options, MAX_ACTION_DIM);
        let bias_out = option_bias.forward(&mut g, option_onehot);
        g.add(trunk_logits, bias_out)
    } else {
        let policy = policy::Policy::new(&mut g, latent_dim, MAX_ACTION_DIM, hidden_dim);
        policy.forward(&mut g, z)
    };
    let logits = g.stop_gradient(logits);
    g.set_outputs(vec![logits]);
    g
}

/// Forward-only option-policy graph: `z -> option_logits`.
/// Companion to `build_policy_planner_graph` when num_options > 1.
/// Mirrors the relevant subgraph of `option::build_option_graph`
/// (option.trunk → relu → option.head) so the planner can sample
/// L1 options at each outer step.
fn build_option_planner_graph(
    batch_size: usize,
    latent_dim: usize,
    hidden_dim: usize,
    num_options: usize,
) -> Graph {
    let mut g = Graph::new();
    let z_raw = g.input("z", &[batch_size, latent_dim]);
    let z = g.stop_gradient(z_raw);
    let trunk = nn::Linear::new(&mut g, "option.trunk", latent_dim, hidden_dim);
    let h = trunk.forward(&mut g, z);
    let h = g.relu(h);
    let option_head = nn::Linear::no_bias(&mut g, "option.head", hidden_dim, num_options);
    let logits = option_head.forward(&mut g, h);
    let logits = g.stop_gradient(logits);
    g.set_outputs(vec![logits]);
    g
}

impl Agent {
    /// Build an N-lane agent. `adapters.len()` must equal `config.batch_size`;
    /// mismatching shapes panic at construction. For a single-lane agent,
    /// pass a single-element vec: `Agent::new(cfg, vec![adapter])`.
    ///
    /// The underlying graphs use universal token sizes and will never need
    /// rebuilding for subsequent lane-adapter swaps (`switch_lane`).
    pub fn new(config: AgentConfig, adapters: Vec<Box<dyn EnvAdapter>>) -> Self {
        assert!(
            !adapters.is_empty(),
            "Agent::new requires at least one adapter (one lane)"
        );
        assert_eq!(
            adapters.len(),
            config.batch_size,
            "adapters.len() ({}) must equal config.batch_size ({})",
            adapters.len(),
            config.batch_size
        );

        // --- Shared blade GPU context ---
        //
        // One context for *every* meganeura session this Agent owns
        // (and any sibling compute pipeline added later — see Track B.2
        // V2-S preprocess).  Without sharing, each `build_session` call
        // would `Context::init` a fresh instance and we'd lose the
        // ability to write between sessions without fd/dmabuf interop.
        let gpu: Arc<blade_graphics::Context> = Arc::new(
            meganeura::init_gpu_context()
                .expect("failed to initialize blade GPU context for Agent"),
        );

        // --- World model graph (uses universal token sizes + task) ---
        //
        // The task code is fed as a graph **input** named "task" and persisted
        // CPU-side in `task_embeddings`. It uses either historical dense hashes
        // or configured orthogonal common ids. The encoder learns to map
        // (obs_token, task) into per-env latents.
        let mut wm_session = {
            let mut g = Graph::new();
            let action = g.input("action", &[config.batch_size, WM_ACTION_DIM]);
            // Previous step's latent — the state the action was taken
            // FROM. Kept under the historical input name "z_target"
            // (set_input call sites and the python extension know it
            // by that name) but it is the WM's INPUT, not its loss
            // target: the WM rolls z_prev forward through `action`
            // and is trained against the FRESH encoder latent below.
            //
            // History: until 2026-06-10 the roles were swapped — the
            // WM consumed the fresh encoder latent and was trained to
            // predict this previous latent, i.e. a BACKWARD model
            // (z_{t+1}, a_t) → z_t, while every consumer (CPU/GPU
            // planner rollouts, MCTS, wm_kstep) assumed forward
            // dynamics. Planner action-conditioning was therefore
            // noise, which is consistent with the 2026-05-11 finding
            // that the policy never captured planner-driven paths and
            // the 2026-05-18 L3 wall ("WM accuracy at depth > 10").
            let z_prev = g.input("z_target", &[config.batch_size, config.latent_dim]);
            // Task embedding is fed as a graph **input**, not a parameter.
            // The encoder sees per-env conditioning and can specialize its
            // representations, but we don't backprop into the embedding
            // itself (meganeura's autodiff over a parameter on this code
            // path is unstable). Each env's code is fixed and keyed by
            // env_id; the encoder learns to map (obs_token, task_code) into
            // env-specific latents.
            let task = g.input("task", &[config.batch_size, TASK_DIM]);

            // `obs` (the OBS_TOKEN_DIM-wide pooled-token view) is declared
            // unconditionally so the optional WM-side recon branch below
            // can target it. The MLP encoder uses `obs` as its primary
            // input; CNN encoders ignore it for forward (they consume
            // `visual_obs`), but `obs` is still set every step by
            // `wm_forward_backward_stacked` so the recon target is fresh.
            let obs = g.input("obs", &[config.batch_size, OBS_TOKEN_DIM]);

            // Captured here so the recon branch below can target the
            // raw frame (rank ~30 globally, ~57 mean per-game) instead
            // of the 8×8 pooled obs token (rank ~8 globally, often 1-2
            // per-game). The pooled obs is too low-rank to force
            // higher-rank z; the raw frame isn't.
            let mut visual_node_2d: Option<NodeId> = None;
            let mut visual_recon_dim: usize = 0;

            // Encoder backbone: default MLP on an obs-token vector, or
            // a small CNN on raw visual input when configured.
            let z_t = match config.encoder_kind {
                EncoderKind::Mlp => {
                    let enc = Encoder::new(
                        &mut g,
                        OBS_TOKEN_DIM,
                        TASK_DIM,
                        config.latent_dim,
                        config.hidden_dim,
                    );
                    enc.forward(&mut g, obs, task)
                }
                EncoderKind::Cnn {
                    channels,
                    height,
                    width,
                } => {
                    // Flat NCHW visual input.
                    let per_sample = (channels as usize) * (height as usize) * (width as usize);
                    let flat_dim = per_sample * config.batch_size;
                    let visual = g.input("visual_obs", &[flat_dim]);
                    visual_node_2d = Some(g.reshape(visual, &[config.batch_size, per_sample]));
                    visual_recon_dim = per_sample;
                    let cnn = CnnEncoder::new(
                        &mut g,
                        channels,
                        height,
                        width,
                        config.latent_dim,
                        config.batch_size as u32,
                    );
                    let z_cnn = cnn.forward(&mut g, visual);
                    // Fold the task embedding in post-CNN via a tiny
                    // projection so the visual encoder still gets
                    // per-env conditioning without rebuilding the whole
                    // graph around task concat.
                    let task_proj = nn::Linear::no_bias(
                        &mut g,
                        "encoder.task_proj_cnn",
                        TASK_DIM,
                        config.latent_dim,
                    );
                    let task_h = task_proj.forward(&mut g, task);
                    g.add(z_cnn, task_h)
                }
                EncoderKind::CnnDqn {
                    channels,
                    height,
                    width,
                } => {
                    // Same wiring as Cnn but uses CnnEncoderDqn
                    // (Nature-DQN-scale, ~1.7M params, no global pool).
                    let per_sample = (channels as usize) * (height as usize) * (width as usize);
                    let flat_dim = per_sample * config.batch_size;
                    let visual = g.input("visual_obs", &[flat_dim]);
                    visual_node_2d = Some(g.reshape(visual, &[config.batch_size, per_sample]));
                    visual_recon_dim = per_sample;
                    let cnn = crate::encoder::CnnEncoderDqn::new(
                        &mut g,
                        channels,
                        height,
                        width,
                        config.latent_dim,
                        config.batch_size as u32,
                    );
                    let z_cnn = cnn.forward(&mut g, visual);
                    let task_proj = nn::Linear::no_bias(
                        &mut g,
                        "encoder.task_proj_cnn",
                        TASK_DIM,
                        config.latent_dim,
                    );
                    let task_h = task_proj.forward(&mut g, task);
                    g.add(z_cnn, task_h)
                }
                EncoderKind::EfficientNetV2S => {
                    // V2-S feeds a fixed (160, 12, 12) feature map into
                    // the same internal CnnEncoder used by `Cnn`. V2-S
                    // itself runs in a sibling session and uploads
                    // features into `visual_obs` via host copy before
                    // each WM step.
                    let channels = EFFICIENTNET_V2S_OUT_CHANNELS;
                    let height = EFFICIENTNET_V2S_OUT_HW;
                    let width = EFFICIENTNET_V2S_OUT_HW;
                    let per_sample = (channels as usize) * (height as usize) * (width as usize);
                    let flat_dim = per_sample * config.batch_size;
                    let visual = g.input("visual_obs", &[flat_dim]);
                    visual_node_2d = Some(g.reshape(visual, &[config.batch_size, per_sample]));
                    visual_recon_dim = per_sample;
                    let cnn = CnnEncoder::new(
                        &mut g,
                        channels,
                        height,
                        width,
                        config.latent_dim,
                        config.batch_size as u32,
                    );
                    let z_cnn = cnn.forward(&mut g, visual);
                    let task_proj = nn::Linear::no_bias(
                        &mut g,
                        "encoder.task_proj_cnn",
                        TASK_DIM,
                        config.latent_dim,
                    );
                    let task_h = task_proj.forward(&mut g, task);
                    g.add(z_cnn, task_h)
                }
            };
            let wm = if config.wm_stochastic {
                WorldModel::new_stochastic(
                    &mut g,
                    config.latent_dim,
                    WM_ACTION_DIM,
                    config.hidden_dim,
                )
            } else {
                WorldModel::new(&mut g, config.latent_dim, WM_ACTION_DIM, config.hidden_dim)
            };
            // Forward dynamics: ẑ_{t+1} = WM(z_t, a_t), trained against
            // the fresh encoder latent of the post-action observation.
            // stop_grad on the target keeps the WM loss off the encoder
            // (Dreamer-style: the representation is shaped by the recon
            // branch + value head, the dynamics model chases it). Rows
            // at an episode boundary stage z_prev = 0, so the WM learns
            // a "from the null state, actions lead to episode-start
            // states" manifold instead of polluting real transitions.
            let (z_hat, sigma_opt) = wm.forward_with_sigma(&mut g, z_prev, action);
            let z_next_target = g.stop_gradient(z_t);
            let mu_loss = WorldModel::loss(&mut g, z_hat, z_next_target);
            let wm_loss = match sigma_opt {
                Some(sigma) => {
                    // Heteroscedastic σ-head regression loss. Trained
                    // against |z_next − ẑ| with stop-grad on the
                    // residual so the σ training does not bias μ.
                    let sigma_loss = WorldModel::sigma_loss(&mut g, z_hat, sigma, z_next_target);
                    let coef = g.scalar(config.wm_sigma_loss_coef);
                    let sigma_loss_scaled = g.mul(sigma_loss, coef);
                    g.add(mu_loss, sigma_loss_scaled)
                }
                None => mu_loss,
            };

            // Optional obs-reconstruction loss INSIDE the WM session
            // (anti-collapse). Without this, the WM forward-prediction
            // loss admits a trivial low-rank z (verified 2026-05-05:
            // 25-game multi-training collapsed encoder to effective
            // rank 2 of 256 dims). Forcing z to retain enough info to
            // reconstruct the encoder's input forces higher-rank
            // features.
            //
            // Target selection (verified 2026-05-06):
            //   * MLP encoder → target the OBS_TOKEN_DIM pooled `obs`
            //     (it's the encoder's only input).
            //   * CNN/CNN-DQN encoder + `recon_visual_target = false`
            //     → target the OBS_TOKEN_DIM pooled `obs`. Cheap
            //     decoder, but the pooled view is rank ~8 globally and
            //     1–2 per-game on static games — too low-rank to force
            //     the encoder past the rank-2 collapse on its own.
            //   * CNN/CNN-DQN encoder + `recon_visual_target = true`
            //     → target the reshaped raw `visual_obs` (rank ~30
            //     globally, mean ~57 per-game). The decoder must
            //     reproduce frame-level structure, which a rank-2 z
            //     cannot. Decoder size: ~1M params at 64×64×1.
            //
            // Distinct from the `recon.*` decoder built by
            // `policy::build_policy_graph_e2e` — that one only fires
            // in e2e mode and operates on the policy session's
            // separate encoder. This branch operates on the WM
            // session's encoder (which produces the latents stored
            // in lane buffers and used for novelty / surprise / option
            // goals).
            let (wm_recon_loss, recon_loss_raw) = if config.recon_loss_coef > 0.0 {
                let (target_node, target_dim) = match (config.recon_visual_target, visual_node_2d) {
                    (true, Some(v2d)) => (v2d, visual_recon_dim),
                    _ => (obs, OBS_TOKEN_DIM),
                };
                let dec_fc1 =
                    nn::Linear::new(&mut g, "wm.recon.fc1", config.latent_dim, config.hidden_dim);
                let dec_fc2 =
                    nn::Linear::no_bias(&mut g, "wm.recon.fc2", config.hidden_dim, target_dim);
                let dh = dec_fc1.forward(&mut g, z_t);
                let dh = g.relu(dh);
                let recon = dec_fc2.forward(&mut g, dh);
                let target_det = g.stop_gradient(target_node);
                let recon_loss_raw = g.mse_loss(recon, target_det);
                let recon_coef = g.scalar(config.recon_loss_coef);
                let recon_loss = g.mul(recon_loss_raw, recon_coef);
                (g.add(wm_loss, recon_loss), recon_loss_raw)
            } else {
                (wm_loss, g.scalar(0.0))
            };

            // Win-classifier head atop the encoder output. Built when
            // training (`value_head_train_coef > 0`) or planner
            // consumption (`planner_value_alpha > 0`) is on. When OFF,
            // the graph is byte-identical to the pre-classifier
            // version so baseline behaviour is unaffected.
            //
            // Mechanism: V(z) → logit, sigmoid → P(win-trajectory | z),
            // BCE loss vs label γ^(T-t) for win-trajectory transitions,
            // 0 for loss-trajectory transitions.
            //
            // Input routing (2026-06-10): the head reads a dedicated
            // `value_z` input — the STORED latent of the buffered
            // win/loss sample — not the encoder output. The previous
            // wiring read enc(current visual/obs), but value-replay
            // batches stage archived samples whose frames are NOT in
            // the visual slot: in CNN modes the classifier was being
            // trained on stale online frames against unrelated
            // archived labels, so it could only learn a per-game win
            // prior through the task channel (the 2026-05-15
            // cross-game-bias finding, explained). Training on stored
            // latents is exact, encoder-mode-independent, and the
            // planner consumption (V over rollout latents via pulled
            // value_head.* weights) is unchanged — the in-graph V has
            // no other online consumer. Trade-off given up: BCE
            // encoder shaping; the only mode where that shaping ever paired
            // labels with the right states was MLP.
            //
            // Why BCE not MSE on R_to_go: an earlier regression
            // formulation (2026-05-12) collapsed the encoder because
            // R_to_go ≈ 0 dominates (loss episodes) and MSE pulled all
            // latents toward "predict zero". BCE's gradient does NOT
            // vanish for confident negatives.
            let value_branch_on =
                config.value_head_train_coef > 0.0 || config.planner_value_alpha > 0.0;
            let loss = if value_branch_on {
                let vh_hidden = if config.value_head_hidden_dim == 0 {
                    config.hidden_dim
                } else {
                    config.value_head_hidden_dim
                };
                let value_target_in = g.input("value_target", &[config.batch_size, 1]);
                let value_z_in = g.input("value_z", &[config.batch_size, config.latent_dim]);
                let wm_gate_in = g.input("wm_gate", &[1]);
                let value_gate_in = g.input("value_gate", &[1]);
                let wm_gate = g.stop_gradient(wm_gate_in);
                let value_gate = g.stop_gradient(value_gate_in);
                let vh_module =
                    crate::value_head::ValueHead::new(&mut g, config.latent_dim, vh_hidden);
                let logit = vh_module.forward(&mut g, value_z_in);
                let prob = g.sigmoid(logit);
                let v_target_det = g.stop_gradient(value_target_in);
                let value_loss_raw = g.bce_loss(prob, v_target_det);
                let value_coef_scalar = g.scalar(config.value_head_train_coef);
                let value_loss = g.mul(value_loss_raw, value_coef_scalar);
                let wm_gated = g.mul(wm_recon_loss, wm_gate);
                let value_gated = g.mul(value_loss, value_gate);
                g.add(wm_gated, value_gated)
            } else {
                wm_recon_loss
            };

            // Per-lane squared-error output. Exposing `(z_hat − z_target)²`
            // as `[N, latent_dim]` lets us compute per-lane prediction error
            // (and therefore per-lane surprise reward) on the CPU without
            // mutating the scalar `loss` used for backprop. This is the
            // Phase E.v2 "per-lane surprise" hook that the design doc
            // flagged as the thing to build when the shared mean-loss
            // surrogate starts hurting at large N.
            //
            // We use `z_next − z_hat` (i.e. `add(z_next, neg(z_hat))`)
            // instead of the reverse to keep `z_hat`'s primary consumer the
            // mse_loss node — meganeura's forward-optimize can still fuse
            // the loss path cleanly without fighting for the `z_hat`
            // buffer. Per-lane surprise is therefore the error of the
            // FORWARD prediction: "how wrong was the WM about where
            // this action would lead."
            let neg_zhat = g.neg(z_hat);
            let diff = g.add(z_next_target, neg_zhat);
            let sq = g.mul(diff, diff);
            // Output 3 (z_hat) backs `Agent::wm_predict`. Outputs 4 and 5
            // keep prediction and reconstruction metrics separate from the
            // combined optimization objective at output 0. Appending keeps
            // indices 1 (z_t) and 2 (sq) stable for existing readbacks.
            g.set_outputs(vec![loss, z_t, sq, z_hat, mu_loss, recon_loss_raw]);
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            s
        };

        // --- EfficientNetV2-S session (sibling, runs before WM each step) ---
        let (efficientnet_session, efficientnet_input_size_bytes, efficientnet_output_buf) =
            match config.encoder_kind {
                EncoderKind::EfficientNetV2S => {
                    let weights_path = config
                        .efficientnet_weights_path
                        .as_ref()
                        .expect(
                            "AgentConfig::efficientnet_weights_path must be set when \
                             encoder_kind = EfficientNetV2S",
                        )
                        .clone();
                    let mut g = Graph::new();
                    let out = meganeura::models::efficientnet::build_graph(
                        &mut g,
                        config.batch_size as u32,
                    );
                    g.set_outputs(vec![out]);
                    let mut s = build_inference_session(&g, &gpu);
                    let weights =
                        meganeura::data::safetensors::SafeTensorsModel::load(weights_path.clone())
                            .unwrap_or_else(|e| {
                                panic!(
                                    "EfficientNetV2-S: loading weights from {:?}: {}",
                                    weights_path, e
                                )
                            });
                    for name in meganeura::models::efficientnet::weight_names() {
                        let data = weights.tensor_f32(&name).unwrap_or_else(|e| {
                            panic!("EfficientNetV2-S: parameter '{name}': {e}")
                        });
                        s.set_parameter(&name, &data);
                    }
                    let input_size_bytes = (config.batch_size
                        * (EFFICIENTNET_V2S_IN_CHANNELS as usize)
                        * (EFFICIENTNET_V2S_IN_HW as usize)
                        * (EFFICIENTNET_V2S_IN_HW as usize))
                        * std::mem::size_of::<f32>();
                    let output_size = config.batch_size
                        * (EFFICIENTNET_V2S_OUT_CHANNELS as usize)
                        * (EFFICIENTNET_V2S_OUT_HW as usize)
                        * (EFFICIENTNET_V2S_OUT_HW as usize);
                    (Some(s), input_size_bytes, vec![0.0f32; output_size])
                }
                _ => (None, 0, Vec::new()),
            };

        // --- V2-S preprocess pipeline (shared blade context) ---
        //
        // Built eagerly when V2-S is in use so callers can `register_lane`
        // / `v2s_preprocess_step` without a follow-up "is the pipeline
        // ready?" branch.  When the user picks a non-V2-S encoder the
        // pipeline stays `None` and the legacy `image_input_host_ptr`
        // path remains available.  See `src/v2s_preprocess.rs` and
        // `src/shaders/v2s_preprocess.wgsl` for the GPU side.
        let v2s_preprocess: Option<crate::v2s_preprocess::PreprocessPipeline> =
            efficientnet_session.as_ref().map(|sess| {
                let dst_image = sess
                    .input_buffer("image")
                    .expect("V2-S session must have an `image` input slot");
                crate::v2s_preprocess::PreprocessPipeline::new(
                    Arc::clone(&gpu),
                    dst_image,
                    config.batch_size,
                )
            });

        // --- Policy + value graph ---
        // Pick the policy graph shape from the adapters' action kinds. All
        // adapters must share a kind (agent-wide single policy session), so
        // we check all of them and pick:
        //
        //   Discrete    → cross_entropy_loss(logits, one_hot_action).
        //                 The gradient w.r.t. logits is
        //                 `softmax(logits) − one_hot`, which gives every
        //                 logit a well-conditioned update every step and
        //                 lets the softmax actually narrow toward preferred
        //                 actions. This is what kindle should use for any
        //                 all-discrete setup.
        //
        //   Continuous  → mse_loss(mean, action). Gaussian-NLL with fixed
        //                 unit variance reduces to MSE up to a constant,
        //                 which is the right loss for continuous actions.
        //
        // Mixed-kind adapter sets aren't currently supported because one
        // compiled graph has one loss op. If that ever matters, we'd route
        // per-lane into one of two graphs (different shape) — out of scope
        // here.
        let first_kind = adapters[0].action_kind();
        for (i, a) in adapters.iter().enumerate().skip(1) {
            assert!(
                kinds_match(first_kind, a.action_kind()),
                "Agent::new: all adapters must share the same ActionKind \
                 variant (lane 0 is {:?}, lane {} is {:?}). Mixed discrete/\
                 continuous in one batched session is not supported.",
                first_kind,
                i,
                a.action_kind()
            );
        }
        let is_discrete = matches!(first_kind, ActionKind::Discrete { .. });
        let continuous_dim = match first_kind {
            ActionKind::Continuous { dim, .. } => dim,
            ActionKind::Discrete { .. } => 0,
        };
        assert!(
            is_discrete || !config.use_ppo,
            "use_ppo is not implemented for continuous policies"
        );
        assert!(
            is_discrete || !config.use_kl_ppo,
            "use_kl_ppo is not implemented for continuous policies"
        );
        assert!(
            is_discrete || !config.use_sil,
            "SIL replay stores categorical action indices and is not implemented for continuous policies"
        );
        // Static padding mask: policy graphs are compiled at
        // MAX_ACTION_DIM, but each discrete adapter may expose fewer real
        // actions. Start padded heads invalid even when the environment has
        // no dynamic mask API. Dynamic environments can overwrite these rows
        // through `set_action_masks[_from_envs]` before act().
        let mut initial_action_masks = vec![1.0; config.batch_size * MAX_ACTION_DIM];
        if is_discrete {
            initial_action_masks.fill(0.0);
            for (lane, adapter) in adapters.iter().enumerate() {
                let ActionKind::Discrete { n: action_count } = adapter.action_kind() else {
                    unreachable!("mixed action kinds rejected above")
                };
                let valid = action_count.min(MAX_ACTION_DIM);
                initial_action_masks[lane * MAX_ACTION_DIM..lane * MAX_ACTION_DIM + valid]
                    .fill(1.0);
            }
        }

        // Resolve option_dim: 0 in config means "use latent_dim". This
        // is the dimensionality of the `option_goal` vector used by the
        // goal-alignment reward bonus; it is no longer mixed into the
        // policy's `z` input (Phase G v2 uses a per-option bias head
        // instead, which is plumbed through `option_onehot`).
        let option_dim = if config.option_dim == 0 {
            config.latent_dim
        } else {
            config.option_dim
        };
        let l1_active = config.num_options >= 2;
        // The option-aware planner (build_policy_planner_graph) uses
        // the shared-trunk + option_bias path. If per_option_heads is
        // true, policy_session uses per_option_fc2 (no shared fc2)
        // and param sync to policy_planner fails. Reject the incompatible
        // configuration rather than silently changing the requested model.
        assert!(
            !(l1_active && config.planner_horizon > 0 && config.per_option_heads),
            "per_option_heads=true is incompatible with option-aware planning; \
             set per_option_heads=false or planner_horizon=0"
        );
        assert!(
            !(config.end_to_end_encoder
                && config.planner_horizon > 0
                && (config.planner_policy_mix > 0.0 || l1_active)),
            "end_to_end_encoder uses a separate policy latent space and cannot \
             guide world-model planning; disable policy guidance/options, or \
             use the shared WM encoder policy path"
        );

        // L0 policy graph — `z` is just the encoder latent. When L1 is
        // active, the graph also takes a one-hot `option_onehot` input
        // and routes it through a direct-to-logits bias head (see
        // `policy::build_policy_graph`).
        // Policy graph's batch_size covers the rollout: `lanes ×
        // rollout_length` rows per `session.step()`. At the default
        // `rollout_length = 1` this collapses to `lanes`, identical
        // to the pre-rollout-buffer behavior. Other graphs (WM and
        // option) stay at `lanes` — they update per env-
        // step on lane-current state, not on a rollout.
        let rollout_length = config.rollout_length.max(1);
        let policy_batch = config.batch_size * rollout_length;
        let obs_scratch_rows = if config.end_to_end_encoder {
            policy_batch
        } else {
            config.batch_size
        };
        let mut policy_session = {
            // PPO + end_to_end_encoder: restored 2026-05-01 with the
            // post-autodiff-bug-fix grad path. Validate behavior carefully
            // — historical failure mode documented in
            // docs/failed_experiments.md.
            assert!(
                !config.use_kl_ppo || config.end_to_end_encoder,
                "use_kl_ppo currently requires end_to_end_encoder = true"
            );
            assert!(
                !config.end_to_end_encoder || config.n_step.max(1) >= 2,
                "end_to_end_encoder requires n_step >= 2: the single-step \
                 policy path feeds latents through the `z` input, which e2e \
                 graphs don't have (it would panic mid-training instead)"
            );
            assert!(
                config.rollout_length.max(1) < 2 || config.n_step.max(1) >= 2,
                "rollout_length > 1 requires n_step >= 2: the rollout buffer \
                 trains through the n-step path (the single-step path would \
                 leave the padded rollout rows training the value head \
                 toward zero)"
            );
            assert!(
                !(config.use_kl_ppo && config.use_ppo),
                "use_kl_ppo and use_ppo are mutually exclusive"
            );
            assert!(
                !(config.use_kl_ppo && config.num_options > 1),
                "use_kl_ppo + L1 options not implemented"
            );
            // use_grpo + use_ppo is the canonical DeepSeek-R1 GRPO
            // formulation: PPO-clipped surrogate with GRPO advantage
            // (cross-batch normalized n-step return, no V baseline).
            // use_grpo + use_kl_ppo is also allowed and gives the
            // KL-penalty PPO surrogate with GRPO advantage. When use_grpo
            // is set alone, the plain-PG e2e graph is used (no clip,
            // no KL — just advantage-weighted CE with GRPO advantage).
            assert!(
                !(config.use_grpo && config.use_kl_ppo && config.use_ppo),
                "all three of use_grpo + use_ppo + use_kl_ppo cannot be set"
            );
            assert!(
                !config.use_grpo || config.advantage_normalize,
                "use_grpo requires advantage_normalize = true (within-task \
                 mean/std normalization IS the GRPO advantage definition)"
            );
            let g = if is_discrete && config.end_to_end_encoder && config.use_kl_ppo {
                policy::build_kl_policy_graph_e2e(
                    OBS_TOKEN_DIM,
                    TASK_DIM,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.latent_dim,
                    policy_batch,
                    config.kl_beta,
                    config.value_loss_coef,
                    config.value_clip_scale,
                )
            } else if is_discrete && config.end_to_end_encoder && config.use_ppo {
                assert!(
                    config.num_options <= 1,
                    "use_ppo is not compatible with num_options > 1 (L1 options)"
                );
                policy::build_ppo_policy_graph_e2e(
                    OBS_TOKEN_DIM,
                    TASK_DIM,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.latent_dim,
                    policy_batch,
                    config.ppo_clip_eps,
                    config.value_loss_coef,
                    config.entropy_beta,
                    config.value_clip_scale,
                )
            } else if is_discrete && config.end_to_end_encoder {
                // L1 options + e2e: now supported. The e2e graph routes
                // option_onehot through per_option_fc2 (when
                // per_option_heads=true) or shared-trunk + per-option
                // bias (when false), matching the non-e2e
                // build_policy_graph routing.
                policy::build_policy_graph_e2e(
                    OBS_TOKEN_DIM,
                    TASK_DIM,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.latent_dim,
                    policy_batch,
                    config.entropy_beta,
                    config.value_loss_coef,
                    config.value_clip_scale,
                    config.num_options,
                    config.per_option_heads,
                    config.recon_loss_coef,
                    config.reward_pred_loss_coef,
                )
            } else if is_discrete && config.use_ppo {
                // PPO with optional L1 options (added 2026-05-06).
                // When num_options > 1, the PPO graph routes
                // option_onehot through the same option-conditional
                // logits structure as build_policy_graph. Both old
                // and new policies see the same option (the one
                // chosen at collection time), so the PPO ratio
                // π_new(a | z, opt) / π_old(a | z, opt) is well
                // defined.
                policy::build_ppo_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    policy_batch,
                    config.ppo_clip_eps,
                    config.value_loss_coef,
                    config.entropy_beta,
                    config.value_clip_scale,
                    config.policy_z_layer_norm,
                    config.policy_z_layer_norm_scale,
                    config.num_options,
                    config.per_option_heads,
                )
            } else if is_discrete {
                policy::build_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    policy_batch,
                    config.entropy_beta,
                    config.num_options,
                    config.per_option_heads,
                    config.value_loss_coef,
                    config.value_clip_scale,
                )
            } else if config.end_to_end_encoder {
                policy::build_continuous_policy_graph_e2e(
                    OBS_TOKEN_DIM,
                    TASK_DIM,
                    MAX_ACTION_DIM,
                    continuous_dim,
                    config.hidden_dim,
                    config.latent_dim,
                    policy_batch,
                    config.num_options,
                    config.per_option_heads,
                    config.value_loss_coef,
                    config.value_clip_scale,
                    config.recon_loss_coef,
                    config.reward_pred_loss_coef,
                )
            } else {
                policy::build_continuous_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    continuous_dim,
                    config.hidden_dim,
                    policy_batch,
                    config.num_options,
                    config.per_option_heads,
                    config.value_loss_coef,
                    config.value_clip_scale,
                )
            };
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            // Task #259: enable gradient norm clipping on the policy session
            // to prevent the value head from saturating into NaN. The MSE
            // gradient on a saturated scaled_tanh keeps the pre-tanh logit
            // growing unboundedly until activations overflow; clipping the
            // total grad norm caps each per-step update so the logit stays
            // in the linear regime. 10.0 lets policy_step_batched make
            // meaningful progress while still bounding the worst-case
            // value-head excursion.
            s.set_grad_clip_norm(10.0);
            s
        };

        // M7 approach-reward state. Constructed only when
        // `approach_reward_alpha > 0`. Cheap CPU data structure.
        // Coord-action head. Constructed only when
        // `coord_action_alpha > 0`. Cheap CPU MLP.
        let coord_head = if config.coord_action_alpha > 0.0 {
            let lr = config.coord_lr.unwrap_or(config.learning_rate * 0.3);
            Some(coord::CoordHead::new(
                OBS_TOKEN_DIM + TASK_DIM,
                config.coord_hidden_dim,
                config.batch_size,
                lr,
                config.coord_sigma,
                0xC001_0D25_DECA_FBADu64,
            ))
        } else {
            None
        };

        // RND curiosity state. Constructed only when
        // `rnd_reward_alpha > 0` so the default agent pays no
        // per-step overhead. RND operates on the obs TOKEN
        // (64-dim, pre-encoder) — see the docstring on
        // `rnd_reward_alpha` for the rationale.
        let rnd_state = if config.rnd_reward_alpha > 0.0 {
            let lr = config.rnd_lr.unwrap_or(config.learning_rate * 0.3);
            Some(rnd::RndState::new(
                OBS_TOKEN_DIM,
                config.rnd_feature_dim,
                config.rnd_hidden_dim,
                lr,
                0x42_BEEF_D15E_A53Eu64 ^ (config.batch_size as u64).wrapping_mul(0x9E37_79B9),
            ))
        } else {
            None
        };

        // DIAYN discriminator: built when intrinsic reward weight > 0
        // AND L1 has at least 2 options (the discriminator predicts
        // option-from-z, so 1 option would be degenerate).
        let diayn_state = if config.diayn_reward_alpha > 0.0 && config.num_options >= 2 {
            let lr = config.diayn_lr.unwrap_or(config.lr_option);
            Some(diayn::DiaynState::new(
                config.latent_dim,
                config.num_options,
                config.diayn_hidden_dim,
                lr,
                0xD1AB_1234_5678_9ABCu64 ^ (config.batch_size as u64).wrapping_mul(0x9E37_79B9),
            ))
        } else {
            None
        };

        let approach_state = if config.approach_reward_alpha > 0.0 {
            Some(approach::ApproachState::new(
                config.latent_dim,
                config.approach_buffer_size,
                config.approach_top_frac,
                config.approach_update_interval,
                config.approach_warmup_episodes,
            ))
        } else {
            None
        };

        // M8 delta-goal bank. Shared across lanes so all of them
        // benefit from discoveries by any one lane. Feature vector
        // is the obs TOKEN (OBS_TOKEN_DIM), NOT the post-encoder
        // latent: a well-trained encoder compresses state into a
        // narrow region where per-step latent-deltas fall below
        // any useful threshold (the same saturation RND hit, see
        // `rnd_reward_alpha` doc). The obs token carries the raw
        // per-frame variation that "something just changed in the
        // world" should key off.
        let delta_goal_bank = if config.delta_goal_alpha > 0.0 {
            Some(delta_goals::DeltaGoalBank::new(
                OBS_TOKEN_DIM,
                config.delta_goal_bank_size,
                config.delta_goal_threshold,
                config.delta_goal_merge_radius,
            ))
        } else {
            None
        };

        // Cross-episode state-action memory. Keyed on the encoder
        // latent (post-encoder), quantized by
        // `xeps_grid_resolution` (or `grid_resolution` when
        // unspecified). Shared across lanes — lanes pool
        // exploration credit.
        let xeps_memory = if config.xeps_reward_alpha > 0.0 {
            let res = config
                .xeps_grid_resolution
                .unwrap_or(config.grid_resolution);
            Some(xeps_memory::StateActionMemory::new(res))
        } else {
            None
        };

        // Track 3 model-based planner. Allocated when horizon > 0.
        // Weights are zero-initialized; the first `plan_and_queue`
        // call triggers a refresh that pulls the current WM state.
        let planner = if config.planner_horizon > 0 {
            Some(planner::WmRollout::new(
                config.latent_dim,
                WM_ACTION_DIM,
                config.hidden_dim,
            ))
        } else {
            None
        };
        let planner_samples_cached = config.planner_samples.max(1);
        let planner_batch = if config.planner_horizon > 0 {
            config.batch_size * planner_samples_cached
        } else {
            0
        };
        let vh_hidden_dim_resolved = if config.value_head_hidden_dim == 0 {
            config.hidden_dim
        } else {
            config.value_head_hidden_dim
        };
        let wm_planner_has_value_head =
            config.planner_horizon > 0 && config.planner_value_alpha > 0.0;
        let wm_planner_session = if config.planner_horizon > 0 {
            let g = build_wm_planner_graph(
                planner_batch,
                config.latent_dim,
                config.hidden_dim,
                if wm_planner_has_value_head {
                    Some(vh_hidden_dim_resolved)
                } else {
                    None
                },
                config.wm_stochastic,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };
        // Policy-guided planner is always on when the planner is on.
        // Cheap to build (just a 2-layer MLP) and dramatically
        // changes the planner's effective behavior — trajectories
        // follow the current policy instead of pure random.
        let policy_planner_session = if config.planner_horizon > 0 {
            let g = build_policy_planner_graph(
                planner_batch,
                config.latent_dim,
                config.hidden_dim,
                config.num_options,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };
        // k-step WM training session. Built when wm_kstep_k > 1.
        let wm_kstep_session = if config.wm_kstep_k > 1 {
            let k = config.wm_kstep_k;
            let batch = config.wm_kstep_batch.max(1);
            let g = build_wm_kstep_graph(
                batch,
                config.latent_dim,
                config.hidden_dim,
                k,
                config.wm_stochastic,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            if config.grad_clip_norm > 0.0 {
                s.set_grad_clip_norm(config.grad_clip_norm);
                s.set_grad_clip_every(config.grad_clip_every.max(1));
            }
            Some(s)
        } else {
            None
        };
        let kstep_z_scratch = if config.wm_kstep_k > 1 {
            vec![0.0f32; config.wm_kstep_batch.max(1) * config.latent_dim]
        } else {
            Vec::new()
        };
        let kstep_z_target_scratch = kstep_z_scratch.clone();
        let kstep_action_scratch_per_step = if config.wm_kstep_k > 1 {
            (0..config.wm_kstep_k)
                .map(|_| vec![0.0f32; config.wm_kstep_batch.max(1) * WM_ACTION_DIM])
                .collect()
        } else {
            Vec::new()
        };
        // Option-policy forward-only session for option-aware planning.
        // Built when planner is on AND L1 options are active.
        let option_planner_session = if config.planner_horizon > 0 && config.num_options >= 2 {
            let g = build_option_planner_graph(
                planner_batch,
                config.latent_dim,
                config.hidden_dim,
                config.num_options,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };
        // MCTS uses a smaller batch (n_lanes, one row per lane per
        // simulation step). Only build when MCTS is enabled.
        let wm_mcts_session = if config.planner_horizon > 0 && config.planner_use_mcts {
            let g = build_wm_planner_graph(
                config.batch_size,
                config.latent_dim,
                config.hidden_dim,
                if wm_planner_has_value_head {
                    Some(vh_hidden_dim_resolved)
                } else {
                    None
                },
                config.wm_stochastic,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };
        let planner_z_scratch = vec![0.0f32; planner_batch * config.latent_dim];
        let planner_action_scratch = vec![0.0f32; planner_batch * WM_ACTION_DIM];
        let planner_option_onehot_scratch = if config.num_options >= 2 && config.planner_horizon > 0
        {
            vec![0.0f32; planner_batch * config.num_options]
        } else {
            Vec::new()
        };
        let planner_traj_scratch =
            vec![0.0f32; planner_batch * config.planner_horizon.max(1) * config.latent_dim];
        let planner_v_traj_scratch = if wm_planner_has_value_head {
            vec![0.0f32; planner_batch * config.planner_horizon.max(1)]
        } else {
            Vec::new()
        };

        // Value-head training lives inside `wm_session` (encoder
        // shared). `value_replay_step` runs wm_session in
        // "value gate" mode using (obs, env_id, R_to_go) samples
        // drawn from `value_buffer`. The standalone z-input value
        // session was removed in favour of this end-to-end path.
        let value_target_scratch_vh = vec![0.0f32; config.batch_size];
        let value_z_scratch = vec![0.0f32; config.batch_size * config.latent_dim];
        let wm_gate_scalar = vec![1.0f32];
        let value_gate_scalar = vec![0.0f32];
        // Staging buffer for cross-session parameter syncs (planner
        // weight refresh, kstep WM sync). Sized as max_dim² where
        // max_dim covers every dimension that appears in a synced
        // param shape — hidden_dim × hidden_dim alone under-sizes it
        // whenever latent_dim > hidden_dim (e.g. latent 256 / hidden
        // 128) or hidden_dim < MAX_ACTION_DIM, which panicked on the
        // first weight refresh.
        let planner_param_dim = config
            .hidden_dim
            .max(config.latent_dim)
            .max(WM_ACTION_DIM)
            .max(config.num_options.max(1))
            .max(config.value_head_hidden_dim);
        let planner_param_buf = vec![0.0f32; planner_param_dim * planner_param_dim];

        // M6 outcome-value head (CPU MLP). Constructed only when the
        // user has asked for a non-zero bonus weight. Its LR derives
        // from `lr_outcome` or falls back to `learning_rate × 0.3`.
        let outcome_head = if config.outcome_reward_alpha > 0.0 {
            let lr = config.lr_outcome.unwrap_or(config.learning_rate * 0.3);
            Some(outcome::OutcomeHead::new(
                config.latent_dim,
                config.outcome_window.max(1),
                config.hidden_dim,
                lr,
                0xA11CE ^ 0xD0C_EFF,
            ))
        } else {
            None
        };

        // L1 option-policy session — only built when num_options >= 2.
        let mut option_session = if l1_active {
            let g = option::build_option_graph(
                config.latent_dim,
                config.num_options,
                config.hidden_dim,
                config.batch_size,
                config.option_entropy_beta,
            );
            let mut s = build_session(&g, config.opt_level, &gpu);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };

        // Apply gradient norm clipping to every training session if the
        // user opted in. This is sticky — set once at agent build, not
        // toggled per-step. Sessions without param_grad_pairs (forward
        // only) are unaffected by the setter.
        if config.grad_clip_norm > 0.0 {
            let n = config.grad_clip_norm;
            let every = config.grad_clip_every.max(1);
            wm_session.set_grad_clip_norm(n);
            wm_session.set_grad_clip_every(every);
            policy_session.set_grad_clip_norm(n);
            policy_session.set_grad_clip_every(every);
            if let Some(ref mut s) = option_session {
                s.set_grad_clip_norm(n);
                s.set_grad_clip_every(every);
            }
        }

        let n = config.batch_size;

        // Initialize per-env task embeddings (one TASK_DIM vector per env_id)
        // for every env id present in the initial lane set.
        let mut task_embeddings: HashMap<u32, Vec<f32>> = HashMap::new();
        for adapter in &adapters {
            task_embeddings.entry(adapter.id()).or_insert_with(|| {
                embedding_for(adapter.id(), TASK_DIM, config.orthogonal_task_codes)
            });
        }

        // Build per-lane state. Lanes for the same task share the reward
        // circuit projection seed so synchronous forks receive comparable
        // intrinsic rewards; recurrent digest state remains lane-local.
        let lanes: Vec<Lane> = adapters
            .into_iter()
            .map(|adapter| {
                let reward_seed = 0xA11CE ^ adapter.id() as u64;
                Lane {
                    adapter,
                    buffer: ExperienceBuffer::with_visit_count_config(
                        config.buffer_capacity,
                        config.grid_resolution,
                        config.visit_counts_max,
                        config.visit_count_dims,
                        config.visit_count_proj_dim,
                        config.visit_count_proj_seed,
                    ),
                    reward_circuit: RewardCircuit::with_seed(
                        config.reward_weights.clone(),
                        reward_seed,
                    ),
                    pending_boundary: false,
                    cached_action: None,
                    repeats_left: 0,
                    current_option: 0,
                    option_goal: vec![0.0; option_dim],
                    option_steps_left: 0, // triggers initial option sample
                    option_elapsed: 0,
                    option_return: 0.0,
                    option_start_value: 0.0,
                    option_start_z: vec![0.0; config.latent_dim],
                    last_value: 0.0,
                    last_prob_taken: 1.0,
                    pred_error_ema: 0.0,
                    last_logits: vec![0.0; MAX_ACTION_DIM],
                    last_entropy: 0.0,
                    last_surprise: 0.0,
                    confidence: 0.5,
                    last_novelty: 0.0,
                    last_homeo: 0.0,
                    last_order: 0.0,
                    last_reward: 0.0,
                    last_base_reward: 0.0,
                    outcome_ep_trajectory: Vec::new(),
                    outcome_ep_return: 0.0,
                    last_episode_return: 0.0,
                    sil_ep_return: 0.0,
                    sil_ep_event_count: 0,
                    outcome_last_step_reward: 0.0,
                    outcome_ep_step_rewards: Vec::new(),
                    prev_r_hat: 0.0,
                    outcome_baseline: 0.0,
                    last_r_hat: 0.0,
                    outcome_baseline_seeded: false,
                }
            })
            .collect();

        Self {
            gpu,
            lanes,
            task_embeddings,
            wm_session,
            efficientnet_session,
            efficientnet_input_size_bytes,
            efficientnet_output_buf,
            v2s_preprocess,
            policy_session,
            option_session,
            latent_dim: config.latent_dim,
            step_count: 0,
            probe_obs: None,
            probe_reference: None,
            last_wm_loss: 0.0,
            last_recon_loss: 0.0,
            last_policy_loss: 0.0,
            policy_update_ticks: 0,
            policy_loss_ema: 0.0,
            last_replay_loss: 0.0,
            last_drift: 0.0,
            encoder_lr_scale: 1.0,
            batch_lr_scale: (config.batch_size as f32).sqrt(),
            // When end_to_end_encoder is on with rollout_length>1, the
            // policy training path fills obs+task per rollout row
            // (`policy_batch` rows), not just per lane. Size accordingly.
            // act() and observe() still write only the first `n` rows.
            obs_token_scratch: vec![0.0; obs_scratch_rows * OBS_TOKEN_DIM],
            visual_obs_size_bytes: n
                * config.encoder_kind.visual_dim()
                * std::mem::size_of::<f32>(),
            action_token_scratch: vec![0.0; n * MAX_ACTION_DIM],
            wm_action_token_scratch: vec![0.0; n * WM_ACTION_DIM],
            action_parameter_scratch: vec![0.0; n * ACTION_PARAMETER_DIM],
            action_parameter_active: vec![false; n],
            action_parameter_masks: vec![false; n * MAX_ACTION_DIM],
            last_planned_action_parameters: vec![None; n],
            z_target_scratch: vec![0.0; n * config.latent_dim],
            task_scratch: vec![0.0; obs_scratch_rows * TASK_DIM],
            // Policy-session inputs are sized for the expanded rollout
            // batch (`lanes × rollout_length`). At rollout_length=1
            // this equals `n` — pre-refactor behavior. For rollout
            // mode, act() still fills only the first `n` rows (the
            // rest are ignored — act reads only first `n` output rows);
            // training fills all `policy_batch` rows from the rollout
            // window.
            value_target_scratch: vec![0.0; policy_batch],
            policy_action_scratch: vec![0.0; policy_batch * MAX_ACTION_DIM],
            policy_action_mask_scratch: vec![1.0; policy_batch * MAX_ACTION_DIM],
            policy_action_mask_input_present: is_discrete,
            policy_z_scratch: vec![0.0; policy_batch * config.latent_dim],
            ppo_advantage_scratch: vec![0.0; policy_batch],
            ppo_old_prob_scratch: vec![1.0; policy_batch],
            kl_old_logits_scratch: if config.use_kl_ppo {
                vec![0.0; policy_batch * MAX_ACTION_DIM]
            } else {
                Vec::new()
            },
            // Input is built only when e2e + initial entropy_beta > 0.
            // The KL-PPO graph variant doesn't include the entropy
            // regularization branch (no entropy_beta input), so gate
            // it out here too.
            entropy_beta_input_present: is_discrete
                && config.end_to_end_encoder
                && config.entropy_beta > 0.0
                && !config.use_kl_ppo,
            // Input is built only when use_kl_ppo + kl_beta > 0.
            old_logits_input_present: config.use_kl_ppo && config.kl_beta > 0.0,
            reward_pred_input_present: config.end_to_end_encoder
                && config.reward_pred_loss_coef > 0.0
                && !config.use_kl_ppo
                && !config.use_ppo,
            reward_target_scratch: if config.end_to_end_encoder
                && config.reward_pred_loss_coef > 0.0
                && !config.use_kl_ppo
                && !config.use_ppo
            {
                vec![0.0; policy_batch]
            } else {
                Vec::new()
            },
            last_kl: 0.0,
            kl_snapshot_capture_pending: false,
            option_dim,
            option_onehot_scratch: vec![0.0; policy_batch * config.num_options.max(1)],
            option_taken_scratch: vec![0.0; n * config.num_options],
            option_return_scratch: vec![0.0; n],
            termination_target_scratch: vec![0.0; n],
            goal_table: option::build_goal_table(config.num_options, option_dim),
            outcome_head,
            last_outcome_baseline: 0.0,
            approach_state,
            approach_ep_returns: vec![0.0; n],
            approach_distances: vec![0.0; n],
            rnd_state,
            last_rnd_mse: 0.0,
            diayn_state,
            last_diayn_reward: 0.0,
            coord_head,
            coord_last_reward: vec![0.0; n],
            coord_reward_baseline: vec![0.0; n],
            delta_goal_bank,
            delta_goal_prev_latent: (0..n).map(|_| None).collect(),
            last_delta_goal_events: 0,
            xeps_memory,
            xeps_prev_action: vec![None; n],
            planner,
            wm_planner_session,
            policy_planner_session,
            option_planner_session,
            wm_kstep_session,
            kstep_z_scratch,
            kstep_z_target_scratch,
            kstep_action_scratch_per_step,
            last_wm_kstep_loss: 0.0,
            wm_mcts_session,
            planner_samples_cached,
            planner_z_scratch,
            planner_action_scratch,
            planner_option_onehot_scratch,
            planner_traj_scratch,
            planner_sigma_scratch: Vec::new(),
            surprise_ring_frames: Vec::new(),
            surprise_ring_actions: Vec::new(),
            surprise_ring_zprev: Vec::new(),
            surprise_ring_obs: Vec::new(),
            surprise_ring_env: Vec::new(),
            surprise_ring_prio: Vec::new(),
            surprise_ring_next: 0,
            planner_sigma_traj_scratch: Vec::new(),
            planner_v_traj_scratch,
            planner_param_buf,
            planner_queue: (0..n).map(|_| std::collections::VecDeque::new()).collect(),
            planner_calls_since_refresh: 0,
            extrinsic_reward: vec![0.0; n],
            extrinsic_event: vec![None; n],
            intrinsic_progress_reward: vec![0.0; n],
            last_empowerment: vec![0.0; n],
            goal_states: hashbrown::HashMap::new(),
            subgoal_centroids: hashbrown::HashMap::new(),
            action_masks: initial_action_masks,
            sil_buffer: std::collections::VecDeque::with_capacity(config.sil_buffer_capacity),
            sil_baselines: hashbrown::HashMap::new(),
            sil_updates_attempted: 0,
            sil_updates_fired: 0,
            replan_clears: 0,
            surprise_replays: 0,
            sil_last_active_rows: 0,
            sil_last_param_change: 0.0,
            wm_planner_has_value_head,
            win_buffer: hashbrown::HashMap::new(),
            loss_buffer: hashbrown::HashMap::new(),
            value_target_scratch_vh,
            value_z_scratch,
            wm_gate_scalar,
            value_gate_scalar,
            last_value_head_loss: 0.0,
            value_head_updates: 0,
            config,
        }
    }

    /// Number of lanes (`N = config.batch_size`).
    pub fn num_lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Mark the next observed transition on `lane_idx` as the start of a
    /// new episode within the same env. The world model will zero its
    /// `z_target` row for that lane on the next step, and the stored
    /// transition will be tagged `env_boundary = true`, so the credit
    /// assigner and world model skip attribution across the reset.
    pub fn mark_boundary(&mut self, lane_idx: usize) {
        // Apply the terminal-proximity bonus retroactively if configured
        // and the just-ended episode terminated with a positive-reward
        // outcome. Gated on `pending_boundary` so calling mark_boundary
        // twice for the same terminal doesn't double-apply the bonus.
        // See `apply_terminal_proximity_bonus`.
        if !self.lanes[lane_idx].pending_boundary {
            apply_terminal_proximity_bonus(
                &mut self.lanes[lane_idx].buffer,
                self.config.terminal_proximity_k,
                self.config.terminal_proximity_bonus,
                self.config.terminal_proximity_threshold,
            );
        }
        let lane = &mut self.lanes[lane_idx];
        lane.pending_boundary = true;
        // An episode reset ends any in-flight action repeat: the post-reset
        // state is drawn from a fresh env distribution, so the cached action
        // is semantically stale.
        lane.cached_action = None;
        lane.repeats_left = 0;
        // Episode reset terminates the current option early — force a
        // resample next act().
        lane.option_steps_left = 0;
        // Drop any actions the planner queued for the episode that just
        // ended — they were planned from a now-dead latent and would
        // otherwise be blindly played into the fresh episode.
        self.planner_queue[lane_idx].clear();
        self.last_planned_action_parameters[lane_idx] = None;
    }

    /// Swap the active adapter on one lane. Preserves all learned
    /// parameters; the next transition stored on that lane is marked as
    /// an env boundary so the world model and return estimators do not
    /// cross the switch. Other lanes
    /// are unaffected.
    ///
    /// A new env's task embedding is lazily initialized on first sight;
    /// returning to a previously-seen env reuses the same deterministic
    /// vector, preserving the encoder's per-env specialization.
    pub fn switch_lane(&mut self, lane_idx: usize, adapter: Box<dyn EnvAdapter>) {
        let incoming_id = adapter.id();
        let orthogonal_task_codes = self.config.orthogonal_task_codes;
        self.task_embeddings
            .entry(incoming_id)
            .or_insert_with(|| embedding_for(incoming_id, TASK_DIM, orthogonal_task_codes));

        let lane = &mut self.lanes[lane_idx];
        lane.adapter = adapter;
        lane.pending_boundary = true;
        lane.cached_action = None;
        lane.repeats_left = 0;
        lane.option_steps_left = 0;
        // Queued planner actions were planned for the OUTGOING env —
        // playing them into the incoming one would be nonsense.
        self.planner_queue[lane_idx].clear();
        self.last_planned_action_parameters[lane_idx] = None;
        self.action_parameter_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM]
            .fill(false);
    }

    /// Env id of the adapter currently bound to `lane_idx`.
    pub fn env_id(&self, lane_idx: usize) -> u32 {
        self.lanes[lane_idx].adapter.id()
    }

    /// Select one action per lane. `observations.len()` must equal N.
    /// Returns a `Vec<Action>` of length N, in lane order.
    ///
    /// The obs argument is currently unused inside the policy graph (the
    /// policy conditions on the previous latent), but is kept in the
    /// signature to match the multi-lane contract and to make room for a
    /// future obs-conditioned exploration policy.
    pub fn act<R: Rng>(&mut self, observations: &[Observation], rng: &mut R) -> Vec<Action> {
        self.act_with_mode(observations, rng, false)
    }

    /// Select the highest-logit discrete action (or continuous mean) instead of
    /// sampling. Planner queues and action persistence still take precedence.
    /// This is intended for evaluation, not exploration.
    pub fn act_greedy<R: Rng>(&mut self, observations: &[Observation], rng: &mut R) -> Vec<Action> {
        self.act_with_mode(observations, rng, true)
    }

    fn act_with_mode<R: Rng>(
        &mut self,
        observations: &[Observation],
        rng: &mut R,
        greedy: bool,
    ) -> Vec<Action> {
        let n = self.lanes.len();
        assert_eq!(
            observations.len(),
            n,
            "act: observations.len() ({}) must equal num_lanes ({})",
            observations.len(),
            n
        );

        // E2E mode: policy_session takes raw obs + task as inputs, so
        // populate obs_token_scratch and task_scratch with the CURRENT-
        // step observations (the regular observe() path also fills
        // these but only after this act() returns; without this we'd
        // be feeding the policy session the previous step's obs).
        if self.config.end_to_end_encoder {
            for (i, obs) in observations.iter().enumerate() {
                let obs_row =
                    &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
                self.lanes[i].adapter.obs_to_token(obs, obs_row);
            }
            for (i, lane) in self.lanes.iter().enumerate() {
                let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
                if let Some(emb) = self.task_embeddings.get(&lane.adapter.id()) {
                    task_row.copy_from_slice(emb);
                } else {
                    task_row.fill(0.0);
                }
            }
        }

        // Stack per-lane previous latents. Lanes with a pending episode
        // boundary keep their row at zero: the buffered latent belongs
        // to the episode that just ended, and observe() stages
        // `z_prev = 0` for boundary rows (the WM's episode-start
        // manifold is anchored at the null state) — the first
        // post-reset action, value baseline, and option resample must
        // run from that same null state, not from the dead episode's
        // terminal latent.
        let ld = self.latent_dim;
        let od = self.option_dim;
        let mut z_stack = vec![0.0f32; n * ld];
        for (i, lane) in self.lanes.iter().enumerate() {
            if lane.pending_boundary {
                continue;
            }
            if let Some(prev) = lane.buffer.last() {
                z_stack[i * ld..(i + 1) * ld].copy_from_slice(&prev.latent);
            }
        }

        // --- L1 option management (Phase G) ---
        if let Some(ref mut opt_sess) = self.option_session {
            let num_options = self.config.num_options;
            let horizon = self.config.option_horizon;
            let learned_term = self.config.learned_termination;
            let warmup_done = self.step_count >= self.config.warmup_steps;

            // Forward option session every step to read current option
            // logits, state value, and termination β(z_t). LR = 0 so this
            // is forward-only; the training pass below optionally runs at
            // the real LR on just the terminated-lane rows.
            opt_sess.set_input("z", &z_stack);
            self.option_taken_scratch.fill(0.0);
            opt_sess.set_input("option_taken", &self.option_taken_scratch);
            self.option_return_scratch.fill(0.0);
            opt_sess.set_input("option_return", &self.option_return_scratch);
            self.termination_target_scratch.fill(0.0);
            opt_sess.set_input("termination_target", &self.termination_target_scratch);
            // Forward-only read of logits/value/β — skip the optimizer
            // pass (see act() for why lr = 0 under Adam is not a no-op).
            opt_sess.clear_optimizer();
            opt_sess.step();
            opt_sess.wait();

            let mut logits = vec![0.0f32; n * num_options];
            opt_sess.read_output_by_index(1, &mut logits);
            let mut values = vec![0.0f32; n];
            opt_sess.read_output_by_index(2, &mut values);
            let mut term_probs = vec![0.0f32; n];
            opt_sess.read_output_by_index(3, &mut term_probs);

            // Per-lane termination decision: horizon cap (always) or
            // Bernoulli(β) sample (only when learned_termination is on
            // and past warmup).
            let mut lanes_to_terminate: Vec<usize> = Vec::with_capacity(n);
            for (i, lane) in self.lanes.iter().enumerate() {
                let horizon_expired = lane.option_steps_left == 0;
                let learned_fire =
                    learned_term && warmup_done && rng.random_range(0.0..1.0) < term_probs[i];
                if horizon_expired || learned_fire {
                    lanes_to_terminate.push(i);
                }
            }

            if !lanes_to_terminate.is_empty() {
                // --- L1 backward: train at the lanes that just
                // terminated. Other lanes contribute zero rows and are
                // effectively excluded from this step's gradient.
                self.option_taken_scratch.fill(0.0);
                // Default every row's value target to the value head's
                // own current prediction: MSE(V(z), V(z)) contributes
                // exactly zero gradient, so non-trained lanes are truly
                // excluded instead of being pulled toward V = 0. (CE
                // rows with all-zero labels already contribute zero
                // gradient — meganeura's CE backward scales the softmax
                // term by the per-row label sum.)
                self.option_return_scratch.copy_from_slice(&values);
                self.termination_target_scratch.fill(0.0);
                let mut any_train = false;
                let mut trained_lanes: Vec<usize> = Vec::new();
                for &i in &lanes_to_terminate {
                    let lane = &self.lanes[i];
                    let advantage = (lane.option_return - lane.option_start_value).clamp(-1.0, 1.0);
                    if advantage.abs() < 1e-8 {
                        continue;
                    }
                    any_train = true;
                    trained_lanes.push(i);
                    let row =
                        &mut self.option_taken_scratch[i * num_options..(i + 1) * num_options];
                    row[lane.current_option as usize] = advantage;
                    self.option_return_scratch[i] = lane.option_return;
                    // Termination target at the step where the option
                    // ended, with a deadband: only train β when the
                    // option's realized advantage is clearly signed.
                    //
                    //   |adv| < 0.3 → target = 0 (keep β low; noisy
                    //                  signals shouldn't raise β).
                    //   adv < −0.3   → target = 1 (raise β here).
                    //   adv > +0.3   → target = 0 (correct to have
                    //                  continued).
                    //
                    // Paired with the −3 logit bias in `option.rs`,
                    // this keeps β strongly low by default; β only
                    // rises when states with consistently-negative
                    // option returns accumulate training signal.
                    let deadband = 0.3f32;
                    self.termination_target_scratch[i] =
                        if advantage < -deadband { 1.0 } else { 0.0 };
                }

                if any_train && warmup_done {
                    // The option was CHOSEN at its start state, so the
                    // option-policy CE and the value baseline must train
                    // at `option_start_z`, not at the state where the
                    // option happened to end — training at z_end credits
                    // the choice at a state where it was never made and
                    // turns the baseline into "return of the option that
                    // just ended here". The captured start latents
                    // replace the trained lanes' rows; other rows keep
                    // the current z (their targets are self/zero, so
                    // they contribute no gradient).
                    let mut train_z = z_stack.clone();
                    for &i in &trained_lanes {
                        let lane = &self.lanes[i];
                        if lane.option_start_z.len() == ld {
                            train_z[i * ld..(i + 1) * ld].copy_from_slice(&lane.option_start_z);
                        }
                    }
                    // Termination targets in THIS dispatch pair with the
                    // start-state rows, so keep them 0 here ("don't
                    // terminate an option right where it starts" — the
                    // same keep-β-low prior the design already applies
                    // to continuing lanes). The end-state BCE runs as a
                    // second dispatch below when learned termination is
                    // actually in use.
                    let term_targets = std::mem::take(&mut self.termination_target_scratch);
                    self.termination_target_scratch = vec![0.0; term_targets.len()];
                    opt_sess.set_input("z", &train_z);
                    opt_sess.set_input("option_taken", &self.option_taken_scratch);
                    opt_sess.set_input("option_return", &self.option_return_scratch);
                    opt_sess.set_input("termination_target", &self.termination_target_scratch);
                    let lr_option = self.config.lr_option * self.batch_lr_scale;
                    apply_lr(
                        opt_sess,
                        lr_option,
                        self.config.use_adam,
                        self.config.adam_eps,
                    );
                    opt_sess.step();
                    opt_sess.wait();

                    // Second dispatch: termination BCE at the states
                    // where the options actually ENDED (that's where the
                    // terminate/continue decision β(z) applies). CE rows
                    // are all-zero (zero gradient — meganeura's CE
                    // backward scales by the per-row label sum) and the
                    // value targets are the head's own predictions (zero
                    // MSE gradient), so this trains only the β head.
                    self.termination_target_scratch = term_targets;
                    let has_term_signal = self.termination_target_scratch.iter().any(|&t| t > 0.0);
                    if learned_term && has_term_signal {
                        self.option_taken_scratch.fill(0.0);
                        self.option_return_scratch.copy_from_slice(&values);
                        opt_sess.set_input("z", &z_stack);
                        opt_sess.set_input("option_taken", &self.option_taken_scratch);
                        opt_sess.set_input("option_return", &self.option_return_scratch);
                        opt_sess.set_input("termination_target", &self.termination_target_scratch);
                        opt_sess.step();
                        opt_sess.wait();
                    }
                }

                // Update each terminated option's continuous goal prototype.
                let ema_rate = self.config.goal_ema_rate;
                for &i in &lanes_to_terminate {
                    let lane = &mut self.lanes[i];
                    if lane.option_elapsed > 0 && ema_rate > 0.0 {
                        let old_opt = lane.current_option as usize;
                        let z_end = &z_stack[i * ld..(i + 1) * ld];
                        let base = old_opt * od;
                        let copy_dim = od.min(ld);
                        let goal = &mut self.goal_table[base..base + od];
                        for k in 0..copy_dim {
                            goal[k] += ema_rate * (z_end[k] - goal[k]);
                        }
                    }
                }

                // --- Sample new option for each terminated lane ---
                for &i in &lanes_to_terminate {
                    let lane = &mut self.lanes[i];
                    let row = &logits[i * num_options..(i + 1) * num_options];
                    let opt_idx = crate::adapter::sample_discrete_from_logits(row, rng);
                    lane.current_option = opt_idx as u32;
                    lane.option_start_value = values[i];
                    lane.option_steps_left = horizon;
                    lane.option_elapsed = 0;
                    lane.option_return = 0.0;
                    let z_row = &z_stack[i * ld..(i + 1) * ld];
                    lane.option_start_z.clear();
                    lane.option_start_z.extend_from_slice(z_row);
                    let base = opt_idx * od;
                    lane.option_goal
                        .copy_from_slice(&self.goal_table[base..base + od]);
                }
            }

            // Build per-lane one-hot option encodings and feed both z
            // (pure latent) and option_onehot to the policy graph.
            // The option identity signals directly into the per-option
            // bias head inside the policy graph.
            self.option_onehot_scratch.fill(0.0);
            for (i, lane) in self.lanes.iter().enumerate() {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                row[lane.current_option as usize] = 1.0;
            }
            // L1 + e2e: feed obs/task to the e2e graph directly (encoder
            // is built inside the policy graph). The obs_token_scratch
            // and task_scratch were already populated above when
            // end_to_end_encoder is on.
            // L1 + non-e2e: feed z_stack as before.
            if self.config.end_to_end_encoder {
                self.policy_session
                    .set_input("obs", &self.obs_token_scratch);
                self.policy_session.set_input("task", &self.task_scratch);
            } else {
                // Pad z_stack from [lanes × ld] to the policy graph's
                // expected [policy_batch × ld]. With rollout_length=1
                // these are the same size. With rollout_length>1, rows
                // beyond `lanes` are zeros — their forward outputs are
                // garbage that we throw away (act() only reads the first
                // `lanes` output rows), but the session still requires a
                // valid-sized input.
                self.policy_z_scratch[..n * ld].copy_from_slice(&z_stack);
                for v in self.policy_z_scratch[n * ld..].iter_mut() {
                    *v = 0.0;
                }
                self.policy_session.set_input("z", &self.policy_z_scratch);
            }
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        } else {
            // L0-only path — feed z directly (padded).
            if self.config.end_to_end_encoder {
                // E2E mode: policy_session takes raw obs + task instead
                // of z. obs_token_scratch and task_scratch are already
                // populated above with current-step values.
                self.policy_session
                    .set_input("obs", &self.obs_token_scratch);
                self.policy_session.set_input("task", &self.task_scratch);
            } else {
                self.policy_z_scratch[..n * ld].copy_from_slice(&z_stack);
                for v in self.policy_z_scratch[n * ld..].iter_mut() {
                    *v = 0.0;
                }
                self.policy_session.set_input("z", &self.policy_z_scratch);
            }
        }

        // The policy graph's `action`/`action_mask` inputs are sized
        // for the rollout batch (`lanes × rollout_length` rows), while
        // `action_token_scratch` is only the collection batch's `lanes`-row
        // discrete-label buffer — feeding it here panicked on the byte-size
        // check for any `rollout_length > 1`. Use the policy-batch scratches.
        self.policy_action_scratch.fill(0.0);
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        // Feed the current per-lane action mask so the policy graph's
        // masked-logits subgraph (PPO) sees the right availability.
        // The forward at act-time discards the loss output, so any
        // mismatch here is harmless beyond the per-lane sampling — but
        // keeping it consistent avoids surprises when the same graph
        // is reused for training in `policy_step_batched`.
        if self.policy_action_mask_input_present {
            self.populate_policy_action_mask_scratch();
            self.policy_session
                .set_input("action_mask", &self.policy_action_mask_scratch);
        }
        // KL-PPO graph requires `old_logits` input to compute the loss.
        // act() doesn't read the loss output, but the input must exist
        // for the forward pass to run. Zero out — KL contribution is
        // discarded anyway. Only fires when the input was actually
        // built (kl_beta > 0 at construction).
        if self.old_logits_input_present {
            for v in self.kl_old_logits_scratch.iter_mut() {
                *v = 0.0;
            }
            self.policy_session
                .set_input("old_logits", &self.kl_old_logits_scratch);
        }
        // PPO and continuous graphs require an `advantage` input.
        // act() doesn't read the loss output but the forward pass still
        // computes their policy-loss subgraphs. Zero advantage makes those
        // loss terms inert. PPO additionally needs a unit old probability to
        // keep its ratio finite.
        if self.config.use_ppo || !self.policy_action_mask_input_present {
            self.ppo_advantage_scratch.fill(0.0);
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
        }
        if self.config.use_ppo {
            for v in self.ppo_old_prob_scratch.iter_mut() {
                *v = 1.0;
            }
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        self.value_target_scratch.fill(0.0);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        self.feed_entropy_beta_input();
        self.feed_kl_beta_input();
        // Forward-only: skip the optimizer pass entirely. `set_adam`
        // with lr = 0 is NOT a no-op — the Adam kernel still runs each
        // step(), advancing `adam_step` and folding this pass's fake-
        // loss gradients (zero action target, value_target = 0) into
        // the m/v moment buffers, which corrupts every real training
        // update. `clear_optimizer` is persistent, and every training
        // path re-arms via `apply_lr` before its own step().
        self.policy_session.clear_optimizer();
        self.policy_session.step();
        self.policy_session.wait();

        // Read stacked outputs: head is [N, MAX_ACTION_DIM], value is [N, 1].
        let mut head_stack = vec![0.0f32; n * MAX_ACTION_DIM];
        self.policy_session.read_output_by_index(1, &mut head_stack);
        let mut value_stack = vec![0.0f32; n];
        self.policy_session
            .read_output_by_index(2, &mut value_stack);
        // Stability guard: sanitize any NaN and clamp extreme logits so
        // softmax(head) stays finite. Combined with the graph-internal
        // soft-tanh-clamp (see `policy.rs::scaled_tanh`) this forms a
        // defense-in-depth against long-run numerical drift. ±60 is
        // just beyond the graph clamp (±50) so a well-behaved graph
        // output is never touched here; only truly bad values are
        // sanitized.
        for v in head_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-60.0, 60.0);
            }
        }
        for v in value_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-1e6, 1e6);
            }
        }

        // Per-lane sampling with optional action persistence. The batched
        // policy forward ran for every lane above (so `head`/`value` rows
        // are always fresh); a lane in mid-repeat just ignores its row
        // this step and re-uses the cached action.
        let action_repeat = self.config.action_repeat.max(1);
        let mut actions = Vec::with_capacity(n);
        // Reusable per-lane masked-logits buffer to avoid per-step
        // allocation in the hot loop.
        let mut head_masked = vec![0.0f32; MAX_ACTION_DIM];
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let head_raw = &head_stack[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let mask_row = &self.action_masks[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];

            // Apply action mask: invalid logits → large negative so
            // softmax assigns them ~0 probability. If ALL entries are
            // masked off, fall back to the unmasked logits to avoid a
            // NaN softmax (no valid action to sample).
            let any_valid = mask_row.iter().any(|&m| m >= 0.5);
            if any_valid {
                for j in 0..MAX_ACTION_DIM {
                    head_masked[j] = if mask_row[j] >= 0.5 {
                        head_raw[j]
                    } else {
                        -1e9
                    };
                }
            } else {
                head_masked.copy_from_slice(head_raw);
            }
            let head: &[f32] = &head_masked;

            lane.last_value = value_stack[i];
            lane.last_entropy = lane.adapter.head_entropy(head);

            // Model-based planner: if the queue has a pre-planned
            // action for this lane, play it instead of sampling from
            // the policy head. Skips action_repeat accounting — a
            // planned sequence is semantically one committed plan,
            // not independent samples to repeat.
            let mut queued = self.planner_queue[i].pop_front();
            if any_valid
                && queued.is_some_and(|planned| {
                    let idx = planned.action as usize;
                    idx >= MAX_ACTION_DIM || mask_row[idx] < 0.5
                })
            {
                // Availability can change after every environment step. A
                // stale queued action would otherwise be executed despite the
                // current mask, corrupting PPO's acted-action probability and
                // possibly applying coordinates to a substituted action.
                queued = None;
                self.planner_queue[i].clear();
            }
            self.last_planned_action_parameters[i] = queued.and_then(|p| p.parameters);
            let action = if let Some(planned) = queued {
                let act = Action::Discrete(planned.action as usize);
                lane.cached_action = Some(act.clone());
                lane.repeats_left = 0;
                act
            } else {
                let resample = lane.repeats_left == 0 || lane.cached_action.is_none();
                if resample {
                    let a = if greedy {
                        greedy_policy_action(lane.adapter.action_kind(), head)
                    } else {
                        lane.adapter.sample_action(head, rng)
                    };
                    lane.cached_action = Some(a.clone());
                    lane.repeats_left = action_repeat - 1;
                    a
                } else {
                    lane.repeats_left -= 1;
                    lane.cached_action
                        .clone()
                        .expect("cached_action is Some by branch condition")
                }
            };
            // Cache π_old(a | s) for the PPO path — head is logits for
            // discrete adapters; re-softmax it here rather than threading
            // an extra array out of the batched forward. Continuous
            // actions use a Gaussian with fixed scale, we approximate
            // their π_old as 1.0 (the MSE surrogate is scale-invariant).
            //
            // Also cache the full logits for the KL-PPO path (when
            // use_kl_ppo is on, the rollout-batch training feeds these
            // as `old_logits` graph input to compute KL(π_new ‖ π_old)
            // exactly). We store the MASKED logits so the recorded
            // π_old distribution matches what we actually sampled from.
            if self.config.use_kl_ppo {
                let nlog = head.len().min(lane.last_logits.len());
                lane.last_logits[..nlog].copy_from_slice(&head[..nlog]);
                for v in &mut lane.last_logits[nlog..] {
                    *v = 0.0;
                }
            }
            lane.last_prob_taken = match &action {
                Action::Discrete(a) => {
                    let probs = crate::policy::softmax_probs(head);
                    let idx = (*a).min(probs.len().saturating_sub(1));
                    probs[idx].max(1e-8)
                }
                Action::Continuous(_) => 1.0,
            };
            // Cache the discrete action id for cross-episode memory.
            // Continuous actions don't key the xeps memory (no natural
            // discrete index), so they leave `xeps_prev_action` as None.
            self.xeps_prev_action[i] = match action {
                Action::Discrete(a) => Some(a as u32),
                Action::Continuous(_) => None,
            };
            actions.push(action);
        }
        actions
    }

    /// Observe one synchronous step across all lanes. All input slices must
    /// have length `N = config.batch_size`.
    ///
    /// # Calling convention (post-action)
    ///
    /// `observations[i]` must be the observation that RESULTED from
    /// `actions[i]` — i.e. call order is `act() → env.step(a) →
    /// observe(o_next, a)`. The WM trains forward dynamics
    /// `WM(z_prev, a) → enc(o_next)` and the policy pairs `a` with the
    /// previous step's latent, both of which assume this ordering.
    /// (In CNN modes the same applies to the frames staged via
    /// `visual_obs`: write the POST-step frames before calling
    /// observe.)
    pub fn observe<R: Rng>(
        &mut self,
        observations: &[Observation],
        actions: &[Action],
        envs: &[&dyn Environment],
        rng: &mut R,
    ) {
        let homeostatic = envs
            .iter()
            .map(|env| env.homeostatic_variables())
            .collect::<Vec<_>>();
        self.observe_with_homeostatic(observations, actions, &homeostatic, rng);
    }

    fn observe_with_homeostatic<R: Rng>(
        &mut self,
        observations: &[Observation],
        actions: &[Action],
        homeostatic: &[&[crate::env::HomeostaticVariable]],
        rng: &mut R,
    ) {
        let n = self.lanes.len();
        assert_eq!(observations.len(), n, "observations.len() must equal N");
        assert_eq!(actions.len(), n, "actions.len() must equal N");
        assert_eq!(homeostatic.len(), n, "homeostatic.len() must equal N");

        let ld = self.latent_dim;

        // --- Build stacked inputs: obs, action, z_target, task ---
        for (i, lane) in self.lanes.iter().enumerate() {
            let obs_row = &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            lane.adapter.obs_to_token(&observations[i], obs_row);

            let act_row =
                &mut self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            lane.adapter.action_to_token(&actions[i], act_row);

            // "z_target" row (historical name): the WM's INPUT — the
            // previous step's latent, i.e. the state the action was
            // taken from. Zeros at boundary or bootstrap (the WM
            // learns the episode-start manifold from the null state).
            let z_row = &mut self.z_target_scratch[i * ld..(i + 1) * ld];
            if lane.pending_boundary {
                z_row.fill(0.0);
            } else if let Some(prev) = lane.buffer.last() {
                z_row.copy_from_slice(&prev.latent);
            } else {
                z_row.fill(0.0);
            }

            // task row: lookup per-lane env's embedding.
            let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
            match self.task_embeddings.get(&lane.adapter.id()) {
                Some(emb) => task_row.copy_from_slice(emb),
                None => task_row.fill(0.0),
            }
        }
        self.stage_wm_action_tokens();

        // --- One batched WM forward+backward ---
        let optimization_loss = self.wm_forward_backward_stacked(
            self.config.learning_rate * self.encoder_lr_scale * self.batch_lr_scale,
        );
        if optimization_loss.is_finite() {
            let mut dynamics_loss = [0.0f32];
            let mut recon_loss = [0.0f32];
            self.wm_session.read_output_by_index(4, &mut dynamics_loss);
            self.wm_session.read_output_by_index(5, &mut recon_loss);
            self.last_wm_loss = dynamics_loss[0];
            self.last_recon_loss = recon_loss[0];
        } else {
            log::warn!(
                "WM loss went non-finite at step {}, re-initialized WM params",
                self.step_count
            );
            init_parameters(&mut self.wm_session);
            self.last_wm_loss = 0.0;
            self.last_recon_loss = 0.0;
        }

        // Read stacked z_t output [N, latent_dim] and per-lane
        // squared-error output [N, latent_dim] from the WM graph.
        let mut z_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(1, &mut z_stack);
        let mut sq_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(2, &mut sq_stack);
        // Stability guard: clamp latent components to a finite range
        // before anything downstream (surprise computation, policy
        // input, reward bonus, buffer writes). The encoder's final
        // `fc2` is unbounded — under long training on simple envs it
        // can drive `|z|` arbitrarily large, and the resulting policy
        // logits blow up to NaN within a few tens of thousands of
        // steps. Clamping to ±10 preserves all normal-range
        // representations (Xavier init produces values in ~[-2, 2])
        // while bounding logit magnitudes through the policy's
        // `Linear(latent_dim, hidden)` — exp(10·‖w‖) stays well inside
        // f32 before any softmax.
        for v in z_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-10.0, 10.0);
            }
        }

        // --- Per-lane reward + transition push ---
        // Per-lane surprise: `pred_error_i = ||z_hat_i − z_target_i||` is
        // the L2 norm of the i-th row of `(z_hat − z_target)`, i.e. the
        // sqrt of the sum of that row's squared-errors. Replaces the old
        // Phase E.v1 mean-loss surrogate, which blurred rare-event signal
        // across all N lanes at large batch sizes.
        //
        // M6: capture the outcome-head knobs + handle in one split so the
        // inner loop can both train (episode boundary) and forward (every
        // step) without fighting the borrow checker against `self.lanes`.
        let m6_alpha = self.config.outcome_reward_alpha;
        let m6_ema = self.config.outcome_baseline_ema;
        let m6_clamp = self.config.outcome_clamp.max(0.0);
        let m6_max_ep = self.config.outcome_max_episode_len;
        let m6_target = self.config.outcome_target;
        let m6_bonus_mode = self.config.outcome_bonus;
        let m7_alpha = self.config.approach_reward_alpha;
        let m7_rank_by = self.config.approach_rank_by;
        let m7_clamp = self.config.approach_distance_clamp;
        let m7_saturation = self.config.approach_confidence_saturation;
        let rnd_alpha = self.config.rnd_reward_alpha;
        let mut rnd_state = self.rnd_state.take();
        let mut rnd_mse_sum = 0.0f32;
        let mut rnd_mse_count = 0usize;
        let diayn_alpha = self.config.diayn_reward_alpha;
        let mut diayn_state = self.diayn_state.take();
        let mut diayn_reward_sum = 0.0f32;
        let mut diayn_reward_count = 0usize;
        let dg_alpha = self.config.delta_goal_alpha;
        let dg_clamp = self.config.delta_goal_distance_clamp;
        let dg_surprise_gate = self.config.delta_goal_surprise_threshold;
        let mut dg_bank = self.delta_goal_bank.take();
        let mut dg_events_this_step: usize = 0;
        let xeps_alpha = self.config.xeps_reward_alpha;
        let mut xeps_memory = self.xeps_memory.take();
        let ext_alpha = self.config.extrinsic_reward_alpha;
        let m7_warmup = self.config.approach_warmup_episodes;
        let m7_homeo_taper = self.config.homeo_confidence_taper.clamp(0.0, 1.0);
        let mut m7_state = self.approach_state.take();
        // Confidence for this step. Zero before warmup; linear ramp
        // from 0 → 1 over `saturation` episodes once warmup is met;
        // clamped to [0, 1]. `saturation = 0` disables ramping
        // (c = 1 once warmup is satisfied, matching M7 v1 byte-parity
        // for an M7-enabled config).
        let m7_confidence = match m7_state.as_ref() {
            None => 0.0,
            Some(state) => {
                if state.episodes_seen < m7_warmup {
                    0.0
                } else if m7_saturation == 0 {
                    1.0
                } else {
                    let past_warmup = (state.episodes_seen - m7_warmup) as f32;
                    (past_warmup / m7_saturation as f32).clamp(0.0, 1.0)
                }
            }
        };
        let mut m6_head = self.outcome_head.take();
        let mut m6_baseline_diag = self.last_outcome_baseline;
        // (P4) Lanes whose realized surprise this step invalidates
        // their queued plan; queues cleared after the loop (the lane
        // iterator holds a mutable borrow over self.lanes only).
        let replan_mult = self.config.replan_surprise_mult;
        let mut replan_lanes: Vec<usize> = Vec::new();
        // (P2) Surprise-ring admission candidates: (lane, priority).
        // Frame copies happen after the loop (the visual slot content
        // is stable until the next harness write).
        let ring_cap = if matches!(
            self.config.encoder_kind,
            EncoderKind::Cnn { .. } | EncoderKind::CnnDqn { .. }
        ) {
            self.config.surprise_ring_capacity
        } else {
            // Mlp has no visual slot (replay_step covers it);
            // EfficientNetV2S recomputes the slot from the Dullahan
            // frame inside every dispatch, which would clobber staged
            // replay frames.
            0
        };
        let ring_per_sample = (self.visual_obs_size_bytes / std::mem::size_of::<f32>())
            .checked_div(n)
            .unwrap_or(0);
        let ring_mult = self.config.surprise_replay_mult.max(1.0);
        let mut ring_admits: Vec<(usize, f32)> = Vec::new();
        // Score every synchronous lane against the same RND predictor
        // snapshot, then update it as one batch. Calling `step` inside the
        // lane loop made curiosity depend on iteration order.
        let rnd_mses = if let Some(state) = rnd_state.as_mut() {
            let rows: Vec<&[f32]> = self
                .obs_token_scratch
                .chunks_exact(OBS_TOKEN_DIM)
                .take(n)
                .collect();
            state.step_batch(&rows)
        } else {
            vec![0.0; n]
        };
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let z_row = &z_stack[i * ld..(i + 1) * ld];
            let obs_row = &self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            let act_row = &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let sq_row = &sq_stack[i * ld..(i + 1) * ld];

            let row_sum: f32 = sq_row
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .sum();
            let pred_error = row_sum.max(0.0).sqrt();
            let surprise = RewardCircuit::surprise(pred_error);

            // (P4) Surprise-triggered replanning: track a per-lane EMA
            // of the WM error and mark the lane for a planner-queue
            // clear when this step's error spikes far above it — the
            // world just did something the plan's WM rollout did not
            // anticipate, so the remaining queued actions are built on
            // a stale premise. Lazy-seed the EMA with the first value.
            if lane.pred_error_ema == 0.0 {
                lane.pred_error_ema = pred_error;
            } else {
                lane.pred_error_ema = 0.99 * lane.pred_error_ema + 0.01 * pred_error;
            }
            if replan_mult > 0.0
                && lane.pred_error_ema > 1e-6
                && pred_error > replan_mult * lane.pred_error_ema
            {
                replan_lanes.push(i);
            }
            // (P2) Surprising transition → candidate for the frame
            // replay ring. Boundary rows are skipped: their z_prev is
            // the null state, so their "surprise" is an artifact.
            if ring_cap > 0
                && ring_per_sample > 0
                && !lane.pending_boundary
                && lane.pred_error_ema > 1e-6
                && pred_error > ring_mult * lane.pred_error_ema
            {
                ring_admits.push((i, pred_error));
            }

            let visit_count = lane.buffer.visit_count(z_row);
            let novelty = RewardCircuit::novelty(visit_count);
            let homeo_raw = RewardCircuit::homeostatic(homeostatic[i]);
            // M7↔homeo confidence taper: once M7 has confidence,
            // reduce the homeo contribution so M7's approach signal
            // isn't dominated by homeo's (potentially misaligned)
            // basin. `τ = 0` (default) preserves pre-confidence M7
            // behaviour.
            let homeo = homeo_raw * (1.0 - m7_homeo_taper * m7_confidence);
            let order = lane.reward_circuit.observe_order(obs_row);
            let base_reward = lane.reward_circuit.compute(surprise, novelty, homeo, order);

            // Goal-alignment bonus (Phase G): when L1 is active, add
            // `α · cos(z_t, goal)` — scale-invariant, bounded in
            // `[-α, +α]`. Kept in `last_reward` (advantage input) but
            // NOT in `last_base_reward` (value TD target), so the
            // value baseline can't absorb the option signal.
            let mut bonus = 0.0f32;
            if self.option_session.is_some() && self.config.goal_bonus_alpha > 0.0 {
                // The goal lives in the first `option_dim` dims of z
                // (option_dim ≤ latent_dim); compare over that prefix.
                // Passing full z with option_dim < latent_dim computed
                // the dot over the truncated prefix but the norm over
                // the full vector — a systematically shrunken cosine.
                let gd = lane.option_goal.len().min(z_row.len());
                let cos = unit_cosine(&z_row[..gd], &lane.option_goal[..gd]);
                bonus = self.config.goal_bonus_alpha * cos;
            }
            // `reward_pre_m6` is what trains the outcome head; what the
            // rest of the agent sees also folds in the M6 bonus below.
            let reward_pre_m6 = base_reward + bonus;

            let env_boundary = lane.pending_boundary;
            lane.pending_boundary = false;

            // M7 episode-boundary update: when this step is the first
            // of a new episode, the previous episode's terminal
            // `z_end` is still at `lane.buffer.last()` (current step
            // hasn't been pushed yet) and `approach_ep_returns[i]`
            // holds the previous episode's intrinsic-reward total.
            // Push that pair so the prototype-updater can re-compute.
            if env_boundary {
                if let Some(state) = m7_state.as_mut() {
                    if let Some(prev) = lane.buffer.last() {
                        let rank_score = match m7_rank_by {
                            ApproachRankBy::Return => self.approach_ep_returns[i],
                            ApproachRankBy::Novelty => {
                                // Terminal-specific rarity — reads
                                // from the M7 state's own
                                // `terminal_visit_counts`, NOT the
                                // full buffer's visit_count (which
                                // would conflate "terminated here"
                                // with "passed through here"). Note:
                                // this lookup happens BEFORE the
                                // subsequent `push_terminal` call
                                // increments the count, so the rank
                                // reflects historical rarity, not
                                // including self.
                                state.terminal_novelty(&prev.latent)
                            }
                        };
                        state.push_terminal(&prev.latent, rank_score);
                    }
                }
                self.approach_ep_returns[i] = 0.0;
            }

            // M6 episode-boundary training: when this step is the first
            // after a reset, the PREVIOUS episode's trajectory and
            // summary stats are complete and ready to train the outcome
            // head. We do this BEFORE the per-step M6 forward so the
            // bonus uses the just-updated head (marginally better
            // signal).
            if env_boundary {
                if let Some(head) = m6_head.as_mut() {
                    if !lane.outcome_ep_trajectory.is_empty() {
                        // Build one windowed input per step.
                        let mut windows: Vec<Vec<f32>> =
                            Vec::with_capacity(lane.outcome_ep_trajectory.len());
                        for i in 0..lane.outcome_ep_trajectory.len() {
                            if let Some(w) = head.build_window(&lane.outcome_ep_trajectory, i) {
                                windows.push(w);
                            }
                        }
                        match m6_target {
                            OutcomeTarget::EpisodeSum | OutcomeTarget::TerminalReward => {
                                let target_raw = match m6_target {
                                    OutcomeTarget::EpisodeSum => lane.outcome_ep_return,
                                    OutcomeTarget::TerminalReward => lane.outcome_last_step_reward,
                                    OutcomeTarget::RewardToGo => unreachable!(),
                                };
                                if !lane.outcome_baseline_seeded {
                                    lane.outcome_baseline = target_raw;
                                    lane.outcome_baseline_seeded = true;
                                }
                                let centered = target_raw - lane.outcome_baseline;
                                head.train_batch(&windows, centered);
                                lane.outcome_baseline =
                                    (1.0 - m6_ema) * lane.outcome_baseline + m6_ema * target_raw;
                                m6_baseline_diag = lane.outcome_baseline;
                            }
                            OutcomeTarget::RewardToGo => {
                                // Back-accumulate per-step RTG targets.
                                let l = lane.outcome_ep_step_rewards.len();
                                let mut targets = vec![0.0f32; l];
                                let mut running = 0.0f32;
                                for i in (0..l).rev() {
                                    running += lane.outcome_ep_step_rewards[i];
                                    targets[i] = running;
                                }
                                // Baseline = EMA of RTG[0] = full
                                // episode return, so early-step targets
                                // are centered like EpisodeSum would
                                // be. Late-step targets retain their
                                // raw magnitude (smaller by
                                // construction) because they reference
                                // the same baseline, giving per-step
                                // differentiation.
                                let ep_ret = targets.first().copied().unwrap_or(0.0);
                                if !lane.outcome_baseline_seeded {
                                    lane.outcome_baseline = ep_ret;
                                    lane.outcome_baseline_seeded = true;
                                }
                                for t in targets.iter_mut() {
                                    *t -= lane.outcome_baseline;
                                }
                                // Align lengths in case of a truncated
                                // trajectory cap: train on the shorter
                                // of the two.
                                let n_train = windows.len().min(targets.len());
                                head.train_batch_variable(&windows[..n_train], &targets[..n_train]);
                                lane.outcome_baseline =
                                    (1.0 - m6_ema) * lane.outcome_baseline + m6_ema * ep_ret;
                                m6_baseline_diag = lane.outcome_baseline;
                            }
                        }
                    }
                }
                lane.outcome_ep_trajectory.clear();
                // Snapshot the just-completed episode return for GRPO
                // per-episode advantage mode before resetting.
                lane.last_episode_return = lane.sil_ep_return;
                // SIL: push successful episode's transitions into the
                // replay buffer. "Successful" = return strictly above
                // the EMA baseline of recent episode returns.
                // Uses sil_ep_return (full reward incl. extrinsic), NOT
                // outcome_ep_return (only intrinsic) — see Lane field doc.
                if self.config.use_sil
                    && !(self.config.sil_event_filter && self.config.sil_event_horizon > 0)
                {
                    let ep_ret = lane.sil_ep_return;
                    let event_success = lane.sil_ep_event_count > 0;
                    let push = admit_sil_episode_for_env(
                        &mut self.sil_baselines,
                        lane.adapter.id(),
                        ep_ret,
                        event_success,
                        self.config.sil_event_filter,
                        self.config.sil_baseline_decay,
                    );
                    if push {
                        let samples = sil_samples_from_recent_trajectory(
                            &lane.buffer,
                            usize::MAX,
                            event_success,
                            self.config.value_target_clamp,
                        );
                        append_sil_samples(
                            &mut self.sil_buffer,
                            samples,
                            self.config.sil_buffer_capacity,
                        );
                    }
                }
                // Retroactively annotate the just-completed episode's
                // transitions in lane.buffer with their episode_return
                // and episode_complete=true. This walks back from the
                // most recent transition (which is the terminal step)
                // until we hit the previous env_boundary or the start
                // of the buffer.
                if self.config.use_grpo_episode {
                    let blen = lane.buffer.len();
                    if blen > 0 {
                        let ep_ret = lane.last_episode_return;
                        let mut idx = blen - 1;
                        loop {
                            // Annotate first; the start-of-episode
                            // transition itself (env_boundary=true)
                            // is part of the episode we're closing.
                            let tr = lane.buffer.get_mut(idx);
                            tr.episode_return = ep_ret;
                            tr.episode_complete = true;
                            // env_boundary marks first step of a new
                            // episode — when we hit it we've fully
                            // covered this episode's range. (For the
                            // very first episode, no transition has
                            // env_boundary=true; we fall out at idx=0.)
                            if tr.env_boundary {
                                break;
                            }
                            if idx == 0 {
                                break;
                            }
                            idx -= 1;
                        }
                    }
                }
                // Win-classifier training: on every completed episode,
                // walk the lane buffer back from the terminal step
                // to the most recent boundary. If the episode had any
                // extrinsic event (`sil_ep_event_count > 0`), push each
                // transition's observation into `win_buffer` with label
                // = γ^(T-t) — terminal=1.0, decaying backwards. If the
                // episode had no events, push to `loss_buffer` with
                // label = 0. Balanced sampling at replay time
                // (`value_replay_step`) feeds an equal mix of both into
                // the BCE training, so the encoder learns a separating
                // axis instead of collapsing to "predict 0 always."
                if self.config.value_head_train_coef > 0.0 {
                    let blen = lane.buffer.len();
                    // Per-env cap: half of total budget. Each env gets
                    // its own queue under that cap. In cross-game mode
                    // there is only one key, so use the full budget as
                    // the cap.
                    let per_env_cap = if self.config.value_buffer_cross_game {
                        self.config.value_head_buffer_capacity.max(1)
                    } else {
                        (self.config.value_head_buffer_capacity / 2).max(1)
                    };
                    let gamma = self.config.value_head_gamma.clamp(0.0, 0.999);
                    let is_win = lane.sil_ep_event_count > 0;
                    let env_id = lane.adapter.id();
                    // Cross-game mode: pool all samples under env_id 0
                    // so the sampler draws uniformly without per-game
                    // stratification.
                    let buf_key = if self.config.value_buffer_cross_game {
                        0u32
                    } else {
                        env_id
                    };
                    if blen > 0 {
                        let mut decay = 1.0f32;
                        let mut idx = blen - 1;
                        loop {
                            let tr = lane.buffer.get(idx);
                            let label = if is_win { decay } else { 0.0 };
                            let buf = if is_win {
                                self.win_buffer
                                    .entry(buf_key)
                                    .or_insert_with(std::collections::VecDeque::new)
                            } else {
                                self.loss_buffer
                                    .entry(buf_key)
                                    .or_insert_with(std::collections::VecDeque::new)
                            };
                            if buf.len() >= per_env_cap {
                                buf.pop_front();
                            }
                            buf.push_back((tr.latent.clone(), env_id, label));
                            decay *= gamma;
                            if tr.env_boundary {
                                break;
                            }
                            if idx == 0 {
                                break;
                            }
                            idx -= 1;
                        }
                    }
                }
                // HER-style relabeling: when a failed episode (zero
                // extrinsic events) ends, push its TERMINAL latent into
                // the env's `goal_states` queue with a per-config
                // probability. The planner's cos-sim scorer can then
                // bias rollouts toward "states the agent has actually
                // reached," giving structure to the latent space even
                // before any real win. On envs that DO produce wins,
                // real win-states dominate the FIFO queue over time;
                // HER synthetic goals get evicted.
                let her_prob = self.config.goal_states_her_prob;
                if her_prob > 0.0 && lane.sil_ep_event_count == 0 && self.config.goal_states_cap > 0
                {
                    if rng.random_range(0.0..1.0_f32) < her_prob {
                        if let Some(prev) = lane.buffer.last() {
                            let env_id = lane.adapter.id();
                            let key = if self.config.goal_states_cross_game {
                                0u32
                            } else {
                                env_id
                            };
                            let cap = self.config.goal_states_cap;
                            let q = self
                                .goal_states
                                .entry(key)
                                .or_insert_with(std::collections::VecDeque::new);
                            if q.len() >= cap {
                                q.pop_front();
                            }
                            q.push_back(prev.latent.clone());
                            // Centroid update for HER pushes too.
                            if self.config.subgoal_k > 0 {
                                let centroids =
                                    self.subgoal_centroids.entry(key).or_insert_with(Vec::new);
                                let z = prev.latent.clone();
                                online_kmeans_update(
                                    centroids,
                                    &z,
                                    self.config.subgoal_k,
                                    self.config.subgoal_lr,
                                );
                            }
                        }
                    }
                }
                lane.outcome_ep_return = 0.0;
                lane.sil_ep_return = 0.0;
                lane.sil_ep_event_count = 0;
                lane.outcome_ep_step_rewards.clear();
                lane.prev_r_hat = 0.0;
            }

            // Push z into the current episode's trajectory (cap at
            // `outcome_max_episode_len` — overflow truncates the tail).
            // Done *before* the per-step forward so the window ending
            // at `z_t` is available, including for the first step of a
            // new episode.
            if m6_head.is_some() && lane.outcome_ep_trajectory.len() < m6_max_ep {
                lane.outcome_ep_trajectory.push(z_row.to_vec());
            }

            // M6 per-step forward: read `R̂(window_ending_at_z_t)` for
            // the reward bonus.
            let r_hat = if let Some(head) = m6_head.as_ref() {
                let end = lane.outcome_ep_trajectory.len().saturating_sub(1);
                if let Some(win) = head.build_window(&lane.outcome_ep_trajectory, end) {
                    let raw = head.forward(&win);
                    if raw.is_finite() {
                        raw.clamp(-m6_clamp, m6_clamp)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let m6_bonus = match m6_bonus_mode {
                OutcomeBonus::Raw => m6_alpha * r_hat,
                OutcomeBonus::PotentialDelta => {
                    // α · (R̂_t − R̂_{t-1}). At episode boundary
                    // `prev_r_hat` is reset to 0, so the bonus on the
                    // first step of a new episode is `α · r_hat`
                    // (no previous reference). Ng-et-al shaping is
                    // well-defined here: treating the pre-episode
                    // potential as 0 is equivalent to re-setting the
                    // shaping at each episode boundary.
                    m6_alpha * (r_hat - lane.prev_r_hat)
                }
            };
            lane.prev_r_hat = r_hat;
            lane.last_r_hat = r_hat;

            // M7 approach bonus: `-α · ‖z_t − centroid‖` (clamped).
            // Zero until the prototype has been seeded (warmup
            // episodes across all lanes satisfied, see
            // `approach::ApproachState::reward`).
            // Scale the M7 bonus by confidence — starts at 0 during
            // warmup, ramps to 1 over `approach_confidence_saturation`
            // episodes. When saturation is 0, confidence is 1 once
            // warmup is satisfied (v1 behaviour).
            let m7_reward = if let Some(state) = m7_state.as_ref() {
                m7_confidence * state.reward(z_row, m7_alpha, m7_clamp)
            } else {
                0.0
            };
            // Cache raw distance for diagnostics.
            self.approach_distances[i] = if let Some(state) = m7_state.as_ref() {
                state.distance(z_row)
            } else {
                0.0
            };

            // Accumulate pre-M6 reward into the episode return — the
            // outcome head trains on this (or on the single-step value
            // when `outcome_target == TerminalReward`, read from
            // `outcome_last_step_reward` below) so it can't chase its
            // own output (stability guarantee from the design doc).
            lane.outcome_ep_return += reward_pre_m6;
            lane.outcome_last_step_reward = reward_pre_m6;
            if m6_head.is_some() && lane.outcome_ep_step_rewards.len() < m6_max_ep {
                lane.outcome_ep_step_rewards.push(reward_pre_m6);
            }

            // M7 episode-return accumulator: uses `reward_pre_m6`,
            // NOT the post-bonus reward. Prevents the prototype
            // from being built out of the M7 bonus's own echo.
            self.approach_ep_returns[i] += reward_pre_m6;

            // RND curiosity: train predictor on current z, emit
            // MSE as intrinsic reward. Unlike kindle's surprise
            // (which decays as the WM converges), RND's target is
            // a frozen random net so the curiosity signal stays
            // alive as long as there are states the predictor
            // hasn't been fit against.
            // RND reads the obs TOKEN (pre-encoder, 64-dim) rather
            // than the post-encoder latent. An encoder that has
            // converged tightly clusters latents into a narrow
            // region where the predictor can match the target
            // quickly on any z — killing RND's signal. The obs
            // token carries more raw variation across frames, so
            // MSE stays informative longer.
            let rnd_reward = if rnd_state.is_some() {
                let mse = rnd_mses[i];
                rnd_mse_sum += mse;
                rnd_mse_count += 1;
                rnd_alpha * mse
            } else {
                0.0
            };

            // DIAYN bonus: log q(option | z) - log(1/K). Trains the
            // discriminator on (current z, current option index)
            // simultaneously. Only fires when L1 is active (option_idx
            // is meaningful). The reward pushes options to produce
            // mutually-distinguishable z trajectories.
            let diayn_reward = if let Some(state) = diayn_state.as_mut() {
                let opt_idx = lane.current_option as usize;
                let r = state.step(z_row, opt_idx);
                diayn_reward_sum += r;
                diayn_reward_count += 1;
                diayn_alpha * r
            } else {
                0.0
            };

            // M8 delta-goal reward. First, consider recording a new
            // goal from the (prev, cur) OBS pair — gated by
            // `pred_error >= surprise_threshold` so only
            // WM-surprising transitions populate the bank. Then
            // score the current obs against the (possibly
            // just-updated) bank. `prev_obs` is cleared on episode
            // boundaries so cross-episode jumps never register.
            let dg_reward = if let Some(bank) = dg_bank.as_mut() {
                if env_boundary {
                    self.delta_goal_prev_latent[i] = None;
                }
                if pred_error >= dg_surprise_gate {
                    let prev = self.delta_goal_prev_latent[i].as_deref();
                    if bank.observe_delta(prev, obs_row) {
                        dg_events_this_step += 1;
                    }
                }
                // Always update prev_obs, even when the surprise
                // gate blocks recording, so the next step's delta
                // is measured against this step rather than against
                // a stale pre-gate observation.
                self.delta_goal_prev_latent[i] = Some(obs_row.to_vec());
                bank.reward(obs_row, dg_alpha, dg_clamp)
            } else {
                0.0
            };

            // Cross-episode state-action novelty. Keyed on the
            // PREVIOUS step's (obs_token, action) — that's the
            // transition we're rewarding. The obs token is used
            // (not the post-encoder latent) because a trained
            // encoder clusters latents tight enough that the
            // quantized key collapses to 1-2 cells — verified
            // empirically on cd82 (xeps=6 at default grid, xeps=12
            // at fine grid) before switching to obs. The obs
            // token carries the raw per-frame variation that
            // distinguishes states the agent should care about.
            let xeps_reward = if let Some(memory) = xeps_memory.as_mut() {
                if let (Some(prev), Some(prev_a)) = (lane.buffer.last(), self.xeps_prev_action[i]) {
                    let bonus = if env_boundary {
                        0.0
                    } else {
                        memory.reward(&prev.observation, prev_a, xeps_alpha)
                    };
                    memory.observe(&prev.observation, prev_a);
                    bonus
                } else {
                    0.0
                }
            } else {
                0.0
            };
            // Also clear prev_action on the boundary step itself, so
            // the NEXT step (first full step of the new episode)
            // sees an empty prev — no spurious credit before a
            // fresh action has been sampled.
            if env_boundary {
                self.xeps_prev_action[i] = None;
            }

            // Extrinsic reward (signed, passed through from the env).
            // Kindle's value head absorbs the mean; the per-step
            // variance stays in the advantage. This is the channel
            // stock RL consumes — scaled by alpha it sits alongside
            // kindle's intrinsic primitives. Harness sets
            // `self.extrinsic_reward[i]` via `set_extrinsic_reward`
            // before calling observe; we consume and zero it here.
            let ext_event = is_task_event(self.extrinsic_event[i].take(), self.extrinsic_reward[i]);
            let ext_reward = ext_alpha * self.extrinsic_reward[i];
            self.extrinsic_reward[i] = 0.0;

            // Intrinsic progress reward. Pre-alpha value fed by the
            // harness via `set_intrinsic_progress`. Adds to the
            // per-step reward sum (consumed by SIL accumulator and
            // policy training) BUT does NOT increment
            // `sil_ep_event_count` — that's reserved for real
            // task events. This keeps the win-classifier's
            // is_win label clean: a "win episode" still means an
            // episode with a real event, not just one with high
            // intermediate progress.
            //
            // Scaled by `ext_alpha` so the harness can set values in
            // [0, 1] just like extrinsic. Typical use: dense
            // configurational-progress signal from cell-level diffs
            // and per-episode entropy, magnitudes ~0.001-0.01.
            let progress_reward = ext_alpha * self.intrinsic_progress_reward[i];
            self.intrinsic_progress_reward[i] = 0.0;

            let reward = reward_pre_m6
                + m6_bonus
                + m7_reward
                + rnd_reward
                + diayn_reward
                + dg_reward
                + xeps_reward
                + ext_reward
                + progress_reward;

            // Accumulate the FULL per-step reward into the SIL episode
            // tracker. Distinct from outcome_ep_return (intrinsic only)
            // — SIL needs the actual gym/env signal to gate "successful
            // episode" pushes.
            lane.sil_ep_return += reward;
            // Count explicit task success or a legacy positive reward pulse.
            // Success is not synonymous with positive reward: Gymnasium
            // MountainCar, for example, terminates successfully on a -1 step.
            if ext_event {
                lane.sil_ep_event_count += 1;
                // Confidence: a validated extrinsic event raises C.
                // This is the ONLY signal that increases confidence.
                if self.config.confidence_mode {
                    lane.confidence =
                        (lane.confidence + self.config.confidence_win_increment).min(1.0);
                }
                // Emergent goal discovery: snapshot the latent at this
                // win-event into the per-env goal archive. The planner
                // uses these as attractors via cosine similarity (when
                // `planner_goal_alpha > 0`). FIFO bounded by
                // `goal_states_cap`. Skipped entirely when the goal
                // mechanism is off.
                if self.config.planner_goal_alpha > 0.0 && self.config.goal_states_cap > 0 {
                    let env_id = lane.adapter.id();
                    let key = if self.config.goal_states_cross_game {
                        0u32
                    } else {
                        env_id
                    };
                    let cap = self.config.goal_states_cap;
                    let q = self
                        .goal_states
                        .entry(key)
                        .or_insert_with(std::collections::VecDeque::new);
                    if q.len() >= cap {
                        q.pop_front();
                    }
                    q.push_back(z_row.to_vec());
                    // Sub-goal centroids: online k-means update over the
                    // same pool. Centroids share the same key (per-env
                    // or pooled, matching goal_states_cross_game).
                    if self.config.subgoal_k > 0 {
                        let centroids = self.subgoal_centroids.entry(key).or_insert_with(Vec::new);
                        online_kmeans_update(
                            centroids,
                            z_row,
                            self.config.subgoal_k,
                            self.config.subgoal_lr,
                        );
                    }
                }
            }

            // Cache per-lane reward for the coord head's next
            // REINFORCE update; the head uses this step's reward
            // as the advantage signal for the coordinates it
            // sampled PRE-step. The head's `sample` call caches μ
            // and sample; `train_coord_head` (called by the
            // harness after this `observe` returns) runs
            // train_step using that cached state + this reward.
            self.coord_last_reward[i] = reward;

            lane.buffer.push(Transition {
                observation: obs_row.to_vec(),
                latent: z_row.to_vec(),
                action: act_row.to_vec(),
                action_parameters: self.wm_action_token_scratch
                    [i * WM_ACTION_DIM + MAX_ACTION_DIM..(i + 1) * WM_ACTION_DIM]
                    .to_vec(),
                action_mask: self.action_masks[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM]
                    .to_vec(),
                reward,
                pred_error,
                value: lane.last_value,
                prob_taken: lane.last_prob_taken,
                logits_at_action: if self.config.use_kl_ppo {
                    lane.last_logits.clone()
                } else {
                    Vec::new()
                },
                option_idx: lane.current_option,
                env_id: lane.adapter.id(),
                env_boundary,
                episode_return: 0.0,
                episode_complete: false,
            });

            // Continuing tasks often expose a meaningful credit boundary long
            // before environment termination. Retain a bounded backward window
            // immediately at the positive event, without marking a fake reset
            // or cutting world-model continuity. Episode-level event SIL is
            // disabled when this mode is active, avoiding duplicate replay.
            if ext_event
                && self.config.use_sil
                && self.config.sil_event_filter
                && self.config.sil_event_horizon > 0
            {
                let samples = sil_samples_from_recent_trajectory(
                    &lane.buffer,
                    self.config.sil_event_horizon,
                    true,
                    self.config.value_target_clamp,
                );
                append_sil_samples(
                    &mut self.sil_buffer,
                    samples,
                    self.config.sil_buffer_capacity,
                );
            }

            lane.last_surprise = surprise;
            lane.last_novelty = novelty;
            lane.last_homeo = homeo;
            lane.last_order = order;
            lane.last_reward = reward;
            lane.last_base_reward = base_reward;

            // Confidence: pure per-step decay toward 0. The only signal
            // that increases C is an extrinsic event (handled at the
            // event site above). Decay represents "competence atrophies
            // without validation": no wins for a while → C drifts down →
            // planner shifts to exploration → may stumble on new wins →
            // C rises again. Steady-state C ≈ win_inc * events_per_step
            // / novelty_drop_rate.
            //
            // For tu93 in long_change_5k (~0.0008 events/step per lane,
            // win_inc=0.02, drop_rate=0.001): steady-state C ≈ 0.016,
            // which would mean almost-pure exploration. So scale
            // drop_rate to typical event density; default 0.001 per
            // step targets C ≈ 0.4 at ~0.02 events/step.
            if self.config.confidence_mode {
                lane.confidence =
                    (lane.confidence - self.config.confidence_novelty_drop_rate).max(0.0);
            }

            // L1: accumulate reward into the current option's return and
            // count down. The next act() call will detect steps_left == 0
            // and handle training + resampling.
            lane.option_return += reward;
            lane.option_steps_left = lane.option_steps_left.saturating_sub(1);
            lane.option_elapsed = lane.option_elapsed.saturating_add(1);
        }
        // (P4) Apply surprise-triggered replans: drop the queued plan
        // for lanes whose realized dynamics diverged. The next
        // plan_and_queue call replans from the fresh latent.
        if !replan_lanes.is_empty() {
            self.replan_clears += replan_lanes.len() as u64;
        }
        for i in replan_lanes {
            self.planner_queue[i].clear();
        }

        // (P2) Write admitted surprise-ring entries. The visual slot
        // still holds this step's post-action frames; the scratches
        // still hold this step's action / acted-from-latent / obs
        // rows. FIFO overwrite once at capacity; priorities drive the
        // sampling in surprise_replay_step.
        if !ring_admits.is_empty() {
            if let Some((vis_ptr, _)) = self.wm_session.input_host_ptr("visual_obs") {
                let ld_ = self.latent_dim;
                for (i, prio) in ring_admits {
                    // Safety: the slot is host-visible and the GPU read
                    // of this step completed inside the WM dispatch's
                    // wait(); rows are within visual_obs_size_bytes.
                    let frame = unsafe {
                        std::slice::from_raw_parts(
                            (vis_ptr as *const f32).add(i * ring_per_sample),
                            ring_per_sample,
                        )
                    }
                    .to_vec();
                    let action = self.wm_action_token_scratch
                        [i * WM_ACTION_DIM..(i + 1) * WM_ACTION_DIM]
                        .to_vec();
                    let z_prev = self.z_target_scratch[i * ld_..(i + 1) * ld_].to_vec();
                    let obs =
                        self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM].to_vec();
                    let env_id = self.lanes[i].adapter.id();
                    if self.surprise_ring_frames.len() < ring_cap {
                        self.surprise_ring_frames.push(frame);
                        self.surprise_ring_actions.push(action);
                        self.surprise_ring_zprev.push(z_prev);
                        self.surprise_ring_obs.push(obs);
                        self.surprise_ring_env.push(env_id);
                        self.surprise_ring_prio.push(prio);
                    } else {
                        let at = self.surprise_ring_next % ring_cap;
                        self.surprise_ring_frames[at] = frame;
                        self.surprise_ring_actions[at] = action;
                        self.surprise_ring_zprev[at] = z_prev;
                        self.surprise_ring_obs[at] = obs;
                        self.surprise_ring_env[at] = env_id;
                        self.surprise_ring_prio[at] = prio;
                        self.surprise_ring_next = (self.surprise_ring_next + 1) % ring_cap;
                    }
                }
            }
        }

        // Restore the outcome head after the loop finishes.
        self.outcome_head = m6_head;
        self.last_outcome_baseline = m6_baseline_diag;
        // Restore the M7 state.
        self.approach_state = m7_state;
        self.rnd_state = rnd_state;
        self.last_rnd_mse = if rnd_mse_count > 0 {
            rnd_mse_sum / rnd_mse_count as f32
        } else {
            0.0
        };
        self.diayn_state = diayn_state;
        self.last_diayn_reward = if diayn_reward_count > 0 {
            diayn_reward_sum / diayn_reward_count as f32
        } else {
            0.0
        };
        self.delta_goal_bank = dg_bank;
        self.last_delta_goal_events = dg_events_this_step;
        self.xeps_memory = xeps_memory;

        // --- Policy + value training (one batched dispatch over all lanes) ---
        //
        // Gate is applied per-lane via the reward/advantage signal. Lanes
        // whose entropy is below the floor, or which are still in warmup,
        // contribute a zero gradient signal (LR scale 0) but share the
        // single graph dispatch.
        if self.step_count >= self.config.warmup_steps {
            self.policy_step_batched();
            // SIL is an independent replay update: it must still run when the
            // current on-policy batch has zero advantage. Keeping this call
            // inside the successful policy-gradient paths made all n_step=1
            // agents silently skip SIL and made PPO run it once per epoch.
            let sil_interval = if self.config.n_step >= 2 {
                if self.config.rollout_length > 1 {
                    self.config.rollout_length
                } else {
                    self.config.policy_update_interval.max(1)
                }
            } else {
                1
            };
            if (self.step_count + 1).is_multiple_of(sil_interval) {
                self.maybe_run_sil_update();
            }
        }

        // --- Replay mixing: one batched replay per call, one transition
        // sampled per lane (no zero-row dilution).
        if rng.random_range(0.0..1.0) < self.config.replay_ratio {
            self.replay_step(rng);
        }

        // (P2) Surprise frame-replay: same cadence knob as replay
        // mixing, CNN modes only (replay_step early-returns there).
        if self.config.surprise_ring_capacity > 0
            && rng.random_range(0.0..1.0_f32) < self.config.replay_ratio
        {
            self.surprise_replay_step(rng);
        }

        // --- Win-classifier replay step: dispatch one wm_session
        // forward+backward in "value-gate" mode on a BALANCED batch
        // from win_buffer + loss_buffer. The BCE loss backprops
        // through encoder + classifier head — encoder learns to
        // separate win-trajectory and loss-trajectory latents (the
        // "high-level direction in latent space"). Skipped until
        // both buffers have data.
        if self.config.value_head_train_coef > 0.0
            && self.win_buffer.values().any(|q| !q.is_empty())
            && self.loss_buffer.values().any(|q| !q.is_empty())
        {
            self.value_replay_step(rng);
        }

        // k-step WM training: with configured probability, sample
        // (t, t+k) tuples from lane buffers and train WM iteratively.
        // Forces depth-accuracy; combats compounding error at planner
        // rollout depth >5. Synced back to wm_session each step.
        if self.wm_kstep_session.is_some()
            && self.config.wm_kstep_train_prob > 0.0
            && rng.random_range(0.0..1.0_f32) < self.config.wm_kstep_train_prob
        {
            self.wm_kstep_step(rng);
        }

        // --- Representation drift monitor (shared probe set, WM session) ---
        if self.step_count == self.config.warmup_steps && self.probe_obs.is_none() {
            self.capture_probe_reference();
        }
        if self.step_count > 0 && self.step_count.is_multiple_of(self.config.drift_interval) {
            self.measure_drift();
        }

        // (P5) Plasticity maintenance: gentle shrink-and-perturb on the
        // representation + dynamics stack.
        if self.config.plasticity_interval > 0
            && self.step_count > 0
            && self
                .step_count
                .is_multiple_of(self.config.plasticity_interval)
        {
            self.plasticity_perturb(rng);
        }

        self.step_count += 1;
    }

    /// Run one k-step world-model training step: sample `wm_kstep_batch`
    /// boundary-free windows `(z_t, a_{t+1..t+k}, z_{t+k})` from the lane
    /// buffers, seed the kstep session's mean-dynamics params from the
    /// canonical WM, take one gradient step on the k-step rollout loss,
    /// and sync the updated mean params back into `wm_session`.
    fn wm_kstep_step<R: Rng>(&mut self, rng: &mut R) {
        let Some(ref mut sess) = self.wm_kstep_session else {
            return;
        };
        let k = self.config.wm_kstep_k;
        let batch = self.config.wm_kstep_batch.max(1);
        let ld = self.config.latent_dim;

        let mut staged = 0usize;
        let mut donor_rows: Vec<usize> = Vec::with_capacity(batch);
        let mut failed_rows: Vec<usize> = Vec::new();
        for row in 0..batch {
            let mut found = false;
            for _attempt in 0..8 {
                let lane_idx = rng.random_range(0..self.lanes.len());
                let lane = &self.lanes[lane_idx];
                let blen = lane.buffer.len();
                if blen < k + 1 {
                    continue;
                }
                let t = rng.random_range(0..blen - k);
                // Window validity: transitions t+1..=t+k must not cross
                // an env/episode boundary — a k-step rollout spanning a
                // reset trains the WM on a teleport.
                let crosses_boundary = (t + 1..=t + k).any(|j| lane.buffer.get(j).env_boundary);
                if crosses_boundary {
                    continue;
                }
                let z_start = &lane.buffer.get(t).latent;
                let z_end = &lane.buffer.get(t + k).latent;
                let dst_z = &mut self.kstep_z_scratch[row * ld..(row + 1) * ld];
                let copy_len = z_start.len().min(ld);
                dst_z[..copy_len].copy_from_slice(&z_start[..copy_len]);
                if copy_len < ld {
                    dst_z[copy_len..].fill(0.0);
                }
                let dst_zt = &mut self.kstep_z_target_scratch[row * ld..(row + 1) * ld];
                let tlen = z_end.len().min(ld);
                dst_zt[..tlen].copy_from_slice(&z_end[..tlen]);
                if tlen < ld {
                    dst_zt[tlen..].fill(0.0);
                }
                for i in 0..k {
                    // Transition records store {latent: z_after_action,
                    // action: the action that PRODUCED that latent}. So
                    // rolling forward from tr[t].latent, the first action
                    // applied is tr[t+1].action — the i-th rollout step
                    // uses tr[t+1+i].action (previously `t + i`: every
                    // kstep window was trained with the action sequence
                    // shifted one step into the past).
                    let transition = lane.buffer.get(t + 1 + i);
                    let dst_a = &mut self.kstep_action_scratch_per_step[i]
                        [row * WM_ACTION_DIM..(row + 1) * WM_ACTION_DIM];
                    compose_wm_action_token(
                        &transition.action,
                        &transition.action_parameters,
                        dst_a,
                    );
                }
                found = true;
                staged += 1;
                break;
            }
            if !found {
                failed_rows.push(row);
            } else {
                donor_rows.push(row);
            }
        }
        if staged == 0 {
            return;
        }
        // Donor-fill rows that couldn't find a valid window instead of
        // zero-filling them: zero rows train the WM toward a spurious
        // f(0, 0…0) = 0 fixed point at the origin (same rationale as
        // the donor fill in `replay_step`). Duplicating a real sample
        // just up-weights it slightly.
        for (fi, &row) in failed_rows.iter().enumerate() {
            let donor = donor_rows[fi % donor_rows.len()];
            self.kstep_z_scratch
                .copy_within(donor * ld..(donor + 1) * ld, row * ld);
            self.kstep_z_target_scratch
                .copy_within(donor * ld..(donor + 1) * ld, row * ld);
            for i in 0..k {
                self.kstep_action_scratch_per_step[i].copy_within(
                    donor * WM_ACTION_DIM..(donor + 1) * WM_ACTION_DIM,
                    row * WM_ACTION_DIM,
                );
            }
        }

        // Mean-dynamics params shared between the canonical WM and the
        // kstep session. sigma_proj is deliberately excluded in BOTH
        // directions: it's trained only in wm_session (single-step
        // residual regression) and kstep's sigma_proj never receives
        // gradients — copying its random init around would trash
        // σ-head learning.
        let hd_ = self.config.hidden_dim;
        let ld_ = self.config.latent_dim;
        let wm_params: [(&str, usize); 6] = [
            ("world_model.z_proj.weight", ld_ * hd_),
            ("world_model.z_proj.bias", hd_),
            ("world_model.a_proj.weight", WM_ACTION_DIM * hd_),
            ("world_model.fc2.weight", hd_ * hd_),
            ("world_model.fc2.bias", hd_),
            ("world_model.fc_out.weight", hd_ * ld_),
        ];

        // Seed kstep's mean params from the canonical WM so the k-step
        // gradient below applies ON TOP of the online-trained weights.
        // Without this, the kstep session trains a private lineage
        // from its own random init and the sync-back after the step
        // would overwrite the trained WM with it (including right
        // after load_weights, which doesn't checkpoint this session).
        for (name, n_elem) in wm_params.iter() {
            let buf = &mut self.planner_param_buf[..*n_elem];
            self.wm_session.read_param(name, buf);
            sess.set_parameter(name, buf);
        }

        sess.set_input("z_input", &self.kstep_z_scratch);
        sess.set_input("z_target_kstep", &self.kstep_z_target_scratch);
        for i in 0..k {
            let name = format!("action_step_{}", i);
            sess.set_input(&name, &self.kstep_action_scratch_per_step[i]);
        }
        let lr = self.config.learning_rate * self.config.wm_kstep_loss_coef * self.batch_lr_scale;
        apply_lr(sess, lr, self.config.use_adam, self.config.adam_eps);
        sess.step();
        sess.wait();
        let l = sess.read_loss();
        if l.is_finite() {
            self.last_wm_kstep_loss = l;
        }

        // Sync the k-step-updated mean params back into the canonical
        // WM (used by encoder→z prediction and wm_planner). Combined
        // with the forward seed above, the net effect of this function
        // is exactly one k-step gradient step applied to the online
        // WM's current weights.
        for (name, n_elem) in wm_params.iter() {
            let buf = &mut self.planner_param_buf[..*n_elem];
            sess.read_param(name, buf);
            self.wm_session.set_parameter(name, buf);
        }
    }

    /// Run one win-classifier training step: sample a balanced batch
    /// from `win_buffer` and `loss_buffer` (half from each), stage the
    /// stored latents into wm_session's `value_z` input and the labels
    /// into `value_target`, flip the gates to `(wm_gate=0,
    /// value_gate=1)`, dispatch, and read the per-call BCE loss.
    ///
    /// The classifier head reads the STORED latents through the
    /// `value_z` input — the encoder is not on this gradient path (see
    /// the trade-off note at the `value_z` graph construction site).
    fn value_replay_step<R: Rng>(&mut self, rng: &mut R) {
        let n = self.lanes.len();
        // Collect env_ids with non-empty win and loss queues. Need
        // at least one of each to train balanced batches.
        let win_keys: Vec<u32> = self
            .win_buffer
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(k, _)| *k)
            .collect();
        let loss_keys: Vec<u32> = self
            .loss_buffer
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(k, _)| *k)
            .collect();
        if win_keys.is_empty() || loss_keys.is_empty() {
            return;
        }
        // Balanced batch: half win, half loss. Round-robin across
        // env_ids within each class so games contribute equally
        // regardless of how many of total wins they own. Cross-game
        // classifier bias (2026-05-15 finding) is what this fixes.
        // With an odd batch (notably batch_size == 1) the leftover row
        // goes to a random class: `n / 2 == 0` at batch_size 1 starved
        // the win class entirely and the classifier collapsed to
        // "always loss"; a fixed assignment would under-represent one
        // class on every step.
        let mut n_win = n / 2;
        if !n.is_multiple_of(2) && rng.random_range(0..2u32) == 0 {
            n_win += 1;
        }
        for row in 0..n {
            let from_win = row < n_win;
            let (env_id, copy_z, copy_label) = if from_win {
                let env_id = win_keys[row % win_keys.len()];
                let q = &self.win_buffer[&env_id];
                let i = rng.random_range(0..q.len());
                let entry = &q[i];
                (env_id, entry.0.clone(), entry.2)
            } else {
                let lossi = row - n_win;
                let env_id = loss_keys[lossi % loss_keys.len()];
                let q = &self.loss_buffer[&env_id];
                let i = rng.random_range(0..q.len());
                let entry = &q[i];
                (env_id, entry.0.clone(), entry.2)
            };
            let _ = env_id;
            let ld = self.config.latent_dim;
            let z_dst = &mut self.value_z_scratch[row * ld..(row + 1) * ld];
            let copy_len = copy_z.len().min(ld);
            z_dst[..copy_len].copy_from_slice(&copy_z[..copy_len]);
            if copy_len < ld {
                z_dst[copy_len..].fill(0.0);
            }
            self.value_target_scratch_vh[row] = copy_label;
        }
        // Encoder inputs (obs/task) and the WM inputs (action/z_target)
        // are unused in value-replay mode — the value head reads the
        // staged `value_z` latents directly and the WM branch is gated
        // off by wm_gate=0. Zero them so stale per-step values don't
        // linger in the device buffers.
        self.obs_token_scratch.fill(0.0);
        self.task_scratch.fill(0.0);
        self.wm_action_token_scratch.fill(0.0);
        self.z_target_scratch.fill(0.0);

        // Flip the gates.
        self.wm_gate_scalar[0] = 0.0;
        self.value_gate_scalar[0] = 1.0;

        let n_tok = self.lanes.len();
        self.wm_session
            .set_input("obs", &self.obs_token_scratch[..n_tok * OBS_TOKEN_DIM]);
        if matches!(self.config.encoder_kind, EncoderKind::EfficientNetV2S) {
            self.run_efficientnet_v2s();
        }
        self.wm_session
            .set_input("action", &self.wm_action_token_scratch);
        self.wm_session
            .set_input("z_target", &self.z_target_scratch);
        self.wm_session
            .set_input("task", &self.task_scratch[..n_tok * TASK_DIM]);
        // The classifier head reads the staged latents through the
        // `value_z` graph input — without this upload the head trains
        // against whatever the device buffer last held (zeros).
        self.wm_session.set_input("value_z", &self.value_z_scratch);
        self.wm_session
            .set_input("value_target", &self.value_target_scratch_vh);
        self.wm_session.set_input("wm_gate", &self.wm_gate_scalar);
        self.wm_session
            .set_input("value_gate", &self.value_gate_scalar);
        let lr = self.config.learning_rate
            * self.config.value_head_lr_scale
            * self.encoder_lr_scale
            * self.batch_lr_scale;
        apply_lr(
            &mut self.wm_session,
            lr,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.wm_session.step();
        self.wm_session.wait();
        let l = self.wm_session.read_loss();
        if l.is_finite() {
            self.last_value_head_loss = l;
        }
        self.value_head_updates += 1;

        // Restore default gates so subsequent wm_session calls (the
        // next observe() step) run normal WM forward without needing
        // to re-set the inputs every time.
        self.wm_gate_scalar[0] = 1.0;
        self.value_gate_scalar[0] = 0.0;
    }

    /// Run one world-model forward+backward pass on the currently staged
    /// `obs_token_scratch` / `wm_action_token_scratch` / `z_target_scratch` /
    /// `task_scratch` inputs. Returns the scalar batch-mean loss.
    ///
    /// obs/task are sliced to the first `lanes` rows: with
    /// `end_to_end_encoder && rollout_length > 1` those scratches are
    /// sized for the policy session's rollout batch, while the WM
    /// session's inputs stay `lanes`-row.
    fn wm_forward_backward_stacked(&mut self, lr: f32) -> f32 {
        let n_tok = self.lanes.len();
        self.wm_session
            .set_input("obs", &self.obs_token_scratch[..n_tok * OBS_TOKEN_DIM]);
        if matches!(self.config.encoder_kind, EncoderKind::EfficientNetV2S) {
            self.run_efficientnet_v2s();
        }
        self.wm_session
            .set_input("action", &self.wm_action_token_scratch);
        self.wm_session
            .set_input("z_target", &self.z_target_scratch);
        self.wm_session
            .set_input("task", &self.task_scratch[..n_tok * TASK_DIM]);
        // Value-target / gate inputs only exist when the value branch
        // was built into the graph. When off, skip them entirely — the
        // graph contract has no such inputs and set_input would error.
        if self.value_branch_on() {
            self.value_target_scratch_vh.fill(0.0);
            self.wm_gate_scalar[0] = 1.0;
            self.value_gate_scalar[0] = 0.0;
            self.wm_session
                .set_input("value_target", &self.value_target_scratch_vh);
            self.wm_session.set_input("wm_gate", &self.wm_gate_scalar);
            self.wm_session
                .set_input("value_gate", &self.value_gate_scalar);
        }
        apply_lr(
            &mut self.wm_session,
            lr,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.wm_session.step();
        self.wm_session.wait();
        self.wm_session.read_loss()
    }

    /// True iff the wm_session graph was built with the value-head
    /// branch (and the `value_target`, `wm_gate`, `value_gate` inputs).
    /// Mirrors the condition in `Agent::new`: training ON or planner
    /// consumption ON.
    fn value_branch_on(&self) -> bool {
        self.config.value_head_train_coef > 0.0 || self.config.planner_value_alpha > 0.0
    }

    /// Stage optional normalized action parameters for the next `observe()`.
    /// `parameters` is flat `[batch_size × ACTION_PARAMETER_DIM]`; `active`
    /// selects lanes whose executed action consumed those values. Inactive
    /// rows get a zero parameter tail. The staging is one-shot and clears
    /// after `observe`, preventing stale click coordinates from leaking into
    /// later ordinary actions.
    pub fn set_action_parameters(&mut self, parameters: &[f32], active: &[bool]) {
        assert_eq!(
            parameters.len(),
            self.action_parameter_scratch.len(),
            "set_action_parameters: expected {} parameters, got {}",
            self.action_parameter_scratch.len(),
            parameters.len()
        );
        assert_eq!(
            active.len(),
            self.action_parameter_active.len(),
            "set_action_parameters: expected {} active flags, got {}",
            self.action_parameter_active.len(),
            active.len()
        );
        for (dst, &src) in self.action_parameter_scratch.iter_mut().zip(parameters) {
            *dst = if src.is_finite() {
                src.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        self.action_parameter_active.copy_from_slice(active);
    }

    /// Declare which discrete actions consume the auxiliary parameter tail.
    /// Flat layout `[batch_size × MAX_ACTION_DIM]`. This is deliberately
    /// separate from availability: an action may be legal yet unparameterized.
    /// Changing a lane's declaration invalidates plans built under the old
    /// action schema, so that lane's queue is cleared.
    pub fn set_action_parameter_masks(&mut self, masks: &[bool]) {
        assert_eq!(
            masks.len(),
            self.action_parameter_masks.len(),
            "set_action_parameter_masks: expected {} entries, got {}",
            self.action_parameter_masks.len(),
            masks.len()
        );
        for lane in 0..self.lanes.len() {
            let range = lane * MAX_ACTION_DIM..(lane + 1) * MAX_ACTION_DIM;
            if self.action_parameter_masks[range.clone()] != masks[range.clone()] {
                self.planner_queue[lane].clear();
                self.last_planned_action_parameters[lane] = None;
            }
        }
        self.action_parameter_masks.copy_from_slice(masks);
    }

    /// Reset every action to the unparameterized default.
    pub fn clear_action_parameter_masks(&mut self) {
        self.action_parameter_masks.fill(false);
        for lane in 0..self.lanes.len() {
            self.planner_queue[lane].clear();
            self.last_planned_action_parameters[lane] = None;
        }
    }

    /// Consume planner-supplied parameters for the actions returned by the
    /// most recent `act()` call. Each entry is `Some([x, y])` only when that
    /// action came from a parameterized planner candidate. Calling this before
    /// `act()` or twice returns `None` for the affected lane.
    pub fn take_planned_action_parameters(&mut self) -> Vec<Option<[f32; ACTION_PARAMETER_DIM]>> {
        self.last_planned_action_parameters
            .iter_mut()
            .map(Option::take)
            .collect()
    }

    /// Compose policy action identities and one-shot parameters into the
    /// world-model-only action rows.
    fn stage_wm_action_tokens(&mut self) {
        for lane in 0..self.lanes.len() {
            let base =
                &self.action_token_scratch[lane * MAX_ACTION_DIM..(lane + 1) * MAX_ACTION_DIM];
            let wm =
                &mut self.wm_action_token_scratch[lane * WM_ACTION_DIM..(lane + 1) * WM_ACTION_DIM];
            if self.action_parameter_active[lane] {
                let params = &self.action_parameter_scratch
                    [lane * ACTION_PARAMETER_DIM..(lane + 1) * ACTION_PARAMETER_DIM];
                compose_wm_action_token(base, params, wm);
            } else {
                compose_wm_action_token(base, &[], wm);
            }
        }
        self.action_parameter_active.fill(false);
        self.action_parameter_scratch.fill(0.0);
    }

    /// Set per-lane action availability masks for the NEXT and subsequent
    /// `act()` calls. Flat layout `[batch_size × MAX_ACTION_DIM]`; entries
    /// `>= 0.5` are treated as valid, `< 0.5` as invalid. Invalid actions get
    /// their logits forced to a large-negative value before the categorical
    /// sample, so the policy never picks them. The masked distribution is
    /// also used for `last_prob_taken` (PPO's π_old denominator) and for
    /// the cached `last_logits` (KL-PPO's old-policy reference), keeping
    /// the importance ratio's denominator consistent with the sampling
    /// distribution.
    ///
    /// Persists across calls until re-set. There is no auto-clear (unlike
    /// `extrinsic_reward`) because masks typically stay stable over many env
    /// steps. [`Self::clear_action_masks`] restores the adapter's static
    /// action range.
    ///
    /// Panics if `masks.len() != batch_size * MAX_ACTION_DIM`. A no-op
    /// when at least one entry per lane row is < 0.5; if all entries in a
    /// row are masked off, the row is treated as un-masked to avoid
    /// producing NaN softmax (no valid action to sample).
    pub fn set_action_masks(&mut self, masks: &[f32]) {
        assert_eq!(
            masks.len(),
            self.action_masks.len(),
            "set_action_masks: expected {} entries (batch_size {} × MAX_ACTION_DIM {}), got {}",
            self.action_masks.len(),
            self.lanes.len(),
            MAX_ACTION_DIM,
            masks.len()
        );
        self.action_masks.copy_from_slice(masks);
    }

    /// Reset action masks to each adapter's static action range.
    /// Discrete padded heads remain invalid; continuous rows are all-valid.
    pub fn clear_action_masks(&mut self) {
        self.action_masks.fill(0.0);
        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            let row =
                &mut self.action_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            match lane.adapter.action_kind() {
                ActionKind::Discrete { n } => row[..n.min(MAX_ACTION_DIM)].fill(1.0),
                ActionKind::Continuous { .. } => row.fill(1.0),
            }
        }
    }

    /// Synchronize per-lane action masks from native environments before
    /// [`Self::act`]. Environments without a dynamic mask expose all actions.
    pub fn set_action_masks_from_envs(&mut self, envs: &[&dyn Environment]) {
        let n = self.lanes.len();
        assert_eq!(envs.len(), n, "envs.len() must equal N");
        self.action_masks.fill(0.0);

        for (lane_idx, (lane, env)) in self.lanes.iter().zip(envs).enumerate() {
            let row =
                &mut self.action_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            match lane.adapter.action_kind() {
                ActionKind::Continuous { .. } => row.fill(1.0),
                ActionKind::Discrete { n: action_count } => {
                    assert_eq!(
                        env.num_actions(),
                        action_count,
                        "environment and adapter action counts differ on lane {lane_idx}"
                    );
                    if let Some(mask) = env.action_mask() {
                        assert_eq!(
                            mask.len(),
                            action_count,
                            "environment action mask length differs from num_actions on lane {lane_idx}"
                        );
                        assert!(
                            mask.iter().any(|&valid| valid),
                            "environment action mask has no valid action on lane {lane_idx}"
                        );
                        for (dst, valid) in row.iter_mut().zip(mask) {
                            *dst = f32::from(valid);
                        }
                    } else {
                        row[..action_count].fill(1.0);
                    }
                }
            }
        }
    }

    /// Populate `policy_action_mask_scratch` from current per-lane
    /// `action_masks`. For `policy_batch == n_lanes` (rollout_length=1)
    /// this is a direct copy. For larger `policy_batch`, the first
    /// `n_lanes` rows get the current masks; remaining rows are
    /// filled with all-1.0 (no masking), since per-rollout-step
    /// masks aren't currently tracked in the buffer. Rollout paths
    /// (n_step >= 2 or rollout_length > 1) thus mask only the most
    /// recent step accurately — this is a current limitation.
    fn populate_policy_action_mask_scratch(&mut self) {
        self.policy_action_mask_scratch.fill(1.0);
        let n = self.lanes.len();
        let cap = self.policy_action_mask_scratch.len();
        let copy_len = (n * MAX_ACTION_DIM).min(cap).min(self.action_masks.len());
        self.policy_action_mask_scratch[..copy_len].copy_from_slice(&self.action_masks[..copy_len]);
    }

    /// party that knows the env's native reward. Kindle zeroes the
    /// per-lane value inside `observe` after consuming it, so a
    /// missed call doesn't silently replay the previous step.
    pub fn set_extrinsic_reward(&mut self, rewards: &[f32]) {
        if self.config.extrinsic_reward_alpha == 0.0 {
            return;
        }
        assert_eq!(
            rewards.len(),
            self.extrinsic_reward.len(),
            "set_extrinsic_reward: expected {} rewards, got {}",
            self.extrinsic_reward.len(),
            rewards.len()
        );
        self.extrinsic_reward.copy_from_slice(rewards);
    }

    /// Set explicit per-lane task-event decisions for the next `observe()`.
    /// This is separate from scalar reward so successful transitions with
    /// non-positive reward can seed event-filtered SIL and positive-reward
    /// transitions that are not achievements can explicitly suppress it.
    /// Decisions are one-shot and clear after `observe()`.
    pub fn set_extrinsic_events(&mut self, events: &[bool]) {
        if self.config.extrinsic_reward_alpha == 0.0 {
            return;
        }
        assert_eq!(
            events.len(),
            self.extrinsic_event.len(),
            "set_extrinsic_events: expected {} events, got {}",
            self.extrinsic_event.len(),
            events.len()
        );
        self.extrinsic_event
            .iter_mut()
            .zip(events)
            .for_each(|(slot, &event)| *slot = Some(event));
    }

    /// Per-lane empowerment estimate from the most recent
    /// `plan_and_queue` call (updated at planner cadence, not
    /// every step). Mean-of-per-dim cross-sample variance of
    /// step-0 `z_next` across the m planner samples; rises when
    /// different first actions diverge state, falls when stuck.
    /// Returns zeros for lanes that weren't planned this call,
    /// and an empty vec when the planner is off.
    pub fn empowerment(&self) -> Vec<f32> {
        self.last_empowerment.clone()
    }

    /// Per-lane intrinsic progress reward for the NEXT `observe()`.
    /// Adds to per-step reward sum like extrinsic, but does NOT
    /// increment `sil_ep_event_count` — the win-classifier's is_win
    /// label stays gated on real task events. Use for dense
    /// progress signals (persistent configurational change, per-
    /// episode entropy growth, empowerment) that should bias
    /// exploration without polluting "did this episode win?".
    /// Scaled by `extrinsic_reward_alpha` so values in [0, 1]
    /// give comparable per-step magnitudes to extrinsic. Cleared
    /// inside observe after consumption.
    pub fn set_intrinsic_progress(&mut self, rewards: &[f32]) {
        if self.config.extrinsic_reward_alpha == 0.0 {
            return;
        }
        assert_eq!(
            rewards.len(),
            self.intrinsic_progress_reward.len(),
            "set_intrinsic_progress: expected {} rewards, got {}",
            self.intrinsic_progress_reward.len(),
            rewards.len()
        );
        self.intrinsic_progress_reward.copy_from_slice(rewards);
    }

    /// Mutate the base learning rate at runtime. Picked up by every
    /// per-step `set_learning_rate` call across all sessions on the
    /// next `observe`. Use case: drop LR once a sustained-solve
    /// threshold is reached, to prevent the on-policy AC post-solve
    /// crash. Does NOT modify `lr_policy`, which may need independent tuning.
    ///
    /// **Footgun:** the policy session uses `lr_policy`, not
    /// `learning_rate`. Calling only `set_learning_rate` updates the
    /// encoder/replay/option LRs but leaves the policy LR unchanged —
    /// usually a silent no-op for the post-solve-crash use case, where
    /// the policy is the destabilizer. Prefer `set_all_learning_rates`
    /// for a runtime LR schedule, or pair this call with
    /// `set_lr_policy(lr * 0.5)`.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Set the base and policy learning rates together, preserving the
    /// `0.5×` ratio the Python constructor uses. Intended for schedules that want a
    /// single knob to control the whole agent's update magnitude.
    ///
    /// Equivalent to:
    /// ```ignore
    /// agent.set_learning_rate(lr);
    /// agent.set_lr_policy(lr * 0.5);
    /// ```
    /// without the easy-to-forget pair-call. See `set_learning_rate`
    /// for the footgun this exists to avoid.
    pub fn set_all_learning_rates(&mut self, lr: f32) {
        self.config.learning_rate = lr;
        self.config.lr_policy = lr * 0.5;
    }

    /// Mutate the entropy bonus weight at runtime. Currently only
    /// effective for the end_to_end_encoder graph (entropy_beta is a
    /// graph input there, set per-step by `feed_entropy_beta_input`).
    /// Other graphs bake entropy_beta as a compile-time constant.
    /// Use case: anneal entropy from a high initial value down to 0
    /// over training, to maintain exploration early then commit late
    /// — the standard PPO/A2C entropy schedule that helps escape
    /// local-optimum policy plateaus on dense-reward envs like
    /// LunarLander.
    pub fn set_entropy_beta(&mut self, beta: f32) {
        self.config.entropy_beta = beta;
    }

    /// Change action persistence at runtime. Cached actions are cleared so the
    /// new repeat length takes effect on the next `act()` call rather than
    /// inheriting the remainder of an old macro-action.
    pub fn set_action_repeat(&mut self, repeat: usize) {
        self.config.action_repeat = repeat.max(1);
        for lane in &mut self.lanes {
            lane.cached_action = None;
            lane.repeats_left = 0;
        }
    }

    /// Feed the current `config.entropy_beta` into the e2e policy
    /// graph as input "entropy_beta". No-op when e2e is off OR when
    /// the construction-time entropy_beta was 0 (in which case the
    /// entropy branch was fully elided and the input doesn't exist
    /// in the graph). Called before every `policy_session.step()` in
    /// the e2e path.
    fn feed_entropy_beta_input(&mut self) {
        if self.config.end_to_end_encoder && self.entropy_beta_input_present {
            let buf = [self.config.entropy_beta];
            self.policy_session.set_input("entropy_beta", &buf);
        }
    }

    /// Feed `config.kl_beta` into the KL-PPO graph as input "kl_beta".
    /// No-op when use_kl_ppo is off OR the KL branch was elided.
    fn feed_kl_beta_input(&mut self) {
        if self.old_logits_input_present {
            let buf = [self.config.kl_beta];
            self.policy_session.set_input("kl_beta", &buf);
        }
    }

    /// Runtime setter for KL-PPO's β. Picked up on next training step.
    /// Use case: Schulman 2017's adaptive β schedule — read observed
    /// KL via `last_kl`, double β if KL > target × 1.5, halve if
    /// KL < target / 1.5. Avoids the difficulty of finding the right
    /// fixed β at compile time.
    pub fn set_kl_beta(&mut self, beta: f32) {
        self.config.kl_beta = beta;
    }

    /// Most recent observed KL(π_new ‖ π_old) from the last training
    /// step (averaged over the policy_batch). Returns 0.0 when KL-PPO
    /// is off or no training step has run.
    pub fn last_kl(&self) -> f32 {
        self.last_kl
    }

    /// Mutate the policy-session learning rate at runtime. See
    /// `set_learning_rate` for the post-solve-crash use case; the
    /// policy LR is the dominant lever for this since the e2e encoder
    /// gradient flow makes the policy and value updates the primary
    /// destabilizers post-solve.
    pub fn set_lr_policy(&mut self, lr: f32) {
        self.config.lr_policy = lr;
    }

    /// Print a summary of the largest policy-session gradient norms
    /// (descending), with grad/weight ratios. Diagnostic for "which
    /// layer is the gradient exploding/vanishing through". Use after
    /// a `policy_session.step()` so the gradient buffers contain
    /// fresh values from the last backward pass.
    ///
    /// Backed by `meganeura::Session::dump_grad_summary`.
    pub fn dump_policy_grad_summary(&self, top_n: usize) {
        self.policy_session.dump_grad_summary(top_n);
    }

    /// Bulk read of policy-session per-parameter gradient norms.
    /// Returns (name, ‖grad‖₂) pairs in compile order, for every
    /// parameter that has a gradient. Useful for programmatic
    /// instability detection (e.g., trigger LR drop when a specific
    /// layer's grad norm exceeds a threshold).
    pub fn policy_grad_norms(&self) -> Vec<(String, f32)> {
        self.policy_session.read_all_param_grad_norms()
    }

    /// Bulk read of policy-session per-parameter weight norms.
    /// Same shape as `policy_grad_norms`.
    pub fn policy_weight_norms(&self) -> Vec<(String, f32)> {
        self.policy_session.read_all_param_norms()
    }

    /// Snapshot the world-model session's encoder parameters for diagnostics
    /// and regression tests. This makes it possible to verify that the
    /// representation objective is actually updating the encoder rather than
    /// merely reporting a decreasing dynamics loss over fixed random features.
    pub fn dump_encoder_params(&self) -> Vec<(String, Vec<f32>)> {
        let names: Vec<String> = self
            .wm_session
            .param_names()
            .into_iter()
            .filter(|name| name.starts_with("encoder."))
            .map(str::to_string)
            .collect();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let n = self.wm_session.param_size(&name).expect("param size known");
            let mut buf = vec![0.0f32; n];
            self.wm_session.read_param(&name, &mut buf);
            out.push((name, buf));
        }
        out
    }

    /// Dump all policy-session parameters as `(name, values)` pairs.
    /// Values are flattened f32 vectors in the parameter's natural layout
    /// (compile-order, stable across runs of the same graph). Use
    /// `load_policy_params` to restore the same structure.
    pub fn dump_policy_params(&self) -> Vec<(String, Vec<f32>)> {
        let names: Vec<String> = self
            .policy_session
            .param_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let n = self
                .policy_session
                .param_size(&name)
                .expect("param size known");
            let mut buf = vec![0.0f32; n];
            self.policy_session.read_param(&name, &mut buf);
            out.push((name, buf));
        }
        out
    }

    /// Upload all policy-session parameters from `(name, values)` pairs.
    /// Names and sizes must match the current graph. Missing or
    /// extra names are silently ignored — callers should validate.
    /// Returns the number of params successfully uploaded.
    pub fn load_policy_params(&mut self, params: &[(String, Vec<f32>)]) -> usize {
        let mut loaded = 0;
        for (name, data) in params {
            match self.policy_session.param_size(name) {
                Some(n) if n == data.len() => {
                    self.policy_session.upload_param(name, data);
                    loaded += 1;
                }
                _ => {}
            }
        }
        loaded
    }

    /// Set a per-parameter LR multiplier on the policy session, by
    /// name prefix. Effective LR for params matching the prefix is
    /// `lr_policy * batch_lr_scale * lr_scale * mul`. Longest-prefix-
    /// match wins; default 1.0 if no prefix matches.
    ///
    /// Use case from kindle: gradient inspection on LunarLander showed
    /// the policy_encoder.* params get 50-100× the gradient of policy.*
    /// (the actual policy head). Calling
    /// `set_policy_lr_multiplier("policy.", 5.0)` boosts the policy
    /// head's effective LR without changing the encoder's, rebalancing
    /// the asymmetric gradient flow.
    pub fn set_policy_lr_multiplier(&mut self, prefix: &str, mul: f32) {
        self.policy_session.set_lr_multiplier(prefix, mul);
    }

    /// Clear all per-parameter LR multipliers on the policy session.
    pub fn clear_policy_lr_multipliers(&mut self) {
        self.policy_session.clear_lr_multipliers();
    }

    /// Must be called by the harness when `encoder_kind = Cnn` and
    /// the input shape is flat NCHW of size `batch · channels · h · w`.
    /// No-op when the encoder is `Mlp`.
    ///
    /// This writes directly into meganeura's device-local,
    /// host-visible graph buffer via `Session::input_host_ptr` —
    /// the legacy scratch copy + `set_input` upload is gone, so
    /// one memcpy lands the frames in the GPU-side memory with
    /// no intermediary.
    pub fn set_visual_obs(&mut self, visual_obs: &[f32]) {
        if self.visual_obs_size_bytes == 0 {
            return;
        }
        let want = self.visual_obs_size_bytes / std::mem::size_of::<f32>();
        assert_eq!(
            visual_obs.len(),
            want,
            "set_visual_obs: expected {} floats (batch · channels · h · w), got {}",
            want,
            visual_obs.len()
        );
        let (dst, size) = self
            .wm_session
            .input_host_ptr("visual_obs")
            .expect("visual_obs input slot present under CNN encoder");
        let need = std::mem::size_of_val(visual_obs);
        debug_assert!(need <= size, "visual_obs write {need} > slot {size}");
        // Safety: `dst` points at a host-visible, host-coherent
        // buffer owned by wm_session for the lifetime of this
        // Agent. We write exactly `need` bytes which fits within
        // the slot size. `observe()` calls `wm_session.wait()`
        // before returning, so the previous step's GPU read has
        // completed — no race on the next host write.
        unsafe {
            std::ptr::copy_nonoverlapping(visual_obs.as_ptr() as *const u8, dst, need);
        }
    }

    /// Consume native environment results for one synchronous step.
    ///
    /// This is the preferred native-environment counterpart to [`Self::observe`]:
    /// it forwards task rewards and explicit task events, preserves terminal
    /// homeostatic state even when an environment auto-resets, and marks episode
    /// boundaries consistently.
    pub fn observe_step_results<R: Rng>(
        &mut self,
        results: &[StepResult],
        actions: &[Action],
        rng: &mut R,
    ) {
        let n = self.lanes.len();
        assert_eq!(results.len(), n, "results.len() must equal N");
        assert_eq!(actions.len(), n, "actions.len() must equal N");

        let rewards = results
            .iter()
            .map(|result| result.reward)
            .collect::<Vec<_>>();
        let events = results
            .iter()
            .map(|result| result.task_event)
            .collect::<Vec<_>>();
        let observations = results
            .iter()
            .map(|result| result.observation.clone())
            .collect::<Vec<_>>();
        let homeostatic = results
            .iter()
            .map(|result| result.homeostatic.as_slice())
            .collect::<Vec<_>>();

        self.set_extrinsic_reward(&rewards);
        self.set_extrinsic_events(&events);
        self.observe_with_homeostatic(&observations, actions, &homeostatic, rng);

        for (lane, result) in results.iter().enumerate() {
            if result.done() {
                self.mark_boundary(lane);
            }
        }
    }

    /// Forward-only probe of the dynamics model: ẑ_next = WM(z, a)
    /// for a full batch of rows. `z_rows` is `[N · latent_dim]`,
    /// `action_rows` is `[N · WM_ACTION_DIM]`; base-only
    /// `[N · MAX_ACTION_DIM]` rows are accepted and zero-pad the parameter
    /// tail for compatibility. Returns `[N · latent_dim]` predictions.
    ///
    /// Runs the wm_session at LR = 0 (no parameter change). The
    /// encoder branch runs on whatever obs/visual content is staged —
    /// its output only feeds the (ignored) loss, never ẑ. Intended
    /// for tests and harness diagnostics (e.g. verifying the model
    /// rolls forward in time); costs one GPU dispatch.
    pub fn wm_predict(&mut self, z_rows: &[f32], action_rows: &[f32]) -> Vec<f32> {
        let n = self.lanes.len();
        let ld = self.latent_dim;
        assert_eq!(
            z_rows.len(),
            n * ld,
            "wm_predict: z_rows must be N·latent_dim"
        );
        let padded_actions = if action_rows.len() == n * MAX_ACTION_DIM {
            let mut padded = vec![0.0; n * WM_ACTION_DIM];
            for lane in 0..n {
                compose_wm_action_token(
                    &action_rows[lane * MAX_ACTION_DIM..(lane + 1) * MAX_ACTION_DIM],
                    &[],
                    &mut padded[lane * WM_ACTION_DIM..(lane + 1) * WM_ACTION_DIM],
                );
            }
            Some(padded)
        } else {
            assert_eq!(
                action_rows.len(),
                n * WM_ACTION_DIM,
                "wm_predict: action_rows must be N·MAX_ACTION_DIM or N·WM_ACTION_DIM"
            );
            None
        };
        let action_rows = padded_actions.as_deref().unwrap_or(action_rows);
        self.wm_session
            .set_input("obs", &vec![0.0; n * OBS_TOKEN_DIM]);
        self.wm_session.set_input("action", action_rows);
        self.wm_session.set_input("z_target", z_rows);
        self.wm_session.set_input("task", &vec![0.0; n * TASK_DIM]);
        if self.value_branch_on() {
            self.value_target_scratch_vh.fill(0.0);
            self.value_z_scratch.fill(0.0);
            self.wm_gate_scalar[0] = 1.0;
            self.value_gate_scalar[0] = 0.0;
            self.wm_session
                .set_input("value_target", &self.value_target_scratch_vh);
            self.wm_session.set_input("value_z", &self.value_z_scratch);
            self.wm_session.set_input("wm_gate", &self.wm_gate_scalar);
            self.wm_session
                .set_input("value_gate", &self.value_gate_scalar);
        }
        // Forward-only prediction: skip the optimizer pass. With Adam,
        // lr = 0 still updates the moment buffers from this pass's
        // meaningless gradients (see act()).
        self.wm_session.clear_optimizer();
        self.wm_session.step();
        self.wm_session.wait();
        let mut out = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(3, &mut out);
        out
    }

    /// Raw host pointer + byte size of the WM session's
    /// `visual_obs` input buffer. `None` when the encoder is
    /// `Mlp` (no visual input slot).
    ///
    /// The buffer is allocated as `Memory::Shared` (device-local,
    /// host-visible, host-coherent). Writes through the pointer
    /// are picked up by the next `observe()` without any upload,
    /// staging, or external-memory import — a single memcpy lands
    /// the frame in GPU-visible memory.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this
    /// Agent. The caller must write at most `size_bytes` and must
    /// not initiate a write before the previous `observe()`
    /// returns (wm_session's `wait()` inside `observe` synchronizes
    /// with the GPU's read of the buffer).
    pub fn visual_obs_host_ptr(&self) -> Option<(*mut u8, usize)> {
        self.wm_session.input_host_ptr("visual_obs")
    }

    /// Byte size of the `visual_obs` slot, or 0 when the encoder
    /// is `Mlp`. Diagnostic.
    pub fn visual_obs_host_size(&self) -> usize {
        self.visual_obs_size_bytes
    }

    /// Writable host-mapped pointer + byte size of the V2-S session's
    /// `image` input buffer. `Some` only when
    /// `encoder_kind == EfficientNetV2S`. The harness writes raw
    /// `[batch · 3 · 192 · 192]` f32 NCHW pixels (in [0, 1] range) here
    /// each step before calling `observe`; kindle's V2-S forward picks
    /// up the writes via the implicit host-domain barrier on the next
    /// queue submit.
    pub fn image_input_host_ptr(&self) -> Option<(*mut u8, usize)> {
        self.efficientnet_session
            .as_ref()
            .and_then(|s| s.input_host_ptr("image"))
    }

    /// Byte size of the V2-S session's `image` input buffer, or 0
    /// when V2-S is not in use.
    pub fn image_input_host_size(&self) -> usize {
        self.efficientnet_input_size_bytes
    }

    /// The shared blade GPU context every meganeura `Session` in this
    /// agent was built with.  Hand a clone to a sibling compute pipeline
    /// (e.g. the V2-S preprocess pass) to issue dispatches on the same
    /// device + queue without a fresh context init.  Returns a clone of
    /// the agent's `Arc`; the underlying context lives until both the
    /// agent and the pipeline drop their handles.
    pub fn gpu_context(&self) -> Arc<blade_graphics::Context> {
        Arc::clone(&self.gpu)
    }

    /// Register (or replace) the per-lane Dullahan source buffer that
    /// the V2-S preprocess pipeline samples each step.  Calls through
    /// to [`crate::v2s_preprocess::PreprocessPipeline::register_lane`]
    /// — see that doc for the field semantics.  No-op when the agent
    /// was built with a non-V2-S encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn register_v2s_source(
        &mut self,
        lane: usize,
        fd: i32,
        allocation_size_bytes: u64,
        frame_data_offset_bytes: u64,
        bytes_per_frame: u64,
        src_w: u32,
        src_h: u32,
        is_rgba: bool,
    ) -> Result<(), String> {
        match self.v2s_preprocess.as_mut() {
            Some(pp) => pp.register_lane(
                lane,
                fd,
                allocation_size_bytes,
                frame_data_offset_bytes,
                bytes_per_frame,
                src_w,
                src_h,
                is_rgba,
            ),
            None => Err(
                "agent has no V2-S preprocess pipeline (encoder is not EfficientNetV2S)"
                    .to_string(),
            ),
        }
    }

    /// Drop the per-lane source for `lane` (e.g. before re-registering
    /// with a fresh fd after `reset_lane`).  No-op when the agent isn't
    /// using V2-S or `lane` is out of range.
    pub fn release_v2s_source(&mut self, lane: usize) {
        if let Some(pp) = self.v2s_preprocess.as_mut() {
            pp.release_lane(lane);
        }
    }

    /// Configure HUD-mask rects (in V2-S 192² input space) that the
    /// preprocess shader zeros after sampling.  See
    /// [`crate::v2s_preprocess::PreprocessPipeline::set_hud_masks`].
    /// No-op when the agent has no V2-S preprocess pipeline.
    pub fn set_v2s_hud_masks(&mut self, rects: &[(u32, u32, u32, u32)]) {
        if let Some(pp) = self.v2s_preprocess.as_mut() {
            pp.set_hud_masks(rects);
        }
    }

    /// Run one preprocess dispatch per registered lane: each lane's
    /// most-recently-presented Dullahan frame slot is read, resized,
    /// channel-swizzled, and written into V2-S's `image` input buffer
    /// at `[lane, :, :, :]`.  `slot_indices[lane]` matches what
    /// `VectorRustGameEnv::step_no_copy` returns; lanes whose source
    /// isn't registered or whose slot is `None` are skipped.
    ///
    /// Submits and waits — `observe()` can be called immediately after.
    /// No-op when the agent was built with a non-V2-S encoder.
    pub fn v2s_preprocess_step(&mut self, slot_indices: &[Option<u32>]) {
        if let Some(pp) = self.v2s_preprocess.as_mut() {
            pp.dispatch_step(slot_indices);
        }
    }

    pub fn bind_v2s_image_external(
        &mut self,
        source: blade_graphics::ExternalMemorySource,
        size: u64,
    ) -> Result<(), BindV2sImageError> {
        let session = self
            .efficientnet_session
            .as_mut()
            .ok_or(BindV2sImageError::NoV2sSession)?;
        session
            .bind_external_buffer(meganeura::ExternalSlot::Input("image"), source, size)
            .map_err(BindV2sImageError::BindFailed)
    }

    /// Run the V2-S forward on whatever raw RGB the harness wrote into
    /// `image_input_host_ptr`, then upload the output features into
    /// the WM session's `visual_obs` input so the WM step can consume
    /// them. Called automatically by the WM step path when
    /// `encoder_kind == EfficientNetV2S`; not a public API.
    fn run_efficientnet_v2s(&mut self) {
        let session = self
            .efficientnet_session
            .as_mut()
            .expect("EfficientNetV2S branch entered without session");
        session.step();
        session.wait();
        let n = self.efficientnet_output_buf.len();
        session.read_output_by_index(0, &mut self.efficientnet_output_buf);
        // Copy features into wm_session's visual_obs buffer (host-mapped).
        let (dst, dst_size) = self
            .wm_session
            .input_host_ptr("visual_obs")
            .expect("visual_obs slot present under EfficientNetV2S");
        let need = n * std::mem::size_of::<f32>();
        debug_assert!(
            need <= dst_size,
            "V2-S output {need} bytes > visual_obs slot {dst_size} bytes"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.efficientnet_output_buf.as_ptr() as *const u8,
                dst,
                need,
            );
        }
    }

    /// Sample one replay transition per lane and run a single batched WM
    /// forward+backward over the stacked rows. Each lane retries up to 8
    /// random indices to find a non-boundary pair in its own buffer; if
    /// that fails (buffer < 2, or deeply fragmented by env switches), we
    /// fall back to the most recent valid donor lane's sample so every
    /// batch row carries signal instead of zeros.
    /// (P5) Shrink-and-perturb plasticity maintenance over the wm
    /// session's encoder / world-model / recon parameters:
    /// p <- shrink·p + noise·std(p)·ε, ε ~ N(0, 1). Scale-aware (the
    /// noise rides each tensor's own std), so converged structure is
    /// nudged, not erased, while dormant directions get fresh signal
    /// to grab onto. Value head and task projections excluded — they
    /// are small and label-anchored.
    fn plasticity_perturb<R: Rng>(&mut self, rng: &mut R) {
        use std::f32::consts::TAU;
        let shrink = self.config.plasticity_shrink.clamp(0.5, 1.0);
        let noise = self.config.plasticity_noise.max(0.0);
        let names: Vec<String> = self
            .wm_session
            .param_names()
            .iter()
            .filter(|n| {
                n.starts_with("encoder.")
                    || n.starts_with("world_model.")
                    || n.starts_with("wm.recon.")
            })
            .map(|n| n.to_string())
            .collect();
        for name in names {
            let Some(len) = self.wm_session.param_size(&name) else {
                continue;
            };
            let mut buf = vec![0.0f32; len];
            self.wm_session.read_param(&name, &mut buf);
            let mean = buf.iter().sum::<f32>() / len.max(1) as f32;
            let var = buf.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / len.max(1) as f32;
            let std = var.max(0.0).sqrt();
            let amp = noise * std;
            let mut i = 0;
            while i < len {
                let u1: f32 = rng.random_range(1e-7..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                let mag = (-2.0 * u1.ln()).sqrt();
                buf[i] = shrink * buf[i] + amp * mag * (TAU * u2).cos();
                if i + 1 < len {
                    buf[i + 1] = shrink * buf[i + 1] + amp * mag * (TAU * u2).sin();
                }
                i += 2;
            }
            self.wm_session.set_parameter(&name, &buf);
        }
    }

    /// (P2) One surprise-replay dispatch: priority-sample a full batch
    /// of stored surprising transitions (post-action frame, action,
    /// acted-from latent) from the ring and run one encoder+WM
    /// forward/backward over them. This is the CNN-mode counterpart
    /// of `replay_step`, and the mechanism that concentrates encoder
    /// and WM gradient on novel elements the moment they appear —
    /// instead of waiting for the online stream to revisit them.
    ///
    /// Clobbers the visual slot; safe because the harness rewrites
    /// post-action frames before every observe() and nothing else
    /// reads the slot between observes (act() uses cached latents).
    fn surprise_replay_step<R: Rng>(&mut self, rng: &mut R) {
        let n = self.lanes.len();
        let len = self.surprise_ring_frames.len();
        if len < n.max(8) {
            return;
        }
        let per_sample = self.visual_obs_size_bytes / std::mem::size_of::<f32>() / n.max(1);
        if per_sample == 0 {
            return;
        }
        let ld = self.latent_dim;
        let total: f32 = self.surprise_ring_prio.iter().map(|p| p.max(1e-6)).sum();
        let mut visual = vec![0.0f32; n * per_sample];
        for row in 0..n {
            // Inverse-CDF priority-proportional sample (with
            // replacement across rows).
            let mut u = rng.random_range(0.0..total.max(1e-6));
            let mut idx = len - 1;
            for (j, p) in self.surprise_ring_prio.iter().enumerate() {
                let p = p.max(1e-6);
                if u <= p {
                    idx = j;
                    break;
                }
                u -= p;
            }
            visual[row * per_sample..(row + 1) * per_sample]
                .copy_from_slice(&self.surprise_ring_frames[idx]);
            self.wm_action_token_scratch[row * WM_ACTION_DIM..(row + 1) * WM_ACTION_DIM]
                .copy_from_slice(&self.surprise_ring_actions[idx]);
            self.z_target_scratch[row * ld..(row + 1) * ld]
                .copy_from_slice(&self.surprise_ring_zprev[idx]);
            self.obs_token_scratch[row * OBS_TOKEN_DIM..(row + 1) * OBS_TOKEN_DIM]
                .copy_from_slice(&self.surprise_ring_obs[idx]);
            let task_row = &mut self.task_scratch[row * TASK_DIM..(row + 1) * TASK_DIM];
            match self.task_embeddings.get(&self.surprise_ring_env[idx]) {
                Some(emb) => task_row.copy_from_slice(emb),
                None => task_row.fill(0.0),
            }
        }
        self.set_visual_obs(&visual);
        let loss = self.wm_forward_backward_stacked(
            self.config.learning_rate * self.encoder_lr_scale * self.batch_lr_scale * 0.5,
        );
        if loss.is_finite() {
            self.last_replay_loss = loss;
            self.surprise_replays += 1;
        } else {
            log::warn!(
                "surprise replay loss went non-finite at step {}, re-initialized WM params",
                self.step_count
            );
            init_parameters(&mut self.wm_session);
        }
    }

    fn replay_step<R: Rng>(&mut self, rng: &mut R) {
        // Replay in CNN-encoder mode would need visual frames
        // stored per-transition, which they aren't (buffer holds
        // the 64-dim token). Skip replay in that case; the online
        // WM gradient still flows every step.
        if matches!(
            self.config.encoder_kind,
            EncoderKind::Cnn { .. } | EncoderKind::CnnDqn { .. } | EncoderKind::EfficientNetV2S
        ) {
            return;
        }
        let ld = self.latent_dim;
        let n = self.lanes.len();

        // Per-lane sampled transitions; `None` for lanes that couldn't
        // find a valid non-boundary pair after 8 retries.
        let mut samples: Vec<Option<ReplaySample>> = Vec::with_capacity(n);
        for lane in &self.lanes {
            let buf_len = lane.buffer.len();
            if buf_len < 2 {
                samples.push(None);
                continue;
            }
            let mut found: Option<ReplaySample> = None;
            for _ in 0..8 {
                let idx = rng.random_range(0..buf_len - 1);
                let ti = lane.buffer.get(idx);
                let tj = lane.buffer.get(idx + 1);
                // Skip replay across env boundaries (latent → latent is
                // meaningless when env just switched).
                if tj.env_boundary || ti.env_id != tj.env_id {
                    continue;
                }
                // Replay row contract mirrors the online step: the
                // staged "z_target" slot is the WM's INPUT (state acted
                // from = ti.latent); the staged obs is the post-action
                // observation whose fresh encoding is the loss target
                // (tj.observation); the action is the one that produced
                // tj (tj.action). Pre-2026-06-10 this staged
                // (ti.observation, ti.action, tj.latent), which under
                // the old backward graph trained forward-with-wrong-
                // action — a third, conflicting direction signal.
                found = Some(ReplaySample {
                    obs: tj.observation.clone(),
                    action: tj.action.clone(),
                    action_parameters: tj.action_parameters.clone(),
                    z_target: ti.latent.clone(),
                    env_id: ti.env_id,
                });
                break;
            }
            samples.push(found);
        }

        // Fall back every failed lane to the most recent valid donor so
        // the batch has no zero rows (which dilute the shared gradient).
        // If *no* lane produced a sample, bail — nothing to replay.
        let donor = samples.iter().rev().find_map(|s| s.clone());
        let Some(donor) = donor else {
            return;
        };
        for s in samples.iter_mut() {
            if s.is_none() {
                *s = Some(donor.clone());
            }
        }

        for (i, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref().expect("filled above");
            let obs = &sample.obs;
            let act = &sample.action;
            let action_parameters = &sample.action_parameters;
            let z_target = &sample.z_target;
            let env_id = &sample.env_id;
            let obs_row = &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            obs_row.copy_from_slice(obs);
            let act_row =
                &mut self.wm_action_token_scratch[i * WM_ACTION_DIM..(i + 1) * WM_ACTION_DIM];
            compose_wm_action_token(act, action_parameters, act_row);
            let z_row = &mut self.z_target_scratch[i * ld..(i + 1) * ld];
            z_row.copy_from_slice(z_target);

            let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
            if let Some(emb) = self.task_embeddings.get(env_id) {
                task_row.copy_from_slice(emb);
            } else {
                task_row.fill(0.0);
            }
        }

        let loss = self.wm_forward_backward_stacked(
            self.config.learning_rate * self.encoder_lr_scale * self.batch_lr_scale * 0.5,
        );
        if loss.is_finite() {
            self.last_replay_loss = loss;
        } else {
            log::warn!(
                "replay loss went non-finite at step {}, re-initialized WM params",
                self.step_count
            );
            init_parameters(&mut self.wm_session);
            self.last_replay_loss = 0.0;
        }
    }

    /// Capture a shared probe reference set from lane 0's buffer (any lane
    /// would do — drift is a global representation-stability signal).
    fn capture_probe_reference(&mut self) {
        let lane0 = &self.lanes[0];
        let n_probe = 16.min(lane0.buffer.len());
        if n_probe == 0 {
            return;
        }
        let step = lane0.buffer.len() / n_probe.max(1);
        let mut observations = Vec::with_capacity(n_probe);
        let mut references = Vec::with_capacity(n_probe);
        for i in 0..n_probe {
            let idx = i * step;
            if idx < lane0.buffer.len() {
                let t = lane0.buffer.get(idx);
                observations.push(t.observation.clone());
                references.push(t.latent.clone());
            }
        }
        self.probe_obs = Some(observations);
        self.probe_reference = Some(references);
    }

    /// Measure representation drift by forwarding the probe set through the
    /// batched WM graph. The probe set isn't necessarily a multiple of N;
    /// we pad the remaining rows with zeros and ignore their outputs.
    fn measure_drift(&mut self) {
        // Drift probes are 64-dim obs tokens captured at warmup;
        // they're not visual frames, so a CNN encoder can't consume
        // them. Leave `last_drift = 0` for CNN-mode agents.
        if matches!(
            self.config.encoder_kind,
            EncoderKind::Cnn { .. } | EncoderKind::CnnDqn { .. } | EncoderKind::EfficientNetV2S
        ) {
            return;
        }
        let (probes, references) = match (self.probe_obs.as_ref(), self.probe_reference.as_ref()) {
            (Some(p), Some(r)) => (p.clone(), r.clone()),
            _ => return,
        };

        let n = self.lanes.len();
        let ld = self.latent_dim;

        let mut total = 0.0f32;
        let mut count = 0;

        // Use lane 0's env's task embedding for all probe rows — drift is a
        // global representation signal and its absolute scale depends on
        // task conditioning anyway. Unused (padded) rows see the same
        // embedding but their outputs are discarded.
        let task_emb = self
            .task_embeddings
            .get(&self.lanes[0].adapter.id())
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; TASK_DIM]);

        for chunk_start in (0..probes.len()).step_by(n) {
            // Stage batch inputs.
            self.obs_token_scratch.fill(0.0);
            self.wm_action_token_scratch.fill(0.0);
            self.z_target_scratch.fill(0.0);
            for i in 0..n {
                let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
                task_row.copy_from_slice(&task_emb);
            }
            let chunk_len = (probes.len() - chunk_start).min(n);
            for i in 0..chunk_len {
                let probe = &probes[chunk_start + i];
                let obs_row =
                    &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
                let copy_len = probe.len().min(OBS_TOKEN_DIM);
                obs_row[..copy_len].copy_from_slice(&probe[..copy_len]);
            }

            self.wm_session
                .set_input("obs", &self.obs_token_scratch[..n * OBS_TOKEN_DIM]);
            self.wm_session
                .set_input("action", &self.wm_action_token_scratch);
            self.wm_session
                .set_input("z_target", &self.z_target_scratch);
            self.wm_session
                .set_input("task", &self.task_scratch[..n * TASK_DIM]);
            if self.value_branch_on() {
                self.value_target_scratch_vh.fill(0.0);
                self.wm_gate_scalar[0] = 0.0;
                self.value_gate_scalar[0] = 0.0;
                self.wm_session
                    .set_input("value_target", &self.value_target_scratch_vh);
                self.wm_session.set_input("wm_gate", &self.wm_gate_scalar);
                self.wm_session
                    .set_input("value_gate", &self.value_gate_scalar);
            }
            // Forward-only probe: skip the optimizer pass. With Adam,
            // lr = 0 still updates the moment buffers from this pass's
            // meaningless gradients (see act()).
            self.wm_session.clear_optimizer();
            self.wm_session.step();
            self.wm_session.wait();
            self.wm_gate_scalar[0] = 1.0;

            let mut z_stack = vec![0.0f32; n * ld];
            self.wm_session.read_output_by_index(1, &mut z_stack);

            for i in 0..chunk_len {
                let current = &z_stack[i * ld..(i + 1) * ld];
                let reference = &references[chunk_start + i];
                let dist: f32 = current
                    .iter()
                    .zip(reference.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist.is_finite() {
                    total += dist;
                    count += 1;
                }
            }
        }

        let drift = if count > 0 { total / count as f32 } else { 0.0 };
        self.last_drift = drift;

        if drift > self.config.drift_threshold {
            self.encoder_lr_scale = (self.encoder_lr_scale * 0.5).max(0.01);
        } else {
            self.encoder_lr_scale = (self.encoder_lr_scale * 1.1).min(1.0);
        }
    }

    /// Batched policy + value step over all lanes. The graph input `"z"` is
    /// the stacked per-lane latents produced this step; `"action"` is the
    /// stacked action tokens already in `action_token_scratch`;
    /// `"value_target"` is the stacked per-lane rewards. Categorical plain-PG
    /// feeds `"action"` as a per-row advantage-weighted scaled one-hot;
    /// continuous PG feeds the raw action and a separate scalar advantage.
    ///
    /// Per-row categorical advantage weighting, without a graph change: the
    /// cross-entropy gradient w.r.t. logits is
    /// `target_weight · (pred − one_hot)`. So if we feed the action
    /// input as `advantage_i · one_hot_i` (signed, clamped) rather than
    /// just `one_hot_i`, each lane's gradient magnitude and sign come
    /// from its own advantage. Lanes with positive advantage push the
    /// policy toward the taken action, lanes with negative advantage
    /// push it away, and lanes with ~zero advantage contribute ~nothing.
    /// The shared LR is a fixed `lr_policy · batch_lr_scale`.
    ///
    /// This is the zero-graph-change analog of a proper per-row loss-weight
    /// input for categorical cross-entropy. Continuous squared error cannot
    /// use that trick, so its graph explicitly weights each row loss.
    fn policy_step_batched(&mut self) {
        let n_step = self.config.n_step.max(1);
        // When the user opts into n-step lookahead, defer to the
        // dedicated path that trains on old (z, action, value) with
        // an n-step Monte-Carlo return as the advantage.
        if n_step >= 2 {
            // Rollout-gated dispatch. With `policy_update_interval = 1`
            // (default) this fires every env-step on the single ripe
            // transition at `buf_len - n_step - bootstrap_headroom` —
            // identical to the old per-step behavior. With
            // `policy_update_interval > 1` we accumulate transitions
            // and, when the tick hits the interval, fire `interval`
            // sequential gradient steps on the last `interval` ripe
            // transitions (ripe_back_offset = 0, 1, …, interval-1).
            // Each step's advantage is still computed from the stored
            // V's, so `interval = n_step + 1` keeps the whole rollout
            // on-policy w.r.t. the collector.
            // When `rollout_length > 1`, it supersedes
            // `policy_update_interval` — the update cadence and the
            // per-update batch size both come from rollout_length.
            let rollout_length = self.config.rollout_length.max(1);
            let interval = if rollout_length > 1 {
                rollout_length
            } else {
                self.config.policy_update_interval.max(1)
            };
            self.policy_update_ticks += 1;
            if self.policy_update_ticks < interval {
                return;
            }
            self.policy_update_ticks = 0;
            if rollout_length > 1 {
                // Big-batch rollout path: one policy session.step() per
                // epoch over a flat `lanes × rollout_length`-row batch.
                // This is what actually lowers the per-update gradient
                // variance and makes the PPO clip fire on genuine trust-
                // region excursions rather than on noise.
                let n_epochs = if self.config.use_ppo || self.config.use_kl_ppo {
                    self.config.ppo_n_epochs.max(1)
                } else {
                    1
                };
                // KL-PPO frozen-π_old snapshot capture (when
                // kl_use_snapshot is on). Captures current policy's logits
                // on the rollout batch BEFORE the K training epochs, holds
                // them across all K iterations. KL grows monotonically as
                // policy drifts from the captured point. Without this,
                // KL stays near 0 and the trust region is inactive.
                if self.config.use_kl_ppo
                    && self.config.kl_use_snapshot
                    && self.old_logits_input_present
                {
                    self.capture_kl_snapshot_logits(n_step, rollout_length);
                }
                for _ in 0..n_epochs {
                    self.policy_step_rollout_batch(n_step, rollout_length);
                }
                return;
            }
            let n_epochs = if self.config.use_ppo {
                self.config.ppo_n_epochs.max(1)
            } else {
                1
            };
            // Epoch loop: replay the same `interval`-step rollout
            // `n_epochs` times. With PPO, epoch 1's ratio ≈ 1 (π_new
            // ≈ π_old since collection), so it runs as plain PG.
            // Epochs 2+ see the policy drifted from π_old (because
            // epoch 1 updated weights), so ratios diverge from 1 and
            // the PPO clip starts actually capping updates that would
            // push past the trust region. This is the standard PPO
            // training schedule (e.g. SB3 defaults to n_epochs=10).
            //
            // Non-PPO path keeps n_epochs=1 — replaying the same
            // rollout under plain PG would just multiply the update
            // magnitude (every epoch is unclipped, so each repeat is
            // basically an LR boost).
            //
            // Within each epoch, the offsets are traversed in a
            // RANDOMIZED order (Fisher-Yates shuffle). This is the
            // PPO minibatch-shuffling trick: without it, the
            // consecutive gradient steps in a single epoch all come
            // from adjacent time positions in the rollout, which
            // keeps the optimizer moving in a temporally-correlated
            // direction and amplifies the commit/uncommit feedback
            // loop. Shuffling breaks that correlation — each
            // consecutive gradient step sees a different time slice,
            // so the mean update direction is closer to the true
            // rollout-averaged gradient.
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            // Seed a local RNG from step_count so shuffles are
            // deterministic per-agent across runs (reproducibility),
            // and vary between updates (so epochs aren't identical).
            let mut shuffle_rng = rand::rngs::StdRng::seed_from_u64(
                0x9E37_79B9_7F4A_7C15u64 ^ (self.step_count as u64),
            );
            let mut offsets: Vec<usize> = (0..interval).collect();
            for _ in 0..n_epochs {
                offsets.shuffle(&mut shuffle_rng);
                for &k in offsets.iter() {
                    self.policy_step_n_step_at(n_step, k);
                }
            }
            return;
        }
        // Build stacked value targets: use the *base* reward (no L1
        // goal-alignment bonus). The bonus is retained in `last_reward`
        // for the advantage computation below — so the value baseline
        // doesn't learn to cancel the option-discriminating signal.
        // When L1 is inactive the two are identical. Clamp the target
        // to a sane range so a runaway reward stream (e.g. a very
        // negative homeo deviation) can't drive the value MSE into
        // gradient magnitudes that explode the value-head weights.
        let vtc = self.config.value_target_clamp;
        for (i, lane) in self.lanes.iter().enumerate() {
            let v = lane.last_base_reward;
            self.value_target_scratch[i] = if v.is_finite() {
                v.clamp(-vtc, vtc)
            } else {
                0.0
            };
        }

        // Build per-row advantage-weighted action targets.
        //
        // Raw clamped advantages (not normalized). Normalization was
        // tested and made L0 worse at N=64: compressing all advantages
        // to similar magnitudes removes the signal diversity that lets
        // the cross-entropy gradient find a good local optimum. The
        // 462-landing best (commit 18d422f) used raw clamped advantages;
        // re-introducing normalization later caused L0-42 to degrade
        // from −131 to −849.
        //
        // Phase G v3: `entropy_floor` no longer suppresses updates
        // below the floor (which locked deterministic collapse once
        // entropy hit zero — with batch_size=1 the policy then never
        // recovered). Instead, `entropy_deficit = (floor - entropy) /
        // floor`, clamped to `[0, 1]`, drives two self-correcting
        // changes to the update target for that lane:
        //
        //   1. Label-smoothing `eps` is amplified toward `1.0` —
        //      the softmax target shifts from the taken action's
        //      one-hot toward the uniform distribution. The
        //      cross-entropy gradient then pulls the dominant logit
        //      down and the rest up, restoring entropy.
        //   2. When the real reward advantage is near zero (value
        //      tracks reward well, so the policy gradient would
        //      otherwise carry no signal), `entropy_deficit` is used
        //      as the effective advantage magnitude — a positive
        //      recovery signal applied to the now-uniform target.
        //
        // Both reduce to the old behaviour at entropy ≥ floor
        // (deficit = 0 → eps = eps_base, effective_adv = advantage).
        self.policy_action_scratch.fill(0.0);
        let use_ppo = self.config.use_ppo;
        let continuous = !self.policy_action_mask_input_present;
        if use_ppo || continuous {
            // PPO surrogate inputs: rows for skipped lanes stay at
            // advantage = 0, which zeroes both surrogate terms — the
            // masking equivalent of the CE path's all-zero label rows.
            // Continuous PG uses the same scalar scratch as its explicit
            // per-row loss weight.
            self.ppo_advantage_scratch.fill(0.0);
        }
        if use_ppo {
            self.ppo_old_prob_scratch.fill(1.0);
        }
        let mut any_active = false;
        let eps_base = self.config.label_smoothing;
        let floor = self.config.entropy_floor;
        for (i, lane) in self.lanes.iter().enumerate() {
            // The policy input below is the acted-from latent (the
            // z_target_scratch staged this step). Boundary lanes have
            // no in-episode acted-from state (their row is zeros) —
            // skip them rather than train π(a | null state).
            if lane.buffer.last().is_none_or(|t| t.env_boundary) {
                continue;
            }
            let adv_clamp = self.config.advantage_clamp.max(0.0);
            let advantage = (lane.last_reward - lane.last_value).clamp(-adv_clamp, adv_clamp);
            let entropy_deficit = if floor > 0.0 {
                ((floor - lane.last_entropy) / floor).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Nothing to do only when there's no reward signal AND
            // entropy is comfortably above the floor.
            if advantage.abs() < 1e-8 && entropy_deficit < 1e-6 {
                continue;
            }
            any_active = true;

            if use_ppo {
                // The PPO graph's policy loss is the clipped surrogate
                // built from the `advantage` and `old_prob_taken`
                // inputs; the `action` input is a plain one-hot
                // selecting π(a|s) (no label smoothing / advantage
                // weighting — those are CE-target mechanics). Before
                // this, the single-step PPO path never staged
                // `advantage`, so act()'s zeroed buffer silently
                // nulled the policy gradient and the policy froze.
                let act_src =
                    &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM]
                    .copy_from_slice(act_src);
                self.ppo_advantage_scratch[i] = advantage;
                self.ppo_old_prob_scratch[i] = lane.last_prob_taken.max(1e-6);
                continue;
            }

            if continuous {
                let act_src =
                    &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                let act_dst =
                    &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                act_dst.copy_from_slice(act_src);
                self.ppo_advantage_scratch[i] = advantage;
                continue;
            }

            // Amplify label-smoothing toward uniform when entropy is low.
            let eps = (eps_base + (1.0 - eps_base) * entropy_deficit).min(1.0);
            // Synthesize a recovery advantage only when the real
            // reward advantage has gone silent — otherwise use the
            // signal advantage unchanged. The soft-tanh-clamp on the
            // policy logits (see `policy.rs`) prevents the NaN-from-
            // unbounded-logits failure mode on its own, so we don't
            // need to amplify recovery to dominate the signal.
            let effective_adv = if advantage.abs() < 1e-3 && entropy_deficit > 0.0 {
                entropy_deficit
            } else {
                advantage
            };
            let act_src = &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let act_dst =
                &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let mask = &self.action_masks[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let valid_count = mask.iter().filter(|&&m| m >= 0.5).count().max(1) as f32;
            for ((dst, &src), &valid) in act_dst.iter_mut().zip(act_src.iter()).zip(mask) {
                *dst = if valid >= 0.5 {
                    effective_adv * ((1.0 - eps) * src + eps / valid_count)
                } else {
                    0.0
                };
            }
        }
        if !any_active {
            return;
        }

        // Global advantage-norm clip — see `policy_step_n_step` for
        // rationale. Under PPO the advantage lives in its own input,
        // so clip that; otherwise it's folded into the action targets.
        let clip = self.config.policy_adv_global_clip;
        if clip > 0.0 {
            let buf: &mut [f32] = if use_ppo || continuous {
                &mut self.ppo_advantage_scratch
            } else {
                &mut self.policy_action_scratch
            };
            let sum_sq: f32 = buf.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > clip {
                let scale = clip / norm;
                for v in buf.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Adaptive learning-rate scaling — see `policy_step_n_step`.
        let target = self.config.policy_lr_adaptive_target;
        let lr_scale = if target > 0.0 && self.policy_loss_ema > target {
            target / self.policy_loss_ema
        } else {
            1.0
        };

        // Feed the latent of the state each action was taken FROM —
        // that's the previous step's latent, which observe() staged
        // into `z_target_scratch` for the WM dispatch and which is
        // untouched since. (`z_stack` holds the POST-action latents;
        // training π(a_t | z_{t+1}) — the pre-2026-06-10 behaviour —
        // pairs every action with the state it produced instead of
        // the state it was chosen in.) The stored `last_value`
        // baseline was computed at act() time on this same acted-from
        // latent, so the advantage is consistent with the input.
        // Plus, when L1 is active, the per-lane one-hot
        // `option_onehot` so the policy graph's option bias head
        // receives the current option identity for training.
        // The policy graph's `z` input is policy_batch-row; pad the
        // lanes-row acted-from latents through `policy_z_scratch`
        // (identical at rollout_length = 1, required at > 1).
        let n_rows = self.lanes.len() * self.latent_dim;
        self.policy_z_scratch[..n_rows].copy_from_slice(&self.z_target_scratch[..n_rows]);
        for v in self.policy_z_scratch[n_rows..].iter_mut() {
            *v = 0.0;
        }
        self.policy_session.set_input("z", &self.policy_z_scratch);
        if self.option_session.is_some() {
            let num_options = self.config.num_options;
            self.option_onehot_scratch.fill(0.0);
            for (i, lane) in self.lanes.iter().enumerate() {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                row[lane.current_option as usize] = 1.0;
            }
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        if use_ppo || continuous {
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
        }
        if use_ppo {
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        if self.policy_action_mask_input_present {
            self.populate_policy_action_mask_scratch();
            self.policy_session
                .set_input("action_mask", &self.policy_action_mask_scratch);
        }
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        self.feed_entropy_beta_input();
        self.feed_kl_beta_input();
        apply_lr(
            &mut self.policy_session,
            self.config.lr_policy * self.batch_lr_scale * lr_scale,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        // Watchdog: reset on non-finite OR absolute magnitude above
        // `policy_loss_watchdog_threshold`. The latter catches the
        // "finite but runaway" regime observed on LunarLander after
        // a brief performance peak. Setting the threshold very high
        // effectively disables the magnitude branch (NaN branch
        // remains active).
        let wd = self.config.policy_loss_watchdog_threshold;
        if !loss.is_finite() {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} non-finite at step {} (NaN/Inf), re-initialized policy params (wd={} ignored)",
                loss,
                self.step_count,
                wd,
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
            let rate = self.config.policy_lr_adaptive_ema.clamp(0.001, 1.0);
            self.policy_loss_ema = (1.0 - rate) * self.policy_loss_ema + rate * loss.abs();
        }
    }

    /// N-step advantage variant of `policy_step_batched`.
    ///
    /// Trains the policy on the transition at buffer offset
    /// `len - n_step` — the step at which the lane took an action
    /// whose `n_step`-lookahead reward window is now fully observed.
    /// The advantage is `R_t − V(s_t)` where `R_t` is the γ-discounted
    /// and effective-horizon-normalized sum over the next `n_step`
    /// rewards, truncated at any `env_boundary` flag (so rewards from
    /// after an episode reset don't bleed into the prior option's
    /// credit).
    ///
    /// Value head still trains on the single-step base reward at the
    /// old state, preserving the value-head stability that the
    /// previous discounted-return attempt (commit 8f291e5) lost.
    /// Train the policy on the ripe transition at
    /// `ripe_idx = buf_len - n_step - headroom - ripe_back_offset`.
    /// `ripe_back_offset = 0` → most-recent ripe (the per-step default).
    /// Larger offsets walk back through the rollout for
    /// `policy_update_interval > 1`.
    fn policy_step_n_step_at(&mut self, n_step: usize, ripe_back_offset: usize) {
        let n = self.lanes.len();
        let gamma = self.config.gamma;
        let eps_base = self.config.label_smoothing;
        let floor = self.config.entropy_floor;

        // The n-step return `Σ γ^k r_{t+k}` is compared against a
        // value-head baseline that tracks the *single-step* reward —
        // the previous implementation normalized ret by
        // `Σ γ^k ≈ n_step` to keep the two scales comparable. That
        // works for dense reward streams but is catastrophic on
        // sparse ones: a single ±1 event in a 16-step window divides
        // down to ±0.06, which trips the `advantage.abs() < 1e-3`
        // skip check in the label-smoothing block below and produces
        // a near-zero policy gradient. Empirically (Pong 400k
        // env-steps, all ±1 events averaged away to silence).
        //
        // Correct handling: treat ret as the Monte-Carlo return
        // directly. `ripe.value` is an EMA of single-step reward at
        // that state — a correctly-signed (if slightly biased)
        // baseline. Advantage = ret - value in the reward-scale
        // units. advantage_clamp still bounds the per-step update
        // magnitude.

        self.policy_action_scratch.fill(0.0);
        self.value_target_scratch.fill(0.0);

        // Build old-state z stack and per-row targets. Lanes with a
        // too-short buffer contribute zeros (no gradient).
        let ld = self.latent_dim;
        let mut old_z_stack = vec![0.0f32; n * ld];
        let num_options = self.config.num_options;
        let has_options = self.option_session.is_some();
        if has_options {
            self.option_onehot_scratch.fill(0.0);
        }

        let mut any_active = false;

        // Under `value_bootstrap` OR `gae_lambda > 0`, shift ripe
        // one step earlier so we reserve the last buffer slot as
        // the bootstrap state for `V(s_{ripe+n_step})`. Without it,
        // ripe_idx + n_step would be out of range (== buf_len) —
        // both the value-bootstrap and GAE paths read that slot.
        let use_gae = self.config.gae_lambda > 0.0;
        let needs_bootstrap_slot = self.config.value_bootstrap || use_gae;
        let bootstrap_headroom = needs_bootstrap_slot as usize;
        let value_target_bootstrap = self.config.value_bootstrap || use_gae;

        // First pass: compute raw advantage + value target for each
        // active lane. We collect them all before labeling so that
        // `advantage_normalize` can zero-mean / unit-std the batch
        // before it feeds into the CE labels.
        let mut raw_advantages = vec![0.0f32; n];
        let mut value_targets = vec![0.0f32; n];
        let mut lane_active = vec![false; n];
        let adv_clamp = self.config.advantage_clamp.max(0.0);

        // Optional fresh-V forward pass: recompute V on each lane's
        // ripe.latent using the CURRENT value head (rather than using
        // ripe.value, which was computed n_step env-steps ago under
        // a stale encoder + stale value head). This is what standard
        // A2C does — computes V fresh per rollout sample. Closes the
        // "stored vs current V" mismatch from the encoder-shift
        // hypothesis.
        let mut fresh_v: Option<Vec<f32>> = None;
        if self.config.recompute_base_v && !use_gae {
            // Build the ripe-latent stack for active lanes, padding
            // inactives with zeros (their fresh_v is unused).
            let mut z_pre = vec![0.0f32; n * ld];
            for (i, lane) in self.lanes.iter().enumerate() {
                let buf_len = lane.buffer.len();
                if buf_len < n_step + bootstrap_headroom + ripe_back_offset {
                    continue;
                }
                let ripe_idx = buf_len - n_step - bootstrap_headroom - ripe_back_offset;
                let ripe = lane.buffer.get(ripe_idx);
                // V must be recomputed on the state the action was
                // taken FROM (previous record); boundary rows are
                // skipped by the main loop, so the saturating read
                // below never reaches the trained batch.
                if ripe_idx == 0 || ripe.env_boundary {
                    continue;
                }
                let acted_from = lane.buffer.get(ripe_idx - 1);
                z_pre[i * ld..(i + 1) * ld].copy_from_slice(&acted_from.latent);
            }
            // Forward-only pass at LR=0. Need to seed all session
            // inputs with valid-shaped zeros; the gradient won't flow.
            self.policy_session.set_input("z", &z_pre);
            self.policy_action_scratch.fill(0.0);
            self.policy_session
                .set_input("action", &self.policy_action_scratch);
            if self.policy_action_mask_input_present {
                self.populate_policy_action_mask_scratch();
                self.policy_session
                    .set_input("action_mask", &self.policy_action_mask_scratch);
            }
            self.value_target_scratch.fill(0.0);
            self.policy_session
                .set_input("value_target", &self.value_target_scratch);
            if self.config.use_ppo || !self.policy_action_mask_input_present {
                self.ppo_advantage_scratch.fill(0.0);
                self.policy_session
                    .set_input("advantage", &self.ppo_advantage_scratch);
            }
            if self.config.use_ppo {
                for v in self.ppo_old_prob_scratch.iter_mut() {
                    *v = 1.0;
                }
                self.policy_session
                    .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
            }
            if has_options {
                self.policy_session
                    .set_input("option_onehot", &self.option_onehot_scratch);
            }
            self.feed_entropy_beta_input();
            self.feed_kl_beta_input();
            // Forward-only V read: skip the optimizer pass (with Adam,
            // lr = 0 still pollutes the moment buffers — see act()).
            self.policy_session.clear_optimizer();
            self.policy_session.step();
            self.policy_session.wait();
            // value output is at index 2 in the combined graph.
            let mut v_out = vec![0.0f32; n];
            self.policy_session.read_output_by_index(2, &mut v_out);
            // Sanitize.
            for v in v_out.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            fresh_v = Some(v_out);
        }
        for (i, lane) in self.lanes.iter().enumerate() {
            let buf_len = lane.buffer.len();
            if buf_len < n_step + bootstrap_headroom + ripe_back_offset {
                continue;
            }
            let ripe_idx = buf_len - n_step - bootstrap_headroom - ripe_back_offset;
            let ripe = lane.buffer.get(ripe_idx);
            let (ret, _gk_end, _terminated) = compute_td_n_step_return(
                &lane.buffer,
                ripe_idx,
                n_step,
                gamma,
                value_target_bootstrap,
                self.config.bootstrap_value_clamp,
            );
            let v_for_baseline = match fresh_v.as_ref() {
                Some(fv) => fv[i],
                None => ripe.value,
            };
            let adv_raw = if self.config.use_grpo {
                if self.config.use_grpo_episode {
                    // Per-episode GRPO: use this transition's own
                    // episode return (set retroactively at episode end).
                    // If the transition's episode hasn't ended yet,
                    // skip the lane (set inactive) — credit-assigning
                    // mid-episode transitions before we know the
                    // outcome would just inject noise.
                    if !ripe.episode_complete {
                        continue;
                    }
                    ripe.episode_return
                } else {
                    // Per-step GRPO: n-step return without V bootstrap
                    // (V is unused with value_loss_coef=0, so its
                    // bootstrap contribution is random noise).
                    let (ret_nob, _, _) = compute_td_n_step_return(
                        &lane.buffer,
                        ripe_idx,
                        n_step,
                        gamma,
                        false,
                        self.config.bootstrap_value_clamp,
                    );
                    ret_nob
                }
            } else if use_gae {
                compute_gae_advantage(
                    &lane.buffer,
                    ripe_idx,
                    n_step,
                    gamma,
                    self.config.gae_lambda,
                    self.config.bootstrap_value_clamp,
                )
            } else {
                ret - v_for_baseline
            };
            raw_advantages[i] = adv_raw;
            let vtc = self.config.value_target_clamp;
            value_targets[i] = if value_target_bootstrap {
                ret.clamp(-vtc, vtc)
            } else if ripe.reward.is_finite() {
                ripe.reward.clamp(-vtc, vtc)
            } else {
                0.0
            };
            lane_active[i] = true;
        }

        // Optional zero-mean / unit-std normalization. Plain actor-critic
        // uses the whole active batch. GRPO normalizes independently within
        // each env/task id so forks of one game are never compared with the
        // return scale or difficulty of another game.
        if self.config.advantage_normalize {
            let group_ids = self.config.use_grpo.then(|| {
                self.lanes
                    .iter()
                    .map(|lane| lane.adapter.id())
                    .collect::<Vec<_>>()
            });
            let divide_by_std = !(self.config.use_grpo && self.config.use_grpo_episode);
            normalize_active_advantages(
                &mut raw_advantages,
                &lane_active,
                group_ids.as_deref(),
                divide_by_std,
            );
        }

        let use_ppo = self.config.use_ppo;
        let uses_action_mask = self.policy_action_mask_input_present;
        let continuous = !uses_action_mask;
        if use_ppo || continuous {
            self.ppo_advantage_scratch.fill(0.0);
        }
        if use_ppo {
            self.ppo_old_prob_scratch.fill(1.0);
        }
        if uses_action_mask {
            self.policy_action_mask_scratch.fill(1.0);
        }
        // KL-PPO: stage each ripe transition's stored collection-time
        // logits as π_old. Before this, only the rollout-batch path
        // staged `old_logits`; here the device buffer held the zeros
        // act() wrote, so the "trust region" was KL against the
        // uniform distribution — an entropy regularizer, not a
        // constraint on drift from π_old. Inactive rows stay zero
        // (uniform π_old), matching the pre-fix behavior for rows
        // that carry no policy gradient anyway.
        if self.old_logits_input_present {
            for v in self.kl_old_logits_scratch.iter_mut() {
                *v = 0.0;
            }
        }
        if self.reward_pred_input_present {
            self.reward_target_scratch.fill(0.0);
        }

        for (i, lane) in self.lanes.iter().enumerate() {
            if !lane_active[i] {
                continue;
            }
            let buf_len = lane.buffer.len();
            let ripe_idx = buf_len - n_step - bootstrap_headroom - ripe_back_offset;
            let ripe = lane.buffer.get(ripe_idx);
            // Pair the action with the state it was taken FROM — the
            // previous record. Boundary rows have no in-episode
            // predecessor; skip them. (See policy_step_rollout_batch
            // for the full pairing rationale.)
            if ripe_idx == 0 || ripe.env_boundary {
                continue;
            }
            let acted_from = lane.buffer.get(ripe_idx - 1);
            let advantage = if self.step_count < self.config.policy_warmup_steps {
                0.0
            } else {
                raw_advantages[i].clamp(-adv_clamp, adv_clamp)
            };
            self.value_target_scratch[i] = value_targets[i];
            if self.reward_pred_input_present && ripe.reward.is_finite() {
                // Aux reward-prediction target — previously only the
                // rollout-batch path staged this, leaving the head to
                // train toward the stale device contents here.
                self.reward_target_scratch[i] = ripe.reward;
            }

            let entropy_deficit = if floor > 0.0 {
                ((floor - lane.last_entropy) / floor).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if advantage.abs() < 1e-8 && entropy_deficit < 1e-6 {
                continue;
            }
            any_active = true;

            old_z_stack[i * ld..(i + 1) * ld].copy_from_slice(&acted_from.latent);

            // E2E mode: also fill obs + task scratch from the acted-
            // from record.
            if self.config.end_to_end_encoder {
                let obs_dst =
                    &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
                // Pad short observations with zeros, copy what we have.
                let copy_n = acted_from.observation.len().min(OBS_TOKEN_DIM);
                obs_dst[..copy_n].copy_from_slice(&acted_from.observation[..copy_n]);
                for v in obs_dst[copy_n..].iter_mut() {
                    *v = 0.0;
                }
                let task_dst = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
                if let Some(emb) = self.task_embeddings.get(&ripe.env_id) {
                    task_dst.copy_from_slice(emb);
                } else {
                    task_dst.fill(0.0);
                }
            }

            if uses_action_mask {
                restore_action_mask(
                    &mut self.policy_action_mask_scratch
                        [i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM],
                    &ripe.action_mask,
                );
            }
            if use_ppo {
                // PPO feeds: plain one-hot `action` (sum=1), scalar
                // `advantage`, scalar `old_prob_taken`. The clip and
                // the advantage-weighting live inside the graph.
                let act_dst =
                    &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                for (dst, &src) in act_dst.iter_mut().zip(ripe.action.iter()) {
                    *dst = src;
                }
                self.ppo_advantage_scratch[i] = advantage;
                self.ppo_old_prob_scratch[i] = ripe.prob_taken.max(1e-8);
            } else if continuous {
                let act_dst =
                    &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                act_dst.copy_from_slice(&ripe.action);
                self.ppo_advantage_scratch[i] = advantage;
            } else {
                let eps = (eps_base + (1.0 - eps_base) * entropy_deficit).min(1.0);
                let effective_adv = if advantage.abs() < 1e-3 && entropy_deficit > 0.0 {
                    entropy_deficit
                } else {
                    advantage
                };
                let act_src = &ripe.action;
                let act_dst =
                    &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                let mask = &ripe.action_mask;
                let valid_count = mask.iter().filter(|&&m| m >= 0.5).count().max(1) as f32;
                for ((dst, &src), &valid) in act_dst.iter_mut().zip(act_src.iter()).zip(mask) {
                    *dst = if valid >= 0.5 {
                        effective_adv * ((1.0 - eps) * src + eps / valid_count)
                    } else {
                        0.0
                    };
                }
            }

            if self.old_logits_input_present && !ripe.logits_at_action.is_empty() {
                let dst =
                    &mut self.kl_old_logits_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                let copy_n = ripe.logits_at_action.len().min(MAX_ACTION_DIM);
                dst[..copy_n].copy_from_slice(&ripe.logits_at_action[..copy_n]);
            }

            if has_options {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                let oi = (ripe.option_idx as usize).min(num_options.saturating_sub(1));
                row[oi] = 1.0;
            }
        }

        if !any_active {
            return;
        }

        // Global advantage-norm clip (policy-gradient-analogue of
        // grad-norm clipping). The per-lane `advantage_clamp` is an
        // L∞ bound; this is the L2 bound across the batch. Since
        // each lane contributes linearly to the policy gradient,
        // bounding the batch L2 norm bounds the update magnitude.
        let clip = self.config.policy_adv_global_clip;
        if clip > 0.0 {
            let policy_signal: &[f32] = if continuous {
                &self.ppo_advantage_scratch
            } else {
                &self.policy_action_scratch
            };
            let sum_sq: f32 = policy_signal.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > clip {
                let scale = clip / norm;
                let policy_signal: &mut [f32] = if continuous {
                    &mut self.ppo_advantage_scratch
                } else {
                    &mut self.policy_action_scratch
                };
                for v in policy_signal {
                    *v *= scale;
                }
            }
        }

        // Adaptive learning-rate scaling based on EMA(|pi_loss|).
        // When the recent policy updates have been unusually loud
        // (EMA loss magnitude far above the target), scale LR down
        // so the next update is smaller. Damps the commit-recover
        // oscillation we observed on CartPole.
        let target = self.config.policy_lr_adaptive_target;
        let lr_scale = if target > 0.0 && self.policy_loss_ema > target {
            target / self.policy_loss_ema
        } else {
            1.0
        };

        if self.config.end_to_end_encoder {
            self.policy_session
                .set_input("obs", &self.obs_token_scratch);
            self.policy_session.set_input("task", &self.task_scratch);
        } else {
            self.policy_session.set_input("z", &old_z_stack);
        }
        if has_options {
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        if uses_action_mask {
            self.policy_session
                .set_input("action_mask", &self.policy_action_mask_scratch);
        }
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        if use_ppo || continuous {
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
        }
        if use_ppo {
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        if self.old_logits_input_present {
            self.policy_session
                .set_input("old_logits", &self.kl_old_logits_scratch);
        }
        if self.reward_pred_input_present {
            self.policy_session
                .set_input("reward_target", &self.reward_target_scratch);
        }
        self.feed_entropy_beta_input();
        self.feed_kl_beta_input();
        apply_lr(
            &mut self.policy_session,
            self.config.lr_policy * self.batch_lr_scale * lr_scale,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        let wd = self.config.policy_loss_watchdog_threshold;
        if !loss.is_finite() {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} non-finite at step {} (n-step, NaN/Inf), re-initialized policy params (wd={} ignored)",
                loss,
                self.step_count,
                wd,
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
            let rate = self.config.policy_lr_adaptive_ema.clamp(0.001, 1.0);
            self.policy_loss_ema = (1.0 - rate) * self.policy_loss_ema + rate * loss.abs();
        }
    }

    /// Big-batch rollout policy update: flatten the `rollout_length`-
    /// step rollout × `lanes` = `policy_batch` samples into a single
    /// `session.step()`, rather than doing `rollout_length` sequential
    /// `lanes`-sized steps. Lower per-update gradient variance is the
    /// point — it's what lets PPO's clip engage on meaningful trust-
    /// region excursions rather than on per-lane noise.
    ///
    /// Row layout in the flat batch: row `r = offset × lanes + lane_i`
    /// corresponds to lane `lane_i`'s ripe transition at
    /// `ripe_back_offset = offset`. Older offsets (larger `offset`)
    /// appear first, matching the temporal order of collection.
    /// KL-PPO snapshot capture: forward-only pass on the rollout
    /// batch's obs/task using current weights, write resulting logits
    /// into kl_old_logits_scratch as the "frozen π_old" for the
    /// upcoming K-epoch cycle.
    ///
    /// Called ONCE before the K-loop when `kl_use_snapshot=true`. The K
    /// subsequent training calls leave kl_old_logits_scratch untouched
    /// (the per-transition refill is gated off in snapshot mode), so KL
    /// grows monotonically as the policy drifts from this captured point.
    ///
    /// Implementation: run policy_step_rollout_batch with the snapshot
    /// flag set; that suppresses both the per-transition refill of
    /// kl_old_logits_scratch AND the backward gradient effect (LR=0).
    /// All other bookkeeping (read_loss, EMA, watchdog) runs normally
    /// to keep the call's side effects clean. After return, read
    /// output index 1 (logits) into kl_old_logits_scratch.
    fn capture_kl_snapshot_logits(&mut self, n_step: usize, rollout_length: usize) {
        let lanes = self.lanes.len();
        let policy_batch = lanes * rollout_length;
        self.kl_snapshot_capture_pending = true;
        self.policy_step_rollout_batch(n_step, rollout_length);
        self.kl_snapshot_capture_pending = false;
        let n = policy_batch * MAX_ACTION_DIM;
        if self.kl_old_logits_scratch.len() < n {
            self.kl_old_logits_scratch.resize(n, 0.0);
        }
        self.policy_session
            .read_output_by_index(1, &mut self.kl_old_logits_scratch[..n]);
        // Defensive NaN/Inf guard. With stop_gradient(z) on the KL term
        // in the graph (see build_kl_policy_graph_e2e), the KL gradient
        // shouldn't push encoder weights toward overflow, but this is a
        // belt-and-suspenders sanity check.
        for v in self.kl_old_logits_scratch[..n].iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }

    fn policy_step_rollout_batch(&mut self, n_step: usize, rollout_length: usize) {
        let lanes = self.lanes.len();
        let policy_batch = lanes * rollout_length;
        let gamma = self.config.gamma;
        let ld = self.latent_dim;
        let num_options = self.config.num_options;
        let has_options = self.option_session.is_some();
        let eps_base = self.config.label_smoothing;
        let floor = self.config.entropy_floor;

        let use_gae = self.config.gae_lambda > 0.0;
        let needs_bootstrap_slot = self.config.value_bootstrap || use_gae;
        let bootstrap_headroom = needs_bootstrap_slot as usize;
        let value_target_bootstrap = self.config.value_bootstrap || use_gae;
        let use_ppo = self.config.use_ppo;
        let uses_action_mask = self.policy_action_mask_input_present;
        let continuous = !uses_action_mask;

        // Initialize scratch buffers for the whole batch.
        self.policy_action_scratch.fill(0.0);
        self.value_target_scratch.fill(0.0);
        self.policy_z_scratch.fill(0.0);
        if has_options {
            self.option_onehot_scratch.fill(0.0);
        }
        if use_ppo || continuous {
            self.ppo_advantage_scratch.fill(0.0);
        }
        if use_ppo {
            for v in self.ppo_old_prob_scratch.iter_mut() {
                *v = 1.0;
            }
        }
        if uses_action_mask {
            self.policy_action_mask_scratch.fill(1.0);
        }

        // First pass: compute raw advantages + value targets for
        // every (offset, lane) pair. Also record which rows are
        // "active" (buffer long enough). Inactive rows stay at 0
        // scratch and contribute no gradient through the CE/PPO loss.
        let mut raw_advantages = vec![0.0f32; policy_batch];
        let mut value_targets = vec![0.0f32; policy_batch];
        let mut row_active = vec![false; policy_batch];
        let mut ripe_action = vec![vec![0.0f32; MAX_ACTION_DIM]; policy_batch];
        let mut ripe_action_mask = vec![vec![1.0f32; MAX_ACTION_DIM]; policy_batch];
        let mut ripe_latent = vec![vec![0.0f32; ld]; policy_batch];
        let mut ripe_option = vec![0u32; policy_batch];
        let mut ripe_prob_taken = vec![1.0f32; policy_batch];

        let adv_clamp = self.config.advantage_clamp.max(0.0);
        // First pass A: collect ripe data (latent, action, option,
        // prob, return, raw advantage) using STORED V's. This is the
        // baseline path. The recompute pass (if enabled) overrides
        // adv_raw next.
        let mut ripe_returns = vec![0.0f32; policy_batch];
        let mut ripe_stored_v = vec![0.0f32; policy_batch];
        for offset in 0..rollout_length {
            for (lane_i, lane) in self.lanes.iter().enumerate() {
                let row = offset * lanes + lane_i;
                let buf_len = lane.buffer.len();
                if buf_len < n_step + bootstrap_headroom + offset {
                    continue;
                }
                let ripe_idx = buf_len - n_step - bootstrap_headroom - offset;
                let ripe = lane.buffer.get(ripe_idx);
                // Transitions record {latent: state AFTER the action,
                // action: the action that produced it}. The policy must
                // be trained on the state the action was taken FROM —
                // the previous record's latent. Boundary rows (first
                // record after a reset) have no in-episode predecessor:
                // skip them (the pre-reset latent belongs to another
                // episode). Pre-2026-06-10 this path paired tr.action
                // with tr.latent, training π(a_t | z_{t+1}) throughout.
                if ripe_idx == 0 || ripe.env_boundary {
                    continue;
                }
                let acted_from = lane.buffer.get(ripe_idx - 1);
                let (ret, _, _) = compute_td_n_step_return(
                    &lane.buffer,
                    ripe_idx,
                    n_step,
                    gamma,
                    value_target_bootstrap,
                    self.config.bootstrap_value_clamp,
                );
                let adv_raw = if self.config.use_grpo {
                    if self.config.use_grpo_episode {
                        // Per-episode GRPO. See use_grpo_episode docs.
                        if !ripe.episode_complete {
                            continue;
                        }
                        ripe.episode_return
                    } else {
                        // Per-step GRPO: no V bootstrap.
                        let (ret_nob, _, _) = compute_td_n_step_return(
                            &lane.buffer,
                            ripe_idx,
                            n_step,
                            gamma,
                            false,
                            self.config.bootstrap_value_clamp,
                        );
                        ret_nob
                    }
                } else if use_gae {
                    compute_gae_advantage(
                        &lane.buffer,
                        ripe_idx,
                        n_step,
                        gamma,
                        self.config.gae_lambda,
                        self.config.bootstrap_value_clamp,
                    )
                } else {
                    ret - ripe.value
                };
                raw_advantages[row] = adv_raw;
                ripe_returns[row] = ret;
                ripe_stored_v[row] = ripe.value;
                let vtc = self.config.value_target_clamp;
                value_targets[row] = if value_target_bootstrap {
                    ret.clamp(-vtc, vtc)
                } else if ripe.reward.is_finite() {
                    ripe.reward.clamp(-vtc, vtc)
                } else {
                    0.0
                };
                ripe_latent[row].copy_from_slice(&acted_from.latent);
                ripe_action[row].copy_from_slice(&ripe.action);
                restore_action_mask(&mut ripe_action_mask[row], &ripe.action_mask);
                ripe_option[row] = ripe.option_idx;
                ripe_prob_taken[row] = ripe.prob_taken.max(1e-8);
                // KL-PPO old_logits sourcing:
                // - When `kl_snapshot_capture_pending` is on: this is the
                //   snapshot capture pass; don't touch kl_old_logits_scratch
                //   (it'll be filled from output 1 after this call).
                // - When `kl_use_snapshot` is on (real training in
                //   snapshot mode): also don't touch — the snapshot
                //   captured by the previous capture pass should persist
                //   across all K training epochs.
                // - Otherwise (per-transition mode): refill from
                //   ripe.logits_at_action like before.
                if self.config.use_kl_ppo
                    && !self.kl_snapshot_capture_pending
                    && !self.config.kl_use_snapshot
                    && !ripe.logits_at_action.is_empty()
                {
                    let dst = &mut self.kl_old_logits_scratch
                        [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
                    let n = ripe.logits_at_action.len().min(MAX_ACTION_DIM);
                    dst[..n].copy_from_slice(&ripe.logits_at_action[..n]);
                    for v in dst[n..].iter_mut() {
                        *v = 0.0;
                    }
                }
                row_active[row] = true;
            }
        }

        // Optional fresh-V pass: forward through policy_session with
        // z=ripe_latent (across the whole rollout), read V output,
        // override raw_advantages = ret - V_fresh in the non-GAE
        // path. This closes the "stored V vs current V" mismatch
        // for the rollout batch — kindle's encoder shifts every WM
        // step, so by the time a transition becomes ripe its stored
        // V is computed under a different encoder than is now.
        if self.config.recompute_base_v && !use_gae {
            let e2e_pre = self.config.end_to_end_encoder;
            if e2e_pre {
                // E2E mode: stage obs+task per row so the policy
                // session forwards through its own encoder. The legacy
                // z input doesn't exist on this graph variant.
                self.obs_token_scratch.fill(0.0);
                self.task_scratch.fill(0.0);
                for row in 0..policy_batch {
                    if !row_active[row] {
                        continue;
                    }
                    let lane_i = row % lanes;
                    let buf_len = self.lanes[lane_i].buffer.len();
                    let offset = row / lanes;
                    if buf_len < n_step + bootstrap_headroom + offset {
                        continue;
                    }
                    let ripe_idx = buf_len - n_step - bootstrap_headroom - offset;
                    let ripe = self.lanes[lane_i].buffer.get(ripe_idx);
                    // State acted FROM = previous record's observation
                    // (rows with ripe_idx == 0 or env_boundary are
                    // inactive — collection skipped them — so the
                    // saturating read below is never trained on).
                    let from = self.lanes[lane_i].buffer.get(ripe_idx.saturating_sub(1));
                    let obs_dst =
                        &mut self.obs_token_scratch[row * OBS_TOKEN_DIM..(row + 1) * OBS_TOKEN_DIM];
                    let copy_n = from.observation.len().min(OBS_TOKEN_DIM);
                    obs_dst[..copy_n].copy_from_slice(&from.observation[..copy_n]);
                    let task_dst = &mut self.task_scratch[row * TASK_DIM..(row + 1) * TASK_DIM];
                    if let Some(emb) = self.task_embeddings.get(&ripe.env_id) {
                        task_dst.copy_from_slice(emb);
                    }
                }
                self.policy_session
                    .set_input("obs", &self.obs_token_scratch);
                self.policy_session.set_input("task", &self.task_scratch);
            } else {
                // Non-e2e: legacy z input.
                for row in 0..policy_batch {
                    if row_active[row] {
                        self.policy_z_scratch[row * ld..(row + 1) * ld]
                            .copy_from_slice(&ripe_latent[row]);
                    }
                }
                self.policy_session.set_input("z", &self.policy_z_scratch);
            }
            // Need all other inputs valid-shaped. Action / value
            // target / PPO inputs already filled with zeros above.
            self.policy_session
                .set_input("action", &self.policy_action_scratch);
            if uses_action_mask {
                self.populate_policy_action_mask_scratch();
                self.policy_session
                    .set_input("action_mask", &self.policy_action_mask_scratch);
            }
            self.policy_session
                .set_input("value_target", &self.value_target_scratch);
            if use_ppo || continuous {
                self.policy_session
                    .set_input("advantage", &self.ppo_advantage_scratch);
            }
            if use_ppo {
                self.policy_session
                    .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
            }
            if has_options {
                self.policy_session
                    .set_input("option_onehot", &self.option_onehot_scratch);
            }
            // Reward-pred aux input needs to be valid-shaped too.
            // (recon shares `obs` already set above.)
            if self.reward_pred_input_present {
                self.reward_target_scratch.fill(0.0);
                self.policy_session
                    .set_input("reward_target", &self.reward_target_scratch);
            }
            self.feed_entropy_beta_input();
            self.feed_kl_beta_input();
            // Forward-only V read: skip the optimizer pass (with Adam,
            // lr = 0 still pollutes the moment buffers — see act()).
            self.policy_session.clear_optimizer();
            self.policy_session.step();
            self.policy_session.wait();
            let mut v_fresh = vec![0.0f32; policy_batch];
            self.policy_session.read_output_by_index(2, &mut v_fresh);
            for v in v_fresh.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            // Override advantages with fresh-V baseline.
            for row in 0..policy_batch {
                if row_active[row] {
                    raw_advantages[row] = ripe_returns[row] - v_fresh[row];
                }
            }
            // Suppress unused-warnings on ripe_stored_v in this branch.
            let _ = &ripe_stored_v;
        } else {
            let _ = (&ripe_returns, &ripe_stored_v);
        }

        // Advantage normalization across the rollout batch. Plain
        // actor-critic uses all active rows. GRPO uses one group per
        // environment/task id, retaining all rollout-time samples for that
        // group while never comparing unrelated games.
        if self.config.advantage_normalize {
            let group_ids = self.config.use_grpo.then(|| {
                (0..policy_batch)
                    .map(|row| self.lanes[row % lanes].adapter.id())
                    .collect::<Vec<_>>()
            });
            let divide_by_std = !(self.config.use_grpo && self.config.use_grpo_episode);
            normalize_active_advantages(
                &mut raw_advantages,
                &row_active,
                group_ids.as_deref(),
                divide_by_std,
            );
        }

        // Second pass: fill scratch buffers.
        let mut any_active = false;
        let e2e = self.config.end_to_end_encoder;
        if e2e {
            self.obs_token_scratch.fill(0.0);
            self.task_scratch.fill(0.0);
        }
        let rp_aux = self.reward_pred_input_present;
        if rp_aux {
            self.reward_target_scratch.fill(0.0);
        }
        for row in 0..policy_batch {
            if !row_active[row] {
                continue;
            }
            any_active = true;
            let advantage = if self.step_count < self.config.policy_warmup_steps {
                0.0
            } else {
                raw_advantages[row].clamp(-adv_clamp, adv_clamp)
            };
            self.value_target_scratch[row] = value_targets[row];

            // z_scratch row (used in non-e2e path; redundant under e2e
            // but cheap to fill).
            self.policy_z_scratch[row * ld..(row + 1) * ld].copy_from_slice(&ripe_latent[row]);

            // E2E mode: fill obs + task per row from ripe transition.
            if e2e {
                let lane_i = row % lanes;
                let buf_len = self.lanes[lane_i].buffer.len();
                let offset = row / lanes;
                if buf_len >= n_step + bootstrap_headroom + offset {
                    let ripe_idx = buf_len - n_step - bootstrap_headroom - offset;
                    let ripe = self.lanes[lane_i].buffer.get(ripe_idx);
                    // State acted FROM = previous record's observation;
                    // inactive rows (ripe_idx == 0 / boundary) carry no
                    // gradient so the saturating read is harmless.
                    let from = self.lanes[lane_i].buffer.get(ripe_idx.saturating_sub(1));
                    let obs_dst =
                        &mut self.obs_token_scratch[row * OBS_TOKEN_DIM..(row + 1) * OBS_TOKEN_DIM];
                    let copy_n = from.observation.len().min(OBS_TOKEN_DIM);
                    obs_dst[..copy_n].copy_from_slice(&from.observation[..copy_n]);
                    let task_dst = &mut self.task_scratch[row * TASK_DIM..(row + 1) * TASK_DIM];
                    if let Some(emb) = self.task_embeddings.get(&ripe.env_id) {
                        task_dst.copy_from_slice(emb);
                    }

                    // WM aux: feed next_obs (the obs recorded at
                    // ripe_idx+1) and raw_action (one-hot of ripe.action).
                    // Skip when ripe_idx+1 is an env_boundary (the
                    // transition crossed an episode reset; predicting
                    // across that boundary would hand the WM loss a
                    // discontinuity it can't model). For boundary rows
                    // we leave next_obs at the current obs — predicted
                    // delta = 0 then has zero loss, contributing no
                    // gradient.
                    // Reward-prediction aux: feed the per-row single-
                    // step reward as the regression target. Clamped to
                    // ±200 to bound the loss against the lander +100/
                    // -100 terminal events without saturating.
                    if rp_aux {
                        let r = if ripe.reward.is_finite() {
                            ripe.reward.clamp(-200.0, 200.0)
                        } else {
                            0.0
                        };
                        self.reward_target_scratch[row] = r;
                    }
                }
            }

            if uses_action_mask {
                self.policy_action_mask_scratch[row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM]
                    .copy_from_slice(&ripe_action_mask[row]);
            }
            if use_ppo {
                let act_dst = &mut self.policy_action_scratch
                    [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
                act_dst.copy_from_slice(&ripe_action[row]);
                self.ppo_advantage_scratch[row] = advantage;
                self.ppo_old_prob_scratch[row] = ripe_prob_taken[row];
            } else if continuous {
                let act_dst = &mut self.policy_action_scratch
                    [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
                act_dst.copy_from_slice(&ripe_action[row]);
                self.ppo_advantage_scratch[row] = advantage;
            } else {
                // Plain advantage-weighted CE path. Use the corresponding
                // lane's current entropy as the entropy-deficit source;
                // this is an approximation (the ripe transition was
                // collected under possibly different entropy), but is
                // only used to trigger the label-smoothing fallback when
                // the live policy is near-deterministic.
                let lane_i = row % lanes;
                let live_entropy = self.lanes[lane_i].last_entropy;
                let entropy_deficit = if floor > 0.0 {
                    ((floor - live_entropy) / floor).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let eps = (eps_base + (1.0 - eps_base) * entropy_deficit).min(1.0);
                let effective_adv = if advantage.abs() < 1e-3 && entropy_deficit > 0.0 {
                    entropy_deficit
                } else {
                    advantage
                };
                let act_dst = &mut self.policy_action_scratch
                    [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
                let mask = &ripe_action_mask[row];
                let valid_count = mask.iter().filter(|&&m| m >= 0.5).count().max(1) as f32;
                for ((dst, &src), &valid) in
                    act_dst.iter_mut().zip(ripe_action[row].iter()).zip(mask)
                {
                    *dst = if valid >= 0.5 {
                        effective_adv * ((1.0 - eps) * src + eps / valid_count)
                    } else {
                        0.0
                    };
                }
            }

            if has_options {
                let r = &mut self.option_onehot_scratch[row * num_options..(row + 1) * num_options];
                let oi = (ripe_option[row] as usize).min(num_options.saturating_sub(1));
                r[oi] = 1.0;
            }
        }

        if !any_active {
            return;
        }

        // Global advantage-norm clip (L2 bound across the whole batch).
        let clip = self.config.policy_adv_global_clip;
        if clip > 0.0 {
            let policy_signal: &[f32] = if continuous {
                &self.ppo_advantage_scratch
            } else {
                &self.policy_action_scratch
            };
            let sum_sq: f32 = policy_signal.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > clip {
                let scale = clip / norm;
                let policy_signal: &mut [f32] = if continuous {
                    &mut self.ppo_advantage_scratch
                } else {
                    &mut self.policy_action_scratch
                };
                for v in policy_signal {
                    *v *= scale;
                }
            }
        }

        // Adaptive LR scaling (same rule as the per-step path).
        let target = self.config.policy_lr_adaptive_target;
        let lr_scale = if target > 0.0 && self.policy_loss_ema > target {
            target / self.policy_loss_ema
        } else {
            1.0
        };

        if e2e {
            self.policy_session
                .set_input("obs", &self.obs_token_scratch);
            self.policy_session.set_input("task", &self.task_scratch);
        } else {
            self.policy_session.set_input("z", &self.policy_z_scratch);
        }
        if has_options {
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        if uses_action_mask {
            self.policy_session
                .set_input("action_mask", &self.policy_action_mask_scratch);
        }
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        if use_ppo || continuous {
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
        }
        if use_ppo {
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        if self.old_logits_input_present {
            self.policy_session
                .set_input("old_logits", &self.kl_old_logits_scratch);
        }
        if self.reward_pred_input_present {
            self.policy_session
                .set_input("reward_target", &self.reward_target_scratch);
        }
        self.feed_entropy_beta_input();
        self.feed_kl_beta_input();
        // LR is not scaled down by rollout_length — the loss is already
        // mean-reduced over `policy_batch` rows, so gradient magnitude
        // per parameter is comparable to a `lanes`-batch update.
        // Snapshot capture pass: skip the optimizer so weights (and
        // Adam moments — see act()) don't move; we just want the
        // forward pass's logits to capture as the frozen π_old. All
        // other bookkeeping (read_loss, EMA, watchdog) still runs.
        if self.kl_snapshot_capture_pending {
            self.policy_session.clear_optimizer();
        } else {
            apply_lr(
                &mut self.policy_session,
                self.config.lr_policy * self.batch_lr_scale * lr_scale,
                self.config.use_adam,
                self.config.adam_eps,
            );
        }
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        // Read KL diagnostic from output index 3 (only present when
        // KL-PPO is on with kl_beta > 0 at construction). Skip during
        // snapshot capture pass — the KL by definition would be 0 there
        // (old_logits is whatever was previously in scratch, and we'll
        // overwrite it with this pass's output anyway).
        if self.old_logits_input_present && !self.kl_snapshot_capture_pending {
            let mut kl_buf = [0.0f32; 1];
            self.policy_session.read_output_by_index(3, &mut kl_buf);
            self.last_kl = kl_buf[0];
        }
        let wd = self.config.policy_loss_watchdog_threshold;
        if !loss.is_finite() {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} non-finite at step {} (rollout, NaN/Inf), re-initialized policy params (wd={} ignored)",
                loss,
                self.step_count,
                wd,
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
            let rate = self.config.policy_lr_adaptive_ema.clamp(0.001, 1.0);
            self.policy_loss_ema = (1.0 - rate) * self.policy_loss_ema + rate * loss.abs();
        }
    }

    /// Self-Imitation Learning step: run one supervised update on
    /// a sampled batch of (state, action) from the SIL buffer of past
    /// successful episodes. No-op when use_sil is off or the buffer
    /// has fewer than policy_batch samples.
    ///
    /// The policy graph is reused (no separate session), with the
    /// staging matched to the graph variant:
    ///   * e2e: obs + task inputs, `action = weight · one_hot(taken)`
    ///     makes the CE term `-weight · log π(taken)`.
    ///   * non-e2e plain-PG: `z` input from the stored acted-from
    ///     latent, same weighted-CE labels.
    ///   * non-e2e PPO: `z` input, plain one-hot action, advantage =
    ///     weight, old_prob = 1. The ratio r = π(a)/1 ≤ 1 stays below
    ///     the 1+ε clip for positive advantage, so the surrogate
    ///     reduces to weight-scaled vanilla PG toward the stored
    ///     action — exactly the BC pull we want.
    ///
    /// KL-PPO is skipped (its graph wants old_logits we don't track
    /// for SIL samples). Pre-2026-06-10 this function ALSO returned
    /// early for `use_ppo` and for every non-e2e config — meaning SIL
    /// and BC-from-planner pushed samples that were never trained on
    /// in any ARC run (all of which are non-e2e + PPO).
    fn maybe_run_sil_update(&mut self) {
        if !self.config.use_sil || !self.policy_action_mask_input_present {
            return;
        }
        self.sil_updates_attempted += 1;
        if self.config.use_kl_ppo {
            return;
        }
        let e2e = self.config.end_to_end_encoder;
        let use_ppo = self.config.use_ppo;
        let policy_batch = self.config.batch_size * self.config.rollout_length;
        if self.sil_buffer.len() < policy_batch {
            return;
        }

        // Sample reproducibly and round-robin across tasks active in the
        // current lanes. Uniform sampling from the global FIFO let long
        // successful episodes drown out short ones; replaying inactive task
        // labels after a switch also blocked the new task from adapting.
        // Inactive samples remain capacity-protected and become eligible again
        // when their task returns.
        let sample_seed = 0x511A_710A_7EED_u64
            ^ (self.step_count as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ self.sil_updates_attempted;
        let active_env_ids = self
            .lanes
            .iter()
            .map(|lane| lane.adapter.id())
            .collect::<std::collections::BTreeSet<_>>();
        let sampled_idx = balanced_sil_sample_indices(
            &self.sil_buffer,
            policy_batch,
            sample_seed,
            &active_env_ids,
        );

        // Fill scratch buffers from the sampled SIL transitions.
        let ld = self.latent_dim;
        self.obs_token_scratch.fill(0.0);
        self.task_scratch.fill(0.0);
        self.policy_action_scratch.fill(0.0);
        self.value_target_scratch.fill(0.0);
        if !e2e {
            self.policy_z_scratch.fill(0.0);
        }
        if self.policy_action_mask_input_present {
            self.policy_action_mask_scratch.fill(1.0);
        }
        if use_ppo {
            self.ppo_advantage_scratch.fill(0.0);
            self.ppo_old_prob_scratch.fill(1.0);
        }

        let coef = self.config.sil_loss_coef;
        // Clamp the SIL per-sample advantage to the same scale as the
        // regular policy update's `advantage_clamp` (default ±1 after
        // advantage_normalize). Without this clamp, raw R_to_go values
        // (which can be 50–200 on dense-reward envs) produce SIL weights
        // 50–200× larger than the regular advantage, and the SIL update
        // dominates, destabilizing training.
        let adv_cap = self.config.advantage_clamp.max(0.1);
        let mut n_active = 0;
        for (row, &si) in (0..policy_batch).zip(sampled_idx.iter()) {
            let sample = &self.sil_buffer[si];
            // Standard SIL uses max(0, R_to_go - V_at_collect). In explicit
            // event-filter mode, every transition from a real event episode
            // receives at least unit BC advantage: otherwise environments
            // with negative step costs discard the early, essential part of
            // every successful trajectory and imitate only the terminal tail.
            let adv = sil_imitation_advantage(
                sample.r_to_go,
                sample.v_at_collect,
                sample.event_success,
                self.config.sil_event_filter,
                adv_cap,
            );
            let weight = coef * adv;
            if e2e {
                // obs
                let obs_dst =
                    &mut self.obs_token_scratch[row * OBS_TOKEN_DIM..(row + 1) * OBS_TOKEN_DIM];
                let copy_n = sample.obs.len().min(OBS_TOKEN_DIM);
                obs_dst[..copy_n].copy_from_slice(&sample.obs[..copy_n]);
                // task
                let task_dst = &mut self.task_scratch[row * TASK_DIM..(row + 1) * TASK_DIM];
                if let Some(emb) = self.task_embeddings.get(&sample.env_id) {
                    task_dst.copy_from_slice(emb);
                }
            } else {
                // Non-e2e: stage the stored acted-from latent.
                let z_dst = &mut self.policy_z_scratch[row * ld..(row + 1) * ld];
                let copy_n = sample.z.len().min(ld);
                z_dst[..copy_n].copy_from_slice(&sample.z[..copy_n]);
            }
            let act_dst =
                &mut self.policy_action_scratch[row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
            if self.policy_action_mask_input_present {
                restore_action_mask(
                    &mut self.policy_action_mask_scratch
                        [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM],
                    &sample.action_mask,
                );
            }
            if use_ppo {
                // Plain one-hot; the weight rides in `advantage`.
                if sample.action_idx < MAX_ACTION_DIM && weight > 0.0 {
                    act_dst[sample.action_idx] = 1.0;
                }
                self.ppo_advantage_scratch[row] = weight;
                self.ppo_old_prob_scratch[row] = 1.0;
            } else {
                // action: weight * one_hot(taken). When weight = 0 the
                // CE contribution from this row is zero (no gradient).
                if sample.action_idx < MAX_ACTION_DIM {
                    act_dst[sample.action_idx] = weight;
                }
            }
            // value target: train V toward R_to_go (helps V learn
            // to predict actual outcomes from successful states,
            // which ALSO sharpens future SIL filtering).
            self.value_target_scratch[row] = sample.r_to_go;
            if weight > 0.0 {
                n_active += 1;
            }
        }
        self.sil_last_active_rows = n_active as u32;
        // No-op if all sampled rows had non-positive advantage.
        if n_active == 0 {
            return;
        }
        self.sil_updates_fired += 1;

        // Diagnostic: snapshot a small param slice BEFORE the SIL step.
        let mut p_before = vec![0.0f32; 32];
        self.policy_session
            .read_param("policy.fc1.weight", &mut p_before);

        if e2e {
            self.policy_session
                .set_input("obs", &self.obs_token_scratch);
            self.policy_session.set_input("task", &self.task_scratch);
        } else {
            self.policy_session.set_input("z", &self.policy_z_scratch);
            if self.option_session.is_some() {
                // SIL samples carry no option identity; zero the
                // one-hot so the option bias head contributes nothing
                // (stale rows from the last regular update otherwise
                // persist in the device buffer).
                self.option_onehot_scratch.fill(0.0);
                self.policy_session
                    .set_input("option_onehot", &self.option_onehot_scratch);
            }
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        if self.policy_action_mask_input_present {
            self.policy_session
                .set_input("action_mask", &self.policy_action_mask_scratch);
        }
        if use_ppo {
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        if self.reward_pred_input_present {
            // SIL samples don't store a per-step reward, so feed a
            // zero target rather than let the aux reward head train
            // on the PREVIOUS rollout's rewards paired with these SIL
            // observations (the device buffer keeps the last upload).
            self.reward_target_scratch.fill(0.0);
            self.policy_session
                .set_input("reward_target", &self.reward_target_scratch);
        }
        self.feed_entropy_beta_input();
        // Use the same effective LR as a regular update.
        let lr = self.config.lr_policy * self.batch_lr_scale;
        apply_lr(
            &mut self.policy_session,
            lr,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.policy_session.step();
        self.policy_session.wait();

        // Diagnostic: param L1 change. Stored in `sil_last_param_change`.
        let mut p_after = vec![0.0f32; 32];
        self.policy_session
            .read_param("policy.fc1.weight", &mut p_after);
        let change: f32 = p_before
            .iter()
            .zip(p_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        self.sil_last_param_change = change;
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Save trainable weights to a directory: one safetensors file per
    /// neural-network session (wm, policy, and optional option), plus the
    /// CPU coordinate head when enabled. Lane state, step count, and harness scratch
    /// buffers are NOT saved — load expects a freshly-instantiated
    /// agent with the same architecture (latent_dim, hidden_dim,
    /// encoder kind, etc.).
    pub fn save_weights(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        self.wm_session
            .save_checkpoint(&dir.join("wm.safetensors"))?;
        self.policy_session
            .save_checkpoint(&dir.join("policy.safetensors"))?;
        if let Some(ref mut s) = self.option_session {
            s.save_checkpoint(&dir.join("option.safetensors"))?;
        }
        if let Some(ref head) = self.coord_head {
            head.save(&dir.join("coord.bin"))?;
        }
        Ok(())
    }

    /// Load ONLY the wm_session checkpoint from a single safetensors
    /// file. Used for partial pretrained-encoder injection: the file
    /// can contain just `encoder.conv*.weight` etc; missing entries
    /// are tolerated by meganeura's load_checkpoint (the matching
    /// param keeps its xavier init). The policy session is untouched.
    pub fn load_wm_checkpoint(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        self.wm_session.load_checkpoint(path)?;
        Ok(())
    }

    /// Set per-prefix LR multiplier on wm_session. mul=0.0 freezes
    /// matching parameters. Used to keep a pretrained encoder fixed
    /// while WM/value/policy adapt.
    pub fn set_wm_lr_multiplier(&mut self, prefix: &str, mul: f32) {
        self.wm_session.set_lr_multiplier(prefix, mul);
    }

    /// Load trainable weights from a directory previously written by
    /// `save_weights`. The agent must already be constructed with the same
    /// graph topology. This is not a resumable training checkpoint: replay,
    /// optimizer, RNG, lane, archive, and schedule state are not restored.
    pub fn load_weights(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        self.wm_session
            .load_checkpoint(&dir.join("wm.safetensors"))?;
        self.policy_session
            .load_checkpoint(&dir.join("policy.safetensors"))?;
        if let Some(ref mut s) = self.option_session {
            let p = dir.join("option.safetensors");
            if p.exists() {
                s.load_checkpoint(&p)?;
            }
        }
        if let Some(ref mut head) = self.coord_head {
            let p = dir.join("coord.bin");
            if p.exists() {
                head.load(&p)?;
            }
        }
        Ok(())
    }

    /// Cheap per-lane read of `last_r_hat` without building the full
    /// Diagnostics (used by M6 instrumentation harnesses that log this
    /// every step).
    pub fn r_hats(&self) -> Vec<f32> {
        self.lanes.iter().map(|l| l.last_r_hat).collect()
    }

    /// Per-lane confidence C ∈ [0, 1] used by the confidence-weighted
    /// planner. Constant 0.5 when `confidence_mode=false`.
    pub fn confidence(&self) -> Vec<f32> {
        self.lanes.iter().map(|l| l.confidence).collect()
    }

    /// Per-lane value baseline V(s_t) cached from the most recent act().
    /// Used for diagnosing whether the value head is discriminating
    /// across states or just predicting the mean return everywhere.
    pub fn values(&self) -> Vec<f32> {
        self.lanes.iter().map(|l| l.last_value).collect()
    }

    /// Per-lane policy entropy from the most recent act().
    pub fn entropies(&self) -> Vec<f32> {
        self.lanes.iter().map(|l| l.last_entropy).collect()
    }

    /// Per-lane current latent z (the encoder output that feeds the
    /// policy / WM / decoder). Returns one Vec<f32> of length
    /// `latent_dim` per lane. Empty Vec if the lane has no transitions
    /// yet (pre-first-observe). Used for diagnosing whether the encoder
    /// produces game-distinguishing latents under multi-env training.
    pub fn latents(&self) -> Vec<Vec<f32>> {
        self.lanes
            .iter()
            .map(|l| {
                l.buffer
                    .last()
                    .map(|t| t.latent.clone())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// One behavior-cloning update from externally labeled discrete states.
    ///
    /// This is deliberately narrower than the online RL path: it requires the
    /// plain end-to-end categorical graph and trains only positive masked
    /// cross-entropy labels. It retains simulator/controller paths without
    /// inventing synthetic rewards or depending on an untrained value baseline.
    pub fn train_policy_supervised(
        &mut self,
        observations: &[Observation],
        action_indices: &[usize],
        weight: f32,
    ) -> Result<(), String> {
        let n = self.lanes.len();
        if observations.len() != n {
            return Err(format!(
                "train_policy_supervised: observations.len() {} must equal num_lanes {n}",
                observations.len()
            ));
        }
        if action_indices.len() != n {
            return Err(format!(
                "train_policy_supervised: action_indices.len() {} must equal num_lanes {n}",
                action_indices.len()
            ));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err("train_policy_supervised: weight must be finite and positive".into());
        }
        if !(self.config.end_to_end_encoder
            && !self.config.use_ppo
            && !self.config.use_kl_ppo
            && self.config.num_options == 1
            && self.policy_action_mask_input_present
            && self.config.value_loss_coef == 0.0
            && self.config.recon_loss_coef == 0.0
            && self.config.reward_pred_loss_coef == 0.0
            && self.config.entropy_beta == 0.0)
        {
            return Err(
                "train_policy_supervised requires the plain end-to-end categorical policy graph with auxiliary losses disabled"
                    .into(),
            );
        }

        let policy_batch = n * self.config.rollout_length.max(1);
        self.obs_token_scratch.fill(0.0);
        self.task_scratch.fill(0.0);
        self.policy_action_scratch.fill(0.0);
        self.policy_action_mask_scratch.fill(0.0);
        self.value_target_scratch.fill(0.0);

        for row in 0..policy_batch {
            let lane = row % n;
            let action_index = action_indices[lane];
            if action_index >= MAX_ACTION_DIM {
                return Err(format!(
                    "train_policy_supervised: action index {action_index} exceeds MAX_ACTION_DIM"
                ));
            }
            let source_mask =
                &self.action_masks[lane * MAX_ACTION_DIM..(lane + 1) * MAX_ACTION_DIM];
            if source_mask[action_index] < 0.5 {
                return Err(format!(
                    "train_policy_supervised: labeled action {action_index} is masked on lane {lane}"
                ));
            }

            let obs_row =
                &mut self.obs_token_scratch[row * OBS_TOKEN_DIM..(row + 1) * OBS_TOKEN_DIM];
            self.lanes[lane]
                .adapter
                .obs_to_token(&observations[lane], obs_row);
            if let Some(embedding) = self.task_embeddings.get(&self.lanes[lane].adapter.id()) {
                self.task_scratch[row * TASK_DIM..(row + 1) * TASK_DIM].copy_from_slice(embedding);
            }
            self.policy_action_scratch[row * MAX_ACTION_DIM + action_index] = weight;
            self.policy_action_mask_scratch[row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM]
                .copy_from_slice(source_mask);
        }

        self.policy_session
            .set_input("obs", &self.obs_token_scratch);
        self.policy_session.set_input("task", &self.task_scratch);
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        self.policy_session
            .set_input("action_mask", &self.policy_action_mask_scratch);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        if self.reward_pred_input_present {
            self.reward_target_scratch.fill(0.0);
            self.policy_session
                .set_input("reward_target", &self.reward_target_scratch);
        }
        self.feed_entropy_beta_input();
        let lr = self.config.lr_policy * self.batch_lr_scale;
        apply_lr(
            &mut self.policy_session,
            lr,
            self.config.use_adam,
            self.config.adam_eps,
        );
        self.policy_session.step();
        self.policy_session.wait();
        let mut loss = [0.0f32; 1];
        self.policy_session.read_output_by_index(0, &mut loss);
        self.last_policy_loss = loss[0];
        Ok(())
    }

    /// Sample per-lane `(x, y)` coordinates from the coord head's
    /// current policy. Returns a `[(x, y); N]` vector in `[−1, 1]`
    /// space; callers rescale to the target env's coord range.
    /// Returns zeros when the coord head is disabled.
    ///
    /// After the next `observe()` completes, call
    /// `train_coord_head()` to update the head via REINFORCE on
    /// whatever advantage signal the harness chooses (typically
    /// the last per-step reward minus a baseline).
    pub fn sample_coords<R: Rng>(&mut self, rng: &mut R) -> Vec<(f32, f32)> {
        let n = self.lanes.len();
        let Some(head) = self.coord_head.as_mut() else {
            return vec![(0.0, 0.0); n];
        };
        let mut out = Vec::with_capacity(n);
        for (i, lane) in self.lanes.iter().enumerate() {
            let task_embedding = self
                .task_embeddings
                .get(&lane.adapter.id())
                .map_or(&[][..], Vec::as_slice);
            let features = lane.buffer.last().map_or_else(
                || compose_coord_features(&[], task_embedding),
                |transition| compose_coord_features(&transition.observation, task_embedding),
            );
            let [sx, sy] = head.sample(i, &features, || {
                // Box-Muller from a pair of uniforms.
                let u1: f32 = rng.random_range(1e-7..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                (r * theta.cos(), r * theta.sin())
            });
            out.push((sx, sy));
        }
        out
    }

    /// Deterministic coordinate-head means for supplied current observations.
    /// Unlike `sample_coords`, this tokenizes the provided states directly and
    /// does not depend on the lane buffer or run the world-model encoder.
    pub fn coord_means_for_observations(
        &mut self,
        observations: &[Observation],
    ) -> Vec<(f32, f32)> {
        assert_eq!(
            observations.len(),
            self.lanes.len(),
            "coord_means_for_observations: observations.len() must equal num_lanes"
        );
        for (i, lane) in self.lanes.iter().enumerate() {
            lane.adapter.obs_to_token(
                &observations[i],
                &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM],
            );
        }
        let Some(head) = self.coord_head.as_ref() else {
            return vec![(0.0, 0.0); self.lanes.len()];
        };
        (0..self.lanes.len())
            .map(|i| {
                let task_embedding = self
                    .task_embeddings
                    .get(&self.lanes[i].adapter.id())
                    .map_or(&[][..], Vec::as_slice);
                let features = compose_coord_features(
                    &self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM],
                    task_embedding,
                );
                let [x, y] = head.forward(&features);
                (x, y)
            })
            .collect()
    }

    /// Supervised coordinate imitation from externally successful states.
    /// Updates only the CPU coordinate head from adapter observation tokens
    /// and the lanes' fixed task embeddings.
    /// Returns mean weighted MSE over active lanes.
    pub fn train_coord_head_supervised(
        &mut self,
        observations: &[Observation],
        targets: &[[f32; 2]],
        active: &[bool],
        weight: f32,
    ) -> f32 {
        let n = self.lanes.len();
        assert_eq!(
            observations.len(),
            n,
            "train_coord_head_supervised: observations.len() must equal num_lanes"
        );
        assert_eq!(
            targets.len(),
            n,
            "train_coord_head_supervised: targets.len() must equal num_lanes"
        );
        assert_eq!(
            active.len(),
            n,
            "train_coord_head_supervised: active.len() must equal num_lanes"
        );
        for (i, lane) in self.lanes.iter().enumerate() {
            lane.adapter.obs_to_token(
                &observations[i],
                &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM],
            );
        }
        let mut features = Vec::with_capacity(n * (OBS_TOKEN_DIM + TASK_DIM));
        for i in 0..n {
            let task_embedding = self
                .task_embeddings
                .get(&self.lanes[i].adapter.id())
                .map_or(&[][..], Vec::as_slice);
            features.extend(compose_coord_features(
                &self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM],
                task_embedding,
            ));
        }
        let mut active_per_task = HashMap::<u32, usize>::new();
        let mut active_count = 0usize;
        for (lane, &enabled) in self.lanes.iter().zip(active) {
            if enabled {
                *active_per_task.entry(lane.adapter.id()).or_default() += 1;
                active_count += 1;
            }
        }
        if active_count == 0 {
            return 0.0;
        }
        let task_count = active_per_task.len();
        let sample_weights = self
            .lanes
            .iter()
            .zip(active)
            .map(|(lane, &enabled)| {
                if !enabled {
                    return 0.0;
                }
                let task_samples = active_per_task[&lane.adapter.id()];
                weight * active_count as f32 / (task_count * task_samples) as f32
            })
            .collect::<Vec<_>>();
        let Some(head) = self.coord_head.as_mut() else {
            return 0.0;
        };
        head.train_supervised_batch_weighted(&features, targets, &sample_weights)
    }

    /// Train every lane of the coord head on the most recent step's rewards.
    /// Prefer [`Self::train_coord_head_masked`] when only some executed actions
    /// consumed coordinates.
    pub fn train_coord_head(&mut self) {
        let active = vec![true; self.lanes.len()];
        self.train_coord_head_masked(&active);
    }

    /// Train only lanes whose executed action consumed the sampled coordinate.
    /// The reward baseline is computed from and updated by active lanes only, so
    /// rewards from ordinary discrete actions cannot reinforce unrelated click
    /// locations. No-op when the coord head is disabled or no lane is active.
    pub fn train_coord_head_masked(&mut self, active: &[bool]) {
        let rewards = self.coord_last_reward.clone();
        self.train_coord_head_masked_with_rewards(active, &rewards);
    }

    /// Train selected coordinate lanes from an explicit per-lane reward signal.
    /// This lets an environment use sparse task reward for click credit while
    /// the main policy continues to optimize Kindle's combined reward. Each
    /// lane is centered against its own EMA, avoiding cross-task leakage.
    pub fn train_coord_head_masked_with_rewards(&mut self, active: &[bool], rewards: &[f32]) {
        assert_eq!(
            active.len(),
            self.lanes.len(),
            "train_coord_head_masked_with_rewards: active.len() must equal num_lanes"
        );
        assert_eq!(
            rewards.len(),
            self.lanes.len(),
            "train_coord_head_masked_with_rewards: rewards.len() must equal num_lanes"
        );
        let Some(head) = self.coord_head.as_mut() else {
            return;
        };
        let alpha = self.config.coord_action_alpha;
        let ema = 0.02f32;
        for i in 0..self.lanes.len() {
            if !active[i] {
                continue;
            }
            let reward = if rewards[i].is_finite() {
                rewards[i]
            } else {
                0.0
            };
            self.coord_reward_baseline[i] =
                (1.0 - ema) * self.coord_reward_baseline[i] + ema * reward;
            let adv = (reward - self.coord_reward_baseline[i]) * alpha;
            // The head backprops through the latent it cached at
            // sample() time — passing the lane's current latent here
            // paired the previous state's action-sample residual with
            // the NEXT state's activations (worst across episode
            // boundaries, where the latent jumps).
            head.train_step(i, adv);
        }
    }

    /// Re-initialize the RND predictor (keeps target frozen). Used
    /// by the harness to re-activate curiosity when the agent
    /// enters a qualitatively new state distribution (e.g. on
    /// level-up in ARC-AGI-3). No-op when RND is disabled.
    pub fn reset_rnd_predictor(&mut self) {
        if let Some(state) = self.rnd_state.as_mut() {
            state.reset_predictor();
        }
    }

    /// Current size of the M8 delta-goal bank, or 0 when M8 is off.
    pub fn delta_goal_bank_size(&self) -> usize {
        self.delta_goal_bank.as_ref().map_or(0, |b| b.len())
    }

    /// Number of M8 goal-events recorded in the most recent
    /// `observe()` call, summed across lanes. Zero when M8 is off.
    pub fn last_delta_goal_events(&self) -> usize {
        self.last_delta_goal_events
    }

    /// Distinct `(quantized_state, action)` pairs in the
    /// cross-episode memory. 0 when disabled. Diagnostic.
    pub fn xeps_distinct_pairs(&self) -> usize {
        self.xeps_memory.as_ref().map_or(0, |m| m.distinct_pairs())
    }

    /// Run the Track 3 model-based planner for every lane whose
    /// planner queue is currently empty. For each such lane:
    /// - Sample `planner_samples` random discrete action sequences
    ///   of length `planner_horizon` over `num_actions` (passed by
    ///   the caller — typically from the env's action space).
    /// - Roll each sequence out through the frozen WM starting
    ///   from the lane's most recent latent.
    /// - For slots marked by `set_action_parameter_masks`, sample normalized
    ///   `(x, y)` and include them in the WM rollout.
    /// - Score each by the sum of `1/sqrt(1+visit_count)` across
    ///   predicted latents — pulling toward under-visited regions
    ///   of the state space.
    /// - Queue the highest-scoring action/parameter sequence for consumption
    ///   by `act()` and `take_planned_action_parameters()`.
    ///
    /// No-op when `planner_horizon == 0` or when a lane has no
    /// prior observation (first step of the agent's life).
    /// `num_actions` is clamped to the WM's compiled action width.
    pub fn plan_and_queue<R: Rng>(&mut self, num_actions: usize, rng: &mut R) {
        let k = self.config.planner_horizon;
        let m = self.config.planner_samples;
        if k == 0 || m == 0 {
            return;
        }

        // The current MCTS tree has one child per discrete identity, so it
        // cannot compare multiple parameter values for the same action. Route
        // parameterized action spaces through batched random shooting, whose
        // candidate rows do represent distinct `(action, x, y)` tuples.
        let has_parameterized_actions = self.action_parameter_masks.iter().any(|&active| active);
        if self.config.planner_use_mcts
            && !has_parameterized_actions
            && self.wm_mcts_session.is_some()
        {
            self.plan_and_queue_mcts(num_actions, rng);
            return;
        }
        // GPU path when wm_planner_session is constructed.
        if self.wm_planner_session.is_some() {
            self.plan_and_queue_gpu(num_actions, rng);
            return;
        }

        // CPU fallback (legacy path; kept for completeness when the
        // GPU planner session can't be built).
        let Some(planner) = self.planner.as_mut() else {
            return;
        };
        if self.planner_calls_since_refresh == 0
            || self.planner_calls_since_refresh >= self.config.planner_refresh_interval
        {
            planner.refresh_from_session(&self.wm_session);
            self.planner_calls_since_refresh = 0;
        }
        self.planner_calls_since_refresh += 1;

        let num_actions_eff = num_actions.clamp(1, MAX_ACTION_DIM);
        let mut traj = vec![0.0f32; k * planner.latent_dim];
        let mut action_tokens = vec![0.0f32; k * planner.action_dim];

        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            if !self.planner_queue[lane_idx].is_empty() {
                continue;
            }
            let z0 = match lane.buffer.last() {
                Some(t) => t.latent.clone(),
                None => continue,
            };
            let mask_row =
                &self.action_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            let valid: Vec<u32> = (0..num_actions_eff)
                .filter(|&j| mask_row[j] >= 0.5)
                .map(|j| j as u32)
                .collect();
            let valid: Vec<u32> = if valid.is_empty() {
                (0..num_actions_eff).map(|j| j as u32).collect()
            } else {
                valid
            };

            let mut best_score = f32::NEG_INFINITY;
            let parameter_mask = &self.action_parameter_masks
                [lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            let mut best_actions: Vec<PlannedAction> = Vec::new();
            for _ in 0..m {
                let actions: Vec<u32> = (0..k)
                    .map(|_| valid[rng.random_range(0..valid.len())])
                    .collect();
                action_tokens.fill(0.0);
                let mut planned = Vec::with_capacity(k);
                for (step, &action) in actions.iter().enumerate() {
                    let action_idx = action as usize;
                    let row = &mut action_tokens
                        [step * planner.action_dim..(step + 1) * planner.action_dim];
                    row[action_idx] = 1.0;
                    let parameters = if parameter_mask[action_idx] {
                        let values = [rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)];
                        row[MAX_ACTION_DIM..MAX_ACTION_DIM + ACTION_PARAMETER_DIM]
                            .copy_from_slice(&values);
                        Some(values)
                    } else {
                        None
                    };
                    planned.push(PlannedAction { action, parameters });
                }
                planner.rollout_tokens(&z0, &action_tokens, &mut traj);

                let mut score = 0.0f32;
                for step in 0..k {
                    let z_step = &traj[step * planner.latent_dim..(step + 1) * planner.latent_dim];
                    let c = lane.buffer.visit_count(z_step);
                    score += 1.0 / ((c as f32 + 1.0).sqrt());
                }
                if score > best_score {
                    best_score = score;
                    best_actions = planned;
                }
            }
            for planned in best_actions {
                self.planner_queue[lane_idx].push_back(planned);
            }
        }
    }

    /// GPU-side WM rollout planner. For each lane with an empty queue,
    /// samples `planner_samples` random valid-action sequences of length
    /// `planner_horizon`, batches them through a forward-only WM session
    /// in `planner_horizon` GPU dispatches (instead of `samples * horizon`
    /// CPU matmul loops), scores by latent-visit-count novelty, and
    /// queues the best sequence's actions.
    fn plan_and_queue_gpu<R: Rng>(&mut self, num_actions: usize, rng: &mut R) {
        let k = self.config.planner_horizon;
        let m = self.planner_samples_cached;
        let n = self.lanes.len();
        let ld = self.config.latent_dim;
        let batch = n * m;
        let num_actions_eff = num_actions.clamp(1, MAX_ACTION_DIM);

        // Periodic WM-weight sync: read from wm_session, write to
        // wm_planner_session. First call always syncs.
        if self.planner_calls_since_refresh == 0
            || self.planner_calls_since_refresh >= self.config.planner_refresh_interval
        {
            self.refresh_wm_planner_weights();
            self.planner_calls_since_refresh = 0;
        }
        self.planner_calls_since_refresh += 1;

        // Which lanes need planning this call (queue empty + buffer has data).
        // Lanes without buffer data fill their batch rows with z0=0 — the
        // trajectory will be noise we ignore at scoring time.
        let mut lane_active = vec![false; n];
        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            if !self.planner_queue[lane_idx].is_empty() {
                continue;
            }
            if lane.buffer.last().is_some() {
                lane_active[lane_idx] = true;
            }
        }
        if !lane_active.iter().any(|&b| b) {
            return;
        }

        // Build per-lane valid-action lists from the mask.
        let mut valid_per_lane: Vec<Vec<u32>> = Vec::with_capacity(n);
        for lane_idx in 0..n {
            let mask_row =
                &self.action_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            let valid: Vec<u32> = (0..num_actions_eff)
                .filter(|&j| mask_row[j] >= 0.5)
                .map(|j| j as u32)
                .collect();
            if valid.is_empty() {
                valid_per_lane.push((0..num_actions_eff as u32).collect());
            } else {
                valid_per_lane.push(valid);
            }
        }

        // We'll fill `all_actions` step-by-step as we sample from the
        // policy at each WM rollout time. Layout: row-major
        // `[batch, k, inner_iters_max]`. In non-option mode, inner_iters
        // = planner_action_repeat (action just repeats). In option-aware
        // mode, inner_iters = option_horizon (each inner step re-samples
        // option-conditional action). Storage is allocated for the max
        // path; non-option mode just writes the same action across inner.
        let no = self.config.num_options;
        let use_options = no >= 2 && !self.planner_option_onehot_scratch.is_empty();
        let inner_iters = if use_options {
            self.config.option_horizon.max(1)
        } else {
            self.config.planner_action_repeat.max(1)
        };
        let mut all_actions = vec![0u32; batch * k * inner_iters];
        let mut all_action_parameters =
            vec![0.0f32; batch * k * inner_iters * ACTION_PARAMETER_DIM];

        // Seed z input: replicate each lane's z0 m times.
        self.planner_z_scratch.fill(0.0);
        for lane_idx in 0..n {
            if !lane_active[lane_idx] {
                continue;
            }
            let z0 = self.lanes[lane_idx].buffer.last().unwrap().latent.clone();
            for s in 0..m {
                let row = lane_idx * m + s;
                self.planner_z_scratch[row * ld..(row + 1) * ld].copy_from_slice(&z0);
            }
        }

        // Scratch for one-step policy logits read [batch, MAX_ACTION_DIM].
        let mut policy_logits = vec![0.0f32; batch * MAX_ACTION_DIM];

        let policy_mix = self.config.planner_policy_mix.clamp(0.0, 1.0);
        let policy_temp = self.config.planner_policy_temperature.max(0.01);
        let need_policy =
            (policy_mix > 0.0 || use_options) && self.policy_planner_session.is_some();
        let policy_mix_eff = if use_options { 1.0 } else { policy_mix };

        // Per-step rollout. For each outer step `t`:
        //   - If option-aware, pick a per-row option (held constant for
        //     all `inner_iters` inner WM steps).
        //   - For each inner step `i`:
        //     - Policy forward (option-conditional if use_options): z → logits
        //     - Sample action per row from logits (or uniform with prob 1−mix)
        //     - WM forward (z, action_one_hot) → z_next
        //     - z := z_next; record action into all_actions[row, t, i]
        //
        // Effective atomic-action reach per planner call = k × inner_iters.
        // Trajectory endpoint stored per outer step in planner_traj_scratch[t]
        // is the z AFTER inner_iters atomic steps.
        for t in 0..k {
            // Option choice (constant across inner loop within this t)
            if use_options {
                self.planner_option_onehot_scratch.fill(0.0);
                for row in 0..batch {
                    let opt = rng.random_range(0..no);
                    self.planner_option_onehot_scratch[row * no + opt] = 1.0;
                }
            }
            let wm_planner_ptr: *mut Session = self.wm_planner_session.as_mut().unwrap();
            for inner_idx in 0..inner_iters {
                // Policy forward (option-conditional when use_options)
                if need_policy && (use_options || inner_idx == 0) {
                    let policy_planner = self.policy_planner_session.as_mut().unwrap();
                    policy_planner.set_input("z", &self.planner_z_scratch);
                    if use_options {
                        policy_planner
                            .set_input("option_onehot", &self.planner_option_onehot_scratch);
                    }
                    policy_planner.step();
                    policy_planner.wait();
                    policy_planner.read_output_by_index(0, &mut policy_logits);
                }

                // Sample actions per row
                self.planner_action_scratch.fill(0.0);
                for lane_idx in 0..n {
                    if !lane_active[lane_idx] {
                        continue;
                    }
                    let valid = &valid_per_lane[lane_idx];
                    let parameter_mask = &self.action_parameter_masks
                        [lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
                    for s in 0..m {
                        let row = lane_idx * m + s;
                        let sequence_idx = row * k * inner_iters + t * inner_iters + inner_idx;
                        let action_idx: u32 = if !use_options && inner_idx > 0 {
                            all_actions[sequence_idx - inner_idx]
                        } else {
                            let use_policy =
                                need_policy && rng.random_range(0.0..1.0_f32) < policy_mix_eff;
                            if !use_policy {
                                valid[rng.random_range(0..valid.len())]
                            } else {
                                let lrow = &policy_logits
                                    [row * MAX_ACTION_DIM..(row + 1) * MAX_ACTION_DIM];
                                let mut max_l = f32::NEG_INFINITY;
                                for &v in valid {
                                    let l = lrow[v as usize] / policy_temp;
                                    if l.is_finite() && l > max_l {
                                        max_l = l;
                                    }
                                }
                                if !max_l.is_finite() {
                                    max_l = 0.0;
                                }
                                let mut sum = 0.0f32;
                                let probs: Vec<f32> = valid
                                    .iter()
                                    .map(|&v| {
                                        let p = (lrow[v as usize] / policy_temp - max_l).exp();
                                        sum += p;
                                        p
                                    })
                                    .collect();
                                if sum <= 0.0 || !sum.is_finite() {
                                    valid[rng.random_range(0..valid.len())]
                                } else {
                                    let u: f32 = rng.random_range(0.0..1.0) * sum;
                                    let mut cum = 0.0;
                                    let mut chosen = 0;
                                    for (i, &p) in probs.iter().enumerate() {
                                        cum += p;
                                        if u <= cum {
                                            chosen = i;
                                            break;
                                        }
                                    }
                                    valid[chosen]
                                }
                            }
                        };
                        // Store in (row, t, inner_idx) layout
                        all_actions[sequence_idx] = action_idx;
                        if (action_idx as usize) < MAX_ACTION_DIM {
                            self.planner_action_scratch
                                [row * WM_ACTION_DIM + action_idx as usize] = 1.0;
                            if parameter_mask[action_idx as usize] {
                                let parameter_offset = sequence_idx * ACTION_PARAMETER_DIM;
                                if !use_options && inner_idx > 0 {
                                    let source = (sequence_idx - inner_idx) * ACTION_PARAMETER_DIM;
                                    all_action_parameters.copy_within(
                                        source..source + ACTION_PARAMETER_DIM,
                                        parameter_offset,
                                    );
                                } else {
                                    all_action_parameters[parameter_offset] =
                                        rng.random_range(-1.0..1.0);
                                    all_action_parameters[parameter_offset + 1] =
                                        rng.random_range(-1.0..1.0);
                                }
                                self.planner_action_scratch[row * WM_ACTION_DIM + MAX_ACTION_DIM
                                    ..row * WM_ACTION_DIM + WM_ACTION_DIM]
                                    .copy_from_slice(
                                        &all_action_parameters[parameter_offset
                                            ..parameter_offset + ACTION_PARAMETER_DIM],
                                    );
                            }
                        }
                    }
                }

                // WM forward
                let wm_planner = unsafe { &mut *wm_planner_ptr };
                wm_planner.set_input("z", &self.planner_z_scratch);
                wm_planner.set_input("action", &self.planner_action_scratch);
                wm_planner.step();
                wm_planner.wait();
                // We only need to keep the LATEST z_next for the next
                // outer step's input; final intermediate value gets
                // stored as the per-step trajectory endpoint below.
                let traj_offset_inner = t * batch * ld;
                wm_planner.read_output_by_index(
                    0,
                    &mut self.planner_traj_scratch
                        [traj_offset_inner..traj_offset_inner + batch * ld],
                );
                // GRAM-style stochastic perturbation: add Gaussian noise
                // per element. σ is either fixed (planner_noise_sigma
                // alone) or the product of planner_noise_sigma and the
                // WM's learned per-state σ_φ(z, a) when wm_stochastic
                // is enabled. Each of the m parallel rollouts per lane
                // gets independent N(0, 1) draws, scaled in place.
                let scale = self.config.planner_noise_sigma;
                if scale > 0.0 {
                    // Read learned σ if the WM has the head wired up.
                    // Layout matches z_next: [batch * ld].
                    if self.config.wm_stochastic {
                        if self.planner_sigma_scratch.len() < batch * ld {
                            self.planner_sigma_scratch.resize(batch * ld, 0.0f32);
                        }
                        wm_planner
                            .read_output_by_index(1, &mut self.planner_sigma_scratch[..batch * ld]);
                    }
                    let slice = &mut self.planner_traj_scratch
                        [traj_offset_inner..traj_offset_inner + batch * ld];
                    let sigma_buf = if self.config.wm_stochastic {
                        Some(&self.planner_sigma_scratch[..batch * ld])
                    } else {
                        None
                    };
                    let mut i = 0;
                    while i + 1 < slice.len() {
                        // Box-Muller: two uniforms → two N(0,1) samples.
                        let u1: f32 = rng.random_range(1e-7..1.0);
                        let u2: f32 = rng.random_range(0.0..1.0);
                        let mag = (-2.0 * u1.ln()).sqrt();
                        let n0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
                        let n1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
                        let s0 = sigma_buf.map(|b| b[i]).unwrap_or(1.0);
                        let s1 = sigma_buf.map(|b| b[i + 1]).unwrap_or(1.0);
                        slice[i] += scale * s0 * n0;
                        slice[i + 1] += scale * s1 * n1;
                        i += 2;
                    }
                    if i < slice.len() {
                        let u1: f32 = rng.random_range(1e-7..1.0);
                        let u2: f32 = rng.random_range(0.0..1.0);
                        let mag = (-2.0 * u1.ln()).sqrt();
                        let s_last = sigma_buf.map(|b| b[i]).unwrap_or(1.0);
                        slice[i] += scale * s_last * mag * (2.0 * std::f32::consts::PI * u2).cos();
                    }
                }
                self.planner_z_scratch.copy_from_slice(
                    &self.planner_traj_scratch[traj_offset_inner..traj_offset_inner + batch * ld],
                );
            }
            // Value-head branch reads per-row scalar V for the t-th
            // step's final z_next. Stored row-major `[t * batch + row]`
            // so the scorer below can fetch it without re-running the
            // session. Output index depends on whether σ also occupies
            // an output slot: [z_next, optional sigma, optional value].
            // (P3) Capture per-row mean σ at this outer step's endpoint
            // (σ of the final inner transition) for the information-
            // seeking score bonus and the σ-budgeted horizon.
            let need_sigma_traj = self.config.wm_stochastic
                && (self.config.planner_sigma_alpha > 0.0
                    || self.config.planner_sigma_horizon > 0.0);
            if need_sigma_traj {
                if self.planner_sigma_scratch.len() < batch * ld {
                    self.planner_sigma_scratch.resize(batch * ld, 0.0f32);
                }
                if self.planner_sigma_traj_scratch.len() < k * batch {
                    self.planner_sigma_traj_scratch.resize(k * batch, 0.0f32);
                }
                let wm_planner = unsafe { &mut *wm_planner_ptr };
                wm_planner.read_output_by_index(1, &mut self.planner_sigma_scratch[..batch * ld]);
                for row in 0..batch {
                    let mut sum = 0.0f32;
                    for d in 0..ld {
                        let v = self.planner_sigma_scratch[row * ld + d];
                        if v.is_finite() {
                            sum += v;
                        }
                    }
                    self.planner_sigma_traj_scratch[t * batch + row] = sum / ld as f32;
                }
            }
            if self.wm_planner_has_value_head {
                let v_off = t * batch;
                let wm_planner = unsafe { &mut *wm_planner_ptr };
                let v_idx = if self.config.wm_stochastic { 2 } else { 1 };
                wm_planner.read_output_by_index(
                    v_idx,
                    &mut self.planner_v_traj_scratch[v_off..v_off + batch],
                );
            }
        }

        // Empowerment estimate: per-lane variance of step-0 z_next
        // across the m samples. High variance = different first
        // actions lead to different futures = state has options.
        // Stored for harness consumption via `empowerment()`.
        for lane_idx in 0..n {
            if !lane_active[lane_idx] {
                self.last_empowerment[lane_idx] = 0.0;
                continue;
            }
            let mut mean = vec![0.0f32; ld];
            for s in 0..m {
                let row = lane_idx * m + s;
                let off = row * ld; // t=0 starts at offset 0
                for d in 0..ld {
                    mean[d] += self.planner_traj_scratch[off + d];
                }
            }
            let m_f = m as f32;
            for d in 0..ld {
                mean[d] /= m_f;
            }
            let mut var = 0.0f32;
            for s in 0..m {
                let row = lane_idx * m + s;
                let off = row * ld;
                for d in 0..ld {
                    let v = self.planner_traj_scratch[off + d] - mean[d];
                    if v.is_finite() {
                        var += v * v;
                    }
                }
            }
            let raw = var / m_f / (ld as f32);
            // Clamp to [0, 1] — wm_planner_session is forward-only
            // and doesn't have the wm_session's latent clamp, so a
            // misbehaving WM can produce inf/large variance. Cap so
            // the harness doesn't see infinities.
            self.last_empowerment[lane_idx] = if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        // Score each row by sum-of-novelty over its trajectory.
        // Pick best row per lane and queue its actions.
        let sigma_alpha = self.config.planner_sigma_alpha;
        let sigma_budget = self.config.planner_sigma_horizon;
        let have_sigma_traj = self.config.wm_stochastic
            && (sigma_alpha > 0.0 || sigma_budget > 0.0)
            && self.planner_sigma_traj_scratch.len() >= k * batch;
        let goal_alpha = self.config.planner_goal_alpha;
        let value_alpha = self.config.planner_value_alpha;
        let change_alpha = self.config.planner_change_alpha;
        let subgoal_alpha = self.config.planner_subgoal_alpha;
        let has_value = self.wm_planner_has_value_head;
        for lane_idx in 0..n {
            if !lane_active[lane_idx] {
                continue;
            }
            let lane = &self.lanes[lane_idx];
            let goal_key = if self.config.goal_states_cross_game {
                0u32
            } else {
                lane.adapter.id()
            };
            let goals = if goal_alpha > 0.0 {
                self.goal_states.get(&goal_key)
            } else {
                None
            };
            let centroids = if subgoal_alpha > 0.0 {
                self.subgoal_centroids.get(&goal_key)
            } else {
                None
            };
            // Initial z for change-alpha: the lane's current latent (z0).
            let z0_for_change: Option<Vec<f32>> = if change_alpha > 0.0 {
                lane.buffer.last().map(|t| t.latent.clone())
            } else {
                None
            };
            // Confidence weighting: scale exploit terms (goal/value/
            // centroid) by C, explore terms (change/rnd) by (1-C).
            // visit_count stays as always-on base novelty.
            let (w_exploit, w_explore) = if self.config.confidence_mode {
                (lane.confidence, 1.0 - lane.confidence)
            } else {
                (1.0, 1.0)
            };
            let mut best_score = f32::NEG_INFINITY;
            let mut best_row = lane_idx * m;
            let mut best_trusted_k = k;
            for s in 0..m {
                let row = lane_idx * m + s;
                let mut score = 0.0f32;
                // (P3) σ-budgeted horizon: stop scoring (trusting) the
                // trajectory once cumulative mean-σ exceeds the budget.
                let mut cum_sigma = 0.0f32;
                let mut trusted_k = k;
                for t in 0..k {
                    if have_sigma_traj {
                        let sg = self.planner_sigma_traj_scratch[t * batch + row];
                        if sigma_alpha > 0.0 && sg.is_finite() {
                            // Information-seeking: reward trajectories
                            // that pass through transitions the WM is
                            // still uncertain about.
                            score += w_explore * sigma_alpha * sg;
                        }
                        if sigma_budget > 0.0 && sg.is_finite() {
                            cum_sigma += sg;
                            if cum_sigma > sigma_budget {
                                trusted_k = t + 1;
                            }
                        }
                    }
                    let off = t * batch * ld + row * ld;
                    let z_step = &self.planner_traj_scratch[off..off + ld];
                    let c = lane.buffer.visit_count(z_step);
                    score += 1.0 / ((c as f32 + 1.0).sqrt());
                    if let Some(gq) = goals {
                        if !gq.is_empty() {
                            score += w_exploit * goal_alpha * max_goal_similarity(z_step, gq);
                        }
                    }
                    if let Some(cl) = centroids {
                        if !cl.is_empty() {
                            score +=
                                w_exploit * subgoal_alpha * max_centroid_similarity(z_step, cl);
                        }
                    }
                    // Value-head signal: V(z_step) ≈ expected discounted
                    // return-to-go. Adding it to the score biases the
                    // planner toward latents the head believes are
                    // close to past wins — the snowball mechanism.
                    if has_value && value_alpha > 0.0 {
                        let v = self.planner_v_traj_scratch[t * batch + row];
                        if v.is_finite() {
                            score += w_exploit * value_alpha * v;
                        }
                    }
                    // Affordance / state-change signal: ||z_step − z_prev||.
                    // z_prev = z0 at t=0, otherwise the previous step's
                    // predicted latent (same sample row).
                    if change_alpha > 0.0 {
                        let z_prev: &[f32] = if t == 0 {
                            match z0_for_change.as_deref() {
                                Some(z) => z,
                                None => continue,
                            }
                        } else {
                            let prev_off = (t - 1) * batch * ld + row * ld;
                            &self.planner_traj_scratch[prev_off..prev_off + ld]
                        };
                        let mut sq = 0.0f32;
                        for d in 0..ld {
                            let dv = z_step[d] - z_prev[d];
                            if dv.is_finite() {
                                sq += dv * dv;
                            }
                        }
                        let mag = sq.sqrt();
                        if mag.is_finite() {
                            score += w_explore * change_alpha * mag;
                        }
                    }
                    if trusted_k != k && t + 1 >= trusted_k {
                        // Budget exhausted at this step — the rest of
                        // the rollout is noise; don't score it.
                        break;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_row = row;
                    best_trusted_k = trusted_k;
                }
            }
            // Queue only the trusted prefix: confident rollouts commit
            // deep, uncertain ones come back to the planner early.
            for t in 0..best_trusted_k {
                for inner_idx in 0..inner_iters {
                    let act = all_actions[best_row * k * inner_iters + t * inner_iters + inner_idx];
                    let sequence_idx = best_row * k * inner_iters + t * inner_iters + inner_idx;
                    let parameters =
                        if self.action_parameter_masks[lane_idx * MAX_ACTION_DIM + act as usize] {
                            let offset = sequence_idx * ACTION_PARAMETER_DIM;
                            Some([
                                all_action_parameters[offset],
                                all_action_parameters[offset + 1],
                            ])
                        } else {
                            None
                        };
                    self.planner_queue[lane_idx].push_back(PlannedAction {
                        action: act,
                        parameters,
                    });
                }
            }
            // BC-from-planner: push the (s, planner-chosen a) pair into
            // sil_buffer so the policy can learn to imitate the planner.
            // s is the lane's current latent's observation; a is the
            // first action of the committed trajectory. SIL's existing
            // CE update mechanism then trains the policy toward this
            // (s, a) with synthetic positive R, closing the policy-
            // planner gap.
            if self.config.bc_planner_synthetic_r > 0.0 && self.config.use_sil {
                let lane = &self.lanes[lane_idx];
                if let Some(last) = lane.buffer.last() {
                    // First executed action of the committed rollout.
                    // Layout is [row, k, inner_iters] — indexing with
                    // `best_row * k` landed inside a DIFFERENT row's
                    // sequence whenever inner_iters > 1 (always true in
                    // option-aware planning), feeding SIL an action the
                    // planner never chose.
                    let action_idx = all_actions[best_row * k * inner_iters] as usize;
                    let r = self.config.bc_planner_synthetic_r;
                    // Pairing note: `last` is the most recent record,
                    // whose latent IS the state the planner just
                    // planned from — the (s, a) pair is correct as-is.
                    let sample = SilSample {
                        obs: last.observation.clone(),
                        z: last.latent.clone(),
                        action_idx,
                        action_mask: self.action_masks
                            [lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM]
                            .to_vec(),
                        r_to_go: r,
                        v_at_collect: 0.0,
                        event_success: false,
                        env_id: lane.adapter.id(),
                    };
                    append_sil_samples(
                        &mut self.sil_buffer,
                        std::iter::once(sample),
                        self.config.sil_buffer_capacity,
                    );
                }
            }
        }
    }

    /// MCTS planner — replaces random-shooting at `plan_and_queue` time
    /// when `planner_use_mcts = true`. Per-lane tree built fresh each
    /// call; runs `mcts_simulations` expansions; queues the most-visited
    /// root path (depth `planner_horizon`).
    fn plan_and_queue_mcts<R: Rng>(&mut self, num_actions: usize, _rng: &mut R) {
        use crate::mcts::McTree;

        let n = self.lanes.len();
        let n_sim = self.config.mcts_simulations.max(1);
        let c_puct = self.config.mcts_c_puct;
        let horizon = self.config.planner_horizon.max(1);
        let ld = self.config.latent_dim;
        let num_actions_eff = num_actions.clamp(1, MAX_ACTION_DIM);

        // Periodic WM-weight sync (shared cadence with random planner).
        if self.planner_calls_since_refresh == 0
            || self.planner_calls_since_refresh >= self.config.planner_refresh_interval
        {
            self.refresh_wm_planner_weights();
            self.planner_calls_since_refresh = 0;
        }
        self.planner_calls_since_refresh += 1;

        // Identify active lanes (queue empty + has buffer data).
        let mut lane_active = vec![false; n];
        let mut lane_valid: Vec<Vec<u8>> = vec![Vec::new(); n];
        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            if !self.planner_queue[lane_idx].is_empty() {
                continue;
            }
            if lane.buffer.last().is_none() {
                continue;
            }
            lane_active[lane_idx] = true;
            let mask_row =
                &self.action_masks[lane_idx * MAX_ACTION_DIM..(lane_idx + 1) * MAX_ACTION_DIM];
            let valid: Vec<u8> = (0..num_actions_eff)
                .filter(|&j| mask_row[j] >= 0.5)
                .map(|j| j as u8)
                .collect();
            lane_valid[lane_idx] = if valid.is_empty() {
                (0..num_actions_eff as u8).collect()
            } else {
                valid
            };
        }
        if !lane_active.iter().any(|&b| b) {
            return;
        }

        // Build per-lane trees seeded with the current latent.
        let mut trees: Vec<Option<McTree>> = (0..n)
            .map(|lane_idx| {
                if lane_active[lane_idx] {
                    let z0 = self.lanes[lane_idx].buffer.last().unwrap().latent.clone();
                    Some(McTree::new(z0, lane_valid[lane_idx].clone()))
                } else {
                    None
                }
            })
            .collect();

        // Scratch for the n_lanes-sized WM dispatch (z in, action in,
        // z_next out). Sized once per call.
        let mut z_in = vec![0.0f32; n * ld];
        let mut a_in = vec![0.0f32; n * WM_ACTION_DIM];
        let mut z_out = vec![0.0f32; n * ld];

        for _sim in 0..n_sim {
            // 1) Select per lane: descend tree to find a node with an
            //    unexpanded valid action. Skip lanes that have no
            //    expansion to do (fully expanded subtree at depth bound).
            let mut leaves: Vec<Option<(Vec<usize>, u8)>> = vec![None; n];
            for lane_idx in 0..n {
                if let Some(ref tree) = trees[lane_idx] {
                    let (path, action_opt) = tree.select(c_puct);
                    if let Some(action) = action_opt {
                        leaves[lane_idx] = Some((path, action));
                    }
                }
            }
            let any_active = leaves.iter().any(|x| x.is_some());
            if !any_active {
                break;
            }

            // 2) Build batched WM input (one row per active lane;
            //    inactive lane rows are zeros — output ignored).
            z_in.fill(0.0);
            a_in.fill(0.0);
            for lane_idx in 0..n {
                if let Some((ref path, action)) = leaves[lane_idx] {
                    let tree = trees[lane_idx].as_ref().unwrap();
                    let parent_idx = *path.last().unwrap();
                    let parent_z = &tree.nodes[parent_idx].z;
                    z_in[lane_idx * ld..(lane_idx + 1) * ld].copy_from_slice(parent_z);
                    if (action as usize) < MAX_ACTION_DIM {
                        a_in[lane_idx * WM_ACTION_DIM + action as usize] = 1.0;
                    }
                }
            }
            // 3) WM forward
            let wm_mcts = self.wm_mcts_session.as_mut().unwrap();
            wm_mcts.set_input("z", &z_in);
            wm_mcts.set_input("action", &a_in);
            wm_mcts.step();
            wm_mcts.wait();
            wm_mcts.read_output_by_index(0, &mut z_out);

            // 4) Per-lane: add child node, compute novelty score, backup.
            let goal_alpha = self.config.planner_goal_alpha;
            for lane_idx in 0..n {
                if let Some((path, action)) = leaves[lane_idx].take() {
                    let child_z = z_out[lane_idx * ld..(lane_idx + 1) * ld].to_vec();
                    let count = self.lanes[lane_idx].buffer.visit_count(&child_z);
                    let mut value = 1.0 / ((count as f32 + 1.0).sqrt());
                    if goal_alpha > 0.0 || self.config.planner_subgoal_alpha > 0.0 {
                        let goal_key = if self.config.goal_states_cross_game {
                            0u32
                        } else {
                            self.lanes[lane_idx].adapter.id()
                        };
                        if goal_alpha > 0.0 {
                            if let Some(gq) = self.goal_states.get(&goal_key) {
                                if !gq.is_empty() {
                                    value += goal_alpha * max_goal_similarity(&child_z, gq);
                                }
                            }
                        }
                        if self.config.planner_subgoal_alpha > 0.0 {
                            if let Some(cl) = self.subgoal_centroids.get(&goal_key) {
                                if !cl.is_empty() {
                                    value += self.config.planner_subgoal_alpha
                                        * max_centroid_similarity(&child_z, cl);
                                }
                            }
                        }
                    }
                    let parent_idx = *path.last().unwrap();
                    let tree = trees[lane_idx].as_mut().unwrap();
                    let child_valid = lane_valid[lane_idx].clone();
                    let child_idx = tree.add_child(parent_idx, action, child_z, child_valid);
                    let mut full_path = path;
                    full_path.push(child_idx);
                    tree.backup(&full_path, value);
                }
            }
        }

        // 5) Extract the most-visited path from each tree (up to `horizon`
        //    actions). Queue them so the policy plays them like the
        //    random-shooting planner's output.
        for lane_idx in 0..n {
            if let Some(ref tree) = trees[lane_idx] {
                let mut cur = 0usize;
                for _ in 0..horizon {
                    let node = &tree.nodes[cur];
                    let mut best_a: Option<u8> = None;
                    let mut best_n: u32 = 0;
                    for &a in &node.valid {
                        let ci = node.children[a as usize];
                        if ci == usize::MAX {
                            continue;
                        }
                        let cn = tree.nodes[ci].n;
                        if cn > best_n {
                            best_n = cn;
                            best_a = Some(a);
                        }
                    }
                    match best_a {
                        Some(a) => {
                            self.planner_queue[lane_idx].push_back(PlannedAction {
                                action: a as u32,
                                parameters: None,
                            });
                            cur = node.children[a as usize];
                        }
                        None => break,
                    }
                }
            }
        }
    }

    /// Copy WM + policy parameters from the live training sessions to
    /// the planner's forward-only sessions. Cheap relative to a
    /// planning call — only happens at `planner_refresh_interval`
    /// cadence. Six WM params + three policy params = nine read+write
    /// pairs.
    fn refresh_wm_planner_weights(&mut self) {
        if self.wm_planner_session.is_none() {
            return;
        }
        let ld = self.config.latent_dim;
        let hd = self.config.hidden_dim;
        // WM params (graph: world_model.*). When wm_stochastic is on,
        // the sigma_proj is appended (one more no-bias linear).
        let mut wm_params: Vec<(&str, usize)> = vec![
            ("world_model.z_proj.weight", ld * hd),
            ("world_model.z_proj.bias", hd),
            ("world_model.a_proj.weight", WM_ACTION_DIM * hd),
            ("world_model.fc2.weight", hd * hd),
            ("world_model.fc2.bias", hd),
            ("world_model.fc_out.weight", hd * ld),
        ];
        if self.config.wm_stochastic {
            wm_params.push(("world_model.sigma_proj.weight", hd * ld));
        }
        for (name, n_elem) in wm_params.iter() {
            let buf = &mut self.planner_param_buf[..*n_elem];
            self.wm_session.read_param(name, buf);
            self.wm_planner_session
                .as_mut()
                .unwrap()
                .set_parameter(name, buf);
            // Mirror to the MCTS session if present (same param names).
            if let Some(ref mut mcts_sess) = self.wm_mcts_session {
                mcts_sess.set_parameter(name, buf);
            }
        }
        // Policy params (graph: policy.fc1.weight/bias, policy.fc2.weight).
        // policy.fc2 has no bias.
        if let Some(ref mut policy_planner) = self.policy_planner_session {
            let policy_params: [(&str, usize); 3] = [
                ("policy.fc1.weight", ld * hd),
                ("policy.fc1.bias", hd),
                ("policy.fc2.weight", hd * MAX_ACTION_DIM),
            ];
            for (name, n_elem) in policy_params.iter() {
                let buf = &mut self.planner_param_buf[..*n_elem];
                self.policy_session.read_param(name, buf);
                policy_planner.set_parameter(name, buf);
            }
            // Option bias for option-conditional policy planner
            // (only present when num_options > 1; it adds an additive
            // bias selected by option_onehot).
            if self.config.num_options >= 2 {
                let n_elem = self.config.num_options * MAX_ACTION_DIM;
                let buf = &mut self.planner_param_buf[..n_elem];
                self.policy_session
                    .read_param("policy.option_bias.weight", buf);
                policy_planner.set_parameter("policy.option_bias.weight", buf);
            }
        }
        // Option-policy params (when option_planner exists).
        if let Some(ref mut option_planner) = self.option_planner_session {
            if let Some(ref mut option_sess) = self.option_session {
                let no = self.config.num_options;
                let opt_params: [(&str, usize); 3] = [
                    ("option.trunk.weight", ld * hd),
                    ("option.trunk.bias", hd),
                    ("option.head.weight", hd * no),
                ];
                for (name, n_elem) in opt_params.iter() {
                    let buf = &mut self.planner_param_buf[..*n_elem];
                    option_sess.read_param(name, buf);
                    option_planner.set_parameter(name, buf);
                }
            }
        }
        // Value-head params (when wm_planner has a value branch).
        // Source: `wm_session` (the value head lives inside the WM
        // graph, sharing the encoder). Sink: wm_planner_session /
        // wm_mcts_session, where the value head computes V(z_next)
        // for each rolled-out step.
        if self.wm_planner_has_value_head {
            let vh = if self.config.value_head_hidden_dim == 0 {
                hd
            } else {
                self.config.value_head_hidden_dim
            };
            let vh_params: [(&str, usize); 3] = [
                ("value_head.fc1.weight", ld * vh),
                ("value_head.fc1.bias", vh),
                ("value_head.fc2.weight", vh),
            ];
            for (name, n_elem) in vh_params.iter() {
                let buf = &mut self.planner_param_buf[..*n_elem];
                self.wm_session.read_param(name, buf);
                if let Some(ref mut wm_p) = self.wm_planner_session {
                    wm_p.set_parameter(name, buf);
                }
                if let Some(ref mut mcts_sess) = self.wm_mcts_session {
                    mcts_sess.set_parameter(name, buf);
                }
            }
        }
    }

    /// Total number of actions currently queued by the planner
    /// across all lanes (diagnostic).
    pub fn planner_queue_len(&self) -> usize {
        self.planner_queue.iter().map(|q| q.len()).sum()
    }

    /// External-API push: append a sequence of action indices to a
    /// specific lane's planner queue. Lets the harness inject learned
    /// skills/macros that the agent commits to instead of running a
    /// fresh plan_and_queue. Invalid `lane` is silently ignored.
    pub fn queue_actions(&mut self, lane: usize, actions: &[u32]) {
        if lane >= self.planner_queue.len() {
            return;
        }
        for &a in actions {
            if a as usize >= MAX_ACTION_DIM {
                continue;
            }
            self.planner_queue[lane].push_back(PlannedAction {
                action: a,
                parameters: None,
            });
        }
    }

    /// External-API push for explicit `(action, [x, y])` decisions. Values are
    /// sanitized to the same normalized range as `set_action_parameters`.
    pub fn queue_parameterized_actions(
        &mut self,
        lane: usize,
        actions: &[(u32, [f32; ACTION_PARAMETER_DIM])],
    ) {
        if lane >= self.planner_queue.len() {
            return;
        }
        for &(action, mut parameters) in actions {
            if action as usize >= MAX_ACTION_DIM {
                continue;
            }
            for value in &mut parameters {
                *value = if value.is_finite() {
                    value.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
            }
            self.planner_queue[lane].push_back(PlannedAction {
                action,
                parameters: Some(parameters),
            });
        }
    }

    /// Current M7 confidence ∈ [0, 1]. Zero until warmup is met;
    /// ramps linearly to 1 over `approach_confidence_saturation`
    /// episodes. One once warmup is met if saturation is zero, and always
    /// zero when M7 is disabled.
    pub fn approach_confidence(&self) -> f32 {
        let Some(state) = self.approach_state.as_ref() else {
            return 0.0;
        };
        let warmup = self.config.approach_warmup_episodes;
        if state.episodes_seen < warmup {
            return 0.0;
        }
        let saturation = self.config.approach_confidence_saturation;
        if saturation == 0 {
            return 1.0;
        }
        let past_warmup = (state.episodes_seen - warmup) as f32;
        (past_warmup / saturation as f32).clamp(0.0, 1.0)
    }

    /// Snapshot the most recent `n` transitions of `lane_idx` (oldest
    /// first), cloning fields needed for offline diagnostics. Returns an
    /// empty vec on out-of-range lane.
    ///
    /// Diagnostic-only — used by Python harnesses to inspect rollouts
    /// (WM pred error trajectory, V vs. discounted return, action
    /// distribution near terminal events). Cost is O(n × latent_dim);
    /// callers should pass small n (≤ 1024).
    pub fn recent_transitions(&self, lane_idx: usize, n: usize) -> Vec<crate::buffer::Transition> {
        let Some(lane) = self.lanes.get(lane_idx) else {
            return Vec::new();
        };
        lane.buffer.recent_window(n).into_iter().cloned().collect()
    }

    /// Per-lane diagnostics, one entry per lane in lane order.
    ///
    /// Global (batch-shared) signals — `loss_world_model`,
    /// `loss_policy`, `loss_replay`, `repr_drift` — are broadcast to every
    /// lane's row. Lane-specific fields (`env_id`, `reward_*`,
    /// `policy_entropy` and `buffer_len`) vary per row.
    /// Number of (s, a, R_to_go, V) samples currently in the SIL buffer.
    /// Useful for verifying SIL is actually populating during training.
    pub fn sil_buffer_size(&self) -> usize {
        self.sil_buffer.len()
    }

    /// Number of SIL updates fired (passed all guards including the
    /// positive-advantage check producing at least one active row).
    /// (P4) Number of planner-queue clears triggered by surprise
    /// spikes (replan_surprise_mult). Diagnostic.
    pub fn replan_clears_count(&self) -> u64 {
        self.replan_clears
    }

    /// (P2) Number of surprise-replay dispatches fired, and the
    /// current ring occupancy. Diagnostic.
    pub fn surprise_replay_stats(&self) -> (u64, usize) {
        (self.surprise_replays, self.surprise_ring_frames.len())
    }

    pub fn sil_updates_fired_count(&self) -> u64 {
        self.sil_updates_fired
    }

    /// Number of times maybe_run_sil_update was called (regardless of
    /// whether it actually ran).
    pub fn sil_updates_attempted_count(&self) -> u64 {
        self.sil_updates_attempted
    }

    /// Number of active rows (positive advantage) in the most recent
    /// SIL update. 0 if SIL never fired.
    pub fn sil_last_active(&self) -> u32 {
        self.sil_last_active_rows
    }

    /// L1 change in policy.fc1.weight (first 32 entries) caused by
    /// the most recent SIL update. Useful diagnostic to confirm the
    /// SIL session step is actually moving parameters.
    pub fn sil_last_param_change_value(&self) -> f32 {
        self.sil_last_param_change
    }

    /// Current SIL "successful episode" baseline for lane 0's task. Returns
    /// 0 before that task completes its first eligible episode. Kept as the
    /// single-value compatibility diagnostic; multi-task callers can use
    /// [`Self::sil_baseline_value_for_env`].
    pub fn sil_baseline_value(&self) -> f32 {
        self.lanes
            .first()
            .and_then(|lane| self.sil_baselines.get(&lane.adapter.id()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Current SIL admission baseline for a specific task id.
    pub fn sil_baseline_value_for_env(&self, env_id: u32) -> Option<f32> {
        self.sil_baselines.get(&env_id).copied()
    }

    pub fn diagnostics(&self) -> Vec<Diagnostics> {
        self.lanes
            .iter()
            .enumerate()
            .map(|(lane_idx, lane)| {
                let goal_diversity =
                    if self.option_session.is_some() && self.config.num_options >= 2 {
                        let n_opt = self.config.num_options;
                        let od = self.goal_table.len() / n_opt.max(1);
                        let mut sum = 0.0f32;
                        let mut pairs = 0usize;
                        for a in 0..n_opt {
                            for b in (a + 1)..n_opt {
                                let ga = &self.goal_table[a * od..(a + 1) * od];
                                let gb = &self.goal_table[b * od..(b + 1) * od];
                                let d2: f32 =
                                    ga.iter().zip(gb.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                                sum += d2.sqrt();
                                pairs += 1;
                            }
                        }
                        if pairs == 0 { 0.0 } else { sum / pairs as f32 }
                    } else {
                        0.0
                    };

                // L1 goal distance: ‖last_latent − goal‖.
                let goal_distance = if self.option_session.is_some() {
                    if let Some(prev) = lane.buffer.last() {
                        prev.latent
                            .iter()
                            .zip(lane.option_goal.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                Diagnostics {
                    step: self.step_count,
                    env_id: lane.adapter.id(),
                    loss_world_model: self.last_wm_loss,
                    loss_reconstruction: self.last_recon_loss,
                    loss_policy: self.last_policy_loss,
                    loss_replay: self.last_replay_loss,
                    reward_mean: lane.last_reward,
                    reward_surprise: lane.last_surprise,
                    reward_novelty: lane.last_novelty,
                    reward_homeo: lane.last_homeo,
                    reward_order: lane.last_order,
                    policy_entropy: lane.last_entropy,
                    repr_drift: self.last_drift,
                    buffer_len: lane.buffer.len(),
                    current_option: lane.current_option,
                    option_return: lane.option_return,
                    goal_distance,
                    goal_diversity,
                    r_hat: lane.last_r_hat,
                    outcome_baseline: lane.outcome_baseline,
                    outcome_loss: self
                        .outcome_head
                        .as_ref()
                        .map(|h| h.last_loss)
                        .unwrap_or(0.0),
                    approach_distance: self
                        .approach_distances
                        .get(lane_idx)
                        .copied()
                        .unwrap_or(0.0),
                    approach_buffer_fill: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.buffer.len())
                        .unwrap_or(0),
                    approach_centroid_drift: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.last_centroid_drift)
                        .unwrap_or(0.0),
                    approach_centroid_age: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.centroid_age)
                        .unwrap_or(0),
                    approach_confidence: self.approach_confidence(),
                    rnd_mse: self.last_rnd_mse,
                }
            })
            .collect()
    }
}

/// Compatibility for one shared policy graph. Discrete lanes may expose
/// different categorical widths because masking handles padded heads;
/// continuous lanes must agree on both live dimension and exploration scale.
fn kinds_match(a: ActionKind, b: ActionKind) -> bool {
    match (a, b) {
        (ActionKind::Discrete { .. }, ActionKind::Discrete { .. }) => true,
        (
            ActionKind::Continuous {
                dim: left_dim,
                scale: left_scale,
            },
            ActionKind::Continuous {
                dim: right_dim,
                scale: right_scale,
            },
        ) => left_dim == right_dim && left_scale.to_bits() == right_scale.to_bits(),
        _ => false,
    }
}

/// Deterministic task code. In orthogonal mode, the first `dim` ids receive
/// one-hot directions, so training one common task does not update every column
/// of the trainable task projection and poison a later task's adapter bias.
/// Larger ids fall back to a normalized dense hash. Compatibility mode returns
/// the historical unnormalized dense hash for every id.
fn embedding_for(env_id: u32, dim: usize, orthogonal: bool) -> Vec<f32> {
    if orthogonal && (env_id as usize) < dim {
        let mut embedding = vec![0.0; dim];
        embedding[env_id as usize] = 1.0;
        return embedding;
    }
    use std::f32::consts::PI;
    let mut embedding = (0..dim)
        .map(|i| {
            let h =
                ((env_id as f64 + i as f64 * 17.0 + 1.0) * 0.618_033_988_749_895).fract() as f32;
            (h * PI * 2.0).sin() * 0.5
        })
        .collect::<Vec<_>>();
    if !orthogonal {
        return embedding;
    }
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

/// Initialize all parameters with Xavier (Glorot) initialization.
#[allow(clippy::pattern_type_mismatch)]
fn init_parameters(session: &mut Session) {
    let params: Vec<(String, usize)> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(name, buf_ref)| {
            let size_bytes = session.plan().buffers[buf_ref.0 as usize];
            (name.clone(), size_bytes / 4)
        })
        .collect();

    for (i, (name, num_elements)) in params.iter().enumerate() {
        let num_elements = *num_elements;
        if name.ends_with(".bias") || name.ends_with(".weight") && num_elements <= 1 {
            let data = vec![0.0f32; num_elements];
            session.set_parameter(name, &data);
        } else if name.contains("norm") {
            let data = vec![1.0f32; num_elements];
            session.set_parameter(name, &data);
        } else {
            // xavier_init returns fan*fan elements; pad/truncate to exactly
            // num_elements so set_parameter overwrites the whole buffer
            // (otherwise tail bytes hold whatever was previously there).
            let fan = (num_elements as f32).sqrt().max(1.0) as usize;
            let mut data = xavier_init(fan, fan, i as u64 * 7919);
            data.resize(num_elements, 0.0);
            session.set_parameter(name, &data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{ExperienceBuffer, Transition};

    #[test]
    fn task_code_modes_preserve_dense_compatibility_and_isolate_common_ids() {
        for env_id in 0..TASK_DIM as u32 {
            let embedding = embedding_for(env_id, TASK_DIM, true);
            assert_eq!(embedding[env_id as usize], 1.0);
            assert_eq!(embedding.iter().filter(|&&value| value != 0.0).count(), 1);
        }
        let fallback = embedding_for(TASK_DIM as u32 + 7, TASK_DIM, true);
        let norm = fallback
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");

        let env_id = 3;
        let historical = (0..TASK_DIM)
            .map(|i| {
                let h = ((env_id as f64 + i as f64 * 17.0 + 1.0) * 0.618_033_988_749_895).fract()
                    as f32;
                (h * std::f32::consts::PI * 2.0).sin() * 0.5
            })
            .collect::<Vec<_>>();
        assert_eq!(embedding_for(env_id, TASK_DIM, false), historical);
    }

    /// Build an ExperienceBuffer with one transition per tuple
    /// `(reward, value, env_boundary)`. Latent and other fields are
    /// filled with zeros; they don't matter for the TD-target math.
    fn mk_buffer(spec: &[(f32, f32, bool)]) -> ExperienceBuffer {
        let mut buf = ExperienceBuffer::new(64, 0.5);
        for &(reward, value, env_boundary) in spec {
            buf.push(Transition {
                observation: vec![0.0; 4],
                latent: vec![0.0; 4],
                action: vec![0.0; MAX_ACTION_DIM],
                action_parameters: vec![0.0; ACTION_PARAMETER_DIM],
                action_mask: vec![1.0; MAX_ACTION_DIM],
                reward,
                pred_error: 0.0,
                value,
                prob_taken: 1.0,
                logits_at_action: Vec::new(),
                option_idx: 0,
                env_id: 0,
                env_boundary,
                episode_return: 0.0,
                episode_complete: false,
            });
        }
        buf
    }

    #[test]
    fn event_filtered_sil_imitates_negative_return_successes() {
        assert_eq!(sil_imitation_advantage(-20.0, 0.0, false, true, 2.0), 0.0);
        assert_eq!(sil_imitation_advantage(-20.0, 0.0, true, false, 2.0), 0.0);
        assert_eq!(sil_imitation_advantage(-20.0, 0.0, true, true, 2.0), 1.0);
        assert_eq!(sil_imitation_advantage(5.0, 1.0, true, true, 2.0), 2.0);
    }

    #[test]
    fn greedy_policy_actions_respect_live_discrete_width_and_continuous_dim() {
        let discrete = greedy_policy_action(ActionKind::Discrete { n: 3 }, &[0.1, 0.9, 0.2, 99.0]);
        assert!(matches!(discrete, Action::Discrete(1)));

        let continuous = greedy_policy_action(
            ActionKind::Continuous { dim: 2, scale: 0.5 },
            &[0.25, -0.75, 8.0],
        );
        assert!(matches!(continuous, Action::Continuous(ref values) if values == &[0.25, -0.75]));
    }

    #[test]
    fn event_filtered_sil_baseline_compares_successes_only() {
        let mut baseline = 0.0;
        let mut initialized = false;
        assert!(!admit_sil_episode(
            &mut baseline,
            &mut initialized,
            -200.0,
            false,
            true,
            0.99,
        ));
        assert!(!initialized, "failed episodes must not seed event baseline");
        assert!(admit_sil_episode(
            &mut baseline,
            &mut initialized,
            -80.0,
            true,
            true,
            0.99,
        ));
        assert_eq!(baseline, -80.0);
        assert!(!admit_sil_episode(
            &mut baseline,
            &mut initialized,
            -100.0,
            true,
            true,
            0.99,
        ));
        assert!(baseline < -80.0 && baseline > -81.0);
        assert!(admit_sil_episode(
            &mut baseline,
            &mut initialized,
            -60.0,
            true,
            true,
            0.99,
        ));
    }

    #[test]
    fn sil_admission_baselines_are_independent_per_task() {
        let mut baselines = hashbrown::HashMap::new();
        assert!(admit_sil_episode_for_env(
            &mut baselines,
            0,
            500.0,
            true,
            true,
            0.99,
        ));
        assert!(admit_sil_episode_for_env(
            &mut baselines,
            1,
            -80.0,
            true,
            true,
            0.99,
        ));
        assert_eq!(baselines.get(&0), Some(&500.0));
        assert_eq!(baselines.get(&1), Some(&-80.0));

        let mut failed_only = hashbrown::HashMap::new();
        assert!(!admit_sil_episode_for_env(
            &mut failed_only,
            7,
            -200.0,
            false,
            true,
            0.99,
        ));
        assert!(!failed_only.contains_key(&7));
    }

    #[test]
    fn sil_sampling_balances_tasks_not_transition_volume() {
        let sample = |env_id| SilSample {
            obs: vec![0.0; 4],
            z: vec![0.0; 4],
            action_idx: 0,
            action_mask: vec![1.0; MAX_ACTION_DIM],
            r_to_go: 1.0,
            v_at_collect: 0.0,
            event_success: true,
            env_id,
        };
        let mut buffer = std::collections::VecDeque::new();
        buffer.extend((0..100).map(|_| sample(0)));
        buffer.push_back(sample(1));

        let indices = balanced_sil_sample_indices(&buffer, 20, 42, &[0, 1].into_iter().collect());
        let env_ids = indices
            .iter()
            .map(|&index| buffer[index].env_id)
            .collect::<Vec<_>>();
        assert_eq!(env_ids.iter().filter(|&&env_id| env_id == 0).count(), 10);
        assert_eq!(env_ids.iter().filter(|&&env_id| env_id == 1).count(), 10);

        let active_only = balanced_sil_sample_indices(&buffer, 20, 42, &[1].into_iter().collect());
        assert!(
            active_only.iter().all(|&index| buffer[index].env_id == 1),
            "inactive task labels must not train the current lanes"
        );
    }

    #[test]
    fn sil_capacity_retains_inactive_tasks_and_single_task_fifo() {
        let sample = |env_id, marker| SilSample {
            obs: vec![marker],
            z: vec![0.0; 4],
            action_idx: 0,
            action_mask: vec![1.0; MAX_ACTION_DIM],
            r_to_go: 1.0,
            v_at_collect: 0.0,
            event_success: true,
            env_id,
        };

        let mut buffer = std::collections::VecDeque::new();
        append_sil_samples(&mut buffer, (0..10).map(|i| sample(0, i as f32)), 10);
        append_sil_samples(&mut buffer, (10..20).map(|i| sample(0, i as f32)), 10);
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.front().expect("sample").obs, vec![10.0]);
        assert_eq!(buffer.back().expect("sample").obs, vec![19.0]);

        append_sil_samples(
            &mut buffer,
            (0..10).map(|i| sample(1, 100.0 + i as f32)),
            10,
        );
        assert_eq!(buffer.iter().filter(|sample| sample.env_id == 0).count(), 5);
        assert_eq!(buffer.iter().filter(|sample| sample.env_id == 1).count(), 5);
        assert_eq!(
            buffer
                .iter()
                .filter(|sample| sample.env_id == 0)
                .map(|sample| sample.obs[0])
                .collect::<Vec<_>>(),
            vec![15.0, 16.0, 17.0, 18.0, 19.0]
        );
    }

    #[test]
    fn event_sil_window_is_bounded_and_does_not_cross_environment_boundaries() {
        let mut buffer = mk_buffer(&[
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (1.0, 0.0, false),
        ]);
        for idx in 0..buffer.len() {
            buffer.get_mut(idx).observation[0] = idx as f32;
            buffer.get_mut(idx).action[idx] = 1.0;
            buffer.get_mut(idx).env_id = 7;
        }

        let samples = sil_samples_from_recent_trajectory(&buffer, 3, true, 100.0);

        assert_eq!(samples.len(), 3);
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.action_idx)
                .collect::<Vec<_>>(),
            vec![4, 3, 2]
        );
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.obs[0])
                .collect::<Vec<_>>(),
            vec![3.0, 2.0, 1.0]
        );
        assert!(samples.iter().all(|sample| sample.r_to_go == 1.0));
        assert!(samples.iter().all(|sample| sample.event_success));
        assert!(samples.iter().all(|sample| sample.env_id == 7));

        buffer.get_mut(3).env_boundary = true;
        let bounded = sil_samples_from_recent_trajectory(&buffer, 10, true, 100.0);
        assert_eq!(bounded.len(), 1, "event replay must stop at a real reset");
        assert_eq!(bounded[0].action_idx, 4);
    }

    #[test]
    fn standard_sil_keeps_first_episode_warmup() {
        let mut baseline = 0.0;
        let mut initialized = false;
        assert!(!admit_sil_episode(
            &mut baseline,
            &mut initialized,
            10.0,
            false,
            false,
            0.99,
        ));
        assert!(initialized);
        assert!(admit_sil_episode(
            &mut baseline,
            &mut initialized,
            11.0,
            false,
            false,
            0.99,
        ));
    }

    #[test]
    fn wm_action_token_keeps_policy_identity_and_appends_parameters() {
        let mut base = vec![0.0; MAX_ACTION_DIM];
        base[5] = 1.0;
        let mut out = vec![f32::NAN; WM_ACTION_DIM];

        compose_wm_action_token(&base, &[-0.75, 0.5], &mut out);

        assert_eq!(&out[..MAX_ACTION_DIM], base.as_slice());
        assert_eq!(&out[MAX_ACTION_DIM..], &[-0.75, 0.5]);

        compose_wm_action_token(&base, &[], &mut out);
        assert_eq!(&out[..MAX_ACTION_DIM], base.as_slice());
        assert_eq!(&out[MAX_ACTION_DIM..], &[0.0, 0.0]);
    }

    #[test]
    fn explicit_terminal_event_does_not_require_positive_reward() {
        assert!(is_task_event(Some(true), -1.0));
        assert!(!is_task_event(Some(false), 1.0));
        assert!(is_task_event(None, 1.0));
        assert!(!is_task_event(None, -1.0));
    }

    #[test]
    fn grpo_advantages_normalize_within_task_groups() {
        let mut advantages = vec![1.0, 3.0, 100.0, 104.0];
        let active = vec![true; 4];
        let groups = vec![7, 7, 9, 9];
        normalize_active_advantages(&mut advantages, &active, Some(&groups), true);
        for (actual, expected) in advantages.iter().zip([-1.0, 1.0, -1.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-5, "{advantages:?}");
        }
    }

    #[test]
    fn stored_action_mask_is_restored_or_defaults_to_all_actions() {
        let mut dst = vec![0.0; 4];
        restore_action_mask(&mut dst, &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(dst, [1.0, 0.0, 1.0, 0.0]);

        restore_action_mask(&mut dst, &[]);
        assert_eq!(dst, [1.0; 4]);
    }

    #[test]
    fn grpo_advantages_leave_singletons_and_inactive_rows_unchanged() {
        let mut advantages = vec![5.0, 99.0, 7.0];
        let active = vec![true, false, true];
        let groups = vec![1, 1, 2];
        normalize_active_advantages(&mut advantages, &active, Some(&groups), true);
        assert_eq!(advantages, vec![5.0, 99.0, 7.0]);
    }

    #[test]
    fn episode_grpo_mean_centers_without_rescaling() {
        let mut advantages = vec![2.0, 6.0];
        let active = vec![true, true];
        let groups = vec![3, 3];
        normalize_active_advantages(&mut advantages, &active, Some(&groups), false);
        assert_eq!(advantages, vec![-2.0, 2.0]);
    }

    #[test]
    fn td_n_step_return_no_bootstrap_matches_plain_sum() {
        // Three rewards, no bootstrap. Expect ret = Σ γ^k r_k.
        let buf = mk_buffer(&[(1.0, 0.0, false), (2.0, 0.0, false), (4.0, 0.0, false)]);
        let gamma = 0.9;
        let (ret, gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, false, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!((gk - 0.9f32.powi(3)).abs() < 1e-5, "gk: {}", gk);
        assert!(!term, "should not be terminated");
    }

    #[test]
    fn td_n_step_return_bootstrap_adds_gamma_n_v() {
        // Same three rewards + a 4th transition carrying value=10.
        // With bootstrap, expect ret = Σ γ^k r_k + γ^3 · 10.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (4.0, 0.0, false),
            (99.0, 10.0, false), // reward ignored; only its .value is read for bootstrap
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0 + 0.729 * 10.0;
        assert!(
            (ret - expected).abs() < 1e-4,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!(!term);
    }

    #[test]
    fn td_n_step_return_termination_inside_window_suppresses_bootstrap() {
        // Transition at k=2 is env_boundary → window terminates at k=1,
        // bootstrap suppressed even though the 4th transition has value.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (999.0, 0.0, true), // boundary
            (99.0, 10.0, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        // Only k=0 and k=1 rewards included; no bootstrap.
        let expected = 1.0 + 0.9 * 2.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!(term, "should be terminated");
    }

    #[test]
    fn td_n_step_return_bootstrap_state_across_boundary_suppresses_bootstrap() {
        // Window fills cleanly (k=0..2 no boundary), but the bootstrap
        // transition at index 3 IS a boundary — means it's the first
        // transition of a new episode, not a continuation. Bootstrap
        // should be dropped.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (4.0, 0.0, false),
            (99.0, 10.0, true), // boundary at bootstrap point
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        // The WINDOW itself didn't terminate — we just have no bootstrap.
        assert!(!term);
    }

    #[test]
    fn td_n_step_return_bootstrap_clamp_bounds_runaway_value() {
        // Value head diverged to +1e6; clamp should keep target sane.
        let buf = mk_buffer(&[
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 1.0e6, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        // With clamp=100 on the bootstrap, ret = γ^3 · 100 = 72.9.
        let expected = 0.729 * 100.0;
        assert!(
            (ret - expected).abs() < 1e-3,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_bootstrap_ignores_nonfinite_value() {
        // NaN value at bootstrap position — should be skipped.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (3.0, 0.0, false),
            (99.0, f32::NAN, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_single_reward_bootstrap_is_td0() {
        // n_step=1 with bootstrap = classical TD(0) target.
        let buf = mk_buffer(&[(2.0, 0.0, false), (0.0, 5.0, false)]);
        let gamma = 0.95;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 1, gamma, true, 100.0);
        let expected = 2.0 + 0.95 * 5.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_gamma_zero_myopic() {
        // With γ=0, only the ripe reward counts — discount annihilates everything else.
        let buf = mk_buffer(&[
            (7.0, 0.0, false),
            (999.0, 0.0, false),
            (999.0, 0.0, false),
            (99.0, 999.0, false),
        ]);
        let (ret, gk, _term) = compute_td_n_step_return(&buf, 0, 3, 0.0, true, 100.0);
        // γ^0 · 7 + 0 · 999 + 0 · 999 + 0 · bootstrap = 7
        assert!((ret - 7.0).abs() < 1e-5, "ret: {}", ret);
        assert_eq!(gk, 0.0);
    }

    // ---- GAE advantage tests ----

    #[test]
    fn gae_lambda_zero_is_pure_td0() {
        // λ=0 → Â_t = δ_0 = r_0 + γ·V(s_1) − V(s_0).
        // ripe has reward=2, value=1; next has value=5.
        let buf = mk_buffer(&[
            (2.0, 1.0, false),
            (0.0, 5.0, false),
            (0.0, 0.0, false), // padding so buf_len > ripe+n_step
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.0, 100.0);
        let expected = 2.0 + 0.9 * 5.0 - 1.0;
        assert!(
            (adv - expected).abs() < 1e-5,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_lambda_one_equals_mc_minus_value() {
        // λ=1 → Â_t = Σ γ^k · δ_{t+k} = (Σ γ^k · r_{t+k}) + γ^n · V(s_{t+n}) − V(s_t).
        // Verify via telescoping.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),   // ripe: r=1, V=0.5
            (2.0, 0.3, false),   // t+1: r=2, V=0.3
            (4.0, 0.2, false),   // t+2: r=4, V=0.2
            (99.0, 10.0, false), // bootstrap slot: V=10
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, 1.0, 100.0);
        // MC return + γ^3·V_boot − V(s_ripe)
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0 + 0.729 * 10.0 - 0.5;
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_recursive_identity() {
        // Â_t = δ_t + γλ · Â_{t+1}. Compute twice — once for the
        // full window at ripe_idx=0, once for the shorter window at
        // ripe_idx=1 — and verify the identity numerically.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (4.0, 0.2, false),
            (0.0, 6.0, false),
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv_full = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        let adv_tail = compute_gae_advantage(&buf, 1, 2, gamma, lambda, 100.0);
        // δ_0 = 1 + 0.9·0.3 − 0.5 = 0.77
        let delta0 = 1.0 + 0.9 * 0.3 - 0.5;
        let expected = delta0 + gamma * lambda * adv_tail;
        assert!(
            (adv_full - expected).abs() < 1e-5,
            "adv_full: {} vs δ_0 + γλ·adv_tail = {}",
            adv_full,
            expected
        );
    }

    #[test]
    fn gae_termination_inside_window_stops_accumulation() {
        // Boundary at k=2 (index 2 is the start of a new episode)
        // → only δ's for k=0 and k=1 accumulate.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (999.0, 777.0, true), // boundary → don't accumulate this δ,
            //                     and its V is NOT read as V(s_{t+1}) for k=1
            (0.0, 0.0, false),
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        // δ_0: r=1, V(s_1)=0.3 (next has no boundary) → 1 + 0.9·0.3 − 0.5 = 0.77
        // δ_1: r=2, V(s_2) — but index 2 has env_boundary=true → V(s_2) treated as 0
        //      → 2 + 0.9·0 − 0.3 = 1.7
        // Stop (index 2 boundary also cuts accumulation for k≥2).
        // Â_0 = δ_0 + γλ · δ_1 = 0.77 + 0.9·0.95·1.7 = 0.77 + 1.4535 = 2.2235
        let d0 = 1.0 + 0.9 * 0.3 - 0.5;
        let d1 = 2.0 + 0.9 * 0.0 - 0.3;
        let expected = d0 + gamma * lambda * d1;
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_boundary_at_bootstrap_zeros_last_next_v() {
        // No boundary inside the window, but the n+1-th slot IS a
        // boundary → V(s_{ripe+n_step}) treated as 0 for the last δ.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (4.0, 0.2, false),
            (99.0, 777.0, true), // boundary → V at bootstrap slot → 0
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        let d0 = 1.0 + 0.9 * 0.3 - 0.5;
        let d1 = 2.0 + 0.9 * 0.2 - 0.3;
        // δ_2: V(s_3) treated as 0 (boundary) → 4 + 0.9·0 − 0.2 = 3.8
        let d2 = 4.0 + 0.9 * 0.0 - 0.2;
        let expected = d0 + gamma * lambda * (d1 + gamma * lambda * d2);
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_clamp_bounds_runaway_value() {
        // V=1e6 at ripe; clamp to ±100 before computing δ.
        let buf = mk_buffer(&[(0.0, 1e6, false), (0.0, 0.0, false), (0.0, 0.0, false)]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.5, 100.0);
        // δ_0 = 0 + 0.9·0 − 100 (V clamped) = −100. λ=0.5 has only 1 step, so Â=δ_0.
        assert!(
            (adv - (-100.0)).abs() < 1e-3,
            "adv: {} (expected −100)",
            adv
        );
    }

    #[test]
    fn gae_nonfinite_value_treated_as_zero() {
        // NaN at ripe's V — treated as 0.
        let buf = mk_buffer(&[(3.0, f32::NAN, false), (0.0, 2.0, false), (0.0, 0.0, false)]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.0, 100.0);
        // δ_0 = 3 + 0.9·2 − 0 = 4.8
        let expected = 3.0 + 0.9 * 2.0;
        assert!(
            (adv - expected).abs() < 1e-5,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_gamma_zero_is_single_step_residual() {
        // γ=0 → all later δ's have γλ=0 weight, and within each δ
        // the γ·V(s_{t+1}) term is also zero. Â_0 = r_0 − V(s_0).
        let buf = mk_buffer(&[
            (5.0, 1.0, false),
            (999.0, 999.0, false),
            (999.0, 999.0, false),
        ]);
        let adv = compute_gae_advantage(&buf, 0, 2, 0.0, 0.95, 100.0);
        assert!((adv - 4.0).abs() < 1e-5, "adv: {}", adv);
    }

    #[test]
    fn proximity_bonus_fires_on_positive_terminal() {
        // 5-step episode ending with reward +30 (landing). With
        // K=3 and bonus=2.0 and threshold=0.0, the last 3
        // *pre-terminal* transitions should each get +2.0; the terminal
        // step itself is unchanged, and earlier steps are unchanged.
        let mut buf = mk_buffer(&[
            (-0.3, 0.0, false), // step 0: should NOT receive bonus (outside K=3 lookback from idx=4)
            (-0.3, 0.0, false), // step 1: receives bonus (idx=4-3)
            (-0.3, 0.0, false), // step 2: receives bonus (idx=4-2)
            (-0.3, 0.0, false), // step 3: receives bonus (idx=4-1)
            (30.0, 0.0, false), // step 4: terminal — unchanged
        ]);
        apply_terminal_proximity_bonus(&mut buf, 3, 2.0, 0.0);
        assert!(
            (buf.get(0).reward - (-0.3)).abs() < 1e-5,
            "step 0 reward: {}",
            buf.get(0).reward
        );
        assert!(
            (buf.get(1).reward - (-0.3 + 2.0)).abs() < 1e-5,
            "step 1 reward: {}",
            buf.get(1).reward
        );
        assert!(
            (buf.get(2).reward - (-0.3 + 2.0)).abs() < 1e-5,
            "step 2 reward: {}",
            buf.get(2).reward
        );
        assert!(
            (buf.get(3).reward - (-0.3 + 2.0)).abs() < 1e-5,
            "step 3 reward: {}",
            buf.get(3).reward
        );
        assert!(
            (buf.get(4).reward - 30.0).abs() < 1e-5,
            "terminal reward: {}",
            buf.get(4).reward
        );
    }

    #[test]
    fn proximity_bonus_skips_negative_terminal() {
        // Same buffer shape but terminal is -100 (crash). Bonus must
        // NOT fire — the asymmetry is the whole point.
        let mut buf = mk_buffer(&[
            (-0.3, 0.0, false),
            (-0.3, 0.0, false),
            (-0.3, 0.0, false),
            (-0.3, 0.0, false),
            (-100.0, 0.0, false),
        ]);
        apply_terminal_proximity_bonus(&mut buf, 3, 2.0, 0.0);
        for i in 0..5 {
            let expected = if i < 4 { -0.3 } else { -100.0 };
            assert!(
                (buf.get(i).reward - expected).abs() < 1e-5,
                "step {} reward: {}",
                i,
                buf.get(i).reward
            );
        }
    }

    #[test]
    fn proximity_bonus_stops_at_episode_boundary() {
        // Two-episode buffer: ep1 ends at idx=2 with -100 (crash),
        // ep2 starts at idx=3 (env_boundary=true on idx=3) and ends
        // at idx=5 with +30 (landing). K=10 lookback should NOT cross
        // back into ep1 — only idx=3 and idx=4 receive bonus.
        let mut buf = mk_buffer(&[
            (-0.3, 0.0, false),   // ep1 step 0
            (-0.3, 0.0, false),   // ep1 step 1
            (-100.0, 0.0, false), // ep1 step 2 (terminal, but no boundary flag yet — flag is on next ep's first transition)
            (-0.3, 0.0, true),    // ep2 step 0 (env_boundary=true: first of new ep)
            (-0.3, 0.0, false),   // ep2 step 1
            (30.0, 0.0, false),   // ep2 step 2 (terminal)
        ]);
        apply_terminal_proximity_bonus(&mut buf, 10, 2.0, 0.0);
        // Ep1 transitions must be untouched:
        assert!((buf.get(0).reward - (-0.3)).abs() < 1e-5);
        assert!((buf.get(1).reward - (-0.3)).abs() < 1e-5);
        assert!((buf.get(2).reward - (-100.0)).abs() < 1e-5);
        // Ep2 idx=3 has env_boundary=true and is the start; the loop
        // stops AT this transition and doesn't bonus it.
        assert!(
            (buf.get(3).reward - (-0.3)).abs() < 1e-5,
            "boundary tx reward: {}",
            buf.get(3).reward
        );
        // idx=4 (one before terminal) gets bonus:
        assert!((buf.get(4).reward - (-0.3 + 2.0)).abs() < 1e-5);
        // Terminal unchanged:
        assert!((buf.get(5).reward - 30.0).abs() < 1e-5);
    }

    #[test]
    fn proximity_bonus_disabled_by_zero_k_or_bonus() {
        let mut buf1 = mk_buffer(&[(-0.3, 0.0, false), (30.0, 0.0, false)]);
        apply_terminal_proximity_bonus(&mut buf1, 0, 2.0, 0.0);
        assert!((buf1.get(0).reward - (-0.3)).abs() < 1e-5);
        let mut buf2 = mk_buffer(&[(-0.3, 0.0, false), (30.0, 0.0, false)]);
        apply_terminal_proximity_bonus(&mut buf2, 5, 0.0, 0.0);
        assert!((buf2.get(0).reward - (-0.3)).abs() < 1e-5);
    }

    #[test]
    fn proximity_bonus_threshold_gates_firing() {
        // Terminal reward exactly at threshold should NOT fire (>, not >=).
        let mut buf1 = mk_buffer(&[(-0.3, 0.0, false), (5.0, 0.0, false)]);
        apply_terminal_proximity_bonus(&mut buf1, 1, 2.0, 5.0);
        assert!((buf1.get(0).reward - (-0.3)).abs() < 1e-5);
        // Terminal reward above threshold fires.
        let mut buf2 = mk_buffer(&[(-0.3, 0.0, false), (10.0, 0.0, false)]);
        apply_terminal_proximity_bonus(&mut buf2, 1, 2.0, 5.0);
        assert!((buf2.get(0).reward - (-0.3 + 2.0)).abs() < 1e-5);
    }
}
