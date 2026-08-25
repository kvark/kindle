//! Online acting, replay learning, and latent imagination.

use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use meganeura::{Mode, Session};
use rand::{SeedableRng, rngs::StdRng};

use super::behavior;
use super::config::DreamerConfig;
use super::distributions::{
    PercentileNormalizer, TwoHotBins, lambda_returns, sample_logits, sample_probabilities,
    softmax_unimix,
};
use super::replay::{FrameFlags, ReplayFrame, Reward, SequenceBatch, SequenceReplay};
use super::runtime::{
    build_session, configure_d3_optimizer, ema_matching, initialize_d3, sync_matching,
};
use super::world;
use super::{BLADE_REV, DREAMERV3_UPSTREAM_REV, MEGANEURA_REV};
use crate::env::{RgbFrame, Transition};
use crate::vision::{
    DINOV3_UPSTREAM_REV, DINOVISION_SOURCE_REV, DinoObservation, DinoPerception,
    OBSERVATION_CHANNELS, OBSERVATION_GRID, PROJECTION_SEED, VITS16_CHECKPOINT_REV,
    VITS16_MODEL_ID,
};

const CHECKPOINT_FORMAT: u32 = 2;
const CHECKPOINT_ARCHITECTURE: &str = "dreamerv3-dinov3-vits16";
const CHECKPOINT_METADATA: &str = "metadata.json";
const CHECKPOINT_WORLD: &str = "world.safetensors";
const CHECKPOINT_BEHAVIOR: &str = "behavior.safetensors";
const CHECKPOINT_SLOW_VALUE: &str = "slow_value.safetensors";
const RNG_POLICY: u64 = 0x706f_6c69_6379_0001;
const RNG_LIVE_POSTERIOR: u64 = 0x6c69_7665_706f_7374;
const RNG_REPLAY: u64 = 0x7265_706c_6179_0001;
const RNG_TRAIN_POSTERIOR: u64 = 0x7472_6169_6e70_6f73;
const RNG_IMAGINATION: u64 = 0x696d_6167_696e_6501;
const RNG_DIAGNOSTIC: u64 = 0x6469_6167_6e6f_7374;
const RNG_RESUME: u64 = 0x7265_7375_6d65_0001;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct CheckpointMetadata {
    format: u32,
    architecture: String,
    dreamerv3_revision: String,
    meganeura_revision: String,
    blade_revision: String,
    dinov3_revision: String,
    dinovision_revision: String,
    dino_model_id: String,
    dino_checkpoint_revision: String,
    projection_seed: u64,
    observation_grid: usize,
    observation_channels: usize,
    config: DreamerConfig,
    learner_step: u64,
    environment_step: u64,
    return_low: f32,
    return_high: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActionMode {
    #[default]
    Sample,
    Greedy,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct WorldMetrics {
    pub total_loss: f32,
    pub reconstruction_loss: f32,
    /// Unclipped posterior-to-prior KL before either free-nat floor.
    pub raw_kl: f32,
    pub dynamics_kl: f32,
    pub representation_kl: f32,
    pub reward_loss: f32,
    pub continuation_loss: f32,
    /// Replay-value loss as seen by the frozen critic inside the world graph.
    /// Its gradient updates RSSM/representation parameters, not critic weights.
    pub replay_value_loss: f32,
    pub replay_reward_prediction_mean: f32,
    pub replay_reward_target_mean: f32,
    pub replay_reward_mae: f32,
    pub rewarded_prediction_mean: f32,
    pub unrewarded_prediction_mean: f32,
    pub rewarded_count: usize,
    pub replay_continuation_prediction_mean: f32,
    pub replay_continuation_target_mean: f32,
    pub replay_continuation_mae: f32,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BehaviorMetrics {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub replay_value_loss: f32,
    /// Mean categorical entropy before trajectory weighting.
    pub policy_entropy: f32,
    /// Entropy after the same continuation weighting used by the actor loss.
    pub weighted_policy_entropy: f32,
    pub imagined_reward_mean: f32,
    pub imagined_continuation_mean: f32,
    pub return_mean: f32,
    pub return_scale: f32,
    pub advantage_abs_mean: f32,
    pub weighted_advantage_abs_mean: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LearnReport {
    pub learner_step: u64,
    pub replay_len: usize,
    pub world: WorldMetrics,
    pub behavior: BehaviorMetrics,
}

struct PosteriorBatch {
    deter: Vec<Vec<f32>>,
    stoch: Vec<Vec<f32>>,
}

struct BehaviorTrainingBatch {
    imagined_feature: Vec<f32>,
    action_target: Vec<f32>,
    imagined_weight: Vec<f32>,
    imagined_value_target: Vec<f32>,
    imagined_slow_target: Vec<f32>,
    replay_feature: Vec<f32>,
    replay_weight: Vec<f32>,
    replay_value_target: Vec<f32>,
    replay_slow_target: Vec<f32>,
    imagined_reward_mean: f32,
    imagined_continuation_mean: f32,
    return_mean: f32,
    return_scale: f32,
    advantage_abs_mean: f32,
    weighted_advantage_abs_mean: f32,
    replay_reward_prediction_mean: f32,
    replay_reward_target_mean: f32,
    replay_reward_mae: f32,
    rewarded_prediction_mean: f32,
    unrewarded_prediction_mean: f32,
    rewarded_count: usize,
    replay_continuation_prediction_mean: f32,
    replay_continuation_target_mean: f32,
    replay_continuation_mae: f32,
}

/// D3's runner does not call its ratio scheduler before replay is ready. The
/// first eligible call yields one update; only later environment steps accrue
/// fractional train-ratio credit.
#[derive(Default)]
struct D3TrainScheduler {
    started: bool,
    credit: f32,
}

struct DreamerRngs {
    policy: StdRng,
    live_posterior: StdRng,
    replay: StdRng,
    train_posterior: StdRng,
    imagination: StdRng,
}

impl DreamerRngs {
    fn new(seed: u64) -> Self {
        Self {
            policy: StdRng::seed_from_u64(seed ^ RNG_POLICY),
            live_posterior: StdRng::seed_from_u64(seed ^ RNG_LIVE_POSTERIOR),
            replay: StdRng::seed_from_u64(seed ^ RNG_REPLAY),
            train_posterior: StdRng::seed_from_u64(seed ^ RNG_TRAIN_POSTERIOR),
            imagination: StdRng::seed_from_u64(seed ^ RNG_IMAGINATION),
        }
    }

    fn resumed(seed: u64, learner_step: u64, environment_step: u64) -> Self {
        Self::new(
            seed ^ RNG_RESUME ^ learner_step.rotate_left(17) ^ environment_step.rotate_left(41),
        )
    }

    fn diagnostic(seed: u64, learner_step: u64, environment_step: u64) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ RNG_DIAGNOSTIC ^ learner_step.rotate_left(23) ^ environment_step.rotate_left(47),
        )
    }
}

impl D3TrainScheduler {
    fn observe(&mut self, credit: f32) {
        if self.started {
            self.credit += credit;
        }
    }

    fn update_due(&mut self, replay_ready: bool) -> bool {
        if !replay_ready {
            return false;
        }
        if !self.started {
            self.started = true;
            self.credit = 1.0;
        }
        self.credit >= 1.0
    }

    fn consume_update(&mut self) {
        assert!(self.credit >= 1.0);
        self.credit -= 1.0;
    }
}

/// DreamerV3 learner over precomputed frozen-DINO observations.
///
/// Acting never trains implicitly. Call [`Self::learn`] explicitly, or use
/// [`Self::learn_scheduled`] to honor D3's replay-samples-per-environment-step
/// ratio without putting learner work on the control deadline.
pub struct DreamerCore {
    config: DreamerConfig,
    bins: TwoHotBins,
    replay: SequenceReplay,
    rngs: DreamerRngs,
    return_normalizer: PercentileNormalizer,
    world_train: Session,
    world_observe_batch: Session,
    world_observe_live: Session,
    world_transition: Session,
    world_transition_live: Session,
    world_heads: Session,
    world_heads_live: Session,
    behavior_train: Session,
    behavior_online: Session,
    behavior_slow: Session,
    policy_live: Session,
    deter: Vec<f32>,
    stoch: Vec<f32>,
    encoded_observation: Vec<f32>,
    feature: Vec<f32>,
    pending_action: Option<usize>,
    active: bool,
    needs_reset: bool,
    learner_step: u64,
    environment_step: u64,
    train_scheduler: D3TrainScheduler,
}

impl DreamerCore {
    pub fn new(config: DreamerConfig) -> Result<Self, blade_graphics::NotSupportedError> {
        let gpu = Arc::new(meganeura::init_gpu_context()?);
        Ok(Self::with_gpu(config, gpu))
    }

    /// Restore a model-sized training checkpoint into a fresh runtime.
    ///
    /// Replay and the in-flight episode are deliberately not checkpointed.
    /// A restored agent starts between episodes with empty replay, while both
    /// optimizer moments, the slow critic, counters, and return normalization
    /// continue exactly from the saved learner state.
    pub fn restore(checkpoint: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = Arc::new(meganeura::init_gpu_context()?);
        Self::restore_with_gpu(checkpoint.as_ref(), gpu)
    }

    fn restore_with_gpu(
        checkpoint: &Path,
        gpu: Arc<blade_graphics::Context>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = read_checkpoint_metadata(checkpoint)?;
        validate_checkpoint_metadata(&metadata)?;
        let mut core = Self::with_gpu(metadata.config.clone(), gpu);
        core.world_train
            .load_checkpoint(&checkpoint.join(CHECKPOINT_WORLD))?;
        core.behavior_train
            .load_checkpoint(&checkpoint.join(CHECKPOINT_BEHAVIOR))?;
        core.behavior_slow
            .load_checkpoint(&checkpoint.join(CHECKPOINT_SLOW_VALUE))?;
        core.learner_step = metadata.learner_step;
        core.environment_step = metadata.environment_step;
        core.return_normalizer
            .restore(metadata.return_low, metadata.return_high);
        core.rngs =
            DreamerRngs::resumed(core.config.seed, core.learner_step, core.environment_step);
        core.sync_world_inference();
        sync_matching(&core.behavior_train, &mut core.behavior_online, "behavior.");
        sync_matching(
            &core.behavior_train,
            &mut core.policy_live,
            "behavior.actor.",
        );
        sync_matching(
            &core.behavior_train,
            &mut core.world_train,
            "behavior.value.",
        );
        Ok(core)
    }

    pub fn with_gpu(config: DreamerConfig, gpu: Arc<blade_graphics::Context>) -> Self {
        config.validate();
        let starts = config.batch_size * config.batch_length;
        let imagined_rows = starts * config.imagination_length;
        let replay_rows = config.batch_size * (config.batch_length - 1);

        let world_train_graph = world::build_training_graph(&config, config.world_backprop_length);
        let world_observe_batch_graph = world::build_observe_graph(&config, config.batch_size);
        let world_observe_live_graph = world::build_observe_graph(&config, 1);
        let world_transition_graph = world::build_transition_graph(&config, starts);
        let world_transition_live_graph = world::build_transition_graph(&config, 1);
        let world_head_graph = world::build_head_graph(&config, starts);
        let world_head_live_graph = world::build_head_graph(&config, 1);
        let behavior_train_graph =
            behavior::build_training_graph(&config, imagined_rows, replay_rows);
        let behavior_online_graph = behavior::build_inference_graph(&config, starts);
        let behavior_slow_graph = behavior::build_value_inference_graph(&config, starts);
        let policy_live_graph = behavior::build_actor_inference_graph(&config, 1);

        let mut world_train = build_session(
            &world_train_graph,
            &gpu,
            Mode::Training,
            config.skip_full_optimize,
        );
        let mut world_observe_batch =
            build_session(&world_observe_batch_graph, &gpu, Mode::Inference, false);
        let mut world_observe_live =
            build_session(&world_observe_live_graph, &gpu, Mode::Inference, false);
        let mut world_transition =
            build_session(&world_transition_graph, &gpu, Mode::Inference, false);
        let mut world_transition_live =
            build_session(&world_transition_live_graph, &gpu, Mode::Inference, false);
        let mut world_heads = build_session(&world_head_graph, &gpu, Mode::Inference, false);
        let mut world_heads_live =
            build_session(&world_head_live_graph, &gpu, Mode::Inference, false);
        let mut behavior_train = build_session(
            &behavior_train_graph,
            &gpu,
            Mode::Training,
            config.skip_full_optimize,
        );
        let mut behavior_online =
            build_session(&behavior_online_graph, &gpu, Mode::Inference, false);
        let mut behavior_slow = build_session(&behavior_slow_graph, &gpu, Mode::Inference, false);
        let mut policy_live = build_session(&policy_live_graph, &gpu, Mode::Inference, false);

        initialize_d3(&mut world_train, &world_train_graph, config.seed);
        initialize_d3(
            &mut behavior_train,
            &behavior_train_graph,
            config.seed ^ 0x5eed_0000_0000_0001,
        );
        for target in [
            &mut world_observe_batch,
            &mut world_observe_live,
            &mut world_transition,
            &mut world_transition_live,
            &mut world_heads,
            &mut world_heads_live,
        ] {
            sync_matching(&world_train, target, "world.");
        }
        sync_matching(&behavior_train, &mut behavior_online, "behavior.");
        sync_matching(&behavior_train, &mut behavior_slow, "behavior.value.");
        sync_matching(&behavior_train, &mut policy_live, "behavior.actor.");
        sync_matching(&behavior_train, &mut world_train, "behavior.value.");

        let size = config.network();
        Self {
            bins: TwoHotBins::new(config.value_bins),
            replay: SequenceReplay::new(config.replay_capacity),
            rngs: DreamerRngs::new(config.seed),
            return_normalizer: PercentileNormalizer::new(config.return_norm_rate, 1.0),
            deter: vec![0.0; size.deter],
            stoch: vec![0.0; size.stoch * size.classes],
            encoded_observation: vec![0.0; OBSERVATION_GRID * OBSERVATION_GRID * size.vision_depth],
            feature: vec![0.0; config.feature_dim()],
            pending_action: None,
            active: false,
            needs_reset: false,
            learner_step: 0,
            environment_step: 0,
            train_scheduler: D3TrainScheduler::default(),
            config,
            world_train,
            world_observe_batch,
            world_observe_live,
            world_transition,
            world_transition_live,
            world_heads,
            world_heads_live,
            behavior_train,
            behavior_online,
            behavior_slow,
            policy_live,
        }
    }

    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    pub fn learner_step(&self) -> u64 {
        self.learner_step
    }

    pub fn environment_step(&self) -> u64 {
        self.environment_step
    }

    /// Trainable world and behavior parameter counts, excluding frozen DINO,
    /// inference-session copies, and the EMA value model.
    pub fn trainable_parameter_counts(&self) -> (usize, usize) {
        (
            trainable_parameter_count(&self.world_train),
            trainable_parameter_count(&self.behavior_train),
        )
    }

    /// Current posterior feature (`deter` followed by flattened categoricals).
    ///
    /// This read-only view is useful for representation probes. It must not be
    /// persisted as agent state; replay owns the recurrent context it needs.
    pub fn latent_feature(&self) -> &[f32] {
        &self.feature
    }

    /// Current output of the trainable adapter between DINO and the RSSM.
    pub fn encoded_observation(&self) -> &[f32] {
        &self.encoded_observation
    }

    /// Reward predicted from the current posterior state.
    ///
    /// This runs the synchronized inference head without changing recurrent
    /// state and is intended for frozen evaluation and representation probes.
    pub fn posterior_reward_prediction(&mut self) -> f32 {
        self.world_heads_live.set_input("deter", &self.deter);
        self.world_heads_live.set_input("stoch", &self.stoch);
        self.world_heads_live.step();
        self.world_heads_live.wait();
        let mut reward_logits = vec![0.0; self.config.value_bins];
        self.world_heads_live
            .read_output_by_index(0, &mut reward_logits);
        decode_rows(&reward_logits, 1, &self.bins)[0]
    }

    /// Reward predicted after one prior transition from the current state.
    ///
    /// This read-only diagnostic clones the categorical RNG, so probing does
    /// not perturb subsequent acting or posterior sampling.
    pub fn prior_reward_prediction(&mut self, action: usize) -> f32 {
        self.prior_reward_rollout(&[action])[0]
    }

    /// Open-loop prior reward predictions for a proposed action sequence.
    /// The live posterior state and RNG are left unchanged.
    pub fn prior_reward_rollout(&mut self, actions: &[usize]) -> Vec<f32> {
        assert!(self.active, "call begin_episode before probing the prior");
        assert!(!actions.is_empty());
        assert!(
            actions
                .iter()
                .all(|action| *action < self.config.action_count)
        );
        let size = self.config.network();
        let mut deter = self.deter.clone();
        let mut stoch = self.stoch.clone();
        let mut rng =
            DreamerRngs::diagnostic(self.config.seed, self.learner_step, self.environment_step);
        let mut predictions = Vec::with_capacity(actions.len());
        for &action in actions {
            let mut action_one_hot = vec![0.0; self.config.action_count];
            action_one_hot[action] = 1.0;
            self.world_transition_live.set_input("deter", &deter);
            self.world_transition_live.set_input("stoch", &stoch);
            self.world_transition_live
                .set_input("action", &action_one_hot);
            self.world_transition_live.step();
            self.world_transition_live.wait();
            let mut prior_logits = vec![0.0; size.stoch * size.classes];
            self.world_transition_live
                .read_output_by_index(0, &mut deter);
            self.world_transition_live
                .read_output_by_index(1, &mut prior_logits);
            stoch = sample_latents(
                &prior_logits,
                1,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut rng,
            );
            self.world_heads_live.set_input("deter", &deter);
            self.world_heads_live.set_input("stoch", &stoch);
            self.world_heads_live.step();
            self.world_heads_live.wait();
            let mut reward_logits = vec![0.0; self.config.value_bins];
            self.world_heads_live
                .read_output_by_index(0, &mut reward_logits);
            predictions.push(decode_rows(&reward_logits, 1, &self.bins)[0]);
        }
        predictions
    }

    /// Save model parameters, optimizer state, the EMA critic, and scalar learner
    /// state. Replay is intentionally excluded and can be refilled online.
    pub fn save_checkpoint(&mut self, checkpoint: impl AsRef<Path>) -> io::Result<()> {
        let checkpoint = checkpoint.as_ref();
        fs::create_dir_all(checkpoint)?;
        let world_temporary = checkpoint.join(format!("{CHECKPOINT_WORLD}.tmp"));
        let behavior_temporary = checkpoint.join(format!("{CHECKPOINT_BEHAVIOR}.tmp"));
        let slow_temporary = checkpoint.join(format!("{CHECKPOINT_SLOW_VALUE}.tmp"));
        self.world_train.save_checkpoint(&world_temporary)?;
        self.behavior_train.save_checkpoint(&behavior_temporary)?;
        self.behavior_slow.save_checkpoint(&slow_temporary)?;
        fs::rename(&world_temporary, checkpoint.join(CHECKPOINT_WORLD))?;
        fs::rename(&behavior_temporary, checkpoint.join(CHECKPOINT_BEHAVIOR))?;
        fs::rename(&slow_temporary, checkpoint.join(CHECKPOINT_SLOW_VALUE))?;

        let (return_low, return_high) = self.return_normalizer.state();
        let metadata = CheckpointMetadata {
            format: CHECKPOINT_FORMAT,
            architecture: CHECKPOINT_ARCHITECTURE.to_owned(),
            dreamerv3_revision: DREAMERV3_UPSTREAM_REV.to_owned(),
            meganeura_revision: MEGANEURA_REV.to_owned(),
            blade_revision: BLADE_REV.to_owned(),
            dinov3_revision: DINOV3_UPSTREAM_REV.to_owned(),
            dinovision_revision: DINOVISION_SOURCE_REV.to_owned(),
            dino_model_id: VITS16_MODEL_ID.to_owned(),
            dino_checkpoint_revision: VITS16_CHECKPOINT_REV.to_owned(),
            projection_seed: PROJECTION_SEED,
            observation_grid: OBSERVATION_GRID,
            observation_channels: OBSERVATION_CHANNELS,
            config: self.config.clone(),
            learner_step: self.learner_step,
            environment_step: self.environment_step,
            return_low,
            return_high,
        };
        let encoded = serde_json::to_vec_pretty(&metadata).map_err(io::Error::other)?;
        let metadata_temporary = checkpoint.join(format!("{CHECKPOINT_METADATA}.tmp"));
        fs::write(&metadata_temporary, encoded)?;
        fs::rename(metadata_temporary, checkpoint.join(CHECKPOINT_METADATA))?;
        Ok(())
    }

    /// Begin the first episode or the next episode after `is_last`.
    /// Replay and all optimizer/model state survive this recurrent reset.
    pub fn begin_episode(&mut self, observation: DinoObservation) {
        assert!(
            self.pending_action.is_none(),
            "cannot reset with a pending action"
        );
        assert!(
            !self.active || self.needs_reset,
            "begin_episode is only legal initially or after an is_last frame"
        );
        self.deter.fill(0.0);
        self.stoch.fill(0.0);
        self.posterior_live(observation.as_slice(), None, true);
        self.replay.push(
            ReplayFrame {
                observation,
                previous_action: None,
                reward: Reward::default(),
                flags: FrameFlags {
                    is_first: true,
                    is_last: false,
                    is_terminal: false,
                },
                deter: self.deter.clone().into_boxed_slice(),
                stoch: self.stoch.clone().into_boxed_slice(),
            },
            &self.config,
        );
        self.active = true;
        self.needs_reset = false;
    }

    pub fn act(&mut self, mode: ActionMode, action_mask: Option<&[bool]>) -> usize {
        assert!(self.active, "call begin_episode before act");
        assert!(
            !self.needs_reset,
            "call begin_episode after an is_last frame"
        );
        assert!(
            self.pending_action.is_none(),
            "observe the pending action first"
        );
        let probabilities = self.posterior_action_probabilities(action_mask);
        let action = match mode {
            ActionMode::Sample => sample_probabilities(&probabilities, &mut self.rngs.policy),
            ActionMode::Greedy => argmax(&probabilities),
        };
        self.pending_action = Some(action);
        action
    }

    /// Current posterior policy after optional action masking and actor
    /// unimix. This read-only diagnostic does not sample or alter live state.
    pub fn posterior_action_probabilities(&mut self, action_mask: Option<&[bool]>) -> Vec<f32> {
        assert!(self.active, "call begin_episode before probing the policy");
        if let Some(mask) = action_mask {
            assert_eq!(mask.len(), self.config.action_count);
            assert!(mask.iter().any(|valid| *valid));
        }
        self.policy_live.set_input("feature", &self.feature);
        self.policy_live.step();
        self.policy_live.wait();
        let mut logits = vec![0.0; self.config.action_count];
        self.policy_live.read_output_by_index(0, &mut logits);
        if let Some(mask) = action_mask {
            for (logit, valid) in logits.iter_mut().zip(mask) {
                if !valid {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        let mut probabilities = vec![0.0; self.config.action_count];
        if let Some(mask) = action_mask {
            let valid_count = mask.iter().filter(|valid| **valid).count();
            softmax_unimix(&logits, 0.0, &mut probabilities);
            for (probability, valid) in probabilities.iter_mut().zip(mask) {
                *probability = if *valid {
                    (1.0 - self.config.actor_unimix) * *probability
                        + self.config.actor_unimix / valid_count as f32
                } else {
                    0.0
                };
            }
        } else {
            softmax_unimix(&logits, self.config.actor_unimix, &mut probabilities);
        }
        probabilities
    }

    pub fn observe(&mut self, observation: DinoObservation, reward: Reward, flags: FrameFlags) {
        assert!(!flags.is_first, "use begin_episode for an is_first frame");
        if flags.is_terminal {
            assert!(flags.is_last);
        }
        let action = self
            .pending_action
            .take()
            .expect("act must precede observe");
        self.posterior_live(observation.as_slice(), Some(action), false);
        self.replay.push(
            ReplayFrame {
                observation,
                previous_action: Some(action),
                reward,
                flags,
                deter: self.deter.clone().into_boxed_slice(),
                stoch: self.stoch.clone().into_boxed_slice(),
            },
            &self.config,
        );
        self.environment_step += 1;
        self.train_scheduler.observe(
            self.config.train_ratio / (self.config.batch_size * self.config.batch_length) as f32,
        );
        self.needs_reset = flags.is_last;
    }

    /// Execute at most `maximum_updates` updates due under D3's train ratio.
    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        let mut reports = Vec::new();
        let replay_ready =
            self.replay.valid_sequence_count(&self.config) >= self.config.replay_warmup_sequences();
        while reports.len() < maximum_updates && self.train_scheduler.update_due(replay_ready) {
            let Some(report) = self.learn() else {
                break;
            };
            self.train_scheduler.consume_update();
            reports.push(report);
        }
        reports
    }

    /// Perform one D3 learner update, independent of scheduler credit.
    pub fn learn(&mut self) -> Option<LearnReport> {
        let batch = self.replay.sample(&self.config, &mut self.rngs.replay)?;
        let posterior = self.sample_posterior_batch(&batch);
        // D3 forms all targets from the same pre-update parameters. Keeping
        // this ordering also gives the world graph replay-value targets while
        // its frozen critic routes that auxiliary gradient into the RSSM.
        let behavior_batch = self.imagine_and_target(&batch, &posterior);
        let world = self.train_world(&batch, &posterior, &behavior_batch);
        self.replay
            .update_context(&batch, &posterior.deter, &posterior.stoch, &self.config);
        self.sync_world_inference();
        let behavior = self.train_behavior(&behavior_batch);
        self.learner_step += 1;
        Some(LearnReport {
            learner_step: self.learner_step,
            replay_len: self.replay.len(),
            world,
            behavior,
        })
    }

    fn posterior_live(&mut self, observation: &[f32], action: Option<usize>, reset: bool) {
        let size = self.config.network();
        let mut action_one_hot = vec![0.0; self.config.action_count];
        if let Some(action) = action {
            action_one_hot[action] = 1.0;
        }
        let keep = if reset { 0.0 } else { 1.0 };
        self.world_observe_live
            .set_input("previous_deter", &self.deter);
        self.world_observe_live
            .set_input("previous_stoch", &self.stoch);
        self.world_observe_live
            .set_input("previous_action", &action_one_hot);
        self.world_observe_live
            .set_input("observation", observation);
        self.world_observe_live
            .set_input("keep_deter", &vec![keep; size.deter]);
        self.world_observe_live
            .set_input("keep_stoch", &vec![keep; size.stoch * size.classes]);
        self.world_observe_live
            .set_input("keep_action", &vec![keep; self.config.action_count]);
        self.world_observe_live.step();
        self.world_observe_live.wait();
        self.world_observe_live
            .read_output_by_index(0, &mut self.deter);
        let mut logits = vec![0.0; size.stoch * size.classes];
        self.world_observe_live.read_output_by_index(1, &mut logits);
        self.world_observe_live
            .read_output_by_index(3, &mut self.encoded_observation);
        self.stoch = sample_latents(
            &logits,
            1,
            size.stoch,
            size.classes,
            self.config.unimix,
            &mut self.rngs.live_posterior,
        );
        self.feature = join_features(&self.deter, &self.stoch, 1, &self.config);
        assert!(self.feature.iter().all(|value| value.is_finite()));
    }

    fn sample_posterior_batch(&mut self, batch: &SequenceBatch) -> PosteriorBatch {
        let size = self.config.network();
        let rows = self.config.batch_size;
        let mut previous_deter = batch.initial_deter.clone();
        let mut previous_stoch = batch.initial_stoch.clone();
        let mut deters = Vec::with_capacity(self.config.batch_length);
        let mut stochs = Vec::with_capacity(self.config.batch_length);
        for time in 0..self.config.batch_length {
            let (keep_deter, keep_stoch, keep_action) = keep_masks(batch, time, &self.config);
            self.world_observe_batch
                .set_input("previous_deter", &previous_deter);
            self.world_observe_batch
                .set_input("previous_stoch", &previous_stoch);
            self.world_observe_batch
                .set_input("previous_action", &batch.previous_actions[time]);
            self.world_observe_batch
                .set_input("observation", &batch.observations[time]);
            self.world_observe_batch
                .set_input("keep_deter", &keep_deter);
            self.world_observe_batch
                .set_input("keep_stoch", &keep_stoch);
            self.world_observe_batch
                .set_input("keep_action", &keep_action);
            self.world_observe_batch.step();
            self.world_observe_batch.wait();
            let mut deter = vec![0.0; rows * size.deter];
            let mut logits = vec![0.0; rows * size.stoch * size.classes];
            self.world_observe_batch.read_output_by_index(0, &mut deter);
            self.world_observe_batch
                .read_output_by_index(1, &mut logits);
            let stoch = sample_latents(
                &logits,
                rows,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut self.rngs.train_posterior,
            );
            previous_deter = deter.clone();
            previous_stoch = stoch.clone();
            deters.push(deter);
            stochs.push(stoch);
        }
        PosteriorBatch {
            deter: deters,
            stoch: stochs,
        }
    }

    fn train_world(
        &mut self,
        batch: &SequenceBatch,
        posterior: &PosteriorBatch,
        behavior: &BehaviorTrainingBatch,
    ) -> WorldMetrics {
        let rows = self.config.batch_size;
        let chunk_length = self.config.world_backprop_length;
        let chunk_count = self.config.batch_length / chunk_length;
        self.world_train
            .set_grad_accumulate(chunk_count.try_into().unwrap());
        self.world_train.zero_grad();
        let mut metrics = WorldMetrics::default();

        for chunk in 0..chunk_count {
            let start = chunk * chunk_length;
            if start == 0 {
                self.world_train
                    .set_input("initial_deter", &batch.initial_deter);
                self.world_train
                    .set_input("initial_stoch", &batch.initial_stoch);
            } else {
                self.world_train
                    .set_input("initial_deter", &posterior.deter[start - 1]);
                self.world_train
                    .set_input("initial_stoch", &posterior.stoch[start - 1]);
            }

            for local_time in 0..chunk_length {
                let time = start + local_time;
                let (keep_deter, keep_stoch, keep_action) = keep_masks(batch, time, &self.config);
                self.world_train.set_input(
                    &format!("observation_{local_time}"),
                    &batch.observations[time],
                );
                self.world_train.set_input(
                    &format!("previous_action_{local_time}"),
                    &batch.previous_actions[time],
                );
                self.world_train
                    .set_input(&format!("keep_deter_{local_time}"), &keep_deter);
                self.world_train
                    .set_input(&format!("keep_stoch_{local_time}"), &keep_stoch);
                self.world_train
                    .set_input(&format!("keep_action_{local_time}"), &keep_action);
                self.world_train.set_input(
                    &format!("posterior_sample_{local_time}"),
                    &posterior.stoch[time],
                );
                let mut reward_target = vec![0.0; rows * self.config.value_bins];
                for row in 0..rows {
                    self.bins.encode(
                        batch.rewards[time][row],
                        &mut reward_target
                            [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                    );
                }
                self.world_train
                    .set_input(&format!("reward_target_{local_time}"), &reward_target);
                let continuation_target = batch.flags[time]
                    .iter()
                    .map(|flags| {
                        if flags.is_terminal {
                            0.0
                        } else {
                            self.config.continuation_discount()
                        }
                    })
                    .collect::<Vec<_>>();
                self.world_train.set_input(
                    &format!("continuation_target_{local_time}"),
                    &continuation_target,
                );
                let mut replay_value_target = vec![0.0; rows * self.config.value_bins];
                let mut replay_slow_target = vec![0.0; rows * self.config.value_bins];
                let mut replay_value_weight = vec![0.0; rows];
                if time + 1 < self.config.batch_length {
                    let row_start = time * rows;
                    let value_start = row_start * self.config.value_bins;
                    let value_end = value_start + rows * self.config.value_bins;
                    replay_value_target
                        .copy_from_slice(&behavior.replay_value_target[value_start..value_end]);
                    replay_slow_target
                        .copy_from_slice(&behavior.replay_slow_target[value_start..value_end]);
                    // This graph averages across all T states, whereas D3's
                    // replay-value loss contains T-1 targets. Correct that
                    // denominator while keeping the final state's weight 0.
                    let denominator_correction =
                        self.config.batch_length as f32 / (self.config.batch_length - 1) as f32;
                    for (target, source) in replay_value_weight
                        .iter_mut()
                        .zip(&behavior.replay_weight[row_start..row_start + rows])
                    {
                        *target = *source * denominator_correction;
                    }
                }
                self.world_train.set_input(
                    &format!("replay_value_target_{local_time}"),
                    &replay_value_target,
                );
                self.world_train.set_input(
                    &format!("replay_slow_target_{local_time}"),
                    &replay_slow_target,
                );
                self.world_train.set_input(
                    &format!("replay_value_weight_{local_time}"),
                    &replay_value_weight,
                );
            }

            if chunk + 1 == chunk_count {
                configure_d3_optimizer(
                    &mut self.world_train,
                    &self.config,
                    self.learner_step,
                    self.config.learning_rate,
                );
            } else {
                self.world_train.clear_optimizer();
            }
            self.world_train.step();
            self.world_train.wait();
            metrics.total_loss += read_scalar(&self.world_train, world::LOSS_TOTAL);
            metrics.reconstruction_loss +=
                read_scalar(&self.world_train, world::LOSS_RECONSTRUCTION);
            metrics.raw_kl += read_scalar(&self.world_train, world::RAW_KL);
            metrics.dynamics_kl += read_scalar(&self.world_train, world::LOSS_DYNAMICS);
            metrics.representation_kl += read_scalar(&self.world_train, world::LOSS_REPRESENTATION);
            metrics.reward_loss += read_scalar(&self.world_train, world::LOSS_REWARD);
            metrics.continuation_loss += read_scalar(&self.world_train, world::LOSS_CONTINUATION);
            metrics.replay_value_loss += read_scalar(&self.world_train, world::LOSS_REPLAY_VALUE);
        }
        self.world_train.clear_grad_accumulate();
        let scale = 1.0 / chunk_count as f32;
        metrics.total_loss *= scale;
        metrics.reconstruction_loss *= scale;
        metrics.raw_kl *= scale;
        metrics.dynamics_kl *= scale;
        metrics.representation_kl *= scale;
        metrics.reward_loss *= scale;
        metrics.continuation_loss *= scale;
        metrics.replay_value_loss *= scale;
        metrics.replay_reward_prediction_mean = behavior.replay_reward_prediction_mean;
        metrics.replay_reward_target_mean = behavior.replay_reward_target_mean;
        metrics.replay_reward_mae = behavior.replay_reward_mae;
        metrics.rewarded_prediction_mean = behavior.rewarded_prediction_mean;
        metrics.unrewarded_prediction_mean = behavior.unrewarded_prediction_mean;
        metrics.rewarded_count = behavior.rewarded_count;
        metrics.replay_continuation_prediction_mean = behavior.replay_continuation_prediction_mean;
        metrics.replay_continuation_target_mean = behavior.replay_continuation_target_mean;
        metrics.replay_continuation_mae = behavior.replay_continuation_mae;
        assert!(all_finite(&[
            metrics.total_loss,
            metrics.reconstruction_loss,
            metrics.raw_kl,
            metrics.dynamics_kl,
            metrics.representation_kl,
            metrics.reward_loss,
            metrics.continuation_loss,
            metrics.replay_value_loss,
            metrics.replay_reward_prediction_mean,
            metrics.replay_reward_target_mean,
            metrics.replay_reward_mae,
            metrics.rewarded_prediction_mean,
            metrics.unrewarded_prediction_mean,
            metrics.replay_continuation_prediction_mean,
            metrics.replay_continuation_target_mean,
            metrics.replay_continuation_mae,
        ]));
        metrics
    }

    fn sync_world_inference(&mut self) {
        for target in [
            &mut self.world_observe_batch,
            &mut self.world_observe_live,
            &mut self.world_transition,
            &mut self.world_transition_live,
            &mut self.world_heads,
            &mut self.world_heads_live,
        ] {
            sync_matching(&self.world_train, target, "world.");
        }
    }

    fn imagine_and_target(
        &mut self,
        batch: &SequenceBatch,
        posterior: &PosteriorBatch,
    ) -> BehaviorTrainingBatch {
        let size = self.config.network();
        let starts = self.config.batch_size * self.config.batch_length;
        let horizon = self.config.imagination_length;
        let mut deter = flatten_time(&posterior.deter);
        let mut stoch = flatten_time(&posterior.stoch);
        let mut features = Vec::with_capacity(horizon + 1);
        let mut actions = Vec::with_capacity(horizon);
        let mut rewards = Vec::with_capacity(horizon + 1);
        let mut continuations = Vec::with_capacity(horizon + 1);
        let mut values = Vec::with_capacity(horizon + 1);
        let mut slow_values = Vec::with_capacity(horizon + 1);

        for time in 0..=horizon {
            let state_feature = join_features(&deter, &stoch, starts, &self.config);
            self.behavior_online.set_input("feature", &state_feature);
            self.behavior_online.step();
            self.behavior_online.wait();
            let mut actor_logits = vec![0.0; starts * self.config.action_count];
            let mut value_logits = vec![0.0; starts * self.config.value_bins];
            self.behavior_online
                .read_output_by_index(0, &mut actor_logits);
            self.behavior_online
                .read_output_by_index(1, &mut value_logits);

            self.behavior_slow.set_input("feature", &state_feature);
            self.behavior_slow.step();
            self.behavior_slow.wait();
            let mut slow_logits = vec![0.0; starts * self.config.value_bins];
            self.behavior_slow.read_output_by_index(0, &mut slow_logits);

            self.world_heads.set_input("deter", &deter);
            self.world_heads.set_input("stoch", &stoch);
            self.world_heads.step();
            self.world_heads.wait();
            let mut reward_logits = vec![0.0; starts * self.config.value_bins];
            let mut continuation = vec![0.0; starts];
            self.world_heads.read_output_by_index(0, &mut reward_logits);
            self.world_heads.read_output_by_index(1, &mut continuation);

            let decoded_reward = decode_rows(&reward_logits, starts, &self.bins);
            let decoded_value = decode_rows(&value_logits, starts, &self.bins);
            let decoded_slow = decode_rows(&slow_logits, starts, &self.bins);
            features.push(state_feature);
            rewards.push(decoded_reward);
            continuations.push(continuation);
            values.push(decoded_value);
            slow_values.push(decoded_slow);

            if time == horizon {
                break;
            }
            let mut action_indices = vec![0; starts];
            let mut action_one_hot = vec![0.0; starts * self.config.action_count];
            for row in 0..starts {
                let logits = &actor_logits
                    [row * self.config.action_count..(row + 1) * self.config.action_count];
                let action =
                    sample_logits(logits, self.config.actor_unimix, &mut self.rngs.imagination);
                action_indices[row] = action;
                action_one_hot[row * self.config.action_count + action] = 1.0;
            }
            actions.push(action_indices);
            self.world_transition.set_input("deter", &deter);
            self.world_transition.set_input("stoch", &stoch);
            self.world_transition.set_input("action", &action_one_hot);
            self.world_transition.step();
            self.world_transition.wait();
            let mut next_deter = vec![0.0; starts * size.deter];
            let mut prior_logits = vec![0.0; starts * size.stoch * size.classes];
            self.world_transition
                .read_output_by_index(0, &mut next_deter);
            self.world_transition
                .read_output_by_index(1, &mut prior_logits);
            let next_stoch = sample_latents(
                &prior_logits,
                starts,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut self.rngs.imagination,
            );
            deter = next_deter;
            stoch = next_stoch;
        }

        let mut returns = (0..horizon).map(|_| vec![0.0; starts]).collect::<Vec<_>>();
        let mut weights = (0..horizon).map(|_| vec![0.0; starts]).collect::<Vec<_>>();
        for start in 0..starts {
            let reward = (0..=horizon)
                .map(|time| rewards[time][start])
                .collect::<Vec<_>>();
            let continuation = (0..=horizon)
                .map(|time| continuations[time][start].clamp(0.0, 1.0))
                .collect::<Vec<_>>();
            let terminal = continuation
                .iter()
                .map(|continuation| 1.0 - continuation)
                .collect::<Vec<_>>();
            let value = (0..=horizon)
                .map(|time| values[time][start])
                .collect::<Vec<_>>();
            let trajectory = lambda_returns(
                &vec![0.0; horizon + 1],
                &terminal,
                &reward,
                &value,
                &value,
                1.0,
                self.config.lambda,
            );
            let mut weight = 1.0;
            for time in 0..horizon {
                returns[time][start] = trajectory[time];
                weight *= continuation[time];
                weights[time][start] = weight;
            }
        }
        let all_returns = flatten_time(&returns);
        self.return_normalizer.update(&all_returns);
        let return_scale = self.return_normalizer.scale();

        let imagined_rows = starts * horizon;
        let mut imagined_feature = Vec::with_capacity(imagined_rows * self.config.feature_dim());
        let mut action_target = vec![0.0; imagined_rows * self.config.action_count];
        let mut imagined_weight = vec![0.0; imagined_rows];
        let mut imagined_value_target = vec![0.0; imagined_rows * self.config.value_bins];
        let mut imagined_slow_target = vec![0.0; imagined_rows * self.config.value_bins];
        let mut advantage_abs_sum = 0.0;
        let mut weighted_advantage_abs_sum = 0.0;
        for time in 0..horizon {
            imagined_feature.extend_from_slice(&features[time]);
            for start in 0..starts {
                let row = time * starts + start;
                let advantage = (returns[time][start] - values[time][start]) / return_scale;
                advantage_abs_sum += advantage.abs();
                weighted_advantage_abs_sum += (weights[time][start] * advantage).abs();
                action_target[row * self.config.action_count + actions[time][start]] =
                    weights[time][start] * advantage;
                imagined_weight[row] = weights[time][start];
                self.bins.encode(
                    returns[time][start],
                    &mut imagined_value_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
                self.bins.encode(
                    slow_values[time][start],
                    &mut imagined_slow_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
            }
        }

        let replay_reward_target = flatten_time(&batch.rewards);
        let replay_reward_prediction = &rewards[0];
        let replay_reward_prediction_mean = mean(replay_reward_prediction);
        let replay_reward_target_mean = mean(&replay_reward_target);
        let replay_reward_mae =
            mean_absolute_error(replay_reward_prediction, &replay_reward_target);
        let mut rewarded_prediction_sum = 0.0;
        let mut unrewarded_prediction_sum = 0.0;
        let mut rewarded_count = 0;
        let mut unrewarded_count = 0;
        for (prediction, target) in replay_reward_prediction.iter().zip(&replay_reward_target) {
            if *target != 0.0 {
                rewarded_prediction_sum += prediction;
                rewarded_count += 1;
            } else {
                unrewarded_prediction_sum += prediction;
                unrewarded_count += 1;
            }
        }
        let rewarded_prediction_mean = rewarded_prediction_sum / rewarded_count.max(1) as f32;
        let unrewarded_prediction_mean = unrewarded_prediction_sum / unrewarded_count.max(1) as f32;
        let replay_continuation_prediction = &continuations[0];
        let replay_continuation_target = batch
            .flags
            .iter()
            .flatten()
            .map(|flags| {
                if flags.is_terminal {
                    0.0
                } else {
                    self.config.continuation_discount()
                }
            })
            .collect::<Vec<_>>();

        let replay_rows = self.config.batch_size * (self.config.batch_length - 1);
        let mut replay_feature = vec![0.0; replay_rows * self.config.feature_dim()];
        let mut replay_weight = vec![0.0; replay_rows];
        let mut replay_value_target = vec![0.0; replay_rows * self.config.value_bins];
        let mut replay_slow_target = vec![0.0; replay_rows * self.config.value_bins];
        for batch_row in 0..self.config.batch_size {
            let last = (0..self.config.batch_length)
                .map(|time| f32::from(batch.flags[time][batch_row].is_last))
                .collect::<Vec<_>>();
            let terminal = (0..self.config.batch_length)
                .map(|time| f32::from(batch.flags[time][batch_row].is_terminal))
                .collect::<Vec<_>>();
            let reward = (0..self.config.batch_length)
                .map(|time| batch.rewards[time][batch_row])
                .collect::<Vec<_>>();
            let value = (0..self.config.batch_length)
                .map(|time| values[0][time * self.config.batch_size + batch_row])
                .collect::<Vec<_>>();
            let bootstrap = (0..self.config.batch_length)
                .map(|time| returns[0][time * self.config.batch_size + batch_row])
                .collect::<Vec<_>>();
            let replay_return = lambda_returns(
                &last,
                &terminal,
                &reward,
                &value,
                &bootstrap,
                self.config.continuation_discount(),
                self.config.lambda,
            );
            for time in 0..self.config.batch_length - 1 {
                let start = time * self.config.batch_size + batch_row;
                let row = time * self.config.batch_size + batch_row;
                replay_feature
                    [row * self.config.feature_dim()..(row + 1) * self.config.feature_dim()]
                    .copy_from_slice(
                        &features[0][start * self.config.feature_dim()
                            ..(start + 1) * self.config.feature_dim()],
                    );
                replay_weight[row] = 1.0 - last[time];
                self.bins.encode(
                    replay_return[time],
                    &mut replay_value_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
                self.bins.encode(
                    slow_values[0][start],
                    &mut replay_slow_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
            }
        }

        BehaviorTrainingBatch {
            imagined_feature,
            action_target,
            imagined_weight,
            imagined_value_target,
            imagined_slow_target,
            replay_feature,
            replay_weight,
            replay_value_target,
            replay_slow_target,
            imagined_reward_mean: mean_nested(&rewards[1..]),
            imagined_continuation_mean: mean_nested(&continuations),
            return_mean: mean_nested(&returns),
            return_scale,
            advantage_abs_mean: advantage_abs_sum / imagined_rows as f32,
            weighted_advantage_abs_mean: weighted_advantage_abs_sum / imagined_rows as f32,
            replay_reward_prediction_mean,
            replay_reward_target_mean,
            replay_reward_mae,
            rewarded_prediction_mean,
            unrewarded_prediction_mean,
            rewarded_count,
            replay_continuation_prediction_mean: mean(replay_continuation_prediction),
            replay_continuation_target_mean: mean(&replay_continuation_target),
            replay_continuation_mae: mean_absolute_error(
                replay_continuation_prediction,
                &replay_continuation_target,
            ),
        }
    }

    fn train_behavior(&mut self, batch: &BehaviorTrainingBatch) -> BehaviorMetrics {
        self.behavior_train
            .set_input("imagined_feature", &batch.imagined_feature);
        self.behavior_train
            .set_input("action_target", &batch.action_target);
        self.behavior_train
            .set_input("imagined_weight", &batch.imagined_weight);
        self.behavior_train
            .set_input("imagined_value_target", &batch.imagined_value_target);
        self.behavior_train
            .set_input("imagined_slow_target", &batch.imagined_slow_target);
        self.behavior_train
            .set_input("replay_feature", &batch.replay_feature);
        self.behavior_train
            .set_input("replay_weight", &batch.replay_weight);
        self.behavior_train
            .set_input("replay_value_target", &batch.replay_value_target);
        self.behavior_train
            .set_input("replay_slow_target", &batch.replay_slow_target);
        configure_d3_optimizer(
            &mut self.behavior_train,
            &self.config,
            self.learner_step,
            self.config.behavior_learning_rate(),
        );
        self.behavior_train.step();
        self.behavior_train.wait();
        let metrics = BehaviorMetrics {
            total_loss: read_scalar(&self.behavior_train, behavior::LOSS_TOTAL),
            policy_loss: read_scalar(&self.behavior_train, behavior::LOSS_POLICY),
            value_loss: read_scalar(&self.behavior_train, behavior::LOSS_VALUE),
            replay_value_loss: read_scalar(&self.behavior_train, behavior::LOSS_REPLAY_VALUE),
            policy_entropy: read_scalar(&self.behavior_train, behavior::POLICY_ENTROPY),
            weighted_policy_entropy: read_scalar(
                &self.behavior_train,
                behavior::WEIGHTED_POLICY_ENTROPY,
            ),
            imagined_reward_mean: batch.imagined_reward_mean,
            imagined_continuation_mean: batch.imagined_continuation_mean,
            return_mean: batch.return_mean,
            return_scale: batch.return_scale,
            advantage_abs_mean: batch.advantage_abs_mean,
            weighted_advantage_abs_mean: batch.weighted_advantage_abs_mean,
        };
        assert!(all_finite(&[
            metrics.total_loss,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.replay_value_loss,
            metrics.policy_entropy,
            metrics.weighted_policy_entropy,
            metrics.imagined_reward_mean,
            metrics.imagined_continuation_mean,
            metrics.return_mean,
            metrics.return_scale,
            metrics.advantage_abs_mean,
            metrics.weighted_advantage_abs_mean,
        ]));
        sync_matching(&self.behavior_train, &mut self.behavior_online, "behavior.");
        ema_matching(
            &self.behavior_train,
            &mut self.behavior_slow,
            "behavior.value.",
            self.config.slow_value_rate,
        );
        sync_matching(
            &self.behavior_train,
            &mut self.policy_live,
            "behavior.actor.",
        );
        // Keep the world graph's fixed critic copy ready for the next replay
        // update and consistent in checkpoints.
        sync_matching(
            &self.behavior_train,
            &mut self.world_train,
            "behavior.value.",
        );
        metrics
    }
}

/// Pixel-facing agent combining the frozen DINO frontend and Dreamer core.
pub struct DreamerAgent {
    perception: DinoPerception,
    core: DreamerCore,
}

impl DreamerAgent {
    pub fn new(
        config: DreamerConfig,
        dino_checkpoint: impl AsRef<Path>,
        dino_plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = Arc::new(meganeura::init_gpu_context()?);
        let perception =
            DinoPerception::load_vits16(dino_checkpoint, Some(Arc::clone(&gpu)), dino_plan_cache)?;
        let core = DreamerCore::with_gpu(config, gpu);
        Ok(Self { perception, core })
    }

    pub fn restore(
        dreamer_checkpoint: impl AsRef<Path>,
        dino_checkpoint: impl AsRef<Path>,
        dino_plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = Arc::new(meganeura::init_gpu_context()?);
        let core = DreamerCore::restore_with_gpu(dreamer_checkpoint.as_ref(), Arc::clone(&gpu))?;
        let perception = DinoPerception::load_vits16(dino_checkpoint, Some(gpu), dino_plan_cache)?;
        Ok(Self { perception, core })
    }

    pub fn core(&self) -> &DreamerCore {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut DreamerCore {
        &mut self.core
    }

    /// Current posterior feature for representation diagnostics.
    pub fn latent_feature(&self) -> &[f32] {
        self.core.latent_feature()
    }

    /// Current output of the trainable adapter between DINO and the RSSM.
    pub fn encoded_observation(&self) -> &[f32] {
        self.core.encoded_observation()
    }

    /// Reward predicted from the current posterior state.
    pub fn posterior_reward_prediction(&mut self) -> f32 {
        self.core.posterior_reward_prediction()
    }

    /// One-step prior reward prediction for a proposed categorical action.
    pub fn prior_reward_prediction(&mut self, action: usize) -> f32 {
        self.core.prior_reward_prediction(action)
    }

    /// Open-loop prior reward predictions for a proposed action sequence.
    pub fn prior_reward_rollout(&mut self, actions: &[usize]) -> Vec<f32> {
        self.core.prior_reward_rollout(actions)
    }

    /// Current posterior policy after optional action masking and actor
    /// unimix. This does not sample or alter the live recurrent state.
    pub fn posterior_action_probabilities(&mut self, action_mask: Option<&[bool]>) -> Vec<f32> {
        self.core.posterior_action_probabilities(action_mask)
    }

    pub fn begin_episode(&mut self, frame: &RgbFrame) {
        let observation =
            self.perception
                .encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
        self.core.begin_episode(observation);
    }

    pub fn act(&mut self, mode: ActionMode, action_mask: Option<&[bool]>) -> usize {
        self.core.act(mode, action_mask)
    }

    pub fn observe(&mut self, transition: &Transition) {
        let observation = self.perception.encode_frame_rgb8(
            transition.frame.pixels(),
            transition.frame.width(),
            transition.frame.height(),
        );
        self.core
            .observe(observation, transition.reward, transition.flags());
    }

    pub fn learn(&mut self) -> Option<LearnReport> {
        self.core.learn()
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        self.core.learn_scheduled(maximum_updates)
    }

    pub fn save_checkpoint(&mut self, checkpoint: impl AsRef<Path>) -> io::Result<()> {
        self.core.save_checkpoint(checkpoint)
    }
}

fn read_checkpoint_metadata(checkpoint: &Path) -> io::Result<CheckpointMetadata> {
    let encoded = fs::read(checkpoint.join(CHECKPOINT_METADATA))?;
    serde_json::from_slice(&encoded)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn validate_checkpoint_metadata(metadata: &CheckpointMetadata) -> io::Result<()> {
    let expected = [
        (
            "architecture",
            CHECKPOINT_ARCHITECTURE,
            metadata.architecture.as_str(),
        ),
        (
            "DreamerV3 revision",
            DREAMERV3_UPSTREAM_REV,
            metadata.dreamerv3_revision.as_str(),
        ),
        (
            "Meganeura revision",
            MEGANEURA_REV,
            metadata.meganeura_revision.as_str(),
        ),
        (
            "Blade revision",
            BLADE_REV,
            metadata.blade_revision.as_str(),
        ),
        (
            "DINOv3 revision",
            DINOV3_UPSTREAM_REV,
            metadata.dinov3_revision.as_str(),
        ),
        (
            "DINO transcription revision",
            DINOVISION_SOURCE_REV,
            metadata.dinovision_revision.as_str(),
        ),
        (
            "DINO model",
            VITS16_MODEL_ID,
            metadata.dino_model_id.as_str(),
        ),
        (
            "DINO checkpoint revision",
            VITS16_CHECKPOINT_REV,
            metadata.dino_checkpoint_revision.as_str(),
        ),
    ];
    if metadata.format != CHECKPOINT_FORMAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported Kindle checkpoint format {}, expected {}",
                metadata.format, CHECKPOINT_FORMAT
            ),
        ));
    }
    for (name, wanted, actual) in expected {
        if actual != wanted {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint {name} is {actual:?}, expected {wanted:?}"),
            ));
        }
    }
    let expected_vision = [
        ("projection seed", PROJECTION_SEED, metadata.projection_seed),
        (
            "observation grid",
            OBSERVATION_GRID as u64,
            metadata.observation_grid as u64,
        ),
        (
            "observation channels",
            OBSERVATION_CHANNELS as u64,
            metadata.observation_channels as u64,
        ),
    ];
    for (name, wanted, actual) in expected_vision {
        if actual != wanted {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint {name} is {actual}, expected {wanted}"),
            ));
        }
    }
    metadata
        .config
        .check()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn keep_masks(
    batch: &SequenceBatch,
    time: usize,
    config: &DreamerConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = config.network();
    let mut deter = vec![0.0; config.batch_size * size.deter];
    let mut stoch = vec![0.0; config.batch_size * size.stoch * size.classes];
    let mut action = vec![0.0; config.batch_size * config.action_count];
    for row in 0..config.batch_size {
        let keep = batch.keep(time, row);
        deter[row * size.deter..(row + 1) * size.deter].fill(keep);
        let width = size.stoch * size.classes;
        stoch[row * width..(row + 1) * width].fill(keep);
        action[row * config.action_count..(row + 1) * config.action_count].fill(keep);
    }
    (deter, stoch, action)
}

fn sample_latents(
    logits: &[f32],
    batch: usize,
    stoch: usize,
    classes: usize,
    unimix: f32,
    rng: &mut StdRng,
) -> Vec<f32> {
    assert_eq!(logits.len(), batch * stoch * classes);
    let mut output = vec![0.0; logits.len()];
    for row in 0..batch * stoch {
        let start = row * classes;
        let class = sample_logits(&logits[start..start + classes], unimix, rng);
        output[start + class] = 1.0;
    }
    output
}

fn join_features(deter: &[f32], stoch: &[f32], batch: usize, config: &DreamerConfig) -> Vec<f32> {
    let size = config.network();
    assert_eq!(deter.len(), batch * size.deter);
    let stoch_width = size.stoch * size.classes;
    assert_eq!(stoch.len(), batch * stoch_width);
    let mut output = vec![0.0; batch * config.feature_dim()];
    for row in 0..batch {
        let target = &mut output[row * config.feature_dim()..(row + 1) * config.feature_dim()];
        target[..size.deter].copy_from_slice(&deter[row * size.deter..(row + 1) * size.deter]);
        target[size.deter..].copy_from_slice(&stoch[row * stoch_width..(row + 1) * stoch_width]);
    }
    output
}

fn flatten_time(values: &[Vec<f32>]) -> Vec<f32> {
    let capacity = values.iter().map(Vec::len).sum();
    let mut output = Vec::with_capacity(capacity);
    for value in values {
        output.extend_from_slice(value);
    }
    output
}

fn decode_rows(logits: &[f32], rows: usize, bins: &TwoHotBins) -> Vec<f32> {
    assert_eq!(logits.len(), rows * bins.len());
    (0..rows)
        .map(|row| bins.decode_logits(&logits[row * bins.len()..(row + 1) * bins.len()]))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    assert!(!values.is_empty());
    let mut best = 0;
    for index in 1..values.len() {
        if values[index].total_cmp(&values[best]).is_gt() {
            best = index;
        }
    }
    best
}

fn read_scalar(session: &Session, index: usize) -> f32 {
    let mut output = [0.0];
    session.read_output_by_index(index, &mut output);
    output[0]
}

fn trainable_parameter_count(session: &Session) -> usize {
    session
        .param_names()
        .into_iter()
        .filter(|name| session.has_param_grad(name))
        .map(|name| session.param_size(name).expect("known parameter size"))
        .sum()
}

fn mean_nested(values: &[Vec<f32>]) -> f32 {
    let count = values.iter().map(Vec::len).sum::<usize>();
    if count == 0 {
        0.0
    } else {
        values.iter().flatten().sum::<f32>() / count as f32
    }
}

fn mean(values: &[f32]) -> f32 {
    assert!(!values.is_empty());
    values.iter().sum::<f32>() / values.len() as f32
}

fn mean_absolute_error(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    assert!(!left.is_empty());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / left.len() as f32
}

fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn categorical_latents_are_one_hot_per_variable() {
        let mut rng = StdRng::seed_from_u64(3);
        let sampled = sample_latents(&[0.0; 2 * 3 * 4], 2, 3, 4, 0.01, &mut rng);
        for row in sampled.as_chunks::<4>().0 {
            assert_eq!(row.iter().sum::<f32>(), 1.0);
            assert_eq!(row.iter().filter(|value| **value == 1.0).count(), 1);
        }
    }

    #[test]
    fn d3_scheduler_discards_prefill_credit() {
        let mut scheduler = D3TrainScheduler::default();
        for _ in 0..100 {
            scheduler.observe(0.25);
        }
        assert!(!scheduler.update_due(false));
        assert!(scheduler.update_due(true));
        scheduler.consume_update();
        assert!(!scheduler.update_due(true));

        for _ in 0..3 {
            scheduler.observe(0.25);
            assert!(!scheduler.update_due(true));
        }
        scheduler.observe(0.25);
        assert!(scheduler.update_due(true));
    }

    #[test]
    fn dreamer_rng_streams_advance_independently() {
        let mut advanced = DreamerRngs::new(7);
        let mut baseline = DreamerRngs::new(7);
        for _ in 0..100 {
            let _: u64 = advanced.imagination.random();
            let _: u64 = advanced.replay.random();
            let _: u64 = advanced.train_posterior.random();
        }
        assert_eq!(
            advanced.policy.random::<u64>(),
            baseline.policy.random::<u64>()
        );
        assert_eq!(
            advanced.live_posterior.random::<u64>(),
            baseline.live_posterior.random::<u64>()
        );
        assert_ne!(
            advanced.imagination.random::<u64>(),
            baseline.imagination.random::<u64>()
        );

        let mut left = DreamerRngs::diagnostic(7, 11, 13);
        let mut right = DreamerRngs::diagnostic(7, 11, 13);
        let mut next_step = DreamerRngs::diagnostic(7, 11, 14);
        assert_eq!(left.random::<u64>(), right.random::<u64>());
        assert_ne!(left.random::<u64>(), next_step.random::<u64>());
    }

    #[test]
    fn state_features_concatenate_deter_and_flat_categoricals() {
        let config = DreamerConfig::tiny(3);
        let size = config.network();
        let deter = vec![2.0; 2 * size.deter];
        let stoch = vec![1.0; 2 * size.stoch * size.classes];
        let feature = join_features(&deter, &stoch, 2, &config);
        assert_eq!(feature.len(), 2 * config.feature_dim());
        assert!(feature[..size.deter].iter().all(|value| *value == 2.0));
        assert!(
            feature[size.deter..config.feature_dim()]
                .iter()
                .all(|value| *value == 1.0)
        );
    }

    #[test]
    #[ignore = "builds and runs all Dreamer GPU sessions"]
    fn tiny_agent_completes_an_act_and_learn_cycle() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_length = 8;
        config.world_backprop_length = 4;
        config.dynamics_free_nats = Some(0.0);
        config.skip_full_optimize = true;
        let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, Arc::clone(&gpu));
        let (world_parameters, behavior_parameters) = agent.trainable_parameter_counts();
        assert!(world_parameters > 0 && behavior_parameters > 0);
        assert!(
            !agent
                .world_train
                .has_param_grad("behavior.value.layer0.weight"),
            "the world optimizer must not own critic parameters"
        );
        assert!(
            agent
                .world_train
                .has_param_grad("world.dynamics.core.dynin0.weight"),
            "replay value must retain a gradient path into the world model"
        );
        let observation = || DinoObservation::from_vec(vec![0.0; DinoObservation::LEN]);

        agent.begin_episode(observation());
        assert!(agent.posterior_reward_prediction().is_finite());
        let probabilities = agent.posterior_action_probabilities(None);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let masked = agent.posterior_action_probabilities(Some(&[true, false, true]));
        assert_eq!(masked[1], 0.0);
        assert!((masked.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for step in 0..8 {
            let action = agent.act(ActionMode::Greedy, None);
            assert!(action < 3);
            agent.observe(
                observation(),
                Reward {
                    extrinsic: step as f32,
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: step == 7,
                    is_terminal: step == 7,
                    ..FrameFlags::default()
                },
            );
        }

        let report = agent.learn().expect("nine frames fill one tiny sequence");
        assert_eq!(report.learner_step, 1);
        assert_eq!(report.replay_len, 9);
        assert!(report.world.total_loss.is_finite());
        assert!(report.world.raw_kl >= 0.0);
        assert!((report.world.dynamics_kl - report.world.raw_kl).abs() < 1e-6);
        assert!(report.world.representation_kl >= agent.config.free_nats);
        assert!(report.behavior.total_loss.is_finite());
        assert!(
            (report.world.reward_loss - (agent.config.value_bins as f32).ln()).abs() < 1e-3,
            "uniform initial reward logits should report full cross-entropy"
        );
        assert!(
            (report.behavior.policy_entropy - (agent.config.action_count as f32).ln()).abs() < 0.02,
            "initial raw actor entropy should be close to uniform"
        );
        assert!(report.behavior.weighted_policy_entropy <= report.behavior.policy_entropy);

        let mut world_parameter = vec![0.0; config_value_bins(&agent)];
        let mut world_momentum = vec![0.0; config_value_bins(&agent)];
        let mut behavior_parameter = vec![0.0; 3];
        let mut slow_parameter = vec![0.0; config_value_bins(&agent)];
        agent
            .world_train
            .read_param("world.reward.out.bias", &mut world_parameter);
        agent
            .world_train
            .read_adam_m("world.reward.out.bias", &mut world_momentum);
        agent
            .behavior_train
            .read_param("behavior.actor.out.bias", &mut behavior_parameter);
        agent
            .behavior_slow
            .read_param("behavior.value.out.bias", &mut slow_parameter);
        let return_state = agent.return_normalizer.state();

        let checkpoint = std::env::temp_dir().join(format!(
            "kindle-dreamer-checkpoint-test-{}",
            std::process::id()
        ));
        if checkpoint.exists() {
            fs::remove_dir_all(&checkpoint).unwrap();
        }
        agent.save_checkpoint(&checkpoint).unwrap();
        drop(agent);

        let mut restored = DreamerCore::restore_with_gpu(&checkpoint, gpu).unwrap();
        assert_eq!(restored.learner_step(), 1);
        assert_eq!(restored.environment_step(), 8);
        assert_eq!(restored.replay_len(), 0);
        let mut restored_world = vec![0.0; config_value_bins(&restored)];
        let mut restored_momentum = vec![0.0; config_value_bins(&restored)];
        let mut restored_behavior = vec![0.0; 3];
        let mut restored_slow = vec![0.0; config_value_bins(&restored)];
        restored
            .world_train
            .read_param("world.reward.out.bias", &mut restored_world);
        restored
            .world_train
            .read_adam_m("world.reward.out.bias", &mut restored_momentum);
        restored
            .behavior_train
            .read_param("behavior.actor.out.bias", &mut restored_behavior);
        restored
            .behavior_slow
            .read_param("behavior.value.out.bias", &mut restored_slow);
        assert_eq!(restored_world, world_parameter);
        assert_eq!(restored_momentum, world_momentum);
        assert_eq!(restored_behavior, behavior_parameter);
        assert_eq!(restored_slow, slow_parameter);
        assert_eq!(restored.return_normalizer.state(), return_state);

        restored.begin_episode(observation());
        for step in 0..8 {
            restored.act(ActionMode::Greedy, None);
            restored.observe(
                observation(),
                Reward::default(),
                FrameFlags {
                    is_last: step == 7,
                    is_terminal: step == 7,
                    ..FrameFlags::default()
                },
            );
        }
        assert_eq!(
            restored
                .learn()
                .expect("refilled replay learns")
                .learner_step,
            2
        );
        drop(restored);
        fs::remove_dir_all(checkpoint).unwrap();
    }

    #[test]
    #[ignore = "runs a sustained tiny-core GPU learning diagnostic"]
    fn tiny_core_learns_action_conditioned_reward() {
        let mut config = DreamerConfig::tiny(2);
        config.learning_rate = 1e-3;
        config.learning_rate_warmup = 0;
        let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, gpu);
        let observation = || DinoObservation::from_vec(vec![0.0; DinoObservation::LEN]);

        for _ in 0..512 {
            agent.begin_episode(observation());
            let action = agent.act(ActionMode::Sample, None);
            agent.observe(
                observation(),
                Reward {
                    extrinsic: f32::from(action == 1),
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: true,
                    is_terminal: true,
                    ..Default::default()
                },
            );
            let _ = agent.learn();
        }

        let mut optimal = 0;
        for _ in 0..100 {
            agent.begin_episode(observation());
            let action = agent.act(ActionMode::Greedy, None);
            optimal += usize::from(action == 1);
            agent.observe(
                observation(),
                Reward {
                    extrinsic: f32::from(action == 1),
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: true,
                    is_terminal: true,
                    ..FrameFlags::default()
                },
            );
        }
        assert!(
            optimal >= 90,
            "greedy policy chose the paying action {optimal}/100 times"
        );
    }

    #[test]
    #[ignore = "runs a sustained tiny-core GPU learning diagnostic"]
    fn tiny_core_learns_observation_conditioned_reward() {
        let mut config = DreamerConfig::tiny(2);
        // Make this a wiring canary rather than a small-scale benchmark: the
        // only trainable objective is reward prediction and every replay row
        // resets the RSSM, so class information can only arrive through the
        // observation encoder and posterior straight-through gradient.
        config.learning_rate = 1e-3;
        config.learning_rate_warmup = 0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.continuation = 0.0;
        config.loss_scales.dynamics = 0.0;
        config.loss_scales.representation = 0.0;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, gpu);
        let observation = |class: usize| {
            let mut values = vec![0.0; DinoObservation::LEN];
            for (index, value) in values.iter_mut().enumerate() {
                if index % 2 == class {
                    *value = 1.0;
                }
            }
            DinoObservation::from_vec(values)
        };

        let size = agent.config.network();
        for index in 0..512 {
            let class = index % 2;
            agent.replay.push(
                ReplayFrame {
                    observation: observation(class),
                    previous_action: None,
                    reward: Reward {
                        extrinsic: class as f32,
                        intrinsic: 0.0,
                    },
                    flags: FrameFlags {
                        is_first: true,
                        is_last: true,
                        is_terminal: true,
                    },
                    deter: vec![0.0; size.deter].into_boxed_slice(),
                    stoch: vec![0.0; size.stoch * size.classes].into_boxed_slice(),
                },
                &agent.config,
            );
        }
        for update in 0..512 {
            let report = agent.learn().expect("pre-filled replay learns");
            if (update + 1) % 128 == 0 {
                eprintln!(
                    "update={} reward_loss={} rewarded={} unrewarded={}",
                    update + 1,
                    report.world.reward_loss,
                    report.world.rewarded_prediction_mean,
                    report.world.unrewarded_prediction_mean,
                );
            }
        }

        let predict = |agent: &mut DreamerCore, class: usize| {
            agent.posterior_live(observation(class).as_slice(), None, true);
            agent.posterior_reward_prediction()
        };
        let negative = (0..32).map(|_| predict(&mut agent, 0)).sum::<f32>() / 32.0;
        let positive = (0..32).map(|_| predict(&mut agent, 1)).sum::<f32>() / 32.0;
        assert!(
            positive > negative + 0.5,
            "observation-conditioned reward predictions did not separate: positive={positive}, negative={negative}"
        );
    }

    fn config_value_bins(agent: &DreamerCore) -> usize {
        agent.config.value_bins
    }

    #[test]
    #[ignore = "builds the full 12M baseline GPU sessions"]
    fn default_agent_sessions_build() {
        let agent = DreamerCore::new(DreamerConfig::new(4)).unwrap();
        assert_eq!(agent.config().model_size, super::super::ModelSize::Size12M);
    }
}
