//! Fixed-data world learning without actor, critic, imitation or exploration.

use std::{io, path::Path, sync::Arc};

use meganeura::{Graph, Mode, Session};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{
    DreamerConfig,
    agent::{join_features, sample_latents},
    checkpoint,
    distributions::{TwoHotBins, sample_probabilities},
    runtime, world,
};
use crate::{env::RgbFrame, vision::preprocess_gpu::GpuPreprocessor};

mod frozen_visual;
use frozen_visual::TrainingScope;
mod fixed_teacher;
use fixed_teacher::TeacherAnchor;

#[cfg(test)]
mod clean_prior_tests;
#[cfg(test)]
mod fixed_event_tests;
#[cfg(test)]
mod gaussian_tests;
#[cfg(test)]
mod gradient_diagnostic;
#[cfg(test)]
mod image_feedback_tests;
#[cfg(test)]
mod loss_gradient_diagnostic;
#[cfg(test)]
mod motion_tests;
#[cfg(test)]
mod observation_code_tests;
#[cfg(test)]
mod posterior_correction_diagnostic;
#[cfg(test)]
mod previous_frame_tests;
#[cfg(test)]
mod spatial_decoder_tests;
#[cfg(test)]
mod spatial_fusion_tests;
#[cfg(test)]
mod spatial_residual_tests;
#[cfg(test)]
mod state_diagnostic;
#[cfg(test)]
mod suffix_event_tests;
#[cfg(test)]
mod transport_tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelLoss {
    #[default]
    SquaredError,
    CrossEntropy,
    MotionCrossEntropy,
    MotionImageFeedback,
    MotionEventRollout,
    MotionCleanPrior,
    MotionLatentOvershooting,
    MotionRecurrentEnergy,
    MotionFixedTeacher,
    MotionFixedTeacherImage,
    MotionFixedTeacherEvents,
    PredictiveMotionEnergy,
    PredictiveMotionEnergyScoreFunction,
}

impl PixelLoss {
    pub(crate) fn motion_attention(self) -> bool {
        matches!(
            self,
            Self::MotionCrossEntropy
                | Self::MotionImageFeedback
                | Self::MotionEventRollout
                | Self::MotionCleanPrior
                | Self::MotionLatentOvershooting
                | Self::MotionRecurrentEnergy
                | Self::MotionFixedTeacher
                | Self::MotionFixedTeacherImage
                | Self::MotionFixedTeacherEvents
                | Self::PredictiveMotionEnergy
                | Self::PredictiveMotionEnergyScoreFunction
        )
    }

    pub(crate) fn rollout_events(self) -> bool {
        matches!(
            self,
            Self::MotionImageFeedback | Self::MotionEventRollout | Self::MotionCleanPrior
        )
    }

    pub(crate) fn clean_prior(self) -> bool {
        self == Self::MotionCleanPrior
    }

    pub(crate) fn predictive(self) -> bool {
        matches!(
            self,
            Self::PredictiveMotionEnergy | Self::PredictiveMotionEnergyScoreFunction
        )
    }

    pub(crate) fn score_function(self) -> bool {
        self == Self::PredictiveMotionEnergyScoreFunction
    }

    pub(crate) fn latent_overshooting(self) -> bool {
        matches!(
            self,
            Self::MotionLatentOvershooting
                | Self::MotionFixedTeacher
                | Self::MotionFixedTeacherEvents
        )
    }

    pub(crate) fn recurrent_energy(self) -> bool {
        matches!(
            self,
            Self::MotionRecurrentEnergy | Self::MotionFixedTeacher | Self::MotionFixedTeacherEvents
        )
    }

    pub(crate) fn fixed_teacher_events(self) -> bool {
        self == Self::MotionFixedTeacherEvents
    }

    pub(crate) fn fixed_teacher(self) -> bool {
        matches!(
            self,
            Self::MotionFixedTeacher
                | Self::MotionFixedTeacherImage
                | Self::MotionFixedTeacherEvents
        )
    }
}

fn checkpoint_pixel_loss(
    metadata: &serde_json::Value,
) -> Result<PixelLoss, Box<dyn std::error::Error>> {
    let transport = metadata["config"]["rgb_spatial_transport"] == true;
    if transport != (metadata["format"] == "kindle-world-only-v17") {
        return Err("spatial transport requires checkpoint format v17 and its decoder flag".into());
    }
    if transport
        && (metadata["config"]["rgb_spatial_residual"] == true
            || metadata["config"]["rgb_spatial_fusion"] != true
            || metadata["pixel_loss"] != "motion_event_rollout"
            || metadata["prediction_length"].as_u64().unwrap_or(0) == 0)
    {
        return Err("invalid spatial transport checkpoint objective or geometry".into());
    }
    let residual = metadata["config"]["rgb_spatial_residual"] == true;
    if residual != (metadata["format"] == "kindle-world-only-v15") {
        return Err("spatial residual requires checkpoint format v15 and its decoder flag".into());
    }
    if residual
        && (metadata["config"]["rgb_spatial_fusion"] != true
            || metadata["pixel_loss"] != "motion_event_rollout"
            || metadata["prediction_length"].as_u64().unwrap_or(0) == 0)
    {
        return Err("invalid spatial residual checkpoint objective or geometry".into());
    }
    let clean_prior = metadata["pixel_loss"] == "motion_clean_prior";
    if clean_prior && metadata["format"] != "kindle-world-only-v14" {
        return Err("clean prior objective requires checkpoint format v14".into());
    }
    if metadata["format"] == "kindle-world-only-v14"
        && (!clean_prior
            || metadata["config"]["rgb_spatial_fusion"] != true
            || metadata["prediction_length"].as_u64().unwrap_or(0) == 0)
    {
        return Err("invalid clean prior checkpoint objective or geometry".into());
    }
    let event_rollout = metadata["pixel_loss"] == "motion_event_rollout";
    let fusion = metadata["config"]["rgb_spatial_fusion"]
        .as_bool()
        .unwrap_or(false);
    if event_rollout && metadata["format"] != "kindle-world-only-v13" && !residual && !transport {
        return Err("suffix event objective requires checkpoint format v13".into());
    }
    if metadata["format"] == "kindle-world-only-v13"
        && (!event_rollout || !fusion || metadata["prediction_length"].as_u64().unwrap_or(0) == 0)
    {
        return Err("invalid suffix event checkpoint objective or geometry".into());
    }
    if fusion
        && !matches!(
            metadata["format"].as_str(),
            Some(
                "kindle-world-only-v12"
                    | "kindle-world-only-v13"
                    | "kindle-world-only-v14"
                    | "kindle-world-only-v15"
                    | "kindle-world-only-v17"
            )
        )
    {
        return Err("spatial fusion requires checkpoint format v12".into());
    }
    if matches!(
        metadata["format"].as_str(),
        Some(
            "kindle-world-only-v12"
                | "kindle-world-only-v13"
                | "kindle-world-only-v14"
                | "kindle-world-only-v15"
                | "kindle-world-only-v17"
        )
    ) && (!fusion
        || metadata["config"]["rgb_spatial_decoder"] != true
        || metadata["config"]["world_image_feedback"] == true
        || (!event_rollout && !clean_prior && metadata["pixel_loss"] != "motion_cross_entropy"))
    {
        return Err("invalid spatial fusion checkpoint architecture or objective".into());
    }
    let feedback = metadata["config"]["world_image_feedback"]
        .as_bool()
        .unwrap_or(false);
    if feedback && metadata["format"] != "kindle-world-only-v11" {
        return Err("image feedback requires checkpoint format v11".into());
    }
    if (metadata["format"] == "kindle-world-only-v11"
        && (!feedback
            || metadata["config"]["rgb_spatial_decoder"] != true
            || metadata["pixel_loss"] != "motion_image_feedback"))
        || (metadata["pixel_loss"] == "motion_image_feedback" && !feedback)
    {
        return Err("invalid image feedback checkpoint architecture or objective".into());
    }
    let spatial = metadata["config"]["rgb_spatial_decoder"]
        .as_bool()
        .unwrap_or(false);
    if spatial && !feedback && !fusion && metadata["format"] != "kindle-world-only-v10" {
        return Err("spatial decoder requires checkpoint format v10".into());
    }
    if matches!(
        metadata["format"].as_str(),
        Some(
            "kindle-world-only-v10"
                | "kindle-world-only-v12"
                | "kindle-world-only-v13"
                | "kindle-world-only-v14"
                | "kindle-world-only-v15"
                | "kindle-world-only-v17"
        )
    ) && (!spatial
        || (!event_rollout && !clean_prior && metadata["pixel_loss"] != "motion_cross_entropy")
        || metadata["config"]["world_gaussian"] == true
        || metadata["config"]["rgb_previous_frame"] == true
        || metadata["config"]["rgb_latent_only"] == true)
    {
        return Err("invalid spatial decoder checkpoint architecture or objective".into());
    }
    let gaussian = metadata["config"]["world_gaussian"]
        .as_bool()
        .unwrap_or(false);
    let compact = metadata["config"]["world_gaussian_dim"].is_number();
    if !metadata["config"]["world_gaussian_dim"].is_null() && !compact {
        return Err("invalid Gaussian dimension".into());
    }
    if compact && (!gaussian || metadata["format"] != "kindle-world-only-v9") {
        return Err("compact Gaussian world requires checkpoint format v9".into());
    }
    if metadata["format"] == "kindle-world-only-v9"
        && (!compact
            || !gaussian
            || metadata["config"]["world_gaussian_dim"]
                .as_u64()
                .unwrap_or(0)
                == 0)
    {
        return Err("invalid compact Gaussian checkpoint architecture".into());
    }
    if gaussian && !compact && metadata["format"] != "kindle-world-only-v8" {
        return Err("Gaussian world requires checkpoint format v8".into());
    }
    if matches!(
        metadata["format"].as_str(),
        Some("kindle-world-only-v8" | "kindle-world-only-v9")
    ) && (!gaussian
        || metadata["pixel_loss"] != "motion_cross_entropy"
        || metadata["config"]["rgb_previous_frame"] == true
        || metadata["config"]["rgb_latent_only"] == true)
    {
        return Err("invalid Gaussian checkpoint architecture or objective".into());
    }
    let latent_only = metadata["config"]["rgb_latent_only"]
        .as_bool()
        .unwrap_or(false);
    let previous_frame = metadata["config"]["rgb_previous_frame"]
        .as_bool()
        .unwrap_or(false);
    if previous_frame && metadata["format"] != "kindle-world-only-v5" {
        return Err("previous-frame RGB requires world-only checkpoint format v5".into());
    }
    if metadata["format"] == "kindle-world-only-v5" && (!previous_frame || latent_only) {
        return Err("invalid previous-frame RGB checkpoint architecture".into());
    }
    if latent_only && metadata["format"] != "kindle-world-only-v4" {
        return Err("latent-only RGB requires world-only checkpoint format v4".into());
    }
    if metadata["format"] == "kindle-world-only-v4" && !latent_only {
        return Err("invalid latent-only RGB checkpoint architecture".into());
    }
    let fixed_events = metadata["pixel_loss"] == "motion_fixed_teacher_events";
    if fixed_events != (metadata["format"] == "kindle-world-only-v16") {
        return Err("fixed teacher events require world-only checkpoint format v16".into());
    }
    if fixed_events && (latent_only || previous_frame) {
        return Err("invalid fixed-teacher event checkpoint architecture".into());
    }
    let fixed_teacher = metadata["pixel_loss"] == "motion_fixed_teacher";
    let fixed_image = metadata["pixel_loss"] == "motion_fixed_teacher_image";
    if fixed_image && metadata["format"] != "kindle-world-only-v7" {
        return Err("fixed teacher image requires world-only checkpoint format v7".into());
    }
    if fixed_teacher && metadata["format"] != "kindle-world-only-v6" {
        return Err("fixed teacher requires world-only checkpoint format v6".into());
    }
    if metadata["format"] == "kindle-world-only-v6"
        && (!fixed_teacher || latent_only || previous_frame)
    {
        return Err("invalid fixed-teacher checkpoint architecture".into());
    }
    if metadata["format"] == "kindle-world-only-v7"
        && (!fixed_image || latent_only || previous_frame)
    {
        return Err("invalid fixed-teacher image checkpoint architecture".into());
    }
    match metadata["format"].as_str() {
        Some("kindle-world-only-v1") => Ok(metadata
            .get("pixel_loss")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_default()),
        Some(
            "kindle-world-only-v2"
            | "kindle-world-only-v3"
            | "kindle-world-only-v4"
            | "kindle-world-only-v5"
            | "kindle-world-only-v6"
            | "kindle-world-only-v7"
            | "kindle-world-only-v8"
            | "kindle-world-only-v9"
            | "kindle-world-only-v10"
            | "kindle-world-only-v11"
            | "kindle-world-only-v12"
            | "kindle-world-only-v13"
            | "kindle-world-only-v14"
            | "kindle-world-only-v15"
            | "kindle-world-only-v16"
            | "kindle-world-only-v17",
        ) => Ok(serde_json::from_value(
            metadata
                .get("pixel_loss")
                .ok_or("missing pixel objective")?
                .clone(),
        )?),
        _ => Err(io::Error::other("not a supported world-only checkpoint").into()),
    }
}

pub struct WorldPrediction {
    /// Centered CHW RGB, batch-major.
    pub posterior: Vec<f32>,
    /// Centered CHW RGB, future-time then batch order.
    pub observations: Vec<f32>,
    pub rewards: Vec<Vec<f32>>,
    pub reward_positive_probability: Vec<Vec<f32>>,
    /// Includes the configured continuation discount, as in the online learner.
    pub continuations: Vec<Vec<f32>>,
    /// Optional diagnostic readbacks, never inputs or learning targets.
    pub encoded: Vec<f32>,
    pub posterior_features: Vec<f32>,
    /// Self-observed imagined belief when image feedback is active; otherwise prior state.
    pub prior_features: Vec<f32>,
}

pub struct WorldModelLearner {
    config: DreamerConfig,
    train: Session,
    observe: Session,
    transition: Session,
    decoder: Session,
    heads: Session,
    feedback: Option<Session>,
    resize: Session,
    preprocessor: GpuPreprocessor,
    rng: StdRng,
    rollout_rng: StdRng,
    prediction_length: usize,
    pixel_loss: PixelLoss,
    training_scope: TrainingScope,
    updates: u64,
    dirty: bool,
    fixed_teacher: Option<TeacherAnchor>,
}

fn validate_config(config: &DreamerConfig) -> Result<(), &'static str> {
    if config.world_backprop_length != config.batch_length
        || config.world_microbatch_size() != config.batch_size
    {
        return Err("standalone world learner requires full-sequence BPTT and a complete batch");
    }
    if !config.observation_kind.is_rgb()
        || config.video_encoder.is_some()
        || config.rgb_frame_pair
        || config.observation_crop.is_some()
        || config.loss_scales.reconstruction <= 0.0
        || config.loss_scales.future_prediction != 0.0
        || config.cdp_centered
    {
        return Err("standalone pixel world requires plain RGB and reconstruction only");
    }
    if config.actor_critic_gradient
        || config.expert_imitation.is_some()
        || config.disagreement_bonus
        || config.visitation_bonus
        || config.loss_scales.replay_value != 0.0
        || config.loss_scales.policy != 0.0
        || config.loss_scales.value != 0.0
        || config.open_loop_supervision.is_some()
        || config.color_centroid_supervision.is_some()
        || config.prior_reward_scale.is_some()
        || config.prior_continuation_scale.is_some()
    {
        return Err("standalone world configuration contains behavior or auxiliary supervision");
    }
    Ok(())
}

fn validate_history(
    config: &DreamerConfig,
    observations: &[f32],
    actions: &[i32],
) -> Result<usize, &'static str> {
    let batch = config.batch_size;
    if actions.is_empty() || !actions.len().is_multiple_of(batch) {
        return Err("expected complete time-major action batches");
    }
    let length = actions.len() / batch;
    if length > 256 || observations.len() != actions.len() * config.observation_dim() {
        return Err("observation/history shape mismatch");
    }
    if observations
        .iter()
        .any(|x| !x.is_finite() || !(-0.5..=0.5).contains(x))
    {
        return Err("RGB must contain finite centered GPU-resized pixels");
    }
    if actions
        .iter()
        .any(|&a| a < -1 || a >= config.action_count as i32)
        || actions[..batch].iter().any(|&a| a != -1)
    {
        return Err("history starts from reset (-1); other actions must be valid");
    }
    Ok(length)
}

fn action_inputs(
    config: &DreamerConfig,
    actions: &[i32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = config.network();
    let mut one_hot = vec![0.0; actions.len() * config.action_count];
    for (row, &action) in actions.iter().enumerate() {
        if action >= 0 {
            one_hot[row * config.action_count + action as usize] = 1.0;
        }
    }
    let mask = |width| {
        actions
            .iter()
            .flat_map(|&a| std::iter::repeat_n(f32::from(a >= 0), width))
            .collect()
    };
    (
        one_hot,
        mask(size.deter),
        mask(size.stoch * size.classes),
        mask(config.action_count),
    )
}

/// Select the preceding resized image in the same lane, duplicating the current
/// image at a clipped start or episode reset. No future image is read.
fn previous_images(observations: &[f32], actions: &[i32], time: usize, width: usize) -> Vec<f32> {
    let batch = actions.len();
    let start = time * batch * width;
    let current = &observations[start..start + batch * width];
    if time == 0 {
        return current.to_vec();
    }
    let mut previous = observations[start - batch * width..start].to_vec();
    for (row, &action) in actions.iter().enumerate() {
        if action == -1 {
            previous[row * width..(row + 1) * width]
                .copy_from_slice(&current[row * width..(row + 1) * width]);
        }
    }
    previous
}

fn rollout_weights(actions: &[i32], batch: usize, horizon: usize) -> Vec<Vec<f32>> {
    let mut keep = vec![1.0; batch];
    actions[actions.len() - horizon * batch..]
        .chunks_exact(batch)
        .map(|row| {
            for (keep, &action) in keep.iter_mut().zip(row) {
                if action == -1 {
                    *keep = 0.0;
                }
            }
            keep.clone()
        })
        .collect()
}

fn validate_targets(
    actions: &[i32],
    rewards: &[f32],
    terminals: &[bool],
    batch: usize,
) -> Result<(), &'static str> {
    for (i, &terminal) in terminals.iter().enumerate() {
        // A clipped context clears history but can start at a real terminal or reward.
        // Later -1 arrivals are actual episode resets, which carry neither.
        if (i >= batch && actions[i] == -1 && (terminal || rewards[i] != 0.0))
            || (terminal && i + batch < actions.len() && actions[i + batch] != -1)
        {
            return Err("terminal/reset alignment mismatch");
        }
    }
    Ok(())
}

impl WorldModelLearner {
    pub fn new(config: DreamerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_prediction_length(config, 0)
    }

    pub fn with_prediction_length(
        config: DreamerConfig,
        prediction_length: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_pixel_loss(config, prediction_length, PixelLoss::SquaredError)
    }

    pub fn with_pixel_loss(
        config: DreamerConfig,
        prediction_length: usize,
        pixel_loss: PixelLoss,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if pixel_loss.fixed_teacher() {
            return Err("fixed teacher requires an explicit source phase".into());
        }
        Self::with_pixel_loss_inner(config, prediction_length, pixel_loss, None)
    }

    fn with_pixel_loss_inner(
        config: DreamerConfig,
        prediction_length: usize,
        pixel_loss: PixelLoss,
        fixed_teacher: Option<TeacherAnchor>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if pixel_loss.fixed_teacher() != fixed_teacher.is_some()
            || (pixel_loss.fixed_teacher()
                && (prediction_length != 16 || config.rgb_previous_frame || config.rgb_latent_only))
        {
            return Err("invalid fixed teacher objective, source or geometry".into());
        }
        if config.world_gaussian && pixel_loss != PixelLoss::MotionCrossEntropy {
            return Err("Gaussian world requires the motion RGB objective without a phase".into());
        }
        if config.rgb_spatial_transport
            && (pixel_loss != PixelLoss::MotionEventRollout
                || prediction_length == 0
                || !config.rgb_spatial_fusion
                || config.rgb_spatial_residual)
        {
            return Err("spatial transport requires the suffix event objective, fusion and positive prediction length without residual".into());
        }
        if config.rgb_spatial_residual
            && (pixel_loss != PixelLoss::MotionEventRollout || prediction_length == 0)
        {
            return Err("spatial residual requires the suffix event objective and positive prediction length".into());
        }
        if config.world_image_feedback != (pixel_loss == PixelLoss::MotionImageFeedback)
            || (config.world_image_feedback && prediction_length == 0)
        {
            return Err(
                "image feedback requires its explicit objective and positive prediction length"
                    .into(),
            );
        }
        if matches!(
            pixel_loss,
            PixelLoss::MotionEventRollout | PixelLoss::MotionCleanPrior
        ) && (!config.rgb_spatial_fusion || prediction_length == 0)
        {
            return Err(
                "suffix event objective requires spatial fusion and a positive prediction length"
                    .into(),
            );
        }
        if config.rgb_spatial_decoder
            && !matches!(
                pixel_loss,
                PixelLoss::MotionCrossEntropy
                    | PixelLoss::MotionImageFeedback
                    | PixelLoss::MotionEventRollout
                    | PixelLoss::MotionCleanPrior
            )
        {
            return Err("spatial decoder requires the motion RGB objective without a phase".into());
        }
        validate_config(&config)?;
        config.check().map_err(io::Error::other)?;
        if pixel_loss != PixelLoss::SquaredError
            && config.reconstruction_target_standardization.is_some()
        {
            return Err("pixel cross-entropy requires unstandardized RGB targets".into());
        }
        if prediction_length >= config.batch_length {
            return Err("prediction length must leave an observed prefix".into());
        }
        if (pixel_loss.latent_overshooting() || pixel_loss.recurrent_energy())
            && prediction_length == 0
        {
            return Err("state rollout objective requires a positive prediction length".into());
        }
        if pixel_loss.predictive() && prediction_length != 0 {
            return Err("predictive motion requires no suffix image objective".into());
        }
        let gpu = Arc::new(crate::init_gpu_context()?);
        let graph = match (pixel_loss, prediction_length) {
            (PixelLoss::SquaredError, 0) => world::build_world_only_training_graph(&config),
            (PixelLoss::SquaredError, length) => {
                world::build_world_only_rollout_training_graph(&config, length)
            }
            (
                PixelLoss::CrossEntropy
                | PixelLoss::MotionCrossEntropy
                | PixelLoss::MotionImageFeedback
                | PixelLoss::MotionEventRollout
                | PixelLoss::MotionCleanPrior
                | PixelLoss::MotionLatentOvershooting
                | PixelLoss::MotionRecurrentEnergy
                | PixelLoss::MotionFixedTeacher
                | PixelLoss::MotionFixedTeacherImage
                | PixelLoss::MotionFixedTeacherEvents
                | PixelLoss::PredictiveMotionEnergy
                | PixelLoss::PredictiveMotionEnergyScoreFunction,
                length,
            ) => world::build_world_only_pixel_training_graph(&config, length, pixel_loss),
        };
        let mut train = runtime::build_training_session(&graph, &gpu, &config);
        runtime::initialize_d3(&mut train, &graph, config.seed);
        if let Some(anchor) = &fixed_teacher {
            // Validate the complete logical checkpoint before copying its
            // matching weights into an independently allocated filter.
            checkpoint::load_session(&mut train, &anchor.path)?;
        }
        let batch = config.batch_size;
        let inference = |graph: Graph, train: &mut Session| {
            let mut session = runtime::build_session(&graph, &gpu, Mode::Inference, false);
            runtime::share_matching(train, &mut session, "world.");
            session
        };
        let observe = if let Some(anchor) = &fixed_teacher {
            let mut session = runtime::build_session(
                &world::build_observe_graph(&config, batch),
                &gpu,
                Mode::Inference,
                false,
            );
            runtime::sync_matching(&train, &mut session, "world.");
            anchor.verify(&session)?;
            session
        } else {
            inference(world::build_observe_graph(&config, batch), &mut train)
        };
        let transition = inference(world::build_transition_graph(&config, batch), &mut train);
        let decoder = inference(
            world::build_observation_prediction_graph(&config, batch),
            &mut train,
        );
        let heads = inference(world::build_head_graph(&config, batch), &mut train);
        let feedback = config.world_image_feedback.then(|| {
            inference(
                world::build_image_feedback_graph(&config, batch),
                &mut train,
            )
        });
        let mut graph = Graph::new();
        let pixels = graph.input("observation", &[batch, config.observation_dim()]);
        graph.set_outputs(vec![pixels]);
        let resize = runtime::build_session(&graph, &gpu, Mode::Inference, false);
        let preprocessor = GpuPreprocessor::rgb(gpu, batch, config.observation_grid(), None);
        let rng = StdRng::seed_from_u64(config.seed ^ 0x776f_726c_646f_6e6c);
        let rollout_rng = StdRng::seed_from_u64(config.seed ^ 0x7069_7865_6c72_6f6c);
        Ok(Self {
            config,
            train,
            observe,
            transition,
            decoder,
            heads,
            feedback,
            resize,
            preprocessor,
            rng,
            rollout_rng,
            prediction_length,
            pixel_loss,
            training_scope: TrainingScope::All,
            updates: 0,
            dirty: false,
            fixed_teacher,
        })
    }

    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn training_scope(&self) -> &'static str {
        self.training_scope.name()
    }
    pub fn phase_start_updates(&self) -> Option<u64> {
        self.training_scope.start_updates()
    }
    pub fn prediction_length(&self) -> usize {
        self.prediction_length
    }
    pub fn pixel_loss(&self) -> PixelLoss {
        self.pixel_loss
    }
    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.train.device_information())
    }

    pub fn gpu_memory_budget(&self) -> Option<crate::GpuMemoryBudget> {
        self.train
            .device_memory_stats()
            .map(|s| crate::GpuMemoryBudget {
                usage_bytes: s.usage_bytes,
                budget_bytes: s.budget_bytes,
            })
    }

    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = self
            .train
            .param_names()
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// One native resize, with no learned processing or optimizer execution.
    pub fn preprocess(&mut self, frames: &[RgbFrame]) -> Result<Vec<f32>, &'static str> {
        if frames.len() != self.config.batch_size {
            return Err("complete frame batch required");
        }
        let frames = frames
            .iter()
            .enumerate()
            .map(|(i, frame)| (i, frame.pixels(), frame.width(), frame.height()))
            .collect::<Vec<_>>();
        self.preprocessor.cpu_frames(&mut self.resize, &frames);
        self.resize.step();
        self.resize.wait();
        let mut pixels = vec![0.0; self.config.batch_size * self.config.observation_dim()];
        self.resize.read_output_by_index(0, &mut pixels);
        Ok(pixels)
    }

    /// Observations are arrivals; actions/rewards/terminals describe arrival's incoming transition.
    pub fn learn(
        &mut self,
        observations: &[f32],
        actions: &[i32],
        rewards: &[f32],
        terminals: &[bool],
    ) -> Result<Vec<f32>, &'static str> {
        let length = validate_history(&self.config, observations, actions)?;
        let batch = self.config.batch_size;
        if length != self.config.batch_length
            || rewards.len() != actions.len()
            || terminals.len() != actions.len()
            || rewards.iter().any(|x| !x.is_finite())
        {
            return Err("training batch shape or reward mismatch");
        }
        validate_targets(actions, rewards, terminals, batch)?;
        let weights = rollout_weights(actions, batch, self.prediction_length);
        let size = self.config.network();
        let teacher = if self.pixel_loss.fixed_teacher() {
            Some(self.fixed_teacher_targets(
                observations,
                actions,
                &fixed_teacher::sampling_seeds(self.config.seed, self.updates, batch),
            )?)
        } else {
            None
        };
        self.train
            .set_input("initial_deter", &vec![0.0; batch * size.deter]);
        self.train.set_input(
            "initial_stoch",
            &vec![0.0; batch * size.stoch * size.classes],
        );
        let bins = TwoHotBins::new(self.config.value_bins);
        for time in 0..length {
            let range = time * batch..(time + 1) * batch;
            let (action, deter, stoch, keep) = action_inputs(&self.config, &actions[range.clone()]);
            for (name, values) in [
                ("previous_action", action),
                ("keep_deter", deter),
                ("keep_stoch", stoch),
                ("keep_action", keep),
            ] {
                self.train.set_input(&format!("{name}_{time}"), &values);
            }
            let width = self.config.observation_dim();
            self.train.set_input(
                &format!("observation_{time}"),
                &observations[time * batch * width..(time + 1) * batch * width],
            );
            if self.config.rgb_previous_frame {
                self.train.set_input(
                    &format!("previous_observation_{time}"),
                    &previous_images(observations, &actions[range.clone()], time, width),
                );
            }
            if let Some((deter, stoch)) = &teacher {
                let dw = batch * size.deter;
                let sw = batch * size.stoch * size.classes;
                self.train.set_input(
                    &format!("fixed_teacher_deter_{time}"),
                    &deter[time * dw..(time + 1) * dw],
                );
                self.train.set_input(
                    &format!("fixed_teacher_stoch_{time}"),
                    &stoch[time * sw..(time + 1) * sw],
                );
            } else {
                let uniforms = super::gaussian::noise(&self.config, &mut self.rng);
                self.train.set_input(&format!("uniforms_{time}"), &uniforms);
            }
            if self.pixel_loss.clean_prior() {
                let uniforms = super::gaussian::noise(&self.config, &mut self.rollout_rng);
                self.train
                    .set_input(&format!("clean_prior_uniforms_{time}"), &uniforms);
            }
            if self.pixel_loss.predictive() {
                for (name, width) in [
                    ("predictive_uniforms", size.stoch * size.classes),
                    ("predictive_second_uniforms", size.stoch * size.classes),
                    ("predictive_reward_uniforms0", bins.len()),
                    ("predictive_reward_uniforms1", bins.len()),
                    ("predictive_continuation_uniforms0", 2),
                    ("predictive_continuation_uniforms1", 2),
                ] {
                    let uniforms = (0..batch * width)
                        .map(|_| self.rollout_rng.random_range(1e-6..1.0 - 1e-6))
                        .collect::<Vec<_>>();
                    self.train.set_input(&format!("{name}_{time}"), &uniforms);
                }
            }
            let mut target = vec![0.0; batch * bins.len()];
            for (row, &reward) in rewards[range.clone()].iter().enumerate() {
                bins.encode(
                    reward,
                    &mut target[row * bins.len()..(row + 1) * bins.len()],
                );
            }
            self.train
                .set_input(&format!("reward_target_{time}"), &target);
            if self.pixel_loss.predictive() {
                let sampled = target
                    .chunks_exact(bins.len())
                    .map(|row| {
                        let index = sample_probabilities(row, &mut self.rollout_rng);
                        let value = bins.values()[index];
                        value.signum() * value.abs().ln_1p()
                    })
                    .collect::<Vec<_>>();
                self.train
                    .set_input(&format!("predictive_reward_sample_{time}"), &sampled);
            }
            let target = terminals[range]
                .iter()
                .map(|&terminal| {
                    if terminal {
                        0.0
                    } else {
                        self.config.continuation_discount()
                    }
                })
                .collect::<Vec<_>>();
            self.train
                .set_input(&format!("continuation_target_{time}"), &target);
            if self.pixel_loss.predictive() {
                let sampled = target
                    .iter()
                    .map(|&p| f32::from(self.rollout_rng.random::<f32>() < p))
                    .collect::<Vec<_>>();
                self.train
                    .set_input(&format!("predictive_continuation_sample_{time}"), &sampled);
            }
        }
        for (time, weight) in weights.iter().enumerate() {
            if self.pixel_loss.recurrent_energy()
                && (time + 1 < weights.len() || self.pixel_loss.fixed_teacher_events())
            {
                for name in ["recurrent_first", "recurrent_second"] {
                    let uniforms = (0..batch * size.stoch * size.classes)
                        .map(|_| self.rollout_rng.random_range(1e-6..1.0 - 1e-6))
                        .collect::<Vec<_>>();
                    self.train
                        .set_input(&format!("{name}_uniforms_{time}"), &uniforms);
                }
            }
            if (self.pixel_loss.fixed_teacher() || !self.pixel_loss.recurrent_energy())
                && (!self.pixel_loss.latent_overshooting() || time + 1 < weights.len())
            {
                let uniforms = super::gaussian::noise(&self.config, &mut self.rollout_rng);
                let prefix = if self.pixel_loss.latent_overshooting() {
                    "latent_rollout"
                } else {
                    "pixel_rollout"
                };
                self.train
                    .set_input(&format!("{prefix}_uniforms_{time}"), &uniforms);
                if self.config.world_image_feedback {
                    let noise = super::gaussian::noise(&self.config, &mut self.rollout_rng);
                    self.train
                        .set_input(&format!("pixel_feedback_uniforms_{time}"), &noise);
                }
            }
            self.train
                .set_input(&format!("pixel_rollout_weight_{time}"), weight);
            if self.pixel_loss.fixed_teacher() && self.pixel_loss.recurrent_energy() {
                self.train
                    .set_input(&format!("fixed_teacher_recurrent_weight_{time}"), weight);
            }
        }
        runtime::configure_d3_optimizer(
            &mut self.train,
            &self.config,
            self.updates + 1,
            self.config.learning_rate,
        );
        self.training_scope.apply(&mut self.train);
        self.train.step();
        self.train.wait();
        let mut losses = vec![0.0; 9];
        for (index, loss) in losses.iter_mut().enumerate() {
            self.train
                .read_output_by_index(index, std::slice::from_mut(loss));
        }
        if self.pixel_loss.fixed_teacher() && self.pixel_loss.recurrent_energy() {
            let mut value = [0.0];
            self.train
                .read_output_by_index(world::LOSS_WORLD_ROLLOUT, &mut value);
            losses.push(value[0]);
            for index in world::LOSS_RECURRENT_WORLD..world::LOSS_RECURRENT_WORLD + 4 {
                self.train.read_output_by_index(index, &mut value);
                losses.push(value[0]);
            }
        } else if self.prediction_length > 0 && !self.pixel_loss.recurrent_energy() {
            let mut loss = [0.0];
            self.train
                .read_output_by_index(world::LOSS_WORLD_ROLLOUT, &mut loss);
            losses.push(loss[0]);
            losses.push(
                weights.iter().flatten().sum::<f32>() / (batch * self.prediction_length) as f32,
            );
        }
        if self.pixel_loss.predictive() {
            for index in world::LOSS_PREDICTIVE_WORLD..world::LOSS_PREDICTIVE_WORLD + 4 {
                let mut loss = [0.0];
                self.train.read_output_by_index(index, &mut loss);
                losses.push(loss[0]);
            }
        }
        if self.pixel_loss.rollout_events() || self.pixel_loss.fixed_teacher_events() {
            for index in world::LOSS_IMAGE_FEEDBACK..world::LOSS_IMAGE_FEEDBACK + 2 {
                let mut loss = [0.0];
                self.train.read_output_by_index(index, &mut loss);
                losses.push(loss[0]);
            }
        }
        if self.pixel_loss.clean_prior() {
            let mut fraction = [0.0];
            self.train
                .read_output_by_index(world::CLEAN_PRIOR_VALID_FRACTION, &mut fraction);
            losses.push(fraction[0]);
        }
        if self.pixel_loss.recurrent_energy() && !self.pixel_loss.fixed_teacher() {
            for index in world::LOSS_RECURRENT_WORLD..world::LOSS_RECURRENT_WORLD + 4 {
                let mut loss = [0.0];
                self.train.read_output_by_index(index, &mut loss);
                losses.push(loss[0]);
            }
        }
        if losses.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite standalone world loss");
        }
        self.updates += 1;
        self.dirty = true;
        Ok(losses)
    }

    /// No true future image argument exists. Feedback consumes only generated images.
    pub fn predict(
        &mut self,
        observations: &[f32],
        actions: &[i32],
        future: &[usize],
        seed: u64,
    ) -> Result<WorldPrediction, &'static str> {
        self.predict_with_features(observations, actions, future, seed, false)
    }

    pub fn predict_with_features(
        &mut self,
        observations: &[f32],
        actions: &[i32],
        future: &[usize],
        seed: u64,
        include_features: bool,
    ) -> Result<WorldPrediction, &'static str> {
        let length = validate_history(&self.config, observations, actions)?;
        let batch = self.config.batch_size;
        if future.is_empty()
            || !future.len().is_multiple_of(batch)
            || future.len() / batch > 64
            || future.iter().any(|&a| a >= self.config.action_count)
        {
            return Err("expected 1..64 complete future action batches");
        }
        if self.dirty {
            if self.fixed_teacher.is_none() {
                runtime::sync_matching(&self.train, &mut self.observe, "world.");
            }
            for session in [&mut self.transition, &mut self.decoder, &mut self.heads] {
                runtime::sync_matching(&self.train, session, "world.");
            }
            if let Some(session) = &mut self.feedback {
                runtime::sync_matching(&self.train, session, "world.");
            }
            self.dirty = false;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let size = self.config.network();
        let mut deter = vec![0.0; batch * size.deter];
        let mut stoch = vec![0.0; batch * size.stoch * size.classes];
        let mut logits = vec![0.0; batch * size.stoch * self.config.latent_parameter_classes()];
        let width = self.config.observation_dim();
        let mut encoded = if include_features {
            vec![0.0; batch * self.config.encoded_observation_dim()]
        } else {
            Vec::new()
        };
        for time in 0..length {
            let (action, keep_deter, keep_stoch, keep_action) =
                action_inputs(&self.config, &actions[time * batch..(time + 1) * batch]);
            for (name, values) in [
                ("previous_deter", &deter),
                ("previous_stoch", &stoch),
                ("previous_action", &action),
                ("keep_deter", &keep_deter),
                ("keep_stoch", &keep_stoch),
                ("keep_action", &keep_action),
            ] {
                self.observe.set_input(name, values);
            }
            self.observe.set_input(
                "observation",
                &observations[time * batch * width..(time + 1) * batch * width],
            );
            if self.config.rgb_previous_frame {
                self.observe.set_input(
                    "previous_observation",
                    &previous_images(
                        observations,
                        &actions[time * batch..(time + 1) * batch],
                        time,
                        width,
                    ),
                );
            }
            self.observe.step();
            self.observe.wait();
            self.observe.read_output_by_index(0, &mut deter);
            self.observe.read_output_by_index(1, &mut logits);
            if include_features && time + 1 == length {
                self.observe.read_output_by_index(3, &mut encoded);
            }
            stoch = super::gaussian::sample_cpu(&self.config, &logits, &mut rng);
        }
        let decode = |session: &mut Session, deter: &[f32], stoch: &[f32], history: &[f32]| {
            session.set_input("deter", deter);
            session.set_input("stoch", stoch);
            if self.config.rgb_spatial_decoder {
                session.set_input("decoder_history", history);
            }
            session.step();
            session.wait();
            let mut pixels = vec![0.0; batch * width];
            session.read_output_by_index(0, &mut pixels);
            pixels
        };
        let posterior_history = if self.config.rgb_spatial_decoder {
            super::spatial_history::before_cpu(observations, actions, length - 1, batch, width)
        } else {
            Vec::new()
        };
        let posterior = decode(&mut self.decoder, &deter, &stoch, &posterior_history);
        let mut history = if self.config.rgb_spatial_decoder {
            super::spatial_history::after_cpu(observations, actions, length - 1, batch, width)
        } else {
            Vec::new()
        };
        let mut result = WorldPrediction {
            posterior,
            observations: Vec::new(),
            rewards: Vec::new(),
            reward_positive_probability: Vec::new(),
            continuations: Vec::new(),
            encoded,
            posterior_features: if include_features {
                join_features(&deter, &stoch, batch, &self.config)
            } else {
                Vec::new()
            },
            prior_features: Vec::new(),
        };
        let bins = TwoHotBins::new(self.config.value_bins);
        for actions in future.chunks_exact(batch) {
            let actions = actions.iter().map(|&a| a as i32).collect::<Vec<_>>();
            let (action, _, _, _) = action_inputs(&self.config, &actions);
            self.transition.set_input("deter", &deter);
            self.transition.set_input("stoch", &stoch);
            self.transition.set_input("action", &action);
            self.transition.step();
            self.transition.wait();
            self.transition.read_output_by_index(0, &mut deter);
            self.transition.read_output_by_index(1, &mut logits);
            stoch = super::gaussian::sample_cpu(&self.config, &logits, &mut rng);
            let image = decode(&mut self.decoder, &deter, &stoch, &history);
            if let Some(session) = &mut self.feedback {
                session.set_input("deter", &deter);
                session.set_input("generated_image", &image);
                session.step();
                session.wait();
                session.read_output_by_index(0, &mut logits);
                stoch = super::gaussian::sample_cpu(&self.config, &logits, &mut rng);
            }
            if self.config.rgb_spatial_decoder {
                history = super::spatial_history::shift_cpu(&history, &image, width);
            }
            result.observations.extend(image);
            if include_features {
                result
                    .prior_features
                    .extend(join_features(&deter, &stoch, batch, &self.config));
            }
            self.heads.set_input("deter", &deter);
            self.heads.set_input("stoch", &stoch);
            self.heads.step();
            self.heads.wait();
            let mut reward_logits = vec![0.0; batch * bins.len()];
            let mut continuation = vec![0.0; batch];
            self.heads.read_output_by_index(0, &mut reward_logits);
            self.heads.read_output_by_index(1, &mut continuation);
            let mut scratch = vec![0.0; bins.len()];
            let mut rewards = Vec::new();
            let mut positive = Vec::new();
            for row in reward_logits.chunks_exact(bins.len()) {
                rewards.push(bins.decode_logits_with_scratch(row, &mut scratch));
                positive.push(
                    (scratch[bins.len() / 2 + 1..].iter().sum::<f64>()
                        / scratch.iter().sum::<f64>()) as f32,
                );
            }
            result.rewards.push(rewards);
            result.reward_positive_probability.push(positive);
            result.continuations.push(continuation);
        }
        if result
            .posterior
            .iter()
            .chain(&result.observations)
            .chain(result.rewards.iter().flatten())
            .chain(result.continuations.iter().flatten())
            .chain(result.reward_positive_probability.iter().flatten())
            .chain(&result.encoded)
            .chain(&result.posterior_features)
            .chain(&result.prior_features)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite standalone forecast");
        }
        Ok(result)
    }

    pub fn save(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if self.train.memory_summary().adam_state_bytes == 0 {
            if self.train.adam_step_count() != 0 {
                return Err(io::Error::other("stepped world optimizer has no moments").into());
            }
            runtime::configure_d3_optimizer(&mut self.train, &self.config, 0, 0.0);
            self.train.clear_optimizer();
        }
        std::fs::create_dir(path)?;
        self.train
            .save_checkpoint(&path.join("world.safetensors"))?;
        let format = if self.config.rgb_spatial_transport {
            "kindle-world-only-v17"
        } else if self.pixel_loss.fixed_teacher_events() {
            "kindle-world-only-v16"
        } else if self.config.rgb_spatial_residual {
            "kindle-world-only-v15"
        } else if self.pixel_loss.clean_prior() {
            "kindle-world-only-v14"
        } else if self.pixel_loss == PixelLoss::MotionEventRollout {
            "kindle-world-only-v13"
        } else if self.config.rgb_spatial_fusion {
            "kindle-world-only-v12"
        } else if self.config.world_image_feedback {
            "kindle-world-only-v11"
        } else if self.config.rgb_spatial_decoder {
            "kindle-world-only-v10"
        } else if self.config.world_gaussian_dim.is_some() {
            "kindle-world-only-v9"
        } else if self.config.world_gaussian {
            "kindle-world-only-v8"
        } else if self.pixel_loss == PixelLoss::MotionFixedTeacherImage {
            "kindle-world-only-v7"
        } else if self.config.rgb_previous_frame {
            assert_eq!(self.training_scope, TrainingScope::All);
            "kindle-world-only-v5"
        } else if self.config.rgb_latent_only {
            assert_eq!(self.training_scope, TrainingScope::All);
            "kindle-world-only-v4"
        } else {
            self.training_scope.format()
        };
        let mut metadata = serde_json::json!({"format": format, "config": self.config, "updates": self.updates, "prediction_length": self.prediction_length, "pixel_loss": self.pixel_loss});
        if let Some(anchor) = &self.fixed_teacher {
            let (parameters, scalars) = anchor.verify(&self.observe)?;
            std::fs::copy(&anchor.path, path.join("teacher.safetensors"))?;
            metadata["teacher_sha256"] = serde_json::json!(anchor.sha256);
            metadata["teacher_parameters"] = serde_json::json!(parameters);
            metadata["teacher_scalars"] = serde_json::json!(scalars);
            metadata["teacher_session_allocation_bytes"] =
                serde_json::json!(self.observe.memory_summary().total_allocated_bytes());
        }
        if let Some(start) = self.training_scope.start_updates() {
            metadata["training_scope"] = serde_json::json!(self.training_scope.name());
            metadata["phase_start_updates"] = serde_json::json!(start);
        }
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    pub fn restore(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json"))?)?;
        let pixel_loss = checkpoint_pixel_loss(&metadata)?;
        let config = serde_json::from_value(metadata["config"].clone())?;
        let prediction_length = match metadata.get("prediction_length") {
            None => 0,
            Some(value) => value.as_u64().ok_or("invalid world prediction length")?,
        };
        let prediction_length = usize::try_from(prediction_length)?;
        let updates = metadata["updates"].as_u64().ok_or("missing update count")?;
        let scope =
            TrainingScope::from_metadata(&metadata, pixel_loss, prediction_length, updates)?;
        let teacher = if pixel_loss.fixed_teacher() {
            Some(TeacherAnchor::restore(path, &metadata)?)
        } else {
            None
        };
        let mut learner =
            Self::with_pixel_loss_inner(config, prediction_length, pixel_loss, teacher)?;
        learner.load_training_checkpoint(path, updates)?;
        learner.training_scope = scope;
        Ok(learner)
    }

    /// Start a distinct H16 phase from an experienced H0 motion world. Rebuild
    /// the training graph while preserving all weights, moments and optimizer age.
    /// Ordinary restore never changes objectives or training scope.
    pub fn start_frozen_visual_phase(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json"))?)?;
        let pixel_loss = checkpoint_pixel_loss(&metadata)?;
        if metadata["config"]["rgb_spatial_decoder"].as_bool() == Some(true) {
            return Err("frozen visual phase requires the original RGB decoder".into());
        }
        if metadata["config"]["world_gaussian"].as_bool() == Some(true) {
            return Err("frozen visual phase requires categorical latents".into());
        }
        if metadata["config"]["rgb_previous_frame"].as_bool() == Some(true) {
            return Err("frozen visual phase requires single-frame RGB".into());
        }
        if metadata["config"]["rgb_latent_only"].as_bool() == Some(true) {
            return Err("frozen visual phase requires recurrent-conditioned RGB".into());
        }
        let prediction_length = metadata
            .get("prediction_length")
            .map(|value| value.as_u64().ok_or("invalid world prediction length"))
            .transpose()?
            .unwrap_or(0);
        let prediction_length = usize::try_from(prediction_length)?;
        let updates = metadata["updates"].as_u64().ok_or("missing update count")?;
        let scope =
            TrainingScope::from_metadata(&metadata, pixel_loss, prediction_length, updates)?;
        let phase = scope.start_frozen_visual(pixel_loss, prediction_length, updates)?;
        let config = serde_json::from_value(metadata["config"].clone())?;
        let mut learner = Self::with_pixel_loss(config, 16, pixel_loss)?;
        learner.load_training_checkpoint(path, updates)?;
        learner.training_scope = phase;
        Ok(learner)
    }

    fn load_training_checkpoint(
        &mut self,
        path: &Path,
        updates: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        checkpoint::load_session(&mut self.train, &path.join("world.safetensors"))?;
        if u64::from(self.train.adam_step_count()) != updates {
            return Err(io::Error::other("world checkpoint optimizer age mismatch").into());
        }
        self.updates = updates;
        self.dirty = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dreamer::ObservationKind;
    use meganeura::graph::Op;

    fn pixel_loss_fixture() -> (Graph, Vec<f32>, Vec<f32>, Vec<f32>) {
        let logits = vec![-80., -10., -1., 0., 1., 10., 80., -8., 8., -4., 4., 0.];
        let targets = vec![
            0.5, -0.5, 0.2, 0., -0.2, 0.5, -0.5, 0.5, -0.5, 0.25, -0.25, 0.5,
        ];
        let weights = vec![1., 0., 1.];
        let mut graph = Graph::new();
        let x = graph.parameter("logits", &[3, 4]);
        let y = graph.input("target", &[3, 4]);
        let elements = super::super::networks::pixel_cross_entropy(&mut graph, x, y);
        let rows = graph.sum_inner(elements);
        let mask = graph.constant(weights.clone(), &[3, 1]);
        let masked = graph.mul(rows, mask);
        let loss = graph.mean_all(masked);
        graph.set_outputs(vec![loss]);
        (graph, logits, targets, weights)
    }

    fn expected_pixel_loss(logits: &[f32], targets: &[f32], weights: &[f32]) -> (f64, Vec<f64>) {
        let mut loss = 0.;
        let mut gradient = Vec::new();
        for (i, (&x, &y)) in logits.iter().zip(targets).enumerate() {
            let x = f64::from(x);
            let y = f64::from(y) + 0.5;
            let weight = f64::from(weights[i / 4]) / 3.;
            loss += (x.max(0.) + (-x.abs()).exp().ln_1p() - x * y) * weight;
            gradient.push((1. / (1. + (-x).exp()) - y) * weight);
        }
        (loss, gradient)
    }

    #[test]
    fn pixel_cross_entropy_matches_f64_values_and_derivatives() {
        use meganeura::reference::{Feeds, evaluate};
        let (graph, logits, targets, weights) = pixel_loss_fixture();
        let backward = meganeura::autodiff::differentiate(&graph);
        let mut feeds = Feeds::new();
        feeds.set("logits", &logits).set("target", &targets);
        let values = evaluate(&backward, &feeds).unwrap();
        let (loss, gradient) = expected_pixel_loss(&logits, &targets, &weights);
        assert!((values[backward.outputs()[0] as usize].data[0] - loss).abs() < 1e-10);
        for (actual, expected) in values[backward.outputs()[1] as usize]
            .data
            .iter()
            .zip(gradient)
        {
            // Autodiff stores the mean's reciprocal as an F32 scale constant.
            assert!((actual - expected).abs() < 2e-8, "{actual} vs {expected}");
        }
    }

    #[test]
    #[ignore = "requires guarded GPU; pixel cross-entropy values and derivatives"]
    fn pixel_cross_entropy_native_matches_f64() {
        let (graph, logits, targets, weights) = pixel_loss_fixture();
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        let mut session = runtime::build_session(&graph, &gpu, Mode::Training, false);
        session.set_parameter("logits", &logits);
        session.set_input("target", &targets);
        session.step();
        session.wait();
        let mut actual = [0.];
        let mut gradients = vec![0.; logits.len()];
        session.read_output_by_index(0, &mut actual);
        session.read_param_grad("logits", &mut gradients);
        let (loss, gradient) = expected_pixel_loss(&logits, &targets, &weights);
        assert!((f64::from(actual[0]) - loss).abs() < 2e-5);
        for (actual, expected) in gradients.iter().zip(gradient) {
            assert!(
                (f64::from(*actual) - expected).abs() < 2e-6,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn cross_entropy_keeps_parameter_shapes_and_has_no_behavior_dependencies() {
        let config = config();
        let graphs = [PixelLoss::SquaredError, PixelLoss::CrossEntropy]
            .map(|loss| world::build_world_only_pixel_training_graph(&config, 2, loss));
        let names = graphs.each_ref().map(|graph| {
            graph
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                    Op::Input { name } => {
                        assert!(!name.contains("value") && !name.contains("expert"));
                        None
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(names[0], names[1]);
        assert!(names[0].iter().all(|(name, _)| name.starts_with("world.")));
        meganeura::compile_training_graph(&graphs[1]);
        assert_eq!(
            serde_json::from_str::<PixelLoss>("\"cross_entropy\"").unwrap(),
            PixelLoss::CrossEntropy
        );
        assert!(serde_json::from_str::<PixelLoss>("\"unknown\"").is_err());
    }

    #[test]
    fn rollout_masks_keep_terminal_arrivals_and_ignore_all_later_episode_rows() {
        let actions = [-1, -1, 1, 2, 1, 2, -1, 2, 1, -1, 1, 2];
        assert_eq!(
            rollout_weights(&actions, 2, 4),
            vec![
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0]
            ]
        );
        assert!(rollout_weights(&actions, 2, 0).is_empty());
        assert_eq!(
            rollout_weights(&[-1, -1, 0, 2], 2, 2),
            vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
    }

    #[test]
    fn clipped_context_can_begin_with_a_real_terminal_or_reward() {
        assert!(validate_targets(&[-1, -1, 2], &[7.0, 0.0, 0.0], &[true, false, false], 1).is_ok());
        assert!(validate_targets(&[-1, 2, 2], &[0.0; 3], &[false, true, false], 1).is_err());
        assert!(validate_targets(&[-1, -1, 2], &[0.0, 1.0, 0.0], &[false; 3], 1).is_err());
    }

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        config.disagreement_bonus = false;
        config.actor_critic_gradient = false;
        config
    }

    #[test]
    fn standalone_graph_has_no_behavior_parameters_or_targets() {
        let config = config();
        validate_config(&config).unwrap();
        let graph = world::build_world_only_training_graph(&config);
        let mut parameters = Vec::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Parameter { name } => {
                    assert!(name.starts_with("world."), "{name}");
                    parameters.push(name);
                }
                Op::Input { name } => assert!(
                    !name.contains("value")
                        && !name.contains("expert")
                        && !name.contains("initial_action_target"),
                    "{name}"
                ),
                _ => (),
            }
        }
        for prefix in [
            "world.dynamics.",
            "world.representation.",
            "world.decoder.",
            "world.reward.",
            "world.continuation.",
        ] {
            assert!(
                parameters.iter().any(|name| name.starts_with(prefix)),
                "{prefix}"
            );
        }
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(
            plan.param_buffers
                .iter()
                .all(|(name, _)| name.starts_with("world."))
        );
    }

    #[test]
    fn history_rejects_future_style_alignment_and_invalid_pixels() {
        let config = config();
        let batch = config.batch_size;
        let mut pixels = vec![0.0; 2 * batch * config.observation_dim()];
        let mut actions = vec![-1; 2 * batch];
        actions[batch..].fill(1);
        assert_eq!(validate_history(&config, &pixels, &actions), Ok(2));
        actions[0] = 1;
        assert!(validate_history(&config, &pixels, &actions).is_err());
        actions[0] = -1;
        pixels[0] = f32::NAN;
        assert!(validate_history(&config, &pixels, &actions).is_err());
    }

    #[test]
    fn reset_masks_zero_only_the_reset_rows() {
        let config = config();
        let (actions, deter, stoch, keep) = action_inputs(&config, &[-1, 2]);
        assert_eq!(actions, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        for (mask, width) in [
            (deter, config.network().deter),
            (stoch, config.network().stoch * config.network().classes),
            (keep, 3),
        ] {
            assert!(mask[..width].iter().all(|&x| x == 0.0));
            assert!(mask[width..].iter().all(|&x| x == 1.0));
        }
    }
}
