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

#[cfg(test)]
mod gradient_diagnostic;
#[cfg(test)]
mod motion_tests;
#[cfg(test)]
mod state_diagnostic;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelLoss {
    #[default]
    SquaredError,
    CrossEntropy,
    MotionCrossEntropy,
    MotionLatentOvershooting,
    MotionRecurrentEnergy,
    PredictiveMotionEnergy,
    PredictiveMotionEnergyScoreFunction,
}

impl PixelLoss {
    pub(crate) fn motion_attention(self) -> bool {
        matches!(
            self,
            Self::MotionCrossEntropy
                | Self::MotionLatentOvershooting
                | Self::MotionRecurrentEnergy
                | Self::PredictiveMotionEnergy
                | Self::PredictiveMotionEnergyScoreFunction
        )
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
        self == Self::MotionLatentOvershooting
    }

    pub(crate) fn recurrent_energy(self) -> bool {
        self == Self::MotionRecurrentEnergy
    }
}

fn checkpoint_pixel_loss(
    metadata: &serde_json::Value,
) -> Result<PixelLoss, Box<dyn std::error::Error>> {
    match metadata["format"].as_str() {
        Some("kindle-world-only-v1") => Ok(metadata
            .get("pixel_loss")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_default()),
        Some("kindle-world-only-v2") => Ok(serde_json::from_value(
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
    pub prior_features: Vec<f32>,
}

pub struct WorldModelLearner {
    config: DreamerConfig,
    train: Session,
    observe: Session,
    transition: Session,
    decoder: Session,
    heads: Session,
    resize: Session,
    preprocessor: GpuPreprocessor,
    rng: StdRng,
    rollout_rng: StdRng,
    prediction_length: usize,
    pixel_loss: PixelLoss,
    updates: u64,
    dirty: bool,
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
        validate_config(&config)?;
        config.validate();
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
        if pixel_loss.motion_attention()
            && !pixel_loss.latent_overshooting()
            && !pixel_loss.recurrent_energy()
            && prediction_length != 0
        {
            return Err("motion pixel attention requires no prior-image objective".into());
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
                | PixelLoss::MotionLatentOvershooting
                | PixelLoss::MotionRecurrentEnergy
                | PixelLoss::PredictiveMotionEnergy
                | PixelLoss::PredictiveMotionEnergyScoreFunction,
                length,
            ) => world::build_world_only_pixel_training_graph(&config, length, pixel_loss),
        };
        let mut train = runtime::build_training_session(&graph, &gpu, &config);
        runtime::initialize_d3(&mut train, &graph, config.seed);
        let batch = config.batch_size;
        let mut inference = |graph: Graph| {
            let mut session = runtime::build_session(&graph, &gpu, Mode::Inference, false);
            runtime::share_matching(&mut train, &mut session, "world.");
            session
        };
        let observe = inference(world::build_observe_graph(&config, batch));
        let transition = inference(world::build_transition_graph(&config, batch));
        let decoder = inference(world::build_observation_prediction_graph(&config, batch));
        let heads = inference(world::build_head_graph(&config, batch));
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
            resize,
            preprocessor,
            rng,
            rollout_rng,
            prediction_length,
            pixel_loss,
            updates: 0,
            dirty: false,
        })
    }

    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
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
            let uniforms = (0..batch * size.stoch * size.classes)
                .map(|_| self.rng.random_range(1e-6..1.0 - 1e-6))
                .collect::<Vec<_>>();
            self.train.set_input(&format!("uniforms_{time}"), &uniforms);
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
            if self.pixel_loss.recurrent_energy() {
                if time + 1 < weights.len() {
                    for name in ["recurrent_first", "recurrent_second"] {
                        let uniforms = (0..batch * size.stoch * size.classes)
                            .map(|_| self.rollout_rng.random_range(1e-6..1.0 - 1e-6))
                            .collect::<Vec<_>>();
                        self.train
                            .set_input(&format!("{name}_uniforms_{time}"), &uniforms);
                    }
                }
            } else if !self.pixel_loss.latent_overshooting() || time + 1 < weights.len() {
                let uniforms = (0..batch * size.stoch * size.classes)
                    .map(|_| self.rollout_rng.random_range(1e-6..1.0 - 1e-6))
                    .collect::<Vec<_>>();
                let prefix = if self.pixel_loss.latent_overshooting() {
                    "latent_rollout"
                } else {
                    "pixel_rollout"
                };
                self.train
                    .set_input(&format!("{prefix}_uniforms_{time}"), &uniforms);
            }
            self.train
                .set_input(&format!("pixel_rollout_weight_{time}"), weight);
        }
        runtime::configure_d3_optimizer(
            &mut self.train,
            &self.config,
            self.updates + 1,
            self.config.learning_rate,
        );
        self.train.step();
        self.train.wait();
        let mut losses = vec![0.0; 9];
        for (index, loss) in losses.iter_mut().enumerate() {
            self.train
                .read_output_by_index(index, std::slice::from_mut(loss));
        }
        if self.prediction_length > 0 && !self.pixel_loss.recurrent_energy() {
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
        if self.pixel_loss.recurrent_energy() {
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

    /// No future image argument exists. All future states come from prior transitions.
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
            for session in [
                &mut self.observe,
                &mut self.transition,
                &mut self.decoder,
                &mut self.heads,
            ] {
                runtime::sync_matching(&self.train, session, "world.");
            }
            self.dirty = false;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let size = self.config.network();
        let mut deter = vec![0.0; batch * size.deter];
        let mut stoch = vec![0.0; batch * size.stoch * size.classes];
        let mut logits = vec![0.0; stoch.len()];
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
            self.observe.step();
            self.observe.wait();
            self.observe.read_output_by_index(0, &mut deter);
            self.observe.read_output_by_index(1, &mut logits);
            if include_features && time + 1 == length {
                self.observe.read_output_by_index(3, &mut encoded);
            }
            stoch = sample_latents(
                &logits,
                batch,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut rng,
            );
        }
        let decode = |session: &mut Session, deter: &[f32], stoch: &[f32]| {
            session.set_input("deter", deter);
            session.set_input("stoch", stoch);
            session.step();
            session.wait();
            let mut pixels = vec![0.0; batch * width];
            session.read_output_by_index(0, &mut pixels);
            pixels
        };
        let posterior = decode(&mut self.decoder, &deter, &stoch);
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
            stoch = sample_latents(
                &logits,
                batch,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut rng,
            );
            result
                .observations
                .extend(decode(&mut self.decoder, &deter, &stoch));
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
        let metadata = serde_json::json!({"format": "kindle-world-only-v2", "config": self.config, "updates": self.updates, "prediction_length": self.prediction_length, "pixel_loss": self.pixel_loss});
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
        let mut learner = Self::with_pixel_loss(config, prediction_length, pixel_loss)?;
        checkpoint::load_session(&mut learner.train, &path.join("world.safetensors"))?;
        learner.updates = metadata["updates"].as_u64().ok_or("missing update count")?;
        if u64::from(learner.train.adam_step_count()) != learner.updates {
            return Err(io::Error::other("world checkpoint optimizer age mismatch").into());
        }
        learner.dirty = true;
        Ok(learner)
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
