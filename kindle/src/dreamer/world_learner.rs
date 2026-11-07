//! Fixed-data world learning without actor, critic, imitation or exploration.

use std::{io, path::Path, sync::Arc};

use meganeura::{Graph, Mode, Session};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{
    DreamerConfig, agent::sample_latents, checkpoint, distributions::TwoHotBins, runtime, world,
};
use crate::{env::RgbFrame, vision::preprocess_gpu::GpuPreprocessor};

pub struct WorldPrediction {
    /// Centered CHW RGB, batch-major.
    pub posterior: Vec<f32>,
    /// Centered CHW RGB, future-time then batch order.
    pub observations: Vec<f32>,
    pub rewards: Vec<Vec<f32>>,
    /// Includes the configured continuation discount, as in the online learner.
    pub continuations: Vec<Vec<f32>>,
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
    updates: u64,
    dirty: bool,
}

fn validate_config(config: &DreamerConfig) -> Result<(), &'static str> {
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

impl WorldModelLearner {
    pub fn new(config: DreamerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        validate_config(&config)?;
        config.validate();
        let gpu = Arc::new(crate::init_gpu_context()?);
        let graph = world::build_world_only_training_graph(&config);
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
        for (i, &terminal) in terminals.iter().enumerate() {
            if (actions[i] == -1 && (terminal || rewards[i] != 0.0))
                || (terminal && i + batch < actions.len() && actions[i + batch] != -1)
            {
                return Err("terminal/reset alignment mismatch");
            }
        }
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
            let mut target = vec![0.0; batch * bins.len()];
            for (row, &reward) in rewards[range.clone()].iter().enumerate() {
                bins.encode(
                    reward,
                    &mut target[row * bins.len()..(row + 1) * bins.len()],
                );
            }
            self.train
                .set_input(&format!("reward_target_{time}"), &target);
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
            continuations: Vec::new(),
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
            self.heads.set_input("deter", &deter);
            self.heads.set_input("stoch", &stoch);
            self.heads.step();
            self.heads.wait();
            let mut reward_logits = vec![0.0; batch * bins.len()];
            let mut continuation = vec![0.0; batch];
            self.heads.read_output_by_index(0, &mut reward_logits);
            self.heads.read_output_by_index(1, &mut continuation);
            let mut scratch = vec![0.0; bins.len()];
            result.rewards.push(
                reward_logits
                    .chunks_exact(bins.len())
                    .map(|row| bins.decode_logits_with_scratch(row, &mut scratch))
                    .collect(),
            );
            result.continuations.push(continuation);
        }
        if result
            .posterior
            .iter()
            .chain(&result.observations)
            .chain(result.rewards.iter().flatten())
            .chain(result.continuations.iter().flatten())
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
        let metadata = serde_json::json!({"format": "kindle-world-only-v1", "config": self.config, "updates": self.updates});
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    pub fn restore(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json"))?)?;
        if metadata["format"] != "kindle-world-only-v1" {
            return Err(io::Error::other("not a world-only checkpoint").into());
        }
        let config = serde_json::from_value(metadata["config"].clone())?;
        let mut learner = Self::new(config)?;
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
