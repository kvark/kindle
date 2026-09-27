//! Synchronous vector collection for one learner. No independent model replicas.

use super::super::acting_gpu::ActingGpu;
use super::*;
use crate::vision::{levjepa::LeVJepaPerception, preprocess_gpu::GpuFrame};

struct LiveStream {
    policy_rng: StdRng,
    posterior_rng: StdRng,
    active: bool,
    needs_reset: bool,
    pending_action: Option<usize>,
}

pub(super) struct VectorCore {
    acting: ActingGpu,
    copies: DeviceCopies,
    pub(super) learner: DreamerCore,
    streams: Vec<LiveStream>,
    observe: Session,
    policy: Session,
}

fn check_capacity(config: &DreamerConfig, streams: usize) -> Result<(), &'static str> {
    if config.visitation_bonus {
        return Err(
            "GPU collection does not support host hash visitation; supply adapter intrinsic rewards instead",
        );
    }
    if streams == 0 {
        return Err("at least one environment is required");
    }
    let needed = (config.replay_context + config.batch_length - 1)
        .checked_mul(streams)
        .and_then(|context| context.checked_add(config.replay_warmup_sequences()));
    if needed.is_none_or(|needed| needed > config.replay_capacity) {
        return Err("replay capacity cannot warm up this many independent streams");
    }
    Ok(())
}

fn check_action_overrides(
    overrides: &[Option<usize>],
    streams: usize,
    action_count: usize,
) -> Result<(), &'static str> {
    if overrides.len() != streams {
        return Err("action overrides must have one entry per stream");
    }
    if overrides
        .iter()
        .flatten()
        .any(|&action| action >= action_count)
    {
        return Err("action override is outside the action vocabulary");
    }
    Ok(())
}

#[cfg(test)]
fn select_action(
    logits: &[f32],
    unimix: f32,
    mode: ActionMode,
    rng: &mut StdRng,
    override_action: Option<usize>,
) -> usize {
    let mut probabilities = vec![0.0; logits.len()];
    softmax_unimix(logits, unimix, &mut probabilities);
    let proposed = match mode {
        ActionMode::Sample => sample_probabilities(&probabilities, rng),
        ActionMode::Greedy => argmax(&probabilities),
    };
    override_action.unwrap_or(proposed)
}

impl VectorCore {
    fn new(mut learner: DreamerCore, streams: usize) -> Self {
        check_capacity(&learner.config, streams).unwrap();
        assert_eq!(learner.replay_len(), 0);
        learner.collection_streams = streams;
        learner.replay = SequenceReplay::with_streams(learner.config.replay_capacity, streams);
        learner
            .replay
            .enable_device(Arc::clone(&learner.gpu), &learner.config);
        assert!(
            learner.visitation.is_none(),
            "GPU collection does not support host hash visitation"
        );
        let config = &learner.config;
        let mut observe = build_session(
            &world::build_observe_graph(config, streams),
            &learner.gpu,
            Mode::Inference,
            false,
        );
        let mut policy = build_session(
            &behavior::build_actor_inference_graph(config, streams),
            &learner.gpu,
            Mode::Inference,
            false,
        );
        share_matching(&mut learner.world_train, &mut observe, "world.");
        share_matching(&mut learner.behavior_train, &mut policy, "behavior.actor.");
        let acting = ActingGpu::new(Arc::clone(&learner.gpu), config, streams);
        let copies = DeviceCopies::new(Arc::clone(&learner.gpu));
        let size = config.network();
        observe.set_input("previous_deter", &vec![0.0; streams * size.deter]);
        observe.set_input(
            "previous_stoch",
            &vec![0.0; streams * size.stoch * size.classes],
        );
        let streams = (0..streams)
            .map(|stream| {
                let seed = config.seed.wrapping_add(stream as u64);
                let rngs = if learner.environment_step == 0 && learner.learner_step == 0 {
                    DreamerRngs::new(seed)
                } else {
                    DreamerRngs::resumed(seed, learner.learner_step, learner.environment_step)
                };
                LiveStream {
                    policy_rng: rngs.policy,
                    posterior_rng: rngs.live_posterior,
                    active: false,
                    needs_reset: false,
                    pending_action: None,
                }
            })
            .collect();
        Self {
            acting,
            copies,
            learner,
            streams,
            observe,
            policy,
        }
    }

    fn check_arrivals(&self, arrivals: impl Iterator<Item = (usize, FrameFlags, Reward)>) {
        let mut seen = vec![false; self.streams.len()];
        for (id, flags, reward) in arrivals {
            assert!(id < seen.len() && !seen[id], "invalid or repeated stream");
            seen[id] = true;
            assert!(reward.extrinsic.is_finite() && reward.intrinsic.is_finite());
            assert!(!flags.is_terminal || flags.is_last);
            let stream = &self.streams[id];
            if flags.is_first {
                assert!(!flags.is_last && !flags.is_terminal);
                assert!(
                    stream.pending_action.is_none(),
                    "cannot reset a pending action"
                );
                assert!(
                    !stream.active || stream.needs_reset,
                    "reset requires an episode boundary"
                );
            } else {
                assert!(stream.pending_action.is_some(), "act must precede observe");
            }
        }
    }

    #[cfg(test)]
    fn ingest(&mut self, arrivals: Vec<(usize, Observation, FrameFlags, Reward)>) -> Vec<Reward> {
        self.check_arrivals(arrivals.iter().map(|a| (a.0, a.2, a.3)));
        if arrivals.is_empty() {
            return Vec::new();
        }
        self.acting.wait();
        self.learner.replay.wait_device();
        let mut observations =
            vec![0.0; self.streams.len() * self.learner.config.observation_dim()];
        for (id, observation, _, _) in &arrivals {
            observations[id * Observation::LEN..(id + 1) * Observation::LEN]
                .copy_from_slice(observation.as_slice());
        }
        self.observe.set_input("observation", &observations);
        self.ingest_prepared(&arrivals.iter().map(|a| (a.0, a.2, a.3)).collect::<Vec<_>>())
    }

    #[cfg(test)]
    fn feature(&mut self, stream: usize) -> Vec<f32> {
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            let device = self.learner.gpu_device();
            assert!(!device.is_software_emulated);
            assert_eq!(device.device_name, expected);
            let memory = self.learner.gpu_memory_budget();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
            eprintln!("device={} memory={memory:?}", device.device_name);
        }
        let width = self.learner.config.feature_dim();
        let mut source = self.policy.input_buffer("feature").unwrap();
        source.offset += (stream * width * 4) as u64;
        let mut values = vec![0.0; width];
        self.learner
            .readback
            .read_regions(&mut [(source, &mut values)]);
        values
    }

    fn ingest_encoded(
        &mut self,
        perception: &LeVJepaPerception,
        arrivals: &[(usize, FrameFlags, Reward)],
    ) -> Vec<Reward> {
        if arrivals.is_empty() {
            return Vec::new();
        }
        self.copies.copy(&[DeviceCopy {
            source: (perception.session(), ExternalSlot::Output(0)),
            target: (&self.observe, "observation"),
            target_offset_bytes: 0,
        }]);
        self.ingest_prepared(arrivals)
    }

    fn ingest_prepared(&mut self, arrivals: &[(usize, FrameFlags, Reward)]) -> Vec<Reward> {
        self.check_arrivals(arrivals.iter().copied());
        let config = &self.learner.config;
        let size = config.network();
        let rows = self.streams.len();
        let mut actions = vec![0.0; rows * config.action_count];
        let mut keep_deter = vec![0.0; rows * size.deter];
        let mut keep_stoch = vec![0.0; rows * size.stoch * size.classes];
        let mut keep_action = vec![0.0; rows * config.action_count];
        let mut active = vec![0; rows];
        let mut draws = vec![0.0; rows * size.stoch];
        for &(id, flags, _) in arrivals {
            active[id] = 1;
            for draw in &mut draws[id * size.stoch..(id + 1) * size.stoch] {
                *draw = self.streams[id].posterior_rng.random();
            }
            if !flags.is_first {
                actions[id * config.action_count + self.streams[id].pending_action.unwrap()] = 1.0;
                keep_deter[id * size.deter..(id + 1) * size.deter].fill(1.0);
                keep_stoch[id * size.stoch * size.classes..(id + 1) * size.stoch * size.classes]
                    .fill(1.0);
                keep_action[id * config.action_count..(id + 1) * config.action_count].fill(1.0);
            }
        }
        self.observe.wait();
        for (name, values) in [
            ("previous_action", &actions),
            ("keep_deter", &keep_deter),
            ("keep_stoch", &keep_stoch),
            ("keep_action", &keep_action),
        ] {
            self.observe.set_input(name, values);
        }
        self.observe.step();
        self.acting
            .posterior(&self.observe, &self.policy, &active, &draws);
        let frames: Vec<_> = arrivals
            .iter()
            .map(|&(id, flags, reward)| {
                let stream = &mut self.streams[id];
                let previous_action = stream.pending_action.take();
                stream.active = true;
                stream.needs_reset = flags.is_last;
                if !flags.is_first {
                    self.learner.environment_step += 1;
                    self.learner.train_scheduler.observe(
                        config.train_ratio / (config.batch_size * config.batch_length) as f32,
                    );
                }
                (id, previous_action, reward, flags)
            })
            .collect();
        self.learner
            .replay
            .push_device(&self.observe, &frames, config);
        arrivals.iter().map(|a| a.2).collect()
    }

    fn act(&mut self, mode: ActionMode) -> Vec<usize> {
        self.act_inner(mode, None, None)
    }

    fn act_with_overrides(
        &mut self,
        mode: ActionMode,
        overrides: &[Option<usize>],
    ) -> Result<Vec<usize>, &'static str> {
        check_action_overrides(
            overrides,
            self.streams.len(),
            self.learner.config.action_count,
        )?;
        Ok(self.act_inner(mode, Some(overrides), None))
    }

    pub(super) fn act_inner(
        &mut self,
        mode: ActionMode,
        overrides: Option<&[Option<usize>]>,
        mask: Option<&[bool]>,
    ) -> Vec<usize> {
        if let Some(mask) = mask {
            assert_eq!(
                mask.len(),
                self.streams.len() * self.learner.config.action_count
            );
            assert!(
                mask.chunks(self.learner.config.action_count)
                    .all(|row| row.iter().any(|&v| v))
            );
        }
        for stream in &self.streams {
            assert!(
                stream.active && !stream.needs_reset,
                "begin every episode before act"
            );
            assert!(
                stream.pending_action.is_none(),
                "observe every pending action first"
            );
        }
        self.policy.step();
        let draws: Vec<f32> = self
            .streams
            .iter_mut()
            .map(|stream| match mode {
                ActionMode::Sample => stream.policy_rng.random(),
                ActionMode::Greedy => 0.0,
            })
            .collect();
        let selected = self.acting.actions(&self.policy, &draws, mode, mask);
        let mut actions = vec![0.0; self.streams.len()];
        self.learner
            .readback
            .read_regions(&mut [(selected, &mut actions)]);
        actions
            .iter()
            .enumerate()
            .map(|(id, &proposed)| {
                assert!(
                    proposed.is_finite()
                        && proposed >= 0.0
                        && proposed < self.learner.config.action_count as f32
                        && proposed.fract() == 0.0,
                    "invalid GPU policy action: {proposed}"
                );
                let action = overrides.and_then(|a| a[id]).unwrap_or(proposed as usize);
                self.streams[id].pending_action = Some(action);
                action
            })
            .collect()
    }

    pub(super) fn learn(&mut self) -> Option<LearnReport> {
        let mut report = self.learner.learn()?;
        self.sync_live(&mut report);
        Some(report)
    }

    pub(super) fn read_diagnostics(&mut self) {
        assert_eq!(self.streams.len(), 1);
        let learner = &mut self.learner;
        learner.readback.read_regions(&mut [
            (
                self.policy.input_buffer("feature").unwrap(),
                &mut learner.feature,
            ),
            (
                self.observe.input_buffer("previous_deter").unwrap(),
                &mut learner.deter,
            ),
            (
                self.observe.input_buffer("previous_stoch").unwrap(),
                &mut learner.stoch,
            ),
            (
                self.observe.input_buffer("observation").unwrap(),
                &mut learner.observation,
            ),
            (
                self.observe.output_buffer(3).unwrap(),
                &mut learner.encoded_observation,
            ),
        ]);
        learner.active = self.streams[0].active;
    }

    fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        let mut reports = self.learner.learn_scheduled(maximum_updates);
        if let Some(last) = reports.last_mut() {
            self.sync_live(last);
        }
        reports
    }

    fn sync_live(&mut self, last: &mut LearnReport) {
        let started = Instant::now();
        sync_matching(&self.learner.world_train, &mut self.observe, "world.");
        let world = started.elapsed().as_secs_f64();
        let started = Instant::now();
        sync_matching(
            &self.learner.behavior_train,
            &mut self.policy,
            "behavior.actor.",
        );
        let behavior = started.elapsed().as_secs_f64();
        last.timing.world_sync_seconds += world;
        last.timing.behavior_sync_seconds += behavior;
        last.timing.total_seconds += world + behavior;
    }
}

/// Multiple independent environments using one frozen LeVJEPA and Dreamer learner.
/// Dense perception, posterior and policy inference are batched on the GPU.
pub struct VectorDreamerAgent {
    perception: LeVJepaPerception,
    pub(super) core: VectorCore,
}

impl VectorDreamerAgent {
    pub fn new(
        config: DreamerConfig,
        streams: usize,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_perception(
            config,
            streams,
            PerceptionKind::LeVJepaTiny,
            encoder_checkpoint,
        )
    }

    pub fn with_perception(
        config: DreamerConfig,
        streams: usize,
        kind: PerceptionKind,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        config.check()?;
        check_capacity(&config, streams)?;
        let architecture = kind
            .levjepa_architecture()
            .ok_or("vector perception requires LeVJEPA")?;
        let identity = kind.identity(crate::vision::checkpoint_sha256(
            encoder_checkpoint.as_ref(),
        )?);
        let gpu = Arc::new(crate::init_gpu_context()?);
        let perception = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            encoder_checkpoint,
            streams,
            Some(Arc::clone(&gpu)),
            None,
        )?;
        let mut learner = DreamerCore::with_gpu(config, gpu);
        learner.perception_identity = Some(identity);
        Ok(Self {
            perception,
            core: VectorCore::new(learner, streams),
        })
    }

    pub fn restore(
        checkpoint: impl AsRef<Path>,
        streams: usize,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if streams == 0 {
            return Err("at least one environment is required".into());
        }
        let metadata = read_checkpoint_metadata(checkpoint.as_ref())?;
        check_capacity(&metadata.config, streams)?;
        let identity = metadata
            .perception
            .as_ref()
            .ok_or("missing perception identity")?;
        let architecture = identity
            .kind
            .levjepa_architecture()
            .ok_or("vector perception requires LeVJEPA")?;
        identity.verify_file(encoder_checkpoint.as_ref())?;
        let gpu = Arc::new(crate::init_gpu_context()?);
        let perception = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            encoder_checkpoint,
            streams,
            Some(Arc::clone(&gpu)),
            None,
        )?;
        let learner = DreamerCore::restore_with_gpu(checkpoint.as_ref(), gpu, metadata)?;
        Ok(Self {
            perception,
            core: VectorCore::new(learner, streams),
        })
    }

    pub fn config(&self) -> &DreamerConfig {
        self.core.learner.config()
    }
    pub fn provenance(&self) -> ModelProvenance {
        self.core.learner.provenance()
    }
    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        self.core.learner.gpu_device()
    }
    pub fn gpu_memory_budget(&self) -> crate::GpuMemoryBudget {
        self.core.learner.gpu_memory_budget()
    }
    pub fn trainable_parameter_counts(&self) -> (usize, usize) {
        self.core.learner.trainable_parameter_counts()
    }
    pub fn learner_step(&self) -> u64 {
        self.core.learner.learner_step()
    }
    pub fn environment_step(&self) -> u64 {
        self.core.learner.environment_step()
    }
    pub fn replay_len(&self) -> usize {
        self.core.learner.replay_len()
    }
    pub fn cpu_worker_threads(&self) -> usize {
        self.core.learner.cpu_worker_threads()
    }
    pub fn stream_count(&self) -> usize {
        self.core.streams.len()
    }
    pub fn training_debt(&self) -> f32 {
        self.core.learner.train_scheduler.credit
    }

    pub fn begin_episodes(&mut self, frames: &[(usize, RgbFrame)]) {
        self.begin_refs(
            &frames
                .iter()
                .map(|(id, frame)| (*id, frame))
                .collect::<Vec<_>>(),
        );
    }

    pub(super) fn begin_refs(&mut self, frames: &[(usize, &RgbFrame)]) {
        let flags = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        self.core
            .check_arrivals(frames.iter().map(|(id, _)| (*id, flags, Reward::default())));
        let arrivals: Vec<_> = frames
            .iter()
            .map(|(id, frame)| (*id, *frame, true))
            .collect();
        self.perception.submit_frames_rgb8(&arrivals);
        self.core.ingest_encoded(
            &self.perception,
            &frames
                .iter()
                .map(|(id, _)| (*id, flags, Reward::default()))
                .collect::<Vec<_>>(),
        );
    }

    pub fn act(&mut self, mode: ActionMode) -> Vec<usize> {
        self.core.act(mode)
    }

    /// Choose actions, replacing selected streams with explicitly executed controls.
    /// Policy RNG draws are retained, and RSSM/replay consume the overridden action.
    pub fn act_with_overrides(
        &mut self,
        mode: ActionMode,
        overrides: &[Option<usize>],
    ) -> Result<Vec<usize>, &'static str> {
        self.core.act_with_overrides(mode, overrides)
    }

    pub fn observe(&mut self, transitions: &[(usize, Transition)]) -> Vec<Reward> {
        self.observe_refs(
            &transitions
                .iter()
                .map(|(id, t)| (*id, t))
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn observe_refs(&mut self, transitions: &[(usize, &Transition)]) -> Vec<Reward> {
        self.core
            .check_arrivals(transitions.iter().map(|(id, t)| (*id, t.flags(), t.reward)));
        let arrivals: Vec<_> = transitions
            .iter()
            .map(|(id, t)| (*id, &t.frame, false))
            .collect();
        self.perception.submit_frames_rgb8(&arrivals);
        self.core.ingest_encoded(
            &self.perception,
            &transitions
                .iter()
                .map(|(id, t)| (*id, t.flags(), t.reward))
                .collect::<Vec<_>>(),
        )
    }

    /// Context used by the complete perception/belief/policy pipeline.
    pub fn gpu_context(&self) -> Arc<blade_graphics::Context> {
        Arc::clone(&self.core.learner.gpu)
    }

    /// Consume already synchronized resident frames; reset and transition
    /// metadata have the same contract as begin_episodes/observe.
    /// This waits for the GPU to finish reading capture buffers before returning.
    pub fn observe_gpu(
        &mut self,
        arrivals: &[(usize, GpuFrame<'_>, FrameFlags, Reward)],
    ) -> Vec<Reward> {
        let metadata: Vec<_> = arrivals.iter().map(|a| (a.0, a.2, a.3)).collect();
        self.core.check_arrivals(metadata.iter().copied());
        self.perception.submit_frames_gpu8(
            &arrivals
                .iter()
                .map(|a| (a.0, a.1, a.2.is_first))
                .collect::<Vec<_>>(),
        );
        let rewards = self.core.ingest_encoded(&self.perception, &metadata);
        // GpuFrame's borrowed capture ownership ends on return. No data readback.
        self.core.acting.wait();
        rewards
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        self.core.learn_scheduled(maximum_updates)
    }
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.core.learner.save_checkpoint(path)
    }
}

impl Drop for VectorDreamerAgent {
    fn drop(&mut self) {
        self.core.acting.wait();
        self.core.copies.wait();
        self.core.learner.replay.wait_device();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_overrides_validate_every_stream_before_mutation() {
        assert!(check_action_overrides(&[None, Some(2), Some(0)], 3, 3).is_ok());
        assert!(check_action_overrides(&[None], 3, 3).is_err());
        assert!(check_action_overrides(&[None, Some(3), None], 3, 3).is_err());
        assert!(check_action_overrides(&[Some(usize::MAX)], 1, 3).is_err());
    }

    #[test]
    fn action_overrides_preserve_policy_rng_draws_and_default_selection() {
        let logits = [0.4, -0.9, 2.0];
        let mut reference = StdRng::seed_from_u64(73);
        let mut default = reference.clone();
        let mut overridden = reference.clone();
        for time in 0..128 {
            let mut probabilities = vec![0.0; logits.len()];
            softmax_unimix(&logits, 0.01, &mut probabilities);
            let expected = sample_probabilities(&probabilities, &mut reference);
            assert_eq!(
                select_action(&logits, 0.01, ActionMode::Sample, &mut default, None),
                expected
            );
            assert_eq!(
                select_action(
                    &logits,
                    0.01,
                    ActionMode::Sample,
                    &mut overridden,
                    Some(time % 3)
                ),
                time % 3
            );
        }
        let next = reference.random::<u64>();
        assert_eq!(next, default.random::<u64>());
        assert_eq!(next, overridden.random::<u64>());
    }

    #[test]
    fn greedy_overrides_do_not_consume_rng() {
        let mut rng = StdRng::seed_from_u64(73);
        let mut reference = rng.clone();
        assert_eq!(
            select_action(&[0.0, 1.0], 0.01, ActionMode::Greedy, &mut rng, None),
            1
        );
        assert_eq!(
            select_action(&[0.0, 1.0], 0.01, ActionMode::Greedy, &mut rng, Some(0)),
            0
        );
        assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    }

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 4;
        config.world_backprop_length = 4;
        config.world_microbatch_size = Some(2);
        config.train_ratio = 8.0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        config
    }

    fn observation(stream: usize, time: usize) -> Observation {
        Observation::from_vec(
            (0..Observation::LEN)
                .map(|i| ((i + stream * 17 + time * 31) % 101) as f32 / 101.0)
                .collect(),
        )
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    #[ignore = "requires GPU; checks executed-action belief/replay causality and independent RNGs"]
    fn vector_overrides_match_serial_beliefs_and_replay_actions() {
        let config = config();
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let learner = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut vector = VectorCore::new(learner, 3);
        let mut serial: Vec<_> = (0..3)
            .map(|id| {
                let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
                core.rngs = DreamerRngs::new(config.seed + id);
                core
            })
            .collect();
        let first = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        vector.ingest(
            (0..3)
                .map(|id| (id, observation(id, 0), first, Reward::default()))
                .collect(),
        );
        for (id, core) in serial.iter_mut().enumerate() {
            core.begin_episode(observation(id, 0));
        }
        assert!(
            vector
                .act_with_overrides(ActionMode::Sample, &[None])
                .is_err()
        );
        assert!(
            vector
                .act_with_overrides(ActionMode::Sample, &[None, Some(3), None])
                .is_err()
        );
        assert!(
            vector
                .streams
                .iter()
                .all(|stream| stream.pending_action.is_none())
        );
        for time in 1..25 {
            let overrides = [None, Some(2), (time % 2 == 0).then_some(0)];
            let actions = vector
                .act_with_overrides(ActionMode::Sample, &overrides)
                .unwrap();
            let mut arrivals = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                let mask = overrides[id]
                    .map(|forced| (0..3).map(|action| action == forced).collect::<Vec<_>>());
                assert_eq!(actions[id], core.act(ActionMode::Sample, mask.as_deref()));
                assert_eq!(vector.streams[id].pending_action, Some(actions[id]));
                let flags = FrameFlags {
                    is_last: time % (3 + id) == 0,
                    is_terminal: id != 1 && time % (3 + id) == 0,
                    ..Default::default()
                };
                let reward = Reward {
                    extrinsic: actions[id] as f32,
                    intrinsic: 0.0,
                };
                core.observe(observation(id, time), reward, flags);
                arrivals.push((id, observation(id, time), flags, reward));
            }
            vector.ingest(arrivals);
            let mut resets = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                assert_close(&core.feature, &vector.feature(id));
                assert_eq!(
                    core.rngs.policy.clone().random::<u64>(),
                    vector.streams[id].policy_rng.clone().random::<u64>()
                );
                if core.needs_reset {
                    core.begin_episode(observation(id, 100 + time));
                    resets.push((id, observation(id, 100 + time), first, Reward::default()));
                }
            }
            vector.ingest(resets);
        }
        let mut rng = StdRng::seed_from_u64(97);
        for _ in 0..20 {
            let batch = vector.learner.replay.sample(&config, &mut rng).unwrap();
            for time in 0..batch.previous_actions.len() {
                for row in 0..config.batch_size {
                    let actual = &batch.previous_actions[time][row * 3..(row + 1) * 3];
                    let mut expected = [0.0; 3];
                    if !batch.flags[time][row].is_first {
                        expected[batch.rewards[time][row] as usize] = 1.0;
                    }
                    assert_eq!(actual, expected);
                }
            }
        }
        assert_eq!(vector.learner.environment_step, 72);
        assert_eq!(vector.learner.learner_step, 0);
    }

    #[test]
    #[ignore = "requires GPU; compares batched live states to independent serial streams"]
    fn vector_live_beliefs_and_policy_match_serial() {
        let mut config = config();
        config.replay_capacity = 32;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let learner = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut vector = VectorCore::new(learner, 3);
        let mut serial: Vec<_> = (0..3)
            .map(|id| {
                let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
                // Same model initialization, independent live RNGs as in the vector.
                core.rngs = DreamerRngs::new(config.seed + id);
                core
            })
            .collect();
        let first = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        vector.ingest(
            (0..3)
                .map(|id| (id, observation(id, 0), first, Reward::default()))
                .collect(),
        );
        for (id, core) in serial.iter_mut().enumerate() {
            core.begin_episode(observation(id, 0));
        }
        for time in 1..20 {
            let mask: Vec<_> = (0..9)
                .map(|i| time % 3 == 0 || i % 3 != (time + i / 3) % 3)
                .collect();
            let mode = if time % 4 == 0 {
                ActionMode::Greedy
            } else {
                ActionMode::Sample
            };
            let actions = vector.act_inner(mode, None, Some(&mask));
            let mut arrivals = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                assert_eq!(
                    actions[id],
                    core.act(mode, Some(&mask[id * 3..(id + 1) * 3]))
                );
                let last = time % (3 + id) == 0;
                let flags = FrameFlags {
                    is_last: last,
                    is_terminal: last && id != 1,
                    ..Default::default()
                };
                let reward = Reward {
                    extrinsic: actions[id] as f32,
                    intrinsic: 0.0,
                };
                core.observe(observation(id, time), reward, flags);
                arrivals.push((id, observation(id, time), flags, reward));
            }
            vector.ingest(arrivals);
            let mut resets = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                assert_close(core.latent_feature(), &vector.feature(id));
                if core.needs_reset {
                    core.begin_episode(observation(id, 100 + time));
                    resets.push((id, observation(id, 100 + time), first, Reward::default()));
                }
            }
            vector.ingest(resets);
            for (id, core) in serial.iter().enumerate() {
                assert_close(core.latent_feature(), &vector.feature(id));
            }
            assert_eq!(vector.learner.environment_step, 3 * time as u64);
        }
    }

    #[test]
    #[ignore = "requires GPU; includes scheduled updates and checkpoint restore"]
    fn vector_one_matches_serial_learning_and_checkpoint() {
        let mut config = config();
        config.replay_capacity = 20;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut serial = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let learner = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut vector = VectorCore::new(learner, 1);
        let first = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        serial.begin_episode(observation(0, 0));
        vector.ingest(vec![(0, observation(0, 0), first, Reward::default())]);
        for time in 1..24 {
            assert_eq!(
                vector.act(ActionMode::Sample)[0],
                serial.act(ActionMode::Sample, None)
            );
            let flags = FrameFlags {
                is_last: time % 5 == 0,
                is_terminal: time % 10 == 0,
                ..Default::default()
            };
            let reward = Reward {
                extrinsic: (time % 3) as f32 - 1.0,
                intrinsic: 0.0,
            };
            serial.observe(observation(0, time), reward, flags);
            vector.ingest(vec![(0, observation(0, time), flags, reward)]);
            let a = serial.learn_scheduled(usize::MAX);
            let b = vector.learn_scheduled(usize::MAX);
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(&b) {
                assert_eq!(a.world.total_loss, b.world.total_loss);
                assert_eq!(a.behavior.total_loss, b.behavior.total_loss);
            }
            assert_close(&serial.feature, &vector.feature(0));
            if flags.is_last {
                serial.begin_episode(observation(0, time + 100));
                vector.ingest(vec![(
                    0,
                    observation(0, time + 100),
                    first,
                    Reward::default(),
                )]);
            }
        }
        assert!(vector.learner.learner_step > 0);
        assert_eq!(serial.learner_step, vector.learner.learner_step);
        assert_eq!(
            serial.train_scheduler.credit,
            vector.learner.train_scheduler.credit
        );
        let names = serial.world_train.param_names();
        assert_eq!(
            serial.world_train.read_params(&names),
            vector.learner.world_train.read_params(&names)
        );
        let checkpoint =
            std::env::temp_dir().join(format!("kindle-vector-test-{}", std::process::id()));
        assert!(!checkpoint.exists());
        vector.learner.save_checkpoint(&checkpoint).unwrap();
        let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
        assert_eq!(metadata.collection_streams, 1);
        let learner = DreamerCore::restore_with_gpu(&checkpoint, gpu, metadata).unwrap();
        let mut restored = VectorCore::new(learner, 2);
        assert_eq!(restored.learner.replay_len(), 0);
        assert_eq!(restored.learner.environment_step, 23);
        restored.ingest(
            (0..2)
                .map(|id| (id, observation(id, 0), first, Reward::default()))
                .collect(),
        );
        assert_eq!(restored.act(ActionMode::Sample).len(), 2);
        fs::remove_dir_all(checkpoint).unwrap();
    }

    #[test]
    fn vector_scheduler_counts_actions_not_ticks_or_resets() {
        let mut scheduler = D3TrainScheduler::default();
        for _ in 0..400 {
            scheduler.observe(0.25);
        }
        assert!(!scheduler.update_due(false));
        assert!(scheduler.update_due(true));
        scheduler.consume_update();
        for _ in 0..3 {
            for _ in 0..8 {
                scheduler.observe(0.25);
            }
            for _ in 0..2 {
                assert!(scheduler.update_due(true));
                scheduler.consume_update();
            }
            assert!(!scheduler.update_due(true));
        }
        assert_eq!(scheduler.credit, 0.0);
    }

    #[test]
    fn vector_capacity_includes_each_stream_context() {
        let mut config = config();
        config.replay_capacity = config.replay_warmup_frames();
        assert!(check_capacity(&config, 1).is_ok());
        assert!(check_capacity(&config, 2).is_err());
        assert!(check_capacity(&config, 0).is_err());
        assert!(check_capacity(&config, usize::MAX).is_err());
        config.replay_capacity += config.replay_context + config.batch_length - 1;
        assert!(check_capacity(&config, 2).is_ok());
    }
}
