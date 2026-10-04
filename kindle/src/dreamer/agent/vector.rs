//! Synchronous vector collection for one learner. No independent model replicas.

use super::super::acting_gpu::ActingGpu;
use super::*;
use crate::dreamer::ObservationKind;
use crate::vision::{
    levjepa::LeVJepaPerception,
    preprocess_gpu::{GpuFrame, GpuPreprocessor},
};

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
    pub(super) learner: Box<DreamerCore>,
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
        .and_then(|context| {
            let stride = if config.video_encoder.is_some() {
                config.batch_length
            } else {
                1
            };
            config
                .replay_warmup_sequences()
                .checked_mul(stride)
                .and_then(|items| context.checked_add(items))
        });
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
            learner: Box::new(learner),
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

    fn ingest(&mut self, arrivals: Vec<(usize, Observation, FrameFlags, Reward)>) -> Vec<Reward> {
        self.check_arrivals(arrivals.iter().map(|a| (a.0, a.2, a.3)));
        if arrivals.is_empty() {
            return Vec::new();
        }
        self.acting.wait();
        self.learner.replay.wait_device();
        let mut observations =
            vec![0.0; self.streams.len() * self.learner.config.observation_dim()];
        let width = self.learner.config.observation_dim();
        for (id, observation, _, _) in &arrivals {
            observations[id * width..(id + 1) * width].copy_from_slice(observation.as_slice());
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
        if self.learner.config.video_encoder.is_some() {
            self.learner.replay.store_pixels(
                perception.session(),
                &arrivals.iter().map(|a| a.0).collect::<Vec<_>>(),
            );
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
        let learner = &mut *self.learner;
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

/// Small-image or symbolic observations with a jointly learned encoder.
///
/// The adapter supplies losslessly packed `[7 * 7, 64]` observations. It does
/// not supply pretrained features, privileged reward information or an RSSM
/// state. Batched belief/policy, replay and learning use the same GPU path as
/// the pixel agent; only the frozen LeVJEPA frontend is bypassed.
pub struct FeatureVectorAgent {
    core: VectorCore,
}

impl FeatureVectorAgent {
    pub fn new(config: DreamerConfig, streams: usize) -> Result<Self, Box<dyn std::error::Error>> {
        config.check()?;
        check_capacity(&config, streams)?;
        if config.observation_kind != ObservationKind::Features || config.video_encoder.is_some() {
            return Err(
                "FeatureVectorAgent requires supplied features, not pixel/causal replay".into(),
            );
        }
        Ok(Self {
            core: VectorCore::new(DreamerCore::new(config)?, streams),
        })
    }

    pub fn learner(&self) -> &DreamerCore {
        &self.core.learner
    }

    pub fn stream_count(&self) -> usize {
        self.core.streams.len()
    }

    pub fn training_debt(&self) -> f32 {
        self.core.learner.train_scheduler.credit
    }

    pub fn begin_episodes(&mut self, observations: Vec<(usize, Observation)>) {
        self.core.ingest(
            observations
                .into_iter()
                .map(|(id, observation)| {
                    (
                        id,
                        observation,
                        FrameFlags {
                            is_first: true,
                            ..Default::default()
                        },
                        Reward::default(),
                    )
                })
                .collect(),
        );
    }

    pub fn act(&mut self, mode: ActionMode) -> Vec<usize> {
        self.core.act(mode)
    }

    pub fn observe(
        &mut self,
        arrivals: Vec<(usize, Observation, FrameFlags, Reward)>,
    ) -> Vec<Reward> {
        self.core.ingest(arrivals)
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        self.core.learn_scheduled(maximum_updates)
    }

    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.core.learner.save_checkpoint(path)
    }
}

impl Drop for FeatureVectorAgent {
    fn drop(&mut self) {
        self.core.acting.wait();
        self.core.copies.wait();
        self.core.learner.replay.wait_device();
    }
}

enum VectorPerception {
    Causal(Box<LeVJepaPerception>),
    Learned(Box<GpuPreprocessor>),
}

impl VectorPerception {
    fn ingest(
        &mut self,
        core: &mut VectorCore,
        frames: &[(usize, &RgbFrame)],
        metadata: &[(usize, FrameFlags, Reward)],
    ) -> Vec<Reward> {
        match self {
            Self::Causal(perception) => {
                let frames = frames
                    .iter()
                    .zip(metadata)
                    .map(|(&(id, frame), &(_, flags, _))| (id, frame, flags.is_first))
                    .collect::<Vec<_>>();
                perception.submit_frames_rgb8(&frames);
                core.ingest_encoded(perception, metadata)
            }
            Self::Learned(pixels) => {
                core.acting.wait();
                core.learner.replay.wait_device();
                let frames = frames
                    .iter()
                    .map(|&(id, frame)| (id, frame.pixels(), frame.width(), frame.height()))
                    .collect::<Vec<_>>();
                pixels.cpu_frames(&mut core.observe, &frames);
                core.ingest_prepared(metadata)
            }
        }
    }
}

/// Independent environments sharing one batched GPU perception/policy and learner.
/// Frozen LeVJEPA retains features; the jointly learned CNN retains RGB pixels.
pub struct VectorDreamerAgent {
    perception: VectorPerception,
    history: Option<PixelHistory>,
    pub(super) core: VectorCore,
}

/// Only current-chunk pixels, on GPU. Updating Tiny invalidates all its KVs;
/// replaying these prefixes rebuilds them without resetting any RSSM stream.
struct PixelHistory {
    gpu: Arc<blade_graphics::Context>,
    buffer: blade_graphics::Buffer,
    copies: DeviceCopies,
    streams: usize,
}

impl PixelHistory {
    fn new(gpu: Arc<blade_graphics::Context>, streams: usize) -> Self {
        use crate::vision::levjepa::{FRAMES, joint::PIXELS};
        let buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "kindle_live_pixel_history",
            size: (streams * FRAMES * PIXELS * 4) as u64,
            memory: blade_graphics::Memory::Device,
        });
        Self {
            copies: DeviceCopies::new(Arc::clone(&gpu)),
            gpu,
            buffer,
            streams,
        }
    }

    fn store(&mut self, perception: &LeVJepaPerception, streams: &[usize]) {
        use crate::vision::levjepa::{FRAMES, joint::PIXELS};
        let source = perception.session().input_buffer("patches").unwrap();
        let copies = streams
            .iter()
            .map(|&stream| {
                let phase = (perception.chunk_positions()[stream] + FRAMES - 1) % FRAMES;
                let mut source = source;
                source.offset += (stream * PIXELS * 4) as u64;
                (
                    source,
                    self.buffer
                        .at(((stream * FRAMES + phase) * PIXELS * 4) as u64),
                    PIXELS * 4,
                )
            })
            .collect::<Vec<_>>();
        self.copies.copy_regions(&copies);
    }

    fn refresh(&mut self, perception: &mut LeVJepaPerception) {
        use crate::vision::levjepa::{FRAMES, PATCHES, joint::PIXELS};
        let positions = perception.chunk_positions().to_vec();
        let session = perception.session_mut();
        session.wait();
        self.copies.wait();
        for phase in 0..positions.iter().copied().max().unwrap_or(0) {
            let destination = session.input_buffer("patches").unwrap();
            let copies = (0..self.streams)
                .filter(|&stream| phase < positions[stream])
                .map(|stream| {
                    let mut destination = destination;
                    destination.offset += (stream * PIXELS * 4) as u64;
                    (
                        self.buffer
                            .at(((stream * FRAMES + phase) * PIXELS * 4) as u64),
                        destination,
                        PIXELS * 4,
                    )
                })
                .collect::<Vec<_>>();
            self.copies.copy_regions(&copies);
            for (stream, &position) in positions.iter().enumerate() {
                // An inactive stream's unused next slot may be overwritten;
                // its valid prefix and arrival counter are untouched.
                let frame = phase.min(position);
                session.set_input_u32(&format!("frame.{stream}"), &[frame as u32]);
                session.set_input_u32(
                    &format!("last_token.{stream}"),
                    &[((frame + 1) * PATCHES - 1) as u32],
                );
            }
            session.step();
            session.wait();
        }
    }
}

impl Drop for PixelHistory {
    fn drop(&mut self) {
        self.copies.wait();
        self.gpu.destroy_buffer(self.buffer);
    }
}

impl VectorDreamerAgent {
    pub fn learned_rgb(
        config: DreamerConfig,
        streams: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        config.check()?;
        check_capacity(&config, streams)?;
        if config.observation_kind != ObservationKind::Rgb64 {
            return Err("learned RGB requires observation_kind=rgb64".into());
        }
        let learner = DreamerCore::new(config)?;
        let pixels = GpuPreprocessor::rgb64(Arc::clone(&learner.gpu), streams);
        Ok(Self {
            perception: VectorPerception::Learned(Box::new(pixels)),
            history: None,
            core: VectorCore::new(learner, streams),
        })
    }

    pub fn restore_rgb(
        checkpoint: impl AsRef<Path>,
        streams: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = read_checkpoint_metadata(checkpoint.as_ref())?;
        check_capacity(&metadata.config, streams)?;
        if metadata.config.observation_kind != ObservationKind::Rgb64
            || metadata.perception.is_some()
        {
            return Err("checkpoint is not a jointly learned RGB model".into());
        }
        let gpu = Arc::new(crate::init_gpu_context()?);
        let pixels = GpuPreprocessor::rgb64(Arc::clone(&gpu), streams);
        let learner = DreamerCore::restore_with_gpu(checkpoint.as_ref(), gpu, metadata)?;
        Ok(Self {
            perception: VectorPerception::Learned(Box::new(pixels)),
            history: None,
            core: VectorCore::new(learner, streams),
        })
    }

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
        if config.observation_kind != ObservationKind::Features {
            return Err("frozen LeVJEPA requires feature observations".into());
        }
        let architecture = kind
            .levjepa_architecture()
            .ok_or("vector perception requires LeVJEPA")?;
        if config.video_encoder.is_some()
            && architecture != crate::vision::levjepa::Architecture::Tiny
        {
            return Err("joint/re-encoded replay requires the complete Tiny encoder".into());
        }
        let identity = kind.identity(crate::vision::checkpoint_sha256(
            encoder_checkpoint.as_ref(),
        )?);
        let gpu = Arc::new(crate::init_gpu_context()?);
        let mut perception = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            encoder_checkpoint.as_ref(),
            streams,
            Some(Arc::clone(&gpu)),
            None,
        )?;
        let mut learner = DreamerCore::with_gpu(config, gpu);
        learner.perception_identity = Some(identity);
        if learner.config.video_encoder.is_some() {
            let model = meganeura::data::safetensors::SafeTensorsModel::load(
                encoder_checkpoint.as_ref().to_path_buf(),
            )?;
            crate::vision::levjepa::load_weights(
                &mut learner.world_train,
                &model,
                0,
                architecture,
            )?;
            learner.sync_world_inference();
            share_matching(
                &mut learner.world_train,
                perception.session_mut(),
                "encoder.",
            );
        }
        let history = (learner.config.video_encoder == Some(crate::dreamer::VideoEncoder::Joint))
            .then(|| PixelHistory::new(Arc::clone(&learner.gpu), streams));
        Ok(Self {
            perception: VectorPerception::Causal(Box::new(perception)),
            history,
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
        let mut perception = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            encoder_checkpoint,
            streams,
            Some(Arc::clone(&gpu)),
            None,
        )?;
        let mut learner = DreamerCore::restore_with_gpu(checkpoint.as_ref(), gpu, metadata)?;
        if learner.config.video_encoder.is_some() {
            share_matching(
                &mut learner.world_train,
                perception.session_mut(),
                "encoder.",
            );
        }
        let history = (learner.config.video_encoder == Some(crate::dreamer::VideoEncoder::Joint))
            .then(|| PixelHistory::new(Arc::clone(&learner.gpu), streams));
        Ok(Self {
            perception: VectorPerception::Causal(Box::new(perception)),
            history,
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
        self.perception.ingest(
            &mut self.core,
            frames,
            &frames
                .iter()
                .map(|(id, _)| (*id, flags, Reward::default()))
                .collect::<Vec<_>>(),
        );
        self.store_history(&frames.iter().map(|a| a.0).collect::<Vec<_>>());
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
        let rewards = self.perception.ingest(
            &mut self.core,
            &transitions
                .iter()
                .map(|(id, t)| (*id, &t.frame))
                .collect::<Vec<_>>(),
            &transitions
                .iter()
                .map(|(id, t)| (*id, t.flags(), t.reward))
                .collect::<Vec<_>>(),
        );
        self.store_history(&transitions.iter().map(|a| a.0).collect::<Vec<_>>());
        rewards
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
        let frames = arrivals.iter().map(|a| (a.0, a.1)).collect::<Vec<_>>();
        let rewards = match &mut self.perception {
            VectorPerception::Causal(perception) => {
                perception.submit_frames_gpu8(
                    &arrivals
                        .iter()
                        .map(|a| (a.0, a.1, a.2.is_first))
                        .collect::<Vec<_>>(),
                );
                self.core.ingest_encoded(perception, &metadata)
            }
            VectorPerception::Learned(pixels) => {
                self.core.acting.wait();
                self.core.learner.replay.wait_device();
                pixels.gpu_frames(&mut self.core.observe, &frames);
                self.core.ingest_prepared(&metadata)
            }
        };
        // GpuFrame's borrowed capture ownership ends on return. No data readback.
        self.core.acting.wait();
        self.store_history(&arrivals.iter().map(|a| a.0).collect::<Vec<_>>());
        rewards
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        let mut reports = self.core.learn_scheduled(maximum_updates);
        if let Some(last) = reports.last_mut() {
            self.refresh_encoder(last);
        }
        reports
    }

    pub fn learn(&mut self) -> Option<LearnReport> {
        let mut report = self.core.learn()?;
        self.refresh_encoder(&mut report);
        Some(report)
    }

    fn refresh_encoder(&mut self, report: &mut LearnReport) {
        if let (Some(history), VectorPerception::Causal(perception)) =
            (&mut self.history, &mut self.perception)
        {
            let started = Instant::now();
            sync_matching(
                &self.core.learner.world_train,
                perception.session_mut(),
                "encoder.",
            );
            history.refresh(perception);
            let elapsed = started.elapsed().as_secs_f64();
            report.timing.world_sync_seconds += elapsed;
            report.timing.total_seconds += elapsed;
        }
    }

    fn store_history(&mut self, streams: &[usize]) {
        if let (Some(history), VectorPerception::Causal(perception)) =
            (&mut self.history, &self.perception)
        {
            history.store(perception, streams);
        }
    }
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.core.learner.save_checkpoint(path)
    }
}

impl Drop for VectorDreamerAgent {
    fn drop(&mut self) {
        if let Some(history) = &mut self.history {
            history.copies.wait();
        }
        self.core.acting.wait();
        self.core.copies.wait();
        self.core.learner.replay.wait_device();
    }
}

#[cfg(test)]
mod joint_qualification {
    use super::*;

    #[test]
    #[ignore = "requires declared GPU, oracle directory, KINDLE_JOINT_PROBE_MODE and fresh KINDLE_JOINT_PROBE_OUTPUT"]
    fn full_video_updates() {
        let mode = match std::env::var("KINDLE_JOINT_PROBE_MODE").unwrap().as_str() {
            "joint" => crate::dreamer::VideoEncoder::Joint,
            "frozen" => crate::dreamer::VideoEncoder::Frozen,
            other => panic!("unknown probe mode {other}"),
        };
        let root = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let checkpoint = Path::new(manifest["checkpoint"].as_str().unwrap());
        assert_eq!(
            crate::vision::checkpoint_sha256(checkpoint).unwrap(),
            manifest["checkpoint_sha256"].as_str().unwrap()
        );
        let output = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_PROBE_OUTPUT").unwrap());
        std::fs::create_dir(&output).unwrap();
        let mut config = DreamerConfig::new(18);
        config.model_size = crate::dreamer::ModelSize::Size1M;
        config.batch_size = 8;
        config.batch_length = 16;
        config.world_backprop_length = 16;
        config.world_microbatch_size = Some(1);
        config.replay_capacity = 8192;
        config.seed = 1009;
        config.video_encoder = Some(mode);
        config.actor_critic_gradient = std::env::var_os("KINDLE_AC_GRADS").is_some();
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        config.validate();
        eprintln!("full video probe: constructing {mode:?}");
        let start = Instant::now();
        let mut agent = VectorDreamerAgent::new(config.clone(), 8, checkpoint).unwrap();
        let construction_seconds = start.elapsed().as_secs_f64();
        let inspect = |agent: &VectorDreamerAgent| {
            assert_eq!(
                agent.gpu_device().device_name,
                std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
            );
            assert!(!agent.gpu_device().is_software_emulated);
            let memory = agent.gpu_memory_budget();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
            serde_json::to_value(memory).unwrap()
        };
        let constructed_memory = inspect(&agent);
        eprintln!("constructed in {construction_seconds:.3}s; memory={constructed_memory}");
        let parameter = |agent: &mut VectorDreamerAgent| {
            let name = "encoder.patch_embed.proj.weight";
            let session = &mut agent.core.learner.world_train;
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_param(name, &mut values);
            values
        };
        let before = parameter(&mut agent);
        let frame = |stream: usize, tick: usize| {
            RgbFrame::new(
                224,
                224,
                (0..224 * 224 * 3)
                    .map(|i| ((i * 37 + tick * 13 + stream * 71) % 256) as u8)
                    .collect(),
            )
        };
        agent.begin_episodes(&(0..8).map(|s| (s, frame(s, 0))).collect::<Vec<_>>());
        // Leave a live prefix so the timing includes actual cache refresh.
        for tick in 1..35 {
            assert!(agent.act(ActionMode::Sample).iter().all(|&a| a < 18));
            agent.observe(
                &(0..8)
                    .map(|s| {
                        (
                            s,
                            Transition {
                                frame: frame(s, tick),
                                reward: Reward {
                                    extrinsic: if (tick + s).is_multiple_of(7) {
                                        1.0
                                    } else {
                                        0.0
                                    },
                                    intrinsic: 0.0,
                                },
                                terminated: false,
                                truncated: false,
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let mut reports = Vec::new();
        eprintln!("replay populated; starting three full updates");
        for _ in 0..3 {
            let report = agent.learn().expect("eight complete causal replay chunks");
            assert!(report.world.total_loss.is_finite());
            assert!(report.world.encoder_spread.is_finite() && report.world.encoder_spread > 0.0);
            assert!(report.behavior.total_loss.is_finite());
            println!("{}", serde_json::to_string(&report).unwrap());
            reports.push(report);
            inspect(&agent);
        }
        let after = parameter(&mut agent);
        assert!(after.iter().all(|x| x.is_finite()));
        assert_eq!(before != after, mode == crate::dreamer::VideoEncoder::Joint);
        let updated_memory = inspect(&agent);
        let saved = output.join("checkpoint");
        agent.save_checkpoint(&saved).unwrap();
        let last_session_gpu_timings = [
            ("posterior", &agent.core.learner.world_posterior),
            ("world", &agent.core.learner.world_train),
            ("imagination", &agent.core.learner.imagination),
            ("behavior", &agent.core.learner.behavior_train),
        ]
        .map(|(name, session)| {
            (
                name,
                session
                    .gpu_timings()
                    .into_iter()
                    .map(|(label, time)| (label, time.as_secs_f64()))
                    .collect::<Vec<_>>(),
            )
        });
        std::fs::write(output.join("result.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "config": config, "streams": 8, "synthetic_actions": 272, "game_actions": 0,
            "construction_seconds": construction_seconds, "reports": reports,
            "constructed_memory": constructed_memory, "updated_memory": updated_memory,
            "encoder_moved": before != after,
            "last_session_gpu_timings": last_session_gpu_timings,
            "limitations": ["three synthetic updates, not learning or steady-state throughput", "replay only partly populated", "restore is separately qualified", "last session timings are not whole-update or device utilization"]
        })).unwrap()).unwrap();
        if std::env::var_os("KINDLE_JOINT_PROBE_PROFILE").is_some() {
            agent
                .core
                .learner
                .profile_sessions(output.join("profiles"))
                .unwrap();
        }
    }

    #[test]
    #[ignore = "requires declared GPU, oracle directory and KINDLE_JOINT_RESTORE_CHECKPOINT"]
    fn video_checkpoint_restore_is_frozen() {
        let root = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let saved =
            std::path::PathBuf::from(std::env::var("KINDLE_JOINT_RESTORE_CHECKPOINT").unwrap());
        let metadata = read_checkpoint_metadata(&saved).unwrap();
        let tensors =
            meganeura::data::safetensors::SafeTensorsModel::load(saved.join("world.safetensors"))
                .unwrap();
        let mut agent =
            VectorDreamerAgent::restore(&saved, 8, manifest["checkpoint"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            agent.gpu_device().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!agent.gpu_device().is_software_emulated);
        let memory = agent.gpu_memory_budget();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        assert_eq!(agent.learner_step(), metadata.learner_step);
        assert_eq!(agent.environment_step(), metadata.environment_step);
        let session = &mut agent.core.learner.world_train;
        let names = session
            .param_names()
            .iter()
            .filter(|n| n.starts_with("encoder."))
            .map(|n| n.to_string())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 148);
        for name in names {
            let mut actual = vec![0.0; session.param_size(&name).unwrap()];
            session.read_param(&name, &mut actual);
            assert_eq!(actual, tensors.tensor_f32_auto(&name).unwrap(), "{name}");
        }
        agent.begin_episodes(
            &(0..8)
                .map(|s| {
                    (
                        s,
                        RgbFrame::new(224, 224, vec![s as u8 * 17; 224 * 224 * 3]),
                    )
                })
                .collect::<Vec<_>>(),
        );
        assert!(agent.act(ActionMode::Greedy).iter().all(|&a| a < 18));
        assert_eq!(agent.learner_step(), metadata.learner_step);
        println!(
            "148 encoder tensors restored exactly; frozen action has zero updates; memory={memory:?}"
        );
    }

    #[test]
    #[ignore = "requires separately declared GPU and KINDLE_JOINT_TINY_REFERENCE oracle directory"]
    fn joint_encoder_cache_matches_fresh_encoding_of_live_prefixes() {
        let root = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let checkpoint = Path::new(manifest["checkpoint"].as_str().unwrap());
        assert_eq!(
            crate::vision::checkpoint_sha256(checkpoint).unwrap(),
            manifest["checkpoint_sha256"].as_str().unwrap()
        );
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let device = gpu.device_information();
        assert_eq!(
            device.device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!device.is_software_emulated);
        let load = || {
            LeVJepaPerception::load_batched_with_architecture(
                crate::vision::levjepa::Architecture::Tiny,
                checkpoint,
                2,
                Some(Arc::clone(&gpu)),
                None,
            )
            .unwrap()
        };
        let mut actual = load();
        let mut reference = load();
        let mut history = PixelHistory::new(Arc::clone(&gpu), 2);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        let frame = |stream: usize, tick: usize| {
            RgbFrame::new(
                224,
                224,
                (0..224 * 224 * 3)
                    .map(|i| ((i * 37 + tick * 13 + stream * 71) % 256) as u8)
                    .collect(),
            )
        };
        let mut prefixes = [Vec::new(), Vec::new()];
        for tick in 0..24 {
            for (stream, prefix) in prefixes.iter_mut().enumerate() {
                if stream == 1 && tick % 3 == 0 {
                    continue;
                }
                let reset = prefix.is_empty() || (stream == 1 && tick == 11);
                if reset || actual.chunk_positions()[stream] == 0 {
                    prefix.clear();
                }
                prefix.push(frame(stream, tick));
                actual.encode_frames_rgb8(&[(stream, prefix.last().unwrap(), reset)]);
                history.store(&actual, &[stream]);
            }
        }
        // Emulate an optimizer weight change; compare refresh to a separately
        // rebuilt prefix using the original RGB, not PixelHistory's buffers.
        let name = "encoder.patch_embed.proj.bias";
        let mut bias = vec![0.0; 192];
        actual.session().read_param(name, &mut bias);
        for (i, value) in bias.iter_mut().enumerate() {
            *value += (i % 5) as f32 * 0.007;
        }
        actual.session_mut().set_parameter(name, &bias);
        reference.session_mut().set_parameter(name, &bias);
        let positions = actual.chunk_positions().to_vec();
        history.refresh(&mut actual);
        assert_eq!(actual.chunk_positions(), positions);
        for phase in 0..positions.iter().copied().max().unwrap() {
            let arrivals = prefixes
                .iter()
                .enumerate()
                .filter(|(stream, _)| phase < positions[*stream])
                .map(|(stream, frames)| (stream, &frames[phase], phase == 0))
                .collect::<Vec<_>>();
            reference.encode_frames_rgb8(&arrivals);
        }
        for tick in 24..27 {
            let next = [frame(0, tick), frame(1, tick)];
            let arrivals = [(0, &next[0], false), (1, &next[1], tick == 26)];
            let actual = actual.encode_frames_rgb8(&arrivals);
            let reference = reference.encode_frames_rgb8(&arrivals);
            for (a, b) in actual.iter().zip(&reference) {
                for (&a, &b) in a.as_slice().iter().zip(b.as_slice()) {
                    assert!((a - b).abs() < 1e-4 * b.abs() + 1e-5, "{a} != {b}");
                }
            }
        }
        // History must complete before its session-owned sources are dropped.
        history.copies.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_capacity_accounts_for_nonoverlapping_chunks() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 8;
        config.batch_length = 16;
        config.world_backprop_length = 16;
        config.video_encoder = Some(crate::dreamer::VideoEncoder::Joint);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        assert_eq!(config.replay_warmup_sequences(), 8);
        assert_eq!(config.replay_warmup_frames(), 144);
        config.replay_capacity = 255;
        assert!(check_capacity(&config, 8).is_err());
        config.replay_capacity = 256;
        assert!(check_capacity(&config, 8).is_ok());
        config.check().unwrap();
        assert!(FeatureVectorAgent::new(config.clone(), 8).is_err());
        assert!(
            VectorDreamerAgent::with_perception(config, 8, PerceptionKind::LeVJepa, "unused")
                .is_err()
        );
    }

    #[test]
    fn rgb_configuration_refusals_precede_device_initialization() {
        let config = DreamerConfig::tiny(3);
        assert!(VectorDreamerAgent::learned_rgb(config.clone(), 2).is_err());
        let mut rgb = config.clone();
        rgb.observation_kind = ObservationKind::Rgb64;
        assert!(VectorDreamerAgent::learned_rgb(rgb.clone(), 0).is_err());
        assert!(FeatureVectorAgent::new(rgb.clone(), 2).is_err());
        assert!(
            VectorDreamerAgent::with_perception(
                rgb.clone(),
                2,
                PerceptionKind::LeVJepaTiny,
                "unused"
            )
            .is_err()
        );
        rgb.loss_scales.future_prediction = 0.25;
        assert!(rgb.check().is_err());
        rgb.loss_scales.future_prediction = 0.0;
        rgb.visitation_bonus = true;
        assert!(rgb.check().is_err());
    }

    #[test]
    #[ignore = "requires GPU; joint RGB learning, pixel replay, stream isolation and restore"]
    fn tiny_rgb_replay_is_reencoded_and_checkpoint_restores() {
        check_rgb_replay_and_restore(false);
    }

    #[test]
    #[ignore = "requires separately guarded GPU; CDP encoder learning, pixel replay, stream isolation and restore"]
    fn tiny_cdp_replay_is_reencoded_and_checkpoint_restores() {
        check_rgb_replay_and_restore(true);
    }

    fn check_rgb_replay_and_restore(cdp: bool) {
        let checkpoint = check_rgb_vector_replay_and_restore(cdp);
        check_rgb_single_restore(&checkpoint);
        fs::remove_dir_all(checkpoint).unwrap();
    }

    fn check_rgb_vector_replay_and_restore(cdp: bool) -> std::path::PathBuf {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        config.replay_capacity = 32;
        config.train_ratio = 0.0;
        if cdp {
            config.loss_scales.reconstruction = 0.0;
            config.loss_scales.future_prediction = 500.0;
            config.encoder_learning_rate = Some(6e-6);
            config.dynamics_learning_rate = Some(4e-4);
        }
        let mut agent = VectorDreamerAgent::learned_rgb(config.clone(), 2).unwrap();
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            assert_eq!(agent.gpu_device().device_name, expected);
            assert!(!agent.gpu_device().is_software_emulated);
            let memory = agent.gpu_memory_budget();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        }
        let frame = |id: usize, time: usize| {
            RgbFrame::new(
                19,
                13,
                (0..19 * 13 * 3)
                    .map(|i| ((i * 17 + time * 11 + id * 31) % 256) as u8)
                    .collect(),
            )
        };
        agent.begin_episodes(&[(0, frame(0, 0)), (1, frame(1, 0))]);
        for time in 1..12 {
            agent.act(ActionMode::Sample);
            agent.observe(
                &(0..2)
                    .map(|id| {
                        (
                            id,
                            Transition {
                                frame: frame(id, time),
                                reward: Reward {
                                    extrinsic: (time % 3) as f32,
                                    intrinsic: 0.0,
                                },
                                terminated: time == 11 && id == 0,
                                truncated: false,
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let untouched = agent.core.feature(1);
        agent.begin_episodes(&[(0, frame(0, 12))]);
        assert_eq!(untouched, agent.core.feature(1));
        let sample = |agent: &mut VectorDreamerAgent| {
            agent
                .core
                .learner
                .replay
                .sample(&config, &mut StdRng::seed_from_u64(773))
                .unwrap()
        };
        // Fresh-arrival sampling is a consumable queue, independent of RNG.
        // Exhaust it before comparing the same seeded random replay windows.
        for _ in 0..agent.replay_len() {
            sample(&mut agent);
        }
        let before = sample(&mut agent);
        assert!(
            before
                .observations
                .iter()
                .all(|row| row.len() == config.batch_size * 12288
                    && row.iter().all(|x| (-0.5..=0.5).contains(x)))
        );
        let names = agent
            .core
            .learner
            .world_train
            .param_names()
            .into_iter()
            .filter(|name| name.starts_with("world.representation.encoder."))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        let weights = agent.core.learner.world_train.read_params(&names);
        let encoder_output = |agent: &mut VectorDreamerAgent| {
            agent.core.observe.step();
            agent.core.observe.wait();
            let mut values = vec![0.0; 2 * config.encoded_observation_dim()];
            agent.core.observe.read_output_by_index(3, &mut values);
            values
        };
        let encoded_before = encoder_output(&mut agent);
        let report = agent.core.learn().unwrap();
        assert!(report.world.total_loss.is_finite());
        if cdp {
            assert_eq!(report.world.reconstruction_loss, 0.0);
            assert!(report.world.future_prediction_loss > 0.0);
            assert!(
                !agent
                    .core
                    .learner
                    .world_train
                    .param_names()
                    .iter()
                    .any(|n| n.starts_with("world.decoder."))
            );
        } else {
            assert!(report.world.reconstruction_loss > 0.0);
        }
        let updated = agent.core.learner.world_train.read_params(&names);
        assert!(
            weights
                .iter()
                .zip(&updated)
                .all(|(a, b)| a != b && b.iter().all(|x| x.is_finite()))
        );
        assert_ne!(encoded_before, encoder_output(&mut agent));
        assert_eq!(before.observations, sample(&mut agent).observations);
        let moments = agent.core.learner.world_train.read_adam_states(&names);
        let checkpoint =
            std::env::temp_dir().join(format!("kindle-rgb-test-{}", std::process::id()));
        assert!(!checkpoint.exists());
        agent.save_checkpoint(&checkpoint).unwrap();
        drop(agent);
        let mut restored = VectorDreamerAgent::restore_rgb(&checkpoint, 2).unwrap();
        assert_eq!(restored.config(), &config);
        assert_eq!(restored.replay_len(), 0);
        assert_eq!(restored.learner_step(), 1);
        assert_eq!(
            updated,
            restored.core.learner.world_train.read_params(&names)
        );
        assert_eq!(
            moments,
            restored.core.learner.world_train.read_adam_states(&names)
        );
        restored.begin_episodes(&[(0, frame(0, 0)), (1, frame(1, 0))]);
        restored.act(ActionMode::Greedy);
        assert_eq!(
            updated,
            restored.core.learner.world_train.read_params(&names)
        );
        assert_eq!(
            moments,
            restored.core.learner.world_train.read_adam_states(&names)
        );
        drop(restored);
        checkpoint
    }

    fn check_rgb_single_restore(checkpoint: &Path) {
        let mut single = DreamerAgent::restore_rgb(checkpoint).unwrap();
        single.begin_episode(&RgbFrame::new(19, 13, vec![127; 19 * 13 * 3]));
        let width = single.core().config().encoded_observation_dim();
        assert_eq!(single.encoded_observation().len(), width);
        let prediction_width = single.core().config().prediction_dim();
        let prediction = single.observation_prediction();
        assert_eq!(prediction.len(), prediction_width);
        assert!(prediction.iter().all(|x| x.is_finite()));
        let before = single.latent_feature().to_vec();
        let forecast = single.prior_state_rollout(&[1, 2]);
        assert!(forecast.1.iter().all(|row| row.len() == prediction_width));
        assert_eq!(forecast, single.prior_state_rollout(&[1, 2]));
        assert_eq!(before, single.latent_feature());
        assert_eq!(single.core().learner_step(), 1);
    }

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
