//! Online-first sequence replay with D3-aligned frame semantics.

use std::collections::VecDeque;

use rand::Rng;

use super::config::DreamerConfig;
use super::replay_gpu::DeviceReplay;
use crate::vision::Observation;

#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct Reward {
    pub extrinsic: f32,
    pub intrinsic: f32,
}

impl Reward {
    pub fn combined(self, config: &DreamerConfig) -> f32 {
        let combined = config.extrinsic_reward_scale * self.extrinsic
            + config.intrinsic_reward_scale * self.intrinsic;
        assert!(combined.is_finite(), "scaled reward must remain finite");
        combined
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FrameFlags {
    /// Reset the recurrent state before incorporating this observation.
    pub is_first: bool,
    /// This frame ends a replay return trace (timeout or terminal).
    pub is_last: bool,
    /// This frame is a true terminal and must not bootstrap.
    pub is_terminal: bool,
}

#[derive(Clone, Debug)]
pub struct ReplayFrame {
    pub observation: Observation,
    /// Action taken from the preceding frame to reach this observation.
    pub previous_action: Option<usize>,
    /// Reward observed together with this frame, caused by `previous_action`.
    pub reward: Reward,
    pub flags: FrameFlags,
    /// Cached posterior context from acting or the latest replay refresh.
    pub deter: Box<[f32]>,
    pub stoch: Box<[f32]>,
}

impl ReplayFrame {
    pub fn validate(&self, config: &DreamerConfig) {
        let size = config.network();
        assert_eq!(self.observation.as_slice().len(), config.observation_dim());
        assert!(
            self.previous_action
                .is_none_or(|action| action < config.action_count)
        );
        assert!(self.reward.extrinsic.is_finite() && self.reward.intrinsic.is_finite());
        assert_eq!(self.deter.len(), size.deter);
        assert_eq!(self.stoch.len(), size.stoch * size.classes);
        if self.flags.is_first {
            assert!(self.previous_action.is_none());
        }
        if self.flags.is_terminal {
            assert!(self.flags.is_last);
        }
    }
}

pub struct SequenceReplay {
    capacity: usize,
    streams: Vec<ReplayStream>,
    arrival_order: VecDeque<usize>,
    fresh_starts: VecDeque<(usize, u64)>,
    device: Option<DeviceReplay>,
    next_device_slot: usize,
}

struct StoredFrame {
    values: FrameValues,
    previous_action: Option<usize>,
    reward: Reward,
    flags: FrameFlags,
}

struct ReplayContext {
    deter: Box<[f32]>,
    stoch: Box<[f32]>,
}

enum FrameValues {
    Host {
        observation: Observation,
        deter: Box<[f32]>,
        stoch: Box<[f32]>,
    },
    Device {
        slot: usize,
        refreshed: Option<ReplayContext>,
    },
}

#[derive(Default)]
struct ReplayStream {
    frames: VecDeque<StoredFrame>,
    total_frames: u64,
    next_encoder_phase: usize,
    video_starts: VecDeque<u64>,
}

impl ReplayStream {
    fn oldest_frame(&self) -> u64 {
        self.total_frames - self.frames.len() as u64
    }
}

impl SequenceReplay {
    pub fn new(capacity: usize) -> Self {
        Self::with_streams(capacity, 1)
    }

    pub fn with_streams(capacity: usize, streams: usize) -> Self {
        assert!(capacity > 1);
        assert!(streams > 0);
        Self {
            capacity,
            streams: (0..streams).map(|_| ReplayStream::default()).collect(),
            arrival_order: VecDeque::with_capacity(capacity.min(65_536)),
            fresh_starts: VecDeque::new(),
            device: None,
            next_device_slot: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.arrival_order.len()
    }

    /// Number of complete context-plus-training sequences currently eligible
    /// for sampling. Upstream D3 gates scheduled training on at least
    /// `batch_size * batch_length` such replay items.
    pub fn valid_sequence_count(&self, config: &DreamerConfig) -> usize {
        let required = config.replay_context + config.batch_length;
        self.streams
            .iter()
            .map(|stream| {
                if config.video_encoder.is_some() {
                    stream.video_starts.len()
                } else {
                    stream.frames.len().saturating_sub(required - 1)
                }
            })
            .sum()
    }

    pub fn push(&mut self, frame: ReplayFrame, config: &DreamerConfig) {
        self.push_stream(0, frame, config);
    }

    pub fn push_stream(&mut self, stream: usize, frame: ReplayFrame, config: &DreamerConfig) {
        assert!(
            config.video_encoder.is_none(),
            "causal video replay requires GPU pixel arrivals"
        );
        assert!(
            self.device.is_none(),
            "cannot mix host and device replay storage"
        );
        frame.validate(config);
        self.push_stored(
            stream,
            StoredFrame {
                values: FrameValues::Host {
                    observation: frame.observation,
                    deter: frame.deter,
                    stoch: frame.stoch,
                },
                previous_action: frame.previous_action,
                reward: frame.reward,
                flags: frame.flags,
            },
            config,
        );
    }

    pub(super) fn enable_device(
        &mut self,
        gpu: std::sync::Arc<blade_graphics::Context>,
        config: &DreamerConfig,
    ) {
        assert_eq!(self.len(), 0);
        self.device = Some(DeviceReplay::new(gpu, config));
    }

    pub(super) fn wait_device(&mut self) {
        if let Some(device) = &mut self.device {
            device.wait();
        }
    }

    pub(super) fn store_pixels(&mut self, perception: &meganeura::Session, streams: &[usize]) {
        let slots = streams
            .iter()
            .enumerate()
            .map(|(i, &stream)| (stream, (self.next_device_slot + i) % self.capacity))
            .collect::<Vec<_>>();
        self.device
            .as_mut()
            .unwrap()
            .store_pixels(perception, &slots);
    }

    /// Called after posterior commit, before overwriting observation/state inputs.
    pub(super) fn push_device(
        &mut self,
        session: &meganeura::Session,
        arrivals: &[(usize, Option<usize>, Reward, FrameFlags)],
        config: &DreamerConfig,
    ) {
        assert!(arrivals.len() <= self.capacity);
        let slots: Vec<_> = arrivals
            .iter()
            .enumerate()
            .map(|(i, a)| (a.0, (self.next_device_slot + i) % self.capacity))
            .collect();
        self.device
            .as_mut()
            .expect("device replay enabled")
            .store(session, &slots);
        for (&(stream, previous_action, reward, flags), &(_, slot)) in arrivals.iter().zip(&slots) {
            self.push_stored(
                stream,
                StoredFrame {
                    values: FrameValues::Device {
                        slot,
                        refreshed: None,
                    },
                    previous_action,
                    reward,
                    flags,
                },
                config,
            );
        }
        self.next_device_slot = (self.next_device_slot + arrivals.len()) % self.capacity;
    }

    fn push_stored(&mut self, stream: usize, frame: StoredFrame, config: &DreamerConfig) {
        assert!(stream < self.streams.len());
        if self.len() == self.capacity {
            let oldest = self.arrival_order.pop_front().unwrap();
            self.streams[oldest].frames.pop_front();
            let oldest_frame = self.streams[oldest].oldest_frame();
            while self.streams[oldest]
                .video_starts
                .front()
                .is_some_and(|&start| start < oldest_frame)
            {
                self.streams[oldest].video_starts.pop_front();
            }
        }
        self.arrival_order.push_back(stream);
        let history = &mut self.streams[stream];
        if frame.flags.is_first {
            history.next_encoder_phase = 0;
        }
        let phase = history.next_encoder_phase;
        history.next_encoder_phase = (phase + 1) % crate::vision::levjepa::FRAMES;
        history.frames.push_back(frame);

        history.total_frames = history
            .total_frames
            .checked_add(1)
            .expect("replay frame index overflowed");
        let required = (config.replay_context + config.batch_length) as u64;
        // D3's online counter is checked before it is incremented, so it skips
        // start zero and queues fresh non-overlapping starts 1, 1 + required,
        // and so on. Learner batches drain this queue before sampling uniformly.
        if config.video_encoder.is_some() {
            if phase + 1 == crate::vision::levjepa::FRAMES && history.total_frames >= required {
                let start = history.total_frames - required;
                if start >= history.oldest_frame() {
                    history.video_starts.push_back(start);
                    self.fresh_starts.push_back((stream, start));
                }
            }
        } else if history.total_frames > required
            && (history.total_frames - 1).is_multiple_of(required)
        {
            self.fresh_starts
                .push_back((stream, history.total_frames - required));
        }

        // Prune lazily at the head; stale items later in the queue are skipped
        // when reached. No scan proportional to replay capacity on every action.
        while self
            .fresh_starts
            .front()
            .is_some_and(|&(stream, start)| start < self.streams[stream].oldest_frame())
        {
            self.fresh_starts.pop_front();
        }
    }

    pub fn sample(&mut self, config: &DreamerConfig, rng: &mut impl Rng) -> Option<SequenceBatch> {
        let required = config.replay_context + config.batch_length;
        let counts: Vec<_> = self
            .streams
            .iter()
            .map(|stream| {
                if config.video_encoder.is_some() {
                    stream.video_starts.len()
                } else {
                    stream.frames.len().saturating_sub(required - 1)
                }
            })
            .collect();
        let valid_count: usize = counts.iter().sum();
        if valid_count == 0 {
            return None;
        }
        let size = config.network();
        let batch = config.batch_size;
        let length = config.batch_length;
        let mut initial_deter = vec![0.0; batch * size.deter];
        let mut initial_stoch = vec![0.0; batch * size.stoch * size.classes];
        let mut observations = (0..length)
            .map(|_| vec![0.0; batch * config.observation_dim()])
            .collect::<Vec<_>>();
        let mut pixels = config.video_encoder.map(|_| {
            (0..length)
                .map(|_| vec![0.0; batch * crate::vision::levjepa::joint::PIXELS])
                .collect::<Vec<_>>()
        });
        let mut previous_actions = (0..length)
            .map(|_| vec![0.0; batch * config.action_count])
            .collect::<Vec<_>>();
        let mut rewards = (0..length).map(|_| vec![0.0; batch]).collect::<Vec<_>>();
        let mut flags = (0..length)
            .map(|_| vec![FrameFlags::default(); batch])
            .collect::<Vec<_>>();
        let mut frame_indices = (0..length).map(|_| vec![(0, 0); batch]).collect::<Vec<_>>();
        let mut device_copies = Vec::new();

        for row in 0..batch {
            let fresh = loop {
                match self.fresh_starts.pop_front() {
                    Some((stream, start)) if start >= self.streams[stream].oldest_frame() => {
                        break Some((
                            stream,
                            (start - self.streams[stream].oldest_frame()) as usize,
                        ));
                    }
                    Some(_) => continue,
                    None => break None,
                }
            };
            let (stream, start) = if let Some(fresh) = fresh {
                fresh
            } else {
                let mut start = rng.random_range(0..valid_count);
                let stream = counts
                    .iter()
                    .position(|&count| {
                        if start < count {
                            true
                        } else {
                            start -= count;
                            false
                        }
                    })
                    .unwrap();
                if config.video_encoder.is_some() {
                    start = (self.streams[stream].video_starts[start]
                        - self.streams[stream].oldest_frame()) as usize;
                }
                (stream, start)
            };
            assert!(
                start + required <= self.streams[stream].frames.len(),
                "fresh replay start is invalid"
            );
            let frames = &self.streams[stream].frames;
            let context = &frames[start + config.replay_context - 1];
            let stoch_width = size.stoch * size.classes;
            let deter_target = &mut initial_deter[row * size.deter..(row + 1) * size.deter];
            let stoch_target = &mut initial_stoch[row * stoch_width..(row + 1) * stoch_width];
            match &context.values {
                FrameValues::Host { deter, stoch, .. }
                | FrameValues::Device {
                    refreshed: Some(ReplayContext { deter, stoch }),
                    ..
                } => {
                    deter_target.copy_from_slice(deter);
                    stoch_target.copy_from_slice(stoch);
                }
                FrameValues::Device {
                    slot,
                    refreshed: None,
                } => {
                    device_copies.push((
                        *slot,
                        config.observation_dim(),
                        0,
                        row * size.deter,
                        size.deter,
                    ));
                    device_copies.push((
                        *slot,
                        config.observation_dim() + size.deter,
                        1,
                        row * stoch_width,
                        stoch_width,
                    ));
                }
            }

            for time in 0..length {
                let index = start + config.replay_context + time;
                let frame = &frames[index];
                match &frame.values {
                    FrameValues::Host { observation, .. } => observations[time]
                        [row * config.observation_dim()..(row + 1) * config.observation_dim()]
                        .copy_from_slice(observation.as_slice()),
                    FrameValues::Device { slot, .. } => {
                        if pixels.is_some() {
                            let width = crate::vision::levjepa::joint::PIXELS;
                            device_copies.push((
                                *slot,
                                config.observation_dim() + config.feature_dim(),
                                time + length + 2,
                                row * width,
                                width,
                            ));
                        } else {
                            device_copies.push((
                                *slot,
                                0,
                                time + 2,
                                row * config.observation_dim(),
                                config.observation_dim(),
                            ));
                        }
                    }
                }
                if let Some(action) = frame.previous_action {
                    previous_actions[time][row * config.action_count + action] = 1.0;
                }
                rewards[time][row] = frame.reward.combined(config);
                flags[time][row] = frame.flags;
                frame_indices[time][row] = (stream, index);
            }
        }

        if !device_copies.is_empty() {
            let device = self.device.as_mut().unwrap();
            let mut downloaded: Vec<Vec<f32>> =
                device_copies.iter().map(|row| vec![0.0; row.4]).collect();
            let mut regions: Vec<_> = device_copies
                .iter()
                .zip(&mut downloaded)
                .map(|(row, values)| (device.region(row.0, row.1), values.as_mut_slice()))
                .collect();
            device.readback.read_regions(&mut regions);
            for ((_, _, target, offset, width), values) in device_copies.into_iter().zip(downloaded)
            {
                let target = match target {
                    0 => &mut initial_deter,
                    1 => &mut initial_stoch,
                    time if time >= length + 2 => &mut pixels.as_mut().unwrap()[time - length - 2],
                    time => &mut observations[time - 2],
                };
                target[offset..offset + width].copy_from_slice(&values);
            }
        }

        Some(SequenceBatch {
            initial_deter,
            initial_stoch,
            observations,
            pixels,
            previous_actions,
            rewards,
            flags,
            frame_indices,
        })
    }

    pub fn update_context(
        &mut self,
        batch: &SequenceBatch,
        deter: &[Vec<f32>],
        stoch: &[Vec<f32>],
        config: &DreamerConfig,
    ) {
        let size = config.network();
        assert_eq!(deter.len(), config.batch_length);
        assert_eq!(stoch.len(), config.batch_length);
        for time in 0..config.batch_length {
            assert_eq!(deter[time].len(), config.batch_size * size.deter);
            assert_eq!(
                stoch[time].len(),
                config.batch_size * size.stoch * size.classes
            );
            for row in 0..config.batch_size {
                let (stream, index) = batch.frame_indices[time][row];
                let frame = &mut self.streams[stream].frames[index];
                let width = size.stoch * size.classes;
                let next_deter = &deter[time][row * size.deter..(row + 1) * size.deter];
                let next_stoch = &stoch[time][row * width..(row + 1) * width];
                match &mut frame.values {
                    FrameValues::Host { deter, stoch, .. } => {
                        deter.copy_from_slice(next_deter);
                        stoch.copy_from_slice(next_stoch);
                    }
                    FrameValues::Device { refreshed, .. } => {
                        *refreshed = Some(ReplayContext {
                            deter: next_deter.into(),
                            stoch: next_stoch.into(),
                        });
                    }
                }
            }
        }
    }
}

#[cfg_attr(test, derive(Default))]
pub struct SequenceBatch {
    pub initial_deter: Vec<f32>,
    pub initial_stoch: Vec<f32>,
    pub observations: Vec<Vec<f32>>,
    pub pixels: Option<Vec<Vec<f32>>>,
    pub previous_actions: Vec<Vec<f32>>,
    pub rewards: Vec<Vec<f32>>,
    pub flags: Vec<Vec<FrameFlags>>,
    frame_indices: Vec<Vec<(usize, usize)>>,
}

impl SequenceBatch {
    pub fn keep(&self, time: usize, row: usize) -> f32 {
        if self.flags[time][row].is_first {
            0.0
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn video_replay_chunks_follow_arrivals_resets_and_eviction() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = crate::vision::levjepa::FRAMES;
        config.world_backprop_length = config.batch_length;
        config.video_encoder = Some(super::super::VideoEncoder::Joint);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let mut replay = SequenceReplay::with_streams(100, 2);
        let mut expected = [Vec::new(), Vec::new()];
        let mut positions = [0, 0];
        let mut totals = [0, 0];
        for tick in 0..120 {
            for stream in 0..2 {
                // Different arrival rates, real resets and wraparound.
                if stream == 1 && tick % 3 == 0 {
                    continue;
                }
                let first = totals[stream] == 0 || (stream == 1 && tick == 43);
                if first {
                    positions[stream] = 0;
                }
                if positions[stream] == 15 && totals[stream] >= 16 {
                    expected[stream].push(totals[stream] - 16);
                }
                positions[stream] = (positions[stream] + 1) % 16;
                totals[stream] += 1;
                // Exercise metadata only; production host pushes are rejected
                // because ReplayFrame intentionally cannot fabricate pixels.
                replay.push_stored(
                    stream,
                    StoredFrame {
                        values: FrameValues::Host {
                            observation: Observation::from_vec(vec![0.0; config.observation_dim()]),
                            deter: Box::new([]),
                            stoch: Box::new([]),
                        },
                        previous_action: (!first).then_some(0),
                        reward: Reward::default(),
                        flags: FrameFlags {
                            is_first: first,
                            ..Default::default()
                        },
                    },
                    &config,
                );
                for (id, starts) in expected.iter().enumerate() {
                    let history = &replay.streams[id];
                    let retained = starts
                        .iter()
                        .copied()
                        .filter(|&start| start >= history.oldest_frame())
                        .collect::<VecDeque<_>>();
                    assert_eq!(history.video_starts, retained);
                    for &start in &history.video_starts {
                        let offset = (start - history.oldest_frame()) as usize;
                        assert_eq!(
                            history
                                .frames
                                .range(offset + 2..offset + 17)
                                .filter(|f| f.flags.is_first)
                                .count(),
                            0
                        );
                    }
                }
                assert_eq!(
                    replay.valid_sequence_count(&config),
                    replay
                        .streams
                        .iter()
                        .map(|s| s.video_starts.len())
                        .sum::<usize>()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "requires GPU pixel arrivals")]
    fn video_replay_cannot_silently_train_on_missing_pixels() {
        let mut config = DreamerConfig::tiny(3);
        config.video_encoder = Some(super::super::VideoEncoder::Joint);
        SequenceReplay::new(32).push(frame(0, &config), &config);
    }

    fn frame(index: usize, config: &DreamerConfig) -> ReplayFrame {
        let size = config.network();
        ReplayFrame {
            observation: Observation::from_vec(vec![index as f32; config.observation_dim()]),
            previous_action: (index > 0).then_some(index % config.action_count),
            reward: Reward {
                extrinsic: index as f32,
                intrinsic: 100.0,
            },
            flags: FrameFlags {
                is_first: index == 0,
                ..FrameFlags::default()
            },
            deter: vec![index as f32; size.deter].into_boxed_slice(),
            stoch: vec![index as f32; size.stoch * size.classes].into_boxed_slice(),
        }
    }

    #[test]
    fn context_frame_precedes_every_training_sequence() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 1;
        config.batch_length = 3;
        config.intrinsic_reward_scale = 0.5;
        let mut replay = SequenceReplay::new(16);
        for index in 0..4 {
            replay.push(frame(index, &config), &config);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let batch = replay.sample(&config, &mut rng).unwrap();
        assert_eq!(replay.valid_sequence_count(&config), 1);
        assert_eq!(batch.initial_deter[0], 0.0);
        assert_eq!(batch.observations[0][0], 1.0);
        assert_eq!(batch.rewards[0][0], 51.0);
        assert_eq!(batch.previous_actions[0], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn replay_preserves_episode_boundaries_and_action_alignment() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 1;
        config.batch_length = 3;
        let mut replay = SequenceReplay::new(16);

        replay.push(frame(0, &config), &config);
        let mut terminal = frame(1, &config);
        terminal.flags.is_last = true;
        terminal.flags.is_terminal = true;
        let terminal_flags = terminal.flags;
        replay.push(terminal, &config);
        let mut reset = frame(2, &config);
        reset.previous_action = None;
        reset.reward = Reward::default();
        reset.flags.is_first = true;
        let reset_flags = reset.flags;
        replay.push(reset, &config);
        replay.push(frame(3, &config), &config);

        let mut rng = StdRng::seed_from_u64(1);
        let batch = replay.sample(&config, &mut rng).unwrap();

        assert_eq!(batch.flags[0][0], terminal_flags);
        assert_eq!(batch.flags[1][0], reset_flags);
        assert_eq!(batch.flags[2][0], FrameFlags::default());
        assert_eq!(batch.previous_actions[0], vec![0.0, 1.0, 0.0]);
        assert_eq!(batch.previous_actions[1], vec![0.0, 0.0, 0.0]);
        assert_eq!(batch.previous_actions[2], vec![1.0, 0.0, 0.0]);
        assert_eq!(
            [batch.keep(0, 0), batch.keep(1, 0), batch.keep(2, 0)],
            [1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn d3_prefill_counts_complete_sequence_starts() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 4;
        let mut replay = SequenceReplay::new(16);
        for index in 0..11 {
            replay.push(frame(index, &config), &config);
        }
        assert_eq!(replay.valid_sequence_count(&config), 7);
        replay.push(frame(11, &config), &config);
        assert_eq!(replay.valid_sequence_count(&config), 8);
    }

    #[test]
    fn d3_online_queue_precedes_uniform_sampling() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 3;
        let mut replay = SequenceReplay::new(16);
        for index in 0..9 {
            replay.push(frame(index, &config), &config);
        }

        let mut rng = StdRng::seed_from_u64(1);
        let batch = replay.sample(&config, &mut rng).unwrap();
        assert_eq!(batch.frame_indices[0], vec![(0, 2), (0, 6)]);
    }

    #[test]
    fn evicted_online_sequences_are_pruned() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 1;
        config.batch_length = 3;
        let mut replay = SequenceReplay::new(4);
        for index in 0..9 {
            replay.push(frame(index, &config), &config);
        }

        let mut rng = StdRng::seed_from_u64(1);
        let batch = replay.sample(&config, &mut rng).unwrap();
        assert_eq!(batch.frame_indices[0], vec![(0, 1)]);
        assert_eq!(batch.observations[0][0], 6.0);
    }

    #[test]
    fn terminal_implies_last() {
        let config = DreamerConfig::tiny(2);
        let mut invalid = frame(1, &config);
        invalid.flags.is_terminal = true;
        let result = std::panic::catch_unwind(|| invalid.validate(&config));
        assert!(result.is_err());
    }

    #[test]
    fn vector_sequences_never_cross_streams_even_after_eviction_and_refresh() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 8;
        config.batch_length = 3;
        let mut replay = SequenceReplay::with_streams(19, 3);
        for tick in 0..30 {
            for stream in 0..3 {
                // Unequal arrival rates and a shared global eviction budget.
                if stream == 2 && tick % 2 == 0 {
                    continue;
                }
                let index = replay.streams[stream].total_frames as usize;
                replay.push_stream(stream, frame(stream * 100 + index, &config), &config);
            }
        }
        assert_eq!(replay.len(), 19);
        assert_eq!(replay.valid_sequence_count(&config), 10);
        let mut rng = StdRng::seed_from_u64(17);
        for _ in 0..20 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            for row in 0..config.batch_size {
                let (stream, index) = batch.frame_indices[0][row];
                for time in 0..config.batch_length {
                    assert_eq!(batch.frame_indices[time][row], (stream, index + time));
                    assert_eq!(
                        batch.observations[time][row * config.observation_dim()],
                        batch.observations[0][row * config.observation_dim()] + time as f32
                    );
                    assert_eq!((batch.rewards[time][row] / 100.0) as usize, stream);
                }
            }
            let size = config.network();
            let deter = vec![vec![123.0; config.batch_size * size.deter]; config.batch_length];
            let stoch = vec![
                vec![456.0; config.batch_size * size.stoch * size.classes];
                config.batch_length
            ];
            replay.update_context(&batch, &deter, &stoch, &config);
            for indices in &batch.frame_indices {
                for &(stream, index) in indices {
                    let FrameValues::Host { deter, stoch, .. } =
                        &replay.streams[stream].frames[index].values
                    else {
                        panic!("expected host fixture")
                    };
                    assert_eq!(deter[0], 123.0);
                    assert_eq!(stoch[0], 456.0);
                }
            }
        }
    }
}
