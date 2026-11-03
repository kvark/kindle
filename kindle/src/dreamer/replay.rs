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
    completed_return: Option<f32>,
    actor_return_boundary: bool,
    previous_expert_action: Option<usize>,
    previous_action: Option<usize>,
    reward: Reward,
    flags: FrameFlags,
}

pub(super) type DeviceReplayArrival = (usize, Option<usize>, Reward, FrameFlags, Option<usize>);

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

    fn complete_actor_returns(&mut self, config: &DreamerConfig) {
        let mut next = 0.0_f64;
        for (offset, frame) in self.frames.iter_mut().rev().enumerate() {
            if offset > 0 && frame.flags.is_last {
                break;
            }
            // A live boundary arrival starts the next segment, whose outgoing
            // return remains unknown until that segment ends.
            if offset > 0 || frame.flags.is_terminal {
                let value = next as f32;
                assert!(value.is_finite(), "completed observed return overflowed");
                frame.completed_return = Some(value);
            }
            if frame.flags.is_first || (offset > 0 && frame.actor_return_boundary) {
                break;
            }
            next =
                frame.reward.combined(config) as f64 + config.continuation_discount() as f64 * next;
        }
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
                previous_expert_action: None,
                completed_return: None,
                actor_return_boundary: false,
                previous_action: frame.previous_action,
                reward: frame.reward,
                flags: frame.flags,
            },
            config,
        );
    }

    pub(super) fn mark_actor_return_boundary(&mut self, stream: usize, config: &DreamerConfig) {
        assert!(config.replay_actor_auxiliary_boundaries);
        let history = &mut self.streams[stream];
        let arrival = history
            .frames
            .back_mut()
            .expect("boundary needs an arrival");
        assert!(!arrival.flags.is_first && arrival.previous_action.is_some());
        assert!(!arrival.actor_return_boundary, "duplicate actor boundary");
        arrival.actor_return_boundary = true;
        // Real terminals have already completed their segment; timeouts remain
        // ineligible even if a life loss coincides with the cutoff.
        if !arrival.flags.is_last {
            history.complete_actor_returns(config);
        }
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
        arrivals: &[DeviceReplayArrival],
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
        for (&(stream, previous_action, reward, flags, previous_expert_action), &(_, slot)) in
            arrivals.iter().zip(&slots)
        {
            self.push_stored(
                stream,
                StoredFrame {
                    values: FrameValues::Device {
                        slot,
                        refreshed: None,
                    },
                    previous_action,
                    previous_expert_action,
                    completed_return: None,
                    actor_return_boundary: false,
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
        let complete = config.replay_actor_scale.is_some() && frame.flags.is_terminal;
        if frame.flags.is_first {
            history.next_encoder_phase = 0;
        }
        let phase = history.next_encoder_phase;
        history.next_encoder_phase = (phase + 1) % crate::vision::levjepa::FRAMES;
        history.frames.push_back(frame);
        if complete {
            history.complete_actor_returns(config);
        }

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
        let prefix_length = config.replay_context - 1;
        let length = prefix_length + config.batch_length;
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
        let mut previous_expert_action = vec![vec![None; batch]; length];
        let mut completed_returns = vec![vec![None; batch]; length];
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
            let context = &frames[start];
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
                let index = start + 1 + time;
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
                previous_expert_action[time][row] = frame.previous_expert_action;
                completed_returns[time][row] = frame.completed_return;
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

        let prefix = if prefix_length == 0 {
            None
        } else {
            assert!(pixels.is_none(), "causal video replay requires context1");
            let training_observations = observations.split_off(prefix_length);
            let training_actions = previous_actions.split_off(prefix_length);
            let training_flags = flags.split_off(prefix_length);
            let prefix = ReplayPrefix {
                observations,
                previous_actions,
                flags,
            };
            observations = training_observations;
            previous_actions = training_actions;
            flags = training_flags;
            previous_expert_action = previous_expert_action.split_off(prefix_length);
            completed_returns = completed_returns.split_off(prefix_length);
            rewards = rewards.split_off(prefix_length);
            frame_indices = frame_indices.split_off(prefix_length);
            Some(prefix)
        };

        Some(SequenceBatch {
            initial_deter,
            initial_stoch,
            prefix,
            observations,
            pixels,
            previous_actions,
            previous_expert_action,
            completed_returns,
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
    pub prefix: Option<ReplayPrefix>,
    pub observations: Vec<Vec<f32>>,
    pub pixels: Option<Vec<Vec<f32>>>,
    pub previous_actions: Vec<Vec<f32>>,
    pub previous_expert_action: Vec<Vec<Option<usize>>>,
    /// Return through a known terminal or auxiliary boundary; absent otherwise.
    pub completed_returns: Vec<Vec<Option<f32>>>,
    pub rewards: Vec<Vec<f32>>,
    pub flags: Vec<Vec<FrameFlags>>,
    frame_indices: Vec<Vec<(usize, usize)>>,
}

pub struct ReplayPrefix {
    pub observations: Vec<Vec<f32>>,
    pub previous_actions: Vec<Vec<f32>>,
    pub flags: Vec<Vec<FrameFlags>>,
}

impl SequenceBatch {
    pub fn replay_actor_targets(
        &self,
        config: &DreamerConfig,
        values: &[f32],
        return_scale: f32,
    ) -> (Vec<f32>, usize, usize) {
        if config.replay_actor_scale.is_none() {
            return (Vec::new(), 0, 0);
        }
        let rows = config.batch_size * (config.batch_length - 1);
        assert!(values.len() >= rows && return_scale.is_finite() && return_scale >= 1.0);
        let mut targets = vec![0.0; rows * config.action_count];
        let (mut completed, mut positive) = (0, 0);
        for time in 0..config.batch_length - 1 {
            for lane in 0..config.batch_size {
                if self.flags[time][lane].is_last || self.flags[time + 1][lane].is_first {
                    continue;
                }
                let Some(value) = self.completed_returns[time][lane] else {
                    continue;
                };
                let row = time * config.batch_size + lane;
                assert!(value.is_finite() && values[row].is_finite());
                let weight = (((value as f64 - values[row] as f64) / return_scale as f64)
                    .clamp(0.0, 1.0)) as f32;
                let actions = &self.previous_actions[time + 1]
                    [lane * config.action_count..(lane + 1) * config.action_count];
                assert_eq!(actions.iter().filter(|&&a| a == 1.0).count(), 1);
                assert!(actions.iter().all(|&a| a == 0.0 || a == 1.0));
                completed += 1;
                positive += usize::from(weight > 0.0);
                for (target, action) in targets
                    [row * config.action_count..(row + 1) * config.action_count]
                    .iter_mut()
                    .zip(actions)
                {
                    *target = weight * action;
                }
            }
        }
        (targets, completed, positive)
    }

    /// Prior rollout targets after `start`, retaining a terminal arrival but
    /// never crossing an episode end/reset. Invalid rows stay masked thereafter.
    pub fn open_loop_weights(
        &self,
        start: usize,
        horizon: usize,
        first_row: usize,
        row_count: usize,
    ) -> Vec<Vec<f32>> {
        assert!(start + horizon < self.flags.len());
        let mut valid = vec![1.0; row_count];
        (start + 1..=start + horizon)
            .map(|time| {
                for (local, weight) in valid.iter_mut().enumerate() {
                    let row = first_row + local;
                    if self.flags[time - 1][row].is_last || self.flags[time][row].is_first {
                        *weight = 0.0;
                    }
                }
                valid.clone()
            })
            .collect()
    }

    /// Pair the outgoing expert recommendation stored on frame t+1 with state t.
    /// Last/reset boundaries and the sequence tail never acquire a label.
    pub fn expert_targets(&self, config: &DreamerConfig) -> (Vec<f32>, usize) {
        let Some(expert) = &config.expert_imitation else {
            return (Vec::new(), 0);
        };
        let rows = config.batch_size * (config.batch_length - 1);
        let mut targets = vec![0.0; rows * config.action_count];
        let mut labeled = 0;
        for time in 0..config.batch_length - 1 {
            for lane in 0..config.batch_size {
                if self.flags[time][lane].is_last || self.flags[time + 1][lane].is_first {
                    continue;
                }
                let Some(action) = self.previous_expert_action[time + 1][lane] else {
                    continue;
                };
                assert!(action < config.action_count);
                labeled += 1;
                let row = time * config.batch_size + lane;
                targets[row * config.action_count + action] = expert.action_weights[action];
            }
        }
        (targets, labeled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn completed_returns_use_future_rewards_and_true_terminal_only() {
        let mut config = DreamerConfig::tiny(3);
        config.horizon = 2;
        config.replay_actor_scale = Some(1.0);
        config.extrinsic_reward_scale = 2.0;
        config.intrinsic_reward_scale = 3.0;
        let mut replay = SequenceReplay::with_streams(32, 2);
        for index in 0..4 {
            let mut arrival = frame(index, &config);
            arrival.reward = Reward {
                extrinsic: index as f32,
                intrinsic: 1.0,
            };
            arrival.flags.is_last = index == 3;
            arrival.flags.is_terminal = index == 3;
            replay.push_stream(0, arrival, &config);
            replay.push_stream(1, frame(index, &config), &config);
            if index < 3 {
                assert!(
                    replay.streams[0]
                        .frames
                        .iter()
                        .all(|f| f.completed_return.is_none())
                );
            }
        }
        let values: Vec<_> = replay.streams[0]
            .frames
            .iter()
            .map(|f| f.completed_return)
            .collect();
        assert_eq!(values, [Some(10.75), Some(11.5), Some(9.0), Some(0.0)]);
        assert!(
            replay.streams[1]
                .frames
                .iter()
                .all(|f| f.completed_return.is_none())
        );
    }

    #[test]
    fn auxiliary_boundary_keeps_arrival_pending_and_preserves_prior_segment() {
        let mut config = DreamerConfig::tiny(3);
        config.horizon = 2;
        config.replay_actor_scale = Some(1.0);
        config.replay_actor_auxiliary_boundaries = true;
        let mut replay = SequenceReplay::new(32);
        for index in 0..3 {
            replay.push(frame(index, &config), &config);
        }
        replay.mark_actor_return_boundary(0, &config);
        let values = |r: &SequenceReplay| {
            r.streams[0]
                .frames
                .iter()
                .map(|f| f.completed_return)
                .collect::<Vec<_>>()
        };
        assert_eq!(values(&replay), [Some(2.0), Some(2.0), None]);
        assert!(!replay.streams[0].frames[2].flags.is_last);
        replay.push(frame(3, &config), &config);
        let mut terminal = frame(4, &config);
        terminal.flags.is_last = true;
        terminal.flags.is_terminal = true;
        replay.push(terminal, &config);
        replay.mark_actor_return_boundary(0, &config);
        assert_eq!(
            values(&replay),
            [Some(2.0), Some(2.0), Some(5.0), Some(4.0), Some(0.0)]
        );
        assert_eq!(replay.len(), 5);
        assert_eq!(replay.streams[0].total_frames, 5);
    }

    #[test]
    fn adjacent_actor_cuts_do_not_finalize_timeout_or_cross_reset() {
        let mut config = DreamerConfig::tiny(3);
        config.replay_actor_scale = Some(1.0);
        config.replay_actor_auxiliary_boundaries = true;
        let mut replay = SequenceReplay::new(32);
        replay.push(frame(0, &config), &config);
        for index in 1..3 {
            replay.push(frame(index, &config), &config);
            replay.mark_actor_return_boundary(0, &config);
        }
        let mut timeout = frame(3, &config);
        timeout.flags.is_last = true;
        replay.push(timeout, &config);
        replay.mark_actor_return_boundary(0, &config);
        replay.push(frame(0, &config), &config);
        let mut terminal = frame(9, &config);
        terminal.flags.is_last = true;
        terminal.flags.is_terminal = true;
        replay.push(terminal, &config);
        assert_eq!(
            replay.streams[0]
                .frames
                .iter()
                .map(|f| f.completed_return)
                .collect::<Vec<_>>(),
            [Some(1.0), Some(2.0), None, None, Some(9.0), Some(0.0)]
        );
    }

    #[test]
    fn segmented_replay_matches_forward_returns_without_changing_samples() {
        let mut config = DreamerConfig::tiny(3);
        config.horizon = 7;
        config.replay_context = 3;
        config.replay_actor_scale = Some(1.0);
        config.replay_actor_auxiliary_boundaries = true;
        let mut replay = SequenceReplay::with_streams(30, 2);
        let mut control = SequenceReplay::with_streams(30, 2);
        for index in 0..30 {
            for lane in 0..2 {
                let mut arrival = frame(index, &config);
                arrival.reward.extrinsic = (index * (lane + 1)) as f32;
                arrival.flags.is_last = index == 29;
                arrival.flags.is_terminal = index == 29;
                replay.push_stream(lane, arrival.clone(), &config);
                control.push_stream(lane, arrival, &config);
                if index > 0 && index % (4 + lane) == 0 {
                    replay.mark_actor_return_boundary(lane, &config);
                }
            }
        }
        for lane in 0..2 {
            for (offset, stored) in replay.streams[lane].frames.iter().enumerate() {
                let index = replay.streams[lane].oldest_frame() as usize + offset;
                let end = (index + 1..30).find(|i| i % (4 + lane) == 0).unwrap_or(29);
                let expected = (index + 1..=end)
                    .enumerate()
                    .map(|(offset, i)| {
                        (i * (lane + 1)) as f64
                            * (config.continuation_discount() as f64).powi(offset as i32)
                    })
                    .sum::<f64>() as f32;
                assert_eq!(stored.completed_return, Some(expected));
            }
        }
        let mut rng = StdRng::seed_from_u64(91);
        let mut other_rng = rng.clone();
        for _ in 0..10 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            let other = control.sample(&config, &mut other_rng).unwrap();
            assert_eq!(batch.observations, other.observations);
            assert_eq!(batch.previous_actions, other.previous_actions);
            assert_eq!(batch.flags, other.flags);
            assert_eq!(batch.rewards, other.rewards);
        }
    }

    #[test]
    fn timeouts_and_resets_do_not_acquire_later_episode_returns() {
        let mut config = DreamerConfig::tiny(3);
        config.replay_actor_scale = Some(1.0);
        let mut replay = SequenceReplay::new(32);
        replay.push(frame(0, &config), &config);
        let mut timeout = frame(1, &config);
        timeout.flags.is_last = true;
        replay.push(timeout, &config);
        replay.push(frame(0, &config), &config);
        let mut terminal = frame(2, &config);
        terminal.flags.is_last = true;
        terminal.flags.is_terminal = true;
        replay.push(terminal, &config);
        assert_eq!(
            replay.streams[0]
                .frames
                .iter()
                .map(|f| f.completed_return)
                .collect::<Vec<_>>(),
            [None, None, Some(2.0), Some(0.0)]
        );
        // Even an explicit reset without a preceding last frame is a boundary.
        replay.push(frame(0, &config), &config);
        replay.push(frame(3, &config), &config);
        replay.push(frame(0, &config), &config);
        let mut terminal = frame(4, &config);
        terminal.flags.is_last = true;
        terminal.flags.is_terminal = true;
        replay.push(terminal, &config);
        assert!(replay.streams[0].frames[5].completed_return.is_none());
        assert_eq!(replay.streams[0].frames[6].completed_return, Some(4.0));
    }

    #[test]
    fn evicted_prefix_keeps_exact_returns_for_retained_suffix() {
        let mut config = DreamerConfig::tiny(3);
        config.horizon = 2;
        config.replay_actor_scale = Some(1.0);
        let mut replay = SequenceReplay::new(3);
        for index in 0..7 {
            let mut arrival = frame(index, &config);
            arrival.flags.is_last = index == 6;
            arrival.flags.is_terminal = index == 6;
            replay.push(arrival, &config);
        }
        assert_eq!(replay.streams[0].oldest_frame(), 4);
        assert_eq!(
            replay.streams[0]
                .frames
                .iter()
                .map(|f| f.completed_return)
                .collect::<Vec<_>>(),
            [Some(8.0), Some(6.0), Some(0.0)]
        );
    }

    #[test]
    fn completed_targets_follow_sampled_states_through_prefix_slicing() {
        let mut config = DreamerConfig::tiny(3);
        config.replay_context = 3;
        config.horizon = 2;
        config.replay_actor_scale = Some(1.0);
        let mut legacy = config.clone();
        legacy.replay_actor_scale = None;
        let mut replay = SequenceReplay::new(64);
        let mut control = SequenceReplay::new(64);
        for index in 0..20 {
            let mut arrival = frame(index, &config);
            arrival.flags.is_last = index == 19;
            arrival.flags.is_terminal = index == 19;
            replay.push(arrival.clone(), &config);
            control.push(arrival, &legacy);
        }
        let mut rng = StdRng::seed_from_u64(81);
        let mut control_rng = rng.clone();
        for _ in 0..10 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            let same = control.sample(&legacy, &mut control_rng).unwrap();
            assert_eq!(batch.observations, same.observations);
            assert_eq!(batch.previous_actions, same.previous_actions);
            assert_eq!(batch.flags, same.flags);
            assert_eq!(batch.completed_returns.len(), config.batch_length);
            for t in 0..config.batch_length {
                for row in 0..config.batch_size {
                    let index = batch.observations[t][row * config.observation_dim()] as usize;
                    let expected = (index + 1..20)
                        .enumerate()
                        .map(|(offset, r)| r as f64 * 0.5_f64.powi(offset as i32))
                        .sum::<f64>() as f32;
                    assert_eq!(batch.completed_returns[t][row], Some(expected));
                    assert!(same.completed_returns[t][row].is_none());
                }
            }
        }
    }

    #[test]
    fn replay_actor_uses_outgoing_actions_positive_advantage_and_masks() {
        let mut config = DreamerConfig::tiny(3);
        config.replay_actor_scale = Some(1.0);
        let mut batch = SequenceBatch {
            previous_actions: vec![vec![0.0; 6]; 4],
            flags: vec![vec![FrameFlags::default(); 2]; 4],
            completed_returns: vec![
                vec![Some(3.0), None],
                vec![Some(4.0), Some(2.0)],
                vec![Some(5.0), Some(10.0)],
                vec![None, None],
            ],
            ..Default::default()
        };
        for t in 1..4 {
            batch.previous_actions[t][2] = 1.0;
            batch.previous_actions[t][4] = 1.0;
        }
        batch.flags[1][0].is_last = true;
        batch.flags[3][0].is_first = true;
        let (target, completed, positive) =
            batch.replay_actor_targets(&config, &[1., 0., 0., 4., 0., 1.], 4.);
        assert_eq!((completed, positive), (3, 2));
        let mut expected = vec![0.0; 18];
        expected[2] = 0.5;
        expected[16] = 1.0;
        assert_eq!(target, expected);
        config.replay_actor_scale = None;
        assert_eq!(
            SequenceBatch::default().replay_actor_targets(&config, &[], 1.),
            (vec![], 0, 0)
        );
    }

    #[test]
    fn open_loop_masks_retain_terminal_arrival_without_crossing_boundaries() {
        let mut batch = SequenceBatch {
            flags: vec![vec![FrameFlags::default(); 5]; 7],
            ..Default::default()
        };
        // Start at an offset and slice rows, including a reset first state.
        batch.flags[2][1].is_first = true;
        batch.flags[3][1].is_last = true;
        batch.flags[3][1].is_terminal = true;
        batch.flags[4][2].is_first = true;
        batch.flags[2][3].is_last = true;
        // A time-limit end is also a boundary, even without is_terminal.
        batch.flags[4][4].is_last = true;
        assert_eq!(
            batch.open_loop_weights(2, 3, 1, 4),
            vec![
                vec![1.0, 1.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ]
        );
        let full = batch.open_loop_weights(2, 3, 0, 5);
        let part = batch.open_loop_weights(2, 3, 2, 2);
        for (full, part) in full.iter().zip(part) {
            assert_eq!(full[2..4], part);
        }
    }

    #[test]
    fn expert_targets_align_outgoing_actions_and_exclude_reset_tail_and_untrusted_rows() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 4;
        config.expert_imitation = Some(super::super::ExpertImitation {
            scale: 2.0,
            action_weights: vec![0.05, 1.0, 2.0],
            world_gradient: false,
            decay: None,
        });
        let mut batch = SequenceBatch {
            previous_actions: vec![vec![0.0; 6]; 4],
            previous_expert_action: vec![vec![None; 2]; 4],
            flags: vec![vec![FrameFlags::default(); 2]; 4],
            ..Default::default()
        };
        batch.previous_expert_action[0][0] = Some(0);
        batch.previous_actions[0][0] = 1.0; // Incoming action must not label state0.
        batch.previous_expert_action[1][0] = Some(2);
        batch.previous_actions[1][1] = 1.0; // The teacher can disagree with the executed action.
        batch.previous_actions[1][4] = 1.0; // Ordinary override is not expert.
        batch.flags[1][0].is_last = true;
        batch.flags[2][0].is_first = true;
        batch.previous_expert_action[2][0] = Some(2); // Mask even a malformed reset label.
        batch.previous_expert_action[2][1] = Some(1);
        batch.previous_actions[2][4] = 1.0;
        batch.previous_expert_action[3][0] = Some(0);
        batch.previous_actions[3][0] = 1.0;
        batch.flags[3][0].is_last = true; // The action leading into terminal is valid.
        let executed = batch.previous_actions.clone();
        let (targets, count) = batch.expert_targets(&config);
        assert_eq!(batch.previous_actions, executed);
        assert_eq!(count, 3);
        assert_eq!(
            targets,
            vec![
                0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.05, 0., 0., 0., 0., 0.
            ]
        );
        config.expert_imitation = None;
        assert_eq!(batch.expert_targets(&config), (Vec::new(), 0));
    }

    #[test]
    fn expert_labels_survive_replay_sampling_without_crossing_streams() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 3;
        let mut replay = SequenceReplay::with_streams(16, 2);
        for index in 0..4 {
            for stream in 0..2 {
                replay.push_stream(stream, frame(index, &config), &config);
                replay.streams[stream]
                    .frames
                    .back_mut()
                    .unwrap()
                    .previous_expert_action = (stream == 1 && index == 2).then_some(2);
            }
        }
        let mut rng = StdRng::seed_from_u64(1);
        let mut seen = [false; 2];
        for _ in 0..16 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            for time in 0..3 {
                for row in 0..2 {
                    let (stream, index) = batch.frame_indices[time][row];
                    seen[stream] = true;
                    assert_eq!(
                        batch.previous_expert_action[time][row],
                        (stream == 1 && index == 2).then_some(2)
                    );
                }
            }
        }
        assert_eq!(seen, [true, true]);
    }

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
                        previous_expert_action: None,
                        completed_return: None,
                        actor_return_boundary: false,
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
        let values = vec![index as f32; config.observation_dim()];
        let observation = if config.rgb_frame_pair {
            Observation::rgb_pair(values, config.observation_grid())
        } else if config.observation_kind.is_rgb() {
            Observation::rgb(values, config.observation_grid())
        } else {
            Observation::from_vec(values)
        };
        ReplayFrame {
            observation,
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
    fn replay_fixture_supports_feature_and_rgb_geometry() {
        use super::super::ObservationKind;
        let mut config = DreamerConfig::tiny(3);
        for kind in [
            ObservationKind::Features,
            ObservationKind::Rgb32,
            ObservationKind::Rgb48,
            ObservationKind::Rgb64,
        ] {
            config.observation_kind = kind;
            frame(7, &config).validate(&config);
            if kind.is_rgb() {
                config.rgb_frame_pair = true;
                frame(7, &config).validate(&config);
                config.rgb_frame_pair = false;
            }
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
        assert!(batch.prefix.is_none());
    }

    #[test]
    fn prefix_preserves_boundaries_and_refreshes_only_training_rows() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 16;
        config.batch_length = 16;
        config.replay_context = 9;
        config.intrinsic_reward_scale = 0.0;
        config.expert_imitation = Some(super::super::ExpertImitation {
            scale: 1.0,
            action_weights: vec![1.0; 3],
            world_gradient: false,
            decay: None,
        });
        let mut replay = SequenceReplay::new(32);
        for index in 0..25 {
            let mut arrival = frame(index, &config);
            if index == 4 {
                arrival.flags.is_last = true;
                arrival.flags.is_terminal = true;
            } else if index == 5 {
                arrival.flags.is_first = true;
                arrival.previous_action = None;
            }
            replay.push(arrival, &config);
            replay.streams[0]
                .frames
                .back_mut()
                .unwrap()
                .previous_expert_action = Some(index % 3);
        }
        assert_eq!(replay.valid_sequence_count(&config), 1);
        let batch = replay
            .sample(&config, &mut StdRng::seed_from_u64(13))
            .unwrap();
        let prefix = batch.prefix.as_ref().unwrap();
        assert_eq!(prefix.observations.len(), 8);
        assert!(batch.initial_deter.iter().all(|&value| value == 0.0));
        for time in 0..8 {
            assert!(
                prefix.observations[time]
                    .iter()
                    .all(|&value| value == (time + 1) as f32)
            );
            for row in 0..16 {
                let action = &prefix.previous_actions[time][row * 3..(row + 1) * 3];
                assert_eq!(action.iter().sum::<f32>(), f32::from(time != 4));
                assert_eq!(prefix.flags[time][row].is_first, time == 4);
                assert_eq!(prefix.flags[time][row].is_terminal, time == 3);
            }
        }
        for time in 0..16 {
            assert!(
                batch.observations[time]
                    .iter()
                    .all(|&value| value == (time + 9) as f32)
            );
            assert_eq!(batch.rewards[time], vec![(time + 9) as f32; 16]);
            assert_eq!(batch.frame_indices[time], vec![(0, time + 9); 16]);
            assert_eq!(
                batch.previous_expert_action[time],
                vec![Some((time + 9) % 3); 16]
            );
        }
        let (targets, count) = batch.expert_targets(&config);
        assert_eq!(count, 240);
        assert_eq!(targets.len(), 240 * 3);
        for time in 0..15 {
            for row in 0..16 {
                let target = &targets[(time * 16 + row) * 3..(time * 16 + row + 1) * 3];
                assert_eq!(target.iter().sum::<f32>(), 1.0);
                assert_eq!(target[(time + 10) % 3], 1.0);
            }
        }
        let size = config.network();
        replay.update_context(
            &batch,
            &vec![vec![123.0; 16 * size.deter]; 16],
            &vec![vec![456.0; 16 * size.stoch * size.classes]; 16],
            &config,
        );
        for (index, stored) in replay.streams[0].frames.iter().enumerate() {
            let FrameValues::Host { deter, stoch, .. } = &stored.values else {
                panic!("host fixture")
            };
            assert!(
                deter
                    .iter()
                    .all(|&value| value == if index < 9 { index as f32 } else { 123.0 })
            );
            assert!(
                stoch
                    .iter()
                    .all(|&value| value == if index < 9 { index as f32 } else { 456.0 })
            );
        }
    }

    #[test]
    fn replay_prefix_stays_with_its_stream_after_eviction() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 8;
        config.batch_length = 3;
        config.replay_context = 3;
        let mut replay = SequenceReplay::with_streams(25, 3);
        for tick in 0..40 {
            for stream in 0..3 {
                if stream == 2 && tick % 2 == 0 {
                    continue;
                }
                let index = replay.streams[stream].total_frames as usize;
                replay.push_stream(stream, frame(stream * 100 + index, &config), &config);
            }
        }
        let mut rng = StdRng::seed_from_u64(19);
        for _ in 0..20 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            let prefix = batch.prefix.as_ref().unwrap();
            for row in 0..8 {
                let (stream, index) = batch.frame_indices[0][row];
                assert!(index >= 3);
                let first = batch.observations[0][row * config.observation_dim()];
                assert_eq!(first as usize / 100, stream);
                assert_eq!(
                    batch.initial_deter[row * config.network().deter],
                    first - 3.0
                );
                for time in 0..2 {
                    assert_eq!(
                        prefix.observations[time][row * config.observation_dim()],
                        first - 2.0 + time as f32
                    );
                }
                for time in 0..3 {
                    assert_eq!(batch.frame_indices[time][row], (stream, index + time));
                    assert_eq!(
                        batch.observations[time][row * config.observation_dim()],
                        first + time as f32
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires separately guarded GPU; device prefix readback, eviction and refresh match stored host rows"]
    fn device_replay_prefix_matches_host_storage() {
        use std::sync::Arc;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            let info = gpu.device_information();
            assert_eq!(info.device_name, expected);
            assert!(!info.is_software_emulated);
            let memory = gpu.memory_stats();
            assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        }
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = super::super::ObservationKind::Rgb32;
        config.batch_length = 16;
        config.world_backprop_length = 16;
        config.replay_context = 9;
        config.replay_capacity = 64;
        let size = config.network();
        let latent = size.stoch * size.classes;
        let mut graph = meganeura::Graph::new();
        let observations = graph.input("observation", &[2, config.observation_dim()]);
        let deters = graph.input("previous_deter", &[2, size.deter]);
        let stochs = graph.input("previous_stoch", &[2, latent]);
        graph.set_outputs(vec![observations, deters, stochs]);
        let mut session =
            super::super::runtime::build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut host = SequenceReplay::with_streams(64, 2);
        let mut device = SequenceReplay::with_streams(64, 2);
        device.enable_device(Arc::clone(&gpu), &config);
        for time in 0..40 {
            let mut observations = Vec::new();
            let mut deters = Vec::new();
            let mut stochs = Vec::new();
            let mut arrivals = Vec::new();
            for stream in 0..2 {
                let mut arrival = frame(stream * 100 + time, &config);
                if time == 0 || (stream == 1 && time == 14) {
                    arrival.flags.is_first = true;
                    arrival.previous_action = None;
                }
                let label = Some((time + stream) % 3);
                arrivals.push((
                    stream,
                    arrival.previous_action,
                    arrival.reward,
                    arrival.flags,
                    label,
                ));
                observations.extend_from_slice(arrival.observation.as_slice());
                deters.extend_from_slice(&arrival.deter);
                stochs.extend_from_slice(&arrival.stoch);
                host.push_stream(stream, arrival, &config);
                host.streams[stream]
                    .frames
                    .back_mut()
                    .unwrap()
                    .previous_expert_action = label;
            }
            session.set_input("observation", &observations);
            session.set_input("previous_deter", &deters);
            session.set_input("previous_stoch", &stochs);
            device.push_device(&session, &arrivals, &config);
            device.wait_device();
        }
        let mut host_rng = StdRng::seed_from_u64(73);
        let mut device_rng = host_rng.clone();
        let mut resets = 0;
        for _ in 0..16 {
            let left = host.sample(&config, &mut host_rng).unwrap();
            let right = device.sample(&config, &mut device_rng).unwrap();
            assert_eq!(left.initial_deter, right.initial_deter);
            assert_eq!(left.initial_stoch, right.initial_stoch);
            assert_eq!(left.observations, right.observations);
            assert_eq!(left.previous_actions, right.previous_actions);
            assert_eq!(left.previous_expert_action, right.previous_expert_action);
            assert_eq!(left.rewards, right.rewards);
            assert_eq!(left.flags, right.flags);
            assert_eq!(left.frame_indices, right.frame_indices);
            let (lp, rp) = (
                left.prefix.as_ref().unwrap(),
                right.prefix.as_ref().unwrap(),
            );
            assert_eq!(lp.observations, rp.observations);
            assert_eq!(lp.previous_actions, rp.previous_actions);
            assert_eq!(lp.flags, rp.flags);
            resets += lp
                .flags
                .iter()
                .flatten()
                .filter(|flag| flag.is_first)
                .count();
            let deters = vec![vec![123.0; 2 * size.deter]; 16];
            let stochs = vec![vec![456.0; 2 * latent]; 16];
            host.update_context(&left, &deters, &stochs, &config);
            device.update_context(&right, &deters, &stochs, &config);
        }
        assert!(resets > 0);
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
    fn replay_prefix_fresh_starts_skip_whole_prefix_and_training_windows() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 3;
        config.replay_context = 9;
        let mut replay = SequenceReplay::new(32);
        for index in 0..25 {
            replay.push(frame(index, &config), &config);
        }
        assert_eq!(replay.valid_sequence_count(&config), 14);
        let batch = replay
            .sample(&config, &mut StdRng::seed_from_u64(1))
            .unwrap();
        assert_eq!(batch.initial_deter[0], 1.0);
        assert_eq!(batch.initial_deter[config.network().deter], 13.0);
        assert_eq!(batch.frame_indices[0], vec![(0, 10), (0, 22)]);
        assert!(replay.fresh_starts.is_empty());
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
