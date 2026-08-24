//! Uniform online sequence replay with D3-aligned frame semantics.

use std::collections::VecDeque;

use rand::Rng;

use super::config::DreamerConfig;
use crate::vision::DinoObservation;

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
    pub observation: DinoObservation,
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
    frames: VecDeque<ReplayFrame>,
}

impl SequenceReplay {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 1);
        Self {
            capacity,
            frames: VecDeque::with_capacity(capacity.min(65_536)),
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Number of complete context-plus-training sequences currently eligible
    /// for uniform sampling. Upstream D3 gates scheduled training on at least
    /// `batch_size * batch_length` such replay items.
    pub fn valid_sequence_count(&self, config: &DreamerConfig) -> usize {
        let required = config.replay_context + config.batch_length;
        self.frames.len().saturating_sub(required - 1)
    }

    pub fn push(&mut self, frame: ReplayFrame, config: &DreamerConfig) {
        frame.validate(config);
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn sample(&self, config: &DreamerConfig, rng: &mut impl Rng) -> Option<SequenceBatch> {
        let required = config.replay_context + config.batch_length;
        if self.frames.len() < required {
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
        let mut previous_actions = (0..length)
            .map(|_| vec![0.0; batch * config.action_count])
            .collect::<Vec<_>>();
        let mut rewards = (0..length).map(|_| vec![0.0; batch]).collect::<Vec<_>>();
        let mut flags = (0..length)
            .map(|_| vec![FrameFlags::default(); batch])
            .collect::<Vec<_>>();
        let mut frame_indices = (0..length).map(|_| vec![0; batch]).collect::<Vec<_>>();

        let last_start = self.frames.len() - required;
        for row in 0..batch {
            let start = rng.random_range(0..=last_start);
            let context = &self.frames[start + config.replay_context - 1];
            initial_deter[row * size.deter..(row + 1) * size.deter].copy_from_slice(&context.deter);
            let stoch_width = size.stoch * size.classes;
            initial_stoch[row * stoch_width..(row + 1) * stoch_width]
                .copy_from_slice(&context.stoch);

            for time in 0..length {
                let index = start + config.replay_context + time;
                let frame = &self.frames[index];
                observations[time]
                    [row * config.observation_dim()..(row + 1) * config.observation_dim()]
                    .copy_from_slice(frame.observation.as_slice());
                if let Some(action) = frame.previous_action {
                    previous_actions[time][row * config.action_count + action] = 1.0;
                }
                rewards[time][row] = frame.reward.combined(config);
                flags[time][row] = frame.flags;
                frame_indices[time][row] = index;
            }
        }

        Some(SequenceBatch {
            initial_deter,
            initial_stoch,
            observations,
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
                let frame = &mut self.frames[batch.frame_indices[time][row]];
                frame
                    .deter
                    .copy_from_slice(&deter[time][row * size.deter..(row + 1) * size.deter]);
                let width = size.stoch * size.classes;
                frame
                    .stoch
                    .copy_from_slice(&stoch[time][row * width..(row + 1) * width]);
            }
        }
    }
}

pub struct SequenceBatch {
    pub initial_deter: Vec<f32>,
    pub initial_stoch: Vec<f32>,
    pub observations: Vec<Vec<f32>>,
    pub previous_actions: Vec<Vec<f32>>,
    pub rewards: Vec<Vec<f32>>,
    pub flags: Vec<Vec<FrameFlags>>,
    frame_indices: Vec<Vec<usize>>,
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

    fn frame(index: usize, config: &DreamerConfig) -> ReplayFrame {
        let size = config.network();
        ReplayFrame {
            observation: DinoObservation::from_vec(vec![index as f32; config.observation_dim()]),
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
    fn terminal_implies_last() {
        let config = DreamerConfig::tiny(2);
        let mut invalid = frame(1, &config);
        invalid.flags.is_terminal = true;
        let result = std::panic::catch_unwind(|| invalid.validate(&config));
        assert!(result.is_err());
    }
}
