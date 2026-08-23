//! Small pixel-environment boundary shared by native and Python runners.

use crate::dreamer::{FrameFlags, Reward};

/// Interleaved row-major RGB8 frame. Perception handles resizing, so game
/// adapters should preserve their native aspect ratio here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RgbFrame {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbFrame {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert!(width > 0 && height > 0);
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .expect("RGB frame dimensions overflow usize");
        assert_eq!(pixels.len(), expected);
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Result of one environment action.
#[derive(Clone, Debug)]
pub struct Transition {
    pub frame: RgbFrame,
    pub reward: Reward,
    /// Natural task terminal; value targets must not bootstrap through it.
    pub terminated: bool,
    /// Time/resource limit; closes the replay trace but may bootstrap.
    pub truncated: bool,
}

impl Transition {
    pub fn flags(&self) -> FrameFlags {
        FrameFlags {
            is_first: false,
            is_last: self.terminated || self.truncated,
            is_terminal: self.terminated,
        }
    }
}

/// Minimal contract for a discrete-action visual environment.
///
/// D3's first Kindle baseline intentionally fixes the action family at
/// construction. Environment-specific controls belong in adapters outside
/// the model; they expose a categorical vocabulary and optional validity
/// mask through this trait.
pub trait Environment {
    fn action_count(&self) -> usize;

    /// Reset the environment and return its first frame.
    fn reset(&mut self) -> RgbFrame;

    /// Optional mask over the current categorical action vocabulary.
    fn action_mask(&self) -> Option<Vec<bool>> {
        None
    }

    fn step(&mut self, action: usize) -> Transition;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_limit_is_last_but_not_terminal() {
        let transition = Transition {
            frame: RgbFrame::new(1, 1, vec![0, 0, 0]),
            reward: Reward::default(),
            terminated: false,
            truncated: true,
        };
        assert_eq!(
            transition.flags(),
            FrameFlags {
                is_first: false,
                is_last: true,
                is_terminal: false,
            }
        );
    }
}
