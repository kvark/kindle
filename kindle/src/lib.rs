#![warn(trivial_numeric_casts, unused_extern_crates)]

//! Kindle's DreamerV3 baseline with frozen DINOv3 perception.
//!
//! The first implementation deliberately exposes a small surface: visual
//! environment types, a discrete-action Dreamer agent, its configuration and
//! metrics, and the frozen DINO encoder used at the perception boundary.

pub mod dreamer;
pub mod env;
pub mod vision;

pub use dreamer::{
    ActionMode, BehaviorMetrics, DreamerAgent, DreamerConfig, DreamerCore, FrameFlags, LearnReport,
    LossScales, ModelSize, Reward, WorldMetrics,
};
pub use env::{Environment, RgbFrame, Transition};
