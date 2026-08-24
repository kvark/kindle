//! DreamerV3 baseline with frozen DINOv3 perception.
//!
//! The implementation follows the pinned upstream DreamerV3 algorithm for
//! replay alignment, categorical RSSM state, KL balancing, distributional
//! reward/value heads, and imagined actor-critic learning. The deliberate
//! perception exception lives in [`crate::vision`].

mod agent;
mod behavior;
mod config;
mod distributions;
mod networks;
mod replay;
mod runtime;
mod world;

pub use agent::{
    ActionMode, BehaviorMetrics, DreamerAgent, DreamerCore, LearnReport, WorldMetrics,
};
pub use config::{DreamerConfig, LossScales, ModelSize, NetworkSize};
pub use replay::{FrameFlags, Reward};

/// Upstream DreamerV3 revision used as the behavioral contract.
pub const DREAMERV3_UPSTREAM_REV: &str = "e3f02248693a79dc8b0ebd62c93683888ddaccfe";
/// Meganeura revision used to compile and optimize the baseline graphs.
pub const MEGANEURA_REV: &str = "3f2b91d95288d625a7616179604c33bba6472aaf";
/// Blade revision providing the shared graphics runtime.
pub const BLADE_REV: &str = "95f5004fb02785a792c883a5312ca5ac37872a75";
