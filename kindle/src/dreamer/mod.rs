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
pub const MEGANEURA_REV: &str = "d904e12e52af6910b041873cd203a5d5e5fd3b3c";
/// Blade revision providing the shared graphics runtime.
pub const BLADE_REV: &str = "824d55fbd6485a102ee220553bc54a30c1af5936";
