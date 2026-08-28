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
pub const MEGANEURA_REV: &str = "5429fcd043acd58dec77f35a44c452737d892fd2";
/// Blade revision providing the shared graphics runtime.
pub const BLADE_REV: &str = "824d55fbd6485a102ee220553bc54a30c1af5936";

#[cfg(test)]
mod tests {
    use super::{BLADE_REV, MEGANEURA_REV};

    #[test]
    fn reported_backend_revisions_match_every_dependency_lock() {
        const KINDLE_MANIFEST: &str = include_str!("../../Cargo.toml");
        const WORKSPACE_LOCK: &str = include_str!("../../../Cargo.lock");
        const PYTHON_LOCK: &str = include_str!("../../../python/Cargo.lock");

        for (repository, revision) in [("meganeura", MEGANEURA_REV), ("blade", BLADE_REV)] {
            let manifest_pin =
                format!("git = \"https://github.com/kvark/{repository}\", rev = \"{revision}\"");
            assert!(
                KINDLE_MANIFEST.contains(&manifest_pin),
                "Kindle manifest does not pin reported {repository} revision {revision}"
            );

            let lock_pin =
                format!("git+https://github.com/kvark/{repository}?rev={revision}#{revision}");
            for (name, lock) in [("workspace", WORKSPACE_LOCK), ("Python", PYTHON_LOCK)] {
                assert!(
                    lock.contains(&lock_pin),
                    "{name} lock does not resolve reported {repository} revision {revision}"
                );
            }
        }
    }
}
