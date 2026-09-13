//! DreamerV3 learner with frozen visual perception and causal feature prediction.
//!
//! The implementation follows the pinned upstream DreamerV3 algorithm for
//! replay alignment, categorical RSSM state, KL balancing, distributional
//! reward/value heads, and imagined actor-critic learning. The deliberate
//! perception exception lives in [`crate::vision`].

mod agent;
mod behavior;
mod checkpoint;
mod config;
mod cpu;
mod device_copy;
mod distributions;
mod intrinsic;
mod networks;
mod readback;
mod replay;
mod runtime;
mod world;

pub use agent::{
    ActionMode, BehaviorMetrics, DreamerAgent, DreamerCore, LearnReport, LearnTiming,
    ModelProvenance, VectorDreamerAgent, WorldMetrics,
};
pub use config::{DreamerConfig, LossScales, ModelSize, NetworkSize};
pub use replay::{FrameFlags, Reward};

/// Upstream DreamerV3 revision used as the behavioral contract.
pub const DREAMERV3_UPSTREAM_REV: &str = "e3f02248693a79dc8b0ebd62c93683888ddaccfe";
/// Meganeura revision used to compile and optimize the baseline graphs.
pub const MEGANEURA_REV: &str = "75dfe901deb87ca0054c438437efd3aa388b7188";
/// Exact Blade revision providing the shared graphics runtime.
pub const BLADE_REV: &str = "f6f2729e850cc0aefdc0bb18523da58a72765169";

#[cfg(test)]
mod tests {
    use super::{BLADE_REV, MEGANEURA_REV};

    #[test]
    fn reported_backend_revisions_match_every_dependency_lock() {
        const KINDLE_MANIFEST: &str = include_str!("../../Cargo.toml");
        const WORKSPACE_LOCK: &str = include_str!("../../../Cargo.lock");
        const PYTHON_LOCK: &str = include_str!("../../../python/Cargo.lock");

        let manifest_pin =
            format!("git = \"https://github.com/kvark/meganeura\", rev = \"{MEGANEURA_REV}\"");
        assert!(KINDLE_MANIFEST.contains(&manifest_pin));
        let blade_pin = format!("git = \"https://github.com/kvark/blade\", rev = \"{BLADE_REV}\"");
        assert!(KINDLE_MANIFEST.contains(&blade_pin));

        for (name, lock) in [("workspace", WORKSPACE_LOCK), ("Python", PYTHON_LOCK)] {
            let packages = |wanted: &str| {
                lock.split("[[package]]")
                    .filter(|package| package.lines().any(|line| line == wanted))
                    .collect::<Vec<_>>()
            };
            let meganeura = packages("name = \"meganeura\"");
            assert_eq!(
                meganeura.len(),
                1,
                "{name} has conflicting Meganeura sources"
            );
            assert!(meganeura[0].contains(&format!(
                "git+https://github.com/kvark/meganeura?rev={MEGANEURA_REV}#{MEGANEURA_REV}"
            )));
            let blade = packages("name = \"blade-graphics\"");
            assert_eq!(blade.len(), 1, "{name} has conflicting GPU context types");
            assert!(
                blade[0].contains(&format!(
                    "git+https://github.com/kvark/blade?rev={BLADE_REV}#{BLADE_REV}"
                )),
                "{name} mismatches {BLADE_REV}"
            );
        }
    }
}
