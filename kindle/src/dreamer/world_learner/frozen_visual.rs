use super::PixelLoss;
use meganeura::Session;

const FROZEN_PREFIXES: [&str; 3] = [
    "world.representation.encoder.",
    "world.representation.posterior.",
    "world.decoder.",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TrainingScope {
    All,
    FrozenVisual { start_updates: u64 },
    FixedTeacher { start_updates: u64 },
}

impl TrainingScope {
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::FrozenVisual { .. } => "frozen_visual",
            Self::FixedTeacher { .. } => "fixed_teacher",
        }
    }

    pub fn format(self) -> &'static str {
        match self {
            Self::All => "kindle-world-only-v2",
            Self::FrozenVisual { .. } => "kindle-world-only-v3",
            Self::FixedTeacher { .. } => "kindle-world-only-v6",
        }
    }

    pub fn start_updates(self) -> Option<u64> {
        match self {
            Self::All => None,
            Self::FrozenVisual { start_updates } | Self::FixedTeacher { start_updates } => {
                Some(start_updates)
            }
        }
    }

    pub fn apply(self, session: &mut Session) {
        if matches!(self, Self::FrozenVisual { .. } | Self::FixedTeacher { .. }) {
            for prefix in FROZEN_PREFIXES {
                session.set_lr_multiplier(prefix, 0.0);
            }
        }
    }

    pub fn start_frozen_visual(
        self,
        pixel_loss: PixelLoss,
        prediction_length: usize,
        updates: u64,
    ) -> Result<Self, &'static str> {
        if self != Self::All
            || pixel_loss != PixelLoss::MotionCrossEntropy
            || prediction_length != 0
            || updates == 0
        {
            return Err("frozen visual phase requires an experienced all-weight H0 motion world");
        }
        Ok(Self::FrozenVisual {
            start_updates: updates,
        })
    }

    pub fn from_metadata(
        metadata: &serde_json::Value,
        pixel_loss: PixelLoss,
        prediction_length: usize,
        updates: u64,
    ) -> Result<Self, &'static str> {
        match metadata["format"].as_str() {
            Some(
                "kindle-world-only-v1"
                | "kindle-world-only-v2"
                | "kindle-world-only-v4"
                | "kindle-world-only-v5"
                | "kindle-world-only-v8"
                | "kindle-world-only-v9"
                | "kindle-world-only-v10",
            ) => {
                if metadata.get("training_scope").is_some()
                    || metadata.get("phase_start_updates").is_some()
                {
                    return Err("world training scope requires checkpoint format v3");
                }
                Ok(Self::All)
            }
            Some("kindle-world-only-v3") => {
                let start = metadata["phase_start_updates"]
                    .as_u64()
                    .filter(|&value| value > 0 && value <= updates)
                    .ok_or("invalid frozen visual phase start")?;
                if metadata["training_scope"].as_str() != Some("frozen_visual")
                    || pixel_loss != PixelLoss::MotionCrossEntropy
                    || prediction_length != 16
                {
                    return Err("invalid frozen visual checkpoint scope or objective");
                }
                Ok(Self::FrozenVisual {
                    start_updates: start,
                })
            }
            Some("kindle-world-only-v6" | "kindle-world-only-v7") => {
                let start = metadata["phase_start_updates"]
                    .as_u64()
                    .filter(|&value| value > 0 && value <= updates)
                    .ok_or("invalid fixed-teacher phase start")?;
                if metadata["training_scope"] != "fixed_teacher"
                    || !pixel_loss.fixed_teacher()
                    || prediction_length != 16
                    || metadata["teacher_sha256"]
                        .as_str()
                        .is_none_or(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
                {
                    return Err("invalid fixed-teacher checkpoint scope or objective");
                }
                Ok(Self::FixedTeacher {
                    start_updates: start,
                })
            }
            _ => Err("not a supported world-only checkpoint"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_only_accepts_experienced_h0_motion_world() {
        let phase = TrainingScope::All
            .start_frozen_visual(PixelLoss::MotionCrossEntropy, 0, 8192)
            .unwrap();
        assert_eq!(phase.start_updates(), Some(8192));
        for (scope, loss, horizon, age) in [
            (TrainingScope::All, PixelLoss::CrossEntropy, 0, 8192),
            (TrainingScope::All, PixelLoss::MotionCrossEntropy, 16, 8192),
            (TrainingScope::All, PixelLoss::MotionCrossEntropy, 0, 0),
            (phase, PixelLoss::MotionCrossEntropy, 0, 8192),
        ] {
            assert!(scope.start_frozen_visual(loss, horizon, age).is_err());
        }
    }

    #[test]
    fn legacy_checkpoints_keep_all_weights_and_reject_disguised_phase_scope() {
        for format in ["kindle-world-only-v1", "kindle-world-only-v2"] {
            let mut metadata = serde_json::json!({"format": format});
            assert_eq!(
                TrainingScope::from_metadata(&metadata, PixelLoss::SquaredError, 0, 7).unwrap(),
                TrainingScope::All
            );
            metadata["training_scope"] = serde_json::json!("frozen_visual");
            assert!(
                TrainingScope::from_metadata(&metadata, PixelLoss::MotionCrossEntropy, 16, 7)
                    .is_err()
            );
        }
        assert_eq!(TrainingScope::All.format(), "kindle-world-only-v2");
        assert_eq!(TrainingScope::All.start_updates(), None);
    }

    #[test]
    fn phase_schema_preserves_start_and_rejects_missing_or_inconsistent_state() {
        let original = serde_json::json!({"format":"kindle-world-only-v3", "training_scope":"frozen_visual", "phase_start_updates":8192});
        let scope =
            TrainingScope::from_metadata(&original, PixelLoss::MotionCrossEntropy, 16, 12288)
                .unwrap();
        assert_eq!(scope.format(), "kindle-world-only-v3");
        assert_eq!(scope.start_updates(), Some(8192));
        for start in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(12289),
            serde_json::json!(1.5),
        ] {
            let mut metadata = original.clone();
            metadata["phase_start_updates"] = start;
            assert!(
                TrainingScope::from_metadata(&metadata, PixelLoss::MotionCrossEntropy, 16, 12288)
                    .is_err()
            );
        }
        for scope in [
            serde_json::Value::Null,
            serde_json::json!("all"),
            serde_json::json!("unknown"),
        ] {
            let mut metadata = original.clone();
            metadata["training_scope"] = scope;
            assert!(
                TrainingScope::from_metadata(&metadata, PixelLoss::MotionCrossEntropy, 16, 12288)
                    .is_err()
            );
        }
        assert!(
            TrainingScope::from_metadata(&original, PixelLoss::SquaredError, 16, 12288).is_err()
        );
        assert!(
            TrainingScope::from_metadata(&original, PixelLoss::MotionCrossEntropy, 0, 12288)
                .is_err()
        );
    }

    #[test]
    fn frozen_prefixes_cover_visual_weights_without_dynamics_or_head_weights() {
        let frozen = |name: &str| {
            FROZEN_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
        };
        for name in [
            "world.representation.encoder.cnn0.weight",
            "world.representation.posterior.layer0.weight",
            "world.decoder.project.weight",
        ] {
            assert!(frozen(name));
        }
        for name in [
            "world.dynamics.gru.weight",
            "world.dynamics.prior.weight",
            "world.reward.out.weight",
            "world.continuation.out.weight",
        ] {
            assert!(!frozen(name));
        }
    }

    #[test]
    fn scope_partitions_the_actual_h16_world_parameters() {
        use crate::dreamer::{DreamerConfig, ObservationKind, world};
        use meganeura::graph::Op;
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        config.batch_length = 32;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        config.loss_scales.future_prediction = 0.0;
        config.disagreement_bonus = false;
        config.actor_critic_gradient = false;
        let graph = world::build_world_only_pixel_training_graph(
            &config,
            16,
            PixelLoss::MotionCrossEntropy,
        );
        let names = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                Op::Parameter { name } => Some(name),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(names.iter().all(|name| name.starts_with("world.")));
        for prefix in FROZEN_PREFIXES {
            assert!(names.iter().any(|name| name.starts_with(prefix)));
        }
        for prefix in ["world.dynamics.", "world.reward.", "world.continuation."] {
            assert!(names.iter().any(|name| name.starts_with(prefix)));
            assert!(
                !FROZEN_PREFIXES
                    .iter()
                    .any(|frozen| prefix.starts_with(frozen))
            );
        }
    }
}
