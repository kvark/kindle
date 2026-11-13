use super::{DreamerConfig, PixelLoss};
use serde_json::Value;

type Error = Box<dyn std::error::Error>;
const FORMAT: &str = "kindle-world-only-v25";

pub(super) fn validate(
    config: &DreamerConfig,
    prediction_length: usize,
    pixel_loss: PixelLoss,
) -> Result<(), Error> {
    if config.world_action_features.is_none() {
        return Ok(());
    }
    config.check().map_err(std::io::Error::other)?;
    let plain = pixel_loss == PixelLoss::MotionCrossEntropy
        && !config.rgb_spatial_decoder
        && !config.rgb_spatial_fusion
        && !config.rgb_spatial_action;
    let local = pixel_loss == PixelLoss::MotionEventRollout
        && config.rgb_spatial_decoder
        && config.rgb_spatial_fusion
        && config.rgb_spatial_action;
    if prediction_length == 0
        || !(plain || local)
        || config.world_mean_latents
        || config.world_gaussian
        || config.world_gaussian_dim.is_some()
        || config.rgb_previous_frame
        || config.rgb_latent_only
        || config.world_image_feedback
        || config.rgb_spatial_transport
        || config.rgb_spatial_residual
        || config.rgb_color_attention.is_some()
        || config.color_centroid_supervision.is_some()
    {
        return Err("action features require a plain or local-action categorical RGB suffix without auxiliary heads".into());
    }
    Ok(())
}

pub(super) fn base_metadata(metadata: &Value) -> Result<Option<Value>, Error> {
    let enabled = !metadata["config"]["world_action_features"].is_null();
    if enabled != (metadata["format"] == FORMAT) {
        return Err("world action features require checkpoint format v25".into());
    }
    if !enabled {
        if metadata.get("action_feature_base_format").is_some() {
            return Err("action feature base format requires action features".into());
        }
        return Ok(None);
    }
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing action feature suffix length")?;
    let pixel_loss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, pixel_loss)?;
    let expected = if config.rgb_spatial_action {
        "kindle-world-only-v18"
    } else {
        "kindle-world-only-v2"
    };
    if metadata["action_feature_base_format"] != expected
        || metadata.get("training_scope").is_some()
        || metadata.get("phase_start_updates").is_some()
    {
        return Err("invalid action feature base format or training scope".into());
    }
    let mut base = metadata.clone();
    base["format"] = expected.into();
    base.as_object_mut()
        .unwrap()
        .remove("action_feature_base_format");
    base["config"]
        .as_object_mut()
        .unwrap()
        .remove("world_action_features");
    Ok(Some(base))
}

pub(super) fn wrap_metadata(metadata: &mut Value) -> Result<(), Error> {
    let base = metadata["format"].clone();
    if base != "kindle-world-only-v2" && base != "kindle-world-only-v18" {
        return Err("unsupported action feature checkpoint composition".into());
    }
    metadata["action_feature_base_format"] = base;
    metadata["format"] = FORMAT.into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dreamer::{ObservationKind, agent::DreamerCore, world_learner};
    use serde_json::json;

    const CANONICAL: [i32; 18] = [0, 1, 0, 3, 4, 0, 3, 4, 3, 4, 1, 11, 12, 1, 11, 12, 11, 12];

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(18);
        config.observation_kind = ObservationKind::Rgb32;
        config.batch_size = 2;
        config.batch_length = 4;
        config.world_backprop_length = 4;
        config.world_microbatch_size = Some(2);
        config.disagreement_bonus = false;
        config.loss_scales.policy = 0.;
        config.loss_scales.value = 0.;
        config.loss_scales.replay_value = 0.;
        config.world_action_features = Some(
            CANONICAL
                .iter()
                .map(|&a| {
                    let mut row = vec![0.; 18];
                    row[0] = f32::from(a == 4 || a == 12);
                    row[1] = f32::from(a == 3 || a == 11);
                    row[2] = f32::from(a == 1 || a == 11 || a == 12);
                    row
                })
                .collect(),
        );
        config
    }

    fn metadata(local: bool) -> Value {
        let mut config = config();
        config.rgb_spatial_decoder = local;
        config.rgb_spatial_fusion = local;
        config.rgb_spatial_action = local;
        let mut value = json!({
            "format": if local { "kindle-world-only-v18" } else { "kindle-world-only-v2" },
            "config": config,
            "pixel_loss": if local { "motion_event_rollout" } else { "motion_cross_entropy" },
            "prediction_length": 2,
            "updates": 1,
        });
        wrap_metadata(&mut value).unwrap();
        value
    }

    #[test]
    fn action_aliases_share_features_and_preserve_reset_masks() {
        let config = config();
        config.check().unwrap();
        let actions = [
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        ];
        let canonical = actions.map(|a| if a < 0 { a } else { CANONICAL[a as usize] });
        let feeds = world_learner::action_inputs(&config, &actions);
        assert_eq!(feeds, world_learner::action_inputs(&config, &canonical));
        assert!(feeds.0[..36].iter().all(|&x| x == 0.));
        for mask in [&feeds.1, &feeds.2, &feeds.3] {
            let width = mask.len() / actions.len();
            assert!(mask[..width].iter().all(|&x| x == 0.));
            assert!(mask[width..].iter().all(|&x| x == 1.));
        }
        assert_eq!(&feeds.0[13 * 18..13 * 18 + 3], &[1., 0., 1.]);
        assert_eq!(&feeds.0[12 * 18..12 * 18 + 3], &[0., 1., 1.]);
    }

    #[test]
    fn invalid_tables_are_rejected_and_defaults_omit_features() {
        let original = config();
        for table in [vec![], vec![vec![0.; 18]; 17], vec![vec![0.; 17]; 18]] {
            let mut invalid = original.clone();
            invalid.world_action_features = Some(table);
            assert!(invalid.check().unwrap_err().contains("action features"));
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.01, -1.01] {
            let mut invalid = original.clone();
            invalid.world_action_features.as_mut().unwrap()[0][0] = value;
            assert!(invalid.check().unwrap_err().contains("action features"));
        }
        let value = serde_json::to_value(&original).unwrap();
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(value).unwrap(),
            original
        );
        let mut default = original;
        default.world_action_features = None;
        let mut value = serde_json::to_value(&default).unwrap();
        assert!(value.get("world_action_features").is_none());
        value["world_action_features"] = Value::Null;
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(value).unwrap(),
            default
        );
    }

    #[test]
    fn feature_headers_preserve_base_objective_and_reject_disguises() {
        for local in [false, true] {
            let value = metadata(local);
            let loss = world_learner::checkpoint_pixel_loss(&value).unwrap();
            assert_eq!(
                loss,
                if local {
                    PixelLoss::MotionEventRollout
                } else {
                    PixelLoss::MotionCrossEntropy
                }
            );
            assert_eq!(
                super::super::TrainingScope::from_metadata(&value, loss, 2, 1).unwrap(),
                super::super::TrainingScope::All
            );
            for (key, replacement) in [
                ("format", json!("kindle-world-only-v2")),
                ("action_feature_base_format", json!("kindle-world-only-v3")),
                ("action_feature_base_format", Value::Null),
                ("training_scope", json!("frozen_visual")),
                ("phase_start_updates", json!(1)),
                ("prediction_length", json!(0)),
                ("pixel_loss", json!("squared_error")),
            ] {
                let mut invalid = value.clone();
                invalid[key] = replacement;
                assert!(
                    world_learner::checkpoint_pixel_loss(&invalid).is_err(),
                    "{key}"
                );
            }
            let mut invalid = value.clone();
            invalid["config"]["world_action_features"] = Value::Null;
            assert!(world_learner::checkpoint_pixel_loss(&invalid).is_err());
            let mut legacy = base_metadata(&value).unwrap().unwrap();
            assert!(world_learner::checkpoint_pixel_loss(&legacy).is_ok());
            legacy["action_feature_base_format"] = value["action_feature_base_format"].clone();
            assert!(world_learner::checkpoint_pixel_loss(&legacy).is_err());
        }
    }

    #[test]
    fn unsupported_compositions_and_whole_agent_fail_before_gpu() {
        let config = config();
        for (horizon, loss) in [
            (0, PixelLoss::MotionCrossEntropy),
            (2, PixelLoss::MotionFixedTeacher),
            (2, PixelLoss::SquaredError),
        ] {
            assert!(
                world_learner::WorldModelLearner::with_pixel_loss(config.clone(), horizon, loss)
                    .is_err()
            );
        }
        let mut incompatible = config.clone();
        incompatible.world_mean_latents = true;
        assert!(
            world_learner::WorldModelLearner::with_pixel_loss(
                incompatible,
                2,
                PixelLoss::MotionCrossEntropy
            )
            .is_err()
        );
        assert!(std::panic::catch_unwind(|| DreamerCore::new(config)).is_err());
    }
}
