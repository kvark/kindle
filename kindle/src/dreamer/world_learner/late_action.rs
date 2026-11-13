use super::{DreamerConfig, PixelLoss};
use serde_json::Value;

type Error = Box<dyn std::error::Error>;
const FORMAT: &str = "kindle-world-only-v26";

pub(super) fn validate(
    config: &DreamerConfig,
    horizon: usize,
    loss: PixelLoss,
) -> Result<(), Error> {
    if !config.rgb_late_action {
        return Ok(());
    }
    config.check().map_err(std::io::Error::other)?;
    if horizon == 0 || loss != PixelLoss::MotionEventRollout {
        return Err("late action requires a positive motion-event suffix".into());
    }
    Ok(())
}

pub(super) fn base_metadata(metadata: &Value) -> Result<Option<Value>, Error> {
    let enabled = metadata["config"]["rgb_late_action"] == true;
    if enabled != (metadata["format"] == FORMAT) {
        return Err("late action requires checkpoint format v26".into());
    }
    if !enabled {
        if metadata.get("late_action_base_format").is_some() {
            return Err("late action base format requires late action".into());
        }
        return Ok(None);
    }
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing late action suffix")?;
    let loss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, loss)?;
    let expected = if config.world_action_features.is_some() {
        "kindle-world-only-v25"
    } else {
        "kindle-world-only-v18"
    };
    if metadata["late_action_base_format"] != expected
        || metadata.get("training_scope").is_some()
        || metadata.get("phase_start_updates").is_some()
    {
        return Err("invalid late action base format or training scope".into());
    }
    let mut base = metadata.clone();
    base["format"] = expected.into();
    base.as_object_mut()
        .unwrap()
        .remove("late_action_base_format");
    base["config"]
        .as_object_mut()
        .unwrap()
        .remove("rgb_late_action");
    Ok(Some(base))
}

pub(super) fn wrap_metadata(metadata: &mut Value) -> Result<(), Error> {
    let expected = if metadata["config"]["world_action_features"].is_null() {
        "kindle-world-only-v18"
    } else {
        "kindle-world-only-v25"
    };
    if metadata["format"] != expected {
        return Err("unsupported late action checkpoint composition".into());
    }
    metadata["late_action_base_format"] = expected.into();
    metadata["format"] = FORMAT.into();
    Ok(())
}

#[cfg(test)]
mod tests;
