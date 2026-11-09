use super::{DreamerConfig, PixelLoss};
use serde_json::Value;

type Error = Box<dyn std::error::Error>;
const FORMAT: &str = "kindle-world-only-v27";

pub(super) fn validate(
    config: &DreamerConfig,
    horizon: usize,
    loss: PixelLoss,
) -> Result<(), Error> {
    if !config.rgb_wide_motion {
        return Ok(());
    }
    config.check().map_err(std::io::Error::other)?;
    if horizon == 0 || loss != PixelLoss::MotionEventRollout {
        return Err("wide motion requires a positive motion-event suffix".into());
    }
    Ok(())
}

pub(super) fn base_metadata(metadata: &Value) -> Result<Option<Value>, Error> {
    let enabled = metadata["config"]["rgb_wide_motion"] == true;
    if enabled != (metadata["format"] == FORMAT) {
        return Err("wide motion requires checkpoint format v27".into());
    }
    if !enabled {
        if metadata.get("wide_motion_base_format").is_some() {
            return Err("wide motion base format requires wide motion".into());
        }
        return Ok(None);
    }
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing wide motion suffix")?;
    let loss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, loss)?;
    let expected = if config.world_action_features.is_some() {
        "kindle-world-only-v25"
    } else {
        "kindle-world-only-v18"
    };
    if metadata["wide_motion_base_format"] != expected
        || metadata.get("training_scope").is_some()
        || metadata.get("phase_start_updates").is_some()
    {
        return Err("invalid wide motion base format or training scope".into());
    }
    let mut base = metadata.clone();
    base["format"] = expected.into();
    base.as_object_mut()
        .unwrap()
        .remove("wide_motion_base_format");
    base["config"]
        .as_object_mut()
        .unwrap()
        .remove("rgb_wide_motion");
    Ok(Some(base))
}

pub(super) fn wrap_metadata(metadata: &mut Value) -> Result<(), Error> {
    let expected = if metadata["config"]["world_action_features"].is_null() {
        "kindle-world-only-v18"
    } else {
        "kindle-world-only-v25"
    };
    if metadata["format"] != expected {
        return Err("unsupported wide motion checkpoint composition".into());
    }
    metadata["wide_motion_base_format"] = expected.into();
    metadata["format"] = FORMAT.into();
    Ok(())
}

#[cfg(test)]
mod tests;
