use super::{DreamerConfig, PixelLoss};
use serde_json::{Value, json};

type Error = Box<dyn std::error::Error>;
const FORMAT: &str = "kindle-world-only-v28";

pub(super) fn validate(
    config: &DreamerConfig,
    horizon: usize,
    loss: PixelLoss,
) -> Result<(), Error> {
    if !config.rgb_event_heads {
        return Ok(());
    }
    config.check().map_err(std::io::Error::other)?;
    super::validate_config(config).map_err(std::io::Error::other)?;
    if horizon != 16 || loss != PixelLoss::MotionEventRollout {
        return Err("RGB event heads require the H16 motion-event objective".into());
    }
    Ok(())
}

fn geometry(config: &DreamerConfig) -> Value {
    json!({
        "latent": config.feature_dim(),
        "visual": config.encoded_observation_dim(),
        "inputs": config.feature_dim() + config.encoded_observation_dim(),
        "units": config.network().units,
    })
}

pub(super) fn base_metadata(metadata: &Value) -> Result<Option<Value>, Error> {
    let enabled = metadata["config"]["rgb_event_heads"] == true;
    if enabled != (metadata["format"] == FORMAT) {
        return Err("RGB event heads require checkpoint format v28".into());
    }
    if !enabled {
        if metadata.get("rgb_event_head_base_format").is_some()
            || metadata.get("rgb_event_head_geometry").is_some()
        {
            return Err("RGB event head metadata requires RGB event heads".into());
        }
        return Ok(None);
    }
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing RGB event head suffix")?;
    let loss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, loss)?;
    let expected = if config.world_action_features.is_some() {
        "kindle-world-only-v25"
    } else {
        "kindle-world-only-v18"
    };
    if metadata["rgb_event_head_base_format"] != expected
        || metadata["rgb_event_head_geometry"] != geometry(&config)
        || metadata.get("training_scope").is_some()
        || metadata.get("phase_start_updates").is_some()
    {
        return Err("invalid RGB event head base, geometry or training scope".into());
    }
    let mut base = metadata.clone();
    base["format"] = expected.into();
    let object = base.as_object_mut().unwrap();
    object.remove("rgb_event_head_base_format");
    object.remove("rgb_event_head_geometry");
    base["config"]
        .as_object_mut()
        .unwrap()
        .remove("rgb_event_heads");
    Ok(Some(base))
}

pub(super) fn wrap_metadata(metadata: &mut Value) -> Result<(), Error> {
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    if !config.rgb_event_heads {
        return Err("RGB event head wrapper requires its flag".into());
    }
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing RGB event head suffix")?;
    let loss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, loss)?;
    let expected = if config.world_action_features.is_some() {
        "kindle-world-only-v25"
    } else {
        "kindle-world-only-v18"
    };
    if metadata["format"] != expected {
        return Err("unsupported RGB event head checkpoint composition".into());
    }
    metadata["rgb_event_head_base_format"] = expected.into();
    metadata["rgb_event_head_geometry"] = geometry(&config);
    metadata["format"] = FORMAT.into();
    Ok(())
}

#[cfg(test)]
mod tests;
