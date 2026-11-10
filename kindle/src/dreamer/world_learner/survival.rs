use super::{DreamerConfig, PixelLoss, rgb_event_heads};
use serde_json::{Value, json};

type Error = Box<dyn std::error::Error>;
const FORMAT: &str = "kindle-world-only-v29";

pub(super) fn validate(
    config: &DreamerConfig,
    horizon: usize,
    loss: PixelLoss,
) -> Result<(), Error> {
    if config.world_survival_scale.is_none() {
        return Ok(());
    }
    config.check().map_err(std::io::Error::other)?;
    rgb_event_heads::validate(config, horizon, loss)
}

pub(super) fn base_metadata(metadata: &Value) -> Result<Option<Value>, Error> {
    let enabled = !metadata["config"]["world_survival_scale"].is_null();
    if enabled != (metadata["format"] == FORMAT) {
        return Err("cumulative survival loss requires checkpoint format v29".into());
    }
    if !enabled {
        if metadata.get("world_survival_base_format").is_some()
            || metadata.get("world_survival_scale").is_some()
        {
            return Err("survival metadata requires a configured cumulative survival loss".into());
        }
        return Ok(None);
    }
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let horizon = metadata["prediction_length"]
        .as_u64()
        .ok_or("missing survival horizon")?;
    let loss: PixelLoss = serde_json::from_value(metadata["pixel_loss"].clone())?;
    validate(&config, usize::try_from(horizon)?, loss)?;
    if metadata["world_survival_base_format"] != "kindle-world-only-v28"
        || metadata["world_survival_scale"] != json!(config.world_survival_scale.unwrap())
        || metadata.get("training_scope").is_some()
        || metadata.get("phase_start_updates").is_some()
    {
        return Err(
            "invalid cumulative survival coefficient, base format or training scope".into(),
        );
    }
    let mut base = metadata.clone();
    base["format"] = "kindle-world-only-v28".into();
    let object = base.as_object_mut().unwrap();
    object.remove("world_survival_base_format");
    object.remove("world_survival_scale");
    base["config"]
        .as_object_mut()
        .unwrap()
        .remove("world_survival_scale");
    Ok(Some(base))
}

pub(super) fn wrap_metadata(metadata: &mut Value) -> Result<(), Error> {
    let config: DreamerConfig = serde_json::from_value(metadata["config"].clone())?;
    let scale = config
        .world_survival_scale
        .ok_or("missing cumulative survival coefficient")?;
    if metadata["format"] != "kindle-world-only-v28" {
        return Err("cumulative survival loss requires the RGB-event base format".into());
    }
    metadata["format"] = FORMAT.into();
    metadata["world_survival_base_format"] = "kindle-world-only-v28".into();
    metadata["world_survival_scale"] = json!(scale);
    base_metadata(metadata)?;
    Ok(())
}

#[cfg(test)]
mod tests;
