//! Integrity checks around the backend's permissive checkpoint loader.

use std::collections::HashSet;
use std::io;
use std::path::Path;

use meganeura::{Session, data::safetensors::SafeTensorsModel};

pub(super) const WORLD: &str = "world.safetensors";
pub(super) const BEHAVIOR: &str = "behavior.safetensors";
pub(super) const SLOW_VALUE: &str = "slow_value.safetensors";

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(super) struct TensorFingerprints {
    world: String,
    behavior: String,
    slow_value: String,
}

impl TensorFingerprints {
    pub fn read(directory: &Path) -> io::Result<Self> {
        Ok(Self {
            world: crate::vision::checkpoint_sha256(&directory.join(WORLD))?,
            behavior: crate::vision::checkpoint_sha256(&directory.join(BEHAVIOR))?,
            slow_value: crate::vision::checkpoint_sha256(&directory.join(SLOW_VALUE))?,
        })
    }

    pub fn verify(&self, directory: &Path) -> io::Result<()> {
        for (name, expected) in [
            (WORLD, &self.world),
            (BEHAVIOR, &self.behavior),
            (SLOW_VALUE, &self.slow_value),
        ] {
            let actual = crate::vision::checkpoint_sha256(&directory.join(name))?;
            if actual != *expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("checkpoint {name} SHA-256 mismatch; damaged or mixed save"),
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn load_session(
    session: &mut Session,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // The backend allows partial model loading and absent optimizer moments.
    // A training restore must not silently keep their initialized values.
    let model = SafeTensorsModel::load(path.to_path_buf())?;
    let mut required = Vec::new();
    for name in session.param_names() {
        required.push(name.to_owned());
        if session.has_param_grad(name) {
            required.push(format!("adam_m.{name}"));
            required.push(format!("adam_v.{name}"));
        }
    }
    check_tensor_names(model.tensor_info().keys().map(String::as_str), &required)?;
    drop(model);
    session.load_checkpoint(path)?;
    Ok(())
}

fn check_tensor_names<'a>(
    present: impl Iterator<Item = &'a str>,
    required: &[String],
) -> io::Result<()> {
    let present = present.collect::<HashSet<_>>();
    for name in required {
        if !present.contains(name.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint missing required tensor {name}"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_restore_requires_parameters_and_both_optimizer_moments() {
        let names = ["weight", "adam_m.weight", "adam_v.weight"];
        let required = names.map(str::to_owned);
        check_tensor_names(names.into_iter(), &required).unwrap();
        for omitted in names {
            let error =
                check_tensor_names(names.into_iter().filter(|name| *name != omitted), &required)
                    .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains(omitted));
        }
    }

    #[test]
    fn inference_restore_may_read_a_subset_of_a_training_checkpoint() {
        check_tensor_names(
            ["weight", "adam_m.weight", "adam_v.weight"].into_iter(),
            &["weight".to_owned()],
        )
        .unwrap();
    }
}
