//! Frozen visual perception for Dreamer.
//!
//! The encoder runs in a separate inference-only Meganeura session. Its
//! parameters therefore cannot accidentally enter the Dreamer optimizer.
//! Dreamer consumes a fixed-size projected patch grid. Encoder identity and
//! temporal semantics are checkpointed; equal shapes do not mean equal features.

use std::path::Path;

#[cfg(target_os = "linux")]
pub mod capture;
pub mod levjepa;
pub mod preprocess;
pub mod preprocess_gpu;
pub mod probe;

/// Channels retained by the fixed Johnson–Lindenstrauss projection.
pub const OBSERVATION_CHANNELS: usize = 64;
/// Spatial side after fixed 2×2 pooling of LeVJEPA's 14×14 patch grid.
pub const OBSERVATION_GRID: usize = 7;
/// Stable seed for the non-trainable projection matrix.
pub const PROJECTION_SEED: u64 = 0xd1_30_00_03_00_00_00_01;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerceptionKind {
    LeVJepa,
    #[serde(rename = "levjepa-tiny")]
    LeVJepaTiny,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PerceptionIdentity {
    pub kind: PerceptionKind,
    pub model_id: String,
    pub checkpoint_revision: String,
    pub encoding_revision: String,
    pub checkpoint_sha256: String,
}

impl PerceptionKind {
    pub fn identity(self, fingerprint: String) -> PerceptionIdentity {
        let (model_id, checkpoint_revision, encoding_revision) = match self {
            Self::LeVJepa => (
                levjepa::MODEL_ID,
                levjepa::CHECKPOINT_REV,
                levjepa::ENCODING_REV,
            ),
            Self::LeVJepaTiny => (
                "kindle/LeVJEPA-Tiny",
                "local-sha256",
                levjepa::Architecture::Tiny.encoding_revision(),
            ),
        };
        PerceptionIdentity {
            kind: self,
            model_id: model_id.to_owned(),
            checkpoint_revision: checkpoint_revision.to_owned(),
            encoding_revision: encoding_revision.to_owned(),
            checkpoint_sha256: fingerprint,
        }
    }

    pub(crate) fn levjepa_architecture(self) -> Option<levjepa::Architecture> {
        match self {
            Self::LeVJepa => Some(levjepa::Architecture::Large),
            Self::LeVJepaTiny => Some(levjepa::Architecture::Tiny),
        }
    }
}

impl PerceptionIdentity {
    pub(crate) fn validate(&self) -> std::io::Result<()> {
        let hash = &self.checkpoint_sha256;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid perception fingerprint",
            ));
        }
        if *self != self.kind.identity(hash.clone()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported perception identity or temporal encoding",
            ));
        }
        Ok(())
    }

    pub(crate) fn verify_file(&self, checkpoint: &Path) -> std::io::Result<()> {
        self.validate()?;
        let actual = checkpoint_sha256(checkpoint)?;
        if actual != self.checkpoint_sha256 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "perception checkpoint SHA-256 is {actual}, expected {}",
                    self.checkpoint_sha256
                ),
            ));
        }
        Ok(())
    }
}

pub(crate) fn checkpoint_sha256(path: &Path) -> std::io::Result<String> {
    hash_reader(std::fs::File::open(path)?)
}

fn hash_reader(mut reader: impl std::io::Read) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    let mut buffer = [0; 65_536];
    loop {
        let count = match reader.read(&mut buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Ok(format!("{:x}", digest.finalize()));
        }
        digest.update(&buffer[..count]);
    }
}

/// One visual observation in token-major `[7 * 7, 64]` order: frozen visual
/// features, or losslessly packed small images for a learned frontend. Replay
/// retains exactly the values observed at collection time.
#[derive(Clone, Debug)]
pub struct Observation {
    values: Box<[f32]>,
}

impl Observation {
    pub const LEN: usize = OBSERVATION_GRID * OBSERVATION_GRID * OBSERVATION_CHANNELS;

    pub fn from_vec(values: Vec<f32>) -> Self {
        assert_eq!(values.len(), Self::LEN);
        assert!(values.iter().all(|value| value.is_finite()));
        Self {
            values: values.into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

pub(crate) fn fixed_projection(input: usize, output: usize, seed: u64) -> Vec<f32> {
    assert!(input > 0 && output > 0);
    let scale = 1.0 / (output as f32).sqrt();
    let mut state = seed;
    (0..input * output)
        .map(|_| {
            // xorshift64*: deterministic across platforms and Rust versions.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bit = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 63;
            if bit == 0 { -scale } else { scale }
        })
        .collect()
}

#[cfg(test)]
fn pool_2x2_token_major(input: &[f32], grid: usize, channels: usize, output: &mut [f32]) {
    assert_eq!(grid % 2, 0);
    assert_eq!(input.len(), grid * grid * channels);
    let out_grid = grid / 2;
    assert_eq!(output.len(), out_grid * out_grid * channels);
    for y in 0..out_grid {
        for x in 0..out_grid {
            for channel in 0..channels {
                let mut sum = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let token = (2 * y + dy) * grid + 2 * x + dx;
                        sum += input[token * channels + channel];
                    }
                }
                output[(y * out_grid + x) * channels + channel] = 0.25 * sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perception_sizes_have_distinct_stable_checkpoint_identities() {
        for (kind, encoded, architecture) in [
            (
                PerceptionKind::LeVJepa,
                "levjepa",
                Some(levjepa::Architecture::Large),
            ),
            (
                PerceptionKind::LeVJepaTiny,
                "levjepa-tiny",
                Some(levjepa::Architecture::Tiny),
            ),
        ] {
            assert_eq!(serde_json::to_value(kind).unwrap(), encoded);
            assert_eq!(kind.levjepa_architecture(), architecture);
            let identity = kind.identity("1".repeat(64));
            identity.validate().unwrap();
            let restored: PerceptionIdentity =
                serde_json::from_value(serde_json::to_value(&identity).unwrap()).unwrap();
            assert_eq!(restored, identity);
        }
        let mut tiny = PerceptionKind::LeVJepaTiny.identity("2".repeat(64));
        tiny.kind = PerceptionKind::LeVJepa;
        assert!(tiny.validate().is_err());
        let large = PerceptionKind::LeVJepa.identity(levjepa::CHECKPOINT_SHA256.into());
        assert_eq!(large.model_id, levjepa::MODEL_ID);
        assert_eq!(large.checkpoint_revision, levjepa::CHECKPOINT_REV);
        assert_eq!(large.encoding_revision, levjepa::ENCODING_REV);
    }

    #[test]
    fn fingerprint_matches_sha256_known_vectors_and_chunked_reads() {
        use sha2::{Digest, Sha256};

        assert_eq!(
            hash_reader(&b"abc"[..]).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hash_reader(&b""[..]).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let bytes = vec![17; 131_079];
        assert_eq!(
            hash_reader(bytes.as_slice()).unwrap(),
            format!("{:x}", Sha256::digest(&bytes))
        );
    }

    #[test]
    fn projection_and_pooling_are_fixed_and_spatial() {
        assert_eq!(
            fixed_projection(384, 64, PROJECTION_SEED),
            fixed_projection(384, 64, PROJECTION_SEED)
        );
        let input: Vec<f32> = (0..14 * 14 * 2).map(|value| value as f32).collect();
        let mut output = vec![0.0; 7 * 7 * 2];
        pool_2x2_token_major(&input, 14, 2, &mut output);
        let expected = (input[0] + input[2] + input[28] + input[30]) / 4.0;
        assert_eq!(output[0], expected);
        assert_eq!(output.len(), 98);
    }
}
