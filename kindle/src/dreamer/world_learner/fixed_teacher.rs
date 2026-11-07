//! A separate immutable observation filter anchors forecast coordinates.

use super::*;
use meganeura::data::safetensors::SafeTensorsModel;
use std::path::PathBuf;

pub(super) struct TeacherAnchor {
    pub path: PathBuf,
    pub sha256: String,
}

impl TeacherAnchor {
    fn new(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let sha256 = crate::vision::checkpoint_sha256(&path)?;
        Ok(Self { path, sha256 })
    }

    pub fn restore(
        directory: &Path,
        metadata: &serde_json::Value,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let anchor = Self::new(directory.join("teacher.safetensors"))?;
        if metadata["teacher_sha256"].as_str() != Some(&anchor.sha256) {
            return Err("fixed teacher anchor SHA-256 mismatch".into());
        }
        Ok(anchor)
    }

    pub fn verify(&self, session: &Session) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if crate::vision::checkpoint_sha256(&self.path)? != self.sha256 {
            return Err("fixed teacher anchor changed".into());
        }
        let model = SafeTensorsModel::load(self.path.clone())?;
        let names = session.param_names();
        let mut scalars = 0;
        for name in &names {
            let expected = model.tensor_f32(name)?;
            let mut actual = vec![0.0; expected.len()];
            session.read_param(name, &mut actual);
            if actual
                .iter()
                .zip(&expected)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("fixed teacher GPU parameter changed: {name}").into());
            }
            if session.has_param_grad(name) {
                return Err("fixed teacher unexpectedly has an optimizer gradient".into());
            }
            scalars += expected.len();
        }
        Ok((names.len(), scalars))
    }
}

// Independent of observations/actions and reproducible at the same optimizer age.
pub(super) fn sampling_seeds(seed: u64, age: u64, batch: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(
        seed ^ age.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x6669_7865_6474_6774,
    );
    (0..batch).map(|_| rng.random()).collect()
}

impl WorldModelLearner {
    pub fn start_fixed_teacher_phase(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::start_fixed_teacher_with_loss(path, PixelLoss::MotionFixedTeacher)
    }

    pub fn start_fixed_teacher_image_phase(
        path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::start_fixed_teacher_with_loss(path, PixelLoss::MotionFixedTeacherImage)
    }

    fn start_fixed_teacher_with_loss(
        path: &Path,
        objective: PixelLoss,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json"))?)?;
        let loss = checkpoint_pixel_loss(&metadata)?;
        let length = metadata
            .get("prediction_length")
            .map(|v| {
                v.as_u64()
                    .ok_or("invalid fixed teacher source prediction length")
            })
            .transpose()?
            .unwrap_or(0);
        let length = usize::try_from(length)?;
        let age = metadata["updates"].as_u64().ok_or("missing update count")?;
        let scope = TrainingScope::from_metadata(&metadata, loss, length, age)?;
        if scope != TrainingScope::All
            || loss != PixelLoss::MotionCrossEntropy
            || length != 0
            || age == 0
            || metadata["config"]["rgb_previous_frame"].as_bool() == Some(true)
            || metadata["config"]["rgb_latent_only"].as_bool() == Some(true)
        {
            return Err("fixed teacher phase requires an experienced single-frame all-weight H0 motion world".into());
        }
        let config = serde_json::from_value(metadata["config"].clone())?;
        let anchor = TeacherAnchor::new(path.join("world.safetensors"))?;
        let mut learner = Self::with_pixel_loss_inner(config, 16, objective, Some(anchor))?;
        learner.load_training_checkpoint(path, age)?;
        learner.training_scope = TrainingScope::FixedTeacher { start_updates: age };
        Ok(learner)
    }

    /// Privileged training targets, never a causal forecast of unseen images.
    /// With the same independent sampling seeds, targets stay exact while the student learns.
    pub fn fixed_teacher_targets(
        &mut self,
        observations: &[f32],
        actions: &[i32],
        sampling_seeds: &[u64],
    ) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        if self.fixed_teacher.is_none() {
            return Err("fixed teacher targets require the explicit phase");
        }
        let length = validate_history(&self.config, observations, actions)?;
        if length != self.config.batch_length {
            return Err("fixed teacher targets require one full training context");
        }
        let b = self.config.batch_size;
        let size = self.config.network();
        let width = self.config.observation_dim();
        let sw = size.stoch * size.classes;
        if sampling_seeds.len() != b {
            return Err("one independent teacher sampling seed per lane required");
        }
        let mut rngs = sampling_seeds
            .iter()
            .map(|&seed| StdRng::seed_from_u64(seed))
            .collect::<Vec<_>>();
        let mut h = vec![0.0; b * size.deter];
        let mut z = vec![0.0; b * sw];
        let mut logits = vec![0.0; b * sw];
        let mut hs = Vec::with_capacity(length * h.len());
        let mut zs = Vec::with_capacity(length * z.len());
        for t in 0..length {
            let (a, kh, kz, ka) = action_inputs(&self.config, &actions[t * b..(t + 1) * b]);
            for (name, values) in [
                ("previous_deter", &h),
                ("previous_stoch", &z),
                ("previous_action", &a),
                ("keep_deter", &kh),
                ("keep_stoch", &kz),
                ("keep_action", &ka),
            ] {
                self.observe.set_input(name, values);
            }
            self.observe.set_input(
                "observation",
                &observations[t * b * width..(t + 1) * b * width],
            );
            self.observe.step();
            self.observe.wait();
            self.observe.read_output_by_index(0, &mut h);
            self.observe.read_output_by_index(1, &mut logits);
            z.clear();
            for (row, rng) in rngs.iter_mut().enumerate() {
                z.extend(sample_latents(
                    &logits[row * sw..(row + 1) * sw],
                    1,
                    size.stoch,
                    size.classes,
                    self.config.unimix,
                    rng,
                ));
            }
            hs.extend_from_slice(&h);
            zs.extend_from_slice(&z);
        }
        if hs.iter().chain(&zs).any(|v| !v.is_finite()) {
            return Err("nonfinite fixed teacher target");
        }
        Ok((hs, zs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn teacher_sampling_stream_is_reproducible_at_saved_age() {
        assert_eq!(
            sampling_seeds(41001, 8192, 8),
            sampling_seeds(41001, 8192, 8)
        );
        assert_ne!(
            sampling_seeds(41001, 8192, 8),
            sampling_seeds(41001, 8193, 8)
        );
        assert_ne!(
            sampling_seeds(41001, 8192, 8),
            sampling_seeds(41002, 8192, 8)
        );
        let s = sampling_seeds(41001, u64::MAX, 8);
        assert_eq!(s.len(), 8);
        assert_eq!(s.iter().collect::<std::collections::HashSet<_>>().len(), 8);
    }
}

#[cfg(test)]
mod phase_tests {
    use super::*;
    use crate::dreamer::ObservationKind;
    use meganeura::graph::Op;
    use serde_json::json;

    fn config() -> DreamerConfig {
        let mut c = DreamerConfig::tiny(18);
        c.observation_kind = ObservationKind::Rgb32;
        c.batch_size = 2;
        c.batch_length = 32;
        c.world_backprop_length = 32;
        c.actor_critic_gradient = false;
        c.loss_scales.policy = 0.0;
        c.loss_scales.value = 0.0;
        c.loss_scales.replay_value = 0.0;
        c.loss_scales.reconstruction = 100.0;
        c.loss_scales.future_prediction = 0.0;
        c.disagreement_bonus = false;
        c
    }

    fn metadata() -> serde_json::Value {
        json!({"format":"kindle-world-only-v6","config":config(),"pixel_loss":"motion_fixed_teacher",
            "prediction_length":16,"training_scope":"fixed_teacher","phase_start_updates":8192,
            "updates":12288,"teacher_sha256":"0".repeat(64)})
    }

    #[test]
    fn fixed_teacher_requires_explicit_phase_before_device_construction() {
        let result =
            WorldModelLearner::with_pixel_loss(config(), 16, PixelLoss::MotionFixedTeacher);
        assert!(matches!(result,Err(e) if e.to_string().contains("explicit source phase")));
    }

    #[test]
    fn fixed_teacher_version_and_scope_cannot_be_disguised() {
        let m = metadata();
        assert_eq!(
            checkpoint_pixel_loss(&m).unwrap(),
            PixelLoss::MotionFixedTeacher
        );
        assert_eq!(
            TrainingScope::from_metadata(&m, PixelLoss::MotionFixedTeacher, 16, 12288).unwrap(),
            TrainingScope::FixedTeacher {
                start_updates: 8192
            }
        );
        for version in [
            "kindle-world-only-v1",
            "kindle-world-only-v2",
            "kindle-world-only-v3",
            "kindle-world-only-v4",
            "kindle-world-only-v5",
        ] {
            let mut bad = m.clone();
            bad["format"] = json!(version);
            assert!(checkpoint_pixel_loss(&bad).is_err());
        }
        for (key, value) in [
            ("training_scope", json!("all")),
            ("phase_start_updates", json!(0)),
            ("phase_start_updates", json!(12289)),
            ("teacher_sha256", json!("bad")),
        ] {
            let mut bad = m.clone();
            bad[key] = value;
            assert!(
                TrainingScope::from_metadata(&bad, PixelLoss::MotionFixedTeacher, 16, 12288)
                    .is_err()
            );
        }
        let mut bad = m;
        bad["pixel_loss"] = json!("motion_cross_entropy");
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }

    #[test]
    fn fixed_teacher_digest_is_checked_before_loading_world_or_gpu() {
        let path =
            std::env::temp_dir().join(format!("kindle-fixed-teacher-hash-{}", std::process::id()));
        assert!(!path.exists());
        std::fs::create_dir(&path).unwrap();
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec(&metadata()).unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("teacher.safetensors"), b"wrong anchor").unwrap();
        let result = WorldModelLearner::restore(&path);
        assert!(matches!(result,Err(e) if e.to_string().contains("anchor SHA-256 mismatch")));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn fixed_teacher_graph_has_both_losses_and_auxiliary_gradients_only_in_dynamics() {
        let c = config();
        let mut g =
            world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionFixedTeacher);
        assert!(g.outputs().len() > world::LOSS_RECURRENT_WORLD + 3);
        let inputs = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        for t in 0..32 {
            assert!(inputs.contains(format!("fixed_teacher_deter_{t}").as_str()));
            assert!(inputs.contains(format!("fixed_teacher_stoch_{t}").as_str()));
            assert!(!inputs.contains(format!("uniforms_{t}").as_str()));
        }
        let input_count = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .count();
        assert_eq!(
            inputs.len(),
            input_count,
            "named training inputs must be unique"
        );
        let kl = g.outputs()[world::LOSS_WORLD_ROLLOUT];
        let energy = g.outputs()[world::LOSS_RECURRENT_WORLD];
        let energy = g.scale(energy, 100.0);
        let aux = g.add(kl, energy);
        g.set_outputs(vec![aux]);
        let (plan, _) = meganeura::compile_training_graph(&g);
        let gradients = plan
            .param_buffers
            .iter()
            .filter(|(_, id)| plan.param_grad_pairs.iter().any(|(p, _)| p == id))
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert!(!gradients.is_empty());
        // Compiled buffers can alias a scalar zero-gradient sentinel. Inspect
        // actual autodiff outputs rather than classifying that storage as a gradient.
        let backward = meganeura::autodiff::differentiate(&g);
        let names = g.nodes().iter().filter_map(|n| match &n.op {
            Op::Parameter { name } => Some(name),
            _ => None,
        });
        for (name, &gradient) in names.zip(&backward.outputs()[1..]) {
            if !name.starts_with("world.dynamics.") {
                assert!(
                    matches!(&backward.node(gradient).op,Op::Constant{data} if data.iter().all(|&v|v==0.0)),
                    "unexpected auxiliary gradient: {name}"
                );
            }
        }
        assert!(
            gradients
                .iter()
                .any(|n| n.starts_with("world.dynamics.core."))
        );
        assert!(
            gradients
                .iter()
                .any(|n| n.starts_with("world.dynamics.prior"))
        );
    }
    #[test]
    fn image_phase_requires_v7_and_an_explicit_source_before_gpu() {
        let mut m = metadata();
        m["format"] = json!("kindle-world-only-v7");
        m["pixel_loss"] = json!("motion_fixed_teacher_image");
        assert_eq!(
            checkpoint_pixel_loss(&m).unwrap(),
            PixelLoss::MotionFixedTeacherImage
        );
        assert_eq!(
            TrainingScope::from_metadata(&m, PixelLoss::MotionFixedTeacherImage, 16, 12288)
                .unwrap(),
            TrainingScope::FixedTeacher {
                start_updates: 8192
            }
        );
        for version in 1..=6 {
            let mut bad = m.clone();
            bad["format"] = json!(format!("kindle-world-only-v{version}"));
            assert!(checkpoint_pixel_loss(&bad).is_err());
        }
        m["pixel_loss"] = json!("motion_fixed_teacher");
        assert!(checkpoint_pixel_loss(&m).is_err());
        assert!(
            matches!(WorldModelLearner::with_pixel_loss(config(),16,PixelLoss::MotionFixedTeacherImage),
            Err(e) if e.to_string().contains("explicit source phase"))
        );
    }

    #[test]
    fn image_phase_inputs_are_unique_and_only_use_pixel_rollout_samples() {
        let c = config();
        let g = world::build_world_only_pixel_training_graph(
            &c,
            16,
            PixelLoss::MotionFixedTeacherImage,
        );
        let old =
            world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
        let inputs = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let unique = inputs
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(inputs.len(), unique.len());
        for t in 0..32 {
            assert!(unique.contains(format!("fixed_teacher_deter_{t}").as_str()));
            assert!(unique.contains(format!("fixed_teacher_stoch_{t}").as_str()));
            assert!(!unique.contains(format!("uniforms_{t}").as_str()));
        }
        for t in 0..16 {
            assert!(unique.contains(format!("pixel_rollout_weight_{t}").as_str()));
            assert!(unique.contains(format!("pixel_rollout_uniforms_{t}").as_str()));
        }
        assert!(inputs.iter().all(|n| !n.starts_with("recurrent_")
            && !n.starts_with("latent_rollout_")
            && !n.starts_with("fixed_teacher_recurrent_")));
        let (plan, _) = meganeura::compile_training_graph(&g);
        let (reference, _) = meganeura::compile_training_graph(&old);
        let geometry = |p: &meganeura::compile::ExecutionPlan| {
            p.param_buffers
                .iter()
                .map(|(n, b)| (n.clone(), p.param_types[b].clone()))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(geometry(&plan), geometry(&reference));
        assert_eq!(plan.param_buffers.len(), 68);
        assert_eq!(g.outputs().len(), world::LOSS_WORLD_ROLLOUT + 3);
        assert!(PixelLoss::MotionFixedTeacherImage.motion_attention());
        assert!(!PixelLoss::MotionFixedTeacherImage.latent_overshooting());
        assert!(!PixelLoss::MotionFixedTeacherImage.recurrent_energy());
    }
}
