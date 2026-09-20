//! GPU AdamW and export-only EMA. Video sampling stays outside the learner.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig, SessionOptions, graph::Op};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{ARCHITECTURE, Config, PATCH_DIM, encoder_weights, graph};

type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub model: Config,
    pub seed: u64,
    pub steps: u32,
    pub warmup_steps: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub ema_decay: f32,
}

impl TrainingConfig {
    pub fn validate(self) -> Result<(), &'static str> {
        self.model.validate()?;
        if self.steps == 0
            || self.warmup_steps >= self.steps
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || !self.ema_decay.is_finite()
            || !(0.0..1.0).contains(&self.ema_decay)
        {
            return Err("invalid pretraining budget or optimizer configuration");
        }
        Ok(())
    }

    /// One-based update: linear warmup, then cosine decay to 1% of peak.
    pub fn rate(self, update: u32) -> f32 {
        assert!((1..=self.steps).contains(&update));
        if update <= self.warmup_steps {
            return self.learning_rate * (update as f32 / self.warmup_steps as f32);
        }
        let progress =
            (update - self.warmup_steps) as f32 / (self.steps - self.warmup_steps) as f32;
        self.learning_rate * (0.01 + 0.495 * (1.0 + (std::f32::consts::PI * progress).cos()))
    }
}

/// Token-major `[kept, clips, 768]` patches and `[kept, clips]` original IDs.
/// Local clips are view-major: the B clips of local view 0 precede view 1.
pub struct Batch<'a> {
    pub global_patches: &'a [f32],
    pub global_ids: &'a [u32],
    pub local_patches: &'a [f32],
    pub local_ids: &'a [u32],
    /// Fresh Gaussian unit-length columns, shared across all views.
    pub directions: &'a [f32],
}

#[derive(Debug, serde::Serialize)]
pub struct Metrics {
    pub step: u32,
    pub learning_rate: f32,
    pub loss: f32,
    pub invariance: f32,
    pub sigreg: f32,
}

pub struct Trainer {
    config: TrainingConfig,
    session: Session,
    ema: Session,
    step: u32,
    ready: bool,
}

fn ema_graph(decay: f32) -> Graph {
    let mut graph = Graph::new();
    let source = encoder_weights(&mut graph);
    let position = graph.input_u32("position", &[1]);
    let mut outputs = Vec::new();
    for (name, parameter) in source {
        let size = graph.node(parameter).ty.num_elements();
        let source = graph.reshape(parameter, &[1, size]);
        let average = graph.parameter(&format!("ema.{name}"), &[1, size]);
        let old = graph.scale(average, decay);
        let new = graph.scale(source, 1.0 - decay);
        let next = graph.add(old, new);
        outputs.push(graph.cache_write(next, average, position));
    }
    graph.set_outputs(outputs);
    graph
}

fn initial_values(
    name: &str,
    shape: &[usize],
    projector_hidden: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let size = shape.iter().product();
    if name.contains("norm") {
        return vec![if name.ends_with("weight") { 1.0 } else { 0.0 }; size];
    }
    if name.starts_with("projector.") {
        let fan_in = if shape.len() == 2 {
            shape[0]
        } else if name.starts_with("projector.fc1.") {
            ARCHITECTURE.hidden()
        } else {
            projector_hidden
        };
        let bound = 1.0 / (fan_in as f32).sqrt();
        return (0..size).map(|_| rng.random_range(-bound..bound)).collect();
    }
    if name.ends_with(".bias") {
        return vec![0.0; size];
    }
    let residual_scale = if name.ends_with("attn.proj.weight") || name.ends_with("mlp.fc2.weight") {
        let layer: usize = name.split('.').nth(2).unwrap().parse().unwrap();
        (2.0 * (layer + 1) as f32).sqrt().recip()
    } else {
        1.0
    };
    let mut values = Vec::with_capacity(size);
    while values.len() < size {
        let radius = (-2.0 * rng.random_range(f32::EPSILON..1.0).ln()).sqrt();
        let angle = std::f32::consts::TAU * rng.random::<f32>();
        for z in [radius * angle.cos(), radius * angle.sin()] {
            // Upstream truncates at absolute [-2,2], not at two std deviations.
            let value = z * 0.02;
            if value.abs() <= 2.0 {
                values.push(value * residual_scale);
                if values.len() == size {
                    break;
                }
            }
        }
    }
    values
}

fn check_memory(session: &Session) -> Result<(), Error> {
    let stats = session
        .device_memory_stats()
        .ok_or("Vulkan memory budget unavailable")?;
    if stats.budget_bytes.saturating_sub(stats.usage_bytes) < 2 * 1024 * 1024 * 1024 {
        return Err("pretraining requires 2 GiB estimated Vulkan budget headroom".into());
    }
    Ok(())
}

fn export_bytes(name: &str, shape: &[usize], native: &[f32]) -> Vec<u8> {
    assert_eq!(native.len(), shape.iter().product::<usize>());
    if shape.len() < 2 || name == "encoder.cls_token" {
        return native.iter().flat_map(|v| v.to_le_bytes()).collect();
    }
    let rows = shape[0];
    let columns = native.len() / rows;
    (0..rows)
        .flat_map(|row| {
            (0..columns).flat_map(move |column| native[column * rows + row].to_le_bytes())
        })
        .collect()
}

fn check_saved_tensors(
    model: &meganeura::data::safetensors::SafeTensorsModel,
    graph: &Graph,
    moments: bool,
) -> Result<(), Error> {
    let mut count = 0;
    for node in graph.nodes() {
        if let Op::Parameter { name } = &node.op {
            for prefix in if moments {
                &["", "adam_m.", "adam_v."][..]
            } else {
                &[""][..]
            } {
                let key = format!("{prefix}{name}");
                let info = model
                    .tensor_info()
                    .get(&key)
                    .ok_or_else(|| format!("missing {key}"))?;
                if info.shape != node.ty.shape
                    || !model.tensor_f32(&key)?.iter().all(|v| v.is_finite())
                {
                    return Err(format!("invalid saved tensor {key}").into());
                }
                count += 1;
            }
        }
    }
    if model.tensor_info().len() != count {
        return Err("unexpected saved tensors".into());
    }
    Ok(())
}

impl Trainer {
    /// Builds native sessions; call only inside a declared direct-process GPU guard.
    pub fn new(config: TrainingConfig, gpu: Arc<blade_graphics::Context>) -> Result<Self, Error> {
        config.validate()?;
        let model = graph(config.model)?;
        let options = || SessionOptions {
            coop: CoopPolicy::Disabled,
            ..Default::default()
        };
        let (mut session, _) = meganeura::build(
            &model,
            SessionConfig {
                gpu: Some(Arc::clone(&gpu)),
                runtime: options(),
                ..Default::default()
            },
        );
        check_memory(&session)?;
        let (mut ema, _) = meganeura::build(
            &ema_graph(config.ema_decay),
            SessionConfig {
                mode: Mode::Inference,
                gpu: Some(gpu),
                runtime: options(),
                ..Default::default()
            },
        );
        ema.set_input_u32("position", &[0]);
        for (name, _) in super::super::weight_shapes(ARCHITECTURE) {
            ema.share_parameter_from(&mut session, &name)?;
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        for node in model.nodes() {
            if let Op::Parameter { name } = &node.op {
                let values = initial_values(
                    name,
                    &node.ty.shape,
                    config.model.projector_hidden,
                    &mut rng,
                );
                session.set_parameter(name, &values);
                if name.starts_with("encoder.") {
                    ema.set_parameter(&format!("ema.{name}"), &values);
                }
            }
        }
        session.set_adam(config.rate(1), 0.9, 0.999, 1e-8);
        session.set_weight_decay(config.weight_decay);
        check_memory(&session)?;
        Ok(Self {
            config,
            session,
            ema,
            step: 0,
            ready: true,
        })
    }

    pub fn step(&mut self, batch: Batch<'_>) -> Result<Metrics, Error> {
        if !self.ready || self.step >= self.config.steps {
            return Err("pretrainer is failed or its declared budget is exhausted".into());
        }
        let config = self.config.model;
        let mut inputs = Vec::new();
        for (name, view, patches, ids) in [
            (
                "global",
                config.global(),
                batch.global_patches,
                batch.global_ids,
            ),
            (
                "local",
                config.local(),
                batch.local_patches,
                batch.local_ids,
            ),
        ] {
            if patches.len() != view.kept * view.clips * PATCH_DIM
                || !patches.iter().all(|x| x.is_finite())
            {
                return Err("invalid pretraining patch tensor".into());
            }
            let (positions, mask) = view.attention_inputs(ids)?;
            inputs.push((name, patches, positions, mask));
        }
        if batch.directions.len() != config.projector_output * config.directions
            || !batch.directions.iter().all(|v| v.is_finite())
        {
            return Err("invalid SIGReg projection tensor".into());
        }
        for column in 0..config.directions {
            let norm: f64 = batch
                .directions
                .chunks_exact(config.directions)
                .map(|row| f64::from(row[column]).powi(2))
                .sum();
            if (norm - 1.0).abs() > 1e-4 {
                return Err("SIGReg projection columns must have unit norm".into());
            }
        }
        self.ready = false;
        for (name, patches, positions, mask) in inputs {
            self.session.set_input(&format!("{name}.patches"), patches);
            self.session
                .set_input_u32(&format!("{name}.positions"), &positions);
            self.session.set_input(&format!("{name}.mask"), &mask);
        }
        self.session
            .set_input("sigreg.directions", batch.directions);
        let learning_rate = self.config.rate(self.step + 1);
        self.session.set_adam(learning_rate, 0.9, 0.999, 1e-8);
        self.session.step();
        self.session.wait();
        check_memory(&self.session)?;
        let mut values = [0.0; 3];
        for (index, value) in values.iter_mut().enumerate() {
            self.session
                .read_output_by_index(index, std::slice::from_mut(value));
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err("non-finite pretraining loss; learner stopped".into());
        }
        self.ema.step();
        self.ema.wait();
        check_memory(&self.session)?;
        self.step += 1;
        self.ready = true;
        Ok(Metrics {
            step: self.step,
            learning_rate,
            loss: values[0],
            invariance: values[1],
            sigreg: values[2],
        })
    }

    pub fn device_info(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    /// Export only the averaged encoder, in the Tiny loader's published tensor
    /// layout. No projector, optimizer moments or live perception caches.
    pub fn export_encoder(&mut self, path: &Path) -> Result<(), Error> {
        use safetensors::{Dtype, tensor::TensorView};
        use std::io::Write;
        if !self.ready {
            return Err("cannot export a failed pretrainer".into());
        }
        self.ema.wait();
        let mut tensors = Vec::new();
        for (name, shape) in super::super::weight_shapes(ARCHITECTURE) {
            let mut values = vec![0.0; shape.iter().product()];
            self.ema.read_param(&format!("ema.{name}"), &mut values);
            if !values.iter().all(|v| v.is_finite()) {
                return Err("non-finite EMA weights".into());
            }
            let bytes = export_bytes(&name, &shape, &values);
            tensors.push((name, shape, bytes));
        }
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                Ok((
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        let metadata = std::collections::HashMap::from([
            (
                "architecture".to_owned(),
                "kindle-levjepa-tiny16-chunk16-v1".to_owned(),
            ),
            ("ema_step".to_owned(), self.step.to_string()),
        ]);
        let bytes = safetensors::tensor::serialize(views, &Some(metadata))?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        output.write_all(&bytes)?;
        Ok(())
    }

    /// Restore all 155 parameters, 310 moments, EMA and step counter. External
    /// batch selection must resume at the same declared seed/step separately.
    pub fn restore(directory: &Path, gpu: Arc<blade_graphics::Context>) -> Result<Self, Error> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("training.json"))?)?;
        if metadata["format"] != 1
            || metadata["meganeura"] != crate::dreamer::MEGANEURA_REV
            || metadata["blade"] != crate::dreamer::BLADE_REV
        {
            return Err("pretraining checkpoint format or backend mismatch".into());
        }
        let config: TrainingConfig = serde_json::from_value(metadata["config"].clone())?;
        config.validate()?;
        let step: u32 = serde_json::from_value(metadata["step"].clone())?;
        if step > config.steps {
            return Err("saved step exceeds training budget".into());
        }
        for name in ["training.safetensors", "ema.safetensors"] {
            let actual = crate::vision::checkpoint_sha256(&directory.join(name))?;
            if metadata["sha256"][name].as_str() != Some(actual.as_str()) {
                return Err(format!("checkpoint hash mismatch: {name}").into());
            }
        }
        let bytes = std::fs::read(directory.join("training.safetensors"))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
        if header.metadata().as_ref().and_then(|m| m.get("adam_step")) != Some(&step.to_string()) {
            return Err("saved optimizer and training steps differ".into());
        }
        let training = meganeura::data::safetensors::SafeTensorsModel::from_bytes(bytes)?;
        let ema = meganeura::data::safetensors::SafeTensorsModel::load(
            directory.join("ema.safetensors"),
        )?;
        check_saved_tensors(&training, &graph(config.model)?, true)?;
        check_saved_tensors(&ema, &ema_graph(config.ema_decay), false)?;
        for (name, _) in super::super::weight_shapes(ARCHITECTURE) {
            // These source tensors will share the learner's allocations.
            if training.tensor_f32(&name)? != ema.tensor_f32(&name)? {
                return Err(format!("EMA source does not match learner: {name}").into());
            }
        }
        drop((training, ema));
        let mut trainer = Self::new(config, gpu)?;
        trainer
            .session
            .load_checkpoint(&directory.join("training.safetensors"))?;
        trainer
            .ema
            .load_checkpoint(&directory.join("ema.safetensors"))?;
        if trainer.session.adam_step_count() != step {
            return Err("optimizer step was not restored".into());
        }
        trainer.step = step;
        check_memory(&trainer.session)?;
        Ok(trainer)
    }

    /// Complete model/moments and EMA state. Does not save external video sampler
    /// state; its declared seed/step mapping must be checkpointed by the adapter.
    pub fn save(&mut self, directory: &Path) -> Result<(), Error> {
        if !self.ready {
            return Err("cannot save a failed pretrainer".into());
        }
        std::fs::create_dir(directory)?;
        self.session
            .save_checkpoint(&directory.join("training.safetensors"))?;
        self.ema
            .save_checkpoint(&directory.join("ema.safetensors"))?;
        for (name, graph, moments) in [
            ("training.safetensors", graph(self.config.model)?, true),
            ("ema.safetensors", ema_graph(self.config.ema_decay), false),
        ] {
            let saved = meganeura::data::safetensors::SafeTensorsModel::load(directory.join(name))?;
            check_saved_tensors(&saved, &graph, moments)?;
        }
        let hashes: BTreeMap<_, _> = ["training.safetensors", "ema.safetensors"]
            .into_iter()
            .map(|name| {
                Ok((
                    name,
                    crate::vision::checkpoint_sha256(&directory.join(name))?,
                ))
            })
            .collect::<Result<_, std::io::Error>>()?;
        let metadata = serde_json::json!({
            "format": 1, "config": self.config, "step": self.step, "sha256": hashes,
            "meganeura": crate::dreamer::MEGANEURA_REV, "blade": crate::dreamer::BLADE_REV,
        });
        std::fs::write(
            directory.join("training.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_reaches_peak_and_final_rate_and_refuses_invalid_budget() {
        let config = TrainingConfig {
            model: Config::default(),
            seed: 7,
            steps: 100,
            warmup_steps: 10,
            learning_rate: 1e-4,
            weight_decay: 0.04,
            ema_decay: 0.999,
        };
        config.validate().unwrap();
        assert!((config.rate(1) - 1e-5).abs() < 1e-10);
        assert_eq!(config.rate(10), config.learning_rate);
        assert!((config.rate(100) - 1e-6).abs() < 1e-10);
        for steps in [0, 10] {
            assert!(TrainingConfig { steps, ..config }.validate().is_err());
        }
        assert!(
            TrainingConfig {
                ema_decay: f32::NAN,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn ema_graph_updates_every_encoder_tensor_on_device_without_projector() {
        let graph = ema_graph(0.999);
        assert_eq!(graph.outputs().len(), 149);
        for &output in graph.outputs() {
            assert!(matches!(graph.node(output).op, Op::CacheWrite));
        }
        let plan = meganeura::compile::compile(&graph);
        assert_eq!(plan.param_buffers.len(), 298);
        assert!(plan.param_grad_pairs.is_empty());
        assert!(
            !plan
                .param_buffers
                .iter()
                .any(|(name, _)| name.contains("projector"))
        );
    }

    #[test]
    fn encoder_initialization_rescales_residuals_and_is_reproducible() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut same = rng.clone();
        let regular = initial_values(
            "encoder.blocks.0.attn.qkv.weight",
            &[64, 64],
            1024,
            &mut rng,
        );
        let residual = initial_values(
            "encoder.blocks.3.attn.proj.weight",
            &[64, 64],
            1024,
            &mut same,
        );
        for (a, b) in regular.iter().zip(residual) {
            assert_eq!(*a * 8.0_f32.sqrt().recip(), b);
        }
        assert_eq!(
            initial_values("encoder.blocks.3.norm1.weight", &[5], 1024, &mut rng),
            [1.0; 5]
        );
        assert_eq!(
            initial_values("encoder.blocks.3.attn.qkv.bias", &[5], 1024, &mut rng),
            [0.0; 5]
        );
    }

    #[test]
    fn exported_linear_and_convolution_layout_inverts_native_transpose() {
        let native = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (name, shape) in [
            ("linear.weight", vec![2, 3]),
            ("encoder.patch_embed.proj.weight", vec![2, 3, 1, 1, 1]),
        ] {
            let bytes = export_bytes(name, &shape, &native);
            let values: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        assert_eq!(
            export_bytes("encoder.cls_token", &[1, 1, 6], &native),
            native
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn checkpoint_requires_complete_shapes_and_both_finite_moments() {
        use safetensors::{Dtype, tensor::TensorView};
        let mut graph = Graph::new();
        graph.parameter("weight", &[2]);
        for omitted in [
            None,
            Some("weight"),
            Some("adam_m.weight"),
            Some("adam_v.weight"),
        ] {
            let bytes = [1.0_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let views = ["weight", "adam_m.weight", "adam_v.weight"]
                .into_iter()
                .filter(|name| Some(*name) != omitted)
                .map(|name| (name, TensorView::new(Dtype::F32, vec![2], &bytes).unwrap()));
            let serialized = safetensors::tensor::serialize(views, &None).unwrap();
            let model =
                meganeura::data::safetensors::SafeTensorsModel::from_bytes(serialized).unwrap();
            assert_eq!(
                check_saved_tensors(&model, &graph, true).is_ok(),
                omitted.is_none()
            );
        }
        for (shape, values) in [(vec![1, 2], [1.0_f32, 2.0]), (vec![2], [f32::NAN, 2.0])] {
            let bytes = values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let views = [(
                "weight",
                TensorView::new(Dtype::F32, shape, &bytes).unwrap(),
            )];
            let model = meganeura::data::safetensors::SafeTensorsModel::from_bytes(
                safetensors::tensor::serialize(views, &None).unwrap(),
            )
            .unwrap();
            assert!(check_saved_tensors(&model, &graph, false).is_err());
        }
    }
}
