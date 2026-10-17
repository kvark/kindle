//! Small patch-convolution autoencoder for the offline representation control.
//! Native pixels use the same GPU preprocessing as LeVJEPA. No RAM or reward.

use std::{path::Path, sync::Arc};

use meganeura::{Graph, Mode, NodeId, Session, SessionConfig, nn};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::preprocess_gpu::GpuPreprocessor;

pub const CHANNELS: usize = 64;
const GRID: usize = 14;
const PATCH: usize = 16;
const PATCH_DIM: usize = 3 * PATCH * PATCH;

/// Transpose each independent matrix, preserving the batch axis.
fn transpose_batch(g: &mut Graph, x: NodeId, batch: usize, rows: usize, cols: usize) -> NodeId {
    let x = g.reshape(x, &[batch * rows * cols]);
    let width = (rows * cols) as u32;
    let mut result = None;
    for index in 0..batch {
        let mut part = x;
        if index > 0 {
            part = g.split_b(part, 1, index as u32, (batch - index) as u32, width);
        }
        if index + 1 < batch {
            part = g.split_a(part, 1, 1, (batch - index - 1) as u32, width);
        }
        let part = g.reshape(part, &[rows, cols]);
        let part = g.transpose(part);
        let part = g.reshape(part, &[rows * cols]);
        result = Some(match result {
            None => part,
            Some(previous) => g.concat(previous, part, 1, index as u32, 1, width),
        });
    }
    result.unwrap()
}

fn graph(batch: usize, grid: usize, patch_dim: usize, channels: usize) -> Graph {
    let mut g = Graph::new();
    let patches = g.input("patches", &[batch * grid * grid, patch_dim]);
    // A stride-16, kernel-16 RGB convolution, expressed in packed patch layout.
    let stem = nn::Linear::new(&mut g, "stem", patch_dim, channels);
    let x = stem.forward(&mut g, patches);
    let x = g.relu(x);
    let mut x = transpose_batch(&mut g, x, batch, grid * grid, channels);
    for name in ["spatial0", "spatial1"] {
        let convolution = nn::Conv2d::new(
            &mut g,
            name,
            channels as u32,
            channels as u32,
            3,
            grid as u32,
            grid as u32,
            1,
            1,
        );
        x = convolution.forward(&mut g, x, batch as u32);
        x = g.relu(x);
    }
    let tokens = transpose_batch(&mut g, x, batch, channels, grid * grid);
    let tokens = g.reshape(tokens, &[batch * grid * grid, channels]);
    let decoder = nn::Linear::new(&mut g, "decoder", channels, patch_dim);
    let reconstruction = decoder.forward(&mut g, tokens);
    let loss = g.mse_loss(reconstruction, patches);
    g.set_outputs(vec![loss, tokens, reconstruction]);
    g
}

fn initialize(session: &mut Session, patch_dim: usize, channels: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for (name, fan_in, fan_out) in [
        ("stem", patch_dim, channels),
        ("spatial0", 9 * channels, 9 * channels),
        ("spatial1", 9 * channels, 9 * channels),
        ("decoder", channels, patch_dim),
    ] {
        let weight = format!("{name}.weight");
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let values = (0..session.param_size(&weight).unwrap())
            .map(|_| rng.random_range(-bound..bound))
            .collect::<Vec<_>>();
        session.set_parameter(&weight, &values);
        let bias = format!("{name}.bias");
        if let Some(size) = session.param_size(&bias) {
            session.set_parameter(&bias, &vec![0.0; size]);
        }
    }
}

pub struct ReconstructionEncoder {
    pixels: GpuPreprocessor,
    session: Session,
    batch: usize,
}

impl ReconstructionEncoder {
    pub fn new(batch: usize, seed: u64) -> Result<Self, Box<dyn std::error::Error>> {
        if batch == 0 || batch > 64 {
            return Err("reconstruction batch must be in 1..=64".into());
        }
        let gpu = Arc::new(crate::init_gpu_context()?);
        let mut session = meganeura::build(
            &graph(batch, GRID, PATCH_DIM, CHANNELS),
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(Arc::clone(&gpu)),
                ..Default::default()
            },
        )
        .0;
        initialize(&mut session, PATCH_DIM, CHANNELS, seed);
        Ok(Self {
            pixels: GpuPreprocessor::new(gpu, batch, GRID * PATCH, PATCH),
            session,
            batch,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    pub fn gpu_memory_budget(&self) -> Option<crate::GpuMemoryBudget> {
        self.session
            .device_memory_stats()
            .map(|s| crate::GpuMemoryBudget {
                usage_bytes: s.usage_bytes,
                budget_bytes: s.budget_bytes,
            })
    }

    /// Fixed-size complete batches; evaluation computes no optimizer update.
    pub fn process(
        &mut self,
        frames: &[crate::RgbFrame],
        learning_rate: Option<f32>,
    ) -> Result<f32, &'static str> {
        if frames.len() != self.batch
            || learning_rate.is_some_and(|lr| !lr.is_finite() || lr <= 0.0)
        {
            return Err("wrong reconstruction batch or learning rate");
        }
        let frames = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| (index, frame.pixels(), frame.width(), frame.height()))
            .collect::<Vec<_>>();
        self.pixels.cpu_frames(&mut self.session, &frames);
        self.session.step();
        self.session.wait();
        let loss = self.session.read_loss();
        if !loss.is_finite() {
            return Err("nonfinite reconstruction loss");
        }
        if let Some(lr) = learning_rate {
            self.session.adam_step(lr, 0.9, 0.999, 1e-8);
            self.session.wait();
        }
        Ok(loss)
    }

    pub fn tokens(&self) -> Vec<f32> {
        let mut values = vec![0.0; self.batch * GRID * GRID * CHANNELS];
        self.session.read_output_by_index(1, &mut values);
        values
    }

    pub fn save(&mut self, path: &Path) -> std::io::Result<()> {
        self.session.save_checkpoint(path)
    }
    pub fn load(&mut self, path: &Path) -> std::io::Result<()> {
        self.session.load_checkpoint(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_batch_precedes_gpu_initialization() {
        assert!(ReconstructionEncoder::new(0, 0).is_err());
        assert!(ReconstructionEncoder::new(65, 0).is_err());
    }

    #[test]
    fn batched_layout_is_differentiable_before_gpu_work() {
        for (batch, grid, patch_dim, channels) in [(2, 2, 2, 2), (16, GRID, PATCH_DIM, CHANNELS)] {
            let forward = graph(batch, grid, patch_dim, channels);
            let backward = meganeura::autodiff::differentiate(&forward);
            assert!(backward.nodes().len() > forward.nodes().len());
        }
    }

    // Independent NHWC scalar reference for two 3x3 spatial convolutions.
    fn reference(patches: &[f32], parameters: &[Vec<f32>]) -> (Vec<f64>, f64) {
        let [stem, stem_bias, first, second, decoder, decoder_bias] = parameters else {
            panic!()
        };
        let mut tokens = vec![0.0_f64; 2 * 4 * 2];
        for (input, output) in patches
            .as_chunks::<2>()
            .0
            .iter()
            .zip(tokens.as_chunks_mut::<2>().0)
        {
            for channel in 0..2 {
                output[channel] = (f64::from(stem_bias[channel])
                    + (0..2)
                        .map(|c| f64::from(input[c]) * f64::from(stem[c * 2 + channel]))
                        .sum::<f64>())
                .max(0.0);
            }
        }
        for weights in [first, second] {
            let mut next = vec![0.0; tokens.len()];
            for batch in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        for output in 0..2 {
                            let mut value = 0.0;
                            for input in 0..2 {
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let iy = y as isize + ky as isize - 1;
                                        let ix = x as isize + kx as isize - 1;
                                        if (0..2).contains(&iy) && (0..2).contains(&ix) {
                                            let source =
                                                ((batch * 2 + iy as usize) * 2 + ix as usize) * 2
                                                    + input;
                                            value += tokens[source]
                                                * f64::from(
                                                    weights
                                                        [((output * 2 + input) * 3 + ky) * 3 + kx],
                                                );
                                        }
                                    }
                                }
                            }
                            next[((batch * 2 + y) * 2 + x) * 2 + output] = value.max(0.0);
                        }
                    }
                }
            }
            tokens = next;
        }
        let mut loss = 0.0;
        for (input, target) in tokens
            .as_chunks::<2>()
            .0
            .iter()
            .zip(patches.as_chunks::<2>().0)
        {
            for channel in 0..2 {
                let value = f64::from(decoder_bias[channel])
                    + (0..2)
                        .map(|c| input[c] * f64::from(decoder[c * 2 + channel]))
                        .sum::<f64>();
                loss += (value - f64::from(target[channel])).powi(2) / patches.len() as f64;
            }
        }
        (tokens, loss)
    }

    #[test]
    #[ignore = "requires GPU; independent patch-CNN values and finite-difference gradients"]
    fn patch_convolutions_match_scalar_reference() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut session = meganeura::build(
            &graph(2, 2, 2, 2),
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(gpu),
                ..Default::default()
            },
        )
        .0;
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            let device = crate::gpu_device_info(session.device_information());
            assert_eq!(device.device_name, expected);
            assert!(!device.is_software_emulated);
            let memory = session.device_memory_stats().expect("Vulkan memory budget");
            assert!(memory.budget_bytes - memory.usage_bytes >= 2 << 30);
            eprintln!("reconstruction control device={device:?} memory={memory:?}");
        }
        let names = [
            "stem.weight",
            "stem.bias",
            "spatial0.weight",
            "spatial1.weight",
            "decoder.weight",
            "decoder.bias",
        ];
        let mut parameters = names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                (0..session.param_size(name).unwrap())
                    .map(|i| 0.01 * ((i * 3 + index) % 11 + 1) as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (name, values) in names.iter().zip(&parameters) {
            session.set_parameter(name, values);
        }
        let patches = (0..16).map(|i| (i + 1) as f32 * 0.03).collect::<Vec<_>>();
        session.set_input("patches", &patches);
        session.step();
        session.wait();
        let (expected, loss) = reference(&patches, &parameters);
        let mut actual = vec![0.0; expected.len()];
        session.read_output_by_index(1, &mut actual);
        for (a, b) in actual.iter().zip(&expected) {
            assert!((f64::from(*a) - b).abs() < 1e-5);
        }
        assert!((f64::from(session.read_loss()) - loss).abs() < 1e-5);
        for (index, name) in names.iter().enumerate() {
            let mut gradient = vec![0.0; parameters[index].len()];
            session.read_param_grad(name, &mut gradient);
            for lane in [0, gradient.len() / 2, gradient.len() - 1] {
                let original = parameters[index][lane];
                parameters[index][lane] = original + 1e-3;
                let plus = reference(&patches, &parameters).1;
                let high = parameters[index][lane];
                parameters[index][lane] = original - 1e-3;
                let minus = reference(&patches, &parameters).1;
                let low = parameters[index][lane];
                parameters[index][lane] = original;
                let expected = (plus - minus) / f64::from(high - low);
                assert!(
                    (f64::from(gradient[lane]) - expected).abs() < 2e-5,
                    "{name}[{lane}]: {} vs {expected}",
                    gradient[lane]
                );
            }
        }
        for _ in 0..50 {
            session.step();
            session.wait();
            session.adam_step(0.003, 0.9, 0.999, 1e-8);
            session.wait();
        }
        session.step();
        session.wait();
        assert!(f64::from(session.read_loss()) < loss * 0.4);
        let weights = session.read_params(&names);
        let moments = session.read_adam_states(&names);
        let age = session.adam_step_count();
        session.step();
        session.wait();
        assert_eq!(weights, session.read_params(&names));
        assert_eq!(moments, session.read_adam_states(&names));
        assert_eq!(age, session.adam_step_count());
    }
}
