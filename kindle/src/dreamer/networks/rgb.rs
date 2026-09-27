//! Small jointly learned RGB control. Replay keeps pixels, not encoder outputs.
//! This patch CNN and dense decoder are deliberately not the upstream CNN pair.

use super::*;

struct ConvNorm {
    convolution: nn::Conv2d,
    weight: NodeId,
    bias: NodeId,
    channels: usize,
}

impl ConvNorm {
    fn new(graph: &mut Graph, name: &str, input: usize, output: usize, stem: bool) -> Self {
        Self {
            convolution: nn::Conv2d::new(
                graph,
                name,
                input as u32,
                output as u32,
                if stem { 8 } else { 3 },
                if stem { 64 } else { 8 },
                if stem { 64 } else { 8 },
                if stem { 8 } else { 1 },
                if stem { 0 } else { 1 },
            ),
            weight: graph.parameter(&format!("{name}.norm.weight"), &[output]),
            bias: graph.parameter(&format!("{name}.norm.bias"), &[output]),
            channels: output,
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let value = self.convolution.forward(graph, input, batch as u32);
        let value = graph.group_norm(
            value,
            self.weight,
            self.bias,
            batch as u32,
            self.channels as u32,
            8 * 8,
            8,
            DREAMER_NORM_EPSILON,
        );
        graph.silu(value)
    }
}

pub(crate) struct Encoder {
    layers: [ConvNorm; 3],
    channels: usize,
}

impl Encoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let channels = 4 * config.network().vision_depth;
        let prefix = "world.representation.encoder";
        Self {
            layers: [
                ConvNorm::new(graph, &format!("{prefix}.stem"), 3, channels, true),
                ConvNorm::new(
                    graph,
                    &format!("{prefix}.spatial0"),
                    channels,
                    channels,
                    false,
                ),
                ConvNorm::new(
                    graph,
                    &format!("{prefix}.spatial1"),
                    channels,
                    channels,
                    false,
                ),
            ],
            channels,
        }
    }

    pub(super) fn output_dim(&self) -> usize {
        4 * 4 * self.channels
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let mut value = graph.reshape(input, &[batch * 3 * 64 * 64]);
        for layer in &self.layers {
            value = layer.forward(graph, value, batch);
        }
        let value = graph.max_pool_2d(value, batch as u32, self.channels as u32, 8, 8, 2, 2, 2, 0);
        graph.reshape(value, &[batch, self.output_dim()])
    }
}

pub(crate) struct Decoder {
    hidden: LinearNorm,
    output: nn::Linear,
}

impl Decoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig, name: &str, input: usize) -> Self {
        let units = config.network().units;
        Self {
            hidden: LinearNorm::new(graph, &format!("{name}.trunk"), input, units),
            output: nn::Linear::new(graph, &format!("{name}.out"), units, 3 * 64 * 64),
        }
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let hidden = self.hidden.forward(graph, input);
        self.output.forward(graph, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dreamer::runtime::{build_session, initialize_d3};
    use std::{collections::HashMap, sync::Arc};

    #[test]
    fn rgb_graphs_keep_pixels_and_joint_encoder_parameters() {
        for size in [crate::ModelSize::Tiny, crate::ModelSize::Size12M] {
            let mut config = DreamerConfig::tiny(3);
            config.model_size = size;
            config.observation_kind = ObservationKind::Rgb64;
            let graph = crate::dreamer::world::build_training_graph(&config, 4);
            let backward = meganeura::autodiff::differentiate(&graph);
            assert!(backward.nodes().len() > graph.nodes().len());
            let mut graph = Graph::new();
            let encoder = Encoder::new(&mut graph, &config);
            assert_eq!(encoder.output_dim(), config.encoded_observation_dim());
            let input = graph.input("pixels", &config.observation_shape(2));
            let encoded = encoder.forward(&mut graph, input, 2);
            assert_eq!(
                graph.node(encoded).ty.shape,
                [2, config.encoded_observation_dim()]
            );
            assert_eq!(config.observation_dim(), 12288);
            assert!(graph.nodes().iter().any(|n| matches!(&n.op, meganeura::graph::Op::Parameter { name } if name == "world.representation.encoder.stem.weight")));
        }
    }

    // Independent scalar NCHW convolution, group normalization, SiLU and pooling.
    fn reference(input: &[f32], parameters: &HashMap<String, Vec<f32>>) -> Vec<f64> {
        let mut values = input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let (mut width, mut channels) = (64, 3);
        for (layer, kernel, stride, padding) in [
            ("stem", 8, 8, 0),
            ("spatial0", 3, 1, 1),
            ("spatial1", 3, 1, 1),
        ] {
            let weights = &parameters[&format!("world.representation.encoder.{layer}.weight")];
            let scales = &parameters[&format!("world.representation.encoder.{layer}.norm.weight")];
            let biases = &parameters[&format!("world.representation.encoder.{layer}.norm.bias")];
            let mut next = vec![0.0; 2 * 16 * 8 * 8];
            for b in 0..2 {
                for output in 0..16 {
                    for y in 0..8 {
                        for x in 0..8 {
                            let mut sum = 0.0;
                            for c in 0..channels {
                                for ky in 0..kernel {
                                    for kx in 0..kernel {
                                        let iy = (y * stride + ky) as isize - padding;
                                        let ix = (x * stride + kx) as isize - padding;
                                        if (0..width as isize).contains(&iy)
                                            && (0..width as isize).contains(&ix)
                                        {
                                            sum += values[((b * channels + c) * width
                                                + iy as usize)
                                                * width
                                                + ix as usize]
                                                * f64::from(
                                                    weights[((output * channels + c) * kernel
                                                        + ky)
                                                        * kernel
                                                        + kx],
                                                );
                                        }
                                    }
                                }
                            }
                            next[((b * 16 + output) * 8 + y) * 8 + x] = sum;
                        }
                    }
                }
                for group in 0..8 {
                    let start = (b * 16 + group * 2) * 64;
                    let data = &mut next[start..start + 128];
                    let mean = data.iter().sum::<f64>() / 128.0;
                    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 128.0;
                    for (i, x) in data.iter_mut().enumerate() {
                        let c = group * 2 + i / 64;
                        let norm =
                            (*x - mean) / (variance + f64::from(DREAMER_NORM_EPSILON)).sqrt();
                        let affine = norm * f64::from(scales[c]) + f64::from(biases[c]);
                        *x = affine / (1.0 + (-affine).exp());
                    }
                }
            }
            values = next;
            width = 8;
            channels = 16;
        }
        let mut output = Vec::new();
        for bc in 0..32 {
            for y in 0..4 {
                for x in 0..4 {
                    let offset = bc * 64 + y * 16 + x * 2;
                    output.push(
                        [offset, offset + 1, offset + 8, offset + 9]
                            .into_iter()
                            .map(|i| values[i])
                            .fold(f64::NEG_INFINITY, f64::max),
                    );
                }
            }
        }
        output
    }

    #[test]
    #[ignore = "requires GPU; scalar CNN values and independent finite-difference gradients"]
    fn tiny_rgb_encoder_matches_independent_reference() {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb64;
        let mut graph = Graph::new();
        let encoder = Encoder::new(&mut graph, &config);
        let input = graph.input("pixels", &[2, 12288]);
        let output = encoder.forward(&mut graph, input, 2);
        let squared = graph.mul(output, output);
        let loss = graph.mean_all(squared);
        graph.set_outputs(vec![loss, output]);
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        initialize_d3(&mut session, &graph, 103);
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            assert_eq!(session.device_information().device_name, expected);
            assert!(!session.device_information().is_software_emulated);
            let memory = session.device_memory_stats().unwrap();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        }
        let names = session
            .param_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut parameters = HashMap::new();
        for name in &names {
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_param(name, &mut values);
            if name.ends_with("stem.weight") {
                assert!(
                    values
                        .iter()
                        .all(|x| x.abs() <= 2.0 * 1.1368 / 192_f32.sqrt())
                );
            }
            parameters.insert(name.clone(), values);
        }
        let pixels = (0..2 * 12288)
            .map(|i| ((i * 37 + i / 43) % 256) as f32 / 255.0 - 0.5)
            .collect::<Vec<_>>();
        session.set_input("pixels", &pixels);
        session.step();
        session.wait();
        let expected = reference(&pixels, &parameters);
        let mut actual = vec![0.0; expected.len()];
        session.read_output_by_index(1, &mut actual);
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((f64::from(a) - b).abs() < 3e-4, "{a} != {b}");
        }
        let scalar_loss = |p: &HashMap<String, Vec<f32>>| {
            let values = reference(&pixels, p);
            values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64
        };
        assert!((f64::from(session.read_loss()) - scalar_loss(&parameters)).abs() < 3e-4);
        for name in &names {
            let mut gradient = vec![0.0; parameters[name].len()];
            session.read_param_grad(name, &mut gradient);
            let lane = (0..gradient.len())
                .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
                .unwrap();
            assert!(gradient[lane].is_finite() && gradient[lane].abs() > 1e-6);
            let original = parameters[name][lane];
            parameters.get_mut(name).unwrap()[lane] = original + 1e-4;
            let plus = scalar_loss(&parameters);
            let high = parameters[name][lane];
            parameters.get_mut(name).unwrap()[lane] = original - 1e-4;
            let minus = scalar_loss(&parameters);
            let low = parameters[name][lane];
            parameters.get_mut(name).unwrap()[lane] = original;
            let expected = (plus - minus) / f64::from(high - low);
            assert!(
                (f64::from(gradient[lane]) - expected).abs() < 3e-4 + expected.abs() * 0.003,
                "{name}: {} != {expected}",
                gradient[lane]
            );
        }
    }
}
