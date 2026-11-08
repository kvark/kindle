//! DreamerV3's RGB64 encoder and decoder. Replay retains centered CHW pixels.
//! Convolutions use NCHW; normalization and encoder tokens use upstream's NHWC.

use super::*;

const DECODER_BLOCKS: usize = 8;

struct Convolution {
    layer: nn::Conv2d,
    bias: NodeId,
    output_side: u32,
}

impl Convolution {
    fn new(graph: &mut Graph, name: &str, input: usize, output: usize, side: usize) -> Self {
        Self::with_stride(graph, name, input, output, side, 1)
    }

    fn with_stride(
        graph: &mut Graph,
        name: &str,
        input: usize,
        output: usize,
        side: usize,
        stride: usize,
    ) -> Self {
        Self {
            layer: nn::Conv2d::new(
                graph,
                name,
                input as u32,
                output as u32,
                5,
                side as u32,
                side as u32,
                stride as u32,
                2,
            ),
            bias: graph.parameter(&format!("{name}.bias"), &[output]),
            output_side: side.div_ceil(stride) as u32,
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let value = self.layer.forward(graph, input, batch as u32);
        graph.add_per_channel(
            value,
            self.bias,
            self.layer.out_channels,
            self.output_side * self.output_side,
        )
    }
}

fn transpose_spatial(
    graph: &mut Graph,
    value: NodeId,
    batch: usize,
    rows: usize,
    columns: usize,
) -> NodeId {
    let value = graph.reshape(value, &[batch, rows, columns]);
    graph.transpose(value)
}

fn normalize_image(
    graph: &mut Graph,
    value: NodeId,
    norm: &nn::RmsNorm,
    batch: usize,
    channels: usize,
    side: usize,
) -> NodeId {
    let value = transpose_spatial(graph, value, batch, channels, side * side);
    let value = graph.reshape(value, &[batch * side * side, channels]);
    let value = norm.forward(graph, value);
    let value = graph.silu(value);
    let value = transpose_spatial(graph, value, batch, side * side, channels);
    graph.reshape(value, &[batch * channels * side * side])
}

struct ConvNorm {
    convolution: Convolution,
    norm: nn::RmsNorm,
}

impl ConvNorm {
    fn new(graph: &mut Graph, name: &str, input: usize, output: usize, side: usize) -> Self {
        Self::with_stride(graph, name, input, output, side, 1)
    }

    fn with_stride(
        graph: &mut Graph,
        name: &str,
        input: usize,
        output: usize,
        side: usize,
        stride: usize,
    ) -> Self {
        Self {
            convolution: Convolution::with_stride(graph, name, input, output, side, stride),
            norm: nn::RmsNorm::new(
                graph,
                &format!("{name}.norm.weight"),
                output,
                DREAMER_NORM_EPSILON,
            ),
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize, pool: bool) -> NodeId {
        let layer = &self.convolution.layer;
        let mut value = self.convolution.forward(graph, input, batch);
        let mut side = self.convolution.output_side;
        if pool {
            value = graph.max_pool_2d(
                value,
                batch as u32,
                layer.out_channels,
                side,
                side,
                2,
                2,
                2,
                0,
            );
            side /= 2;
        }
        normalize_image(
            graph,
            value,
            &self.norm,
            batch,
            layer.out_channels as usize,
            side as usize,
        )
    }
}

pub(crate) struct Encoder {
    layers: Vec<ConvNorm>,
    input_channels: usize,
    coordinates: bool,
    stride2: bool,
    channels: usize,
    side: usize,
}

impl Encoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let depth = config.network().vision_depth;
        let side = config.observation_grid();
        let input_channels =
            config.observation_channels() * if config.rgb_previous_frame { 2 } else { 1 };
        let mut channels = input_channels + if config.rgb_coordinates { 2 } else { 0 };
        let layers = [2, 3, 4, 4]
            .into_iter()
            .enumerate()
            .map(|(i, mult)| {
                let output = depth * mult;
                let layer = ConvNorm::with_stride(
                    graph,
                    &format!("world.representation.encoder.cnn{i}"),
                    channels,
                    output,
                    side >> i,
                    if config.rgb_stride2 { 2 } else { 1 },
                );
                channels = output;
                layer
            })
            .collect();
        Self {
            layers,
            input_channels,
            coordinates: config.rgb_coordinates,
            stride2: config.rgb_stride2,
            channels,
            side,
        }
    }

    pub(super) fn output_dim(&self) -> usize {
        (self.side / 16) * (self.side / 16) * self.channels
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let mut value = graph.reshape(
            input,
            &[batch * self.input_channels * self.side * self.side],
        );
        if self.coordinates {
            let width = self.side * self.side;
            let positions = graph.constant(
                coordinate_planes(self.side).repeat(batch),
                &[batch, 2 * width],
            );
            value = concat_columns(
                graph,
                value,
                positions,
                batch,
                self.input_channels * width,
                2 * width,
            );
            value = graph.reshape(value, &[batch * (self.input_channels + 2) * width]);
        }
        for layer in &self.layers {
            value = layer.forward(graph, value, batch, !self.stride2);
        }
        let value = transpose_spatial(
            graph,
            value,
            batch,
            self.channels,
            (self.side / 16) * (self.side / 16),
        );
        graph.reshape(value, &[batch, self.output_dim()])
    }
}

fn coordinate_planes(side: usize) -> Vec<f32> {
    (0..2 * side * side)
        .map(|index| {
            let coordinate = if index < side * side {
                index % side
            } else {
                (index / side) % side
            };
            (coordinate as f32 + 0.5) / side as f32 - 0.5
        })
        .collect()
}

pub(crate) struct Decoder {
    deterministic: Option<BlockLinear>,
    stochastic_hidden: LinearNorm,
    stochastic_output: nn::Linear,
    spatial_norm: nn::RmsNorm,
    layers: Vec<ConvNorm>,
    output: Convolution,
    motion: Vec<ConvNorm>,
    fusion: Option<Box<ConvNorm>>,
    residual: bool,
    deter: usize,
    stoch: usize,
    channels: usize,
    side: usize,
}

impl Decoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig, name: &str, input: usize) -> Self {
        assert_eq!(input, config.feature_dim());
        let size = config.network();
        let side = config.observation_grid();
        let channels = 4 * size.vision_depth;
        let spatial = (side / 16) * (side / 16) * channels;
        let mut input_channels = channels;
        Self {
            deterministic: (!config.rgb_latent_only).then(|| {
                BlockLinear::new(
                    graph,
                    &format!("{name}.sp0"),
                    DECODER_BLOCKS,
                    size.deter / DECODER_BLOCKS,
                    spatial / DECODER_BLOCKS,
                )
            }),
            stochastic_hidden: LinearNorm::new(
                graph,
                &format!("{name}.sp1"),
                size.stoch * size.classes,
                2 * size.units,
            ),
            stochastic_output: nn::Linear::new(
                graph,
                &format!("{name}.sp2"),
                2 * size.units,
                spatial,
            ),
            spatial_norm: nn::RmsNorm::new(
                graph,
                &format!("{name}.spatial.norm.weight"),
                channels,
                DREAMER_NORM_EPSILON,
            ),
            layers: [2, 3, 4]
                .into_iter()
                .enumerate()
                .rev()
                .map(|(i, mult)| {
                    let output_channels = mult * size.vision_depth;
                    let layer = ConvNorm::new(
                        graph,
                        &format!("{name}.conv{i}"),
                        input_channels,
                        output_channels,
                        side >> (i + 1),
                    );
                    input_channels = output_channels;
                    layer
                })
                .collect(),
            output: Convolution::new(
                graph,
                &format!("{name}.imgout"),
                if config.rgb_spatial_fusion {
                    16
                } else {
                    2 * size.vision_depth + if config.rgb_spatial_decoder { 8 } else { 0 }
                },
                3,
                side,
            ),
            motion: if config.rgb_spatial_decoder {
                (0..3)
                    .map(|i| {
                        ConvNorm::new(
                            graph,
                            &format!("{name}.motion{i}"),
                            if i == 0 { 6 } else { 8 },
                            8,
                            side,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            },
            fusion: config.rgb_spatial_fusion.then(|| {
                Box::new(ConvNorm::new(
                    graph,
                    &format!("{name}.fusion"),
                    2 * size.vision_depth + 8,
                    16,
                    side,
                ))
            }),
            residual: config.rgb_spatial_residual,
            deter: size.deter,
            stoch: size.stoch * size.classes,
            channels,
            side,
        }
    }

    pub(super) fn forward_logits(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        self.forward_logits_with_history(graph, input, None, batch)
    }

    pub(super) fn forward_logits_with_history(
        &self,
        graph: &mut Graph,
        input: NodeId,
        history: Option<NodeId>,
        batch: usize,
    ) -> NodeId {
        assert_eq!(history.is_none(), self.motion.is_empty());
        let stoch = slice_columns(
            graph,
            input,
            batch,
            self.deter + self.stoch,
            self.deter,
            self.stoch,
        );
        let deter = self.deterministic.as_ref().map(|layer| {
            let deter = slice_columns(graph, input, batch, self.deter + self.stoch, 0, self.deter);
            let deter = layer.forward(graph, deter, batch);
            // Upstream's (group, h, w, channel) output becomes NCHW directly.
            let deter = transpose_spatial(
                graph,
                deter,
                batch * DECODER_BLOCKS,
                (self.side / 16) * (self.side / 16),
                self.channels / DECODER_BLOCKS,
            );
            graph.reshape(
                deter,
                &[batch * self.channels * (self.side / 16) * (self.side / 16)],
            )
        });
        let stoch = self.stochastic_hidden.forward(graph, stoch);
        let stoch = self.stochastic_output.forward(graph, stoch);
        let stoch = transpose_spatial(
            graph,
            stoch,
            batch,
            (self.side / 16) * (self.side / 16),
            self.channels,
        );
        let stoch = graph.reshape(
            stoch,
            &[batch * self.channels * (self.side / 16) * (self.side / 16)],
        );
        let value = if let Some(deter) = deter {
            graph.add(deter, stoch)
        } else {
            stoch
        };
        let mut value = normalize_image(
            graph,
            value,
            &self.spatial_norm,
            batch,
            self.channels,
            self.side / 16,
        );
        let mut side = self.side / 16;
        let mut channels = self.channels;
        for layer in &self.layers {
            value = graph.upsample_2x(
                value,
                batch as u32,
                channels as u32,
                side as u32,
                side as u32,
            );
            value = layer.forward(graph, value, batch, false);
            side *= 2;
            channels = layer.convolution.layer.out_channels as usize;
        }
        value = graph.upsample_2x(
            value,
            batch as u32,
            channels as u32,
            side as u32,
            side as u32,
        );
        if let Some(history) = history {
            let mut motion = graph.reshape(history, &[batch * 6 * self.side * self.side]);
            for layer in &self.motion {
                motion = layer.forward(graph, motion, batch, false);
            }
            value = concat_columns(
                graph,
                value,
                motion,
                batch,
                channels * self.side * self.side,
                8 * self.side * self.side,
            );
            value = graph.reshape(value, &[batch * (channels + 8) * self.side * self.side]);
            if let Some(fusion) = &self.fusion {
                value = fusion.forward(graph, value, batch, false);
            }
        }
        value = self.output.forward(graph, value, batch);
        let width = 3 * self.side * self.side;
        let logits = graph.reshape(value, &[batch, width]);
        if self.residual {
            residual_logits(graph, logits, history.unwrap(), batch, width)
        } else {
            logits
        }
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let mut value = self.forward_logits(graph, input, batch);
        value = graph.sigmoid(value);
        // Both replay and diagnostic outputs are centered. This is the same
        // summed pixel MSE as sigmoid prediction versus upstream's RGB / 255.
        value = graph.reshape(value, &[batch * 3 * self.side * self.side, 1]);
        let offset = graph.scalar(-0.5);
        value = graph.bias_add(value, offset);
        graph.reshape(value, &[batch, 3 * self.side * self.side])
    }
}

fn residual_logits(
    graph: &mut Graph,
    correction: NodeId,
    history: NodeId,
    batch: usize,
    width: usize,
) -> NodeId {
    let latest = slice_columns(graph, history, batch, 2 * width, width, width);
    let latest = graph.reshape(latest, &[batch * width, 1]);
    let half = graph.scalar(0.5);
    let probability = graph.bias_add(latest, half);
    let probability = graph.clamp(probability, 1.0 / 255.0, 254.0 / 255.0);
    let positive = graph.log(probability);
    let negative = graph.neg(probability);
    let one = graph.scalar(1.0);
    let negative = graph.bias_add(negative, one);
    let negative = graph.log(negative);
    let logits = graph.sub(positive, negative);
    let logits = graph.reshape(logits, &[batch, width]);
    graph.add(correction, logits)
}

#[cfg(test)]
mod fusion_tests;
#[cfg(test)]
mod residual_tests;
#[cfg(test)]
mod stride_tests;
#[cfg(test)]
mod tests;
