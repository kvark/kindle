//! DreamerV3's RGB64 encoder and decoder. Replay retains centered CHW pixels.
//! Convolutions use NCHW; normalization and encoder tokens use upstream's NHWC.

use super::*;

const SIDE: usize = 64;
const BOTTLENECK: usize = 4;
const DECODER_BLOCKS: usize = 8;

struct Convolution {
    layer: nn::Conv2d,
    bias: NodeId,
}

impl Convolution {
    fn new(graph: &mut Graph, name: &str, input: usize, output: usize, side: usize) -> Self {
        Self {
            layer: nn::Conv2d::new(
                graph,
                name,
                input as u32,
                output as u32,
                5,
                side as u32,
                side as u32,
                1,
                2,
            ),
            bias: graph.parameter(&format!("{name}.bias"), &[output]),
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let value = self.layer.forward(graph, input, batch as u32);
        graph.add_per_channel(
            value,
            self.bias,
            self.layer.out_channels,
            self.layer.in_h * self.layer.in_w,
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
        Self {
            convolution: Convolution::new(graph, name, input, output, side),
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
        let mut side = layer.in_h;
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
    channels: usize,
}

impl Encoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let depth = config.network().vision_depth;
        let mut channels = 3;
        let layers = [2, 3, 4, 4]
            .into_iter()
            .enumerate()
            .map(|(i, mult)| {
                let output = depth * mult;
                let layer = ConvNorm::new(
                    graph,
                    &format!("world.representation.encoder.cnn{i}"),
                    channels,
                    output,
                    SIDE >> i,
                );
                channels = output;
                layer
            })
            .collect();
        Self { layers, channels }
    }

    pub(super) fn output_dim(&self) -> usize {
        BOTTLENECK * BOTTLENECK * self.channels
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let mut value = graph.reshape(input, &[batch * 3 * SIDE * SIDE]);
        for layer in &self.layers {
            value = layer.forward(graph, value, batch, true);
        }
        let value = transpose_spatial(graph, value, batch, self.channels, BOTTLENECK * BOTTLENECK);
        graph.reshape(value, &[batch, self.output_dim()])
    }
}

pub(crate) struct Decoder {
    deterministic: BlockLinear,
    stochastic_hidden: LinearNorm,
    stochastic_output: nn::Linear,
    spatial_norm: nn::RmsNorm,
    layers: Vec<ConvNorm>,
    output: Convolution,
    deter: usize,
    stoch: usize,
    channels: usize,
}

impl Decoder {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig, name: &str, input: usize) -> Self {
        assert_eq!(input, config.feature_dim());
        let size = config.network();
        let channels = 4 * size.vision_depth;
        let spatial = BOTTLENECK * BOTTLENECK * channels;
        let mut input_channels = channels;
        Self {
            deterministic: BlockLinear::new(
                graph,
                &format!("{name}.sp0"),
                DECODER_BLOCKS,
                size.deter / DECODER_BLOCKS,
                spatial / DECODER_BLOCKS,
            ),
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
                        SIDE >> (i + 1),
                    );
                    input_channels = output_channels;
                    layer
                })
                .collect(),
            output: Convolution::new(
                graph,
                &format!("{name}.imgout"),
                2 * size.vision_depth,
                3,
                SIDE,
            ),
            deter: size.deter,
            stoch: size.stoch * size.classes,
            channels,
        }
    }

    pub(super) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let deter = slice_columns(graph, input, batch, self.deter + self.stoch, 0, self.deter);
        let stoch = slice_columns(
            graph,
            input,
            batch,
            self.deter + self.stoch,
            self.deter,
            self.stoch,
        );
        let deter = self.deterministic.forward(graph, deter, batch);
        // Upstream's (group, h, w, channel) output becomes NCHW directly.
        let deter = transpose_spatial(
            graph,
            deter,
            batch * DECODER_BLOCKS,
            BOTTLENECK * BOTTLENECK,
            self.channels / DECODER_BLOCKS,
        );
        let deter = graph.reshape(deter, &[batch * self.channels * BOTTLENECK * BOTTLENECK]);
        let stoch = self.stochastic_hidden.forward(graph, stoch);
        let stoch = self.stochastic_output.forward(graph, stoch);
        let stoch = transpose_spatial(graph, stoch, batch, BOTTLENECK * BOTTLENECK, self.channels);
        let stoch = graph.reshape(stoch, &[batch * self.channels * BOTTLENECK * BOTTLENECK]);
        let value = graph.add(deter, stoch);
        let mut value = normalize_image(
            graph,
            value,
            &self.spatial_norm,
            batch,
            self.channels,
            BOTTLENECK,
        );
        let mut side = BOTTLENECK;
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
        value = self.output.forward(graph, value, batch);
        value = graph.sigmoid(value);
        // Both replay and diagnostic outputs are centered. This is the same
        // summed pixel MSE as sigmoid prediction versus upstream's RGB / 255.
        value = graph.reshape(value, &[batch * 3 * SIDE * SIDE, 1]);
        let offset = graph.scalar(-0.5);
        value = graph.bias_add(value, offset);
        graph.reshape(value, &[batch, 3 * SIDE * SIDE])
    }
}

#[cfg(test)]
mod tests;
