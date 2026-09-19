//! Reusable graph pieces for the DreamerV3 model.

use meganeura::{Graph, graph::NodeId, nn};

use super::config::DreamerConfig;
use crate::vision::{OBSERVATION_CHANNELS, OBSERVATION_GRID};

const DREAMER_NORM_EPSILON: f32 = 1e-4;

pub(crate) fn concat_columns(
    graph: &mut Graph,
    left: NodeId,
    right: NodeId,
    batch: usize,
    left_dim: usize,
    right_dim: usize,
) -> NodeId {
    let left = graph.reshape(left, &[batch * left_dim]);
    let right = graph.reshape(right, &[batch * right_dim]);
    let joined = graph.concat(
        left,
        right,
        batch as u32,
        left_dim as u32,
        right_dim as u32,
        1,
    );
    graph.reshape(joined, &[batch, left_dim + right_dim])
}

pub(crate) fn slice_columns(
    graph: &mut Graph,
    input: NodeId,
    batch: usize,
    total: usize,
    start: usize,
    length: usize,
) -> NodeId {
    assert!(length > 0 && start + length <= total);
    if start == 0 && length == total {
        return graph.reshape(input, &[batch, length]);
    }
    let input = graph.reshape(input, &[batch * total]);
    let suffix = if start == 0 {
        input
    } else {
        graph.split_b(input, batch as u32, start as u32, (total - start) as u32, 1)
    };
    let remaining = total - start;
    let result = if length == remaining {
        suffix
    } else {
        graph.split_a(
            suffix,
            batch as u32,
            length as u32,
            (remaining - length) as u32,
            1,
        )
    };
    graph.reshape(result, &[batch, length])
}

pub(crate) fn scale(graph: &mut Graph, value: NodeId, coefficient: f32) -> NodeId {
    let coefficient = graph.scalar(coefficient);
    graph.mul(value, coefficient)
}

pub(crate) fn sum(graph: &mut Graph, values: &[NodeId]) -> NodeId {
    assert!(!values.is_empty());
    let mut result = values[0];
    for value in &values[1..] {
        result = graph.add(result, *value);
    }
    result
}

pub(crate) fn weighted_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    weight: NodeId,
) -> NodeId {
    let log_probability = graph.log_softmax(logits);
    let product = graph.mul(target, log_probability);
    let per_row = graph.sum_inner(product);
    let per_row = graph.neg(per_row);
    let weighted = graph.mul(per_row, weight);
    graph.mean_all(weighted)
}

// Upstream `nn.Norm("rms")` applies a learned scale but no shift. Its `shift`
// option is only read by the layer-normalization branch.
struct LinearNorm {
    linear: nn::Linear,
    norm: nn::RmsNorm,
}

impl LinearNorm {
    fn new(graph: &mut Graph, name: &str, input: usize, output: usize) -> Self {
        Self {
            linear: nn::Linear::new(graph, name, input, output),
            norm: nn::RmsNorm::new(
                graph,
                &format!("{name}.norm.weight"),
                output,
                DREAMER_NORM_EPSILON,
            ),
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let value = self.linear.forward(graph, input);
        let value = self.norm.forward(graph, value);
        graph.silu(value)
    }

    /// Evaluate with fixed parameters while retaining gradients with respect
    /// to the input. This lets the replay critic shape the RSSM state without
    /// giving two independent optimizers ownership of critic parameters.
    fn forward_frozen(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let weight = graph.stop_gradient(self.linear.weight);
        let value = graph.matmul(input, weight);
        let value = if let Some(bias) = self.linear.bias {
            let bias = graph.stop_gradient(bias);
            graph.bias_add(value, bias)
        } else {
            value
        };
        let norm_weight = graph.stop_gradient(self.norm.weight);
        let value = graph.rms_norm(value, norm_weight, self.norm.eps);
        graph.silu(value)
    }
}

/// D3 BlockLinear expressed with one grouped weight and bias parameter.
/// Individual block views feed ordinary matmuls, preserving upstream's
/// parameter leaves for initialization, AGC, and checkpointing.
struct BlockLinear {
    weight: NodeId,
    bias: NodeId,
    blocks: usize,
    input_per_block: usize,
    output_per_block: usize,
}

impl BlockLinear {
    fn new(
        graph: &mut Graph,
        name: &str,
        blocks: usize,
        input_per_block: usize,
        output_per_block: usize,
    ) -> Self {
        Self {
            weight: graph.parameter(
                &format!("{name}.weight"),
                &[blocks, input_per_block, output_per_block],
            ),
            bias: graph.parameter(&format!("{name}.bias"), &[blocks * output_per_block]),
            blocks,
            input_per_block,
            output_per_block,
        }
    }

    fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        // Keep batch-one GEMV and large-batch cooperative imagination on
        // their original arithmetic. This candidate targets small F32 batches.
        if (2..=16).contains(&batch) {
            assert_eq!(
                graph.node(input).ty.shape,
                [batch, self.blocks * self.input_per_block]
            );
            let value = graph.block_matmul(input, self.weight);
            return graph.bias_add(value, self.bias);
        }
        self.forward_serial(graph, input, batch)
    }

    fn forward_serial(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let total_input = self.input_per_block * self.blocks;
        let weight_size = self.input_per_block * self.output_per_block;
        let mut outputs = Vec::with_capacity(self.blocks);
        for block in 0..self.blocks {
            let block_input = slice_columns(
                graph,
                input,
                batch,
                total_input,
                block * self.input_per_block,
                self.input_per_block,
            );
            let weight = slice_columns(
                graph,
                self.weight,
                1,
                self.blocks * weight_size,
                block * weight_size,
                weight_size,
            );
            let weight = graph.reshape(weight, &[self.input_per_block, self.output_per_block]);
            let value = graph.matmul(block_input, weight);
            let bias = slice_columns(
                graph,
                self.bias,
                1,
                self.blocks * self.output_per_block,
                block * self.output_per_block,
                self.output_per_block,
            );
            let bias = graph.reshape(bias, &[self.output_per_block]);
            outputs.push(graph.bias_add(value, bias));
        }
        let mut joined = outputs[0];
        let mut width = self.output_per_block;
        for output in &outputs[1..] {
            joined = concat_columns(graph, joined, *output, batch, width, self.output_per_block);
            width += self.output_per_block;
        }
        joined
    }
}

pub(crate) struct RssmCore {
    deter_input: LinearNorm,
    stoch_input: LinearNorm,
    action_input: LinearNorm,
    hidden: BlockLinear,
    hidden_norm: nn::RmsNorm,
    gru: BlockLinear,
    deter: usize,
    hidden_width: usize,
    stoch_width: usize,
    action_count: usize,
    blocks: usize,
}

impl RssmCore {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let size = config.network();
        let block_deter = size.deter / size.blocks;
        let block_input = block_deter + 3 * size.hidden;
        Self {
            deter_input: LinearNorm::new(
                graph,
                "world.dynamics.core.dynin0",
                size.deter,
                size.hidden,
            ),
            stoch_input: LinearNorm::new(
                graph,
                "world.dynamics.core.dynin1",
                size.stoch * size.classes,
                size.hidden,
            ),
            action_input: LinearNorm::new(
                graph,
                "world.dynamics.core.dynin2",
                config.action_count,
                size.hidden,
            ),
            hidden: BlockLinear::new(
                graph,
                "world.dynamics.core.dynhid0",
                size.blocks,
                block_input,
                block_deter,
            ),
            hidden_norm: nn::RmsNorm::new(
                graph,
                "world.dynamics.core.dynhid0.norm.weight",
                size.deter,
                DREAMER_NORM_EPSILON,
            ),
            gru: BlockLinear::new(
                graph,
                "world.dynamics.core.dyngru",
                size.blocks,
                block_deter,
                3 * block_deter,
            ),
            deter: size.deter,
            hidden_width: size.hidden,
            stoch_width: size.stoch * size.classes,
            action_count: config.action_count,
            blocks: size.blocks,
        }
    }

    pub(crate) fn forward(
        &self,
        graph: &mut Graph,
        deter: NodeId,
        stoch: NodeId,
        action: NodeId,
        batch: usize,
    ) -> NodeId {
        let stoch = graph.reshape(stoch, &[batch, self.stoch_width]);
        let x0 = self.deter_input.forward(graph, deter);
        let x1 = self.stoch_input.forward(graph, stoch);
        let x2 = self.action_input.forward(graph, action);
        let shared = concat_columns(graph, x0, x1, batch, self.hidden_width, self.hidden_width);
        let shared = concat_columns(
            graph,
            shared,
            x2,
            batch,
            2 * self.hidden_width,
            self.hidden_width,
        );

        let block_deter = self.deter / self.blocks;
        let block_input = block_deter + 3 * self.hidden_width;
        let mut grouped = Vec::with_capacity(self.blocks);
        for block in 0..self.blocks {
            let old = slice_columns(
                graph,
                deter,
                batch,
                self.deter,
                block * block_deter,
                block_deter,
            );
            grouped.push(concat_columns(
                graph,
                old,
                shared,
                batch,
                block_deter,
                3 * self.hidden_width,
            ));
        }
        let mut input = grouped[0];
        let mut width = block_input;
        for block in &grouped[1..] {
            input = concat_columns(graph, input, *block, batch, width, block_input);
            width += block_input;
        }

        let hidden = self.hidden.forward(graph, input, batch);
        let hidden = self.hidden_norm.forward(graph, hidden);
        let hidden = graph.silu(hidden);
        let gates = self.gru.forward(graph, hidden, batch);

        // BlockLinear emits [block0(reset,candidate,update), block1(...), ...].
        let gate_width = 3 * block_deter;
        let mut outputs = Vec::with_capacity(self.blocks);
        for block in 0..self.blocks {
            let gate = slice_columns(
                graph,
                gates,
                batch,
                self.blocks * gate_width,
                block * gate_width,
                gate_width,
            );
            let reset = slice_columns(graph, gate, batch, gate_width, 0, block_deter);
            let candidate = slice_columns(graph, gate, batch, gate_width, block_deter, block_deter);
            let update =
                slice_columns(graph, gate, batch, gate_width, 2 * block_deter, block_deter);
            let reset = graph.sigmoid(reset);
            let candidate = graph.mul(reset, candidate);
            let candidate = graph.tanh(candidate);
            let minus_one = graph.constant(vec![-1.0; batch * block_deter], &[batch, block_deter]);
            let update = graph.add(update, minus_one);
            let update = graph.sigmoid(update);
            let old = slice_columns(
                graph,
                deter,
                batch,
                self.deter,
                block * block_deter,
                block_deter,
            );
            let updated = graph.mul(update, candidate);
            let ones = graph.constant(vec![1.0; batch * block_deter], &[batch, block_deter]);
            let negative_update = graph.neg(update);
            let keep = graph.add(ones, negative_update);
            let kept = graph.mul(keep, old);
            outputs.push(graph.add(updated, kept));
        }
        let mut result = outputs[0];
        let mut width = block_deter;
        for output in &outputs[1..] {
            result = concat_columns(graph, result, *output, batch, width, block_deter);
            width += block_deter;
        }
        debug_assert_eq!(width, self.deter);
        debug_assert_eq!(self.action_count, config_action_width(graph, action));
        result
    }
}

fn config_action_width(graph: &Graph, action: NodeId) -> usize {
    graph.node(action).ty.shape[1]
}

pub(crate) struct Prior {
    layer0: LinearNorm,
    layer1: LinearNorm,
    output: nn::Linear,
    stoch: usize,
    classes: usize,
}

impl Prior {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let size = config.network();
        Self {
            layer0: LinearNorm::new(graph, "world.dynamics.prior0", size.deter, size.hidden),
            layer1: LinearNorm::new(graph, "world.dynamics.prior1", size.hidden, size.hidden),
            output: nn::Linear::new(
                graph,
                "world.dynamics.priorlogit",
                size.hidden,
                size.stoch * size.classes,
            ),
            stoch: size.stoch,
            classes: size.classes,
        }
    }

    pub(crate) fn forward(&self, graph: &mut Graph, deter: NodeId, batch: usize) -> NodeId {
        let value = self.layer0.forward(graph, deter);
        let value = self.layer1.forward(graph, value);
        let logits = self.output.forward(graph, value);
        graph.reshape(logits, &[batch * self.stoch, self.classes])
    }
}

pub(crate) struct ObservationEncoder {
    patch0: LinearNorm,
    patch1: LinearNorm,
    depth: usize,
}

impl ObservationEncoder {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let depth = config.network().vision_depth;
        Self {
            patch0: LinearNorm::new(
                graph,
                "world.representation.encoder.patch0",
                OBSERVATION_CHANNELS,
                depth,
            ),
            patch1: LinearNorm::new(graph, "world.representation.encoder.patch1", depth, depth),
            depth,
        }
    }

    pub(crate) fn output_dim(&self) -> usize {
        OBSERVATION_GRID * OBSERVATION_GRID * self.depth
    }

    pub(crate) fn forward(&self, graph: &mut Graph, observation: NodeId, batch: usize) -> NodeId {
        let patches = OBSERVATION_GRID * OBSERVATION_GRID;
        let observation = graph.reshape(observation, &[batch * patches, OBSERVATION_CHANNELS]);
        let value = self.patch0.forward(graph, observation);
        let value = self.patch1.forward(graph, value);
        graph.reshape(value, &[batch, self.output_dim()])
    }
}

pub(crate) struct Posterior {
    hidden: LinearNorm,
    output: nn::Linear,
    encoded_dim: usize,
    deter: usize,
    stoch: usize,
    classes: usize,
}

impl Posterior {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig, encoded_dim: usize) -> Self {
        let size = config.network();
        Self {
            hidden: LinearNorm::new(
                graph,
                "world.representation.posterior.obs0",
                size.deter + encoded_dim,
                size.hidden,
            ),
            output: nn::Linear::new(
                graph,
                "world.representation.posterior.obslogit",
                size.hidden,
                size.stoch * size.classes,
            ),
            encoded_dim,
            deter: size.deter,
            stoch: size.stoch,
            classes: size.classes,
        }
    }

    pub(crate) fn forward(
        &self,
        graph: &mut Graph,
        deter: NodeId,
        encoded: NodeId,
        batch: usize,
    ) -> NodeId {
        let value = concat_columns(graph, deter, encoded, batch, self.deter, self.encoded_dim);
        let value = self.hidden.forward(graph, value);
        let logits = self.output.forward(graph, value);
        graph.reshape(logits, &[batch * self.stoch, self.classes])
    }
}

pub(crate) struct Representation {
    pub(crate) encoder: ObservationEncoder,
    pub(crate) posterior: Posterior,
}

impl Representation {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let encoder = ObservationEncoder::new(graph, config);
        let posterior = Posterior::new(graph, config, encoder.output_dim());
        Self { encoder, posterior }
    }
}

pub(crate) fn mixed_probabilities(
    graph: &mut Graph,
    logits: NodeId,
    rows: usize,
    classes: usize,
    unimix: f32,
) -> NodeId {
    let probabilities = graph.softmax(logits);
    let retained = graph.constant(vec![1.0 - unimix; rows * classes], &[rows, classes]);
    let probabilities = graph.mul(probabilities, retained);
    let uniform = graph.constant(
        vec![unimix / classes as f32; rows * classes],
        &[rows, classes],
    );
    graph.add(probabilities, uniform)
}

pub(crate) fn straight_through_sample(
    graph: &mut Graph,
    hard_sample: NodeId,
    probabilities: NodeId,
) -> NodeId {
    let detached = graph.stop_gradient(probabilities);
    let negative = graph.neg(detached);
    let correction = graph.add(probabilities, negative);
    graph.add(hard_sample, correction)
}

pub(crate) struct CategoricalKl {
    pub(crate) loss: NodeId,
    pub(crate) raw: NodeId,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn categorical_kl(
    graph: &mut Graph,
    posterior_logits: NodeId,
    prior_logits: NodeId,
    batch: usize,
    stoch: usize,
    classes: usize,
    unimix: f32,
    stop_posterior: bool,
    stop_prior: bool,
    free_nats: f32,
) -> CategoricalKl {
    let posterior_logits = if stop_posterior {
        graph.stop_gradient(posterior_logits)
    } else {
        posterior_logits
    };
    let prior_logits = if stop_prior {
        graph.stop_gradient(prior_logits)
    } else {
        prior_logits
    };
    let rows = batch * stoch;
    let posterior = mixed_probabilities(graph, posterior_logits, rows, classes, unimix);
    let prior = mixed_probabilities(graph, prior_logits, rows, classes, unimix);
    let posterior_log = graph.log(posterior);
    let prior_log = graph.log(prior);
    let negative_prior = graph.neg(prior_log);
    let log_ratio = graph.add(posterior_log, negative_prior);
    let elements = graph.mul(posterior, log_ratio);
    let per_variable = graph.sum_inner(elements);
    let per_variable = graph.reshape(per_variable, &[batch, stoch]);
    let per_state = graph.sum_inner(per_variable);
    let raw = graph.mean_all(per_state);
    let loss = if free_nats == 0.0 {
        raw
    } else {
        let negative_floor = graph.constant(vec![-free_nats; batch], &[batch, 1]);
        let excess = graph.add(per_state, negative_floor);
        let excess = graph.relu(excess);
        let floor = graph.constant(vec![free_nats; batch], &[batch, 1]);
        let clamped = graph.add(excess, floor);
        graph.mean_all(clamped)
    };
    CategoricalKl { loss, raw }
}

pub(crate) fn feature(
    graph: &mut Graph,
    deter: NodeId,
    stoch: NodeId,
    batch: usize,
    config: &DreamerConfig,
) -> NodeId {
    let size = config.network();
    let stoch = graph.reshape(stoch, &[batch, size.stoch * size.classes]);
    concat_columns(
        graph,
        deter,
        stoch,
        batch,
        size.deter,
        size.stoch * size.classes,
    )
}

pub(crate) struct MlpHead {
    layers: Vec<LinearNorm>,
    output: nn::Linear,
}

impl MlpHead {
    pub(crate) fn new(
        graph: &mut Graph,
        name: &str,
        input: usize,
        units: usize,
        layers: usize,
        output: usize,
    ) -> Self {
        let mut network = Vec::with_capacity(layers);
        let mut width = input;
        for layer in 0..layers {
            network.push(LinearNorm::new(
                graph,
                &format!("{name}.layer{layer}"),
                width,
                units,
            ));
            width = units;
        }
        Self {
            layers: network,
            output: nn::Linear::new(graph, &format!("{name}.out"), width, output),
        }
    }

    pub(crate) fn forward(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let mut value = input;
        for layer in &self.layers {
            value = layer.forward(graph, value);
        }
        self.output.forward(graph, value)
    }

    /// Forward pass with frozen weights and an input gradient.
    pub(crate) fn forward_frozen(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let mut value = input;
        for layer in &self.layers {
            value = layer.forward_frozen(graph, value);
        }
        let weight = graph.stop_gradient(self.output.weight);
        let value = graph.matmul(value, weight);
        if let Some(bias) = self.output.bias {
            let bias = graph.stop_gradient(bias);
            graph.bias_add(value, bias)
        } else {
            value
        }
    }
}

pub(crate) struct ObservationDecoder {
    trunk: LinearNorm,
    spatial: nn::Linear,
    patch_norm: nn::RmsNorm,
    output: nn::Linear,
    depth: usize,
}

impl ObservationDecoder {
    pub(crate) fn new(
        graph: &mut Graph,
        config: &DreamerConfig,
        name: &str,
        input_dim: usize,
    ) -> Self {
        let size = config.network();
        let patches = OBSERVATION_GRID * OBSERVATION_GRID;
        let depth = config.observation_decoder_depth();
        Self {
            trunk: LinearNorm::new(graph, &format!("{name}.trunk"), input_dim, size.units),
            spatial: nn::Linear::new(
                graph,
                &format!("{name}.spatial"),
                size.units,
                patches * depth,
            ),
            patch_norm: nn::RmsNorm::new(
                graph,
                &format!("{name}.patch.norm.weight"),
                depth,
                DREAMER_NORM_EPSILON,
            ),
            output: nn::Linear::new(graph, &format!("{name}.out"), depth, OBSERVATION_CHANNELS),
            depth,
        }
    }

    pub(crate) fn forward(&self, graph: &mut Graph, input: NodeId, batch: usize) -> NodeId {
        let patches = OBSERVATION_GRID * OBSERVATION_GRID;
        let value = self.trunk.forward(graph, input);
        let value = self.spatial.forward(graph, value);
        let value = graph.reshape(value, &[batch * patches, self.depth]);
        let value = self.patch_norm.forward(graph, value);
        let value = graph.silu(value);
        self.output.forward(graph, value)
    }
}

#[cfg(test)]
mod block_matmul_tests {
    use super::*;
    use meganeura::{Mode, SessionConfig, graph::Op};

    fn graph(
        batch: usize,
        input_width: usize,
        output_width: usize,
        grouped: bool,
        training: bool,
    ) -> Graph {
        let mut graph = Graph::new();
        let input = graph.parameter("probe.input", &[batch, 8 * input_width]);
        let block = BlockLinear::new(&mut graph, "probe.block", 8, input_width, output_width);
        let value = if grouped {
            block.forward(&mut graph, input, batch)
        } else {
            block.forward_serial(&mut graph, input, batch)
        };
        if training {
            let square = graph.mul(value, value);
            let mean = graph.mean_all(square);
            let loss = graph.scale(mean, -0.7);
            graph.set_outputs(vec![loss, value]);
        } else {
            graph.set_outputs(vec![value]);
        }
        graph
    }

    #[test]
    fn small_batch_blocks_preserve_parameters_and_reduce_dispatches() {
        for batch in [2, 4, 6, 8, 16] {
            for (input, output) in [(1024, 256), (256, 768)] {
                let candidate = graph(batch, input, output, true, false);
                let control = graph(batch, input, output, false, false);
                let parameters = |g: &Graph| {
                    g.nodes()
                        .iter()
                        .filter_map(|n| match &n.op {
                            Op::Parameter { name } => Some((name.clone(), n.ty.clone())),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(parameters(&candidate), parameters(&control));
                for (mode, candidate, control) in [
                    ("raw", &candidate, &control),
                    (
                        "optimized",
                        &meganeura::optimize::optimize(&candidate),
                        &meganeura::optimize::optimize(&control),
                    ),
                ] {
                    let candidate_plan = meganeura::compile::compile(candidate);
                    let control_plan = meganeura::compile::compile(control);
                    eprintln!(
                        "block graph {mode} rows={batch} groups=8 input={input} output={output}: dispatches {} -> {}",
                        control_plan.dispatches.len(),
                        candidate_plan.dispatches.len()
                    );
                    assert_eq!(candidate_plan.dispatches.len(), 2);
                    assert!(control_plan.dispatches.len() > candidate_plan.dispatches.len());
                }
                assert_eq!(
                    candidate
                        .nodes()
                        .iter()
                        .filter(|n| matches!(n.op, Op::BlockMatMul))
                        .count(),
                    1
                );
                assert_eq!(
                    control
                        .nodes()
                        .iter()
                        .filter(|n| matches!(n.op, Op::MatMul))
                        .count(),
                    8
                );
            }
        }
    }

    #[test]
    fn gemv_and_large_batch_graphs_remain_exact() {
        for batch in [1, 17, 64, 1024] {
            let candidate = graph(batch, 5, 7, true, false);
            let control = graph(batch, 5, 7, false, false);
            assert_eq!(
                serde_json::to_vec(candidate.nodes()).unwrap(),
                serde_json::to_vec(control.nodes()).unwrap()
            );
            assert_eq!(candidate.outputs(), control.outputs());
        }
    }

    #[test]
    fn block_training_keeps_all_parameter_gradients_full_precision() {
        let graph = graph(6, 5, 7, true, true);
        let differentiated = meganeura::autodiff::differentiate(&graph);
        let block_gradients = differentiated.nodes()[graph.nodes().len()..]
            .iter()
            .filter(|n| matches!(n.op, Op::BlockMatMulAT { .. } | Op::BlockMatMulBT))
            .collect::<Vec<_>>();
        assert_eq!(block_gradients.len(), 2);
        assert!(block_gradients.iter().all(|n| n.requires_full_precision));
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert_eq!(plan.param_grad_pairs.len(), 3);
    }

    #[test]
    #[ignore = "requires exclusive GPU and precedes full-learning/runtime qualification"]
    fn production_blocks_match_serial_outputs_and_all_gradients_exactly() {
        for batch in [6, 16] {
            for (input, output) in [(1024, 256), (256, 768)] {
                let control_graph = graph(batch, input, output, false, true);
                let mut control = meganeura::build(
                    &control_graph,
                    SessionConfig {
                        mode: Mode::Training,
                        ..SessionConfig::default()
                    },
                )
                .0;
                let candidate_graph = graph(batch, input, output, true, true);
                let mut candidate = meganeura::build(
                    &candidate_graph,
                    SessionConfig {
                        mode: Mode::Training,
                        gpu: Some(control.context()),
                        ..SessionConfig::default()
                    },
                )
                .0;
                for (session, graph) in [
                    (&mut control, &control_graph),
                    (&mut candidate, &candidate_graph),
                ] {
                    crate::dreamer::runtime::initialize_d3(session, graph, 7301);
                    session.set_adam(0.0, 0.9, 0.999, 1e-8);
                    session.step();
                    session.wait();
                }
                let exact = |label: &str, a: &[f32], b: &[f32]| {
                    assert_eq!(a.len(), b.len());
                    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
                        assert!(a.is_finite() && b.is_finite());
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{batch}/{input}/{output} {label}[{index}]: {a} != {b}"
                        );
                    }
                };
                exact("loss", &[control.read_loss()], &[candidate.read_loss()]);
                let mut a = vec![0.0; batch * 8 * output];
                let mut b = a.clone();
                control.read_output_by_index(1, &mut a);
                candidate.read_output_by_index(1, &mut b);
                exact("output", &a, &b);
                for name in ["probe.input", "probe.block.weight", "probe.block.bias"] {
                    let size = control.param_size(name).unwrap();
                    let mut a = vec![0.0; size];
                    let mut b = a.clone();
                    control.read_param_grad(name, &mut a);
                    candidate.read_param_grad(name, &mut b);
                    exact(name, &a, &b);
                }
            }
        }
    }
}
