//! Joint one-step forecast scoring with two independent sampled worlds.

use super::super::distributions::TwoHotBins;
use super::*;

pub(super) struct Uniforms {
    pub reward: [NodeId; 2],
    pub continuation: [NodeId; 2],
    pub reward_target: NodeId,
    pub continuation_target: NodeId,
}

#[derive(Clone, Copy)]
struct Sample {
    rgb: NodeId,
    reward: NodeId,
    continuation: NodeId,
}

pub(super) struct Losses {
    pub energy: NodeId,
    pub target_distance: NodeId,
    pub sample_distance: NodeId,
    pub valid: NodeId,
}

impl Losses {
    pub(super) fn scaled(&self, graph: &mut Graph, config: &DreamerConfig) -> NodeId {
        graph.scale(self.energy, config.loss_scales.reconstruction)
    }
}

fn sample_coordinate(
    graph: &mut Graph,
    logits: NodeId,
    uniforms: NodeId,
    support: &[f32],
    rows: usize,
) -> NodeId {
    let hard = gumbel_sample(graph, logits, uniforms, rows, support.len(), 0.0);
    let hard = graph.stop_gradient(hard);
    let probabilities = graph.softmax(logits);
    let sample = straight_through_sample(graph, hard, probabilities);
    let support = graph.constant(support.to_vec(), &[support.len()]);
    let weighted = graph.bias_mul(sample, support);
    graph.sum_inner(weighted)
}

fn decode(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    state: NodeId,
    uniforms: [NodeId; 2],
    rows: usize,
) -> Sample {
    let rgb = model.decoder.as_ref().unwrap().forward(graph, state, rows);
    let reward_logits = model.heads.reward.forward(graph, state);
    let bins = TwoHotBins::new(config.value_bins);
    let support = bins
        .values()
        .iter()
        .map(|&x| x.signum() * x.abs().ln_1p())
        .collect::<Vec<_>>();
    let reward = sample_coordinate(graph, reward_logits, uniforms[0], &support, rows);
    let continuation_logits = model.heads.continuation.forward(graph, state);
    let continuation_logits = graph.reshape(continuation_logits, &[rows]);
    let zero = graph.constant(vec![0.0; rows], &[rows]);
    let binary_logits = graph.concat(zero, continuation_logits, rows as u32, 1, 1, 1);
    let binary_logits = graph.reshape(binary_logits, &[rows, 2]);
    let continuation = sample_coordinate(graph, binary_logits, uniforms[1], &[0.0, 1.0], rows);
    Sample {
        rgb,
        reward,
        continuation,
    }
}

fn distance(graph: &mut Graph, a: Sample, b: Sample, weights: NodeId, metric: [f32; 2]) -> NodeId {
    let rgb = graph.sub(a.rgb, b.rgb);
    let rgb = graph.mul(rgb, rgb);
    let rgb = graph.mul(rgb, weights);
    let rgb = graph.sum_inner(rgb);
    let reward = graph.sub(a.reward, b.reward);
    let reward = graph.mul(reward, reward);
    let reward = graph.scale(reward, metric[0]);
    let continuation = graph.sub(a.continuation, b.continuation);
    let continuation = graph.mul(continuation, continuation);
    let continuation = graph.scale(continuation, metric[1]);
    let squared = graph.add(rgb, reward);
    let squared = graph.add(squared, continuation);
    let rows = graph.node(squared).ty.shape[0];
    let epsilon = graph.constant(vec![1e-6; rows], &[rows, 1]);
    let squared = graph.add(squared, epsilon);
    graph.sqrt(squared)
}

fn score(
    graph: &mut Graph,
    samples: [Sample; 2],
    target: Sample,
    weights: NodeId,
    keep: NodeId,
    metric: [f32; 2],
) -> Losses {
    let target = Sample {
        rgb: graph.stop_gradient(target.rgb),
        reward: graph.stop_gradient(target.reward),
        continuation: graph.stop_gradient(target.continuation),
    };
    let weights = graph.stop_gradient(weights);
    let keep = graph.stop_gradient(keep);
    let first = distance(graph, samples[0], target, weights, metric);
    let second = distance(graph, samples[1], target, weights, metric);
    let target_distance = graph.add(first, second);
    let target_distance = graph.scale(target_distance, 0.5);
    let sample_distance = distance(graph, samples[0], samples[1], weights, metric);
    let half_pair = graph.scale(sample_distance, 0.5);
    let energy = graph.sub(target_distance, half_pair);
    let mut mean = |value| {
        let masked = graph.mul(value, keep);
        graph.mean_all(masked)
    };
    let energy = mean(energy);
    let target_distance = mean(target_distance);
    let sample_distance = mean(sample_distance);
    let valid = graph.mean_all(keep);
    Losses {
        energy,
        target_distance,
        sample_distance,
        valid,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn losses(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    states: [NodeId; 2],
    target: NodeId,
    previous: NodeId,
    before_previous: NodeId,
    past_keep: NodeId,
    keep_action: NodeId,
    uniforms: Uniforms,
    rows: usize,
) -> Losses {
    let first = decode(
        graph,
        config,
        model,
        states[0],
        [uniforms.reward[0], uniforms.continuation[0]],
        rows,
    );
    let second = decode(
        graph,
        config,
        model,
        states[1],
        [uniforms.reward[1], uniforms.continuation[1]],
        rows,
    );
    let past_keep = graph.mean_inner(past_keep);
    let weights =
        super::super::networks::pixel_motion_weights(graph, previous, before_previous, past_keep);
    let keep = graph.mean_inner(keep_action);
    let target = Sample {
        rgb: target,
        reward: uniforms.reward_target,
        continuation: uniforms.continuation_target,
    };
    score(
        graph,
        [first, second],
        target,
        weights,
        keep,
        [config.loss_scales.reward, config.loss_scales.continuation],
    )
}

pub(super) fn average(graph: &mut Graph, values: &[Losses]) -> Losses {
    let mut mean = |select: fn(&Losses) -> NodeId| {
        let nodes = values.iter().map(select).collect::<Vec<_>>();
        let total = sum(graph, &nodes);
        graph.scale(total, 1.0 / values.len() as f32)
    };
    Losses {
        energy: mean(|v| v.energy),
        target_distance: mean(|v| v.target_distance),
        sample_distance: mean(|v| v.sample_distance),
        valid: mean(|v| v.valid),
    }
}

#[cfg(test)]
mod tests;
