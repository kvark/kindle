//! Standalone diagonal Gaussian world state; categorical defaults are untouched.
use super::{
    DreamerConfig,
    agent::sample_latents,
    networks::{KlLoss, slice_columns},
};
use meganeura::{Graph, graph::NodeId};
use rand::{Rng, rngs::StdRng};

pub(crate) const STD_MIN: f32 = 0.1;

fn parameters(graph: &mut Graph, stats: NodeId, rows: usize, width: usize) -> (NodeId, NodeId) {
    let mean = slice_columns(graph, stats, rows, 2 * width, 0, width);
    let raw = slice_columns(graph, stats, rows, 2 * width, width, width);
    let scale = graph.sigmoid(raw);
    let floor = graph.constant(vec![STD_MIN; rows * width], &[rows, width]);
    (mean, graph.add(scale, floor))
}

pub(crate) fn sample(
    graph: &mut Graph,
    stats: NodeId,
    noise: NodeId,
    rows: usize,
    width: usize,
) -> NodeId {
    let (mean, std) = parameters(graph, stats, rows, width);
    let noise = graph.stop_gradient(noise);
    let scaled = graph.mul(std, noise);
    graph.add(mean, scaled)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn kl(
    graph: &mut Graph,
    q: NodeId,
    p: NodeId,
    batch: usize,
    groups: usize,
    width: usize,
    stop_q: bool,
    stop_p: bool,
    free_nats: f32,
) -> KlLoss {
    let q = if stop_q { graph.stop_gradient(q) } else { q };
    let p = if stop_p { graph.stop_gradient(p) } else { p };
    let (qm, qs) = parameters(graph, q, batch * groups, width);
    let (pm, ps) = parameters(graph, p, batch * groups, width);
    let ratio = graph.div(ps, qs);
    let log_ratio = graph.log(ratio);
    let residual = graph.sub(qm, pm);
    let squared = graph.mul(residual, residual);
    let qvar = graph.mul(qs, qs);
    let numerator = graph.add(qvar, squared);
    let pvar = graph.mul(ps, ps);
    let scaled = graph.div(numerator, pvar);
    let scaled = graph.scale(scaled, 0.5);
    let elements = graph.add(log_ratio, scaled);
    let half = graph.constant(vec![0.5; batch * groups * width], &[batch * groups, width]);
    let elements = graph.sub(elements, half);
    let grouped = graph.sum_inner(elements);
    let grouped = graph.reshape(grouped, &[batch, groups]);
    let state = graph.sum_inner(grouped);
    let raw = graph.mean_all(state);
    let loss = if free_nats == 0.0 {
        raw
    } else {
        let neg_floor = graph.constant(vec![-free_nats; batch], &[batch, 1]);
        let excess = graph.add(state, neg_floor);
        let excess = graph.relu(excess);
        let floor = graph.constant(vec![free_nats; batch], &[batch, 1]);
        let clamped = graph.add(excess, floor);
        graph.mean_all(clamped)
    };
    KlLoss { loss, raw }
}

pub(crate) fn normals(rng: &mut StdRng, count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    while result.len() < count {
        let radius = (-2.0_f64 * rng.random_range(1e-12_f64..1.0).ln()).sqrt();
        let angle = rng.random_range(0.0_f64..std::f64::consts::TAU);
        result.push((radius * angle.cos()) as f32);
        if result.len() < count {
            result.push((radius * angle.sin()) as f32);
        }
    }
    result
}

pub(crate) fn noise(config: &DreamerConfig, rng: &mut StdRng) -> Vec<f32> {
    let size = config.network();
    let count = config.batch_size * size.stoch * size.classes;
    if config.world_gaussian {
        normals(rng, count)
    } else {
        (0..count)
            .map(|_| rng.random_range(1e-6..1.0 - 1e-6))
            .collect()
    }
}

pub(crate) fn sample_cpu(config: &DreamerConfig, stats: &[f32], rng: &mut StdRng) -> Vec<f32> {
    let size = config.network();
    if config.world_mean_latents {
        assert!(!config.world_gaussian);
        assert_eq!(stats.len(), config.batch_size * size.stoch * size.classes);
        let mut result = vec![0.; stats.len()];
        for (logits, out) in stats
            .chunks_exact(size.classes)
            .zip(result.chunks_exact_mut(size.classes))
        {
            super::distributions::softmax_unimix(logits, config.unimix, out);
        }
        return result;
    }
    if !config.world_gaussian {
        return sample_latents(
            stats,
            config.batch_size,
            size.stoch,
            size.classes,
            config.unimix,
            rng,
        );
    }
    assert_eq!(
        stats.len(),
        2 * config.batch_size * size.stoch * size.classes
    );
    let noise = normals(rng, stats.len() / 2);
    stats
        .chunks_exact(2 * size.classes)
        .flat_map(|row| {
            let (mean, raw) = row.split_at(size.classes);
            mean.iter()
                .zip(raw)
                .map(|(&m, &s)| (m, STD_MIN + 1.0 / (1.0 + (-s).exp())))
                .collect::<Vec<_>>()
        })
        .zip(noise)
        .map(|((m, s), n)| m + s * n)
        .collect()
}

#[cfg(test)]
mod tests;
