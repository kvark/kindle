//! Causal image memory shared by spatial-decoder training and inference.
use super::networks::{concat_columns, slice_columns};
use meganeura::{Graph, graph::NodeId};

fn masked(graph: &mut Graph, image: NodeId, keep: NodeId, width: usize) -> NodeId {
    let keep = graph.stop_gradient(keep);
    let keep = graph.broadcast_inner(keep, width);
    graph.mul(image, keep)
}

pub(crate) fn before(
    graph: &mut Graph,
    images: &[NodeId],
    keeps: &[NodeId],
    time: usize,
    batch: usize,
    width: usize,
) -> NodeId {
    let zero = graph.constant(vec![0.0; batch * width], &[batch, width]);
    let newer = if time > 0 {
        masked(graph, images[time - 1], keeps[time], width)
    } else {
        zero
    };
    let older = if time > 1 {
        let keep = graph.mul(keeps[time], keeps[time - 1]);
        masked(graph, images[time - 2], keep, width)
    } else {
        zero
    };
    concat_columns(graph, older, newer, batch, width, width)
}

pub(crate) fn after(
    graph: &mut Graph,
    images: &[NodeId],
    keeps: &[NodeId],
    time: usize,
    batch: usize,
    width: usize,
) -> NodeId {
    let older = if time > 0 {
        masked(graph, images[time - 1], keeps[time], width)
    } else {
        graph.constant(vec![0.0; batch * width], &[batch, width])
    };
    concat_columns(graph, older, images[time], batch, width, width)
}

/// One observed frame from `lag` arrivals back, clamped to the window start and
/// repeated into both history slots; zero at the first arrival and across any
/// reset within the lag.
pub(crate) fn context(
    graph: &mut Graph,
    images: &[NodeId],
    keeps: &[NodeId],
    time: usize,
    lag: usize,
    batch: usize,
    width: usize,
) -> NodeId {
    let frame = if time == 0 {
        graph.constant(vec![0.0; batch * width], &[batch, width])
    } else {
        let lag = lag.min(time);
        let mut keep = keeps[time];
        for step in 1..lag {
            keep = graph.mul(keep, keeps[time - step]);
        }
        masked(graph, images[time - lag], keep, width)
    };
    concat_columns(graph, frame, frame, batch, width, width)
}

/// The last observed frame in both history slots, held fixed through a forecast.
pub(crate) fn fixed(graph: &mut Graph, image: NodeId, batch: usize, width: usize) -> NodeId {
    concat_columns(graph, image, image, batch, width, width)
}

pub(crate) fn shift(
    graph: &mut Graph,
    history: NodeId,
    next: NodeId,
    batch: usize,
    width: usize,
) -> NodeId {
    let latest = slice_columns(graph, history, batch, 2 * width, width, width);
    concat_columns(graph, latest, next, batch, width, width)
}

pub(crate) fn before_cpu(
    images: &[f32],
    actions: &[i32],
    time: usize,
    batch: usize,
    width: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; batch * 2 * width];
    for row in 0..batch {
        if actions[time * batch + row] == -1 {
            continue;
        }
        for lag in 1..=2 {
            if time < lag || (lag == 2 && actions[(time - 1) * batch + row] == -1) {
                continue;
            }
            let source = ((time - lag) * batch + row) * width;
            let target = row * 2 * width + (2 - lag) * width;
            result[target..target + width].copy_from_slice(&images[source..source + width]);
        }
    }
    result
}

pub(crate) fn after_cpu(
    images: &[f32],
    actions: &[i32],
    time: usize,
    batch: usize,
    width: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; batch * 2 * width];
    for row in 0..batch {
        let target = row * 2 * width;
        let source = (time * batch + row) * width;
        result[target + width..target + 2 * width].copy_from_slice(&images[source..source + width]);
        if time > 0 && actions[time * batch + row] != -1 {
            result[target..target + width]
                .copy_from_slice(&images[source - batch * width..source - batch * width + width]);
        }
    }
    result
}

pub(crate) fn context_cpu(
    images: &[f32],
    actions: &[i32],
    time: usize,
    lag: usize,
    batch: usize,
    width: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; batch * 2 * width];
    if time == 0 {
        return result;
    }
    let lag = lag.min(time);
    for row in 0..batch {
        if (0..lag).any(|step| actions[(time - step) * batch + row] == -1) {
            continue;
        }
        let source = ((time - lag) * batch + row) * width;
        for half in 0..2 {
            let target = row * 2 * width + half * width;
            result[target..target + width].copy_from_slice(&images[source..source + width]);
        }
    }
    result
}

pub(crate) fn fixed_cpu(images: &[f32], time: usize, batch: usize, width: usize) -> Vec<f32> {
    let mut result = vec![0.0; batch * 2 * width];
    for row in 0..batch {
        let source = (time * batch + row) * width;
        for half in 0..2 {
            let target = row * 2 * width + half * width;
            result[target..target + width].copy_from_slice(&images[source..source + width]);
        }
    }
    result
}

pub(crate) fn shift_cpu(history: &[f32], next: &[f32], width: usize) -> Vec<f32> {
    history
        .chunks_exact(2 * width)
        .zip(next.chunks_exact(width))
        .flat_map(|(past, current)| past[width..].iter().chain(current).copied())
        .collect()
}

#[cfg(test)]
mod tests;
