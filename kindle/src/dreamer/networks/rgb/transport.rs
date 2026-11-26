//! Color-shared separable pixel transport with the ordinary RGB head as fallback.
use super::*;

const RADIUS: isize = 8;
const OFFSETS: usize = 17;
const FIELDS: usize = 2 * OFFSETS + 1;

pub(super) struct Transport {
    projection: Convolution,
}

impl Transport {
    pub fn new(graph: &mut Graph, name: &str, side: usize) -> Self {
        Self {
            projection: Convolution {
                layer: nn::Conv2d::new(
                    graph,
                    name,
                    16,
                    FIELDS as u32,
                    1,
                    side as u32,
                    side as u32,
                    1,
                    0,
                ),
                bias: graph.parameter(&format!("{name}.bias"), &[FIELDS]),
                output_side: side as u32,
            },
        }
    }

    pub fn forward(
        &self,
        graph: &mut Graph,
        fused: NodeId,
        history: NodeId,
        redraw: NodeId,
        batch: usize,
        side: usize,
    ) -> NodeId {
        let masks = self.projection.forward(graph, fused, batch);
        let masks = graph.reshape(masks, &[batch, FIELDS * side * side]);
        let width = 3 * side * side;
        let latest = slice_columns(graph, history, batch, 2 * width, width, width);
        compose(graph, latest, redraw, masks, batch, side)
    }
}

fn shift_rows(
    graph: &mut Graph,
    input: NodeId,
    rows: usize,
    width: usize,
    offset: isize,
) -> NodeId {
    if offset == 0 {
        return graph.reshape(input, &[rows, width]);
    }
    let count = offset.unsigned_abs();
    if count >= width {
        return graph.constant(vec![0.; rows * width], &[rows, width]);
    }
    let start = if offset < 0 { count } else { 0 };
    let body = slice_columns(graph, input, rows, width, start, width - count);
    let zero = graph.constant(vec![0.; rows * count], &[rows, count]);
    if offset > 0 {
        concat_columns(graph, zero, body, rows, count, width - count)
    } else {
        concat_columns(graph, body, zero, rows, width - count, count)
    }
}

fn mix_axis(
    graph: &mut Graph,
    image: NodeId,
    logits: NodeId,
    batch: usize,
    side: usize,
    vertical: bool,
) -> NodeId {
    let area = side * side;
    let probabilities = transpose_spatial(graph, logits, batch, OFFSETS, area);
    let probabilities = graph.reshape(probabilities, &[batch * area, OFFSETS]);
    let probabilities = graph.softmax(probabilities);
    let probabilities = transpose_spatial(graph, probabilities, batch, area, OFFSETS);
    let probabilities = graph.reshape(probabilities, &[batch, OFFSETS * area]);
    let mut candidates = Vec::new();
    for index in 0..OFFSETS {
        let offset = index as isize - RADIUS;
        let shifted = if vertical {
            shift_rows(graph, image, batch * 3, area, offset * side as isize)
        } else {
            shift_rows(graph, image, batch * 3 * side, side, offset)
        };
        let shifted = graph.reshape(shifted, &[batch, 3 * area]);
        let weight = slice_columns(
            graph,
            probabilities,
            batch,
            OFFSETS * area,
            index * area,
            area,
        );
        let weight = repeat_columns(graph, weight, batch, area, 3);
        candidates.push(graph.mul(shifted, weight));
    }
    sum(graph, &candidates)
}

fn compose(
    graph: &mut Graph,
    latest: NodeId,
    redraw: NodeId,
    masks: NodeId,
    batch: usize,
    side: usize,
) -> NodeId {
    let area = side * side;
    let width = 3 * area;
    let latest = graph.reshape(latest, &[batch * width, 1]);
    let half = graph.scalar(0.5);
    let latest = graph.bias_add(latest, half);
    let latest = graph.reshape(latest, &[batch, width]);
    let horizontal = slice_columns(graph, masks, batch, FIELDS * area, 0, OFFSETS * area);
    let vertical = slice_columns(
        graph,
        masks,
        batch,
        FIELDS * area,
        OFFSETS * area,
        OFFSETS * area,
    );
    let moved = mix_axis(graph, latest, horizontal, batch, side, false);
    let moved = mix_axis(graph, moved, vertical, batch, side, true);
    let blend = slice_columns(graph, masks, batch, FIELDS * area, 2 * OFFSETS * area, area);
    let blend = graph.sigmoid(blend);
    let blend = repeat_columns(graph, blend, batch, area, 3);
    let retained = graph.mul(blend, moved);
    let inverse = graph.neg(blend);
    let inverse = graph.reshape(inverse, &[batch * width, 1]);
    let one = graph.scalar(1.0);
    let inverse = graph.bias_add(inverse, one);
    let inverse = graph.reshape(inverse, &[batch, width]);
    let redraw = graph.sigmoid(redraw);
    let generated = graph.mul(inverse, redraw);
    let probability = graph.add(retained, generated);
    let probability = graph.clamp(probability, 1e-6, 1.0 - 1e-6);
    let positive = graph.log(probability);
    let negative = graph.neg(probability);
    let negative = graph.reshape(negative, &[batch * width, 1]);
    let negative = graph.bias_add(negative, one);
    let negative = graph.log(negative);
    let negative = graph.reshape(negative, &[batch, width]);
    graph.sub(positive, negative)
}

#[cfg(test)]
mod tests;
