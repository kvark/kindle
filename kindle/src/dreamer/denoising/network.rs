use super::*;
use crate::dreamer::networks::{ConvNorm, Convolution, MlpHead, concat_columns};

fn join(
    g: &mut Graph,
    a: NodeId,
    b: NodeId,
    batch: usize,
    ca: usize,
    cb: usize,
    side: usize,
) -> NodeId {
    let joined = concat_columns(g, a, b, batch, ca * side * side, cb * side * side);
    g.reshape(joined, &[batch * (ca + cb) * side * side])
}
fn up(g: &mut Graph, x: NodeId, batch: usize, channels: usize, side: usize) -> NodeId {
    g.upsample_2x(x, batch as u32, channels as u32, side as u32, side as u32)
}

fn denoiser(g: &mut Graph, input: NodeId, batch: usize, actions: usize, side: usize) -> NodeId {
    let channels = 16 + 4 * actions;
    let input = g.reshape(input, &[batch * channels * side * side]);
    let e0 =
        ConvNorm::new(g, "pixel.denoiser.enc0", channels, 16, side).forward(g, input, batch, false);
    let e1 = ConvNorm::with_stride(g, "pixel.denoiser.enc1", 16, 32, side, 2)
        .forward(g, e0, batch, false);
    let e2 = ConvNorm::with_stride(g, "pixel.denoiser.enc2", 32, 64, side / 2, 2)
        .forward(g, e1, batch, false);
    let e3 = ConvNorm::with_stride(g, "pixel.denoiser.enc3", 64, 64, side / 4, 2)
        .forward(g, e2, batch, false);
    let mid = ConvNorm::new(g, "pixel.denoiser.mid", 64, 64, side / 8).forward(g, e3, batch, false);
    let x = up(g, mid, batch, 64, side / 8);
    let x = join(g, x, e2, batch, 64, 64, side / 4);
    let x = ConvNorm::new(g, "pixel.denoiser.up2", 128, 64, side / 4).forward(g, x, batch, false);
    let x = up(g, x, batch, 64, side / 4);
    let x = join(g, x, e1, batch, 64, 32, side / 2);
    let x = ConvNorm::new(g, "pixel.denoiser.up1", 96, 32, side / 2).forward(g, x, batch, false);
    let x = up(g, x, batch, 32, side / 2);
    let x = join(g, x, e0, batch, 32, 16, side);
    let x = ConvNorm::new(g, "pixel.denoiser.up0", 48, 16, side).forward(g, x, batch, false);
    Convolution::new(g, "pixel.denoiser.out", 16, 3, side).forward(g, x, batch)
}

fn events(
    g: &mut Graph,
    input: NodeId,
    batch: usize,
    actions: usize,
    side: usize,
    bins: usize,
) -> (NodeId, NodeId) {
    let c = 15 + 4 * actions;
    let input = g.reshape(input, &[batch * c * side * side]);
    let x = ConvNorm::with_stride(g, "pixel.events.enc0", c, 16, side, 2)
        .forward(g, input, batch, false);
    let x = ConvNorm::with_stride(g, "pixel.events.enc1", 16, 32, side / 2, 2)
        .forward(g, x, batch, false);
    let x = ConvNorm::with_stride(g, "pixel.events.enc2", 32, 64, side / 4, 2)
        .forward(g, x, batch, false);
    let x = g.reshape(x, &[batch, 64 * (side / 8) * (side / 8)]);
    let reward = MlpHead::new(
        g,
        "pixel.events.reward",
        64 * (side / 8) * (side / 8),
        128,
        1,
        bins,
    )
    .forward(g, x);
    let continuation = MlpHead::new(
        g,
        "pixel.events.continuation",
        64 * (side / 8) * (side / 8),
        128,
        1,
        1,
    )
    .forward(g, x);
    (reward, continuation)
}

pub(super) fn training(c: &DenoisingConfig) -> Graph {
    let mut g = Graph::new();
    let b = c.train_batch_size;
    let w = 3 * c.side * c.side;
    let input = g.input(
        "denoiser_input",
        &[b, (16 + 4 * c.action_count) * c.side * c.side],
    );
    let target = g.input("denoiser_target", &[b * w]);
    let weight = g.input("row_weight", &[b, 1]);
    let output = denoiser(&mut g, input, b, c.action_count, c.side);
    let image_loss = if c.context_motion_scale == 0. {
        masked_mse(&mut g, output, target, weight, b, w)
    } else {
        let importance = g.input("pixel_importance", &[b * w]);
        weighted_mse(&mut g, output, target, weight, importance, b, w)
    };
    let event_input = g.input(
        "event_input",
        &[b, (15 + 4 * c.action_count) * c.side * c.side],
    );
    let (reward, continuation) =
        events(&mut g, event_input, b, c.action_count, c.side, c.value_bins);
    let reward_target = g.input("reward_target", &[b, c.value_bins]);
    let reward_loss =
        crate::dreamer::networks::weighted_cross_entropy(&mut g, reward, reward_target, weight);
    let terminal_target = g.input("alive_centered", &[b, 1]);
    let continuation_loss =
        crate::dreamer::networks::pixel_cross_entropy(&mut g, continuation, terminal_target);
    let continuation_loss = g.mul(continuation_loss, weight);
    let continuation_loss = g.mean_all(continuation_loss);
    let total = g.add(image_loss, reward_loss);
    let total = g.add(total, continuation_loss);
    g.set_outputs(vec![
        total,
        image_loss,
        reward_loss,
        continuation_loss,
        output,
    ]);
    g
}

pub(super) fn sampling(c: &DenoisingConfig) -> Graph {
    let mut g = Graph::new();
    let b = c.batch_size;
    let input = g.input(
        "denoiser_input",
        &[b, (16 + 4 * c.action_count) * c.side * c.side],
    );
    let output = denoiser(&mut g, input, b, c.action_count, c.side);
    g.set_outputs(vec![output]);
    g
}
pub(super) fn event_sampling(c: &DenoisingConfig) -> Graph {
    let mut g = Graph::new();
    let b = c.batch_size;
    let input = g.input(
        "event_input",
        &[b, (15 + 4 * c.action_count) * c.side * c.side],
    );
    let (reward, continuation) = events(&mut g, input, b, c.action_count, c.side, c.value_bins);
    g.set_outputs(vec![reward, continuation]);
    g
}

pub(super) fn masked_mse(
    g: &mut Graph,
    output: NodeId,
    target: NodeId,
    weight: NodeId,
    batch: usize,
    width: usize,
) -> NodeId {
    let error = g.sub(output, target);
    let error = g.mul(error, error);
    masked_pixel_mean(g, error, weight, batch, width)
}

pub(super) fn weighted_mse(
    g: &mut Graph,
    output: NodeId,
    target: NodeId,
    weight: NodeId,
    importance: NodeId,
    batch: usize,
    width: usize,
) -> NodeId {
    let error = g.sub(output, target);
    let error = g.mul(error, error);
    let error = g.mul(error, importance);
    masked_pixel_mean(g, error, weight, batch, width)
}

fn masked_pixel_mean(
    g: &mut Graph,
    error: NodeId,
    weight: NodeId,
    batch: usize,
    width: usize,
) -> NodeId {
    let error = g.reshape(error, &[batch, width]);
    let error = g.mean_inner(error);
    let error = g.mul(error, weight);
    g.mean_all(error)
}
