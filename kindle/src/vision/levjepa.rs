//! Frozen LeVJEPA with exact block-causal, 16-frame chunk semantics.
//!
//! Each call processes one frame and appends its keys/values on the GPU. Patch
//! features equal the corresponding causal prefix of the released encoder:
//! spatial attention is bidirectional, temporal attention cannot see the future,
//! and patches never read the CLS sink. No CLS computation is therefore needed.
//! At a chunk boundary only the encoder's history restarts; the RSSM does not.
//! This is not a sliding-window approximation with indefinitely inherited KVs.

use std::{path::Path, sync::Arc};

use meganeura::{Graph, Mode, NodeId, Session, SessionConfig, data::safetensors::SafeTensorsModel};

use super::{
    OBSERVATION_CHANNELS, Observation, PROJECTION_SEED, fixed_projection, pool_2x2_token_major,
    preprocess,
};

pub const MODEL_ID: &str = "galilai-group/LeVJEPA-VideoMix-Large";
pub const CHECKPOINT_REV: &str = "e831a0347737fcaa660b39c57d41c109de399845";
pub const CHECKPOINT_SHA256: &str =
    "da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d";
pub const ENCODING_REV: &str = "levjepa-large-f32-chunk16-letterbox224-jl64-pool2-v1";
pub const FRAMES: usize = 16;
pub const IMAGE_SIZE: usize = 224;
pub const PATCH_SIZE: usize = 16;
pub const GRID: usize = IMAGE_SIZE / PATCH_SIZE;
pub const PATCHES: usize = GRID * GRID;
pub const HIDDEN: usize = 1024;
const HEAD_DIM: usize = 64;
const PATCH_DIM: usize = 3 * PATCH_SIZE * PATCH_SIZE;
type Error = Box<dyn std::error::Error>;

/// Independently trained encoder sizes; Large weights cannot initialize Tiny.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Tiny,
    Large,
}

impl Architecture {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Large => "large",
        }
    }

    pub const fn encoding_revision(self) -> &'static str {
        match self {
            Self::Tiny => "levjepa-tiny-f32-chunk16-letterbox224-jl64-pool2-v1",
            Self::Large => ENCODING_REV,
        }
    }

    pub const fn hidden(self) -> usize {
        match self {
            Self::Tiny => 192,
            Self::Large => HIDDEN,
        }
    }

    pub const fn layers(self) -> usize {
        match self {
            Self::Tiny => 12,
            Self::Large => 24,
        }
    }

    pub const fn heads(self) -> usize {
        self.hidden() / HEAD_DIM
    }

    /// Encoder weights including the pretraining-only CLS readout token.
    pub const fn parameter_count(self) -> usize {
        let hidden = self.hidden();
        self.layers() * (12 * hidden * hidden + 13 * hidden) + (PATCH_DIM + 4) * hidden
    }

    /// Logical F32 key/value elements for one stream, excluding allocator overhead.
    pub const fn cache_elements_per_stream(self) -> usize {
        2 * self.layers() * FRAMES * PATCHES * self.hidden()
    }
}

pub struct LeVJepaPerception {
    architecture: Architecture,
    session: Session,
    frames: Vec<usize>,
    projected: Vec<f32>,
    pooled: Vec<f32>,
}

impl LeVJepaPerception {
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }

    pub fn load(
        checkpoint: impl AsRef<Path>,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Error> {
        Self::load_batched(checkpoint, 1, gpu, plan_cache)
    }

    /// One set of frozen weights, batched dense layers, independent KV caches.
    pub fn load_batched(
        checkpoint: impl AsRef<Path>,
        streams: usize,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Error> {
        Self::load_batched_with_architecture(
            Architecture::Large,
            checkpoint,
            streams,
            gpu,
            plan_cache,
        )
    }

    /// Load a complete checkpoint for the explicitly selected architecture.
    /// Tiny is not a truncation of the released Large encoder.
    pub fn load_batched_with_architecture(
        architecture: Architecture,
        checkpoint: impl AsRef<Path>,
        streams: usize,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Error> {
        if streams == 0 {
            return Err("LeVJEPA needs at least one stream".into());
        }
        let model = SafeTensorsModel::load(checkpoint.as_ref().to_path_buf())?;
        validate_weights(&model, architecture)?;
        let gpu = match gpu {
            Some(gpu) => gpu,
            None => Arc::new(crate::init_gpu_context()?),
        };
        let mut graph = Graph::new();
        let tokens = build_encoder(&mut graph, streams, architecture);
        let projection = graph.constant(
            fixed_projection(architecture.hidden(), OBSERVATION_CHANNELS, PROJECTION_SEED),
            &[architecture.hidden(), OBSERVATION_CHANNELS],
        );
        let projected = graph.matmul(tokens, projection);
        graph.set_outputs(vec![projected, tokens]);
        let (mut session, _) = meganeura::build(
            &graph,
            SessionConfig {
                mode: Mode::Inference,
                gpu: Some(gpu),
                cache: plan_cache,
                runtime: meganeura::SessionOptions {
                    coop: meganeura::CoopPolicy::Disabled,
                    gpu_timing: meganeura::GpuOptions::from_env().timing,
                    ..Default::default()
                },
                ..SessionConfig::default()
            },
        );
        load_weights(&mut session, &model, streams, architecture)?;
        Ok(Self {
            architecture,
            session,
            frames: vec![0; streams],
            projected: vec![0.0; streams * PATCHES * OBSERVATION_CHANNELS],
            pooled: vec![0.0; streams * Observation::LEN],
        })
    }

    /// Forget all visual history at a real environment boundary. Stale cache
    /// rows need not be cleared: the valid prefix excludes them until overwritten.
    pub fn reset(&mut self) {
        assert_eq!(self.frames.len(), 1, "use per-stream resets for a batch");
        self.frames[0] = 0;
    }

    pub fn next_frame_in_chunk(&self) -> usize {
        assert_eq!(self.frames.len(), 1);
        self.frames[0]
    }

    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    pub fn gpu_memory_budget(&self) -> Option<crate::GpuMemoryBudget> {
        self.session
            .device_memory_stats()
            .map(|stats| crate::GpuMemoryBudget {
                usage_bytes: stats.usage_bytes,
                budget_bytes: stats.budget_bytes,
            })
    }

    pub fn encode_frame_rgb8(&mut self, rgb: &[u8], width: usize, height: usize) -> Observation {
        let rgb = preprocess::resize_letterbox_rgb8(rgb, width, height, IMAGE_SIZE);
        self.run(&preprocess::patches_from_rgb8(&rgb, IMAGE_SIZE, PATCH_SIZE))
    }

    pub fn encode_normalized_chw(&mut self, pixels: &[f32]) -> Observation {
        self.run(&preprocess::patches_from_pixels_chw(
            pixels, IMAGE_SIZE, PATCH_SIZE,
        ))
    }

    pub fn projected_patches(&self) -> &[f32] {
        &self.projected
    }

    /// Dense current-frame features for numerical verification. Production
    /// reads only the much smaller projected output.
    pub fn patch_tokens(&self) -> Vec<f32> {
        let mut tokens = vec![0.0; self.frames.len() * PATCHES * self.architecture.hidden()];
        self.session.read_output_by_index(1, &mut tokens);
        tokens
    }

    fn run(&mut self, patches: &[f32]) -> Observation {
        assert_eq!(self.frames.len(), 1);
        self.run_batch(patches, &[0]).pop().unwrap()
    }

    /// Encode selected stream arrivals together. Omitted streams do not advance.
    /// Reset flags belong to real environment boundaries, not vector ticks.
    pub fn encode_frames_rgb8(
        &mut self,
        arrivals: &[(usize, &crate::RgbFrame, bool)],
    ) -> Vec<Observation> {
        let mut present = vec![false; self.frames.len()];
        for &(stream, _, _) in arrivals {
            assert!(
                stream < present.len() && !present[stream],
                "invalid or repeated stream"
            );
            present[stream] = true;
        }
        if arrivals.is_empty() {
            return Vec::new();
        }
        let width = PATCHES * PATCH_DIM;
        let mut patches = vec![0.0; self.frames.len() * width];
        for &(stream, frame, reset) in arrivals {
            let rgb = preprocess::resize_letterbox_rgb8(
                frame.pixels(),
                frame.width(),
                frame.height(),
                IMAGE_SIZE,
            );
            patches[stream * width..(stream + 1) * width]
                .copy_from_slice(&preprocess::patches_from_rgb8(&rgb, IMAGE_SIZE, PATCH_SIZE));
            if reset {
                self.frames[stream] = 0;
            }
        }
        self.run_batch(&patches, &arrivals.iter().map(|a| a.0).collect::<Vec<_>>())
    }

    fn run_batch(&mut self, patches: &[f32], active: &[usize]) -> Vec<Observation> {
        self.session.set_input("patches", patches);
        for (stream, &frame) in self.frames.iter().enumerate() {
            self.session
                .set_input_u32(&format!("frame.{stream}"), &[frame as u32]);
            self.session.set_input_u32(
                &format!("last_token.{stream}"),
                &[((frame + 1) * PATCHES - 1) as u32],
            );
        }
        self.session.step();
        self.session.wait();
        self.session.read_output_by_index(0, &mut self.projected);
        // Inactive rows write only their next, unused cache slot. Before that
        // stream consumes it, a real arrival overwrites it at the same position.
        active
            .iter()
            .map(|&stream| {
                let projected_width = PATCHES * OBSERVATION_CHANNELS;
                let pooled =
                    &mut self.pooled[stream * Observation::LEN..(stream + 1) * Observation::LEN];
                pool_2x2_token_major(
                    &self.projected[stream * projected_width..(stream + 1) * projected_width],
                    GRID,
                    OBSERVATION_CHANNELS,
                    pooled,
                );
                self.frames[stream] = (self.frames[stream] + 1) % FRAMES;
                Observation::from_vec(pooled.to_vec())
            })
            .collect()
    }
}

fn linear(g: &mut Graph, x: NodeId, name: &str, input: usize, output: usize) -> NodeId {
    let weight = g.parameter(&format!("{name}.weight"), &[input, output]);
    let bias = g.parameter(&format!("{name}.bias"), &[output]);
    let y = g.matmul(x, weight);
    g.bias_add(y, bias)
}

fn norm(g: &mut Graph, x: NodeId, name: &str, hidden: usize) -> NodeId {
    let weight = g.parameter(&format!("{name}.weight"), &[hidden]);
    let bias = g.parameter(&format!("{name}.bias"), &[hidden]);
    g.layer_norm(x, weight, bias, 1e-6)
}

fn gelu_erf(g: &mut Graph, x: NodeId) -> NodeId {
    // The release uses erf GELU; Meganeura's gelu() is the tanh variant.
    // Abramowitz–Stegun 7.1.26 approximates erf with absolute error <1.5e-7.
    let absolute = g.abs(x);
    let scaled = scale(g, absolute, 0.327_591_1 * std::f32::consts::FRAC_1_SQRT_2);
    let denominator = shift(g, scaled, 1.0);
    let t = g.recip(denominator);
    let mut polynomial = scale(g, t, 1.061_405_4);
    for coefficient in [-1.453_152_1, 1.421_413_8, -0.284_496_72, 0.254_829_6] {
        polynomial = shift(g, polynomial, coefficient);
        polynomial = g.mul(polynomial, t);
    }
    let square = g.mul(x, x);
    let exponent = scale(g, square, -0.5);
    let exponential = g.exp(exponent);
    let tail = g.mul(polynomial, exponential);
    let negative_tail = g.neg(tail);
    let erf_absolute = shift(g, negative_tail, 1.0);
    let zero = scale(g, x, 0.0);
    let positive = g.greater(x, zero);
    let twice_positive = scale(g, positive, 2.0);
    let sign = shift(g, twice_positive, -1.0);
    let erf = g.mul(sign, erf_absolute);
    let cdf = shift(g, erf, 1.0);
    let half_x = scale(g, x, 0.5);
    g.mul(half_x, cdf)
}

fn scale(g: &mut Graph, x: NodeId, value: f32) -> NodeId {
    let shape = g.node(x).ty.shape.clone();
    let len = g.node(x).ty.num_elements();
    let x = g.reshape(x, &[len]);
    let scalar = g.scalar(value);
    let result = g.mul_per_channel(x, scalar, 1, len as u32);
    g.reshape(result, &shape)
}

fn shift(g: &mut Graph, x: NodeId, value: f32) -> NodeId {
    let shape = g.node(x).ty.shape.clone();
    let len = g.node(x).ty.num_elements();
    let x = g.reshape(x, &[len]);
    let scalar = g.scalar(value);
    let result = g.add_per_channel(x, scalar, 1, len as u32);
    g.reshape(result, &shape)
}

fn rope(g: &mut Graph, x: NodeId, cos: NodeId, sin: NodeId) -> NodeId {
    // The release rotates adjacent pairs, but repeats each axis's frequency
    // vector in two halves. Do not substitute standard interleaved RoPE.
    let shape = g.node(x).ty.shape.clone();
    let pairs = (g.node(x).ty.num_elements() / 2) as u32;
    let even = g.split_a(x, pairs, 1, 1, 1);
    let odd = g.split_b(x, pairs, 1, 1, 1);
    let negative_odd = g.neg(odd);
    let rotated = g.concat(negative_odd, even, pairs, 1, 1, 1);
    let rotated = g.reshape(rotated, &shape);
    let a = g.mul(x, cos);
    let b = g.mul(rotated, sin);
    g.add(a, b)
}

fn rope_tables(architecture: Architecture) -> (Vec<f32>, Vec<f32>) {
    let hidden = architecture.hidden();
    let axis_dim = 2 * ((HEAD_DIM / 3) / 2);
    let mut cos = vec![1.0; FRAMES * PATCHES * hidden];
    let mut sin = vec![0.0; cos.len()];
    for frame in 0..FRAMES {
        for patch in 0..PATCHES {
            let positions = [frame, patch / GRID, patch % GRID];
            for head in 0..architecture.heads() {
                for (axis, position) in positions.into_iter().enumerate() {
                    for d in 0..axis_dim {
                        let frequency = 1.0
                            / 10_000.0_f32
                                .powf((d % (axis_dim / 2)) as f32 / (axis_dim / 2) as f32);
                        let angle = position as f32 * frequency;
                        let index = (frame * PATCHES + patch) * hidden
                            + head * HEAD_DIM
                            + axis * axis_dim
                            + d;
                        cos[index] = angle.cos();
                        sin[index] = angle.sin();
                    }
                }
            }
        }
    }
    (cos, sin)
}

fn stream_rows(g: &mut Graph, x: NodeId, stream: usize, streams: usize, width: usize) -> NodeId {
    let mut x = x;
    if stream > 0 {
        x = g.split_b(x, 1, stream as u32, (streams - stream) as u32, width as u32);
    }
    if stream + 1 < streams {
        x = g.split_a(x, 1, 1, (streams - stream - 1) as u32, width as u32);
    }
    x
}

fn stack_streams(g: &mut Graph, rows: &[NodeId], width: usize) -> NodeId {
    let mut x = rows[0];
    for (stream, &row) in rows.iter().enumerate().skip(1) {
        x = g.concat(x, row, 1, stream as u32, 1, width as u32);
    }
    x
}

fn build_encoder(g: &mut Graph, streams: usize, architecture: Architecture) -> NodeId {
    let hidden = architecture.hidden();
    let heads = architecture.heads();
    let rows = streams * PATCHES;
    let input = g.input("patches", &[rows, PATCH_DIM]);
    let frames: Vec<_> = (0..streams)
        .map(|s| g.input_u32(&format!("frame.{s}"), &[1]))
        .collect();
    let last_tokens: Vec<_> = (0..streams)
        .map(|s| g.input_u32(&format!("last_token.{s}"), &[1]))
        .collect();
    let (cos, sin) = rope_tables(architecture);
    let cos = g.constant(cos, &[FRAMES, PATCHES * hidden]);
    let sin = g.constant(sin, &[FRAMES, PATCHES * hidden]);
    let cos: Vec<_> = frames.iter().map(|&f| g.embedding(f, cos)).collect();
    let sin: Vec<_> = frames.iter().map(|&f| g.embedding(f, sin)).collect();
    let cos = stack_streams(g, &cos, PATCHES * hidden);
    let sin = stack_streams(g, &sin, PATCHES * hidden);
    let cos = g.reshape(cos, &[rows, hidden]);
    let sin = g.reshape(sin, &[rows, hidden]);
    let mut x = linear(g, input, "encoder.patch_embed.proj", PATCH_DIM, hidden);
    for layer in 0..architecture.layers() {
        let name = format!("encoder.blocks.{layer}");
        let n = norm(g, x, &format!("{name}.norm1"), hidden);
        let qkv = linear(g, n, &format!("{name}.attn.qkv"), hidden, 3 * hidden);
        let q = g.split_a(qkv, rows as u32, hidden as u32, (2 * hidden) as u32, 1);
        let kv = g.split_b(qkv, rows as u32, hidden as u32, (2 * hidden) as u32, 1);
        let k = g.split_a(kv, rows as u32, hidden as u32, hidden as u32, 1);
        let v = g.split_b(kv, rows as u32, hidden as u32, hidden as u32, 1);
        let q = g.reshape(q, &[rows, hidden]);
        let k = g.reshape(k, &[rows, hidden]);
        let q = rope(g, q, cos, sin);
        let k = rope(g, k, cos, sin);
        let attention: Vec<_> = (0..streams)
            .map(|stream| {
                let q = stream_rows(g, q, stream, streams, PATCHES * hidden);
                let k = stream_rows(g, k, stream, streams, PATCHES * hidden);
                let v = stream_rows(g, v, stream, streams, PATCHES * hidden);
                let q = g.reshape(q, &[PATCHES, hidden]);
                let k = g.reshape(k, &[1, PATCHES * hidden]);
                let v = g.reshape(v, &[1, PATCHES * hidden]);
                let k_cache = g.parameter(
                    &format!("cache.{layer}.{stream}.k"),
                    &[FRAMES, PATCHES * hidden],
                );
                let v_cache = g.parameter(
                    &format!("cache.{layer}.{stream}.v"),
                    &[FRAMES, PATCHES * hidden],
                );
                let k = g.cache_write(k, k_cache, frames[stream]);
                let v = g.cache_write(v, v_cache, frames[stream]);
                let k = g.reshape(k, &[FRAMES * PATCHES, hidden]);
                let v = g.reshape(v, &[FRAMES * PATCHES, hidden]);
                g.cached_attention(
                    q,
                    k,
                    v,
                    last_tokens[stream],
                    heads as u32,
                    heads as u32,
                    HEAD_DIM as u32,
                )
            })
            .collect();
        let attention = stack_streams(g, &attention, PATCHES * hidden);
        let attention = g.reshape(attention, &[rows, hidden]);
        let attention = linear(g, attention, &format!("{name}.attn.proj"), hidden, hidden);
        x = g.add(x, attention);
        let n = norm(g, x, &format!("{name}.norm2"), hidden);
        let mlp = linear(g, n, &format!("{name}.mlp.fc1"), hidden, 4 * hidden);
        let mlp = gelu_erf(g, mlp);
        let mlp = linear(g, mlp, &format!("{name}.mlp.fc2"), 4 * hidden, hidden);
        x = g.add(x, mlp);
    }
    norm(g, x, "encoder.norm", hidden)
}

fn weight_shapes(architecture: Architecture) -> Vec<(String, Vec<usize>)> {
    let hidden = architecture.hidden();
    let mut shapes = vec![
        ("encoder.cls_token".to_owned(), vec![1, 1, hidden]),
        (
            "encoder.patch_embed.proj.weight".to_owned(),
            vec![hidden, 3, 1, PATCH_SIZE, PATCH_SIZE],
        ),
        ("encoder.patch_embed.proj.bias".to_owned(), vec![hidden]),
        ("encoder.norm.weight".to_owned(), vec![hidden]),
        ("encoder.norm.bias".to_owned(), vec![hidden]),
    ];
    for layer in 0..architecture.layers() {
        let name = format!("encoder.blocks.{layer}");
        for norm in ["norm1", "norm2"] {
            for part in ["weight", "bias"] {
                shapes.push((format!("{name}.{norm}.{part}"), vec![hidden]));
            }
        }
        for (part, input, output) in [
            ("attn.qkv", hidden, 3 * hidden),
            ("attn.proj", hidden, hidden),
            ("mlp.fc1", hidden, 4 * hidden),
            ("mlp.fc2", 4 * hidden, hidden),
        ] {
            shapes.push((format!("{name}.{part}.weight"), vec![output, input]));
            shapes.push((format!("{name}.{part}.bias"), vec![output]));
        }
    }
    shapes
}

fn validate_weights(model: &SafeTensorsModel, architecture: Architecture) -> Result<(), Error> {
    let shapes = weight_shapes(architecture);
    if model.tensor_info().len() != shapes.len() {
        return Err(format!(
            "LeVJEPA expects {} tensors, got {}",
            shapes.len(),
            model.tensor_info().len()
        )
        .into());
    }
    for (name, shape) in shapes {
        let info = model
            .tensor_info()
            .get(&name)
            .ok_or_else(|| format!("missing LeVJEPA tensor {name}"))?;
        if info.shape != shape {
            return Err(format!("LeVJEPA {name}: expected {shape:?}, got {:?}", info.shape).into());
        }
    }
    Ok(())
}

fn load_weights(
    session: &mut Session,
    model: &SafeTensorsModel,
    streams: usize,
    architecture: Architecture,
) -> Result<(), Error> {
    for (name, shape) in weight_shapes(architecture) {
        if name == "encoder.cls_token" {
            continue;
        }
        let values = if shape.len() > 1 {
            let values = model.tensor_f32_auto(&name)?;
            preprocess::conv_weight_to_matmul(&values, shape[0], values.len() / shape[0])
        } else {
            model.tensor_f32_auto(&name)?
        };
        if !values.iter().all(|v| v.is_finite()) {
            return Err(format!("non-finite LeVJEPA tensor {name}").into());
        }
        session.set_parameter(&name, &values);
    }
    let zeros = vec![0.0; FRAMES * PATCHES * architecture.hidden()];
    for layer in 0..architecture.layers() {
        for stream in 0..streams {
            session.set_parameter(&format!("cache.{layer}.{stream}.k"), &zeros);
            session.set_parameter(&format!("cache.{layer}.{stream}.v"), &zeros);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_architecture_and_rope_layout() {
        let shapes = weight_shapes(Architecture::Large);
        assert_eq!(shapes.len(), 293);
        assert_eq!(
            shapes
                .iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            303_099_904
        );
        let (cos, sin) = rope_tables(Architecture::Large);
        let at = |frame, patch, head, d| (frame * PATCHES + patch) * HIDDEN + head * HEAD_DIM + d;
        assert_eq!(cos[at(0, 0, 0, 0)], 1.0);
        assert_eq!(sin[at(0, 0, 0, 0)], 0.0);
        assert_eq!(cos[at(3, 0, 0, 0)], 3.0_f32.cos());
        assert_eq!(cos[at(3, 0, 0, 10)], 3.0_f32.cos());
        assert_eq!(cos[at(0, 2 * GRID + 5, 0, 20)], 2.0_f32.cos());
        assert_eq!(sin[at(0, 2 * GRID + 5, 0, 40)], 5.0_f32.sin());
        assert_eq!(sin[at(15, 195, 15, 63)], 0.0);
        assert_eq!(cos[at(7, 17, 0, 29)], cos[at(7, 17, 15, 29)]);
    }

    #[test]
    fn streaming_graph_builds_without_cls() {
        let mut graph = Graph::new();
        let output = build_encoder(&mut graph, 1, Architecture::Large);
        assert_eq!(graph.node(output).ty.shape, [PATCHES, HIDDEN]);
        assert!(!graph.nodes().iter().any(
            |node| matches!(&node.op, meganeura::graph::Op::Parameter {name} if name.contains("cls"))
        ));
    }

    #[test]
    fn vector_graph_shares_weights_but_not_history() {
        let layers = Architecture::Large.layers();
        let mut expected_weights = None;
        for streams in [1, 2, 3, 4, 6, 8] {
            let mut graph = Graph::new();
            let output = build_encoder(&mut graph, streams, Architecture::Large);
            assert_eq!(graph.node(output).ty.shape, [streams * PATCHES, HIDDEN]);
            let parameters: Vec<_> = graph
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    meganeura::graph::Op::Parameter { name } => Some((name, node)),
                    _ => None,
                })
                .collect();
            let weights = parameters
                .iter()
                .filter(|(name, _)| name.starts_with("encoder."))
                .map(|(name, node)| ((*name).clone(), node.ty.clone()))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(weights.len(), 292);
            if let Some(expected) = &expected_weights {
                assert_eq!(&weights, expected);
            } else {
                expected_weights = Some(weights);
            }
            let caches = parameters
                .iter()
                .filter(|(name, _)| name.starts_with("cache."))
                .map(|(name, node)| ((*name).clone(), *node))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(parameters.len(), 292 + streams * layers * 2);
            assert_eq!(caches.len(), streams * layers * 2);
            for layer in 0..layers {
                for stream in 0..streams {
                    for kind in ["k", "v"] {
                        let cache = caches[&format!("cache.{layer}.{stream}.{kind}")];
                        assert_eq!(cache.ty.shape, [FRAMES, PATCHES * HIDDEN]);
                        assert_eq!(cache.ty.dtype, meganeura::DType::F32);
                    }
                }
            }
            let cache_bytes = caches
                .values()
                .map(|node| node.ty.num_elements() * size_of::<f32>())
                .sum::<usize>();
            assert_eq!(cache_bytes, streams * 588 * 1024 * 1024);
            eprintln!(
                "streams={streams} logical_cache_mib={}",
                cache_bytes / 1024 / 1024
            );
        }
    }

    #[test]
    fn tiny_size_is_independent_and_keeps_the_observation_contract() {
        let architecture = Architecture::Tiny;
        assert_eq!(architecture.name(), "tiny");
        assert_ne!(architecture.encoding_revision(), ENCODING_REV);
        assert_eq!(Architecture::Large.encoding_revision(), ENCODING_REV);
        assert_eq!(
            (
                architecture.hidden(),
                architecture.layers(),
                architecture.heads()
            ),
            (192, 12, 3)
        );
        assert_eq!(architecture.parameter_count(), 5_486_592);
        assert_eq!(Architecture::Large.parameter_count(), 303_099_904);
        assert_eq!(architecture.cache_elements_per_stream() * 4, 57_802_752);
        for model in [architecture, Architecture::Large] {
            let shapes = weight_shapes(model);
            assert_eq!(shapes.len(), 5 + 12 * model.layers());
            assert_eq!(
                shapes
                    .iter()
                    .map(|(_, shape)| shape.iter().product::<usize>())
                    .sum::<usize>(),
                model.parameter_count()
            );
        }
        for streams in [1, 6] {
            let mut graph = Graph::new();
            let output = build_encoder(&mut graph, streams, architecture);
            assert_eq!(
                graph.node(output).ty.shape,
                [streams * PATCHES, architecture.hidden()]
            );
            let mut weights = 0;
            let mut caches = 0;
            for node in graph.nodes() {
                if let meganeura::graph::Op::Parameter { name } = &node.op {
                    if name.starts_with("cache.") {
                        assert_eq!(node.ty.shape, [FRAMES, PATCHES * architecture.hidden()]);
                        caches += node.ty.num_elements();
                    } else {
                        assert!(name.starts_with("encoder.") && !name.contains("cls"));
                        weights += node.ty.num_elements();
                    }
                }
            }
            assert_eq!(
                weights,
                architecture.parameter_count() - architecture.hidden()
            );
            assert_eq!(caches, streams * architecture.cache_elements_per_stream());
            let projection = graph.constant(
                fixed_projection(architecture.hidden(), OBSERVATION_CHANNELS, PROJECTION_SEED),
                &[architecture.hidden(), OBSERVATION_CHANNELS],
            );
            let projected = graph.matmul(output, projection);
            assert_eq!(
                graph.node(projected).ty.shape,
                [streams * PATCHES, OBSERVATION_CHANNELS]
            );
        }
    }

    #[test]
    fn tiny_rope_keeps_each_head_and_causal_position_identical() {
        let architecture = Architecture::Tiny;
        let (small_cos, small_sin) = rope_tables(architecture);
        let (large_cos, large_sin) = rope_tables(Architecture::Large);
        for token in 0..FRAMES * PATCHES {
            for head in 0..architecture.heads() {
                for channel in 0..HEAD_DIM {
                    let small = token * architecture.hidden() + head * HEAD_DIM + channel;
                    let large = token * HIDDEN + head * HEAD_DIM + channel;
                    assert_eq!(small_cos[small], large_cos[large]);
                    assert_eq!(small_sin[small], large_sin[large]);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires GPU and pinned LeVJEPA weights"]
    fn batched_streams_match_serial_with_asymmetric_resets_and_gaps() {
        check_batched_streams_match_serial(2);
    }

    #[test]
    #[ignore = "requires GPU and pinned LeVJEPA weights; run after the active pilot"]
    fn memory_candidate_streams_match_serial() {
        for streams in [4, 6, 8] {
            check_batched_streams_match_serial(streams);
        }
    }

    fn check_batched_streams_match_serial(streams: usize) {
        let checkpoint =
            std::env::var_os("KINDLE_LEVJEPA_WEIGHTS").expect("set KINDLE_LEVJEPA_WEIGHTS");
        assert_eq!(
            super::super::checkpoint_sha256(Path::new(&checkpoint)).unwrap(),
            CHECKPOINT_SHA256
        );
        check_batched_streams(streams, Architecture::Large, Path::new(&checkpoint));
    }

    fn check_reference_device(perception: &LeVJepaPerception) {
        assert_eq!(std::env::var("MEGANEURA_DEVICE_ID").unwrap(), "0x2c02");
        assert_eq!(std::env::var("KINDLE_GPU_DRIVER").unwrap(), "580.178.04");
        let device = perception.gpu_device();
        assert_eq!(device.device_name, "NVIDIA GeForce RTX 5080");
        assert_eq!(device.driver_name, "NVIDIA");
        assert_eq!(device.driver_info, "580.178.04");
        assert!(!device.is_software_emulated);
        let memory = perception.session.device_memory_stats().unwrap();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 * 1024 * 1024 * 1024);
        eprintln!(
            "levjepa_reference_memory: usage={} budget={}",
            memory.usage_bytes, memory.budget_bytes
        );
    }

    fn check_batched_streams(streams: usize, architecture: Architecture, checkpoint: &Path) {
        let hidden = architecture.hidden();
        let frame = |stream: usize, tick: usize| {
            crate::RgbFrame::new(
                64,
                64,
                (0..64 * 64 * 3)
                    .map(|i| ((i * 37 + stream * 71 + tick * 13) % 256) as u8)
                    .collect(),
            )
        };
        let active = |stream, tick| stream == 0 || ![4, 15, 16, 32].contains(&tick);
        let reset = |stream, tick| tick == 0 || (stream > 0 && tick == 7 * stream);
        let ticks = (7 * streams).max(36);
        let mut expected = Vec::new();
        {
            let mut serial = LeVJepaPerception::load_batched_with_architecture(
                architecture,
                checkpoint,
                1,
                None,
                None,
            )
            .unwrap();
            check_reference_device(&serial);
            for stream in 0..streams {
                serial.reset();
                let mut values = Vec::new();
                for tick in 0..ticks {
                    if !active(stream, tick) {
                        values.push(None);
                        continue;
                    }
                    if reset(stream, tick) {
                        serial.reset();
                    }
                    let frame = frame(stream, tick);
                    let observation =
                        serial.encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
                    values.push(Some((observation, serial.patch_tokens())));
                }
                expected.push(values);
                check_reference_device(&serial);
            }
        }
        let mut batch = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            checkpoint,
            streams,
            None,
            None,
        )
        .unwrap();
        check_reference_device(&batch);
        let mut worst = 0.0_f32;
        for tick in 0..ticks {
            let frames = (0..streams)
                .map(|stream| frame(stream, tick))
                .collect::<Vec<_>>();
            // Reversed input order must not change stream ownership.
            let arrivals: Vec<_> = (0..streams)
                .rev()
                .filter(|&s| active(s, tick))
                .map(|s| (s, &frames[s], reset(s, tick)))
                .collect();
            let observations = batch.encode_frames_rgb8(&arrivals);
            let tokens = batch.patch_tokens();
            assert_eq!(observations.len(), arrivals.len());
            assert_eq!(tokens.len(), streams * PATCHES * hidden);
            for ((stream, _, _), observation) in arrivals.iter().zip(observations) {
                let (reference, reference_tokens) = expected[*stream][tick].as_ref().unwrap();
                assert_eq!(observation.as_slice().len(), reference.as_slice().len());
                assert_eq!(reference_tokens.len(), PATCHES * hidden);
                for (actual, expected) in observation.as_slice().iter().zip(reference.as_slice()) {
                    assert!(
                        (actual - expected).abs() < 0.005,
                        "pooled N{streams} stream {stream} tick {tick}"
                    );
                }
                let actual = &tokens[stream * PATCHES * hidden..(stream + 1) * PATCHES * hidden];
                let mut error = 0.0_f64;
                let mut energy = 0.0_f64;
                for (&actual, &expected) in actual.iter().zip(reference_tokens) {
                    worst = worst.max((actual - expected).abs());
                    error += f64::from(actual - expected).powi(2);
                    energy += f64::from(expected).powi(2);
                }
                assert!(
                    (error / energy).sqrt() < 1e-4,
                    "dense N{streams} stream {stream} tick {tick}"
                );
            }
            check_reference_device(&batch);
        }
        assert!(
            worst < 0.005,
            "N{streams} batched maximum absolute error {worst}"
        );
        eprintln!("LeVJEPA N{streams} batched/serial maximum absolute error {worst}");
    }

    #[test]
    #[ignore = "requires a separately declared exclusive Large N6 GPU comparison"]
    fn large_six_streams_match_serial_with_resets_and_gaps() {
        check_batched_streams_match_serial(6);
    }

    fn tiny_reference() -> (std::path::PathBuf, SafeTensorsModel) {
        use sha2::{Digest, Sha256};
        let root =
            std::path::PathBuf::from(std::env::var_os("KINDLE_TINY_STREAM_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["format"], 1);
        assert_eq!(manifest["architecture"], "tiny");
        for (name, source) in [
            (
                "levjepa_tiny_streaming_reference.py",
                &include_bytes!("../../../python/examples/levjepa_tiny_streaming_reference.py")[..],
            ),
            (
                "levjepa_tiny_reference.py",
                &include_bytes!("../../../python/examples/levjepa_tiny_reference.py")[..],
            ),
            (
                "levjepa_reference.py",
                &include_bytes!("../../../python/examples/levjepa_reference.py")[..],
            ),
        ] {
            assert_eq!(
                format!("{:x}", Sha256::digest(source)),
                manifest["sources"][name].as_str().unwrap()
            );
        }
        for name in ["encoder.safetensors", "reference.safetensors"] {
            assert_eq!(
                super::super::checkpoint_sha256(&root.join(name)).unwrap(),
                manifest["files"][name].as_str().unwrap()
            );
        }
        let checkpoint = root.join("encoder.safetensors");
        let weights = SafeTensorsModel::load(checkpoint.clone()).unwrap();
        validate_weights(&weights, Architecture::Tiny).unwrap();
        let fixture = SafeTensorsModel::load(root.join("reference.safetensors")).unwrap();
        for (name, shape) in [
            ("rgb", vec![2, 16, 64, 80, 3]),
            ("pixels", vec![2, 16, 3, 224, 224]),
            ("tokens", vec![2, 16, 196, 192]),
            ("projected", vec![2, 16, 196, 64]),
            ("pooled", vec![2, 16, 49, 64]),
        ] {
            assert_eq!(fixture.tensor_info()[name].shape, shape);
            assert!(
                fixture
                    .tensor_f32(name)
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
        (checkpoint, fixture)
    }

    #[test]
    #[ignore = "requires KINDLE_TINY_STREAM_REFERENCE; CPU only"]
    fn tiny_streaming_reference_is_complete() {
        tiny_reference();
    }

    #[test]
    #[ignore = "requires separately declared exclusive GPU and Tiny streaming reference"]
    fn tiny_six_streams_match_serial_with_resets_and_gaps() {
        assert_eq!(std::env::var("MEGANEURA_DEVICE_ID").unwrap(), "0x2c02");
        assert_eq!(std::env::var("KINDLE_GPU_DRIVER").unwrap(), "580.178.04");
        let (checkpoint, _) = tiny_reference();
        check_batched_streams(6, Architecture::Tiny, &checkpoint);
    }

    #[test]
    #[ignore = "requires separately declared exclusive GPU and Tiny streaming reference"]
    fn tiny_checkpoint_matches_dense_causal_reference_and_resets() {
        assert_eq!(std::env::var("MEGANEURA_DEVICE_ID").unwrap(), "0x2c02");
        assert_eq!(std::env::var("KINDLE_GPU_DRIVER").unwrap(), "580.178.04");
        let (checkpoint, fixture) = tiny_reference();
        check_causal_reference(Architecture::Tiny, &checkpoint, fixture);
    }

    #[test]
    #[ignore = "requires GPU, pinned LeVJEPA weights and PyTorch parity fixture"]
    fn checkpoint_matches_causal_reference_and_resets() {
        let checkpoint =
            std::env::var_os("KINDLE_LEVJEPA_WEIGHTS").expect("set KINDLE_LEVJEPA_WEIGHTS");
        let reference =
            std::env::var_os("KINDLE_LEVJEPA_REFERENCE").expect("set KINDLE_LEVJEPA_REFERENCE");
        assert_eq!(
            super::super::checkpoint_sha256(Path::new(&checkpoint)).unwrap(),
            CHECKPOINT_SHA256
        );
        let fixture = SafeTensorsModel::load(reference.into()).unwrap();
        check_causal_reference(Architecture::Large, Path::new(&checkpoint), fixture);
    }

    fn check_causal_reference(
        architecture: Architecture,
        checkpoint: &Path,
        fixture: SafeTensorsModel,
    ) {
        let hidden = architecture.hidden();
        assert_eq!(
            fixture.tensor_info()["pixels"].shape,
            [2, FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE]
        );
        assert_eq!(
            fixture.tensor_info()["tokens"].shape,
            [2, FRAMES, PATCHES, hidden]
        );
        let pixels = fixture.tensor_f32("pixels").unwrap();
        let rgb_shape = &fixture.tensor_info()["rgb"].shape;
        assert_eq!(rgb_shape.len(), 5);
        assert_eq!(&rgb_shape[..2], &[2, FRAMES]);
        assert_eq!(rgb_shape[4], 3);
        let (height, width) = (rgb_shape[2], rgb_shape[3]);
        let rgb: Vec<u8> = fixture
            .tensor_f32("rgb")
            .unwrap()
            .into_iter()
            .map(|value| {
                assert!(
                    value.is_finite() && (0.0..=255.0).contains(&value) && value.fract() == 0.0
                );
                value as u8
            })
            .collect();
        let expected = fixture.tensor_f32("tokens").unwrap();
        let expected_projected = fixture.tensor_f32("projected").unwrap();
        let expected_pooled = fixture.tensor_f32("pooled").unwrap();
        let mut perception = LeVJepaPerception::load_batched_with_architecture(
            architecture,
            checkpoint,
            1,
            None,
            None,
        )
        .unwrap();
        check_reference_device(&perception);
        let pixel_len = 3 * IMAGE_SIZE * IMAGE_SIZE;
        let token_len = PATCHES * hidden;
        let rgb_len = width * height * 3;
        // Two full chunks verify the automatic boundary. Rewind only this
        // synthetic fixture, not an environment, to exercise explicit reset.
        for (step, source) in (0..2 * FRAMES).chain(0..3).chain(0..2).enumerate() {
            if step == 2 * FRAMES || step == 2 * FRAMES + 3 {
                perception.reset();
            }
            let frame = &rgb[source * rgb_len..(source + 1) * rgb_len];
            let resized = preprocess::resize_letterbox_rgb8(frame, width, height, IMAGE_SIZE);
            let actual_patches = preprocess::patches_from_rgb8(&resized, IMAGE_SIZE, PATCH_SIZE);
            let expected_patches = preprocess::patches_from_pixels_chw(
                &pixels[source * pixel_len..(source + 1) * pixel_len],
                IMAGE_SIZE,
                PATCH_SIZE,
            );
            assert_eq!(
                actual_patches, expected_patches,
                "RGB preprocessing differs at step {step}"
            );
            let started = std::time::Instant::now();
            let observation = perception.encode_frame_rgb8(frame, width, height);
            let elapsed = started.elapsed().as_secs_f64();
            let actual = perception.patch_tokens();
            let reference = &expected[source * token_len..(source + 1) * token_len];
            let mut error_sq = 0.0_f64;
            let mut reference_sq = 0.0_f64;
            let mut worst = 0.0_f32;
            for (&actual, &reference) in actual.iter().zip(reference) {
                assert!(actual.is_finite());
                let difference = actual - reference;
                error_sq += f64::from(difference).powi(2);
                reference_sq += f64::from(reference).powi(2);
                worst = worst.max(difference.abs());
            }
            let relative = (error_sq / reference_sq).sqrt();
            eprintln!(
                "LeVJEPA F32 source={source} step={step} seconds={elapsed:.6} relative_l2={relative:.8} max_abs={worst:.8}"
            );
            assert!(
                relative < 1e-4 && worst < 0.005,
                "LeVJEPA parity: relative {relative}, max {worst}"
            );
            for (actual, reference, label) in [
                (
                    perception.projected_patches(),
                    expected_projected.as_slice(),
                    "projected",
                ),
                (observation.as_slice(), expected_pooled.as_slice(), "pooled"),
            ] {
                let length = actual.len();
                let reference = &reference[source * length..(source + 1) * length];
                let worst = actual
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(worst < 0.005, "{label} step {step}, max error {worst}");
            }
            check_reference_device(&perception);
        }
    }
}
