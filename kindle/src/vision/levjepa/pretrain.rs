//! Native Tiny LeVJEPA pretraining graph: shared causal encoder, multi-view
//! invariance and SIGReg. No teacher, predictor, rewards or policy updates.
//!
//! Token-major batches make independent clip/head attention column-block
//! products. Global and local views share the exact same parameter nodes.
//! Video decoding, crop sampling and selecting retained patches happen outside
//! this graph; every learned operation and its derivative runs on Meganeura.

use std::collections::BTreeMap;

use meganeura::{Graph, NodeId};

mod trainer;
pub use trainer::{Batch, Metrics, Trainer, TrainingConfig};

use super::{Architecture, FRAMES, GRID, HEAD_DIM, PATCH_DIM, rope, rope_tables_for_grid};

const ARCHITECTURE: Architecture = Architecture::Tiny;
const KNOTS: usize = 17;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub batch: usize,
    pub local_views: usize,
    pub projector_hidden: usize,
    pub projector_output: usize,
    pub directions: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch: 128,
            local_views: 4,
            projector_hidden: 1024,
            projector_output: 128,
            directions: 1024,
        }
    }
}

impl Config {
    pub fn validate(self) -> Result<(), &'static str> {
        if !(2..=128).contains(&self.batch)
            || !(1..=4).contains(&self.local_views)
            || !(1..=2048).contains(&self.projector_hidden)
            || !(1..=256).contains(&self.projector_output)
            || !(1..=1024).contains(&self.directions)
        {
            return Err("unsupported Tiny pretraining batch/projector dimensions");
        }
        Ok(())
    }

    pub fn global(self) -> View {
        View::new(self.batch, GRID)
    }

    pub fn local(self) -> View {
        View::new(self.batch * self.local_views, 6)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct View {
    pub clips: usize,
    pub grid: usize,
    pub kept: usize,
}

impl View {
    fn new(clips: usize, grid: usize) -> Self {
        Self {
            clips,
            grid,
            // Exactly 95% dropping, rounded as in the reference (157 / 29).
            kept: (FRAMES * grid * grid + 10) / 20,
        }
    }

    /// Validate and retain original, unsorted token IDs in `[kept, clips]`
    /// order. Returns RoPE indices (CLS first) and an additive attention mask
    /// in `[query, clip, head, key]` order. No GPU access.
    pub fn attention_inputs(self, patch_ids: &[u32]) -> Result<(Vec<u32>, Vec<f32>), &'static str> {
        if self.clips == 0 || self.clips > 512 || self.grid == 0 || self.grid > GRID {
            return Err("invalid view dimensions");
        }
        let limit = FRAMES * self.grid * self.grid;
        if self.kept == 0 || self.kept > limit || patch_ids.len() != self.kept * self.clips {
            return Err("invalid retained patch count");
        }
        let mut seen = vec![false; limit];
        for clip in 0..self.clips {
            seen.fill(false);
            for position in 0..self.kept {
                let id = patch_ids[position * self.clips + clip] as usize;
                if id >= limit || seen[id] {
                    return Err("out-of-range or duplicate patch ID");
                }
                seen[id] = true;
            }
        }
        let mut positions = vec![limit as u32; self.clips]; // identity RoPE for CLS
        positions.extend_from_slice(patch_ids);
        let tokens = self.kept + 1;
        let heads = ARCHITECTURE.heads();
        if tokens * self.clips * heads * tokens > 65_535 * 256 {
            return Err("attention mask exceeds the portable pointwise dispatch size");
        }
        let mut mask = Vec::with_capacity(tokens * self.clips * heads * tokens);
        for query in 0..tokens {
            for clip in 0..self.clips {
                let query_frame =
                    positions[query * self.clips + clip] as usize / (self.grid * self.grid);
                for _head in 0..heads {
                    for key in 0..tokens {
                        let key_frame =
                            positions[key * self.clips + clip] as usize / (self.grid * self.grid);
                        let allowed = query == 0 || (key != 0 && key_frame <= query_frame);
                        mask.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
                    }
                }
            }
        }
        Ok((positions, mask))
    }
}

type Weights = BTreeMap<String, NodeId>;

fn encoder_weights(g: &mut Graph) -> Weights {
    super::weight_shapes(ARCHITECTURE)
        .into_iter()
        .map(|(name, shape)| {
            let shape = if name == "encoder.cls_token" {
                vec![1, ARCHITECTURE.hidden()]
            } else if shape.len() > 1 {
                vec![shape[1..].iter().product(), shape[0]]
            } else {
                shape
            };
            let id = g.parameter(&name, &shape);
            (name, id)
        })
        .collect()
}

fn dense(g: &mut Graph, x: NodeId, weights: &Weights, name: &str) -> NodeId {
    let output = g.matmul(x, weights[&format!("{name}.weight")]);
    g.bias_add(output, weights[&format!("{name}.bias")])
}

fn norm(g: &mut Graph, x: NodeId, weights: &Weights, name: &str) -> NodeId {
    g.layer_norm(
        x,
        weights[&format!("{name}.weight")],
        weights[&format!("{name}.bias")],
        1e-6,
    )
}

fn concat_rows(g: &mut Graph, left: NodeId, right: NodeId) -> NodeId {
    let [left_rows, width]: [usize; 2] = g.node(left).ty.shape.clone().try_into().unwrap();
    let [right_rows, right_width]: [usize; 2] = g.node(right).ty.shape.clone().try_into().unwrap();
    assert_eq!(width, right_width);
    let left = g.reshape(left, &[left_rows * width]);
    let right = g.reshape(right, &[right_rows * width]);
    let joined = g.concat(
        left,
        right,
        1,
        left_rows as u32,
        right_rows as u32,
        width as u32,
    );
    g.reshape(joined, &[left_rows + right_rows, width])
}

fn first_rows(g: &mut Graph, input: NodeId, rows: usize) -> NodeId {
    let [total, width]: [usize; 2] = g.node(input).ty.shape.clone().try_into().unwrap();
    let input = g.reshape(input, &[total * width]);
    let first = g.split_a(input, 1, rows as u32, (total - rows) as u32, width as u32);
    g.reshape(first, &[rows, width])
}

fn attention(g: &mut Graph, q: NodeId, k: NodeId, v: NodeId, mask: NodeId, view: View) -> NodeId {
    let tokens = view.kept + 1;
    let groups = view.clips * ARCHITECTURE.heads();
    let q = g.reshape(q, &[tokens, groups * HEAD_DIM]);
    let k = g.reshape(k, &[tokens, groups * HEAD_DIM]);
    let k = g.transpose(k);
    let k = g.reshape(k, &[groups, HEAD_DIM, tokens]);
    let scores = g.block_matmul(q, k);
    let scores = g.scale(scores, 1.0 / (HEAD_DIM as f32).sqrt());
    let scores = g.add(scores, mask);
    let scores = g.reshape(scores, &[tokens * groups, tokens]);
    let probabilities = g.softmax(scores);
    let probabilities = g.reshape(probabilities, &[tokens, groups * tokens]);
    let v = g.reshape(v, &[tokens, groups * HEAD_DIM]);
    let v = g.transpose(v);
    let v = g.reshape(v, &[groups, HEAD_DIM, tokens]);
    let output = g.block_matmul_bt(probabilities, v);
    g.reshape(output, &[tokens * view.clips, ARCHITECTURE.hidden()])
}

fn encode(g: &mut Graph, weights: &Weights, prefix: &str, view: View) -> NodeId {
    let hidden = ARCHITECTURE.hidden();
    let tokens = view.kept + 1;
    let rows = tokens * view.clips;
    let patches = g.input(
        &format!("{prefix}.patches"),
        &[view.kept * view.clips, PATCH_DIM],
    );
    let patches = dense(g, patches, weights, "encoder.patch_embed.proj");
    let cls_copies = g.constant(vec![1.0; view.clips], &[view.clips, 1]);
    let cls = g.matmul(cls_copies, weights["encoder.cls_token"]);
    let mut x = concat_rows(g, cls, patches);
    let ids = g.input_u32(&format!("{prefix}.positions"), &[rows]);
    let (mut cos, mut sin) = rope_tables_for_grid(ARCHITECTURE, view.grid);
    cos.extend(vec![1.0; hidden]);
    sin.extend(vec![0.0; hidden]);
    let table_rows = FRAMES * view.grid * view.grid + 1;
    let cos = g.constant(cos, &[table_rows, hidden]);
    let sin = g.constant(sin, &[table_rows, hidden]);
    let cos = g.embedding(ids, cos);
    let sin = g.embedding(ids, sin);
    let mask = g.input(
        &format!("{prefix}.mask"),
        &[tokens, view.clips * ARCHITECTURE.heads() * tokens],
    );
    for layer in 0..ARCHITECTURE.layers() {
        let name = format!("encoder.blocks.{layer}");
        let normalized = norm(g, x, weights, &format!("{name}.norm1"));
        let qkv = dense(g, normalized, weights, &format!("{name}.attn.qkv"));
        let qkv = g.reshape(qkv, &[rows * 3 * hidden]);
        let q = g.split_a(qkv, rows as u32, hidden as u32, (2 * hidden) as u32, 1);
        let kv = g.split_b(qkv, rows as u32, hidden as u32, (2 * hidden) as u32, 1);
        let k = g.split_a(kv, rows as u32, hidden as u32, hidden as u32, 1);
        let v = g.split_b(kv, rows as u32, hidden as u32, hidden as u32, 1);
        let q = g.reshape(q, &[rows, hidden]);
        let k = g.reshape(k, &[rows, hidden]);
        let q = rope(g, q, cos, sin);
        let k = rope(g, k, cos, sin);
        let output = attention(g, q, k, v, mask, view);
        let output = dense(g, output, weights, &format!("{name}.attn.proj"));
        x = g.add(x, output);
        let normalized = norm(g, x, weights, &format!("{name}.norm2"));
        let hidden = dense(g, normalized, weights, &format!("{name}.mlp.fc1"));
        let hidden = g.gelu_erf(hidden);
        let output = dense(g, hidden, weights, &format!("{name}.mlp.fc2"));
        x = g.add(x, output);
    }
    let output = norm(g, x, weights, "encoder.norm");
    first_rows(g, output, view.clips)
}

fn projector(g: &mut Graph, cls: NodeId, config: Config) -> NodeId {
    let rows = config.batch * (1 + config.local_views);
    let hidden = config.projector_hidden;
    let x = super::linear(g, cls, "projector.fc1", ARCHITECTURE.hidden(), hidden);
    // Training BatchNorm over all views/clips, with population variance.
    // The projector is discarded at export; no inference running statistics.
    let x = g.transpose(x);
    let ones = g.constant(vec![1.0; rows], &[rows]);
    let zeros = g.constant(vec![0.0; rows], &[rows]);
    let x = g.layer_norm(x, ones, zeros, 1e-5);
    let x = g.transpose(x);
    let weight = g.parameter("projector.norm.weight", &[hidden]);
    let bias = g.parameter("projector.norm.bias", &[hidden]);
    let x = g.bias_mul(x, weight);
    let x = g.bias_add(x, bias);
    let x = g.gelu_erf(x);
    super::linear(g, x, "projector.fc2", hidden, config.projector_output)
}

fn mean_batch(g: &mut Graph, x: NodeId, views: usize, batch: usize, width: usize) -> NodeId {
    let x = g.reshape(x, &[views * batch, width]);
    let x = g.transpose(x);
    let x = g.reshape(x, &[width * views, batch]);
    let sum = g.sum_inner(x);
    let mean = g.scale(sum, 1.0 / batch as f32);
    let mean = g.reshape(mean, &[width, views]);
    g.transpose(mean)
}

fn sigreg(g: &mut Graph, z: NodeId, config: Config) -> NodeId {
    let views = config.local_views + 1;
    // Fresh random unit-length columns each step, shared across all views.
    let projections = g.input(
        "sigreg.directions",
        &[config.projector_output, config.directions],
    );
    let projected = g.matmul(z, projections);
    let projected = g.reshape(projected, &[views * config.batch * config.directions, 1]);
    let dt = 3.0 / (KNOTS - 1) as f32;
    let knots = (0..KNOTS).map(|k| k as f32 * dt).collect::<Vec<_>>();
    let t = g.constant(knots.clone(), &[1, KNOTS]);
    let angles = g.matmul(projected, t);
    let cosine = g.cos(angles);
    let sine = g.sin(angles);
    let width = config.directions * KNOTS;
    let cosine = mean_batch(g, cosine, views, config.batch, width);
    let sine = mean_batch(g, sine, views, config.batch, width);
    let phi = knots
        .iter()
        .map(|t| (-t * t / 2.0).exp())
        .collect::<Vec<_>>();
    let negative_phi = g.constant((0..width).map(|i| -phi[i % KNOTS]).collect(), &[width]);
    let real_error = g.bias_add(cosine, negative_phi);
    let real_square = g.mul(real_error, real_error);
    let imaginary_square = g.mul(sine, sine);
    let error = g.add(real_square, imaginary_square);
    let weights = (0..width)
        .map(|i| {
            let knot = i % KNOTS;
            let trapezoid = if knot == 0 || knot == KNOTS - 1 {
                dt
            } else {
                2.0 * dt
            };
            trapezoid * phi[knot]
        })
        .collect();
    let weights = g.constant(weights, &[width]);
    let weighted = g.bias_mul(error, weights);
    let mean = g.mean_all(weighted);
    // Sum over knots, average over views/projections, multiply by clip batch.
    g.scale(mean, (config.batch * KNOTS) as f32)
}

/// Build all learned operations without creating a GPU context. Inputs use
/// view-major clips: global B, then each local view's B clips. Return outputs
/// are total loss, invariance, SIGReg and projected embeddings respectively.
pub fn graph(config: Config) -> Result<Graph, &'static str> {
    config.validate()?;
    let mut g = Graph::new();
    let weights = encoder_weights(&mut g);
    let global = encode(&mut g, &weights, "global", config.global());
    let local = encode(&mut g, &weights, "local", config.local());
    let cls = concat_rows(&mut g, global, local);
    let z = projector(&mut g, cls, config);
    let global = first_rows(&mut g, z, config.batch);
    let mut targets = global;
    for _ in 0..config.local_views {
        targets = concat_rows(&mut g, targets, global);
    }
    let negative = g.neg(targets);
    let difference = g.add(z, negative);
    let squared = g.mul(difference, difference);
    let invariance = g.mean_all(squared);
    let regularization = sigreg(&mut g, z, config);
    let weighted = g.scale(regularization, 0.02);
    let loss = g.add(invariance, weighted);
    g.set_outputs(vec![loss, invariance, regularization, z]);
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use meganeura::graph::Op;

    pub(super) struct Reference {
        pub(super) graph: Graph,
        pub(super) tensors: meganeura::data::safetensors::SafeTensorsModel,
        positions: Vec<(String, Vec<u32>)>,
    }

    impl Reference {
        pub(super) fn load() -> Self {
            use sha2::{Digest, Sha256};
            let root = std::path::PathBuf::from(
                std::env::var_os("KINDLE_TINY_REFERENCE").expect("reference directory required"),
            );
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap())
                    .unwrap();
            let bytes = std::fs::read(root.join("reference.safetensors")).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                manifest["tensors_sha256"].as_str().unwrap()
            );
            assert_eq!(
                format!(
                    "{:x}",
                    Sha256::digest(include_bytes!(
                        "../../../../python/examples/levjepa_tiny_reference.py"
                    ))
                ),
                manifest["generator_sha256"].as_str().unwrap()
            );
            let config: Config = serde_json::from_value(manifest["config"].clone()).unwrap();
            assert_eq!(
                (config.batch, config.local_views, config.directions),
                (3, 2, 7)
            );
            let graph = graph(config).unwrap();
            let tensors =
                meganeura::data::safetensors::SafeTensorsModel::from_bytes(bytes).unwrap();
            let mut positions = Vec::new();
            for (name, view) in [("global", config.global()), ("local", config.local())] {
                let ids: Vec<u32> =
                    serde_json::from_value(manifest["patch_ids"][name].clone()).unwrap();
                let (ids, mask) = view.attention_inputs(&ids).unwrap();
                assert_eq!(mask, tensors.tensor_f32(&format!("{name}.mask")).unwrap());
                positions.push((format!("{name}.positions"), ids));
            }
            let mut parameters = 0;
            for node in graph.nodes() {
                let names = match &node.op {
                    Op::Parameter { name } => {
                        parameters += 1;
                        vec![format!("weight.{name}"), format!("gradient.{name}")]
                    }
                    Op::Input { name } if !name.ends_with(".positions") => vec![name.clone()],
                    _ => continue,
                };
                for name in names {
                    assert_eq!(tensors.tensor_info()[&name].shape, node.ty.shape, "{name}");
                    let data = tensors.tensor_f32(&name).unwrap();
                    if !name.ends_with(".mask") {
                        assert!(data.iter().all(|v| v.is_finite()), "{name}");
                    }
                }
            }
            assert_eq!(parameters, 155);
            for (index, name) in ["loss", "invariance", "sigreg", "embeddings"]
                .into_iter()
                .enumerate()
            {
                let expected = tensors.tensor_f32(&format!("expected.{name}")).unwrap();
                assert_eq!(
                    expected.len(),
                    graph.node(graph.outputs()[index]).ty.num_elements()
                );
                assert!(expected.iter().all(|v| v.is_finite()));
            }
            Self {
                graph,
                tensors,
                positions,
            }
        }
    }

    fn compare_reference(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        let (mut error_sq, mut reference_sq, mut max_error, mut max_reference) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite(), "{name}: non-finite value");
            let error = (f64::from(a) - f64::from(b)).abs();
            error_sq += error * error;
            reference_sq += f64::from(b).powi(2);
            max_error = max_error.max(error);
            max_reference = max_reference.max(f64::from(b).abs());
        }
        // Fixed before GPU execution, also written in the CPU fixture manifest.
        assert!(
            error_sq.sqrt() <= 0.003 * reference_sq.sqrt() + 1e-5 * (actual.len() as f64).sqrt(),
            "{name}: L2 error {} / {}",
            error_sq.sqrt(),
            reference_sq.sqrt()
        );
        assert!(
            max_error <= 0.01 * max_reference + 3e-5,
            "{name}: max error {max_error} / {max_reference}"
        );
        eprintln!(
            "tiny_reference {name}: l2_error={} reference_l2={} max_error={max_error}",
            error_sq.sqrt(),
            reference_sq.sqrt()
        );
    }

    #[test]
    #[ignore = "requires KINDLE_TINY_REFERENCE fixture; CPU only"]
    fn cpu_reference_fixture_is_complete() {
        Reference::load();
    }

    #[test]
    #[ignore = "requires separately declared exclusive GPU and CPU reference fixture"]
    fn native_losses_and_all_gradients_match_independent_reference() {
        let reference = Reference::load(); // all fixture checks precede GPU access
        assert_eq!(std::env::var("MEGANEURA_DEVICE_ID").unwrap(), "0x2c02");
        assert_eq!(std::env::var("KINDLE_GPU_DRIVER").unwrap(), "580.178.04");
        let (mut session, _) = meganeura::build(
            &reference.graph,
            meganeura::SessionConfig {
                runtime: meganeura::runtime::SessionOptions {
                    coop: meganeura::runtime::CoopPolicy::Disabled,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let device = session.device_information();
        assert_eq!(device.device_name, "NVIDIA GeForce RTX 5080");
        assert_eq!(device.driver_name, "NVIDIA");
        assert_eq!(device.driver_info, "580.178.04");
        assert!(!device.is_software_emulated);
        let check_memory = |session: &meganeura::Session| {
            let memory = session
                .device_memory_stats()
                .expect("Vulkan memory budget required");
            assert!(
                memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 * 1024 * 1024 * 1024
            );
            eprintln!(
                "tiny_reference Vulkan estimated usage={} budget={}",
                memory.usage_bytes, memory.budget_bytes
            );
        };
        check_memory(&session);
        for node in reference.graph.nodes() {
            match &node.op {
                Op::Parameter { name } => session.set_parameter(
                    name,
                    &reference
                        .tensors
                        .tensor_f32(&format!("weight.{name}"))
                        .unwrap(),
                ),
                Op::Input { name } if !name.ends_with(".positions") => {
                    session.set_input(name, &reference.tensors.tensor_f32(name).unwrap())
                }
                _ => (),
            }
        }
        for (name, ids) in &reference.positions {
            session.set_input_u32(name, ids);
        }
        // Compute backward without updating weights or creating Adam moments.
        session.set_learning_rate(0.0);
        session.step();
        session.wait();
        check_memory(&session);
        for (index, name) in ["loss", "invariance", "sigreg", "embeddings"]
            .into_iter()
            .enumerate()
        {
            let expected = reference
                .tensors
                .tensor_f32(&format!("expected.{name}"))
                .unwrap();
            let mut actual = vec![0.0; expected.len()];
            session.read_output_by_index(index, &mut actual);
            compare_reference(name, &actual, &expected);
        }
        let mut gradients = 0;
        for node in reference.graph.nodes() {
            if let Op::Parameter { name } = &node.op {
                let expected = reference
                    .tensors
                    .tensor_f32(&format!("gradient.{name}"))
                    .unwrap();
                let mut actual = vec![0.0; expected.len()];
                session.read_param_grad(name, &mut actual);
                compare_reference(name, &actual, &expected);
                session.read_param(name, &mut actual);
                assert_eq!(
                    actual,
                    reference
                        .tensors
                        .tensor_f32(&format!("weight.{name}"))
                        .unwrap()
                );
                gradients += 1;
            }
        }
        assert_eq!(gradients, 155);
        check_memory(&session);
    }

    #[test]
    fn default_views_keep_original_video_grid_and_declared_token_drop() {
        let config = Config::default();
        config.validate().unwrap();
        assert_eq!(config.global().kept, 157);
        assert_eq!(config.local().kept, 29);
        assert_eq!(config.local().clips, 512);
    }

    #[test]
    fn full_default_batch_differentiates_without_dense_host_constants() {
        let graph = graph(Config::default()).unwrap();
        let differentiated = meganeura::autodiff::differentiate(&graph);
        let constants: usize = differentiated
            .nodes()
            .iter()
            .map(|node| match &node.op {
                Op::Constant { data } => data.len(),
                _ => 0,
            })
            .sum();
        assert!(
            constants < 2_000_000,
            "gradient constants grew to {constants} elements"
        );
        let plan = meganeura::compile::compile(&differentiated);
        assert_eq!(plan.param_grad_pairs.len(), 155);
    }

    #[test]
    fn causal_mask_uses_original_positions_and_never_reads_cls() {
        let view = View {
            clips: 2,
            grid: 2,
            kept: 3,
        };
        let ids = [8, 4, 0, 0, 4, 60];
        let (positions, mask) = view.attention_inputs(&ids).unwrap();
        assert_eq!(positions, [64, 64, 8, 4, 0, 0, 4, 60]);
        let at = |q: usize, clip: usize, head: usize, k: usize| {
            mask[((q * 2 + clip) * 3 + head) * 4 + k]
        };
        for clip in 0..2 {
            for head in 0..3 {
                for k in 0..4 {
                    assert_eq!(at(0, clip, head, k), 0.0);
                }
                for q in 1..4 {
                    assert_eq!(at(q, clip, head, 0), f32::NEG_INFINITY);
                    assert_eq!(at(q, clip, head, q), 0.0);
                }
            }
        }
        assert_eq!(at(1, 0, 0, 3), 0.0); // frame 2 reads frame 1
        assert_eq!(at(1, 1, 0, 3), f32::NEG_INFINITY); // frame 1 cannot read 15
        assert_eq!(at(2, 0, 0, 1), f32::NEG_INFINITY); // frame 0 cannot read 2
        assert!(view.attention_inputs(&[64, 4, 0, 0, 4, 60]).is_err());
        assert!(view.attention_inputs(&[0, 4, 0, 0, 4, 60]).is_err());
        assert!(view.attention_inputs(&ids[..5]).is_err());
    }

    #[test]
    fn all_views_share_one_tiny_encoder_and_receive_gradients() {
        let config = Config {
            batch: 2,
            local_views: 1,
            directions: 4,
            ..Config::default()
        };
        let graph = graph(config).unwrap();
        let parameters = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                Op::Parameter { name } => Some((name.clone(), node.ty.num_elements())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let unique: BTreeMap<_, _> = parameters.iter().cloned().collect();
        assert_eq!(unique.len(), parameters.len());
        assert_eq!(
            parameters
                .iter()
                .filter(|(name, _)| name.starts_with("encoder."))
                .map(|(_, size)| size)
                .sum::<usize>(),
            5_486_592
        );
        assert_eq!(
            parameters.iter().map(|(_, size)| size).sum::<usize>(),
            5_817_472
        );
        let diff = meganeura::autodiff::differentiate(&graph);
        assert_eq!(
            diff.outputs().len(),
            graph.outputs().len() + parameters.len()
        );
        let plan = meganeura::compile::compile(&diff);
        assert_eq!(plan.param_grad_pairs.len(), parameters.len());
        assert!(
            !graph
                .nodes()
                .iter()
                .any(|node| matches!(node.op, Op::CacheWrite | Op::StopGradient))
        );
    }
}
