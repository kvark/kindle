//! Dense causal clips for joint world-model training. Same weights, RoPE,
//! frame-causal attention and spatial output as streaming perception; no KV
//! cache enters autodiff. Replay supplies complete, phase-zero encoder chunks.

use super::*;
use crate::vision::OBSERVATION_GRID;

pub(crate) const PIXELS: usize = PATCHES * PATCH_DIM;
pub(crate) const REGULARIZATION_WEIGHT: f32 = 0.02;

fn rows(g: &mut Graph, x: NodeId, start: usize, count: usize, width: usize) -> NodeId {
    let total = g.node(x).ty.num_elements() / width;
    assert!(count > 0 && start + count <= total);
    let mut x = g.reshape(x, &[total * width]);
    if start > 0 {
        x = g.split_b(x, 1, start as u32, (total - start) as u32, width as u32);
    }
    if start + count < total {
        x = g.split_a(
            x,
            1,
            count as u32,
            (total - start - count) as u32,
            width as u32,
        );
    }
    g.reshape(x, &[count, width])
}

fn join(g: &mut Graph, xs: &[NodeId], width: usize) -> NodeId {
    assert!(!xs.is_empty());
    let mut count = g.node(xs[0]).ty.num_elements() / width;
    let mut x = g.reshape(xs[0], &[count * width]);
    for &next in &xs[1..] {
        let next_count = g.node(next).ty.num_elements() / width;
        let next = g.reshape(next, &[next_count * width]);
        x = g.concat(x, next, 1, count as u32, next_count as u32, width as u32);
        count += next_count;
    }
    g.reshape(x, &[count, width])
}

/// Time-major inputs and outputs, clip-major internal tokens. Each frame's
/// queries see all patches of that frame and only earlier frames in its clip.
pub(crate) fn encode(g: &mut Graph, inputs: &[NodeId], batch: usize) -> Vec<NodeId> {
    let architecture = Architecture::Tiny;
    let hidden = architecture.hidden();
    let length = inputs.len();
    assert!((1..=FRAMES).contains(&length) && batch > 0);
    let mut clips = Vec::with_capacity(batch * length);
    for clip in 0..batch {
        for &input in inputs {
            clips.push(rows(g, input, clip * PATCHES, PATCHES, PATCH_DIM));
        }
    }
    let input = join(g, &clips, PATCH_DIM);
    let count = batch * length * PATCHES;
    let (cos, sin) = rope_tables(architecture);
    let table_len = length * PATCHES * hidden;
    let cos = g.constant(cos[..table_len].repeat(batch), &[count, hidden]);
    let sin = g.constant(sin[..table_len].repeat(batch), &[count, hidden]);
    let mut x = linear(g, input, "encoder.patch_embed.proj", PATCH_DIM, hidden);
    for layer in 0..architecture.layers() {
        let name = format!("encoder.blocks.{layer}");
        let n = norm(g, x, &format!("{name}.norm1"), hidden);
        let qkv = linear(g, n, &format!("{name}.attn.qkv"), hidden, 3 * hidden);
        let qkv = g.reshape(qkv, &[count * 3 * hidden]);
        let q = g.split_a(qkv, count as u32, hidden as u32, (2 * hidden) as u32, 1);
        let kv = g.split_b(qkv, count as u32, hidden as u32, (2 * hidden) as u32, 1);
        let k = g.split_a(kv, count as u32, hidden as u32, hidden as u32, 1);
        let v = g.split_b(kv, count as u32, hidden as u32, hidden as u32, 1);
        let q = g.reshape(q, &[count, hidden]);
        let k = g.reshape(k, &[count, hidden]);
        let q = rope(g, q, cos, sin);
        let k = rope(g, k, cos, sin);
        let mut attention = Vec::with_capacity(batch * length);
        for clip in 0..batch {
            for time in 0..length {
                let start = clip * length * PATCHES;
                let q = rows(g, q, start + time * PATCHES, PATCHES, hidden);
                let k = rows(g, k, start, (time + 1) * PATCHES, hidden);
                let v = rows(g, v, start, (time + 1) * PATCHES, hidden);
                // Cross mode is unmasked. Prefix slicing supplies frame-level
                // causality, unlike token-triangular decoder attention.
                attention.push(g.multi_head_attn(q, k, v, 3, 3, HEAD_DIM as u32, true));
            }
        }
        let attention = join(g, &attention, hidden);
        let attention = linear(g, attention, &format!("{name}.attn.proj"), hidden, hidden);
        x = g.add(x, attention);
        let n = norm(g, x, &format!("{name}.norm2"), hidden);
        let mlp = linear(g, n, &format!("{name}.mlp.fc1"), hidden, 4 * hidden);
        let mlp = gelu_erf(g, mlp);
        let mlp = linear(g, mlp, &format!("{name}.mlp.fc2"), 4 * hidden, hidden);
        x = g.add(x, mlp);
    }
    let tokens = norm(g, x, "encoder.norm", hidden);
    let projection = g.constant(
        fixed_projection(hidden, OBSERVATION_CHANNELS, PROJECTION_SEED),
        &[hidden, OBSERVATION_CHANNELS],
    );
    let projected = g.matmul(tokens, projection);
    let pooled = pool_patches(g, projected, batch * length);
    (0..length)
        .map(|time| {
            let clips = (0..batch)
                .map(|clip| rows(g, pooled, clip * length + time, 1, Observation::LEN))
                .collect::<Vec<_>>();
            let frame = join(g, &clips, Observation::LEN);
            g.reshape(
                frame,
                &[
                    batch * OBSERVATION_GRID * OBSERVATION_GRID,
                    OBSERVATION_CHANNELS,
                ],
            )
        })
        .collect()
}

/// Variance/covariance regularization on spatially averaged frame features.
/// This online VICReg-style stabilizer is not LeVJEPA's pretraining objective.
pub(crate) fn regularization(
    g: &mut Graph,
    observations: &[NodeId],
    batch: usize,
) -> (NodeId, NodeId) {
    let width = OBSERVATION_CHANNELS;
    let samples = observations.len() * batch;
    assert!(samples > 1);
    let mut means = Vec::with_capacity(samples);
    for &observation in observations {
        for row in 0..batch {
            let frame = rows(g, observation, row, 1, Observation::LEN);
            let frame = g.reshape(frame, &[OBSERVATION_GRID * OBSERVATION_GRID, width]);
            let frame = g.transpose(frame);
            let mean = g.sum_inner(frame);
            let mean = g.reshape(mean, &[width]);
            means.push(g.scale(mean, 1.0 / (OBSERVATION_GRID * OBSERVATION_GRID) as f32));
        }
    }
    let z = join(g, &means, width);
    let transposed = g.transpose(z);
    let sum = g.sum_inner(transposed);
    let negative_mean = g.scale(sum, -1.0 / samples as f32);
    let negative_mean = g.reshape(negative_mean, &[width]);
    let centered = g.bias_add(z, negative_mean);
    let squared = g.mul(centered, centered);
    let transposed = g.transpose(squared);
    let sum = g.sum_inner(transposed);
    let variance = g.scale(sum, 1.0 / (samples - 1) as f32);
    let stabilized = shift(g, variance, 1e-4);
    let log_variance = g.log(stabilized);
    let log_std = g.scale(log_variance, 0.5);
    let std = g.exp(log_std);
    let spread = g.mean_all(std);
    let negative_std = g.neg(std);
    let shortfall = shift(g, negative_std, 1.0);
    let shortfall = g.relu(shortfall);
    let variance_loss = g.mean_all(shortfall);
    let covariance = g.matmul_at(centered, centered);
    let covariance = g.scale(covariance, 1.0 / (samples - 1) as f32);
    let squared = g.mul(covariance, covariance);
    let off_diagonal = g.constant(
        (0..width * width)
            .map(|i| if i / width == i % width { 0.0 } else { 1.0 })
            .collect(),
        &[width, width],
    );
    let squared = g.mul(squared, off_diagonal);
    let covariance_loss = g.mean_all(squared);
    let covariance_loss = g.scale(covariance_loss, 0.04 * width as f32);
    (g.add(variance_loss, covariance_loss), spread)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_device(gpu: &blade_graphics::Context) {
        let info = gpu.device_information();
        assert_eq!(
            info.device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!info.is_software_emulated);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }

    #[test]
    fn dense_graph_has_complete_tiny_weights_and_no_stateful_cache() {
        let mut g = Graph::new();
        let inputs = (0..2)
            .map(|t| g.input(&format!("pixels_{t}"), &[2, PIXELS]))
            .collect::<Vec<_>>();
        let outputs = encode(&mut g, &inputs, 2);
        for &x in &outputs {
            assert_eq!(g.node(x).ty.shape, [98, 64]);
        }
        let parameters = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                meganeura::graph::Op::Parameter { name } => Some((name, n.ty.num_elements())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(parameters.len(), 148);
        for (name, count) in parameters {
            assert!(name.starts_with("encoder.") && !name.contains("cls"));
            let (_, shape) = weight_shapes(Architecture::Tiny)
                .into_iter()
                .find(|(expected, _)| expected == name)
                .unwrap();
            assert_eq!(count, shape.iter().product::<usize>());
        }
    }

    #[test]
    #[ignore = "requires separately declared GPU and KINDLE_JOINT_TINY_REFERENCE oracle directory"]
    fn regularizer_matches_independent_f64_value_and_gradients() {
        let path = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let reference = SafeTensorsModel::load(path.join("reference.safetensors")).unwrap();
        let mut g = Graph::new();
        let features = (0..2)
            .map(|t| g.parameter(&format!("features_{t}"), &[98, 64]))
            .collect::<Vec<_>>();
        let (loss, spread) = regularization(&mut g, &features, 2);
        let loss = g.scale(loss, 5.0);
        g.set_outputs(vec![loss, spread]);
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        check_device(&gpu);
        let (mut session, _) = meganeura::build(
            &g,
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(Arc::clone(&gpu)),
                ..Default::default()
            },
        );
        check_device(&gpu);
        for time in 0..2 {
            let name = format!("features_{time}");
            session.set_parameter(&name, &reference.tensor_f32_auto(&name).unwrap());
        }
        session.clear_optimizer();
        session.step();
        session.wait();
        check_device(&gpu);
        for (index, name) in ["loss", "spread"].iter().enumerate() {
            let mut actual = [0.0];
            session.read_output_by_index(index, &mut actual);
            let expected = reference
                .tensor_f32_auto(&format!("regularization.{name}"))
                .unwrap();
            assert!(
                (actual[0] - expected[0]).abs() < 1e-5,
                "{name}: {actual:?} {expected:?}"
            );
        }
        for time in 0..2 {
            let mut actual = vec![0.0; 2 * Observation::LEN];
            session.read_param_grad(&format!("features_{time}"), &mut actual);
            let expected = reference
                .tensor_f32_auto(&format!("regularization.gradient_{time}"))
                .unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm = expected.iter().map(|b| b * b).sum::<f32>().sqrt();
            assert!(error < 1e-4 * norm + 1e-7, "error={error}, norm={norm}");
        }
    }

    #[test]
    #[ignore = "requires separately declared GPU and KINDLE_JOINT_TINY_REFERENCE oracle directory"]
    fn dense_tiny_matches_independent_f64_outputs_gradients_and_updates() {
        let path = std::path::PathBuf::from(std::env::var("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
        let checkpoint = std::path::Path::new(manifest["checkpoint"].as_str().unwrap());
        assert_eq!(
            crate::vision::checkpoint_sha256(checkpoint).unwrap(),
            manifest["checkpoint_sha256"].as_str().unwrap()
        );
        let reference = SafeTensorsModel::load(path.join("reference.safetensors")).unwrap();
        let weights = SafeTensorsModel::load(checkpoint.to_path_buf()).unwrap();
        let mut g = Graph::new();
        let inputs = (0..2)
            .map(|t| g.input(&format!("pixels_{t}"), &[2, PIXELS]))
            .collect::<Vec<_>>();
        let features = encode(&mut g, &inputs, 2);
        let mut loss = g.scalar(0.0);
        for (t, &features) in features.iter().enumerate() {
            let coefficients = g.constant(
                reference
                    .tensor_f32_auto(&format!("coefficients_{t}"))
                    .unwrap(),
                &[98, 64],
            );
            let squared = g.mul(features, features);
            let weighted = g.mul(squared, coefficients);
            let mean = g.mean_all(weighted);
            let term = g.scale(mean, 17.0 / 2.0);
            loss = g.add(loss, term);
        }
        g.set_outputs(std::iter::once(loss).chain(features).collect());
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        check_device(&gpu);
        let (mut session, _) = meganeura::build(
            &g,
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(Arc::clone(&gpu)),
                runtime: meganeura::SessionOptions {
                    coop: meganeura::CoopPolicy::Disabled,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        check_device(&gpu);
        load_weights(&mut session, &weights, 0, Architecture::Tiny).unwrap();
        for time in 0..2 {
            let name = format!("pixels_{time}");
            session.set_input(&name, &reference.tensor_f32_auto(&name).unwrap());
        }
        session.clear_optimizer();
        session.step();
        session.wait();
        check_device(&gpu);
        let check = |name: &str, actual: &[f32], expected: &[f32], tolerance: f64| {
            assert_eq!(actual.len(), expected.len(), "{name}");
            assert!(actual.iter().all(|x| x.is_finite()), "{name}");
            let error = actual
                .iter()
                .zip(expected)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = expected
                .iter()
                .map(|&b| f64::from(b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                error <= tolerance * norm + 2e-5,
                "{name}: error={error}, norm={norm}"
            );
        };
        check(
            "loss",
            &session.read_output(1),
            &reference.tensor_f32_auto("loss").unwrap(),
            3e-4,
        );
        for time in 0..2 {
            let mut actual = vec![0.0; 2 * Observation::LEN];
            session.read_output_by_index(time + 1, &mut actual);
            check(
                "features",
                &actual,
                &reference
                    .tensor_f32_auto(&format!("features_{time}"))
                    .unwrap(),
                3e-4,
            );
        }
        let names = session
            .param_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 148);
        for name in names {
            assert!(session.has_param_grad(&name));
            let mut actual = vec![0.0; session.param_size(&name).unwrap()];
            session.read_param_grad(&name, &mut actual);
            check(
                &name,
                &actual,
                &reference
                    .tensor_f32_auto(&format!("gradient.{name}"))
                    .unwrap(),
                5e-3,
            );
        }
        let name = "encoder.patch_embed.proj.weight";
        let mut before = vec![0.0; session.param_size(name).unwrap()];
        session.read_param(name, &mut before);
        session.set_adam(1e-5, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();
        check_device(&gpu);
        let mut after = before.clone();
        session.read_param(name, &mut after);
        assert!(after.iter().all(|x| x.is_finite()));
        assert_ne!(
            before, after,
            "nonzero gradients must move the input projection"
        );
    }
}
