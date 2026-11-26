//! Frozen action-space derivatives. This test-only probe never creates a learner.

use super::*;
use crate::dreamer::{readback::Readback, runtime::build_session, world};
use meganeura::{Mode, Session, data::safetensors::SafeTensorsModel};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, sync::Arc, time::Instant};

type Weights = BTreeMap<String, Vec<f32>>;

fn objectives(
    graph: &mut Graph,
    rewards: &[NodeId],
    continuations: &[NodeId],
    value: NodeId,
) -> [NodeId; 2] {
    assert!(!rewards.is_empty() && rewards.len() == continuations.len());
    let rows = graph.node(value).ty.shape[0];
    let mut live = graph.constant(vec![1.0; rows], &[rows, 1]);
    let mut total = graph.constant(vec![0.0; rows], &[rows, 1]);
    for (&reward, &continuation) in rewards.iter().zip(continuations) {
        let reward = graph.mul(live, reward);
        total = graph.add(total, reward);
        live = graph.mul(live, continuation);
    }
    let endpoint = graph.mul(live, value);
    [live, graph.add(total, endpoint)]
}

fn outputs(graph: &mut Graph, scores: [NodeId; 2]) {
    let weighted = scores
        .iter()
        .enumerate()
        .map(|(index, &score)| {
            let gate = graph.input(&format!("gate_{index}"), &[1]);
            let mean = graph.mean_all(score);
            graph.mul(mean, gate)
        })
        .collect::<Vec<_>>();
    let loss = graph.add(weighted[0], weighted[1]);
    graph.set_outputs(vec![loss, scores[0], scores[1]]);
}

fn graph(config: &DreamerConfig, rows: usize, horizon: usize) -> Graph {
    let mut graph = Graph::new();
    let size = config.network();
    let core = RssmCore::new_frozen(&mut graph, config);
    let prior = Prior::new_frozen(&mut graph, config);
    let reward = MlpHead::new(
        &mut graph,
        "world.reward",
        config.feature_dim(),
        size.units,
        1,
        config.value_bins,
    );
    let continuation = MlpHead::new(
        &mut graph,
        "world.continuation",
        config.feature_dim(),
        size.units,
        1,
        1,
    );
    let critic = MlpHead::new(
        &mut graph,
        "behavior.value",
        config.feature_dim(),
        size.units,
        3,
        config.value_bins,
    );
    let mut deter = graph.input("initial_deter", &[rows, size.deter]);
    let mut stoch = graph.input("initial_stoch", &[rows * size.stoch, size.classes]);
    let bins = TwoHotBins::new(config.value_bins);
    let mut rewards = Vec::new();
    let mut continuations = Vec::new();
    let mut value = None;
    for time in 0..horizon {
        let action = graph.parameter(&format!("action_{time}"), &[rows, config.action_count]);
        deter = core.forward(&mut graph, deter, stoch, action, rows);
        let logits = prior.forward(&mut graph, deter, rows);
        let uniforms = graph.input(
            &format!("uniforms_{time}"),
            &[rows * size.stoch, size.classes],
        );
        stoch = sample(
            &mut graph,
            logits,
            uniforms,
            rows * size.stoch,
            size.classes,
            config.unimix,
        );
        let state = feature(&mut graph, deter, stoch, rows, config);
        let logits = reward.forward_frozen(&mut graph, state);
        rewards.push(scalar_mean(&mut graph, logits, &bins));
        let logits = continuation.forward_frozen(&mut graph, state);
        continuations.push(graph.sigmoid(logits));
        if time + 1 == horizon {
            let logits = critic.forward_frozen(&mut graph, state);
            value = Some(scalar_mean(&mut graph, logits, &bins));
        }
    }
    let scores = objectives(&mut graph, &rewards, &continuations, value.unwrap());
    outputs(&mut graph, scores);
    for node in graph.nodes_mut() {
        if let Op::Parameter { name } = &node.op
            && !name.starts_with("action_")
        {
            node.op = Op::Input { name: name.clone() };
        }
    }
    graph
}

fn model_weights(graph: &Graph, checkpoint: &Path) -> Weights {
    let world = SafeTensorsModel::load(checkpoint.join("world.safetensors")).unwrap();
    let behavior = SafeTensorsModel::load(checkpoint.join("behavior.safetensors")).unwrap();
    graph
        .nodes()
        .iter()
        .filter_map(|node| {
            let Op::Input { name } = &node.op else {
                return None;
            };
            let model = if name.starts_with("world.") {
                &world
            } else if name.starts_with("behavior.") {
                &behavior
            } else {
                return None;
            };
            assert_eq!(model.tensor_info()[name].shape, node.ty.shape);
            let values = model.tensor_f32(name).unwrap();
            assert!(values.iter().all(|v| v.is_finite()));
            Some((name.clone(), values))
        })
        .collect()
}

struct Batch {
    deter: Vec<f32>,
    stoch: Vec<f32>,
    actions: Vec<Vec<f32>>,
    uniforms: Vec<Vec<f32>>,
}

fn execute(
    session: &mut Session,
    batch: &Batch,
    objective: usize,
) -> ([Vec<f32>; 2], Vec<Vec<f32>>) {
    session.set_input("initial_deter", &batch.deter);
    session.set_input("initial_stoch", &batch.stoch);
    for (time, (action, uniforms)) in batch.actions.iter().zip(&batch.uniforms).enumerate() {
        session.set_parameter(&format!("action_{time}"), action);
        session.set_input(&format!("uniforms_{time}"), uniforms);
    }
    for index in 0..2 {
        session.set_input(&format!("gate_{index}"), &[f32::from(index == objective)]);
    }
    session.step();
    session.wait();
    let rows = session
        .slot_size(meganeura::runtime::ExternalSlot::Output(1))
        .unwrap()
        / 4;
    let mut scores = [vec![0.0; rows], vec![0.0; rows]];
    for (index, score) in scores.iter_mut().enumerate() {
        session.read_output_by_index(index + 1, score);
        assert!(score.iter().all(|v| v.is_finite()));
    }
    assert!(scores[0].iter().all(|&v| (0.0..=1.0).contains(&v)));
    let gradients = batch
        .actions
        .iter()
        .enumerate()
        .map(|(time, action)| {
            let name = format!("action_{time}");
            assert_eq!(session.read_params(&[&name])[0], *action);
            let mut gradient = vec![0.0; action.len()];
            session.read_param_grad(&name, &mut gradient);
            for value in &mut gradient {
                *value *= rows as f32;
                assert!(value.is_finite());
            }
            gradient
        })
        .collect();
    assert_eq!(session.adam_step_count(), 0);
    (scores, gradients)
}

fn probabilities(logits: &[f32], unimix: f32) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = logits
        .iter()
        .map(|&v| f64::from(v - maximum).exp())
        .collect::<Vec<_>>();
    let sum: f64 = exponentials.iter().sum();
    exponentials
        .iter()
        .map(|v| (1.0 - f64::from(unimix)) * v / sum + f64::from(unimix) / logits.len() as f64)
        .collect()
}

fn decode(logits: &[f32], bins: &TwoHotBins) -> f64 {
    probabilities(logits, 0.0)
        .iter()
        .zip(bins.values())
        .map(|(p, &v)| p * f64::from(v))
        .sum()
}

fn forward_reference(
    config: &DreamerConfig,
    gpu: &Arc<blade_graphics::Context>,
    weights: &Weights,
    batch: &Batch,
) -> [Vec<f64>; 2] {
    let size = config.network();
    let rows = batch.deter.len() / size.deter;
    let mut transition = build_session(
        &world::build_transition_graph(config, rows),
        gpu,
        Mode::Inference,
        false,
    );
    let mut heads = build_session(
        &world::build_head_graph(config, rows),
        gpu,
        Mode::Inference,
        false,
    );
    let mut value = build_session(
        &super::super::build_value_inference_graph(config, rows),
        gpu,
        Mode::Inference,
        false,
    );
    for session in [&mut transition, &mut heads, &mut value] {
        let names = session
            .param_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        for name in names {
            session.set_parameter(&name, &weights[&name]);
        }
    }
    let bins = TwoHotBins::new(config.value_bins);
    let mut deter = batch.deter.clone();
    let mut stoch = batch.stoch.clone();
    let mut survival = vec![1.0_f64; rows];
    let mut total = vec![0.0_f64; rows];
    for (time, action) in batch.actions.iter().enumerate() {
        transition.set_input("deter", &deter);
        transition.set_input("stoch", &stoch);
        transition.set_input("action", action);
        transition.step();
        transition.wait();
        transition.read_output_by_index(0, &mut deter);
        let mut logits = vec![0.0; stoch.len()];
        transition.read_output_by_index(1, &mut logits);
        for (group, destination) in stoch.chunks_exact_mut(size.classes).enumerate() {
            let offset = group * size.classes;
            let p = probabilities(&logits[offset..offset + size.classes], config.unimix);
            let index = (0..size.classes)
                .max_by(|&a, &b| {
                    let score = |i: usize| {
                        p[i].ln() - (-f64::from(batch.uniforms[time][offset + i]).ln()).ln()
                    };
                    score(a).total_cmp(&score(b))
                })
                .unwrap();
            destination.fill(0.0);
            destination[index] = 1.0;
        }
        heads.set_input("deter", &deter);
        heads.set_input("stoch", &stoch);
        heads.step();
        heads.wait();
        let mut rewards = vec![0.0; rows * config.value_bins];
        let mut continuations = vec![0.0; rows];
        heads.read_output_by_index(0, &mut rewards);
        heads.read_output_by_index(1, &mut continuations);
        for row in 0..rows {
            total[row] += survival[row]
                * decode(
                    &rewards[row * config.value_bins..(row + 1) * config.value_bins],
                    &bins,
                );
            survival[row] *= f64::from(continuations[row]);
        }
    }
    let state = deter
        .chunks_exact(size.deter)
        .zip(stoch.chunks_exact(size.stoch * size.classes))
        .flat_map(|(h, z)| h.iter().chain(z).copied())
        .collect::<Vec<_>>();
    value.set_input("feature", &state);
    value.step();
    value.wait();
    let mut logits = vec![0.0; rows * config.value_bins];
    value.read_output_by_index(0, &mut logits);
    for row in 0..rows {
        total[row] += survival[row]
            * decode(
                &logits[row * config.value_bins..(row + 1) * config.value_bins],
                &bins,
            );
    }
    [survival, total]
}

fn check_forward(actual: &[Vec<f32>; 2], expected: &[Vec<f64>; 2]) -> f64 {
    let mut maximum = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        for (&a, &b) in actual.iter().zip(expected) {
            let error = (f64::from(a) - b).abs() / (1.0 + b.abs());
            maximum = maximum.max(error);
            assert!(error <= 1e-3, "forward error {a} vs {b}: {error}");
        }
    }
    maximum
}

#[derive(Deserialize)]
struct Context {
    origin: serde_json::Value,
    feature: Vec<f32>,
}
#[derive(Deserialize)]
struct Capture {
    config: DreamerConfig,
    origins: Vec<Context>,
}
#[derive(Deserialize)]
struct Plan {
    origin: serde_json::Value,
    recorded: Vec<usize>,
    teacher: Vec<usize>,
}
#[derive(Deserialize)]
struct Manifest {
    checkpoint: String,
    capture: String,
    plans: Vec<Plan>,
    draws: usize,
    seed: u64,
}

fn batch(
    capture: &Capture,
    plans: &[Plan],
    start: usize,
    teacher: bool,
    draws: usize,
    seed: u64,
) -> Batch {
    let config = &capture.config;
    let size = config.network();
    let rows = 8 * draws;
    let mut batch = Batch {
        deter: Vec::new(),
        stoch: Vec::new(),
        actions: vec![vec![0.0; rows * config.action_count]; 9],
        uniforms: vec![vec![0.0; rows * size.stoch * size.classes]; 9],
    };
    for local in 0..8 {
        let index = start + local;
        let state = &capture.origins[index];
        let plan = &plans[index];
        assert_eq!(state.origin, plan.origin);
        assert_eq!(state.feature.len(), config.feature_dim());
        assert!(state.feature.iter().all(|x| x.is_finite()));
        let actions = if teacher {
            &plan.teacher
        } else {
            &plan.recorded
        };
        assert_eq!(actions.len(), 9);
        for draw in 0..draws {
            let row = local * draws + draw;
            batch.deter.extend_from_slice(&state.feature[..size.deter]);
            batch.stoch.extend_from_slice(&state.feature[size.deter..]);
            let mut rng = StdRng::seed_from_u64(seed + (index * draws + draw) as u64);
            for (time, &action) in actions.iter().enumerate() {
                assert!(action < config.action_count);
                batch.actions[time][row * config.action_count + action] = 1.0;
                let width = size.stoch * size.classes;
                for value in &mut batch.uniforms[time][row * width..(row + 1) * width] {
                    *value = rng.random_range(1e-5..1.0 - 1e-5);
                }
            }
        }
    }
    batch
}

fn save(path: &Path, value: &impl Serialize) {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(std::io::BufWriter::new(file), value).unwrap();
}

#[test]
fn action_probe_only_differentiates_action_leaves() {
    let graph = graph(&DreamerConfig::tiny(3), 2, 3);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert_eq!(plan.param_buffers.len(), 3);
    assert_eq!(plan.param_grad_pairs.len(), 3);
    assert!(
        plan.param_buffers
            .iter()
            .all(|(name, _)| name.starts_with("action_"))
    );
    assert!(
        plan.input_buffers
            .iter()
            .any(|(name, _)| name.starts_with("world."))
    );
    assert!(
        plan.input_buffers
            .iter()
            .any(|(name, _)| name.starts_with("behavior.value."))
    );
    assert_eq!(graph.outputs().len(), 3);
}

#[test]
#[ignore = "requires separately declared frozen action-path corpus and host guard"]
fn recorded_action_path_derivatives() {
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(std::env::var("KINDLE_ACTION_PROBE_MANIFEST").unwrap()).unwrap(),
    )
    .unwrap();
    let capture: Capture = serde_json::from_slice(&fs::read(&manifest.capture).unwrap()).unwrap();
    assert_eq!(
        (
            manifest.plans.len(),
            capture.origins.len(),
            manifest.draws,
            manifest.seed
        ),
        (120, 120, 8, 2026110600)
    );
    let config = &capture.config;
    assert_eq!(config.feature_dim(), 2560);
    assert_eq!(config.action_count, 18);
    let output = std::path::PathBuf::from(std::env::var("KINDLE_ACTION_PROBE_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let started = Instant::now();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let graph = graph(config, 64, 9);
    let weights = model_weights(&graph, Path::new(&manifest.checkpoint));
    let mut session = build_session(&graph, &gpu, Mode::Training, false);
    session.clear_optimizer();
    session.disable_grad_clip();
    for (name, values) in &weights {
        session.set_input(name, values);
    }
    let mut forward_errors = Vec::new();
    let mut minimum_headroom = u64::MAX;
    for start in (0..120).step_by(8) {
        let memory = gpu.memory_stats();
        let headroom = memory.budget.saturating_sub(memory.usage);
        assert!(headroom >= 2 << 30);
        minimum_headroom = minimum_headroom.min(headroom);
        for teacher in [false, true] {
            let batch = batch(
                &capture,
                &manifest.plans,
                start,
                teacher,
                manifest.draws,
                manifest.seed,
            );
            let mut previous = None;
            for objective in 0..2 {
                let (scores, gradients) = execute(&mut session, &batch, objective);
                if let Some(previous) = previous {
                    assert_eq!(scores, previous);
                }
                if start == 0 && objective == 0 {
                    let reference = forward_reference(config, &gpu, &weights, &batch);
                    forward_errors.push(check_forward(&scores, &reference));
                }
                save(
                    &output.join(format!(
                        "batch-{start}-{}-{objective}.json",
                        usize::from(teacher)
                    )),
                    &serde_json::json!({"start":start,"teacher":teacher,"objective":objective,"scores":scores,"gradients":gradients}),
                );
                previous = Some(scores);
            }
        }
        eprintln!("action-path batch {start} complete");
    }
    let mut readback = Readback::new(Arc::clone(&gpu));
    for (name, expected) in &weights {
        let mut actual = vec![0.0; expected.len()];
        readback.read_regions(&mut [(session.input_buffer(name).unwrap(), &mut actual)]);
        assert_eq!(&actual, expected, "frozen input {name}");
    }
    assert_eq!(session.adam_step_count(), 0);
    save(
        &output.join("result.json"),
        &serde_json::json!({"status":"complete","seconds":started.elapsed().as_secs_f64(),
        "origins":120,"draws":8,"plans":2,"objectives":2,"batches":15,"forward_errors":forward_errors,
        "frozen_input_tensors":weights.len(),"minimum_headroom_bytes":minimum_headroom,
        "updates":0,"actions":0,"optimizer_steps":session.adam_step_count()}),
    );
}

mod tests;
