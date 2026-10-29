//! Fixed-checkpoint derivatives; never enters the learner or applies an optimizer.

use super::*;
use meganeura::{Graph, graph::Op};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

const TERMS: [&str; 9] = [
    "total",
    "cdp",
    "dynamics",
    "representation",
    "reward",
    "continuation",
    "replay_value",
    "disagreement",
    "expert",
];
type Gradients = BTreeMap<String, Vec<f32>>;

fn gated_graph(config: &DreamerConfig) -> Graph {
    assert!(config.is_cdp() && config.expert_world_gradient());
    assert!(config.video_encoder.is_none() && !config.actor_critic_gradient);
    assert!(config.color_centroid_supervision.is_none() && config.open_loop_supervision.is_none());
    assert!(config.uses_disagreement());
    assert_eq!(config.loss_scales.reconstruction, 0.0);
    let mut graph = world::build_staged_training_graph(config);
    let outputs = graph.outputs().to_vec();
    let actor_scale = graph
        .nodes()
        .iter()
        .find_map(|node| match &node.op {
            Op::Input { name } if name == "actor_update_scale" => Some(node.id),
            _ => None,
        })
        .unwrap();
    let scales = config.loss_scales;
    let specifications = [
        (world::LOSS_TOTAL, 1.0),
        (world::LOSS_FUTURE_PREDICTION, scales.future_prediction),
        (world::LOSS_DYNAMICS, scales.dynamics),
        (world::LOSS_REPRESENTATION, scales.representation),
        (world::LOSS_REWARD, scales.reward),
        (world::LOSS_CONTINUATION, scales.continuation),
        (world::LOSS_REPLAY_VALUE, scales.replay_value),
        (world::LOSS_EXPLORATION, 1.0),
        (
            world::LOSS_EXPERT,
            config.expert_imitation.as_ref().unwrap().scale,
        ),
    ];
    let mut weighted = Vec::new();
    for (name, (index, scale)) in TERMS.iter().zip(specifications) {
        let mut value = graph.scale(outputs[index], scale);
        if *name == "expert" {
            value = graph.mul(value, actor_scale);
        }
        let gate = graph.input(&format!("probe_gate_{name}"), &[1]);
        weighted.push(graph.mul(value, gate));
    }
    let selected = super::super::networks::sum(&mut graph, &weighted);
    graph.set_outputs(std::iter::once(selected).chain(outputs).collect());
    graph
}

fn save_json(path: &Path, value: &impl serde::Serialize) {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), value).unwrap();
}

fn floats(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()));
    values
}

#[derive(serde::Deserialize)]
struct BatchMetadata {
    previous_actions: Vec<Vec<usize>>,
    previous_expert_actions: Vec<Vec<usize>>,
    rewards: Vec<Vec<f32>>,
}

fn batch(core: &mut DreamerCore, directory: &Path, index: usize) -> (SequenceBatch, f32) {
    let config = &core.config;
    let (rows, length, size) = (config.batch_size, config.batch_length, config.network());
    let metadata: BatchMetadata =
        serde_json::from_slice(&fs::read(directory.join(format!("batch-{index}.json"))).unwrap())
            .unwrap();
    let contexts = floats(&directory.join(format!("batch-{index}-context.f32")));
    assert_eq!(contexts.len(), rows * config.feature_dim());
    let mut batch = SequenceBatch::default();
    for context in contexts.chunks_exact(config.feature_dim()) {
        batch
            .initial_deter
            .extend_from_slice(&context[..size.deter]);
        batch
            .initial_stoch
            .extend_from_slice(&context[size.deter..]);
        for variable in context[size.deter..].chunks_exact(size.classes) {
            assert_eq!(variable.iter().sum::<f32>(), 1.0);
            assert!(variable.iter().all(|&v| v == 0.0 || v == 1.0));
        }
    }
    let raw = fs::read(directory.join(format!("batch-{index}.rgb"))).unwrap();
    let reference = floats(&directory.join(format!("batch-{index}-reference.f32")));
    let pixels_per_frame = 210 * 160 * 3;
    let width = config.observation_dim();
    assert_eq!(raw.len(), rows * length * pixels_per_frame);
    assert_eq!(reference.len(), rows * length * width);
    assert_eq!(metadata.previous_actions.len(), length);
    assert_eq!(metadata.previous_expert_actions.len(), length);
    assert_eq!(metadata.rewards.len(), length);
    let mut graph = Graph::new();
    let input = graph.input("observation", &[rows, width]);
    let output = graph.neg(input);
    graph.set_outputs(vec![output]);
    let mut session = build_session(&graph, &core.gpu, Mode::Inference, false);
    let mut resize =
        crate::vision::preprocess_gpu::GpuPreprocessor::rgb(Arc::clone(&core.gpu), rows, 64);
    let mut maximum_error = 0.0_f32;
    for time in 0..length {
        let frames = (0..rows)
            .map(|row| {
                let start = (time * rows + row) * pixels_per_frame;
                (row, &raw[start..start + pixels_per_frame], 160, 210)
            })
            .collect::<Vec<_>>();
        resize.cpu_frames(&mut session, &frames);
        let mut observation = vec![0.0; rows * width];
        core.readback.read_regions(&mut [(
            session.input_buffer("observation").unwrap(),
            &mut observation,
        )]);
        for (&actual, &expected) in observation
            .iter()
            .zip(&reference[time * rows * width..(time + 1) * rows * width])
        {
            maximum_error = maximum_error.max((actual - expected).abs());
        }
        batch.observations.push(observation);
        assert_eq!(metadata.previous_actions[time].len(), rows);
        assert_eq!(metadata.previous_expert_actions[time].len(), rows);
        assert_eq!(metadata.rewards[time].len(), rows);
        let mut actions = vec![0.0; rows * config.action_count];
        for (row, &action) in metadata.previous_actions[time].iter().enumerate() {
            assert!(action < config.action_count);
            actions[row * config.action_count + action] = 1.0;
        }
        batch.previous_actions.push(actions);
        batch.previous_expert_action.push(
            metadata.previous_expert_actions[time]
                .iter()
                .map(|&a| {
                    assert!(a < config.action_count);
                    Some(a)
                })
                .collect(),
        );
        batch.rewards.push(metadata.rewards[time].clone());
        batch.flags.push(vec![FrameFlags::default(); rows]);
    }
    assert!(
        maximum_error <= 1e-6,
        "native resize error: {maximum_error}"
    );
    (batch, maximum_error)
}

fn late_targets(core: &mut DreamerCore, batch: &SequenceBatch, behavior: &BehaviorTrainingBatch) {
    // Same CPU-target path and normalization as train_world; no optimizer call.
    assert!(core.targets_gpu.is_none());
    let config = &core.config;
    let rows = config.batch_size;
    let bins = config.value_bins;
    core.world_train.set_input(
        "actor_update_scale",
        &[config.actor_update_scale(core.learner_step)],
    );
    for time in 0..config.batch_length {
        let mut rewards = vec![0.0; rows * bins];
        for row in 0..rows {
            core.bins.encode(
                batch.rewards[time][row],
                &mut rewards[row * bins..(row + 1) * bins],
            );
        }
        core.world_train
            .set_input(&format!("reward_target_{time}"), &rewards);
        let continuation = batch.flags[time]
            .iter()
            .map(|flag| {
                if flag.is_terminal {
                    0.0
                } else {
                    config.continuation_discount()
                }
            })
            .collect::<Vec<_>>();
        core.world_train
            .set_input(&format!("continuation_target_{time}"), &continuation);
        let mut values = vec![0.0; rows * bins];
        let mut slow = values.clone();
        let mut weights = vec![0.0; rows];
        if time + 1 < config.batch_length {
            let start = time * rows;
            values.copy_from_slice(
                &behavior.replay_value_target[start * bins..(start + rows) * bins],
            );
            slow.copy_from_slice(&behavior.replay_slow_target[start * bins..(start + rows) * bins]);
            let correction = config.batch_length as f32 / (config.batch_length - 1) as f32;
            for (weight, source) in weights
                .iter_mut()
                .zip(&behavior.replay_weight[start..start + rows])
            {
                *weight = source * correction;
            }
        }
        for (name, target) in [
            ("replay_value_target", values),
            ("replay_slow_target", slow),
            ("replay_value_weight", weights),
            (
                "expert_action_target",
                world_expert_targets(&behavior.expert_action_target, config, time, 0, rows),
            ),
        ] {
            core.world_train
                .set_input(&format!("{name}_{time}"), &target);
        }
    }
}

fn gradients(session: &Session) -> Gradients {
    session
        .param_names()
        .into_iter()
        .filter(|name| session.has_param_grad(name))
        .map(|name| {
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut values);
            assert!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite gradient: {name}"
            );
            (name.to_owned(), values)
        })
        .collect()
}

fn export(path: &Path, values: &Gradients) {
    let mut file = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap(),
    );
    for gradient in values.values() {
        for value in gradient {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

fn agreement(reference: &Gradients, candidate: &Gradients) -> serde_json::Value {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        candidate.keys().collect::<Vec<_>>()
    );
    let (mut squared_error, mut squared_reference, mut maximum_error, mut maximum_reference) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (name, values) in reference {
        assert_eq!(values.len(), candidate[name].len());
        for (&a, &b) in values.iter().zip(&candidate[name]) {
            let (a, b) = (f64::from(a), f64::from(b));
            squared_error += (a - b).powi(2);
            squared_reference += a * a;
            maximum_error = maximum_error.max((a - b).abs());
            maximum_reference = maximum_reference.max(a.abs());
        }
    }
    let relative = squared_error.sqrt() / squared_reference.sqrt().max(1e-30);
    serde_json::json!({"relative_l2":relative,"maximum_absolute_error":maximum_error,"reference_norm":squared_reference.sqrt(), "reference_maximum":maximum_reference,
        "passed":relative <= 1e-3 && maximum_error <= 1e-3*(1.0+maximum_reference)})
}

fn group(name: &str) -> &'static str {
    if name.starts_with("world.representation.encoder.") {
        "encoder"
    } else if name.starts_with("world.representation.posterior.") {
        "posterior"
    } else if name.starts_with("world.dynamics.") {
        "dynamics"
    } else {
        "other"
    }
}

fn matrices(terms: &[Gradients]) -> serde_json::Value {
    let mut parameters = BTreeMap::new();
    let mut groups: BTreeMap<&str, Vec<Vec<f64>>> = BTreeMap::new();
    for (name, values) in &terms[0] {
        let mut matrix = vec![vec![0.0; TERMS.len()]; TERMS.len()];
        for i in 0..TERMS.len() {
            for j in i..TERMS.len() {
                let dot: f64 = terms[i][name]
                    .iter()
                    .zip(&terms[j][name])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                matrix[i][j] = dot;
                matrix[j][i] = dot;
            }
        }
        let aggregate = groups
            .entry(group(name))
            .or_insert_with(|| vec![vec![0.0; TERMS.len()]; TERMS.len()]);
        for i in 0..TERMS.len() {
            for j in 0..TERMS.len() {
                aggregate[i][j] += matrix[i][j];
            }
        }
        parameters.insert(
            name,
            serde_json::json!({"elements":values.len(),"group":group(name),"dot_products":matrix}),
        );
    }
    let groups = groups
        .into_iter()
        .map(|(name, matrix)| {
            let norms: Vec<_> = (0..TERMS.len()).map(|i| matrix[i][i].sqrt()).collect();
            let cosine: Vec<Vec<_>> = (0..TERMS.len())
                .map(|i| {
                    (0..TERMS.len())
                        .map(|j| {
                            let denominator = norms[i] * norms[j];
                            (denominator > 0.0).then(|| matrix[i][j] / denominator)
                        })
                        .collect()
                })
                .collect();
            (
                name,
                serde_json::json!({"norms":norms,"dot_products":matrix,"cosines":cosine}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    serde_json::json!({"terms":TERMS,"groups":groups,"parameters":parameters})
}

#[test]
fn zero_gradients_have_no_cosine_and_groups_are_separate() {
    let mut terms = vec![Gradients::new(); TERMS.len()];
    for term in &mut terms {
        term.insert("world.representation.encoder.weight".into(), vec![0.0, 0.0]);
        term.insert("world.dynamics.weight".into(), vec![1.0, 2.0]);
    }
    let result = matrices(&terms);
    assert!(result["groups"]["encoder"]["cosines"][0][0].is_null());
    assert_eq!(result["groups"]["dynamics"]["dot_products"][0][0], 5.0);
    assert!(agreement(&terms[0], &terms[1])["passed"].as_bool().unwrap());
    terms[1].get_mut("world.dynamics.weight").unwrap()[0] = 2.0;
    assert!(!agreement(&terms[0], &terms[1])["passed"].as_bool().unwrap());
}

#[test]
fn diagnostic_gates_preserve_original_graph_prefix() {
    let mut config = DreamerConfig::tiny(18);
    config.batch_size = 2;
    config.batch_length = 2;
    config.world_backprop_length = 2;
    config.observation_kind = crate::ObservationKind::Rgb32;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.cdp_centered = true;
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    config.expert_imitation = Some(super::super::config::ExpertImitation {
        scale: 100.0,
        action_weights: vec![1.0; 18],
        world_gradient: true,
    });
    let original = world::build_staged_training_graph(&config);
    let probe = gated_graph(&config);
    assert_eq!(
        format!("{:?}", original.nodes()),
        format!("{:?}", &probe.nodes()[..original.nodes().len()])
    );
    assert_eq!(&probe.outputs()[1..], original.outputs());
    assert_eq!(
        probe
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op,Op::Input{name} if name.starts_with("probe_gate_")))
            .count(),
        TERMS.len()
    );
}

#[test]
#[ignore = "requires guarded RX7900XT and a declared fixed recorded corpus"]
fn recorded_world_gradient_contributions() {
    let directory = PathBuf::from(std::env::var_os("KINDLE_GRADIENT_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var_os("KINDLE_GRADIENT_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["batch_count"], 2);
    assert_eq!(manifest["selected_frames"], 512);
    let checkpoint = PathBuf::from(manifest["checkpoint"].as_str().unwrap());
    let mut core = DreamerCore::restore(&checkpoint).unwrap();
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!core.gpu_device().is_software_emulated);
    let declared: DreamerConfig = serde_json::from_value(manifest["config"].clone()).unwrap();
    assert_eq!(
        serde_json::to_value(&core.config).unwrap(),
        serde_json::to_value(declared).unwrap()
    );
    assert_eq!(
        (
            core.config.batch_size,
            core.config.batch_length,
            core.config.world_backprop_length
        ),
        (16, 16, 16)
    );
    assert_eq!(core.config.observation_kind, crate::ObservationKind::Rgb64);
    assert!(core.world_posterior.is_none() && core.targets_gpu.is_none());
    let counters = (core.learner_step, core.environment_step);
    let config = world_training_config(&core.config);
    let production_graph = world::build_staged_training_graph(&config);
    let graph = gated_graph(&config);
    let mut probe = super::super::runtime::build_training_session(&graph, &core.gpu, &config);
    let names = probe
        .param_names()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    for name in &names {
        let values = core.world_train.read_params(&[name]);
        probe.set_parameter(name, &values[0]);
    }
    for session in [&mut core.world_train, &mut probe] {
        session.clear_optimizer();
        session.disable_grad_clip();
    }
    core.save_checkpoint(output.join("before")).unwrap();
    let mut minimum_headroom = u64::MAX;
    let started = Instant::now();
    let mut results = Vec::new();
    for index in 0..2 {
        let memory = core.gpu_memory_budget();
        let headroom = memory.budget_bytes.saturating_sub(memory.usage_bytes);
        assert!(headroom >= 2 << 30);
        minimum_headroom = minimum_headroom.min(headroom);
        let path = output.join(format!("batch-{index}"));
        fs::create_dir(&path).unwrap();
        let (batch, resize_error) = batch(&mut core, &directory, index);
        core.world_train.zero_grad();
        let posterior = core.sample_posterior_batch(&batch);
        let mut targets = core.imagine_and_target(&batch, &posterior);
        (targets.expert_action_target, targets.expert_labeled_rows) =
            batch.expert_targets(&core.config);
        assert_eq!(targets.expert_labeled_rows, 240);
        late_targets(&mut core, &batch, &targets);
        let mut captured_inputs = BTreeMap::new();
        for node in production_graph.nodes() {
            let Op::Input { name } = &node.op else {
                continue;
            };
            if probe.input_buffer(name).is_none() {
                continue;
            }
            let mut values = vec![0.0; node.ty.num_elements()];
            core.readback
                .read_regions(&mut [(core.world_train.input_buffer(name).unwrap(), &mut values)]);
            assert!(values.iter().all(|x| x.is_finite()), "{name}");
            probe.set_input(name, &values);
            assert!(captured_inputs.insert(name.clone(), values).is_none());
        }
        export(&path.join("inputs.f32"), &captured_inputs);
        save_json(
            &path.join("input-layout.json"),
            &captured_inputs
                .iter()
                .map(|(name, values)| (name, values.len()))
                .collect::<Vec<_>>(),
        );
        core.world_train.finish_prepared_step();
        core.world_train.wait();
        let production_loss = core.world_train.read_loss();
        assert!(production_loss.is_finite());
        let production = gradients(&core.world_train);
        export(&path.join("production.f32"), &production);
        save_json(
            &path.join("gradient-layout.json"),
            &production
                .iter()
                .map(|(n, g)| (n, g.len()))
                .collect::<Vec<_>>(),
        );
        let mut terms = Vec::new();
        let mut losses = Vec::new();
        for (selected, term) in TERMS.iter().enumerate() {
            for (i, name) in TERMS.iter().enumerate() {
                probe.set_input(&format!("probe_gate_{name}"), &[f32::from(i == selected)]);
            }
            probe.step();
            probe.wait();
            let loss = probe.read_loss();
            assert!(loss.is_finite());
            losses.push(loss);
            let values = gradients(&probe);
            export(&path.join(format!("{term}.f32")), &values);
            terms.push(values);
        }
        let qualification = agreement(&production, &terms[0]);
        let group_qualification = ["encoder", "posterior", "dynamics", "other"]
            .map(|selected| {
                let filter = |source: &Gradients| {
                    source
                        .iter()
                        .filter(|(name, _)| group(name) == selected)
                        .map(|(name, values)| (name.clone(), values.clone()))
                        .collect::<Gradients>()
                };
                (
                    selected,
                    agreement(&filter(&production), &filter(&terms[0])),
                )
            })
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let groups_passed = group_qualification
            .values()
            .all(|value| value["passed"].as_bool().unwrap());
        let mut sum = terms[0].clone();
        for values in sum.values_mut() {
            values.fill(0.0);
        }
        for component in &terms[1..] {
            for (name, values) in component {
                for (total, value) in sum.get_mut(name).unwrap().iter_mut().zip(values) {
                    *total += value;
                }
            }
        }
        let sum_check = agreement(&terms[0], &sum);
        let component_norm_sum: f64 = terms[1..]
            .iter()
            .map(|g| {
                g.values()
                    .flatten()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum();
        let error_norm = sum_check["relative_l2"].as_f64().unwrap()
            * sum_check["reference_norm"].as_f64().unwrap();
        let sum_passed = error_norm <= 1e-3 * component_norm_sum
            && sum_check["maximum_absolute_error"].as_f64().unwrap()
                <= 1e-3 * (1.0 + sum_check["reference_maximum"].as_f64().unwrap());
        let scalar_passed = (production_loss - losses[0]).abs()
            <= 1e-3 * (1.0 + production_loss.abs())
            && (losses[1..].iter().sum::<f32>() - losses[0]).abs()
                <= 1e-3 * (1.0 + losses[0].abs());
        let result = serde_json::json!({"index":index,"production_loss":production_loss,"losses":losses,"resize_max_error":resize_error,
            "production_gradient_agreement":qualification,"group_qualification":group_qualification,"component_sum_agreement":sum_check,"component_norm_sum":component_norm_sum,
            "component_sum_passed":sum_passed,"scalar_passed":scalar_passed,"attribution":matrices(&terms)});
        save_json(&path.join("result.json"), &result);
        assert!(
            qualification["passed"].as_bool().unwrap()
                && groups_passed
                && sum_passed
                && scalar_passed,
            "gradient qualification: {result}"
        );
        results.push(result);
        eprintln!("gradient batch {index} complete; source weights retained");
    }
    assert_eq!((core.learner_step, core.environment_step), counters);
    core.save_checkpoint(output.join("after")).unwrap();
    let result = serde_json::json!({"status":"complete","seconds":started.elapsed().as_secs_f64(),"learner_step":core.learner_step,
        "environment_step":core.environment_step,"learner_updates":0,"minimum_sampled_headroom":minimum_headroom,"batches":results});
    save_json(&output.join("result.json"), &result);
}
