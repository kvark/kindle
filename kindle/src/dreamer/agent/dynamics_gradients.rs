//! New dynamics derivatives on archived behavior inputs, without a learner.

use super::*;
use meganeura::{Graph, data::safetensors::SafeTensorsModel, graph::Op};
use std::{collections::BTreeMap, io::Write, path::PathBuf};

type Arrays = BTreeMap<String, Vec<f32>>;

fn gated_graph(config: &DreamerConfig) -> Graph {
    let starts = config.batch_size * config.batch_length;
    let mut graph = behavior::build_training_graph(
        config,
        starts * config.imagination_length,
        config.batch_size * (config.batch_length - 1),
    );
    let outputs = graph.outputs().to_vec();
    let dynamics = graph.scale(
        outputs[behavior::dynamics_actor_output(config)],
        config.loss_scales.policy * config.actor_dynamics_scale.unwrap(),
    );
    let total_gate = graph.input("probe_total", &[1]);
    let dynamics_gate = graph.input("probe_dynamics", &[1]);
    let total = graph.mul(outputs[0], total_gate);
    let dynamics = graph.mul(dynamics, dynamics_gate);
    let selected = graph.add(total, dynamics);
    graph.set_outputs(std::iter::once(selected).chain(outputs).collect());
    graph
}

fn json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn save(path: &Path, value: &impl serde::Serialize) {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), value).unwrap();
}

fn read_arrays(directory: &Path, layout: &str, data: &str) -> Arrays {
    let layout: Vec<(String, usize)> = json(&directory.join(layout));
    assert!(layout.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let bytes = fs::read(directory.join(data)).unwrap();
    assert_eq!(
        bytes.len(),
        layout.iter().map(|(_, n)| n * 4).sum::<usize>()
    );
    let mut offset = 0;
    layout
        .into_iter()
        .map(|(name, count)| {
            let values = bytes[offset..offset + count * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert!(values.iter().all(|v| v.is_finite()));
            offset += count * 4;
            (name, values)
        })
        .collect()
}

fn export(directory: &Path, name: &str, arrays: &Arrays) {
    save(
        &directory.join(format!("{name}-layout.json")),
        &arrays
            .iter()
            .map(|(name, values)| (name, values.len()))
            .collect::<Vec<_>>(),
    );
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(format!("{name}.f32")))
        .unwrap();
    let mut file = std::io::BufWriter::new(file);
    for values in arrays.values() {
        assert!(values.iter().all(|v| v.is_finite()));
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

fn read_parameters(session: &Session) -> Arrays {
    let names = session.param_names();
    names
        .iter()
        .zip(session.read_params(&names))
        .map(|(name, values)| ((*name).to_owned(), values))
        .collect()
}

fn read_gradients(session: &Session) -> Arrays {
    session
        .param_names()
        .into_iter()
        .map(|name| {
            assert!(session.has_param_grad(name));
            let mut values = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut values);
            (name.to_owned(), values)
        })
        .collect()
}

fn load_parameters(session: &mut Session, world: &SafeTensorsModel, behavior: &SafeTensorsModel) {
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in names {
        let model = if name.starts_with("world.") {
            world
        } else {
            assert!(name.starts_with("behavior."));
            behavior
        };
        let values = model.tensor_f32(&name).unwrap();
        assert_eq!(session.param_size(&name), Some(values.len()));
        assert!(values.iter().all(|v| v.is_finite()));
        session.set_parameter(&name, &values);
    }
}

fn errors(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .fold((0.0_f64, 0.0_f64), |(absolute, normalized), (&a, &b)| {
            assert!(a.is_finite() && b.is_finite());
            let error = (f64::from(a) - f64::from(b)).abs();
            (
                absolute.max(error),
                normalized.max(error / (1.0 + f64::from(b).abs())),
            )
        })
}

struct Imagined {
    reward: Vec<f32>,
    value: Vec<f32>,
    continuation: Vec<f32>,
    bonus: Vec<f32>,
    action: Vec<f32>,
}

fn targets(
    config: &DreamerConfig,
    imagined: &Imagined,
    normalizer: &mut PercentileNormalizer,
) -> (Vec<f32>, Vec<f32>, f32) {
    let starts = config.batch_size * config.batch_length;
    let horizon = config.imagination_length;
    let mut returns = vec![vec![0.0; starts]; horizon];
    let mut weights = vec![vec![0.0; starts]; horizon];
    for start in 0..starts {
        let reward = (0..=horizon)
            .map(|t| {
                imagined.reward[t * starts + start]
                    + if t > 0 {
                        config.intrinsic_reward_scale * imagined.bonus[(t - 1) * starts + start]
                    } else {
                        0.0
                    }
            })
            .collect::<Vec<_>>();
        let continuation = (0..=horizon)
            .map(|t| imagined.continuation[t * starts + start].clamp(0.0, 1.0))
            .collect::<Vec<_>>();
        let terminal = continuation.iter().map(|c| 1.0 - c).collect::<Vec<_>>();
        let value = (0..=horizon)
            .map(|t| imagined.value[t * starts + start])
            .collect::<Vec<_>>();
        let trajectory = lambda_returns(
            &vec![0.0; horizon + 1],
            &terminal,
            &reward,
            &value,
            &value,
            1.0,
            config.lambda,
        );
        let trajectory_weights = continuation_weights(&continuation[..horizon]);
        for t in 0..horizon {
            returns[t][start] = trajectory[t];
            weights[t][start] = trajectory_weights[t];
        }
    }
    normalizer.update(&flatten_time(&returns));
    let scale = normalizer.scale();
    let mut action_target = vec![0.0; starts * horizon * config.action_count];
    for (row, sample) in imagined
        .action
        .chunks_exact(config.action_count)
        .enumerate()
    {
        assert_eq!(sample.iter().filter(|&&v| v == 1.0).count(), 1);
        assert!(sample.iter().all(|&v| v == 0.0 || v == 1.0));
        let action = sample.iter().position(|&v| v == 1.0).unwrap();
        let (t, start) = (row / starts, row % starts);
        action_target[row * config.action_count + action] =
            weights[t][start] * ((returns[t][start] - imagined.value[row]) / scale);
    }
    (action_target, flatten_time(&weights), scale)
}

fn gradient_agreement(total: &Arrays, baseline: &Arrays, dynamics: &Arrays) -> serde_json::Value {
    assert_eq!(
        total.keys().collect::<Vec<_>>(),
        baseline.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        total.keys().collect::<Vec<_>>(),
        dynamics.keys().collect::<Vec<_>>()
    );
    let mut groups = BTreeMap::new();
    for (group, prefix) in [("actor", "behavior.actor."), ("critic", "behavior.value.")] {
        let (mut squared_error, mut squared_reference, mut maximum) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (name, actual) in total.iter().filter(|(name, _)| name.starts_with(prefix)) {
            assert_eq!(actual.len(), baseline[name].len());
            assert_eq!(actual.len(), dynamics[name].len());
            for ((&a, &b), &d) in actual.iter().zip(&baseline[name]).zip(&dynamics[name]) {
                assert!(a.is_finite() && b.is_finite() && d.is_finite());
                if group == "critic" {
                    assert_eq!(d, 0.0);
                }
                let expected = f64::from(b) + f64::from(d);
                let error = f64::from(a) - expected;
                squared_error += error * error;
                squared_reference += expected * expected;
                maximum = maximum.max(error.abs() / (1.0 + expected.abs()));
            }
        }
        let relative = squared_error.sqrt() / squared_reference.sqrt().max(1e-30);
        groups.insert(
            group,
            serde_json::json!({"relative_l2":relative, "normalized_maximum":maximum,
            "passed":relative <= 1e-3 && maximum <= 1e-3}),
        );
    }
    serde_json::to_value(groups).unwrap()
}

#[derive(serde::Deserialize)]
struct Manifest {
    checkpoint: PathBuf,
    batches: Vec<PathBuf>,
}

#[test]
#[ignore = "requires declared archived batches and a guarded healthy AMD device"]
fn archived_batches_dynamics_gradients() {
    let input = PathBuf::from(std::env::var_os("KINDLE_DYNAMICS_GRADIENT_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var_os("KINDLE_DYNAMICS_GRADIENT_OUTPUT").unwrap());
    let manifest: Manifest = json(&input.join("manifest.json"));
    assert_eq!(manifest.batches.len(), 2);
    let metadata = read_checkpoint_metadata(&manifest.checkpoint).unwrap();
    assert_eq!(
        (metadata.learner_step, metadata.environment_step),
        (12477, 200000)
    );
    let mut config = metadata.config.clone();
    assert!(config.actor_dynamics_scale.is_none() && config.is_cdp());
    assert_eq!(
        (
            config.batch_size,
            config.batch_length,
            config.imagination_length
        ),
        (16, 16, 15)
    );
    assert!(config.uses_disagreement());
    config.actor_dynamics_scale = Some(0.5);
    config.validate();
    fs::create_dir(&output).unwrap();
    save(&output.join("config.json"), &config);
    let graph = gated_graph(&config);
    let world = SafeTensorsModel::load(manifest.checkpoint.join("world.safetensors")).unwrap();
    let behavior =
        SafeTensorsModel::load(manifest.checkpoint.join("behavior.safetensors")).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let device = crate::gpu_device_info(gpu.device_information());
    assert_eq!(
        device.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let mut imagination = super::super::runtime::build_learner_inference_session(
        &world::build_imagination_graph(&config),
        &gpu,
        &config,
    );
    let mut probe = super::super::runtime::build_training_session(&graph, &gpu, &config);
    load_parameters(&mut imagination, &world, &behavior);
    load_parameters(&mut probe, &world, &behavior);
    probe.clear_optimizer();
    probe.disable_grad_clip();
    assert_eq!(probe.param_names().len(), 22);
    let parameters = read_parameters(&probe);
    let imagination_parameters = read_parameters(&imagination);
    export(&output, "parameters-before", &parameters);
    let snapshot = graph
        .nodes()
        .iter()
        .filter_map(|node| {
            let Op::Input { name } = &node.op else {
                return None;
            };
            let source = name.strip_prefix("dynamics_parameter_")?;
            assert_eq!(world.tensor_info()[source].shape, node.ty.shape);
            Some((name.clone(), world.tensor_f32(source).unwrap()))
        })
        .collect::<Arrays>();
    assert!(!snapshot.is_empty());
    export(&output, "snapshot", &snapshot);
    for (name, values) in &snapshot {
        probe.set_input(name, values);
    }
    let (starts, horizon, size) = (
        config.batch_size * config.batch_length,
        config.imagination_length,
        config.network(),
    );
    let all_rows = starts * (horizon + 1);
    let mut decoder = super::super::categorical_gpu::CategoricalGpu::new(
        Arc::clone(&gpu),
        &TwoHotBins::new(config.value_bins),
        all_rows,
    );
    let mut readback = Readback::new(Arc::clone(&gpu));
    let mut rng = DreamerRngs::resumed(
        config.seed,
        metadata.learner_step,
        metadata.environment_step,
    )
    .imagination;
    let mut normalizer = PercentileNormalizer::new(config.return_norm_rate, 1.0);
    normalizer.restore(metadata.return_low, metadata.return_high);
    let started = Instant::now();
    let mut results = Vec::new();
    let mut minimum_headroom = u64::MAX;
    for (index, source) in manifest.batches.iter().enumerate() {
        let stats = gpu.memory_stats();
        let headroom = stats.budget.saturating_sub(stats.usage);
        minimum_headroom = minimum_headroom.min(headroom);
        assert!(headroom >= 2 << 30);
        let directory = output.join(format!("batch-{index}"));
        fs::create_dir(&directory).unwrap();
        let inputs = read_arrays(source, "input-layout.json", "inputs.f32");
        for (name, values) in &inputs {
            probe.set_input(name, values);
        }
        let mut shared = Arrays::new();
        let initial = &inputs["imagined_feature"][..starts * config.feature_dim()];
        let deter = initial
            .chunks_exact(config.feature_dim())
            .flat_map(|r| r[..size.deter].iter().copied())
            .collect();
        let stoch = initial
            .chunks_exact(config.feature_dim())
            .flat_map(|r| r[size.deter..].iter().copied())
            .collect();
        shared.insert("deter".into(), deter);
        shared.insert("stoch".into(), stoch);
        for time in 0..horizon {
            for (name, count) in [
                ("action_uniforms", starts * config.action_count),
                ("latent_uniforms", starts * size.stoch * size.classes),
            ] {
                shared.insert(
                    format!("{name}_{time}"),
                    (0..count).map(|_| rng.random::<f32>()).collect(),
                );
            }
        }
        export(&directory, "shared", &shared);
        for (name, values) in &shared {
            imagination.set_input(name, values);
            probe.set_input(&format!("dynamics_{name}"), values);
        }
        imagination.step();
        let decoded = decoder.decode(&[
            (&imagination, world::IMAGINATION_REWARD),
            (&imagination, world::IMAGINATION_VALUE),
            (&imagination, world::IMAGINATION_VALUE),
        ]);
        let mut imagined = Imagined {
            reward: vec![0.0; all_rows],
            value: vec![0.0; all_rows],
            continuation: vec![0.0; all_rows],
            bonus: vec![0.0; starts * horizon],
            action: vec![0.0; starts * horizon * config.action_count],
        };
        let mut features = vec![0.0; inputs["imagined_feature"].len()];
        readback.read_regions(&mut [
            (decoded[0], &mut imagined.reward),
            (decoded[1], &mut imagined.value),
        ]);
        readback.read(
            &imagination,
            &mut [
                (world::IMAGINATION_FEATURE, &mut features),
                (world::IMAGINATION_ACTION, &mut imagined.action),
                (world::IMAGINATION_CONTINUATION, &mut imagined.continuation),
                (world::IMAGINATION_BONUS, &mut imagined.bonus),
            ],
        );
        let feature_error = errors(&features, &inputs["imagined_feature"]);
        let action_matches = imagined
            .action
            .chunks_exact(config.action_count)
            .zip(inputs["action_target"].chunks_exact(config.action_count))
            .all(|(sample, target)| {
                target.iter().filter(|&&v| v != 0.0).count() == 1
                    && sample
                        .iter()
                        .zip(target)
                        .all(|(&a, &t)| a == f32::from(t != 0.0))
            });
        let (action_target, weight, scale) = targets(&config, &imagined, &mut normalizer);
        let target_error = errors(&action_target, &inputs["action_target"]);
        let weight_error = errors(&weight, &inputs["imagined_weight"]);
        let comparison = serde_json::json!({"feature_error":feature_error, "action_matches":action_matches,
            "target_error":target_error, "weight_error":weight_error, "return_scale":scale,
            "normalizer":normalizer.state(), "rng_word_position":rng.get_word_pos().to_string()});
        save(&directory.join("reconstruction.json"), &comparison);
        export(
            &directory,
            "reconstructed",
            &Arrays::from([
                ("action_target".into(), action_target),
                ("imagined_weight".into(), weight),
                ("reward".into(), imagined.reward),
                ("value".into(), imagined.value),
                ("continuation".into(), imagined.continuation),
                ("bonus".into(), imagined.bonus.clone()),
            ]),
        );
        assert!(
            feature_error.0 <= 1e-6 && action_matches,
            "archived imagination mismatch: {comparison}"
        );
        assert!(
            target_error.1 <= 1e-5 && weight_error.1 <= 1e-5,
            "archived target mismatch: {comparison}"
        );
        probe.set_input("dynamics_bonus", &imagined.bonus);
        probe.set_input("dynamics_return_scale", &[scale]);
        let mut gradients = Vec::new();
        let mut losses = Vec::new();
        for (name, total, dynamics) in [("combined", 1.0, 0.0), ("dynamics", 0.0, 1.0)] {
            probe.set_input("probe_total", &[total]);
            probe.set_input("probe_dynamics", &[dynamics]);
            probe.zero_grad();
            probe.step();
            probe.wait();
            let loss = probe.read_loss();
            assert!(loss.is_finite());
            losses.push(loss);
            let values = read_gradients(&probe);
            export(&directory, name, &values);
            gradients.push(values);
            assert_eq!(probe.adam_step_count(), 0);
            assert_eq!(read_parameters(&probe), parameters);
        }
        let baseline = read_arrays(source, "gradient-layout.json", "total.f32");
        let agreement = gradient_agreement(&gradients[0], &baseline, &gradients[1]);
        let previous: serde_json::Value = json(&source.join("result.json"));
        let baseline_loss = previous["losses"][0].as_f64().unwrap();
        let scalar_error = (f64::from(losses[0]) - baseline_loss - f64::from(losses[1])).abs()
            / (1.0 + (baseline_loss + f64::from(losses[1])).abs());
        let result = serde_json::json!({"index":index, "reconstruction":comparison, "gradients":agreement,
            "losses":losses, "baseline_loss":baseline_loss, "scalar_error":scalar_error});
        save(&directory.join("result.json"), &result);
        assert!(scalar_error <= 1e-3);
        assert!(
            agreement["actor"]["passed"].as_bool().unwrap()
                && agreement["critic"]["passed"].as_bool().unwrap()
        );
        results.push(result);
    }
    assert_eq!(read_parameters(&imagination), imagination_parameters);
    for (name, expected) in &snapshot {
        let mut actual = vec![0.0; expected.len()];
        readback.read_regions(&mut [(probe.input_buffer(name).unwrap(), &mut actual)]);
        assert_eq!(&actual, expected, "frozen world input {name}");
    }
    export(&output, "parameters-after", &read_parameters(&probe));
    save(
        &output.join("result.json"),
        &serde_json::json!({"status":"complete", "batches":results,
        "seconds":started.elapsed().as_secs_f64(), "minimum_headroom":minimum_headroom,
        "frozen_world_tensors":snapshot.len(), "parameters":parameters.len(),
        "learner_updates":0, "optimizer_steps":probe.adam_step_count(), "atari_actions":0,
        "learner_step":metadata.learner_step, "environment_step":metadata.environment_step}),
    );
}

#[test]
fn gates_preserve_production_prefix_and_parameter_ownership() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_dynamics_scale = Some(0.5);
    let original = behavior::build_training_graph(
        &config,
        config.batch_size * config.batch_length * config.imagination_length,
        config.batch_size * (config.batch_length - 1),
    );
    let graph = gated_graph(&config);
    for (a, b) in original.nodes().iter().zip(graph.nodes()) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
    assert_eq!(&graph.outputs()[1..], original.outputs());
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert_eq!(plan.param_buffers.len(), 22);
    assert_eq!(plan.param_grad_pairs.len(), 22);
    assert!(
        plan.param_buffers
            .iter()
            .all(|(name, _)| name.starts_with("behavior."))
    );
    assert!(
        plan.input_buffers
            .iter()
            .any(|(name, _)| name.starts_with("dynamics_parameter_world."))
    );
    assert_eq!(
        plan.input_buffers
            .iter()
            .filter(|(name, _)| name.starts_with("probe_"))
            .count(),
        2
    );
}

#[test]
fn arrival_bonus_and_time_major_targets_preserve_both_starts() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 1;
    config.batch_length = 2;
    config.imagination_length = 2;
    config.lambda = 1.0;
    config.intrinsic_reward_scale = 1.0;
    let imagined = Imagined {
        reward: vec![100.0, 100.0, 1.0, 10.0, 2.0, 20.0],
        value: vec![0.0; 6],
        continuation: vec![1.0; 6],
        bonus: vec![4.0, 40.0, 8.0, 80.0],
        action: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    };
    let (target, weight, scale) =
        targets(&config, &imagined, &mut PercentileNormalizer::new(0.0, 1.0));
    assert_eq!(
        target,
        vec![
            15.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 10.0, 100.0, 0.0, 0.0
        ]
    );
    assert_eq!(weight, vec![1.0; 4]);
    assert_eq!(scale, 1.0);
}
