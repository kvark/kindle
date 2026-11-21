//! Opt-in, fixed-weight measurement. No optimizer is configured here.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use meganeura::{data::safetensors::SafeTensorsModel, graph::Op};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

type Inputs = BTreeMap<String, Vec<f32>>;

struct Batch {
    observations: Vec<f32>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    terminals: Vec<bool>,
}

// Match learn's input construction and draw order, but reset the two streams
// independently so that only predictive sampling varies between replicas.
fn inputs(
    config: &DreamerConfig,
    batch: &Batch,
    objective: PixelLoss,
    posterior_seed: u64,
    predictive_seed: u64,
) -> Inputs {
    assert_eq!(
        validate_history(config, &batch.observations, &batch.actions).unwrap(),
        config.batch_length
    );
    assert_eq!(batch.rewards.len(), batch.actions.len());
    assert_eq!(batch.terminals.len(), batch.actions.len());
    assert!(batch.rewards.iter().all(|v| v.is_finite()));
    validate_targets(
        &batch.actions,
        &batch.rewards,
        &batch.terminals,
        config.batch_size,
    )
    .unwrap();
    let b = config.batch_size;
    let size = config.network();
    let bins = TwoHotBins::new(config.value_bins);
    let mut posterior = StdRng::seed_from_u64(posterior_seed);
    let mut predictive = StdRng::seed_from_u64(predictive_seed);
    let mut result = Inputs::new();
    result.insert("initial_deter".into(), vec![0.; b * size.deter]);
    result.insert(
        "initial_stoch".into(),
        vec![0.; b * size.stoch * size.classes],
    );
    for t in 0..config.batch_length {
        let range = t * b..(t + 1) * b;
        let (action, deter, stoch, keep) = action_inputs(config, &batch.actions[range.clone()]);
        for (name, values) in [
            ("previous_action", action),
            ("keep_deter", deter),
            ("keep_stoch", stoch),
            ("keep_action", keep),
        ] {
            result.insert(format!("{name}_{t}"), values);
        }
        let width = config.observation_dim();
        result.insert(
            format!("observation_{t}"),
            batch.observations[t * b * width..(t + 1) * b * width].to_vec(),
        );
        result.insert(
            format!("uniforms_{t}"),
            (0..b * size.stoch * size.classes)
                .map(|_| posterior.random_range(1e-6..1.0 - 1e-6))
                .collect(),
        );
        if objective.predictive() {
            for (name, width) in [
                ("predictive_uniforms", size.stoch * size.classes),
                ("predictive_second_uniforms", size.stoch * size.classes),
                ("predictive_reward_uniforms0", bins.len()),
                ("predictive_reward_uniforms1", bins.len()),
                ("predictive_continuation_uniforms0", 2),
                ("predictive_continuation_uniforms1", 2),
            ] {
                result.insert(
                    format!("{name}_{t}"),
                    (0..b * width)
                        .map(|_| predictive.random_range(1e-6..1.0 - 1e-6))
                        .collect(),
                );
            }
        }
        let mut reward = vec![0.; b * bins.len()];
        for (row, &value) in batch.rewards[range.clone()].iter().enumerate() {
            bins.encode(value, &mut reward[row * bins.len()..(row + 1) * bins.len()]);
        }
        if objective.predictive() {
            let sampled = reward
                .chunks_exact(bins.len())
                .map(|row| {
                    let value = bins.values()[sample_probabilities(row, &mut predictive)];
                    value.signum() * value.abs().ln_1p()
                })
                .collect();
            result.insert(format!("predictive_reward_sample_{t}"), sampled);
        }
        result.insert(format!("reward_target_{t}"), reward);
        let continuation = batch.terminals[range]
            .iter()
            .map(|&terminal| {
                if terminal {
                    0.
                } else {
                    config.continuation_discount()
                }
            })
            .collect::<Vec<_>>();
        if objective.predictive() {
            result.insert(
                format!("predictive_continuation_sample_{t}"),
                continuation
                    .iter()
                    .map(|&p| f32::from(predictive.random::<f32>() < p))
                    .collect(),
            );
        }
        result.insert(format!("continuation_target_{t}"), continuation);
    }
    result
}

fn input_hash(feeds: &Inputs, auxiliary: bool) -> String {
    let mut hash = Sha256::new();
    for (name, values) in feeds {
        if name.starts_with("predictive_") != auxiliary {
            continue;
        }
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn validate_feeds(graph: &Graph, feeds: &Inputs) {
    let expected = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name, .. } => Some((name.clone(), n.ty.shape.iter().product::<usize>())),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(expected.len(), feeds.len());
    for (name, len) in expected {
        assert_eq!(feeds[&name].len(), len, "{name}");
        assert!(feeds[&name].iter().all(|v| v.is_finite()), "{name}");
    }
}

fn tiny_batch() -> (DreamerConfig, Batch) {
    let mut config = DreamerConfig::tiny(18);
    config.observation_kind = super::super::ObservationKind::Rgb64;
    config.batch_size = 2;
    config.batch_length = 3;
    config.world_backprop_length = 3;
    config.actor_critic_gradient = false;
    config.disagreement_bonus = false;
    config.loss_scales.policy = 0.;
    config.loss_scales.value = 0.;
    config.loss_scales.replay_value = 0.;
    let batch = Batch {
        observations: vec![0.; 6 * config.observation_dim()],
        actions: vec![-1, -1, 0, 3, -1, 4],
        rewards: vec![0., 0., 1., 0., 0., 0.],
        terminals: vec![false, false, true, false, false, false],
    };
    (config, batch)
}

#[test]
fn diagnostic_separates_posterior_and_predictive_streams() {
    let (config, batch) = tiny_batch();
    let observed = inputs(&config, &batch, PixelLoss::MotionCrossEntropy, 7, 9);
    let st = inputs(&config, &batch, PixelLoss::PredictiveMotionEnergy, 7, 9);
    let sf = inputs(
        &config,
        &batch,
        PixelLoss::PredictiveMotionEnergyScoreFunction,
        7,
        9,
    );
    let other = inputs(&config, &batch, PixelLoss::PredictiveMotionEnergy, 7, 10);
    assert_eq!(st, sf);
    for feeds in [&st, &sf, &other] {
        assert_eq!(input_hash(&observed, false), input_hash(feeds, false));
    }
    assert_ne!(input_hash(&st, true), input_hash(&other, true));
    assert_eq!(observed["continuation_target_1"][0], 0.);
    assert_eq!(observed["keep_action_2"][0], 0.);
    assert_eq!(observed["reward_target_1"].iter().sum::<f32>(), 2.);
}

#[test]
fn diagnostic_inputs_match_all_three_graphs() {
    let (config, batch) = tiny_batch();
    for objective in [
        PixelLoss::MotionCrossEntropy,
        PixelLoss::PredictiveMotionEnergy,
        PixelLoss::PredictiveMotionEnergyScoreFunction,
    ] {
        let graph = world::build_world_only_pixel_training_graph(&config, 0, objective);
        validate_feeds(&graph, &inputs(&config, &batch, objective, 7, 9));
    }
}

#[test]
#[ignore = "requires declared fixed-weight data and guarded GPU; exports raw gradients only"]
fn fixed_weight_gradient_noise() {
    let root = PathBuf::from(std::env::var("KINDLE_GRADIENT_DIAG_ROOT").unwrap());
    for name in ["data-review.json", "execution-pins.json"] {
        let review: Value = serde_json::from_slice(&fs::read(root.join(name)).unwrap()).unwrap();
        for (path, expected) in review["pins"].as_object().unwrap() {
            assert_eq!(
                crate::vision::checkpoint_sha256(Path::new(path)).unwrap(),
                expected.as_str().unwrap(),
                "{path}"
            );
        }
    }
    let settings: Value =
        serde_json::from_slice(&fs::read(root.join("settings.json")).unwrap()).unwrap();
    let config: DreamerConfig = serde_json::from_value(settings["config"].clone()).unwrap();
    validate_config(&config).unwrap();
    let anchor =
        SafeTensorsModel::load(PathBuf::from(settings["checkpoint"].as_str().unwrap())).unwrap();
    assert_eq!(anchor.tensor_info().len(), 204);
    let out = root.join("native-results");
    fs::create_dir(&out).unwrap();
    let mut log = BufWriter::new(File::create_new(out.join("steps.jsonl")).unwrap());
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let started = Instant::now();
    let mut count = 0;
    for (mode, objective) in [
        ("observed", PixelLoss::MotionCrossEntropy),
        ("st", PixelLoss::PredictiveMotionEnergy),
        ("sf", PixelLoss::PredictiveMotionEnergyScoreFunction),
    ] {
        let graph = world::build_world_only_pixel_training_graph(&config, 0, objective);
        let mut parameters = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(parameters.len(), 68);
        let mut offset = 0;
        let layout = parameters
            .iter()
            .map(|(name, shape)| {
                let len = shape.iter().product::<usize>();
                let item = json!({"name":name,"shape":shape,"length":len,"offset_f32":offset});
                offset += len;
                item
            })
            .collect::<Vec<_>>();
        assert_eq!(offset, 8_852_675);
        fs::write(
            out.join(format!("layout-{mode}.json")),
            serde_json::to_vec_pretty(&layout).unwrap(),
        )
        .unwrap();
        let mut session = runtime::build_training_session(&graph, &gpu, &config);
        assert_eq!(session.param_names().len(), 68);
        for (name, _) in &parameters {
            assert!(session.has_param_grad(name), "{name}");
            session.set_parameter(name, &anchor.tensor_f32(name).unwrap());
        }
        assert_eq!(session.adam_step_count(), 0);
        session
            .save_checkpoint(&out.join(format!("before-{mode}.safetensors")))
            .unwrap();
        for batch_spec in settings["batches"].as_array().unwrap() {
            let data = SafeTensorsModel::load(PathBuf::from(batch_spec["path"].as_str().unwrap()))
                .unwrap();
            let raw_actions = data.tensor_f32("action").unwrap();
            let raw_terminal = data.tensor_f32("terminal").unwrap();
            assert!(
                raw_actions
                    .iter()
                    .all(|&v| v.is_finite() && v == v.round() && (-1.0..18.0).contains(&v))
            );
            assert!(raw_terminal.iter().all(|&v| v == 0. || v == 1.));
            let batch = Batch {
                observations: data.tensor_f32("observation").unwrap(),
                actions: raw_actions.iter().map(|&v| v as i32).collect(),
                rewards: data.tensor_f32("reward").unwrap(),
                terminals: raw_terminal.iter().map(|&v| v == 1.).collect(),
            };
            let seeds = settings["predictive_seeds"].as_array().unwrap();
            assert_eq!(seeds.len(), 16);
            let replicas = if objective.predictive() { 16 } else { 1 };
            for step in 0..=replicas {
                let replica = if step == replicas { 0 } else { step };
                let feeds = inputs(
                    &config,
                    &batch,
                    objective,
                    settings["posterior_seed"].as_u64().unwrap(),
                    seeds[replica].as_u64().unwrap(),
                );
                validate_feeds(&graph, &feeds);
                let common_hash = input_hash(&feeds, false);
                let auxiliary_hash = input_hash(&feeds, true);
                for (name, values) in feeds {
                    session.set_input(&name, &values);
                }
                session.step();
                session.wait();
                assert_eq!(session.adam_step_count(), 0);
                let indices = (0..9).chain(if objective.predictive() {
                    world::LOSS_PREDICTIVE_WORLD..world::LOSS_PREDICTIVE_WORLD + 4
                } else {
                    0..0
                });
                let mut losses = Vec::new();
                for index in indices {
                    let mut value = [0.];
                    session.read_output_by_index(index, &mut value);
                    assert!(value[0].is_finite());
                    losses.push(value[0]);
                }
                let filename = format!(
                    "gradient-{mode}-{}-{step}.bin",
                    batch_spec["name"].as_str().unwrap()
                );
                let mut file = BufWriter::new(File::create_new(out.join(&filename)).unwrap());
                for (name, _) in &parameters {
                    let mut gradient = vec![0.; session.param_size(name).unwrap()];
                    session.read_param_grad(name, &mut gradient);
                    assert!(gradient.iter().all(|v| v.is_finite()), "{name}");
                    let bytes = gradient
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    file.write_all(&bytes).unwrap();
                }
                file.flush().unwrap();
                assert_eq!(
                    fs::metadata(out.join(&filename)).unwrap().len(),
                    (offset * 4) as u64
                );
                let item = json!({"mode":mode,"batch":batch_spec["name"],"step":step,"replica":replica,"repeat":step == replicas,"common_input_sha256":common_hash,"auxiliary_input_sha256":auxiliary_hash,"losses":losses,"gradient_file":filename,"elapsed_seconds":started.elapsed().as_secs_f64()});
                writeln!(log, "{item}").unwrap();
                log.flush().unwrap();
                count += 1;
                println!(
                    "gradient diagnostic {count}/108 {mode} {} {step}",
                    batch_spec["name"]
                );
            }
            for (name, _) in &parameters {
                let expected = anchor.tensor_f32(name).unwrap();
                let mut actual = vec![0.; expected.len()];
                session.read_param(name, &mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "weight changed: {name}"
                );
            }
        }
        session
            .save_checkpoint(&out.join(format!("after-{mode}.safetensors")))
            .unwrap();
    }
    assert_eq!(count, 108);
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&json!({"status":"complete","steps":count,"optimizer_steps":0,"elapsed_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
