//! Opt-in loss decomposition at immutable world weights, without an optimizer.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use meganeura::{
    data::safetensors::SafeTensorsModel,
    graph::{NodeId, Op},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

type Inputs = BTreeMap<String, Vec<f32>>;

#[derive(Clone, Copy, Debug)]
enum Loss {
    ObservedRgb,
    Kl,
    ObservedEvents,
    FutureRgb,
    FutureEvents,
    Joint,
}

const MODES: [Loss; 6] = [
    Loss::ObservedRgb,
    Loss::Kl,
    Loss::ObservedEvents,
    Loss::FutureRgb,
    Loss::FutureEvents,
    Loss::Joint,
];

impl Loss {
    fn name(self) -> &'static str {
        match self {
            Self::ObservedRgb => "observed_rgb",
            Self::Kl => "kl",
            Self::ObservedEvents => "observed_events",
            Self::FutureRgb => "future_rgb",
            Self::FutureEvents => "future_events",
            Self::Joint => "joint",
        }
    }
}

fn selected_loss(graph: &mut Graph, config: &DreamerConfig, mode: Loss) -> NodeId {
    let output = graph.outputs().to_vec();
    let scale = config.loss_scales;
    let pair = match mode {
        Loss::ObservedRgb => return output[world::LOSS_RECONSTRUCTION],
        Loss::FutureRgb => return output[world::LOSS_WORLD_ROLLOUT],
        Loss::Joint => return output[world::LOSS_TOTAL],
        Loss::Kl => [
            (world::LOSS_DYNAMICS, scale.dynamics),
            (world::LOSS_REPRESENTATION, scale.representation),
        ],
        Loss::ObservedEvents => [
            (world::LOSS_REWARD, scale.reward),
            (world::LOSS_CONTINUATION, scale.continuation),
        ],
        Loss::FutureEvents => [
            (world::LOSS_IMAGE_FEEDBACK, scale.reward),
            (world::LOSS_IMAGE_FEEDBACK + 1, scale.continuation),
        ],
    };
    let first = graph.scale(output[pair[0].0], pair[0].1);
    let second = graph.scale(output[pair[1].0], pair[1].1);
    graph.add(first, second)
}

fn select(graph: &mut Graph, config: &DreamerConfig, mode: Loss) {
    let loss = selected_loss(graph, config, mode);
    let mut outputs = graph.outputs().to_vec();
    outputs[0] = loss;
    graph.set_outputs(outputs);
}

struct Batch {
    observations: Vec<f32>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    terminals: Vec<bool>,
}

fn inputs(config: &DreamerConfig, batch: &Batch, horizon: usize, seeds: [u64; 2]) -> Inputs {
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
    let mut posterior = StdRng::seed_from_u64(seeds[0]);
    let mut suffix = StdRng::seed_from_u64(seeds[1]);
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
            super::super::gaussian::noise(config, &mut posterior),
        );
        let mut reward = vec![0.; b * bins.len()];
        for (row, &value) in batch.rewards[range.clone()].iter().enumerate() {
            bins.encode(value, &mut reward[row * bins.len()..(row + 1) * bins.len()]);
        }
        result.insert(format!("reward_target_{t}"), reward);
        result.insert(
            format!("continuation_target_{t}"),
            batch.terminals[range]
                .iter()
                .map(|&terminal| {
                    if terminal {
                        0.
                    } else {
                        config.continuation_discount()
                    }
                })
                .collect(),
        );
    }
    for (t, weight) in rollout_weights(&batch.actions, b, horizon)
        .into_iter()
        .enumerate()
    {
        result.insert(
            format!("pixel_rollout_uniforms_{t}"),
            super::super::gaussian::noise(config, &mut suffix),
        );
        result.insert(format!("pixel_rollout_weight_{t}"), weight);
    }
    result
}

fn validate_feeds(graph: &Graph, feeds: &Inputs) {
    let raw = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some((name.clone(), n.ty.num_elements())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let expected = raw.iter().cloned().collect::<BTreeMap<_, _>>();
    assert_eq!(raw.len(), expected.len(), "duplicate input name");
    assert_eq!(expected.len(), feeds.len());
    for (name, length) in expected {
        assert_eq!(feeds[&name].len(), length, "{name}");
        assert!(feeds[&name].iter().all(|x| x.is_finite()), "{name}");
    }
}

fn hash_inputs(feeds: &Inputs) -> String {
    let mut hash = Sha256::new();
    for (name, values) in feeds {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((values.len() as u64).to_le_bytes());
        for v in values {
            hash.update(v.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn loss_components_reconstruct_values_and_derivatives_with_unequal_scales() {
    use meganeura::reference::{Feeds, evaluate};
    let mut c = DreamerConfig::tiny(18);
    c.loss_scales.reconstruction = 7.;
    c.loss_scales.dynamics = 0.6;
    c.loss_scales.representation = 0.2;
    c.loss_scales.reward = 2.;
    c.loss_scales.continuation = 3.;
    let mut g = Graph::new();
    let mut outputs = (0..36)
        .map(|i| g.parameter(&format!("p{i}"), &[1]))
        .collect::<Vec<_>>();
    let mut total = g.scalar(0.);
    for (index, scale) in [
        (1, 7.),
        (2, 0.6),
        (3, 0.2),
        (4, 2.),
        (5, 3.),
        (25, 7.),
        (34, 2.),
        (35, 3.),
    ] {
        let value = g.scale(outputs[index], scale);
        total = g.add(total, value);
    }
    outputs[0] = total;
    g.set_outputs(outputs);
    let mut feeds = Feeds::new();
    for i in 0..36 {
        feeds.set(&format!("p{i}"), &[0.1 + i as f32 / 17.]);
    }
    let mut values = Vec::new();
    let mut gradients = Vec::new();
    for mode in MODES {
        let mut candidate = g.deep_clone();
        select(&mut candidate, &c, mode);
        let backward = meganeura::autodiff::differentiate(&candidate);
        let result = evaluate(&backward, &feeds).unwrap();
        values.push(result[backward.outputs()[0] as usize].data[0]);
        gradients.push(
            backward.outputs()[candidate.outputs().len()..]
                .iter()
                .map(|&n| result[n as usize].data[0])
                .collect::<Vec<_>>(),
        );
    }
    let weights = [7., 1., 1., 7., 1.];
    let sum = values[..5]
        .iter()
        .zip(weights)
        .map(|(v, w)| v * w)
        .sum::<f64>();
    assert!((sum - values[5]).abs() < 1e-6);
    for i in 0..36 {
        let sum = gradients[..5]
            .iter()
            .zip(weights)
            .map(|(v, w)| v[i] * w)
            .sum::<f64>();
        assert!((sum - gradients[5][i]).abs() < 1e-6);
    }
}

fn tiny() -> (DreamerConfig, Batch) {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = super::super::ObservationKind::Rgb32;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.actor_critic_gradient = false;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    let batch = Batch {
        observations: vec![0.; 8 * c.observation_dim()],
        actions: vec![-1, -1, 0, 3, -1, 4, 1, 0],
        rewards: vec![0., 0., 1., 0., 0., 0., 0., 0.],
        terminals: vec![false, false, true, false, false, false, false, false],
    };
    (c, batch)
}

#[test]
fn loss_component_inputs_match_actual_suffix_graph_and_reset_targets() {
    let (c, batch) = tiny();
    let feeds = inputs(&c, &batch, 2, [7, 9]);
    let graph = world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionEventRollout);
    let original = graph.outputs()[1..].to_vec();
    for mode in MODES {
        let mut g = graph.deep_clone();
        select(&mut g, &c, mode);
        assert_eq!(g.outputs()[1..], original);
        validate_feeds(&g, &feeds);
    }
    assert_eq!(feeds["continuation_target_1"][0], 0.);
    assert_eq!(feeds["continuation_target_1"][1], c.continuation_discount());
    assert_eq!(feeds["pixel_rollout_weight_0"], [0., 1.]);
    assert_eq!(feeds["pixel_rollout_weight_1"], [0., 1.]);
    assert_eq!(feeds["reward_target_1"].iter().sum::<f32>(), 2.);
}

#[test]
fn loss_component_posterior_and_suffix_seeds_are_independent_and_repeatable() {
    let (c, batch) = tiny();
    let first = inputs(&c, &batch, 2, [7, 9]);
    assert_eq!(first, inputs(&c, &batch, 2, [7, 9]));
    for (seeds, prefix) in [([8, 9], "uniforms_"), ([7, 10], "pixel_rollout_uniforms_")] {
        let changed = inputs(&c, &batch, 2, seeds);
        let mut count = 0;
        for (name, values) in &first {
            if name.starts_with(prefix) {
                assert_ne!(values, &changed[name]);
                count += 1;
            } else {
                assert_eq!(values, &changed[name]);
            }
        }
        assert!(count > 0);
        assert_ne!(hash_inputs(&first), hash_inputs(&changed));
    }
}

fn load_batch(path: &str) -> Batch {
    let data = SafeTensorsModel::load(PathBuf::from(path)).unwrap();
    let actions = data.tensor_f32("action").unwrap();
    let terminals = data.tensor_f32("terminal").unwrap();
    assert!(
        actions
            .iter()
            .all(|&v| v.is_finite() && v == v.round() && (-1.0..18.0).contains(&v))
    );
    assert!(terminals.iter().all(|&v| v == 0. || v == 1.));
    Batch {
        observations: data.tensor_f32("observation").unwrap(),
        actions: actions.iter().map(|&v| v as i32).collect(),
        rewards: data.tensor_f32("reward").unwrap(),
        terminals: terminals.iter().map(|&v| v == 1.).collect(),
    }
}

#[test]
#[ignore = "requires separately declared guarded fixed-weight loss-gradient measurement"]
fn fixed_weight_loss_gradient_components() {
    let root = PathBuf::from(std::env::var("KINDLE_LOSS_GRADIENT_ROOT").unwrap());
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
    let c: DreamerConfig = serde_json::from_value(settings["config"].clone()).unwrap();
    validate_config(&c).unwrap();
    c.validate();
    assert!(c.rgb_spatial_fusion && !c.rgb_spatial_residual && !c.world_image_feedback);
    assert_eq!(c.batch_size, 8);
    assert_eq!(c.batch_length, 32);
    let checkpoint = PathBuf::from(settings["checkpoint"].as_str().unwrap());
    let anchor_hash = crate::vision::checkpoint_sha256(&checkpoint).unwrap();
    let anchor = SafeTensorsModel::load(checkpoint.clone()).unwrap();
    assert_eq!(anchor.tensor_info().len(), 240);
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
    for mode in MODES {
        let mut graph =
            world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionEventRollout);
        select(&mut graph, &c, mode);
        let mut parameters = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(parameters.len(), 80);
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
        assert_eq!(offset, 8_871_955);
        fs::write(
            out.join(format!("layout-{}.json", mode.name())),
            serde_json::to_vec_pretty(&layout).unwrap(),
        )
        .unwrap();
        let mut session = runtime::build_training_session(&graph, &gpu, &c);
        session.clear_optimizer();
        assert_eq!(session.param_names().len(), 80);
        for (name, _) in &parameters {
            session.set_parameter(name, &anchor.tensor_f32(name).unwrap());
        }
        let presence = parameters
            .iter()
            .map(|(name, _)| (name, session.has_param_grad(name)))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            out.join(format!("gradient-presence-{}.json", mode.name())),
            serde_json::to_vec_pretty(&presence).unwrap(),
        )
        .unwrap();
        session
            .save_checkpoint(&out.join(format!("before-{}.safetensors", mode.name())))
            .unwrap();
        for batch_spec in settings["batches"].as_array().unwrap() {
            let batch = load_batch(batch_spec["path"].as_str().unwrap());
            let seeds = settings["seeds"].as_array().unwrap();
            assert_eq!(seeds.len(), 4);
            let steps = if matches!(mode, Loss::Joint) { 5 } else { 4 };
            for step in 0..steps {
                let draw = step % 4;
                let feeds = inputs(
                    &c,
                    &batch,
                    16,
                    [
                        seeds[draw][0].as_u64().unwrap(),
                        seeds[draw][1].as_u64().unwrap(),
                    ],
                );
                validate_feeds(&graph, &feeds);
                let hash = hash_inputs(&feeds);
                for (name, values) in feeds {
                    session.set_input(&name, &values);
                }
                session.step();
                session.wait();
                assert_eq!(session.adam_step_count(), 0);
                let mut losses = Vec::new();
                for index in (0..9).chain([world::LOSS_WORLD_ROLLOUT]) {
                    let mut value = [0.];
                    session.read_output_by_index(index, &mut value);
                    assert!(value[0].is_finite());
                    losses.push(value[0]);
                }
                let weights = rollout_weights(&batch.actions, c.batch_size, 16);
                losses.push(weights.iter().flatten().sum::<f32>() / 128.);
                for index in world::LOSS_IMAGE_FEEDBACK..world::LOSS_IMAGE_FEEDBACK + 2 {
                    let mut value = [0.];
                    session.read_output_by_index(index, &mut value);
                    assert!(value[0].is_finite());
                    losses.push(value[0]);
                }
                let filename = format!(
                    "gradient-{}-{}-{step}.bin",
                    mode.name(),
                    batch_spec["name"].as_str().unwrap()
                );
                let mut file = BufWriter::new(File::create_new(out.join(&filename)).unwrap());
                for (name, _) in &parameters {
                    let mut values = vec![0.; session.param_size(name).unwrap()];
                    if session.has_param_grad(name) {
                        session.read_param_grad(name, &mut values);
                    }
                    assert!(values.iter().all(|v| v.is_finite()), "{name}");
                    let bytes = values
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
                let record = json!({"mode":mode.name(),"batch":batch_spec["name"],"step":step,"draw":draw,"repeat":step==4,"input_sha256":hash,"losses":losses,"gradient_file":filename,"elapsed_seconds":started.elapsed().as_secs_f64()});
                writeln!(log, "{record}").unwrap();
                log.flush().unwrap();
                count += 1;
                println!(
                    "loss gradients {count}/75: {} {} draw{draw}",
                    mode.name(),
                    batch_spec["name"]
                );
            }
            for (name, _) in &parameters {
                let mut actual = vec![0.; session.param_size(name).unwrap()];
                session.read_param(name, &mut actual);
                assert_eq!(
                    actual,
                    anchor.tensor_f32(name).unwrap(),
                    "weight changed: {name}"
                );
            }
        }
        session
            .save_checkpoint(&out.join(format!("after-{}.safetensors", mode.name())))
            .unwrap();
    }
    assert_eq!(count, 75);
    assert_eq!(
        crate::vision::checkpoint_sha256(&checkpoint).unwrap(),
        anchor_hash
    );
    fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","steps":75,"optimizer_steps":0,"logical_snapshots":12,"weights_per_snapshot":80,"anchor_checkpoint_unchanged":true,"wall_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
