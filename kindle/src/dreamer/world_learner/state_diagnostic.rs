//! Frozen readout interventions. Privileged future states never feed a forecast.

use super::*;
use meganeura::data::safetensors::SafeTensorsModel;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const HORIZONS: [usize; 5] = [1, 4, 8, 16, 32];
const MODES: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

fn native_config_matches(config: &DreamerConfig, declared: &Value) -> bool {
    let expected: DreamerConfig = serde_json::from_value(declared.clone()).unwrap();
    serde_json::to_value(config).unwrap() == serde_json::to_value(expected).unwrap()
}

#[test]
fn state_configuration_compares_f32_values_and_rejects_actual_changes() {
    let config = DreamerConfig::tiny(18);
    let mut declared = serde_json::to_value(&config).unwrap();
    declared["optimizer_beta2"] = json!(0.9990000128746032);
    declared["optimizer_epsilon"] = json!(9.999999682655224e-21);
    assert!(native_config_matches(&config, &declared));
    for (field, value) in [
        ("optimizer_beta2", json!(0.998)),
        ("optimizer_epsilon", json!(2e-20)),
        ("batch_size", json!(config.batch_size + 1)),
    ] {
        let mut changed = declared.clone();
        changed[field] = value;
        assert!(!native_config_matches(&config, &changed), "{field}");
    }
}

fn blocks(
    prior: &[f32],
    teacher: &[f32],
    deter: usize,
    stoch: usize,
    replace: (bool, bool),
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(prior.len(), teacher.len());
    assert!(prior.len().is_multiple_of(deter + stoch));
    let mut h = Vec::new();
    let mut z = Vec::new();
    for (p, q) in prior
        .chunks_exact(deter + stoch)
        .zip(teacher.chunks_exact(deter + stoch))
    {
        h.extend_from_slice(&if replace.0 { q } else { p }[..deter]);
        z.extend_from_slice(&if replace.1 { q } else { p }[deter..]);
    }
    (h, z)
}

#[test]
fn state_swaps_preserve_rows_and_choose_blocks_independently() {
    let p = [1., 2., 3., 4., 5., 6.];
    let q = [10., 20., 30., 40., 50., 60.];
    assert_eq!(
        blocks(&p, &q, 2, 1, (true, false)),
        (vec![10., 20., 40., 50.], vec![3., 6.])
    );
    assert_eq!(
        blocks(&p, &q, 2, 1, (false, true)),
        (vec![1., 2., 4., 5.], vec![30., 60.])
    );
    assert_eq!(
        blocks(&p, &q, 2, 1, (false, false)),
        (vec![1., 2., 4., 5.], vec![3., 6.])
    );
    assert_eq!(
        blocks(&p, &q, 2, 1, (true, true)),
        (vec![10., 20., 40., 50.], vec![30., 60.])
    );
}

fn exact(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{name}"
    );
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-5 + 1e-5 * b.abs()),
        "{name}"
    );
}

fn endpoints(values: &[f32], width: usize) -> Vec<f32> {
    HORIZONS
        .iter()
        .flat_map(|&h| values[(h - 1) * width..h * width].iter().copied())
        .collect()
}

fn flatten(values: &[Vec<f32>]) -> Vec<f32> {
    values.iter().flatten().copied().collect()
}

struct Readout {
    rgb: Vec<f32>,
    reward: Vec<f32>,
    positive: Vec<f32>,
    continuation: Vec<f32>,
}

fn decode(learner: &mut WorldModelLearner, h: &[f32], z: &[f32]) -> Readout {
    let batch = learner.config.batch_size;
    learner.decoder.set_input("deter", h);
    learner.decoder.set_input("stoch", z);
    learner.decoder.step();
    learner.decoder.wait();
    let mut rgb = vec![0.; batch * learner.config.observation_dim()];
    learner.decoder.read_output_by_index(0, &mut rgb);
    learner.heads.set_input("deter", h);
    learner.heads.set_input("stoch", z);
    learner.heads.step();
    learner.heads.wait();
    let bins = TwoHotBins::new(learner.config.value_bins);
    let mut logits = vec![0.; batch * bins.len()];
    let mut continuation = vec![0.; batch];
    learner.heads.read_output_by_index(0, &mut logits);
    learner.heads.read_output_by_index(1, &mut continuation);
    let mut scratch = vec![0.; bins.len()];
    let mut reward = Vec::new();
    let mut positive = Vec::new();
    for row in logits.chunks_exact(bins.len()) {
        reward.push(bins.decode_logits_with_scratch(row, &mut scratch));
        positive.push(
            (scratch[bins.len() / 2 + 1..].iter().sum::<f64>() / scratch.iter().sum::<f64>())
                as f32,
        );
    }
    Readout {
        rgb,
        reward,
        positive,
        continuation,
    }
}

struct Trace {
    teacher_features: Vec<f32>,
    prior_logits: Vec<f32>,
    teacher_logits: Vec<f32>,
    rgb: Vec<f32>,
    movie: Vec<f32>,
    reward: Vec<f32>,
    positive: Vec<f32>,
    continuation: Vec<f32>,
}

fn trace(
    learner: &mut WorldModelLearner,
    observations: &[f32],
    actions: &[i32],
    prior: &WorldPrediction,
    seed: u64,
    movie: bool,
) -> Trace {
    let config = learner.config.clone();
    let size = config.network();
    let b = config.batch_size;
    let width = config.observation_dim();
    let sw = size.stoch * size.classes;
    let fw = size.deter + sw;
    assert_eq!(
        validate_history(&config, observations, actions).unwrap(),
        48
    );
    let mut result = Trace {
        teacher_features: Vec::new(),
        prior_logits: Vec::new(),
        teacher_logits: Vec::new(),
        rgb: vec![0.; 4 * 5 * b * width],
        movie: if movie {
            vec![0.; 4 * 32 * b * width]
        } else {
            Vec::new()
        },
        reward: vec![0.; 4 * 32 * b],
        positive: vec![0.; 4 * 32 * b],
        continuation: vec![0.; 4 * 32 * b],
    };
    let mut h = vec![0.; b * size.deter];
    let mut z = vec![0.; b * sw];
    let mut logits = vec![0.; b * sw];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prior_rng = None;
    for t in 0..48 {
        let (a, kh, kz, ka) = action_inputs(&config, &actions[t * b..(t + 1) * b]);
        for (name, values) in [
            ("previous_deter", &h),
            ("previous_stoch", &z),
            ("previous_action", &a),
            ("keep_deter", &kh),
            ("keep_stoch", &kz),
            ("keep_action", &ka),
        ] {
            learner.observe.set_input(name, values);
        }
        learner.observe.set_input(
            "observation",
            &observations[t * b * width..(t + 1) * b * width],
        );
        learner.observe.step();
        learner.observe.wait();
        learner.observe.read_output_by_index(0, &mut h);
        learner.observe.read_output_by_index(1, &mut logits);
        z = sample_latents(
            &logits,
            b,
            size.stoch,
            size.classes,
            config.unimix,
            &mut rng,
        );
        let teacher = join_features(&h, &z, b, &config);
        if t == 15 {
            exact(&teacher, &prior.posterior_features, "observed prefix");
            prior_rng = Some(rng.clone());
        }
        if t < 16 {
            continue;
        }
        let step = t - 16;
        let p = &prior.prior_features[step * b * fw..(step + 1) * b * fw];
        let previous = if step == 0 {
            &prior.posterior_features[..]
        } else {
            &prior.prior_features[(step - 1) * b * fw..step * b * fw]
        };
        let (ph, pz) = blocks(previous, previous, size.deter, sw, (false, false));
        learner.transition.set_input("deter", &ph);
        learner.transition.set_input("stoch", &pz);
        learner.transition.set_input("action", &a);
        learner.transition.step();
        learner.transition.wait();
        let mut replay_h = vec![0.; b * size.deter];
        let mut replay_logits = vec![0.; b * sw];
        learner.transition.read_output_by_index(0, &mut replay_h);
        learner
            .transition
            .read_output_by_index(1, &mut replay_logits);
        let replay_z = sample_latents(
            &replay_logits,
            b,
            size.stoch,
            size.classes,
            config.unimix,
            prior_rng.as_mut().unwrap(),
        );
        exact(
            &join_features(&replay_h, &replay_z, b, &config),
            p,
            "prior transition replay",
        );
        if step == 0 {
            close(&h, &replay_h, "step1 recurrent state");
        }
        result.teacher_features.extend_from_slice(&teacher);
        result.teacher_logits.extend_from_slice(&logits);
        result.prior_logits.extend(replay_logits);
        let mut first_rgb = Vec::new();
        let mut first_heads = Vec::new();
        let mut z_rgb = Vec::new();
        let mut z_heads = Vec::new();
        for (mode, &replace) in MODES.iter().enumerate() {
            let (mh, mz) = blocks(p, &teacher, size.deter, sw, replace);
            let read = decode(learner, &mh, &mz);
            if mode == 0 {
                exact(
                    &read.rgb,
                    &prior.observations[step * b * width..(step + 1) * b * width],
                    "prior decoder",
                );
                exact(&read.reward, &prior.rewards[step], "prior reward");
                exact(
                    &read.positive,
                    &prior.reward_positive_probability[step],
                    "prior positive",
                );
                exact(
                    &read.continuation,
                    &prior.continuations[step],
                    "prior continuation",
                );
                if step == 0 {
                    first_rgb = read.rgb.clone();
                    first_heads = [
                        read.reward.clone(),
                        read.positive.clone(),
                        read.continuation.clone(),
                    ]
                    .concat();
                }
            }
            if step == 0 {
                let heads = [
                    read.reward.clone(),
                    read.positive.clone(),
                    read.continuation.clone(),
                ]
                .concat();
                if mode == 1 {
                    close(&read.rgb, &first_rgb, "step1 teacher_h pixels");
                    close(&heads, &first_heads, "step1 teacher_h heads");
                }
                if mode == 2 {
                    z_rgb = read.rgb.clone();
                    z_heads = heads.clone();
                }
                if mode == 3 {
                    close(&read.rgb, &z_rgb, "step1 teacher_z pixels");
                    close(&heads, &z_heads, "step1 teacher_z heads");
                }
            }
            if let Some(index) = HORIZONS.iter().position(|&v| v == step + 1) {
                let offset = (mode * 5 + index) * b * width;
                result.rgb[offset..offset + b * width].copy_from_slice(&read.rgb);
            }
            if movie {
                let offset = (mode * 32 + step) * b * width;
                result.movie[offset..offset + b * width].copy_from_slice(&read.rgb);
            }
            let offset = (mode * 32 + step) * b;
            result.reward[offset..offset + b].copy_from_slice(&read.reward);
            result.positive[offset..offset + b].copy_from_slice(&read.positive);
            result.continuation[offset..offset + b].copy_from_slice(&read.continuation);
        }
    }
    result
}

fn write_arrays(path: &Path, arrays: Vec<(&str, Vec<usize>, Vec<f32>)>) -> Vec<Value> {
    let mut file = BufWriter::new(File::create_new(path).unwrap());
    let mut offset = 0;
    let mut layout = Vec::new();
    for (name, shape, values) in arrays {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        assert!(values.iter().all(|v| v.is_finite()), "{name}");
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        layout.push(json!({"name":name,"shape":shape,"offset_f32":offset,"length":values.len()}));
        offset += values.len();
    }
    file.flush().unwrap();
    layout
}

#[test]
#[ignore = "requires guarded frozen-state substitution protocol and cached development inputs"]
fn frozen_state_substitutions() {
    let r = PathBuf::from(std::env::var("KINDLE_STATE_DIAG_ROOT").unwrap());
    for name in ["data-review.json", "execution-pins.json"] {
        let p: Value = serde_json::from_slice(&fs::read(r.join(name)).unwrap()).unwrap();
        for (path, digest) in p["pins"].as_object().unwrap() {
            assert_eq!(
                crate::vision::checkpoint_sha256(Path::new(path)).unwrap(),
                digest.as_str().unwrap(),
                "{path}"
            );
        }
    }
    let settings: Value =
        serde_json::from_slice(&fs::read(r.join("settings.json")).unwrap()).unwrap();
    let out = r.join("native-results");
    fs::create_dir(&out).unwrap();
    let mut log = BufWriter::new(File::create_new(out.join("queries.jsonl")).unwrap());
    let started = Instant::now();
    let mut queries = 0;
    let mut references = 0;
    let mut movies = 0;
    for model in settings["models"].as_array().unwrap() {
        let name = model["name"].as_str().unwrap();
        let metadata: Value = serde_json::from_slice(
            &fs::read(Path::new(model["checkpoint"].as_str().unwrap()).join("metadata.json"))
                .unwrap(),
        )
        .unwrap();
        let declared: DreamerConfig = serde_json::from_value(metadata["config"].clone()).unwrap();
        assert!(native_config_matches(&declared, &model["config"]));
        let mut learner =
            WorldModelLearner::restore(Path::new(model["checkpoint"].as_str().unwrap())).unwrap();
        assert_eq!(
            learner.gpu_device().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert_eq!(learner.updates, 8192);
        assert_eq!(learner.train.adam_step_count(), 8192);
        assert_eq!(learner.parameter_names().len(), 68);
        assert!(native_config_matches(learner.config(), &model["config"]));
        learner.save(&out.join(format!("before-{name}"))).unwrap();
        let size = learner.config.network();
        let b = learner.config.batch_size;
        let width = learner.config.observation_dim();
        let fw = learner.config.feature_dim();
        assert_eq!(b, 8);
        for case in settings["batches"].as_array().unwrap() {
            let data =
                SafeTensorsModel::load(PathBuf::from(case["input"].as_str().unwrap())).unwrap();
            let observations = data.tensor_f32("observation").unwrap();
            let raw_actions = data.tensor_f32("action").unwrap();
            assert!(
                raw_actions
                    .iter()
                    .all(|v| v.is_finite() && *v == v.round() && (-1.0..18.0).contains(v))
            );
            let actions = raw_actions.iter().map(|&v| v as i32).collect::<Vec<_>>();
            assert!(actions[16 * b..].iter().all(|&v| v >= 0));
            let future = actions[16 * b..]
                .iter()
                .map(|&v| v as usize)
                .collect::<Vec<_>>();
            for (draw, seed) in settings["seeds"].as_array().unwrap().iter().enumerate() {
                let seed = seed.as_u64().unwrap();
                let prediction = learner
                    .predict_with_features(
                        &observations[..16 * b * width],
                        &actions[..16 * b],
                        &future,
                        seed,
                        true,
                    )
                    .unwrap();
                let mut checked = false;
                if draw == 0
                    && let Some(path) = case["references"][name].as_str()
                {
                    let expected = SafeTensorsModel::load(PathBuf::from(path)).unwrap();
                    for (key, values) in [
                        ("prior_rgb", endpoints(&prediction.observations, b * width)),
                        ("posterior_rgb", prediction.posterior.clone()),
                        ("encoder_features", prediction.encoded.clone()),
                        ("posterior_features", prediction.posterior_features.clone()),
                        (
                            "prior_features",
                            endpoints(&prediction.prior_features, b * fw),
                        ),
                        ("reward", flatten(&prediction.rewards)),
                        ("positive", flatten(&prediction.reward_positive_probability)),
                        ("continuation", flatten(&prediction.continuations)),
                    ] {
                        exact(&values, &expected.tensor_f32(key).unwrap(), key);
                    }
                    checked = true;
                    references += 1;
                }
                let movie = draw == 0 && case["keep_movie"].as_bool().unwrap();
                let t = trace(
                    &mut learner,
                    &observations,
                    &actions,
                    &prediction,
                    seed,
                    movie,
                );
                movies += usize::from(movie);
                let stem = format!("{name}-{}-{draw}", case["name"].as_str().unwrap());
                let file = format!("{stem}.bin");
                let mut arrays = vec![
                    ("prior_features", vec![32, b, fw], prediction.prior_features),
                    ("teacher_features", vec![32, b, fw], t.teacher_features),
                    (
                        "prior_logits",
                        vec![32, b, size.stoch, size.classes],
                        t.prior_logits,
                    ),
                    (
                        "teacher_logits",
                        vec![32, b, size.stoch, size.classes],
                        t.teacher_logits,
                    ),
                    ("rgb", vec![4, 5, b, 3, 64, 64], t.rgb),
                    ("reward", vec![4, 32, b], t.reward),
                    ("positive", vec![4, 32, b], t.positive),
                    ("continuation", vec![4, 32, b], t.continuation),
                ];
                if movie {
                    arrays.push(("movie_rgb", vec![4, 32, b, 3, 64, 64], t.movie));
                }
                let layout = write_arrays(&out.join(&file), arrays);
                let item = json!({"model":name,"batch":case["name"],"draw":draw,"seed":seed,"file":file,"layout":layout,"baseline_reference_checked":checked,"elapsed_seconds":started.elapsed().as_secs_f64()});
                fs::write(
                    out.join(format!("{stem}.json")),
                    serde_json::to_vec_pretty(&item).unwrap(),
                )
                .unwrap();
                writeln!(log, "{item}").unwrap();
                log.flush().unwrap();
                queries += 1;
                println!(
                    "state substitutions {queries} {name} {} draw{draw}",
                    case["name"]
                );
            }
        }
        assert_eq!(learner.updates, 8192);
        assert_eq!(learner.train.adam_step_count(), 8192);
        learner.save(&out.join(format!("after-{name}"))).unwrap();
    }
    assert_eq!(references, 8);
    assert_eq!(movies, 4);
    assert_eq!(
        queries,
        settings["models"].as_array().unwrap().len()
            * settings["batches"].as_array().unwrap().len()
            * settings["seeds"].as_array().unwrap().len()
    );
    fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","queries":queries,"baseline_references":references,"full_movies":movies,"model_updates":0,"elapsed_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
