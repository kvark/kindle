//! Frozen h/z/image-history readouts. No hybrid feeds either trajectory.

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
const MODES: [(bool, bool, bool); 8] = [
    (false, false, false),
    (true, false, false),
    (false, true, false),
    (true, true, false),
    (false, false, true),
    (true, false, true),
    (false, true, true),
    (true, true, true),
];

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

fn decode(
    learner: &mut WorldModelLearner,
    h: &[f32],
    z: &[f32],
    history: &[f32],
    action: &[f32],
) -> Readout {
    let batch = learner.config.batch_size;
    learner.decoder.set_input("deter", h);
    learner.decoder.set_input("stoch", z);
    learner.decoder.set_input("decoder_history", history);
    learner.decoder.set_input("decoder_action", action);
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

#[test]
fn history_modes_pair_identical_states_and_use_only_past_images() {
    for i in 0..4 {
        assert_eq!((MODES[i].0, MODES[i].1), (MODES[i + 4].0, MODES[i + 4].1));
        assert!(!MODES[i].2 && MODES[i + 4].2);
    }
    let images = [1., 10., 2., 20., 3., 30., 4., 40.];
    let actions = [-1, -1, 0, -1, 0, 0, 0, 0];
    assert_eq!(
        super::super::spatial_history::before_cpu(&images, &actions, 2, 2, 1),
        vec![1., 2., 0., 20.]
    );
    assert_eq!(
        super::super::spatial_history::after_cpu(&images, &actions, 1, 2, 1),
        vec![1., 2., 0., 20.]
    );
    let canonical = super::super::spatial_history::shift_cpu(&[1., 2., 10., 20.], &[5., 50.], 1);
    assert_eq!(canonical, vec![2., 5., 20., 50.]);
    assert_eq!(images, [1., 10., 2., 20., 3., 30., 4., 40.]);
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
        rgb: vec![0.; 8 * 5 * b * width],
        movie: if movie {
            vec![0.; 8 * 32 * b * width]
        } else {
            Vec::new()
        },
        reward: vec![0.; 8 * 32 * b],
        positive: vec![0.; 8 * 32 * b],
        continuation: vec![0.; 8 * 32 * b],
    };
    let mut h = vec![0.; b * size.deter];
    let mut z = vec![0.; b * sw];
    let mut logits = vec![0.; b * sw];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prior_rng = None;
    let mut history = super::super::spatial_history::after_cpu(observations, actions, 15, b, width);
    for time in 0..48 {
        let (a, kh, kz, ka) = action_inputs(&config, &actions[time * b..(time + 1) * b]);
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
            &observations[time * b * width..(time + 1) * b * width],
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
        if time == 15 {
            exact(&teacher, &prior.posterior_features, "observed prefix");
            prior_rng = Some(rng.clone());
        }
        if time < 16 {
            continue;
        }
        let step = time - 16;
        let current = &prior.prior_features[step * b * fw..(step + 1) * b * fw];
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
            current,
            "prior transition replay",
        );
        if step == 0 {
            close(&h, &replay_h, "step1 recurrent state");
        }
        result.teacher_features.extend_from_slice(&teacher);
        result.teacher_logits.extend_from_slice(&logits);
        result.prior_logits.extend(replay_logits);
        let teacher_history =
            super::super::spatial_history::before_cpu(observations, actions, time, b, width);
        if step == 0 {
            exact(&history, &teacher_history, "step1 image history");
        }
        let mut reads = Vec::new();
        for (mode, &replace) in MODES.iter().enumerate() {
            let (mh, mz) = blocks(current, &teacher, size.deter, sw, (replace.0, replace.1));
            let selected = if replace.2 {
                &teacher_history
            } else {
                &history
            };
            let read = decode(learner, &mh, &mz, selected, &a);
            if mode == 0 {
                exact(
                    &read.rgb,
                    &prior.observations[step * b * width..(step + 1) * b * width],
                    "canonical prior RGB",
                );
                exact(&read.reward, &prior.rewards[step], "canonical reward");
                exact(
                    &read.positive,
                    &prior.reward_positive_probability[step],
                    "canonical positive",
                );
                exact(
                    &read.continuation,
                    &prior.continuations[step],
                    "canonical continuation",
                );
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
            reads.push(read);
        }
        for i in 0..4 {
            exact(
                &reads[i].reward,
                &reads[i + 4].reward,
                "history-invariant reward",
            );
            exact(
                &reads[i].positive,
                &reads[i + 4].positive,
                "history-invariant positive",
            );
            exact(
                &reads[i].continuation,
                &reads[i + 4].continuation,
                "history-invariant continuation",
            );
            if step == 0 {
                exact(&reads[i].rgb, &reads[i + 4].rgb, "step1 history pair");
            }
        }
        if step == 0 {
            for i in [0, 2, 4, 6] {
                close(&reads[i].rgb, &reads[i + 1].rgb, "step1 h pair RGB");
                close(
                    &reads[i].reward,
                    &reads[i + 1].reward,
                    "step1 h pair reward",
                );
                close(
                    &reads[i].positive,
                    &reads[i + 1].positive,
                    "step1 h pair positive",
                );
                close(
                    &reads[i].continuation,
                    &reads[i + 1].continuation,
                    "step1 h pair continuation",
                );
            }
        }
        history = super::super::spatial_history::shift_cpu(
            &history,
            &prior.observations[step * b * width..(step + 1) * b * width],
            width,
        );
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
#[ignore = "requires separately declared guarded state/history diagnostic"]
fn frozen_state_and_history_readouts() {
    let r = PathBuf::from(std::env::var("KINDLE_STATE_HISTORY_ROOT").unwrap());
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
    assert_eq!(settings["models"].as_array().unwrap().len(), 2);
    assert_eq!(settings["modes"].as_array().unwrap().len(), 8);
    for model in settings["models"].as_array().unwrap() {
        let name = model["name"].as_str().unwrap();
        let expected_updates = model["expected_updates"].as_u64().unwrap();
        assert!(matches!(
            (name, expected_updates),
            ("local_h16", 8192) | ("local_h32", 4096)
        ));
        let checkpoint = Path::new(model["checkpoint"].as_str().unwrap());
        let metadata: Value =
            serde_json::from_slice(&fs::read(checkpoint.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["format"], "kindle-world-only-v18");
        let declared: DreamerConfig = serde_json::from_value(metadata["config"].clone()).unwrap();
        assert!(native_config_matches(&declared, &model["config"]));
        let mut learner = WorldModelLearner::restore(checkpoint).unwrap();
        assert_eq!(
            learner.gpu_device().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(
            learner.config.rgb_spatial_action
                && learner.config.rgb_spatial_fusion
                && learner.config.rgb_spatial_decoder
                && !learner.config.world_image_feedback
                && !learner.config.world_gaussian
        );
        assert_eq!(learner.updates, expected_updates);
        assert_eq!(u64::from(learner.train.adam_step_count()), expected_updates);
        assert_eq!(learner.parameter_names().len(), 80);
        assert!(native_config_matches(learner.config(), &model["config"]));
        learner.save(&out.join(format!("before-{name}"))).unwrap();
        let size = learner.config.network();
        let b = learner.config.batch_size;
        let width = learner.config.observation_dim();
        let fw = learner.config.feature_dim();
        assert_eq!((b, width, fw), (8, 12288, 2560));
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
                // The API requires a future batch; only its earlier posterior is checked.
                let full_teacher = learner
                    .predict_with_features(&observations, &actions, &vec![0; b], seed, true)
                    .unwrap();
                let last = &t.rgb[(7 * 5 + 4) * b * width..(7 * 5 + 5) * b * width];
                let same_features =
                    t.teacher_features[31 * b * fw..] == full_teacher.posterior_features[..];
                let same_rgb = last == full_teacher.posterior.as_slice();
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
                    ("rgb", vec![8, 5, b, 3, 64, 64], t.rgb),
                    ("reward", vec![8, 32, b], t.reward),
                    ("positive", vec![8, 32, b], t.positive),
                    ("continuation", vec![8, 32, b], t.continuation),
                    (
                        "canonical_teacher_posterior",
                        vec![b, 3, 64, 64],
                        full_teacher.posterior,
                    ),
                ];
                if movie {
                    arrays.push(("movie_rgb", vec![8, 32, b, 3, 64, 64], t.movie));
                }
                let layout = write_arrays(&out.join(&file), arrays);
                let item = json!({"model":name,"batch":case["name"],"draw":draw,"seed":seed,"file":file,"layout":layout,"baseline_reference_checked":checked,"canonical_teacher_final_features_exact":same_features,"canonical_teacher_final_rgb_exact":same_rgb,"elapsed_seconds":started.elapsed().as_secs_f64()});
                fs::write(
                    out.join(format!("{stem}.json")),
                    serde_json::to_vec_pretty(&item).unwrap(),
                )
                .unwrap();
                writeln!(log, "{item}").unwrap();
                log.flush().unwrap();
                assert!(
                    same_features && same_rgb,
                    "canonical teacher final control; report retained"
                );
                queries += 1;
                println!("state/history {queries} {name} {} draw{draw}", case["name"]);
            }
        }
        assert_eq!(learner.updates, expected_updates);
        assert_eq!(u64::from(learner.train.adam_step_count()), expected_updates);
        learner.save(&out.join(format!("after-{name}"))).unwrap();
    }
    assert_eq!(references, 8);
    assert_eq!(movies, 4);
    assert_eq!(queries, 48);
    fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","queries":queries,"baseline_references":references,"full_movies":movies,"modes":8,"model_updates":0,"elapsed_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
