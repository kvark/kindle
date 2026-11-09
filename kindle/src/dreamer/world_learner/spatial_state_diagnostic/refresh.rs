//! One privileged refresh, followed by an otherwise causal closed-loop forecast.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Prior,
    History,
    Latent,
    Both,
}

const REFRESH_MODES: [Mode; 4] = [Mode::Prior, Mode::History, Mode::Latent, Mode::Both];
const HISTORY: usize = 16;
const PULSE: usize = 4;
const FUTURE: usize = 32;

fn cutoff(history: usize, pulse: usize) -> usize {
    assert!(history >= 2 && pulse > 0);
    history + pulse - 1
}

fn refresh(mode: Mode, step: usize) -> (bool, bool) {
    (
        step + 1 == PULSE && matches!(mode, Mode::History | Mode::Both),
        step + 1 == PULSE && matches!(mode, Mode::Latent | Mode::Both),
    )
}

#[test]
fn one_refresh_has_independent_factors_and_no_future_arrival() {
    assert_eq!(cutoff(16, 4), 19);
    for mode in REFRESH_MODES {
        let factors = (0..FUTURE).map(|s| refresh(mode, s)).collect::<Vec<_>>();
        assert_eq!(
            factors.iter().filter(|&&(h, z)| h || z).count(),
            usize::from(mode != Mode::Prior)
        );
        assert_eq!(
            factors[3],
            (
                matches!(mode, Mode::History | Mode::Both),
                matches!(mode, Mode::Latent | Mode::Both)
            )
        );
        assert!(
            factors[..3]
                .iter()
                .chain(&factors[4..])
                .all(|&v| v == (false, false))
        );
    }
}

#[test]
fn refresh_history_stops_before_the_incoming_action() {
    let images = (0..48).map(|v| v as f32).collect::<Vec<_>>();
    let mut actions = [0; 48];
    actions[0] = -1;
    let n = cutoff(HISTORY, PULSE);
    let actual =
        super::super::super::spatial_history::after_cpu(&images[..n], &actions[..n], n - 1, 1, 1);
    assert_eq!(actual, [17., 18.]);
    assert!(!actual.contains(&images[n]));
}

struct State {
    h: Vec<f32>,
    z: Vec<f32>,
    rng: StdRng,
}

fn observed_states(
    learner: &mut WorldModelLearner,
    images: &[f32],
    actions: &[i32],
    seed: u64,
) -> (State, State) {
    let c = learner.config.clone();
    let b = c.batch_size;
    let size = c.network();
    let width = c.observation_dim();
    let n = cutoff(HISTORY, PULSE);
    // Truncated inputs ensure the teacher cannot consume action4's arrival.
    assert_eq!(images.len(), n * b * width);
    assert_eq!(actions.len(), n * b);
    let mut h = vec![0.; b * size.deter];
    let mut z = vec![0.; b * size.stoch * size.classes];
    let mut logits = z.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prefix = None;
    for t in 0..n {
        let (a, kh, kz, ka) = action_inputs(&c, &actions[t * b..(t + 1) * b]);
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
        learner
            .observe
            .set_input("observation", &images[t * b * width..(t + 1) * b * width]);
        learner.observe.step();
        learner.observe.wait();
        learner.observe.read_output_by_index(0, &mut h);
        learner.observe.read_output_by_index(1, &mut logits);
        z = sample_latents(&logits, b, size.stoch, size.classes, c.unimix, &mut rng);
        if t + 1 == HISTORY {
            prefix = Some(State {
                h: h.clone(),
                z: z.clone(),
                rng: rng.clone(),
            });
        }
    }
    (prefix.unwrap(), State { h, z, rng })
}

struct Trajectory {
    rgb: Vec<f32>,
    features: Vec<f32>,
    reward: Vec<f32>,
    positive: Vec<f32>,
    continuation: Vec<f32>,
}

fn forecast(
    learner: &mut WorldModelLearner,
    prefix: &State,
    teacher: &State,
    initial_history: &[f32],
    true_history: &[f32],
    future: &[i32],
    mode: Mode,
) -> Trajectory {
    let c = learner.config.clone();
    let b = c.batch_size;
    let size = c.network();
    let width = c.observation_dim();
    assert_eq!(future.len(), FUTURE * b);
    let mut h = prefix.h.clone();
    let mut z = prefix.z.clone();
    let mut logits = z.clone();
    let mut rng = prefix.rng.clone();
    let mut history = initial_history.to_vec();
    let mut result = Trajectory {
        rgb: Vec::new(),
        features: Vec::new(),
        reward: Vec::new(),
        positive: Vec::new(),
        continuation: Vec::new(),
    };
    for (step, actions) in future.chunks_exact(b).enumerate() {
        let (replace_history, replace_latent) = refresh(mode, step);
        if replace_history {
            history.clone_from_slice(true_history);
        }
        if replace_latent {
            h.clone_from(&teacher.h);
            z.clone_from(&teacher.z);
        }
        let (action, _, _, _) = action_inputs(&c, actions);
        learner.transition.set_input("deter", &h);
        learner.transition.set_input("stoch", &z);
        learner.transition.set_input("action", &action);
        learner.transition.step();
        learner.transition.wait();
        learner.transition.read_output_by_index(0, &mut h);
        learner.transition.read_output_by_index(1, &mut logits);
        z = sample_latents(&logits, b, size.stoch, size.classes, c.unimix, &mut rng);
        let read = decode(learner, &h, &z, &history, &action);
        history = super::super::super::spatial_history::shift_cpu(&history, &read.rgb, width);
        result.rgb.extend(read.rgb);
        result.features.extend(join_features(&h, &z, b, &c));
        result.reward.extend(read.reward);
        result.positive.extend(read.positive);
        result.continuation.extend(read.continuation);
    }
    result
}

#[test]
#[ignore = "requires a declared frozen checkpoint and guarded AMD execution"]
fn frozen_refresh_trajectories() {
    let r = PathBuf::from(std::env::var("KINDLE_REFRESH_ROOT").unwrap());
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
    assert_eq!(settings["history"], HISTORY);
    assert_eq!(settings["pulse"], PULSE);
    assert_eq!(settings["horizon"], FUTURE);
    let checkpoint = Path::new(settings["checkpoint"].as_str().unwrap());
    let metadata: Value =
        serde_json::from_slice(&fs::read(checkpoint.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["format"], "kindle-world-only-v20");
    assert_eq!(metadata["updates"], 16384);
    assert_eq!(metadata["prediction_length"], 0);
    let mut learner = WorldModelLearner::restore(checkpoint).unwrap();
    assert_eq!(
        learner.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(native_config_matches(learner.config(), &settings["config"]));
    assert!(
        learner.config.rgb_spatial_action
            && !learner.config.world_gaussian
            && !learner.config.world_image_feedback
    );
    assert_eq!(learner.parameter_names().len(), 80);
    let b = learner.config.batch_size;
    let width = learner.config.observation_dim();
    let fw = learner.config.feature_dim();
    assert_eq!((b, width, fw), (8, 12288, 640));
    let out = r.join("native-results");
    fs::create_dir(&out).unwrap();
    learner.save(&out.join("before")).unwrap();
    let mut log = BufWriter::new(File::create_new(out.join("queries.jsonl")).unwrap());
    let started = Instant::now();
    let mut queries = 0;
    let mut references = 0;
    for case in settings["batches"].as_array().unwrap() {
        let data = SafeTensorsModel::load(PathBuf::from(case["input"].as_str().unwrap())).unwrap();
        let images = data.tensor_f32("observation").unwrap();
        let raw = data.tensor_f32("action").unwrap();
        assert!(
            raw.iter()
                .all(|v| v.is_finite() && *v == v.round() && (-1.0..18.0).contains(v))
        );
        let actions = raw.iter().map(|&v| v as i32).collect::<Vec<_>>();
        assert_eq!(
            validate_history(&learner.config, &images, &actions).unwrap(),
            HISTORY + FUTURE
        );
        assert!(actions[HISTORY * b..].iter().all(|&a| a >= 0));
        let future = actions[HISTORY * b..]
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>();
        let n = cutoff(HISTORY, PULSE);
        let initial_history = super::super::super::spatial_history::after_cpu(
            &images[..HISTORY * b * width],
            &actions[..HISTORY * b],
            HISTORY - 1,
            b,
            width,
        );
        let true_history = super::super::super::spatial_history::after_cpu(
            &images[..n * b * width],
            &actions[..n * b],
            n - 1,
            b,
            width,
        );
        for (draw, seed) in settings["seeds"].as_array().unwrap().iter().enumerate() {
            let seed = seed.as_u64().unwrap();
            let canonical = learner
                .predict_with_features(
                    &images[..HISTORY * b * width],
                    &actions[..HISTORY * b],
                    &future,
                    seed,
                    true,
                )
                .unwrap();
            let (prefix, teacher) = observed_states(
                &mut learner,
                &images[..n * b * width],
                &actions[..n * b],
                seed,
            );
            exact(
                &join_features(&prefix.h, &prefix.z, b, &learner.config),
                &canonical.posterior_features,
                "prefix state",
            );
            if draw == 0 {
                let reference =
                    SafeTensorsModel::load(PathBuf::from(case["reference"].as_str().unwrap()))
                        .unwrap();
                for (key, values) in [
                    ("prior_rgb", endpoints(&canonical.observations, b * width)),
                    ("posterior_rgb", canonical.posterior.clone()),
                    ("encoder_features", canonical.encoded.clone()),
                    ("posterior_features", canonical.posterior_features.clone()),
                    (
                        "prior_features",
                        endpoints(&canonical.prior_features, b * fw),
                    ),
                    ("reward", flatten(&canonical.rewards)),
                    ("positive", flatten(&canonical.reward_positive_probability)),
                    ("continuation", flatten(&canonical.continuations)),
                ] {
                    exact(&values, &reference.tensor_f32(key).unwrap(), key);
                }
                references += 1;
            }
            let branches = REFRESH_MODES.map(|mode| {
                forecast(
                    &mut learner,
                    &prefix,
                    &teacher,
                    &initial_history,
                    &true_history,
                    &actions[HISTORY * b..],
                    mode,
                )
            });
            let stem = format!("{}-{draw}", case["name"].as_str().unwrap());
            let file = format!("{stem}.bin");
            let arrays = vec![
                (
                    "rgb",
                    vec![4, FUTURE, b, 3, 64, 64],
                    branches
                        .iter()
                        .flat_map(|v| v.rgb.iter().copied())
                        .collect(),
                ),
                (
                    "features",
                    vec![4, FUTURE, b, fw],
                    branches
                        .iter()
                        .flat_map(|v| v.features.iter().copied())
                        .collect(),
                ),
                (
                    "reward",
                    vec![4, FUTURE, b],
                    branches
                        .iter()
                        .flat_map(|v| v.reward.iter().copied())
                        .collect(),
                ),
                (
                    "positive",
                    vec![4, FUTURE, b],
                    branches
                        .iter()
                        .flat_map(|v| v.positive.iter().copied())
                        .collect(),
                ),
                (
                    "continuation",
                    vec![4, FUTURE, b],
                    branches
                        .iter()
                        .flat_map(|v| v.continuation.iter().copied())
                        .collect(),
                ),
            ];
            let layout = write_arrays(&out.join(&file), arrays);
            let item = json!({"batch":case["name"],"draw":draw,"seed":seed,"file":file,"layout":layout,"elapsed_seconds":started.elapsed().as_secs_f64()});
            fs::write(
                out.join(format!("{stem}.json")),
                serde_json::to_vec_pretty(&item).unwrap(),
            )
            .unwrap();
            writeln!(log, "{item}").unwrap();
            log.flush().unwrap();
            exact(
                &branches[0].rgb,
                &canonical.observations,
                "all-step production RGB",
            );
            exact(
                &branches[0].features,
                &canonical.prior_features,
                "all-step production features",
            );
            exact(
                &branches[0].reward,
                &flatten(&canonical.rewards),
                "all-step production reward",
            );
            exact(
                &branches[0].positive,
                &flatten(&canonical.reward_positive_probability),
                "all-step production positive",
            );
            exact(
                &branches[0].continuation,
                &flatten(&canonical.continuations),
                "all-step production continuation",
            );
            for branch in &branches[1..] {
                exact(
                    &branch.rgb[..(PULSE - 1) * b * width],
                    &branches[0].rgb[..(PULSE - 1) * b * width],
                    "no pre-refresh image change",
                );
                exact(
                    &branch.features[..(PULSE - 1) * b * fw],
                    &branches[0].features[..(PULSE - 1) * b * fw],
                    "no pre-refresh latent change",
                );
            }
            exact(
                &branches[1].features,
                &branches[0].features,
                "history-only latent invariance",
            );
            exact(
                &branches[1].reward,
                &branches[0].reward,
                "history-only reward invariance",
            );
            exact(
                &branches[1].positive,
                &branches[0].positive,
                "history-only positive invariance",
            );
            exact(
                &branches[1].continuation,
                &branches[0].continuation,
                "history-only continuation invariance",
            );
            assert_eq!(learner.updates, 16384);
            assert_eq!(learner.train.adam_step_count(), 16384);
            queries += 1;
            println!("refresh {queries}: {} draw{draw}", case["name"]);
        }
    }
    assert_eq!(queries, 16);
    assert_eq!(references, 4);
    learner.save(&out.join("after")).unwrap();
    fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","queries":queries,"references":references,"modes":4,"model_updates":0,"optimizer_age":learner.train.adam_step_count(),"updates":learner.updates,"elapsed_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
