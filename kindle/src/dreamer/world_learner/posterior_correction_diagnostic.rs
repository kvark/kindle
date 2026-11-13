//! Recompute an image-informed posterior at cached forecast h; never feed it back.
use super::*;
use crate::dreamer::networks::Representation;
use meganeura::{data::safetensors::SafeTensorsModel, graph::Op};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Instant,
};

fn posterior_graph(config: &DreamerConfig) -> Graph {
    let mut graph = Graph::new();
    let representation = Representation::new(&mut graph, config);
    let h = graph.input("deter", &[config.batch_size, config.network().deter]);
    let image = graph.input("observation", &config.observation_shape(config.batch_size));
    let encoded = representation
        .encoder
        .forward(&mut graph, image, config.batch_size);
    let logits = representation
        .posterior
        .forward(&mut graph, h, encoded, config.batch_size);
    graph.set_outputs(vec![logits]);
    graph
}

fn state_blocks(values: &[f32], deter: usize, stoch: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(values.len().is_multiple_of(deter + stoch));
    let rows = values.chunks_exact(deter + stoch);
    (
        rows.clone()
            .flat_map(|r| r[..deter].iter().copied())
            .collect(),
        rows.flat_map(|r| r[deter..].iter().copied()).collect(),
    )
}

#[test]
fn cached_state_blocks_preserve_time_and_batch_rows() {
    assert_eq!(
        state_blocks(&[1., 2., 3., 4., 5., 6.], 2, 1),
        (vec![1., 2., 4., 5.], vec![3., 6.])
    );
}

#[test]
fn correction_graph_has_only_visual_parameters_and_no_recurrence() {
    let mut config = DreamerConfig::tiny(18);
    config.observation_kind = super::super::config::ObservationKind::Rgb64;
    let graph = posterior_graph(&config);
    let names = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Parameter { name } => Some(name),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!names.is_empty());
    assert!(names.iter().all(|n| n.starts_with("world.representation.")));
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("world.representation.encoder."))
    );
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("world.representation.posterior."))
    );
    assert_eq!(graph.outputs().len(), 1);
}

fn exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    assert!(
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{label}"
    );
}

fn close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    assert!(
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-5 + 1e-5 * y.abs()),
        "{label}"
    );
}

fn cached(record: &Value, name: &str) -> Vec<f32> {
    let layout = record["layout"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["name"] == name)
        .unwrap();
    let length = layout["length"].as_u64().unwrap() as usize;
    let mut file = File::open(record["path"].as_str().unwrap()).unwrap();
    file.seek(SeekFrom::Start(4 * layout["offset_f32"].as_u64().unwrap()))
        .unwrap();
    let mut bytes = vec![0; 4 * length];
    file.read_exact(&mut bytes).unwrap();
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values.iter().all(|x| x.is_finite()));
    values
}

fn initialize(session: &mut Session, anchor: &SafeTensorsModel) {
    let names = session
        .param_names()
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>();
    for name in names {
        assert!(!session.has_param_grad(&name));
        session.set_parameter(&name, &anchor.tensor_f32(&name).unwrap());
    }
    assert_eq!(session.adam_step_count(), 0);
}

fn invariant(session: &mut Session, anchor: &SafeTensorsModel) -> usize {
    let names = session
        .param_names()
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>();
    for name in &names {
        let expected = anchor.tensor_f32(name).unwrap();
        let mut actual = vec![0.; expected.len()];
        session.read_param(name, &mut actual);
        exact(&actual, &expected, name);
        assert!(!session.has_param_grad(name));
    }
    assert_eq!(session.adam_step_count(), 0);
    names.len()
}

fn write_arrays(path: &Path, arrays: Vec<(&str, Vec<usize>, Vec<f32>)>) -> Vec<Value> {
    let mut file = File::create_new(path).unwrap();
    let mut offset = 0;
    let mut layout = Vec::new();
    for (name, shape, values) in arrays {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        assert!(values.iter().all(|v| v.is_finite()));
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        layout.push(json!({"name":name,"shape":shape,"offset_f32":offset,"length":values.len()}));
        offset += values.len();
    }
    layout
}

#[test]
#[ignore = "requires declared guarded cached-state posterior-correction diagnostic"]
fn correct_posterior_at_cached_forecast_state() {
    let root = PathBuf::from(std::env::var("KINDLE_POSTERIOR_CORRECTION_ROOT").unwrap());
    let declaration: Value =
        serde_json::from_slice(&fs::read(root.join("declaration.json")).unwrap()).unwrap();
    for (path, digest) in declaration["pins"].as_object().unwrap() {
        assert_eq!(
            crate::vision::checkpoint_sha256(Path::new(path)).unwrap(),
            digest.as_str().unwrap(),
            "{path}"
        );
    }
    let settings: Value =
        serde_json::from_slice(&fs::read(root.join("settings.json")).unwrap()).unwrap();
    let config: DreamerConfig = serde_json::from_value(settings["config"].clone()).unwrap();
    let size = config.network();
    let b = config.batch_size;
    let width = config.observation_dim();
    let sw = size.stoch * size.classes;
    assert_eq!(b, 8);
    assert_eq!(config.feature_dim(), size.deter + sw);
    let source = Path::new(settings["checkpoint"].as_str().unwrap()).join("world.safetensors");
    let anchor = SafeTensorsModel::load(source.clone()).unwrap();
    assert_eq!(anchor.tensor_info().len(), 204);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut posterior =
        runtime::build_session(&posterior_graph(&config), &gpu, Mode::Inference, false);
    let mut decoder = runtime::build_session(
        &world::build_observation_prediction_graph(&config, b),
        &gpu,
        Mode::Inference,
        false,
    );
    initialize(&mut posterior, &anchor);
    initialize(&mut decoder, &anchor);
    let parameters = invariant(&mut posterior, &anchor) + invariant(&mut decoder, &anchor);
    assert_eq!(parameters, 36);
    let output = root.join("native-results");
    fs::create_dir(&output).unwrap();
    let started = Instant::now();
    let mut traces = 0;
    let mut steps = 0;
    for record in settings["records"].as_array().unwrap() {
        let input =
            SafeTensorsModel::load(PathBuf::from(record["input"].as_str().unwrap())).unwrap();
        let observation = input.tensor_f32("observation").unwrap();
        assert_eq!(observation.len(), 48 * b * width);
        let prior = cached(record, "prior_features");
        let prior_logits = cached(record, "prior_logits");
        let teacher_logits = cached(record, "teacher_logits");
        let teacher = cached(record, "teacher_features");
        let old_rgb = cached(record, "rgb");
        assert_eq!(prior.len(), 32 * b * config.feature_dim());
        assert_eq!(prior_logits.len(), 32 * b * sw);
        let (prior_h, prior_z) = state_blocks(&prior, size.deter, sw);
        let (_, teacher_z) = state_blocks(&teacher, size.deter, sw);
        let mut rng = StdRng::seed_from_u64(record["seed"].as_u64().unwrap());
        for _ in 0..16 * b * size.stoch {
            let _: f32 = rng.random();
        }
        let mut all_logits = Vec::new();
        let mut all_z = Vec::new();
        let mut images = Vec::new();
        for time in 0..32 {
            let h = &prior_h[time * b * size.deter..(time + 1) * b * size.deter];
            let mut correction_rng = rng.clone();
            let z = sample_latents(
                &prior_logits[time * b * sw..(time + 1) * b * sw],
                b,
                size.stoch,
                size.classes,
                config.unimix,
                &mut rng,
            );
            exact(
                &z,
                &prior_z[time * b * sw..(time + 1) * b * sw],
                "cached prior draw",
            );
            posterior.set_input("deter", h);
            posterior.set_input(
                "observation",
                &observation[(16 + time) * b * width..(17 + time) * b * width],
            );
            posterior.step();
            posterior.wait();
            let mut logits = vec![0.; b * sw];
            posterior.read_output_by_index(0, &mut logits);
            let z = sample_latents(
                &logits,
                b,
                size.stoch,
                size.classes,
                config.unimix,
                &mut correction_rng,
            );
            if time == 0 {
                close(&logits, &teacher_logits[..b * sw], "step1 posterior logits");
                exact(&z, &teacher_z[..b * sw], "step1 posterior sample");
            }
            if let Some(index) = [1, 4, 8, 16, 32]
                .iter()
                .position(|&horizon| horizon == time + 1)
            {
                decoder.set_input("deter", h);
                decoder.set_input("stoch", &z);
                decoder.step();
                decoder.wait();
                let mut rgb = vec![0.; b * width];
                decoder.read_output_by_index(0, &mut rgb);
                if time == 0 {
                    close(
                        &rgb,
                        &old_rgb[3 * 5 * b * width..(3 * 5 + 1) * b * width],
                        "step1 teacher readout",
                    );
                }
                assert_eq!(images.len(), index * b * width);
                images.extend(rgb);
            }
            all_logits.extend(logits);
            all_z.extend(z);
            steps += 1;
        }
        let stem = format!(
            "{}-{}",
            record["batch"].as_str().unwrap(),
            record["draw"].as_u64().unwrap()
        );
        let file = format!("{stem}.bin");
        let layout = write_arrays(
            &output.join(&file),
            vec![
                ("logits", vec![32, b, size.stoch, size.classes], all_logits),
                ("z", vec![32, b, size.stoch, size.classes], all_z),
                ("rgb", vec![5, b, 3, 64, 64], images),
            ],
        );
        fs::write(
            output.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(
                &json!({"file":file,"batch":record["batch"],"draw":record["draw"],"layout":layout}),
            )
            .unwrap(),
        )
        .unwrap();
        traces += 1;
        println!("posterior correction {traces}/24 {stem}");
    }
    assert_eq!(traces, 24);
    assert_eq!(steps, 768);
    assert_eq!(
        invariant(&mut posterior, &anchor) + invariant(&mut decoder, &anchor),
        parameters
    );
    assert_eq!(
        crate::vision::checkpoint_sha256(&source).unwrap(),
        settings["checkpoint_sha256"].as_str().unwrap()
    );
    fs::write(output.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","traces":traces,"correction_steps":steps,"cached_prior_samples_exact":steps,"step1_teacher_checks":traces,"invariant_parameters":parameters,"optimizer_steps":0,"new_forecasts":0,"new_environment_actions":0,"future_images_privileged":true,"world_model_proven":false,"wall_seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
