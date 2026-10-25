use super::*;
use crate::dreamer::runtime::{build_session, configure_d3_optimizer, initialize_d3};
use std::{collections::HashMap, sync::Arc};

fn check_device(session: &meganeura::Session) {
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        assert_eq!(session.device_information().device_name, expected);
        assert!(!session.device_information().is_software_emulated);
        let memory = session.device_memory_stats().unwrap();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
    }
}

#[test]
fn rgb_graphs_match_upstream_shapes_and_encoder_counts() {
    for (size, count) in [
        (crate::ModelSize::Size1M, 14_304),
        (crate::ModelSize::Size12M, 220_416),
    ] {
        let mut config = DreamerConfig::tiny(3);
        config.model_size = size;
        config.observation_kind = ObservationKind::Rgb64;
        let graph = crate::dreamer::world::build_training_graph(&config, 4);
        let backward = meganeura::autodiff::differentiate(&graph);
        assert!(backward.nodes().len() > graph.nodes().len());
        let mut graph = Graph::new();
        let encoder = Encoder::new(&mut graph, &config);
        let parameters = graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, meganeura::graph::Op::Parameter { .. }));
        assert_eq!(
            parameters.map(|n| n.ty.num_elements()).sum::<usize>(),
            count
        );
        assert_eq!(encoder.output_dim(), config.encoded_observation_dim());
        let input = graph.input("pixels", &config.observation_shape(2));
        let encoded = encoder.forward(&mut graph, input, 2);
        assert_eq!(
            graph.node(encoded).ty.shape,
            [2, config.encoded_observation_dim()]
        );
        let decoder = Decoder::new(&mut graph, &config, "world.decoder", config.feature_dim());
        let input = graph.input("feature", &[2, config.feature_dim()]);
        let decoded = decoder.forward(&mut graph, input, 2);
        assert_eq!(graph.node(decoded).ty.shape, [2, 12_288]);
    }
}

// Independent scalar cross-correlation, max-pool, per-pixel channel RMS and SiLU.
fn reference(input: &[f32], parameters: &HashMap<String, Vec<f32>>) -> Vec<f64> {
    let mut values = input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let (mut side, mut channels) = (64, 3);
    for (layer, output) in [8, 12, 16, 16].into_iter().enumerate() {
        let name = format!("world.representation.encoder.cnn{layer}");
        let weights = &parameters[&format!("{name}.weight")];
        let biases = &parameters[&format!("{name}.bias")];
        let scales = &parameters[&format!("{name}.norm.weight")];
        let mut convolution = vec![0.0; 2 * output * side * side];
        for b in 0..2 {
            for c in 0..output {
                for y in 0..side {
                    for x in 0..side {
                        let mut value = f64::from(biases[c]);
                        for ic in 0..channels {
                            for ky in 0..5 {
                                for kx in 0..5 {
                                    let iy = y as isize + ky as isize - 2;
                                    let ix = x as isize + kx as isize - 2;
                                    if (0..side as isize).contains(&iy)
                                        && (0..side as isize).contains(&ix)
                                    {
                                        value += values[((b * channels + ic) * side + iy as usize)
                                            * side
                                            + ix as usize]
                                            * f64::from(
                                                weights[((c * channels + ic) * 5 + ky) * 5 + kx],
                                            );
                                    }
                                }
                            }
                        }
                        convolution[((b * output + c) * side + y) * side + x] = value;
                    }
                }
            }
        }
        let next_side = side / 2;
        let mut next = vec![0.0; 2 * output * next_side * next_side];
        for b in 0..2 {
            for y in 0..next_side {
                for x in 0..next_side {
                    let mut pixel = Vec::with_capacity(output);
                    for c in 0..output {
                        let start = ((b * output + c) * side + 2 * y) * side + 2 * x;
                        pixel.push(
                            [start, start + 1, start + side, start + side + 1]
                                .into_iter()
                                .map(|i| convolution[i])
                                .fold(f64::NEG_INFINITY, f64::max),
                        );
                    }
                    let rms = (pixel.iter().map(|x| x * x).sum::<f64>() / output as f64
                        + f64::from(DREAMER_NORM_EPSILON))
                    .sqrt();
                    for (c, value) in pixel.into_iter().enumerate() {
                        let value = value / rms * f64::from(scales[c]);
                        next[((b * output + c) * next_side + y) * next_side + x] =
                            value / (1.0 + (-value).exp());
                    }
                }
            }
        }
        values = next;
        side = next_side;
        channels = output;
    }
    let mut result = Vec::new();
    for b in 0..2 {
        for p in 0..16 {
            for c in 0..16 {
                result.push(values[(b * 16 + c) * 16 + p]);
            }
        }
    }
    result
}

#[test]
#[ignore = "requires GPU; independent scalar CNN values and finite-difference gradients"]
fn tiny_rgb_encoder_matches_independent_reference() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb64;
    let mut graph = Graph::new();
    let encoder = Encoder::new(&mut graph, &config);
    let input = graph.input("pixels", &[2, 12_288]);
    let output = encoder.forward(&mut graph, input, 2);
    let squared = graph.mul(output, output);
    let loss = graph.mean_all(squared);
    graph.set_outputs(vec![loss, output]);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
    initialize_d3(&mut session, &graph, 103);
    check_device(&session);
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut parameters = HashMap::new();
    for name in names {
        let mut values = vec![0.0; session.param_size(&name).unwrap()];
        session.read_param(&name, &mut values);
        parameters.insert(name, values);
    }
    let pixels = (0..2 * 12_288)
        .map(|i| ((i * 37 + i / 43) % 256) as f32 / 255.0 - 0.5)
        .collect::<Vec<_>>();
    session.set_input("pixels", &pixels);
    session.step();
    session.wait();
    let expected = reference(&pixels, &parameters);
    let mut actual = vec![0.0; expected.len()];
    session.read_output_by_index(1, &mut actual);
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!((f64::from(a) - b).abs() < 3e-4, "{a} != {b}");
    }
    let scalar_loss = |p: &HashMap<String, Vec<f32>>| {
        let values = reference(&pixels, p);
        values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64
    };
    assert!((f64::from(session.read_loss()) - scalar_loss(&parameters)).abs() < 3e-4);
    for suffix in [
        "cnn0.weight",
        "cnn1.bias",
        "cnn2.norm.weight",
        "cnn3.weight",
    ] {
        let name = format!("world.representation.encoder.{suffix}");
        let mut gradient = vec![0.0; parameters[&name].len()];
        session.read_param_grad(&name, &mut gradient);
        let lane = (0..gradient.len())
            .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
            .unwrap();
        assert!(gradient[lane].is_finite() && gradient[lane].abs() > 1e-6);
        let original = parameters[&name][lane];
        parameters.get_mut(&name).unwrap()[lane] = original + 1e-4;
        let plus = scalar_loss(&parameters);
        let high = parameters[&name][lane];
        parameters.get_mut(&name).unwrap()[lane] = original - 1e-4;
        let minus = scalar_loss(&parameters);
        let low = parameters[&name][lane];
        parameters.get_mut(&name).unwrap()[lane] = original;
        let expected = (plus - minus) / f64::from(high - low);
        assert!(
            (f64::from(gradient[lane]) - expected).abs() < 3e-4 + expected.abs() * 0.003,
            "{name}: {} != {expected}",
            gradient[lane]
        );
    }
}

#[test]
#[ignore = "requires GPU and KINDLE_DREAMER_RGB_REFERENCE generated by pinned upstream JAX"]
fn rgb_pair_matches_upstream_values_and_gradients() {
    use meganeura::data::safetensors::SafeTensorsModel;
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_RGB_REFERENCE").unwrap());
    let fixture = SafeTensorsModel::load(root.join("reference.safetensors")).unwrap();
    let diagnostics =
        std::env::var_os("KINDLE_DREAMER_RGB_DIAGNOSTICS").map(std::path::PathBuf::from);
    if let Some(path) = &diagnostics {
        std::fs::create_dir(path).unwrap();
    }
    let mut config = DreamerConfig::tiny(18);
    config.model_size = crate::ModelSize::Size1M;
    config.observation_kind = ObservationKind::Rgb64;
    config.learning_rate_warmup = 2;
    let mut graph = Graph::new();
    let encoder = Encoder::new(&mut graph, &config);
    let decoder = Decoder::new(&mut graph, &config, "world.decoder", config.feature_dim());
    let pixels = graph.input("pixels", &[2, 12_288]);
    let features = graph.input("features", &[2, config.feature_dim()]);
    let encoded = encoder.forward(&mut graph, pixels, 2);
    let decoded = decoder.forward(&mut graph, features, 2);
    let squared = graph.mul(encoded, encoded);
    let encoding_loss = graph.mean_all(squared);
    let encoding_loss = scale(&mut graph, encoding_loss, 0.01);
    let reconstruction = graph.mse_loss(decoded, pixels);
    let reconstruction = scale(&mut graph, reconstruction, 12_288.0);
    let loss = graph.add(encoding_loss, reconstruction);
    graph.set_outputs(vec![loss, encoded, decoded]);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
    check_device(&session);
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let initial = fixture
        .tensor_info()
        .keys()
        .filter(|k| k.starts_with("initial/"))
        .count();
    assert_eq!(initial, names.len());
    for name in &names {
        session.set_parameter(
            name,
            &fixture.tensor_f32(&format!("initial/{name}")).unwrap(),
        );
    }
    let mut failures = Vec::new();
    let mut compare = |label: &str, actual: &[f32], absolute: f32, relative: f32| {
        let expected = fixture.tensor_f32(label).unwrap();
        assert_eq!(actual.len(), expected.len(), "{label}");
        let mut worst = 0.0_f32;
        let mut error_square = 0.0_f64;
        let mut expected_square = 0.0_f64;
        let mut outside = 0;
        for (&a, &b) in actual.iter().zip(&expected) {
            let error = (a - b).abs();
            assert!(a.is_finite(), "{label}: nonfinite value");
            outside += usize::from(error > absolute + relative * b.abs());
            worst = worst.max(error);
            error_square += f64::from(error).powi(2);
            expected_square += f64::from(b).powi(2);
        }
        let relative_l2 = (error_square / expected_square.max(1e-30)).sqrt();
        if outside != 0
            || (label.contains("/gradient/")
                && error_square.sqrt()
                    > 3e-3 * expected_square.sqrt() + 1e-7 * (actual.len() as f64).sqrt())
        {
            failures.push(format!(
                "{label}: outside={outside}/{} max={worst} relative_l2={relative_l2}",
                actual.len()
            ));
        }
        eprintln!("{label}: max absolute error {worst}, relative L2 {relative_l2}");
    };
    for step in 0..4 {
        let prefix = format!("step{step}");
        session.set_input(
            "pixels",
            &fixture.tensor_f32(&format!("{prefix}/pixels")).unwrap(),
        );
        session.set_input(
            "features",
            &fixture.tensor_f32(&format!("{prefix}/features")).unwrap(),
        );
        // Clipping modifies gradient buffers in place. Inspect the raw backward
        // pass, then execute the normal fused backward+optimizer independently.
        session.clear_optimizer();
        session.step();
        session.wait();
        let mut raw_gradients = std::collections::BTreeMap::new();
        for name in &names {
            let mut actual = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut actual);
            compare(&format!("{prefix}/gradient/{name}"), &actual, 3e-3, 3e-3);
            raw_gradients.insert(name.clone(), actual);
        }
        if let Some(path) = &diagnostics {
            serde_json::to_writer(
                std::fs::File::create(path.join(format!("{prefix}-gradients.json"))).unwrap(),
                &raw_gradients,
            )
            .unwrap();
        }
        configure_d3_optimizer(&mut session, &config, step, config.learning_rate);
        session.step();
        session.wait();
        if let Some(path) = &diagnostics {
            session
                .save_checkpoint(&path.join(format!("{prefix}.safetensors")))
                .unwrap();
        }
        for (index, (name, count)) in [
            ("loss", 1),
            ("encoded", 2 * config.encoded_observation_dim()),
            ("decoded", 2 * 12_288),
        ]
        .into_iter()
        .enumerate()
        {
            let mut actual = vec![0.0; count];
            session.read_output_by_index(index, &mut actual);
            compare(&format!("{prefix}/{name}"), &actual, 5e-4, 1e-5);
        }
        // Validate optimizer state separately with dreamer_optimizer_reference.py
        // using these exact raw gradients. First-step sign normalization can
        // amplify harmless cross-backend rounding of a near-zero gradient.
        check_device(&session);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
