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

#[test]
fn smaller_rgb_graphs_keep_shapes_and_cdp_gradients() {
    for (kind, side) in [(ObservationKind::Rgb32, 32), (ObservationKind::Rgb48, 48)] {
        let mut config = DreamerConfig::tiny(18);
        config.observation_kind = kind;
        assert!(config.check().is_ok());
        let mut graph = Graph::new();
        let encoder = Encoder::new(&mut graph, &config);
        let pixels = graph.input("pixels", &config.observation_shape(2));
        let encoded = encoder.forward(&mut graph, pixels, 2);
        assert_eq!(
            graph.node(encoded).ty.shape,
            [2, (side / 16) * (side / 16) * 16]
        );
        assert_eq!(encoder.output_dim(), config.encoded_observation_dim());
        let decoder = Decoder::new(&mut graph, &config, "world.decoder", config.feature_dim());
        let feature = graph.input("feature", &[2, config.feature_dim()]);
        let decoded = decoder.forward(&mut graph, feature, 2);
        assert_eq!(graph.node(decoded).ty.shape, [2, 3 * side * side]);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.cdp_centered = true;
        config.world_backprop_length = config.batch_length;
        config.world_microbatch_size = Some(config.batch_size);
        assert!(config.check().is_ok());
        assert!(config.is_cdp());
        assert_eq!(config.prediction_dim(), encoder.output_dim());
        let encoded_config = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(encoded_config).unwrap(),
            config
        );
        let graph = crate::dreamer::world::build_training_graph(&config, 4);
        let backward = meganeura::autodiff::differentiate(&graph);
        assert!(backward.nodes().len() > graph.nodes().len());
    }
}

// Independent scalar cross-correlation, max-pool, per-pixel channel RMS and SiLU.
fn reference_at_side(
    input: &[f32],
    parameters: &HashMap<String, Vec<f32>>,
    mut side: usize,
) -> Vec<f64> {
    let mut values = input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let input_channels = parameters["world.representation.encoder.cnn0.weight"].len() / (8 * 25);
    let mut channels = input_channels;
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
        for p in 0..side * side {
            for c in 0..channels {
                result.push(values[(b * channels + c) * side * side + p]);
            }
        }
    }
    result
}

#[test]
fn rgb96_scalar_reference_keeps_all_spatial_tokens() {
    let mut parameters = HashMap::new();
    let mut input_channels = 3;
    for (layer, output_channels) in [8, 12, 16, 16].into_iter().enumerate() {
        let name = format!("world.representation.encoder.cnn{layer}");
        parameters.insert(
            format!("{name}.weight"),
            vec![0.0; output_channels * input_channels * 25],
        );
        parameters.insert(
            format!("{name}.bias"),
            (1..=output_channels).map(|i| i as f32 / 16.0).collect(),
        );
        parameters.insert(format!("{name}.norm.weight"), vec![1.0; output_channels]);
        input_channels = output_channels;
    }
    let expected_rms = ((1..=16).map(|i| (i as f64 / 16.0).powi(2)).sum::<f64>() / 16.0
        + f64::from(DREAMER_NORM_EPSILON))
    .sqrt();
    let actual = reference_at_side(&vec![0.0; 2 * 3 * 96 * 96], &parameters, 96);
    assert_eq!(actual.len(), 2 * 36 * 16);
    for token in actual.chunks_exact(16) {
        for (channel, value) in token.iter().enumerate() {
            let normalized = (channel + 1) as f64 / 16.0 / expected_rms;
            let expected = normalized / (1.0 + (-normalized).exp());
            assert!((value - expected).abs() < 1e-12);
        }
    }
}

#[test]
#[ignore = "requires GPU; independent scalar CNN values and finite-difference gradients"]
fn tiny_rgb_encoder_matches_independent_reference() {
    check_encoder_reference(false, false, false);
}

#[test]
#[ignore = "requires guarded GPU; six-channel scalar CNN and both-frame derivatives"]
fn tiny_rgb_frame_pair_encoder_matches_independent_reference() {
    check_encoder_reference(true, false, false);
}

#[test]
#[ignore = "requires guarded GPU and KINDLE_RGB_PAIR_DERIVATIVES output path"]
fn capture_tiny_rgb_frame_pair_encoder_derivatives() {
    check_encoder_reference(true, false, true);
}

#[test]
#[ignore = "requires guarded GPU; coordinate CNN values and RGB/x/y weight derivatives"]
fn tiny_rgb_coordinate_encoder_matches_independent_reference() {
    check_encoder_reference(false, true, false);
}

fn check_encoder_reference(frame_pair: bool, coordinates: bool, capture: bool) {
    check_encoder_at_side(frame_pair, coordinates, capture, ObservationKind::Rgb64);
}

#[test]
#[ignore = "requires separately guarded RGB96 scalar CNN and derivative qualification"]
fn tiny_rgb96_encoder_matches_spatial_reference() {
    check_encoder_at_side(false, false, false, ObservationKind::Rgb96);
}

fn check_encoder_at_side(
    frame_pair: bool,
    coordinates: bool,
    capture: bool,
    kind: ObservationKind,
) {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = kind;
    let side = config.observation_grid();
    config.rgb_frame_pair = frame_pair;
    config.rgb_coordinates = coordinates;
    let mut graph = Graph::new();
    let encoder = Encoder::new(&mut graph, &config);
    let input = graph.input("pixels", &[2, config.observation_dim()]);
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
    let pixels = (0..2 * config.observation_dim())
        .map(|i| ((i * 37 + i / 43) % 256) as f32 / 255.0 - 0.5)
        .collect::<Vec<_>>();
    session.set_input("pixels", &pixels);
    session.step();
    session.wait();
    let reference_pixels = if coordinates {
        let mut augmented = Vec::new();
        for frame in pixels.chunks_exact(config.observation_dim()) {
            augmented.extend_from_slice(frame);
            for axis in 0..2 {
                for y in 0..side {
                    for x in 0..side {
                        let position = if axis == 0 { x } else { y };
                        augmented
                            .push(((2 * position + 1) as f64 / (2 * side) as f64 - 0.5) as f32);
                    }
                }
            }
        }
        augmented
    } else {
        pixels.clone()
    };
    let expected = reference_at_side(&reference_pixels, &parameters, side);
    assert_eq!(expected.len(), 2 * config.encoded_observation_dim());
    let mut actual = vec![0.0; expected.len()];
    session.read_output_by_index(1, &mut actual);
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!((f64::from(a) - b).abs() < 3e-4, "{a} != {b}");
    }
    let scalar_loss = |p: &HashMap<String, Vec<f32>>| {
        let values = reference_at_side(&reference_pixels, p, side);
        values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64
    };
    assert!((f64::from(session.read_loss()) - scalar_loss(&parameters)).abs() < 3e-4);
    if capture {
        let mut gradients = HashMap::new();
        for (name, values) in &parameters {
            let mut gradient = vec![0.0; values.len()];
            session.read_param_grad(name, &mut gradient);
            assert!(gradient.iter().all(|value| value.is_finite()));
            gradients.insert(name, gradient);
        }
        let path = std::env::var_os("KINDLE_RGB_PAIR_DERIVATIVES")
            .expect("explicit derivative capture path");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(file, &serde_json::json!({
            "seed": 103, "input_channels": config.observation_channels(), "batch": 2, "side": side,
            "normalization_epsilon": DREAMER_NORM_EPSILON, "precision": config.training_precision,
            "pixels": pixels, "parameters": parameters, "gradients": gradients,
            "gpu_outputs": actual, "scalar_outputs": expected, "gpu_loss": session.read_loss(),
        })).unwrap();
        return;
    }
    let mut checks = vec![
        ("cnn0.weight", None),
        ("cnn1.bias", None),
        ("cnn2.norm.weight", None),
        ("cnn3.weight", None),
    ];
    if frame_pair {
        checks[0].1 = Some(0);
        checks.push(("cnn0.weight", Some(1)));
    }
    if coordinates {
        checks[0].1 = Some(0);
        checks.extend([("cnn0.weight", Some(1)), ("cnn0.weight", Some(2))]);
    }
    for (suffix, image) in checks {
        let name = format!("world.representation.encoder.{suffix}");
        let mut gradient = vec![0.0; parameters[&name].len()];
        session.read_param_grad(&name, &mut gradient);
        let lane = (0..gradient.len())
            .filter(|&i| {
                image.is_none_or(|image| {
                    if coordinates {
                        let channel = (i / 25) % 5;
                        if image == 0 {
                            channel < 3
                        } else {
                            channel == image + 2
                        }
                    } else {
                        (i / 25) % 6 / 3 == image
                    }
                })
            })
            .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
            .unwrap();
        assert!(gradient[lane].is_finite() && gradient[lane].abs() > 1e-6);
        let original = parameters[&name][lane];
        // The paired fixture crosses max-pool boundaries at 1e-4; 1e-5 and
        // 1e-6 converge without changing winners in the independent reference.
        let step = if frame_pair || coordinates || side == 96 {
            1e-5
        } else {
            1e-4
        };
        parameters.get_mut(&name).unwrap()[lane] = original + step;
        let plus = scalar_loss(&parameters);
        let high = parameters[&name][lane];
        parameters.get_mut(&name).unwrap()[lane] = original - step;
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
fn rgb_coordinate_planes_and_graph_preserve_replay_geometry() {
    for (kind, side) in [
        (ObservationKind::Rgb32, 32),
        (ObservationKind::Rgb48, 48),
        (ObservationKind::Rgb64, 64),
    ] {
        let planes = coordinate_planes(side);
        assert_eq!(planes.len(), 2 * side * side);
        for y in 0..side {
            for x in 0..side {
                let index = y * side + x;
                assert!(
                    (f64::from(planes[index]) - ((x as f64 + 0.5) / side as f64 - 0.5)).abs()
                        < 3e-8
                );
                assert!(
                    (f64::from(planes[side * side + index])
                        - ((y as f64 + 0.5) / side as f64 - 0.5))
                        .abs()
                        < 3e-8
                );
            }
        }
        let mut config = DreamerConfig::tiny(18);
        config.model_size = crate::ModelSize::Size12M;
        config.observation_kind = kind;
        config.rgb_coordinates = true;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.cdp_centered = true;
        assert!(config.check().is_ok());
        assert_eq!(config.observation_dim(), 3 * side * side);
        let mut graph = Graph::new();
        let encoder = Encoder::new(&mut graph, &config);
        let count: usize = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, meganeura::graph::Op::Parameter { .. }))
            .map(|node| node.ty.num_elements())
            .sum();
        assert_eq!(count, 220_416 + 2 * 25 * 32);
        let pixels = graph.input("pixels", &config.observation_shape(2));
        let output = encoder.forward(&mut graph, pixels, 2);
        assert_eq!(
            graph.node(output).ty.shape,
            [2, config.encoded_observation_dim()]
        );
        let world = crate::dreamer::world::build_training_graph(&config, 4);
        assert!(meganeura::autodiff::differentiate(&world).nodes().len() > world.nodes().len());
    }
}

#[test]
fn rgb_frame_pair_graph_preserves_latent_width_and_adds_first_layer_weights() {
    for (kind, side) in [
        (ObservationKind::Rgb32, 32),
        (ObservationKind::Rgb48, 48),
        (ObservationKind::Rgb64, 64),
    ] {
        let mut config = DreamerConfig::tiny(18);
        config.model_size = crate::ModelSize::Size12M;
        config.observation_kind = kind;
        config.rgb_frame_pair = true;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        assert!(config.check().is_ok());
        let mut graph = Graph::new();
        let encoder = Encoder::new(&mut graph, &config);
        let parameters = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, meganeura::graph::Op::Parameter { .. }));
        assert_eq!(
            parameters.map(|node| node.ty.num_elements()).sum::<usize>(),
            220_416 + 3 * 25 * 32
        );
        let pixels = graph.input("pixels", &[2, 6 * side * side]);
        let output = encoder.forward(&mut graph, pixels, 2);
        assert_eq!(
            graph.node(output).ty.shape,
            [2, config.encoded_observation_dim()]
        );
        let square = graph.mul(output, output);
        let loss = graph.mean_all(square);
        graph.set_outputs(vec![loss]);
        let derivatives = meganeura::autodiff::differentiate(&graph);
        assert!(derivatives.nodes().len() > graph.nodes().len());
        let world = crate::dreamer::world::build_training_graph(&config, 4);
        let derivatives = meganeura::autodiff::differentiate(&world);
        assert!(derivatives.nodes().len() > world.nodes().len());
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
