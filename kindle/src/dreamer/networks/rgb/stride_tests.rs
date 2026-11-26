use super::*;
use crate::dreamer::runtime::{build_session, initialize_d3};
use std::{collections::HashMap, sync::Arc};

// Independent scalar cross-correlation at even input centers, channel RMS and SiLU.
fn strided_reference(
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
        let next_side = side / 2;
        let mut convolution = vec![0.0; 2 * output * next_side * next_side];
        for b in 0..2 {
            for c in 0..output {
                for y in 0..next_side {
                    for x in 0..next_side {
                        let mut value = f64::from(biases[c]);
                        for ic in 0..channels {
                            for ky in 0..5 {
                                for kx in 0..5 {
                                    let iy = (2 * y) as isize + ky as isize - 2;
                                    let ix = (2 * x) as isize + kx as isize - 2;
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
                        convolution[((b * output + c) * next_side + y) * next_side + x] = value;
                    }
                }
            }
        }
        let mut next = vec![0.0; 2 * output * next_side * next_side];
        for b in 0..2 {
            for y in 0..next_side {
                for x in 0..next_side {
                    let mut pixel = Vec::with_capacity(output);
                    for c in 0..output {
                        pixel.push(convolution[((b * output + c) * next_side + y) * next_side + x]);
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
fn rgb_stride2_scalar_reference_samples_spatial_centers_and_all_channels() {
    let side = 32;
    let pixels = (0..2 * 3 * side * side)
        .map(|i| ((i * 37 + i / 43) % 251) as f32 / 255.0 - 0.5)
        .collect::<Vec<_>>();
    let mut parameters = HashMap::new();
    let mut channels = 3;
    for (layer, output) in [8, 12, 16, 16].into_iter().enumerate() {
        let name = format!("world.representation.encoder.cnn{layer}");
        let mut weights = vec![0.0; output * channels * 25];
        weights[12] = 1.0;
        parameters.insert(format!("{name}.weight"), weights);
        parameters.insert(format!("{name}.bias"), vec![0.0; output]);
        parameters.insert(format!("{name}.norm.weight"), vec![1.0; output]);
        channels = output;
    }
    let actual = strided_reference(&pixels, &parameters, side);
    assert_eq!(actual.len(), 2 * 2 * 2 * 16);
    for b in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let mut expected = f64::from(pixels[b * 3 * side * side + 16 * y * side + 16 * x]);
                for width in [8, 12, 16, 16] {
                    expected /= (expected * expected / f64::from(width)
                        + f64::from(DREAMER_NORM_EPSILON))
                    .sqrt();
                    expected /= 1.0 + (-expected).exp();
                }
                for c in 0..16 {
                    let value = if c == 0 { expected } else { 0.0 };
                    assert!((actual[((b * 2 + y) * 2 + x) * 16 + c] - value).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn rgb_stride2_quarters_convolution_positions_and_preserves_decoder() {
    use meganeura::graph::Op;
    use std::collections::BTreeMap;
    for kind in [
        ObservationKind::Rgb32,
        ObservationKind::Rgb48,
        ObservationKind::Rgb64,
        ObservationKind::Rgb96,
    ] {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = kind;
        let mut parameters = None;
        let mut original_positions = 0;
        let mut original_decoder = None;
        for enabled in [false, true] {
            config.rgb_stride2 = enabled;
            let mut graph = Graph::new();
            let encoder = Encoder::new(&mut graph, &config);
            let pixels = graph.input("pixels", &config.observation_shape(2));
            let encoded = encoder.forward(&mut graph, pixels, 2);
            assert_eq!(
                graph.node(encoded).ty.shape,
                [2, config.encoded_observation_dim()]
            );
            let shapes = graph
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                    _ => None,
                })
                .collect::<BTreeMap<_, _>>();
            let convolutions = graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::Conv2d { .. }))
                .collect::<Vec<_>>();
            assert_eq!(convolutions.len(), 4);
            for node in &convolutions {
                if let Op::Conv2d { stride, .. } = node.op {
                    assert_eq!(stride, if enabled { 2 } else { 1 });
                }
            }
            let positions = convolutions
                .iter()
                .map(|n| n.ty.num_elements())
                .sum::<usize>();
            let pools = graph
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::MaxPool2d { .. }))
                .count();
            assert_eq!(pools, if enabled { 0 } else { 4 });
            if enabled {
                assert_eq!(parameters.as_ref(), Some(&shapes));
                assert_eq!(4 * positions, original_positions);
            } else {
                parameters = Some(shapes);
                original_positions = positions;
            }
            let mut decoder_graph = Graph::new();
            let decoder = Decoder::new(
                &mut decoder_graph,
                &config,
                "world.decoder",
                config.feature_dim(),
            );
            let features = decoder_graph.input("features", &[2, config.feature_dim()]);
            let decoded = decoder.forward(&mut decoder_graph, features, 2);
            decoder_graph.set_outputs(vec![decoded]);
            let identity = format!("{:?}:{:?}", decoder_graph.nodes(), decoder_graph.outputs());
            if enabled {
                assert_eq!(original_decoder.as_ref(), Some(&identity));
            } else {
                original_decoder = Some(identity);
            }
        }
    }
}

#[test]
#[ignore = "requires guarded RGB stride-two scalar values and finite-difference derivatives"]
fn tiny_rgb_stride2_encoder_matches_independent_reference() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb64;
    config.rgb_stride2 = true;
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
    assert_eq!(
        session.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!session.device_information().is_software_emulated);
    let memory = session.device_memory_stats().unwrap();
    assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
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
    let expected = strided_reference(&pixels, &parameters, 64);
    assert_eq!(expected.len(), 2 * config.encoded_observation_dim());
    let mut actual = vec![0.0; expected.len()];
    session.read_output_by_index(1, &mut actual);
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!((f64::from(a) - b).abs() < 3e-4, "{a} != {b}");
    }
    let scalar_loss = |p: &HashMap<String, Vec<f32>>| {
        let values = strided_reference(&pixels, p, 64);
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
        let index = (0..gradient.len())
            .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
            .unwrap();
        assert!(gradient[index].is_finite() && gradient[index].abs() > 1e-6);
        let original = parameters[&name][index];
        parameters.get_mut(&name).unwrap()[index] = original + 1e-4;
        let plus = scalar_loss(&parameters);
        let high = parameters[&name][index];
        parameters.get_mut(&name).unwrap()[index] = original - 1e-4;
        let minus = scalar_loss(&parameters);
        let low = parameters[&name][index];
        parameters.get_mut(&name).unwrap()[index] = original;
        let expected = (plus - minus) / f64::from(high - low);
        assert!(
            (f64::from(gradient[index]) - expected).abs() < 3e-4 + expected.abs() * 0.003,
            "{name}: {} != {expected}",
            gradient[index]
        );
    }
}
