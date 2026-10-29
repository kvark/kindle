//! Detached color-position targets from the current resized replay pixels.

use super::super::config::ColorCentroidSupervision;
use super::*;

const MIN_MASS: f32 = 1e-6;

fn targets(
    graph: &mut Graph,
    settings: &ColorCentroidSupervision,
    pixels: NodeId,
    rows: usize,
    side: usize,
) -> (NodeId, NodeId) {
    let width = side * side;
    let pixels = graph.stop_gradient(pixels);
    let offset = graph.constant(vec![0.5; width], &[width]);
    let channels = (0..3)
        .map(|channel| {
            let values = slice_columns(graph, pixels, rows, 3 * width, channel * width, width);
            graph.bias_add(values, offset)
        })
        .collect::<Vec<_>>();
    let others = (0..3)
        .filter(|&i| i != settings.channel)
        .collect::<Vec<_>>();
    let difference = graph.sub(channels[others[0]], channels[others[1]]);
    let positive = graph.relu(difference);
    let background = graph.add(channels[others[1]], positive);
    let background = graph.scale(background, settings.background_scale);
    let excess = graph.sub(channels[settings.channel], background);
    let excess = graph.relu(excess);
    let [x0, y0, x1, y1] = settings.region;
    let mask = (0..width)
        .map(|index| {
            let x = (index % side) as f32 + 0.5;
            let y = (index / side) as f32 + 0.5;
            if x / side as f32 >= x0
                && x / (side as f32) < x1
                && y / side as f32 >= y0
                && y / (side as f32) < y1
            {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mask = graph.constant(mask, &[width]);
    let mass = graph.bias_mul(excess, mask);
    let total = graph.sum_inner(mass);
    let minimum = graph.constant(vec![MIN_MASS; rows], &[rows, 1]);
    let valid = graph.greater(total, minimum);
    let denominator = graph.clamp(total, MIN_MASS, f32::MAX);
    let coordinates = [true, false].map(|horizontal| {
        let weights = (0..width)
            .map(|index| {
                let coordinate = if horizontal {
                    index % side
                } else {
                    index / side
                };
                (coordinate as f32 + 0.5) / side as f32 - 0.5
            })
            .collect();
        let weights = graph.constant(weights, &[width]);
        let weighted = graph.bias_mul(mass, weights);
        let numerator = graph.sum_inner(weighted);
        graph.div(numerator, denominator)
    });
    let coordinates =
        super::super::networks::concat_columns(graph, coordinates[0], coordinates[1], rows, 1, 1);
    (graph.stop_gradient(coordinates), graph.stop_gradient(valid))
}

pub(super) fn loss(
    graph: &mut Graph,
    settings: &ColorCentroidSupervision,
    pixels: NodeId,
    prediction: NodeId,
    rows: usize,
    side: usize,
) -> (NodeId, NodeId) {
    let (target, valid) = targets(graph, settings, pixels, rows, side);
    let error = graph.sub(prediction, target);
    let squared = graph.mul(error, error);
    let per_row = graph.sum_inner(squared);
    let weighted = graph.mul(per_row, valid);
    (graph.mean_all(weighted), graph.mean_all(valid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use meganeura::graph::Op;

    fn settings() -> ColorCentroidSupervision {
        ColorCentroidSupervision {
            scale: 7.0,
            channel: 0,
            background_scale: 1.5,
            region: [0.25, 0.25, 0.75, 0.75],
        }
    }

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.batch_size = 2;
        config.batch_length = 2;
        config.world_backprop_length = 2;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.cdp_centered = true;
        config.color_centroid_supervision = Some(settings());
        config
    }

    #[test]
    fn settings_validate_and_legacy_config_keeps_default_graph() {
        let enabled = config();
        assert!(enabled.check().is_ok());
        let mut disabled = enabled.clone();
        disabled.color_centroid_supervision = None;
        let encoded = serde_json::to_value(&disabled).unwrap();
        assert!(encoded.get("color_centroid_supervision").is_none());
        let legacy: DreamerConfig = serde_json::from_value(encoded).unwrap();
        assert!(legacy.color_centroid_supervision.is_none());
        assert_eq!(
            format!("{:?}", build_training_graph(&legacy, 2).nodes()),
            format!("{:?}", build_training_graph(&disabled, 2).nodes())
        );
        for invalid in [
            ColorCentroidSupervision {
                scale: 0.0,
                ..settings()
            },
            ColorCentroidSupervision {
                scale: f32::INFINITY,
                ..settings()
            },
            ColorCentroidSupervision {
                channel: 3,
                ..settings()
            },
            ColorCentroidSupervision {
                background_scale: f32::NAN,
                ..settings()
            },
            ColorCentroidSupervision {
                background_scale: -1.0,
                ..settings()
            },
            ColorCentroidSupervision {
                region: [-0.1, 0.0, 1.0, 1.0],
                ..settings()
            },
            ColorCentroidSupervision {
                region: [0.0, 0.0, 1.1, 1.0],
                ..settings()
            },
            ColorCentroidSupervision {
                region: [0.75, 0.25, 0.25, 0.75],
                ..settings()
            },
            ColorCentroidSupervision {
                region: [0.0, 0.0, 0.001, 0.001],
                ..settings()
            },
        ] {
            let mut value = enabled.clone();
            value.color_centroid_supervision = Some(invalid);
            assert!(value.check().is_err());
        }
        let mut features = DreamerConfig::tiny(3);
        features.color_centroid_supervision = Some(settings());
        assert!(features.check().is_err());
    }

    #[test]
    fn centroid_target_path_is_detached_and_compiles() {
        let mut graph = Graph::new();
        let pixels = graph.parameter("pixels", &[2, 3 * 16 * 16]);
        let prediction = graph.parameter("prediction", &[2, 2]);
        let (loss, _) = loss(&mut graph, &settings(), pixels, prediction, 2, 16);
        graph.set_outputs(vec![loss]);
        let backward = meganeura::autodiff::differentiate(&graph);
        assert_eq!(backward.node(backward.outputs()[1]).ty.shape, [1]);
        assert_eq!(backward.node(backward.outputs()[2]).ty.shape, [2, 2]);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    fn centroid_connects_cnn_and_posterior_but_not_behavior_and_preserves_inference() {
        let enabled = config();
        let mut disabled = enabled.clone();
        disabled.color_centroid_supervision = None;
        let mut graph = build_training_graph(&enabled, 2);
        let control = build_training_graph(&disabled, 2);
        let parameters = |graph: &Graph| {
            graph
                .nodes()
                .iter()
                .filter_map(|node| {
                    if let Op::Parameter { name } = &node.op {
                        Some((name.clone(), node.ty.shape.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        let all = parameters(&graph);
        assert_eq!(
            all.iter()
                .filter(|(n, _)| !n.starts_with("world.color_centroid."))
                .cloned()
                .collect::<Vec<_>>(),
            parameters(&control)
        );
        assert_eq!(
            all.iter()
                .filter(|(n, _)| n.starts_with("world.color_centroid."))
                .count(),
            5
        );
        assert_eq!(
            format!("{:?}", build_observe_graph(&enabled, 2).nodes()),
            format!("{:?}", build_observe_graph(&disabled, 2).nodes())
        );
        assert_eq!(
            format!("{:?}", build_imagination_graph(&enabled).nodes()),
            format!("{:?}", build_imagination_graph(&disabled).nodes())
        );
        graph.set_outputs(vec![graph.outputs()[LOSS_COLOR_CENTROID]]);
        let backward = meganeura::autodiff::differentiate(&graph);
        let mut cnn = 0;
        let mut posterior = 0;
        for (parameter, &gradient) in graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::Parameter { .. }))
            .zip(&backward.outputs()[1..])
        {
            let Op::Parameter { name } = &parameter.op else {
                unreachable!()
            };
            let connected = backward.node(gradient).ty == parameter.ty;
            if name.starts_with("world.representation.encoder.") {
                assert!(connected, "{name}");
                cnn += 1;
            } else if name.starts_with("world.representation.posterior.") {
                assert!(connected, "{name}");
                posterior += 1;
            } else if name.starts_with("behavior.") {
                assert!(!connected, "{name}");
            }
        }
        assert!(cnn > 0 && posterior > 0);
        let staged = build_staged_training_graph(&enabled);
        assert_eq!(staged.outputs().len(), COLOR_CENTROID_VALID_FRACTION + 3);
        assert_eq!(parameters(&staged), all);
    }

    #[test]
    #[ignore = "requires separately guarded GPU; F64 centroid targets, loss, gradients and zero masks"]
    fn centroid_matches_f64_targets_and_gradients() {
        use super::super::super::runtime::build_session;
        const SIDE: usize = 64;
        const ROWS: usize = 6;
        const WIDTH: usize = SIDE * SIDE;
        let settings = settings();
        let mut graph = Graph::new();
        let pixels = graph.parameter("pixels", &[ROWS, 3 * WIDTH]);
        let prediction = graph.parameter("prediction", &[ROWS, 2]);
        let (raw_loss, valid_fraction) =
            loss(&mut graph, &settings, pixels, prediction, ROWS, SIDE);
        let (target, valid) = targets(&mut graph, &settings, pixels, ROWS, SIDE);
        let total = graph.scale(raw_loss, settings.scale);
        graph.set_outputs(vec![total, raw_loss, valid_fraction, target, valid]);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!gpu.device_information().is_software_emulated);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        let mut images = vec![-0.5; ROWS * 3 * WIDTH];
        let mut paint = |row: usize, x: usize, y: usize, rgb: [f32; 3]| {
            for (channel, value) in rgb.into_iter().enumerate() {
                images[row * 3 * WIDTH + channel * WIDTH + y * SIDE + x] = value - 0.5;
            }
        };
        paint(1, 20, 30, [1.0, 0.0, 0.0]);
        paint(2, 18, 22, [0.75, 0.0, 0.0]);
        paint(2, 40, 38, [0.25, 0.0, 0.0]);
        paint(3, 0, 0, [1.0, 0.0, 0.0]);
        paint(3, 63, 63, [1.0, 0.0, 0.0]);
        paint(3, 25, 25, [0.4, 0.5, 0.0]);
        paint(4, 19, 35, [0.8, 0.2, 0.3]);
        paint(4, 41, 24, [0.6, 0.1, 0.05]);
        paint(5, 24, 24, [1e-7, 0.0, 0.0]);
        let predictions = (0..ROWS * 2)
            .map(|i| (i as f32 - 5.0) * 0.03)
            .collect::<Vec<_>>();
        session.set_parameter("pixels", &images);
        session.set_parameter("prediction", &predictions);
        session.step();
        session.wait();
        assert!(
            !session.has_param_grad("pixels"),
            "centroid targets must be detached"
        );
        let mut actual_target = vec![0.0; ROWS * 2];
        let mut actual_valid = vec![0.0; ROWS];
        session.read_output_by_index(3, &mut actual_target);
        session.read_output_by_index(4, &mut actual_valid);
        let mut gradients = vec![0.0; ROWS * 2];
        session.read_param_grad("prediction", &mut gradients);
        let mut expected_loss = 0.0;
        let mut valid_rows = 0;
        for row in 0..ROWS {
            let mut mass = 0.0;
            let mut sums = [0.0; 2];
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let centers = [
                        (x as f64 + 0.5) / SIDE as f64,
                        (y as f64 + 0.5) / SIDE as f64,
                    ];
                    if !(0.25..0.75).contains(&centers[0]) || !(0.25..0.75).contains(&centers[1]) {
                        continue;
                    }
                    let rgb = [0, 1, 2].map(|c| {
                        f64::from(images[row * 3 * WIDTH + c * WIDTH + y * SIDE + x]) + 0.5
                    });
                    let weight = (rgb[0] - 1.5 * rgb[1].max(rgb[2])).max(0.0);
                    mass += weight;
                    for axis in 0..2 {
                        sums[axis] += weight * (centers[axis] - 0.5);
                    }
                }
            }
            let valid = mass > f64::from(MIN_MASS);
            valid_rows += usize::from(valid);
            assert_eq!(actual_valid[row], if valid { 1.0 } else { 0.0 });
            for axis in 0..2 {
                let target = sums[axis] / mass.max(f64::from(MIN_MASS));
                assert!((f64::from(actual_target[row * 2 + axis]) - target).abs() < 2e-6);
                let error = f64::from(predictions[row * 2 + axis]) - target;
                let expected = if valid {
                    2.0 * f64::from(settings.scale) * error / ROWS as f64
                } else {
                    0.0
                };
                assert!((f64::from(gradients[row * 2 + axis]) - expected).abs() < 3e-6);
                if !valid {
                    assert_eq!(gradients[row * 2 + axis], 0.0);
                }
                expected_loss += if valid {
                    error * error / ROWS as f64
                } else {
                    0.0
                };
            }
        }
        assert_eq!(valid_rows, 3);
        assert!(
            (f64::from(session.read_loss()) - f64::from(settings.scale) * expected_loss).abs()
                < 3e-6
        );
        let mut raw_loss = [0.0];
        let mut valid_fraction = [0.0];
        session.read_output_by_index(1, &mut raw_loss);
        session.read_output_by_index(2, &mut valid_fraction);
        assert!((f64::from(raw_loss[0]) - expected_loss).abs() < 3e-6);
        assert_eq!(valid_fraction, [0.5]);
        assert_eq!(session.read_params(&["prediction"])[0], predictions);
        eprintln!("F64 color centroid targets/loss/gradients and exact absent-row zeros pass");
    }
}
