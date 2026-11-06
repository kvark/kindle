use super::*;
use crate::dreamer::{ExpertImitation, LatentShape, ModelSize, ObservationKind};
use meganeura::{Graph, graph::Op};
use std::collections::BTreeMap;

pub(super) fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.cdp_centered = true;
    config.expert_imitation = Some(ExpertImitation {
        scale: 2.0,
        action_weights: vec![1.0; 3],
        world_gradient: true,
        decay: None,
    });
    config.validate();
    config
}

fn graphs(config: &DreamerConfig) -> Vec<Graph> {
    vec![
        world::build_training_graph(config, config.world_backprop_length),
        world::build_posterior_graph(config),
        world::build_observe_graph(config, 2),
        world::build_imagination_graph(config),
        world::build_transition_graph(config, 2),
        behavior::build_training_graph(config, 2, 2),
        behavior::build_inference_graph(config, 2),
        behavior::build_actor_inference_graph(config, 2),
    ]
}

fn parameters(graph: &Graph) -> BTreeMap<String, Vec<usize>> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn latent_shape_legacy_identity_and_roundtrip() {
    let mut config = config();
    let json = serde_json::to_value(&config).unwrap();
    assert!(json.get("latent_shape").is_none());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(json).unwrap(),
        config
    );
    let original = graphs(&config);
    config.latent_shape = Some(LatentShape {
        variables: 4,
        classes: 4,
    });
    for (a, b) in original.iter().zip(graphs(&config)) {
        assert_eq!(format!("{:?}", a.nodes()), format!("{:?}", b.nodes()));
        assert_eq!(a.outputs(), b.outputs());
    }
    config.latent_shape = Some(LatentShape {
        variables: 8,
        classes: 2,
    });
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(
        json["latent_shape"],
        serde_json::json!({"variables": 8, "classes": 2})
    );
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(json).unwrap(),
        config
    );
}

#[test]
fn latent_shape_rejects_invalid_width_overflow_and_scope() {
    for (variables, classes) in [
        (0, 4),
        (16, 1),
        (8, 0),
        (8, 4),
        (usize::MAX, 2),
        (2, usize::MAX),
    ] {
        let mut config = config();
        config.latent_shape = Some(LatentShape { variables, classes });
        assert!(config.check().unwrap_err().contains("latent_shape"));
    }
    for change in [
        |c: &mut DreamerConfig| c.cdp_centered = false,
        |c: &mut DreamerConfig| c.observation_kind = ObservationKind::Features,
        |c: &mut DreamerConfig| c.loss_scales.future_prediction = 0.0,
    ] {
        let mut config = config();
        config.latent_shape = Some(LatentShape {
            variables: 8,
            classes: 2,
        });
        change(&mut config);
        assert!(config.check().unwrap_err().contains("centered RGB CDP"));
    }
}

#[test]
fn latent_shape_preserves_parameters_and_changes_categorical_geometry() {
    for model in [ModelSize::Tiny, ModelSize::Size12M] {
        let mut plain = config();
        plain.model_size = model;
        if model == ModelSize::Size12M {
            plain.observation_kind = ObservationKind::Rgb64;
        }
        let preset = plain.network();
        let mut grouped = plain.clone();
        grouped.latent_shape = Some(LatentShape {
            variables: preset.stoch * 2,
            classes: preset.classes / 2,
        });
        grouped.validate();
        assert_eq!(plain.feature_dim(), grouped.feature_dim());
        for (a, b) in graphs(&plain).iter().zip(graphs(&grouped)) {
            assert_eq!(parameters(a), parameters(&b));
        }
        let graph = world::build_imagination_graph(&grouped);
        let stoch = graph
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, Op::Input { name } if name == "stoch"))
            .unwrap();
        assert_eq!(
            stoch.ty.shape,
            [
                grouped.batch_size * grouped.batch_length * preset.stoch * 2,
                preset.classes / 2
            ]
        );
    }
}

fn categorical_graph(variables: usize, classes: usize, stop_q: bool, stop_p: bool) -> Graph {
    let mut graph = Graph::new();
    let q = graph.parameter("posterior", &[2 * variables, classes]);
    let p = graph.parameter("prior", &[2 * variables, classes]);
    let uniforms = graph.input("uniforms", &[2 * variables, classes]);
    let kl = super::super::networks::categorical_kl(
        &mut graph, q, p, 2, variables, classes, 0.01, stop_q, stop_p, 1.0,
    );
    let sample = super::super::networks::gumbel_sample(
        &mut graph,
        q,
        uniforms,
        2 * variables,
        classes,
        0.01,
    );
    graph.set_outputs(vec![kl.loss, kl.raw, sample]);
    graph
}

fn softmax(values: &[f32]) -> Vec<f64> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = values
        .iter()
        .map(|&x| (f64::from(x) - f64::from(max)).exp())
        .collect::<Vec<_>>();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

fn reference(
    q_logits: &[f32],
    p_logits: &[f32],
    variables: usize,
    classes: usize,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let mixture = f64::from(0.01f32);
    let mut dq = vec![0.0; q_logits.len()];
    let mut dp = vec![0.0; p_logits.len()];
    let (mut loss, mut raw) = (0.0, 0.0);
    for batch in 0..2 {
        let mut state_kl = 0.0;
        for variable in 0..variables {
            let offset = (batch * variables + variable) * classes;
            let range = offset..offset + classes;
            let sq = softmax(&q_logits[range.clone()]);
            let sp = softmax(&p_logits[range.clone()]);
            let q = sq
                .iter()
                .map(|x| (1.0 - mixture) * x + mixture / classes as f64)
                .collect::<Vec<_>>();
            let p = sp
                .iter()
                .map(|x| (1.0 - mixture) * x + mixture / classes as f64)
                .collect::<Vec<_>>();
            let gq = q
                .iter()
                .zip(&p)
                .map(|(q, p)| (q / p).ln() + 1.0)
                .collect::<Vec<_>>();
            let gp = q.iter().zip(&p).map(|(q, p)| -q / p).collect::<Vec<_>>();
            let mean_q: f64 = sq.iter().zip(&gq).map(|(s, g)| s * g).sum();
            let mean_p: f64 = sp.iter().zip(&gp).map(|(s, g)| s * g).sum();
            for i in 0..classes {
                state_kl += q[i] * (q[i] / p[i]).ln();
                dq[offset + i] = (1.0 - mixture) * sq[i] * (gq[i] - mean_q) / 2.0;
                dp[offset + i] = (1.0 - mixture) * sp[i] * (gp[i] - mean_p) / 2.0;
            }
        }
        raw += state_kl / 2.0;
        loss += state_kl.max(1.0) / 2.0;
        if state_kl < 1.0 {
            let range = batch * variables * classes..(batch + 1) * variables * classes;
            dq[range.clone()].fill(0.0);
            dp[range].fill(0.0);
        }
    }
    (loss, raw, dq, dp)
}

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q = (0..1024)
        .map(|i| (i % 17) as f32 * 0.13 - 1.0)
        .collect::<Vec<_>>();
    let mut p = q.clone();
    for x in &mut p[512..] {
        *x = -*x * 0.7;
    }
    let draws = (0..1024)
        .map(|i| ((i * 13 % 97) + 1) as f32 / 99.0)
        .collect();
    (q, p, draws)
}

#[test]
fn latent_shape_reference_and_fixture_cover_clamped_and_active_states() {
    let (q, p, _) = inputs();
    for (variables, classes) in [(32, 16), (64, 8)] {
        let (loss, raw, dq, dp) = reference(&q, &p, variables, classes);
        assert!((loss - raw - 0.5).abs() < 1e-10);
        for gradient in [dq, dp] {
            assert!(gradient[..512].iter().all(|&x| x == 0.0));
            assert!(gradient[512..].iter().any(|&x| x.abs() > 1e-3));
            for row in gradient.chunks(classes) {
                assert!(row.iter().sum::<f64>().abs() < 1e-12);
            }
        }
        for (stop_q, stop_p) in [(false, false), (true, false), (false, true)] {
            let graph = categorical_graph(variables, classes, stop_q, stop_p);
            assert_eq!(
                graph.node(graph.outputs()[2]).ty.shape,
                [2 * variables, classes]
            );
        }
    }
}

#[test]
#[ignore = "requires separately declared guarded latent shape qualification"]
fn latent_shape_native_categorical_values_derivatives_and_samples() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let device = crate::gpu_device_info(gpu.device_information());
    assert_eq!(
        device.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let (q, p, uniforms) = inputs();
    for (variables, classes) in [(32, 16), (64, 8)] {
        let (loss, raw, dq, dp) = reference(&q, &p, variables, classes);
        for (stop_q, stop_p) in [(false, false), (true, false), (false, true)] {
            let graph = categorical_graph(variables, classes, stop_q, stop_p);
            let mut session =
                super::super::runtime::build_training_session(&graph, &gpu, &config());
            session.clear_optimizer();
            session.disable_grad_clip();
            session.set_parameter("posterior", &q);
            session.set_parameter("prior", &p);
            session.set_input("uniforms", &uniforms);
            let before = session.read_params(&["posterior", "prior"]);
            session.zero_grad();
            session.step();
            session.wait();
            assert!((f64::from(session.read_loss()) - loss).abs() < 2e-4);
            assert!((f64::from(session.read_output(1)[0]) - raw).abs() < 2e-4);
            for (name, expected, stopped) in [("posterior", &dq, stop_q), ("prior", &dp, stop_p)] {
                let mut actual = vec![0.0; 1024];
                if session.has_param_grad(name) {
                    session.read_param_grad(name, &mut actual);
                }
                for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                    let b = if stopped { 0.0 } else { b };
                    assert!(
                        (f64::from(a) - b).abs() < 2e-5,
                        "{variables}x{classes} {name}[{i}]: {a} vs {b}"
                    );
                }
            }
            let sample = session.read_output(2);
            assert_eq!(sample.len(), 1024);
            for (row, onehot) in sample.chunks(classes).enumerate() {
                let range = row * classes..(row + 1) * classes;
                let soft = softmax(&q[range.clone()]);
                let scores = soft
                    .iter()
                    .zip(&uniforms[range])
                    .map(|(s, &u)| {
                        ((1.0 - f64::from(0.01f32)) * s + f64::from(0.01f32) / classes as f64).ln()
                            - (-f64::from(u).ln()).ln()
                    })
                    .collect::<Vec<_>>();
                let selected = (0..classes)
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                    .unwrap();
                for (i, &value) in onehot.iter().enumerate() {
                    assert_eq!(value, if i == selected { 1.0 } else { 0.0 });
                }
            }
            assert_eq!(before, session.read_params(&["posterior", "prior"]));
            assert_eq!(session.adam_step_count(), 0);
        }
    }
    eprintln!(
        "Categorical32x16/64x8: F64 mixed-KL values/derivatives, free-nat mask, stopped routes and supplied-draw samples pass; zero optimizer steps"
    );
}
