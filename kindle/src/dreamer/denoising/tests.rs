use super::*;
use meganeura::graph::Op;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn denoising_configuration_rejects_bad_geometry_before_gpu() {
    let c = DenoisingConfig::default();
    assert!(c.validate().is_ok());
    for field in [
        "action_count",
        "batch_size",
        "train_batch_size",
        "side",
        "value_bins",
        "horizon",
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        v[field] = 0.into();
        let bad: DenoisingConfig = serde_json::from_value(v).unwrap();
        assert!(DenoisingWorldLearner::new(bad).is_err(), "{field}");
    }
    let mut bad = c.clone();
    bad.value_bins = 4;
    assert!(bad.validate().is_err());
    bad = c.clone();
    bad.learning_rate = f32::NAN;
    assert!(bad.validate().is_err());
    let mut v = serde_json::to_value(&c).unwrap();
    v["future_target_in_condition"] = true.into();
    assert!(serde_json::from_value::<DenoisingConfig>(v).is_err());
}

#[test]
fn edm_preconditioning_recovers_targets_and_has_three_finite_euler_steps() {
    let s = schedule();
    assert_eq!(s[0], 5.);
    assert_eq!(s[2], 0.002);
    assert_eq!(s[3], 0.);
    assert!(s.windows(2).all(|p| p[0] > p[1]));
    for sigma in [0.002, 0.05, 0.5, 1., 5.] {
        let (cin, cout, skip, noise) = conditioners(sigma);
        let sf = f64::from(sigma);
        let variance = sf * sf + 0.25;
        for (a, b) in [
            (cin, 1. / variance.sqrt()),
            (cout, sf * 0.5 / variance.sqrt()),
            (skip, 0.25 / variance),
            (noise, sf.ln() / 4.),
        ] {
            assert!((f64::from(a) - b).abs() < 2e-7);
        }
        for y in [-1., -0.3, 0., 0.8, 1.] {
            let x = y + sigma * 0.37;
            let target = (y - skip * x) / cout;
            assert!((skip * x + cout * target - y).abs() < 2e-6);
        }
    }
    for byte in 0..=255 {
        let x = byte as f32 / 127.5 - 1.;
        assert!((quantize(x) - x).abs() < 1e-6);
    }
    assert_eq!(quantize(-100.), -1.);
    assert_eq!(quantize(100.), 1.);
}

#[test]
fn history_preserves_lane_action_alignment_and_clears_only_reset_lane() {
    let mut h = History::empty(2, 3);
    for t in 0..4 {
        h.push(&[t as f32; 6], &[t, t + 1]);
    }
    let c = h.condition(0, 4, 6, false);
    assert_eq!(&c[..12], &[0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    for (t, action) in [1, 2, 3, 4].into_iter().enumerate() {
        for k in 0..6 {
            assert_eq!(c[12 + t * 6 + k], if k == action { 1. } else { 0. });
        }
    }
    assert!(h.condition(0, -1, 6, false).iter().all(|&v| v == 0.));
    h.push(&[9., 9., 9., 8., 8., 8.], &[-1, 4]);
    for t in 0..3 {
        assert_eq!(&h.rgb[(t * 2) * 3..(t * 2 + 1) * 3], &[0.; 3]);
        assert_eq!(h.actions[t * 2], -1);
        assert_eq!(
            &h.rgb[(t * 2 + 1) * 3..(t * 2 + 2) * 3],
            &[(t + 1) as f32; 3]
        );
    }
    assert_eq!(&h.rgb[18..21], &[9.; 3]);
}

fn params(g: &Graph) -> BTreeMap<String, Vec<usize>> {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
        .map(|n| {
            (
                match &n.op {
                    Op::Parameter { name } => name.clone(),
                    _ => unreachable!(),
                },
                n.ty.shape.clone(),
            )
        })
        .collect()
}
fn ancestors(g: &Graph, start: NodeId) -> BTreeSet<NodeId> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![start];
    while let Some(id) = todo.pop() {
        if seen.insert(id) {
            todo.extend(&g.node(id).inputs);
        }
    }
    seen
}

#[test]
fn denoising_graphs_share_exact_weights_and_keep_image_and_event_losses_separate() {
    let c = DenoisingConfig::default();
    let g = network::training(&c);
    let sample = network::sampling(&c);
    let event = network::event_sampling(&c);
    assert_eq!(g.outputs().len(), 5);
    assert_eq!(g.node(g.outputs()[4]).ty.num_elements(), 32 * 3 * 64 * 64);
    assert_eq!(
        sample.node(sample.outputs()[0]).ty.num_elements(),
        8 * 3 * 64 * 64
    );
    assert_eq!(event.node(event.outputs()[0]).ty.num_elements(), 8 * 255);
    let p = params(&g);
    let mut both = params(&sample);
    both.extend(params(&event));
    assert_eq!(p, both);
    assert!(p.len() > 30);
    let image = ancestors(&g, g.outputs()[1]);
    let reward = ancestors(&g, g.outputs()[2]);
    let ending = ancestors(&g, g.outputs()[3]);
    for n in g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Parameter { .. }))
    {
        let denoiser = match &n.op {
            Op::Parameter { name } => name.starts_with("pixel.denoiser."),
            _ => unreachable!(),
        };
        assert_eq!(image.contains(&n.id), denoiser);
        assert_eq!(reward.contains(&n.id) || ending.contains(&n.id), !denoiser);
    }
    let sampled = ancestors(&sample, sample.outputs()[0]);
    for n in sample
        .nodes()
        .iter()
        .filter(|n| sampled.contains(&n.id) && matches!(n.op, Op::Input { .. }))
    {
        assert!(matches!(&n.op,Op::Input{name} if name=="denoiser_input"));
    }
}

fn mse_fixture() -> (Graph, meganeura::reference::Feeds, Vec<f32>, Vec<f64>, f64) {
    let mut g = Graph::new();
    let x = g.parameter("prediction", &[16]);
    let y = g.input("target", &[16]);
    let mask = g.input("mask", &[2, 1]);
    let loss = network::masked_mse(&mut g, x, y, mask, 2, 8);
    g.set_outputs(vec![loss]);
    let x = (0..16).map(|i| i as f32 * 0.25 - 0.75).collect::<Vec<_>>();
    let y = (0..16).map(|i| i as f32 * 0.125).collect::<Vec<_>>();
    let grad = x
        .iter()
        .zip(&y)
        .enumerate()
        .map(|(i, (&a, &b))| {
            if i < 8 {
                2. * (f64::from(a) - f64::from(b)) / 8.
            } else {
                0.
            }
        })
        .collect::<Vec<_>>();
    let loss = x[..8]
        .iter()
        .zip(&y)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2) / 8.)
        .sum();
    let mut feeds = meganeura::reference::Feeds::new();
    feeds
        .set("prediction", &x)
        .set("target", &y)
        .set("mask", &[2., 0.]);
    (g, feeds, x, grad, loss)
}
#[test]
fn masked_denoising_loss_and_derivative_match_f64_with_zero_reset_gradient() {
    let (g, feeds, _, grad, loss) = mse_fixture();
    let g = meganeura::autodiff::differentiate(&g);
    let values = meganeura::reference::evaluate(&g, &feeds).unwrap();
    assert!((values[g.outputs()[0] as usize].data[0] - loss).abs() < 1e-10);
    for (&a, &b) in values[g.outputs()[1] as usize].data.iter().zip(&grad) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn checkpoint_header_rejects_disguised_legacy_and_unknown_settings() {
    let v = serde_json::json!({"format":"kindle-denoising-world-v1","config":DenoisingConfig::default(),"updates":42,"architecture":"conditional-unet16-32-64-history4","sampler":"edm-euler3-sigma5-quantized","observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
    assert_eq!(read_header(&serde_json::to_vec(&v).unwrap()).unwrap().1, 42);
    for k in [
        "format",
        "architecture",
        "sampler",
        "observed_reconstruction",
    ] {
        let mut bad = v.clone();
        bad[k] = "legacy".into();
        assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    let mut bad = v.clone();
    bad["seeded_rng_restored"] = true.into();
    assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
    let mut bad = v;
    bad["updates"] = serde_json::json!(u64::MAX);
    assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
}
#[test]
#[ignore = "requires guarded GPU: denoising mask/loss native values and derivatives"]
fn denoising_native_loss_matches_f64() {
    let (g, feeds, x, grad, loss) = mse_fixture();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session = runtime::build_session(&g, &gpu, Mode::Training, false);
    session.set_parameter("prediction", &x);
    for name in ["target", "mask"] {
        session.set_input(name, &feeds.f32(name).unwrap());
    }
    session.step();
    session.wait();
    let mut actual = [0.];
    session.read_output_by_index(0, &mut actual);
    assert!((f64::from(actual[0]) - loss).abs() < 1e-6);
    let mut got = [0.; 16];
    session.read_param_grad("prediction", &mut got);
    for (&a, &b) in got.iter().zip(&grad) {
        assert!((f64::from(a) - b).abs() < 1e-6);
    }
    if let Ok(path) = std::env::var("KINDLE_DENOISING_NATIVE_RECEIPT") {
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"status":"complete","loss":actual[0],"gradients":got,"values_and_derivatives":true,"no_optimizer_update":true})).unwrap()).unwrap();
    }
}
