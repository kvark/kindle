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

#[test]
fn context_importance_is_local_positive_mean_one_and_lane_safe() {
    let side = 16;
    let area = side * side;
    let width = 3 * area;
    let mut h = History::empty(2, width);
    for t in 0..4 {
        let mut rgb = vec![-1.; 2 * width];
        if t >= 2 {
            rgb[8 * side + if t == 2 { 4 } else { 6 }] = 1.;
        }
        h.push(&rgb, &[if t == 0 { -1 } else { 0 }; 2]);
    }
    let weights = h.motion_importance(0, side, 100.);
    let mean = (99. * 101. + 157.) / 256.;
    assert_eq!(weights.len(), width);
    assert!(weights.iter().all(|v| v.is_finite() && *v > 0.));
    assert!((weights.iter().copied().map(f64::from).sum::<f64>() / width as f64 - 1.).abs() < 2e-6);
    for c in 0..3 {
        assert!((weights[c * area + 8 * side + 10] - 101. / mean).abs() < 1e-6);
        assert!((weights[c * area + 8 * side + 11] - 1. / mean).abs() < 1e-6);
        assert_eq!(&weights[..area], &weights[c * area..(c + 1) * area]);
    }
    assert_eq!(h.motion_importance(1, side, 100.), vec![1.; width]);
    h.rgb[..2 * 2 * width].fill(0.9);
    assert_eq!(h.motion_importance(0, side, 100.), weights);
    assert_eq!(h.motion_importance(0, side, 0.), vec![1.; width]);
    h.push(&vec![0.7; 2 * width], &[-1, 0]);
    assert_eq!(h.motion_importance(0, side, 100.), vec![1.; width]);
}

#[test]
fn context_motion_headers_preserve_v1_and_require_v2_for_weighting() {
    let mut c = DenoisingConfig::default();
    assert!(
        !serde_json::to_value(&c)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("context_motion_scale")
    );
    let mut v = serde_json::json!({"format":"kindle-denoising-world-v1","config":c,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":"edm-euler3-sigma5-quantized","observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
    assert!(read_header(&serde_json::to_vec(&v).unwrap()).is_ok());
    c.context_motion_scale = 100.;
    assert!(c.validate().is_ok());
    v["config"] = serde_json::to_value(&c).unwrap();
    assert!(read_header(&serde_json::to_vec(&v).unwrap()).is_err());
    v["format"] = "kindle-denoising-world-v2".into();
    assert_eq!(read_header(&serde_json::to_vec(&v).unwrap()).unwrap().0, c);
    v["config"]
        .as_object_mut()
        .unwrap()
        .remove("context_motion_scale");
    assert!(read_header(&serde_json::to_vec(&v).unwrap()).is_err());
    for scale in [-1., f32::NAN, f32::INFINITY, 1001.] {
        c.context_motion_scale = scale;
        assert!(c.validate().is_err());
    }
}

#[test]
fn motion_importance_changes_only_image_loss_and_keeps_inference_and_weights() {
    let c = DenoisingConfig::default();
    let mut weighted = c.clone();
    weighted.context_motion_scale = 100.;
    assert_eq!(
        params(&network::training(&c)),
        params(&network::training(&weighted))
    );
    for build in [network::sampling, network::event_sampling] {
        assert_eq!(
            format!("{:?}", build(&c).nodes()),
            format!("{:?}", build(&weighted).nodes())
        );
    }
    let g = network::training(&weighted);
    let id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op,Op::Input{name} if name=="pixel_importance"))
        .unwrap()
        .id;
    assert!(ancestors(&g, g.outputs()[1]).contains(&id));
    assert!(!ancestors(&g, g.outputs()[2]).contains(&id));
    assert!(!ancestors(&g, g.outputs()[3]).contains(&id));
}

fn weighted_fixture() -> (Graph, meganeura::reference::Feeds, Vec<f32>, Vec<f64>, f64) {
    let (mut g, mut feeds, x, mut grad, _) = mse_fixture();
    let target = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op,Op::Input{name} if name=="target"))
        .unwrap()
        .id;
    let row = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op,Op::Input{name} if name=="mask"))
        .unwrap()
        .id;
    let prediction = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op,Op::Parameter{name} if name=="prediction"))
        .unwrap()
        .id;
    let importance = g.input("importance", &[16]);
    let loss = network::weighted_mse(&mut g, prediction, target, row, importance, 2, 8);
    g.set_outputs(vec![loss]);
    let weights = (0..16)
        .map(|i| if i % 2 == 0 { 0.5 } else { 1.5 })
        .collect::<Vec<_>>();
    feeds.set("importance", &weights);
    for (v, &w) in grad.iter_mut().zip(&weights) {
        *v *= f64::from(w);
    }
    let y = feeds.f32("target").unwrap();
    let loss = x[..8]
        .iter()
        .zip(&y)
        .zip(&weights)
        .map(|((&a, &b), &w)| (f64::from(a) - f64::from(b)).powi(2) * f64::from(w) / 8.)
        .sum();
    (g, feeds, x, grad, loss)
}
#[test]
fn context_weighted_loss_and_derivatives_match_independent_f64() {
    let (g, feeds, _, grad, loss) = weighted_fixture();
    let g = meganeura::autodiff::differentiate(&g);
    let values = meganeura::reference::evaluate(&g, &feeds).unwrap();
    assert!((values[g.outputs()[0] as usize].data[0] - loss).abs() < 1e-10);
    for (&a, &b) in values[g.outputs()[1] as usize].data.iter().zip(&grad) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
#[ignore = "requires guarded GPU: positive pixel importance and masked derivatives"]
fn context_weighted_native_loss_matches_f64() {
    let (g, feeds, x, grad, loss) = weighted_fixture();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session = runtime::build_session(&g, &gpu, Mode::Training, false);
    session.set_parameter("prediction", &x);
    for name in ["target", "mask", "importance"] {
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
    let p = std::env::var("KINDLE_CONTEXT_NATIVE_RECEIPT").unwrap();
    std::fs::write(p,serde_json::to_vec_pretty(&serde_json::json!({"status":"complete","loss":actual[0],"gradient":got,"values_and_derivatives":true,"no_optimizer_update":true})).unwrap()).unwrap();
}

#[test]
fn residual_coordinates_recover_full_range_rgb_and_preserve_legacy_math() {
    for sigma in [0.002, 0.05, 0.5, 1., 5.] {
        let (_, cout, skip, _) = conditioners(sigma);
        for base in [-1., -0.6, 0., 0.8, 1.] {
            for byte in [0, 17, 64, 128, 211, 255] {
                let y = byte as f32 / 127.5 - 1.;
                let delta = coordinate(y, base, true);
                assert!((f64::from(delta) - (f64::from(y) - f64::from(base))).abs() < 2e-7);
                let x = delta + sigma * 0.37;
                let target = (delta - skip * x) / cout;
                assert!((absolute(skip * x + cout * target, base, true) - y).abs() < 2e-6);
                assert_eq!(coordinate(y, base, false).to_bits(), y.to_bits());
                assert_eq!(absolute(y, base, false), quantize(y));
            }
        }
    }
    // A full bright-to-dark transition needs residuals outside [-1,1].
    let base = 1.;
    let target = -1.;
    let mut x = 3.7;
    for pair in schedule().windows(2) {
        let prediction = coordinate(target, base, true);
        x += (x - prediction) / pair[0] * (pair[1] - pair[0]);
    }
    assert!(x < -1.9);
    assert_eq!(absolute(x, base, true), target);
    assert_eq!(absolute(100., base, true), 1.);
    assert_eq!(absolute(-100., base, true), -1.);
}

#[test]
fn residual_base_uses_only_latest_past_frame_and_respects_reset_lanes() {
    let mut history = History::empty(2, 3);
    for t in 0..4 {
        history.push(&[t as f32, 0., 1., 10. + t as f32, 2., 3.], &[0, 1]);
    }
    assert_eq!(history.base(&[2, 3], true), [3., 0., 1., 13., 2., 3.]);
    assert_eq!(history.base(&[-1, 3], true), [0., 0., 0., 13., 2., 3.]);
    assert_eq!(history.base(&[2, 3], false), [0.; 6]);
    history.rgb[..18].fill(-99.);
    assert_eq!(history.base(&[2, 3], true), [3., 0., 1., 13., 2., 3.]);
}

#[test]
fn residual_mode_changes_no_differentiated_or_inference_graph() {
    let c = DenoisingConfig {
        context_motion_scale: 100.,
        ..DenoisingConfig::default()
    };
    let mut residual = c.clone();
    residual.residual_frame = true;
    for build in [
        network::training,
        network::sampling,
        network::event_sampling,
    ] {
        assert_eq!(
            format!("{:?}", build(&c).nodes()),
            format!("{:?}", build(&residual).nodes())
        );
    }
    let p = params(&network::training(&residual));
    assert_eq!(p.len(), 45);
    assert_eq!(
        p.values()
            .map(|shape| shape.iter().product::<usize>())
            .sum::<usize>(),
        1_787_843
    );
}

#[test]
fn residual_header_requires_v3_and_its_sampler_and_preserves_old_defaults() {
    let mut c = DenoisingConfig::default();
    assert!(
        !serde_json::to_value(&c)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("residual_frame")
    );
    c.residual_frame = true;
    for scale in [0., 100.] {
        c.context_motion_scale = scale;
        let v = serde_json::json!({"format":c.checkpoint_format(),"config":c,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":c.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
        assert_eq!(read_header(&serde_json::to_vec(&v).unwrap()).unwrap().0, c);
        for version in ["kindle-denoising-world-v1", "kindle-denoising-world-v2"] {
            let mut bad = v.clone();
            bad["format"] = version.into();
            assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        let mut bad = v.clone();
        bad["sampler"] = "edm-euler3-sigma5-quantized".into();
        assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
        let mut bad = v;
        bad["config"]["residual_frame"] = false.into();
        assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
}

#[test]
fn half_max_sigmas_preserve_legacy_values_and_balance_every_repeated_clip() {
    let normals = (0..32).map(|i| (i % 4) as f32 - 2.).collect::<Vec<_>>();
    let expected = normals
        .iter()
        .map(|&n| (-1.2 + 1.2 * n).exp().clamp(SIGMA_MIN, SIGMA_MAX))
        .collect::<Vec<_>>();
    for update in [0, 1, 19, u64::MAX] {
        let plain = training_sigmas(normals.clone(), update, false);
        for (a, b) in plain.iter().zip(&expected) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let high = training_sigmas(normals.clone(), update, true);
        assert_eq!(high.iter().filter(|&&v| v == SIGMA_MAX).count(), 16);
        for lane in 0..8 {
            let copies = (0..4).map(|copy| lane + copy * 8).collect::<Vec<_>>();
            assert_eq!(copies.iter().filter(|&&i| high[i] == SIGMA_MAX).count(), 2);
            for i in copies {
                assert_eq!(
                    high[i],
                    if (i < 16) == update.is_multiple_of(2) {
                        SIGMA_MAX
                    } else {
                        expected[i]
                    }
                );
            }
        }
    }
    // Alternating halves balance odd batches over two updates too.
    let first = training_sigmas(vec![0.; 5], 0, true);
    let second = training_sigmas(vec![0.; 5], 1, true);
    for i in 0..5 {
        assert_ne!(first[i] == SIGMA_MAX, second[i] == SIGMA_MAX);
    }
    let mut a = StdRng::seed_from_u64(41001);
    let mut b = a.clone();
    let _ = training_sigmas(gaussian::normals(&mut a, 32), 0, true);
    let _ = training_sigmas(gaussian::normals(&mut b, 32), 0, false);
    assert_eq!(
        gaussian::normals(&mut a, 512),
        gaussian::normals(&mut b, 512)
    );
}

#[test]
fn half_max_requires_v4_preserves_legacy_defaults_and_all_graphs() {
    let mut c = DenoisingConfig::default();
    assert!(
        !serde_json::to_value(&c)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("half_max_noise")
    );
    for residual in [false, true] {
        c.residual_frame = residual;
        c.context_motion_scale = 100.;
        let mut half = c.clone();
        half.half_max_noise = true;
        for build in [
            network::training,
            network::sampling,
            network::event_sampling,
        ] {
            assert_eq!(
                format!("{:?}", build(&c).nodes()),
                format!("{:?}", build(&half).nodes())
            );
        }
        let mut v = serde_json::json!({"format":"kindle-denoising-world-v4","config":half,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":half.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
        assert_eq!(
            read_header(&serde_json::to_vec(&v).unwrap()).unwrap().0,
            half
        );
        for version in [
            "kindle-denoising-world-v1",
            "kindle-denoising-world-v2",
            "kindle-denoising-world-v3",
        ] {
            let mut bad = v.clone();
            bad["format"] = version.into();
            assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        v["config"]["half_max_noise"] = false.into();
        assert!(read_header(&serde_json::to_vec(&v).unwrap()).is_err());
    }
}

mod event_logits;
mod sampler_steps;
mod sampler_trace;

#[test]
fn configurable_sigma_data_matches_f64_and_recovers_rgb_without_rescaling_history() {
    for scale in [0.01, 0.05, 0.1, 0.5, 2.] {
        let c = DenoisingConfig {
            sigma_data: scale,
            ..DenoisingConfig::default()
        };
        assert!(c.validate().is_ok());
        for sigma in [0.002, 0.05, 0.28308183, 0.5, 5.] {
            let (cin, cout, skip, noise) = c.coefficients(sigma);
            let sf = f64::from(sigma);
            let df = f64::from(scale);
            let variance = sf * sf + df * df;
            for (actual, expected) in [
                (cin, 1. / variance.sqrt()),
                (cout, sf * df / variance.sqrt()),
                (skip, df * df / variance),
                (noise, sf.ln() / 4.),
            ] {
                assert!((f64::from(actual) - expected).abs() <= 2e-6 * expected.abs().max(1.));
            }
            for base in [-1., 0., 1.] {
                for byte in [0, 19, 128, 221, 255] {
                    let y = byte as f32 / 127.5 - 1.;
                    let delta = y - base;
                    let x = delta + sigma * 0.37;
                    let target = (delta - skip * x) / cout;
                    assert!((absolute(skip * x + cout * target, base, true) - y).abs() < 2e-6);
                }
            }
        }
    }
    let c = DenoisingConfig::default();
    for sigma in [0.002, 0.05, 0.5, 1., 5.] {
        assert_eq!(c.coefficients(sigma), conditioners(sigma));
    }
    let mut h = History::empty(1, 3);
    for _ in 0..4 {
        h.push(&[0.25, -0.5, 1.], &[0]);
    }
    // Conditioning stays at the original .5 scale, independent of noise-data scale.
    assert_eq!(&h.condition(0, 0, 1, true)[..3], &[0.5, -1., 2.]);
}

#[test]
fn sigma_data_requires_v5_and_changes_no_graph_or_default_header() {
    let mut c = DenoisingConfig::default();
    assert!(
        !serde_json::to_value(&c)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("sigma_data")
    );
    for bad in [0., 0.009, 2.01, f32::NAN, f32::INFINITY] {
        c.sigma_data = bad;
        assert!(c.validate().is_err());
    }
    c = DenoisingConfig::default();
    c.context_motion_scale = 100.;
    c.residual_frame = true;
    c.half_max_noise = true;
    let mut small = c.clone();
    small.sigma_data = 0.05;
    for build in [
        network::training,
        network::sampling,
        network::event_sampling,
    ] {
        assert_eq!(
            format!("{:?}", build(&c).nodes()),
            format!("{:?}", build(&small).nodes())
        );
    }
    let v = serde_json::json!({"format":"kindle-denoising-world-v5","config":small,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":small.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
    assert_eq!(
        read_header(&serde_json::to_vec(&v).unwrap()).unwrap().0,
        small
    );
    for version in ["v1", "v2", "v3", "v4"] {
        let mut bad = v.clone();
        bad["format"] = format!("kindle-denoising-world-{version}").into();
        assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    let mut bad = v;
    bad["config"]["sigma_data"] = 0.5.into();
    assert!(read_header(&serde_json::to_vec(&bad).unwrap()).is_err());
}

#[test]
fn log_noise_mean_matches_f64_and_preserves_default_sigma_math() {
    let normals = (0..32).map(|i| (i % 8) as f32 - 4.).collect::<Vec<_>>();
    for half in [false, true] {
        for update in [0, 1] {
            let default = DenoisingConfig {
                half_max_noise: half,
                ..DenoisingConfig::default()
            };
            assert_eq!(
                default.training_sigmas(normals.clone(), update),
                training_sigmas(normals.clone(), update, half)
            );
            for mean in [-6., -3., -1.2, 1.] {
                let c = DenoisingConfig {
                    log_noise_mean: mean,
                    ..default.clone()
                };
                let actual = c.training_sigmas(normals.clone(), update);
                for (lane, (&value, &normal)) in actual.iter().zip(&normals).enumerate() {
                    let forced = half && (lane < 16) == update.is_multiple_of(2);
                    let expected = if forced {
                        5.
                    } else {
                        (f64::from(mean) + 1.2 * f64::from(normal))
                            .exp()
                            .clamp(0.002, 5.)
                    };
                    assert!((f64::from(value) - expected).abs() <= 3e-6 * expected);
                    if forced {
                        assert_eq!(value, SIGMA_MAX);
                    }
                }
            }
        }
    }
}

#[test]
fn log_noise_mean_keeps_rng_forced_half_and_all_graphs_unchanged() {
    let control = DenoisingConfig {
        sigma_data: 0.05,
        half_max_noise: true,
        context_motion_scale: 100.,
        residual_frame: true,
        ..DenoisingConfig::default()
    };
    let changed = DenoisingConfig {
        log_noise_mean: -3.,
        ..control.clone()
    };
    for update in [0, 1] {
        let mut a = StdRng::seed_from_u64(41001);
        let mut b = StdRng::seed_from_u64(41001);
        let old = control.training_sigmas(gaussian::normals(&mut a, 32), update);
        let new = changed.training_sigmas(gaussian::normals(&mut b, 32), update);
        for lane in 0..32 {
            if (lane < 16) == update.is_multiple_of(2) {
                assert_eq!((old[lane], new[lane]), (5., 5.));
            }
        }
        assert_eq!(gaussian::normals(&mut a, 96), gaussian::normals(&mut b, 96));
    }
    for build in [
        network::training,
        network::sampling,
        network::event_sampling,
    ] {
        assert_eq!(
            format!("{:?}", build(&control).nodes()),
            format!("{:?}", build(&changed).nodes())
        );
    }
    assert_eq!(
        params(&network::training(&control)),
        params(&network::training(&changed))
    );
}

#[test]
fn log_noise_mean_requires_v6_and_preserves_legacy_headers() {
    for (scale, residual, half, data, version) in [
        (0., false, false, 0.5, "v1"),
        (100., false, false, 0.5, "v2"),
        (100., true, false, 0.5, "v3"),
        (100., true, true, 0.5, "v4"),
        (100., true, true, 0.05, "v5"),
    ] {
        let c = DenoisingConfig {
            context_motion_scale: scale,
            residual_frame: residual,
            half_max_noise: half,
            sigma_data: data,
            ..DenoisingConfig::default()
        };
        assert!(
            !serde_json::to_value(&c)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("log_noise_mean")
        );
        assert_eq!(
            c.checkpoint_format(),
            format!("kindle-denoising-world-{version}")
        );
        let changed = DenoisingConfig {
            log_noise_mean: -3.,
            ..c.clone()
        };
        let header = |config: &DenoisingConfig, format: &str| serde_json::json!({"format":format,"config":config,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":config.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"});
        let bytes = |v: serde_json::Value| serde_json::to_vec(&v).unwrap();
        assert_eq!(
            read_header(&bytes(header(&c, c.checkpoint_format())))
                .unwrap()
                .0,
            c
        );
        assert_eq!(changed.checkpoint_format(), "kindle-denoising-world-v6");
        assert_eq!(
            read_header(&bytes(header(&changed, changed.checkpoint_format())))
                .unwrap()
                .0,
            changed
        );
        assert!(read_header(&bytes(header(&changed, c.checkpoint_format()))).is_err());
        assert!(read_header(&bytes(header(&c, "kindle-denoising-world-v6"))).is_err());
    }
    for mean in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -6.01, 1.01] {
        assert!(
            DenoisingConfig {
                log_noise_mean: mean,
                ..DenoisingConfig::default()
            }
            .validate()
            .is_err()
        );
    }
    for mean in [-6., 1.] {
        assert!(
            DenoisingConfig {
                log_noise_mean: mean,
                ..DenoisingConfig::default()
            }
            .validate()
            .is_ok()
        );
    }
}

#[test]
fn bounded_reward_support_matches_f64_and_roundtrips_targets() {
    let c = DenoisingConfig {
        reward_limit: Some(32.),
        ..DenoisingConfig::default()
    };
    let bins = c.reward_bins();
    let values = bins.values();
    assert_eq!((values[0], values[127], values[254]), (-32., 0., 32.));
    assert!(values.windows(2).all(|v| v[0] < v[1]));
    for i in 0..255 {
        assert_eq!(values[i], -values[254 - i]);
        let coordinate = -33f64.ln() + 33f64.ln() * i as f64 / 127.;
        let expected = coordinate.signum() * coordinate.abs().exp_m1();
        assert!((f64::from(values[i]) - expected).abs() <= 3e-6 * expected.abs().max(1.));
    }
    for target in [-32., -7., -1., 0., 0.5, 1., 4., 7., 28., 32.] {
        let mut encoded = vec![0.; 255];
        bins.encode(target, &mut encoded);
        let mean = encoded
            .iter()
            .zip(values)
            .map(|(&p, &v)| f64::from(p) * f64::from(v))
            .sum::<f64>();
        assert!((mean - f64::from(target)).abs() <= 3e-6 * f64::from(target).abs().max(1.));
        let logits = encoded
            .iter()
            .map(|&p| if p == 0. { -1000. } else { p.ln() })
            .collect::<Vec<_>>();
        let decoded = bins.decode_logits_with_scratch(&logits, &mut vec![0.; 255]);
        assert!((decoded - target).abs() <= 3e-6 * target.abs().max(1.));
    }
    assert_eq!(
        DenoisingConfig::default().reward_bins().values(),
        TwoHotBins::new(255).values()
    );
}

#[test]
fn reward_limit_rejects_invalid_targets_and_changes_no_graph() {
    let control = DenoisingConfig {
        sigma_data: 0.05,
        context_motion_scale: 100.,
        residual_frame: true,
        half_max_noise: true,
        ..DenoisingConfig::default()
    };
    let changed = DenoisingConfig {
        reward_limit: Some(32.),
        ..control.clone()
    };
    assert!(changed.valid_rewards(&[-32., 0., 1., 32.]));
    assert!(!changed.valid_rewards(&[32.001]));
    assert!(!changed.valid_rewards(&[-32.001]));
    assert!(!changed.valid_rewards(&[f32::NAN]));
    for limit in [0., -1., 0.009, 1000001., f32::NAN, f32::INFINITY] {
        assert!(
            DenoisingConfig {
                reward_limit: Some(limit),
                ..control.clone()
            }
            .validate()
            .is_err()
        );
    }
    for build in [
        network::training,
        network::sampling,
        network::event_sampling,
    ] {
        assert_eq!(
            format!("{:?}", build(&control).nodes()),
            format!("{:?}", build(&changed).nodes())
        );
    }
    assert_eq!(
        params(&network::training(&control)),
        params(&network::training(&changed))
    );
}

#[test]
fn bounded_reward_header_requires_v7_and_preserves_all_defaults() {
    for (scale, residual, half, data, mean, version) in [
        (0., false, false, 0.5, -1.2, "v1"),
        (100., false, false, 0.5, -1.2, "v2"),
        (100., true, false, 0.5, -1.2, "v3"),
        (100., true, true, 0.5, -1.2, "v4"),
        (100., true, true, 0.05, -1.2, "v5"),
        (100., true, true, 0.05, -3., "v6"),
    ] {
        let c = DenoisingConfig {
            context_motion_scale: scale,
            residual_frame: residual,
            half_max_noise: half,
            sigma_data: data,
            log_noise_mean: mean,
            ..DenoisingConfig::default()
        };
        assert!(
            !serde_json::to_value(&c)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("reward_limit")
        );
        assert_eq!(
            c.checkpoint_format(),
            format!("kindle-denoising-world-{version}")
        );
        let changed = DenoisingConfig {
            reward_limit: Some(32.),
            ..c.clone()
        };
        let header = |config: &DenoisingConfig, format: &str| {
            serde_json::to_vec(&serde_json::json!({"format":format,"config":config,"updates":0,"architecture":"conditional-unet16-32-64-history4","sampler":config.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"})).unwrap()
        };
        assert_eq!(changed.checkpoint_format(), "kindle-denoising-world-v7");
        assert_eq!(
            read_header(&header(&c, c.checkpoint_format())).unwrap().0,
            c
        );
        assert_eq!(
            read_header(&header(&changed, changed.checkpoint_format()))
                .unwrap()
                .0,
            changed
        );
        assert!(read_header(&header(&changed, c.checkpoint_format())).is_err());
        assert!(read_header(&header(&c, "kindle-denoising-world-v7")).is_err());
    }
}
