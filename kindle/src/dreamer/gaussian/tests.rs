use super::*;
use crate::dreamer::{ObservationKind, PixelLoss, WorldModelLearner, runtime};
use meganeura::{
    Mode,
    reference::{Feeds, evaluate},
};
use rand::SeedableRng;
use std::sync::Arc;

fn scale(x: f64) -> f64 {
    f64::from(STD_MIN) + 1.0 / (1.0 + (-x).exp())
}
fn reference(
    q: &[f64],
    p: &[f64],
    noise: &[f64],
    batch: usize,
    groups: usize,
    width: usize,
    floor: f64,
) -> [f64; 3] {
    let mut states = vec![0.0; batch];
    let mut sample = 0.0;
    for (i, (a, b)) in q
        .chunks_exact(2 * width)
        .zip(p.chunks_exact(2 * width))
        .enumerate()
    {
        for j in 0..width {
            let (qs, ps) = (scale(a[width + j]), scale(b[width + j]));
            states[i / groups] +=
                (ps / qs).ln() + (qs * qs + (a[j] - b[j]).powi(2)) / (2.0 * ps * ps) - 0.5;
            sample += a[j] + qs * noise[i * width + j];
        }
    }
    let raw = states.iter().sum::<f64>() / batch as f64;
    let loss = states.iter().map(|v| v.max(floor)).sum::<f64>() / batch as f64;
    let sample = sample / (batch * groups * width) as f64;
    [loss + 0.3 * sample, raw, sample]
}
struct Fixture {
    graph: Graph,
    feeds: Feeds,
    q: Vec<f32>,
    p: Vec<f32>,
    noise: Vec<f32>,
    expected: [f64; 3],
    samples: Vec<f64>,
    grad: [Vec<f64>; 2],
}
fn fixture(
    batch: usize,
    groups: usize,
    width: usize,
    stop_q: bool,
    stop_p: bool,
    floor: f32,
) -> Fixture {
    let count = batch * groups * 2 * width;
    let q = (0..count)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.125)
        .collect::<Vec<_>>();
    let p = (0..count)
        .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.125)
        .collect::<Vec<_>>();
    let noise = (0..count / 2)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25)
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let qnode = graph.parameter("q", &[batch * groups, 2 * width]);
    let pnode = graph.parameter("p", &[batch * groups, 2 * width]);
    let n = graph.input("noise", &[batch * groups, width]);
    let sample_values = sample(&mut graph, qnode, n, batch * groups, width);
    let sampled = graph.mean_all(sample_values);
    let k = kl(
        &mut graph, qnode, pnode, batch, groups, width, stop_q, stop_p, floor,
    );
    let weighted = graph.scale(sampled, 0.3);
    let total = graph.add(k.loss, weighted);
    graph.set_outputs(vec![total, k.raw, sampled, sample_values]);
    let mut feeds = Feeds::new();
    feeds.set("q", &q).set("p", &p).set("noise", &noise);
    let (q64, p64, n64) = (
        q.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        p.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        noise.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
    );
    let expected = reference(&q64, &p64, &n64, batch, groups, width, f64::from(floor));
    let samples = q64
        .chunks_exact(2 * width)
        .enumerate()
        .flat_map(|(row, q)| {
            (0..width)
                .map(|j| q[j] + scale(q[width + j]) * n64[row * width + j])
                .collect::<Vec<_>>()
        })
        .collect();
    let value = |a: &[f64], b: &[f64]| {
        let k = reference(
            if stop_q { &q64 } else { a },
            if stop_p { &p64 } else { b },
            &n64,
            batch,
            groups,
            width,
            f64::from(floor),
        );
        let sample = reference(a, b, &n64, batch, groups, width, f64::from(floor))[2];
        k[0] - 0.3 * k[2] + 0.3 * sample
    };
    let mut grad = [vec![0.0; count], vec![0.0; count]];
    if count > 128 {
        let mut states = vec![0.0; batch];
        for (row, (a, b)) in q64
            .chunks_exact(2 * width)
            .zip(p64.chunks_exact(2 * width))
            .enumerate()
        {
            for j in 0..width {
                let (qs, ps) = (scale(a[width + j]), scale(b[width + j]));
                states[row / groups] +=
                    (ps / qs).ln() + (qs * qs + (a[j] - b[j]).powi(2)) / (2.0 * ps * ps) - 0.5;
            }
        }
        for (row, (a, b)) in q64
            .chunks_exact(2 * width)
            .zip(p64.chunks_exact(2 * width))
            .enumerate()
        {
            let active = f64::from(states[row / groups] > f64::from(floor)) / batch as f64;
            for j in 0..width {
                let (qs, ps) = (scale(a[width + j]), scale(b[width + j]));
                let (q_sig, p_sig) = (qs - f64::from(STD_MIN), ps - f64::from(STD_MIN));
                let (dq, dp) = (q_sig * (1.0 - q_sig), p_sig * (1.0 - p_sig));
                let delta = a[j] - b[j];
                let sample_weight = 0.3 / (batch * groups * width) as f64;
                let i = row * 2 * width + j;
                grad[0][i] = sample_weight
                    + if stop_q {
                        0.0
                    } else {
                        active * delta / (ps * ps)
                    };
                grad[0][i + width] = sample_weight * n64[row * width + j] * dq
                    + if stop_q {
                        0.0
                    } else {
                        active * (qs / (ps * ps) - 1.0 / qs) * dq
                    };
                grad[1][i] = if stop_p {
                    0.0
                } else {
                    -active * delta / (ps * ps)
                };
                grad[1][i + width] = if stop_p {
                    0.0
                } else {
                    active * (1.0 / ps - (qs * qs + delta * delta) / ps.powi(3)) * dp
                };
            }
        }
        return Fixture {
            graph,
            feeds,
            q,
            p,
            noise,
            expected,
            samples,
            grad,
        };
    }
    let step = 1e-5;
    for (side, base) in [&q64, &p64].into_iter().enumerate() {
        for i in 0..count {
            let mut a = base.clone();
            let mut b = base.clone();
            a[i] += step;
            b[i] -= step;
            grad[side][i] = if side == 0 {
                (value(&a, &p64) - value(&b, &p64)) / (2.0 * step)
            } else {
                (value(&q64, &a) - value(&q64, &b)) / (2.0 * step)
            };
        }
    }
    Fixture {
        graph,
        feeds,
        q,
        p,
        noise,
        expected,
        samples,
        grad,
    }
}
#[test]
fn gaussian_sample_and_kl_match_f64_derivatives_stops_and_free_floor() {
    for (stop_q, stop_p, floor) in [
        (false, false, 0.0),
        (true, false, 0.0),
        (false, true, 0.0),
        (false, false, 1e6),
    ] {
        let f = fixture(2, 2, 3, stop_q, stop_p, floor);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let v = evaluate(&backward, &f.feeds).unwrap();
        for (i, &x) in f.expected.iter().enumerate() {
            assert!((v[backward.outputs()[i] as usize].data[0] - x).abs() < 1e-7);
        }
        for (i, g) in f.grad.iter().enumerate() {
            for (a, b) in v[backward.outputs()[4 + i] as usize].data.iter().zip(g) {
                assert!((a - b).abs() < 2e-5, "{a} vs {b}");
            }
        }
    }
}
#[test]
fn gaussian_large_reference_derivatives_match_autodiff() {
    for (stop_q, stop_p, floor) in [
        (false, false, 0.0),
        (true, false, 0.0),
        (false, true, 0.0),
        (false, false, 1e6),
    ] {
        let f = fixture(1, 5, 16, stop_q, stop_p, floor);
        let backward = meganeura::autodiff::differentiate(&f.graph);
        let v = evaluate(&backward, &f.feeds).unwrap();
        for (i, g) in f.grad.iter().enumerate() {
            for (a, b) in v[backward.outputs()[4 + i] as usize].data.iter().zip(g) {
                assert!((a - b).abs() < 2e-8, "{a} vs {b}");
            }
        }
    }
}

#[test]
fn gaussian_normals_repeat_and_have_standard_moments() {
    let mut a = StdRng::seed_from_u64(31);
    let mut b = a.clone();
    assert_eq!(normals(&mut a, 17), normals(&mut b, 17));
    let x = normals(&mut a, 65536);
    let mean = x.iter().map(|&x| f64::from(x)).sum::<f64>() / x.len() as f64;
    let second = x.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / x.len() as f64;
    assert!(mean.abs() < 0.02 && (second - 1.0).abs() < 0.03);
    let c = DreamerConfig::tiny(3);
    let mut a = StdRng::seed_from_u64(5);
    let mut b = a.clone();
    assert_eq!(
        noise(&c, &mut a),
        (0..c.batch_size * c.network().stoch * c.network().classes)
            .map(|_| b.random_range(1e-6..1.0 - 1e-6))
            .collect::<Vec<f32>>()
    );
}
#[test]
fn gaussian_cpu_sampling_uses_both_halves_and_ignores_categorical_mixing() {
    let mut c = config();
    let width = c.network().classes;
    let count = c.batch_size * c.network().stoch;
    let stats = (0..count * 2 * width)
        .map(|i| i as f32 * 0.01 - 0.5)
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(17);
    let n = normals(&mut rng, count * width);
    let mut rng = StdRng::seed_from_u64(17);
    let a = sample_cpu(&c, &stats, &mut rng);
    let expected = stats
        .chunks_exact(2 * width)
        .flat_map(|row| {
            (0..width)
                .map(|i| (row[i], row[width + i]))
                .collect::<Vec<_>>()
        })
        .zip(n)
        .map(|((m, s), n)| f64::from(m) + scale(f64::from(s)) * f64::from(n))
        .collect::<Vec<_>>();
    for (x, y) in a.iter().zip(expected) {
        assert!((f64::from(*x) - y).abs() < 1e-6);
    }
    c.unimix = 0.9;
    assert_eq!(a, sample_cpu(&c, &stats, &mut StdRng::seed_from_u64(17)));
}
fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.world_gaussian = true;
    c.batch_size = 2;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.0;
    c.loss_scales.value = 0.0;
    c.loss_scales.replay_value = 0.0;
    c.loss_scales.reconstruction = 100.0;
    c
}
#[test]
fn gaussian_world_configuration_requires_explicit_supported_architecture() {
    let c = config();
    assert!(c.check().is_ok());
    assert!(
        !serde_json::to_value(DreamerConfig::tiny(3))
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("world_gaussian")
    );
    for loss in [
        PixelLoss::SquaredError,
        PixelLoss::MotionLatentOvershooting,
        PixelLoss::MotionRecurrentEnergy,
        PixelLoss::MotionFixedTeacher,
    ] {
        assert!(WorldModelLearner::with_pixel_loss(c.clone(), 16, loss).is_err());
    }
    assert!(crate::dreamer::VectorDreamerAgent::learned_rgb(c, 8).is_err());
}
#[test]
#[ignore = "requires separately declared guarded GPU Gaussian values/derivatives check"]
fn gaussian_native_values_and_derivatives() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    for (qstop, pstop, floor) in [
        (false, false, 0.0),
        (true, false, 0.0),
        (false, true, 0.0),
        (false, false, 1e6),
    ] {
        let f = fixture(8, 32, 16, qstop, pstop, floor);
        let mut s = runtime::build_session(&f.graph, &gpu, Mode::Training, false);
        s.set_parameter("q", &f.q);
        s.set_parameter("p", &f.p);
        s.set_input("noise", &f.noise);
        s.step();
        s.wait();
        assert_eq!(s.adam_step_count(), 0);
        for (i, &e) in f.expected.iter().enumerate() {
            let mut v = [0.0];
            s.read_output_by_index(i, &mut v);
            assert!((f64::from(v[0]) - e).abs() < 2e-2, "{} vs {e}", v[0]);
        }
        let mut actual_samples = vec![0.0; f.samples.len()];
        s.read_output_by_index(3, &mut actual_samples);
        for (a, b) in actual_samples.iter().zip(&f.samples) {
            assert!((f64::from(*a) - b).abs() < 2e-6);
        }
        for (name, e) in ["q", "p"].iter().zip(&f.grad) {
            let mut v = vec![0.0; e.len()];
            if s.has_param_grad(name) {
                s.read_param_grad(name, &mut v);
            }
            for (a, b) in v.iter().zip(e) {
                assert!(
                    (f64::from(*a) - b).abs() < 2e-6 + 1e-5 * b.abs(),
                    "{a} vs {b}"
                );
            }
        }
        for (name, expected) in [("q", &f.q), ("p", &f.p)] {
            let mut v = vec![0.0; expected.len()];
            s.read_param(name, &mut v);
            assert_eq!(&v, expected);
        }
    }
}
