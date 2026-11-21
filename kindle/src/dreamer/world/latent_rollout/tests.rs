use super::*;
use meganeura::{
    graph::Op,
    reference::{Feeds, evaluate},
};
use std::{collections::HashSet, sync::Arc};

fn probabilities(values: &[f64], mixing: f32) -> (Vec<f64>, Vec<f64>) {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps = values.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
    let total = exps.iter().sum::<f64>();
    let softmax = exps.iter().map(|x| x / total).collect::<Vec<_>>();
    let mixed = softmax
        .iter()
        .map(|p| f64::from(1.0_f32 - mixing) * p + f64::from(mixing / values.len() as f32))
        .collect();
    (softmax, mixed)
}

fn reference(
    config: &DreamerConfig,
    target: &[f64],
    prior: &[f64],
    keep: &[f64],
) -> (f64, Vec<f64>) {
    let size = config.network();
    let mut loss = 0.;
    let mut gradient = vec![0.; prior.len()];
    for (row, (target, prior)) in target
        .chunks_exact(size.classes)
        .zip(prior.chunks_exact(size.classes))
        .enumerate()
    {
        let (_, q) = probabilities(target, config.unimix);
        let (softmax, p) = probabilities(prior, config.unimix);
        let weight = keep[row / size.stoch] / config.batch_size as f64;
        // F64 MeanAll divides exactly; its differentiated graph stores an F32 reciprocal.
        let gradient_weight =
            keep[row / size.stoch] * f64::from((config.batch_size as f32).recip());
        loss += weight * q.iter().zip(&p).map(|(q, p)| q * (q / p).ln()).sum::<f64>();
        let projected = q
            .iter()
            .zip(&softmax)
            .zip(&p)
            .map(|((q, s), p)| q * s / p)
            .sum::<f64>();
        for j in 0..size.classes {
            gradient[row * size.classes + j] = gradient_weight
                * f64::from(1.0_f32 - config.unimix)
                * softmax[j]
                * (projected - q[j] / p[j]);
        }
    }
    (loss, gradient)
}

struct Fixture {
    graph: Graph,
    feeds: Feeds,
    config: DreamerConfig,
    values: [Vec<f32>; 3],
    expected: f64,
    gradient: Vec<f64>,
}

fn fixture(batch: usize, zero: bool) -> Fixture {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = batch;
    config.unimix = 0.01;
    let size = config.network();
    let len = batch * size.stoch * size.classes;
    let prior = (0..len)
        .map(|i| ((i * 7) % 19) as f32 * 0.125 - 1.)
        .collect::<Vec<_>>();
    let target = (0..len)
        .map(|i| ((i * 3 + 5) % 17) as f32 * 0.125 - 1.)
        .collect::<Vec<_>>();
    let keep = (0..batch)
        .map(|i| f32::from(!zero && i % 3 != 0))
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let p = graph.parameter("prior", &[batch * size.stoch, size.classes]);
    let q = graph.parameter("target", &[batch * size.stoch, size.classes]);
    let k = graph.parameter("keep", &[batch, 1]);
    let loss = masked_kl(&mut graph, &config, q, p, k);
    graph.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds
        .set("prior", &prior)
        .set("target", &target)
        .set("keep", &keep);
    let to_f64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let (expected, gradient) =
        reference(&config, &to_f64(&target), &to_f64(&prior), &to_f64(&keep));
    Fixture {
        graph,
        feeds,
        config,
        values: [prior, target, keep],
        expected,
        gradient,
    }
}

#[test]
fn masked_overshooting_kl_matches_f64_and_stops_targets_and_masks() {
    for zero in [false, true] {
        let f = fixture(3, zero);
        let graph = meganeura::autodiff::differentiate(&f.graph);
        let values = evaluate(&graph, &f.feeds).unwrap();
        let outputs = graph.outputs();
        assert!(
            (values[outputs[0] as usize].data[0] - f.expected).abs() < 1e-10,
            "{} vs {}",
            values[outputs[0] as usize].data[0],
            f.expected
        );
        for (a, b) in values[outputs[1] as usize].data.iter().zip(&f.gradient) {
            assert!((a - b).abs() < 1e-10);
        }
        for &output in &outputs[2..] {
            assert!(values[output as usize].data.iter().all(|&x| x == 0.));
        }
        if zero {
            assert_eq!(f.expected, 0.);
        }
    }
}

#[test]
fn mixed_prior_kl_derivative_matches_finite_differences() {
    let f = fixture(3, false);
    let mut prior = f.values[0]
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    let target = f.values[1]
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    let keep = f.values[2]
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    for i in (0..prior.len()).step_by(7) {
        let old = prior[i];
        prior[i] = old + 1e-5;
        let plus = reference(&f.config, &target, &prior, &keep).0;
        prior[i] = old - 1e-5;
        let minus = reference(&f.config, &target, &prior, &keep).0;
        prior[i] = old;
        assert!(((plus - minus) / 2e-5 - f.gradient[i]).abs() < 1e-8);
    }
}

#[test]
fn latent_rollout_is_causal_and_auxiliary_gradients_only_reach_dynamics() {
    let config = DreamerConfig::tiny(3);
    let mut graph = Graph::new();
    let model = WorldModel::with_behavior(&mut graph, &config, false);
    let size = config.network();
    let deter = graph.input("seed_deter", &[config.batch_size, size.deter]);
    let stoch = graph.input(
        "seed_stoch",
        &[config.batch_size * size.stoch, size.classes],
    );
    let actions = (0..3)
        .map(|i| {
            graph.input(
                &format!("action{i}"),
                &[config.batch_size, config.action_count],
            )
        })
        .collect::<Vec<_>>();
    let targets = (0..3)
        .map(|i| {
            graph.input(
                &format!("future_target{i}"),
                &[config.batch_size * size.stoch, size.classes],
            )
        })
        .collect::<Vec<_>>();
    let rollout = build(
        &mut graph,
        &config,
        &model,
        (deter, stoch),
        &actions,
        &targets,
    );
    for (t, prior) in rollout.priors.iter().enumerate() {
        let mut seen = HashSet::new();
        let mut pending = vec![*prior];
        while let Some(id) = pending.pop() {
            if seen.insert(id) {
                pending.extend(&graph.node(id).inputs);
            }
        }
        assert!(targets.iter().all(|target| !seen.contains(target)));
        assert!(actions[..=t].iter().all(|action| seen.contains(action)));
        assert!(actions[t + 1..].iter().all(|action| !seen.contains(action)));
        assert!(seen.iter().all(|&id| !matches!(&graph.node(id).op, Op::Input{name} if name.starts_with("pixel_rollout_weight_"))));
    }
    graph.set_outputs(vec![rollout.loss]);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    let active = plan
        .param_buffers
        .iter()
        .filter(|(_, buffer)| plan.param_grad_pairs.iter().any(|(p, _)| p == buffer))
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert!(active.iter().any(|n| n.starts_with("world.dynamics.core.")));
    assert!(active.iter().any(|n| n.starts_with("world.dynamics.prior")));
    assert_auxiliary_gradient_scope(&graph);
}

fn assert_auxiliary_gradient_scope(graph: &Graph) {
    let backward = meganeura::autodiff::differentiate(graph);
    let names = graph.nodes().iter().filter_map(|n| match &n.op {
        Op::Parameter { name } => Some(name),
        _ => None,
    });
    for (name, &gradient) in names.zip(&backward.outputs()[1..]) {
        if !name.starts_with("world.dynamics.") {
            assert!(
                matches!(&backward.node(gradient).op, Op::Constant { data } if data.iter().all(|&v| v == 0.)),
                "unexpected auxiliary gradient: {name}"
            );
        }
    }
}

#[test]
fn full_latent_objective_preserves_shapes_and_stops_image_informed_targets() {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb64;
    config.batch_length = 3;
    config.world_backprop_length = 3;
    config.actor_critic_gradient = false;
    config.disagreement_bonus = false;
    config.loss_scales.policy = 0.;
    config.loss_scales.value = 0.;
    config.loss_scales.replay_value = 0.;
    let normal = build_world_only_pixel_training_graph(&config, 0, PixelLoss::MotionCrossEntropy);
    let mut latent =
        build_world_only_pixel_training_graph(&config, 2, PixelLoss::MotionLatentOvershooting);
    let parameters = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(parameters(&normal), parameters(&latent));
    meganeura::compile_training_graph(&latent);
    let loss = latent.outputs()[LOSS_WORLD_ROLLOUT];
    latent.set_outputs(vec![loss]);
    assert_auxiliary_gradient_scope(&latent);
}

#[test]
#[ignore = "requires guarded GPU; masked mixed categorical KL and derivatives"]
fn masked_overshooting_kl_native_matches_f64() {
    let f = fixture(256, false);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut session =
        crate::dreamer::runtime::build_session(&f.graph, &gpu, meganeura::Mode::Training, false);
    for (name, values) in ["prior", "target", "keep"].into_iter().zip(&f.values) {
        session.set_parameter(name, values);
    }
    session.step();
    session.wait();
    assert_eq!(session.adam_step_count(), 0);
    let mut loss = [0.];
    session.read_output_by_index(0, &mut loss);
    assert!(
        (f64::from(loss[0]) - f.expected).abs() < 2e-5,
        "{} vs {}",
        loss[0],
        f.expected
    );
    let mut gradient = vec![0.; f.gradient.len()];
    session.read_param_grad("prior", &mut gradient);
    for (a, b) in gradient.iter().zip(f.gradient) {
        assert!((f64::from(*a) - b).abs() < 2e-6);
    }
    for (name, expected) in ["prior", "target", "keep"].into_iter().zip(&f.values) {
        let mut actual = vec![0.; expected.len()];
        session.read_param(name, &mut actual);
        assert_eq!(&actual, expected);
        if name != "prior" && session.has_param_grad(name) {
            session.read_param_grad(name, &mut actual);
            assert!(actual.iter().all(|&v| v == 0.));
        }
    }
}
