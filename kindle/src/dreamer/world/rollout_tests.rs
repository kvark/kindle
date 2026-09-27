use super::*;
use crate::dreamer::{
    behavior,
    distributions::softmax_unimix,
    readback::Readback,
    runtime::{build_session, initialize_d3, sync_matching},
};
use meganeura::{Mode, Session};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

fn gpu() -> Arc<blade_graphics::Context> {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
        let info = gpu.device_information();
        assert_eq!(info.device_name, expected);
        assert!(!info.is_software_emulated);
    }
    check_memory(&gpu);
    gpu
}

fn check_memory(gpu: &blade_graphics::Context) {
    if std::env::var_os("KINDLE_EXPECT_DEVICE_NAME").is_some() {
        let memory = gpu.memory_stats();
        eprintln!("memory usage={} budget={}", memory.usage, memory.budget);
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
}

fn output(session: &Session, index: usize, count: usize, readback: &mut Readback) -> Vec<f32> {
    let mut values = vec![0.0; count];
    readback.read(session, &mut [(index, &mut values)]);
    values
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "element {i}: {a} != {b}"
        );
    }
}

fn sample(logits: &[f32], draws: &[f32], classes: usize, unimix: f32) -> Vec<f32> {
    assert_eq!(logits.len(), draws.len());
    let mut output = vec![0.0; logits.len()];
    let mut probabilities = vec![0.0; classes];
    for ((logits, draws), output) in logits
        .chunks_exact(classes)
        .zip(draws.chunks_exact(classes))
        .zip(output.chunks_exact_mut(classes))
    {
        softmax_unimix(logits, unimix, &mut probabilities);
        let mut best = f32::NEG_INFINITY;
        let mut selected = 0;
        for (index, (&p, &u)) in probabilities.iter().zip(draws).enumerate() {
            let score = p.ln() - (-u.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON).ln()).ln();
            if score > best {
                best = score;
                selected = index;
            }
        }
        output[selected] = 1.0;
    }
    output
}

#[test]
fn fused_rollout_graphs_have_full_horizons_and_no_trainable_sampling_parameters() {
    let config = DreamerConfig::tiny(3);
    let size = config.network();
    let rows = config.batch_size * config.batch_length;
    let posterior = build_posterior_graph(&config);
    assert_eq!(
        posterior.node(posterior.outputs()[0]).ty.shape,
        [rows, size.deter]
    );
    assert_eq!(
        posterior.node(posterior.outputs()[1]).ty.shape,
        [rows, size.stoch * size.classes]
    );
    let imagination = build_imagination_graph(&config);
    assert_eq!(imagination.outputs().len(), 6);
    assert_eq!(
        imagination
            .node(imagination.outputs()[IMAGINATION_FEATURE])
            .ty
            .shape,
        [rows * config.imagination_length, config.feature_dim()]
    );
    assert_eq!(
        imagination
            .node(imagination.outputs()[IMAGINATION_ALL_FEATURES])
            .ty
            .shape,
        [rows * (config.imagination_length + 1), config.feature_dim()]
    );
    for graph in [&posterior, &imagination] {
        for node in graph.nodes() {
            if let meganeura::graph::Op::Parameter { name } = &node.op {
                assert!(name.starts_with("world.") || name.starts_with("behavior."));
            }
        }
    }
}

#[test]
fn categorical_reduction_dispatch_covers_every_row() {
    for classes in [1, 4, 16, 18, 24] {
        let rows = 16_384;
        let mut graph = Graph::new();
        let logits = graph.input("logits", &[rows, classes]);
        let draws = graph.input("draws", &[rows, classes]);
        let sampled = gumbel_sample(&mut graph, logits, draws, rows, classes, 0.01);
        graph.set_outputs(vec![sampled]);
        let plan = meganeura::compile::compile(&graph);
        let reduction = plan
            .dispatches
            .iter()
            .find(|d| d.shader == meganeura::compile::ShaderEntry::MaxPool2d)
            .unwrap();
        assert_eq!(reduction.params[0], rows as u32);
        assert_eq!(reduction.workgroups, [rows.div_ceil(256) as u32, 1, 1]);
    }
}

#[test]
#[ignore = "requires a GPU"]
fn tiny_gumbel_samples_match_cpu_and_categorical_frequencies() {
    let gpu = gpu();
    let rows = 16_384;
    let classes = 4;
    let unimix = 0.1;
    let mut graph = Graph::new();
    let logits = graph.input("logits", &[rows, classes]);
    let draws = graph.input("draws", &[rows, classes]);
    let sampled = gumbel_sample(&mut graph, logits, draws, rows, classes, unimix);
    graph.set_outputs(vec![sampled]);
    let mut session = build_session(&graph, &gpu, Mode::Inference, false);
    check_memory(&gpu);
    let mut readback = Readback::new(gpu);
    let mut rng = StdRng::seed_from_u64(19);
    let logits = [-1.0, 0.0, 0.5, 1.0].repeat(rows);
    let mut draws = (0..rows * classes)
        .map(|_| rng.random::<f32>())
        .collect::<Vec<_>>();
    draws[..4].copy_from_slice(&[0.0, 1.0, 0.5, 0.5]);
    session.set_input("logits", &logits);
    session.set_input("draws", &draws);
    session.step();
    let actual = output(&session, 0, rows * classes, &mut readback);
    let expected = sample(&logits, &draws, classes, unimix);
    for (index, (a, b)) in actual
        .chunks_exact(classes)
        .zip(expected.chunks_exact(classes))
        .enumerate()
    {
        assert_eq!(
            a,
            b,
            "sample row {index}, draws {:?}",
            &draws[index * classes..(index + 1) * classes]
        );
    }
    let mut probabilities = vec![0.0; classes];
    softmax_unimix(&logits[..classes], unimix, &mut probabilities);
    for class in 0..classes {
        let count = actual
            .chunks_exact(classes)
            .filter(|r| r[class] == 1.0)
            .count() as f32;
        let p = probabilities[class];
        assert!((count - rows as f32 * p).abs() < 6.0 * (rows as f32 * p * (1.0 - p)).sqrt());
    }
    session.set_input("logits", &vec![0.0; rows * classes]);
    session.set_input("draws", &vec![0.5; rows * classes]);
    session.step();
    assert_eq!(
        output(&session, 0, rows * classes, &mut readback),
        [1.0, 0.0, 0.0, 0.0].repeat(rows)
    );
}

#[test]
#[ignore = "requires a GPU"]
fn tiny_fused_rollouts_match_sequential_reference() {
    let gpu = gpu();
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 2;
    config.batch_length = 4;
    config.world_backprop_length = 4;
    config.imagination_length = 3;
    let size = config.network();
    let batch = config.batch_size;
    let length = config.batch_length;
    let latent_width = size.stoch * size.classes;
    let posterior_graph = build_posterior_graph(&config);
    let mut posterior = build_session(&posterior_graph, &gpu, Mode::Inference, false);
    initialize_d3(&mut posterior, &posterior_graph, 29);
    let mut observe = build_session(
        &build_observe_graph(&config, batch),
        &gpu,
        Mode::Inference,
        false,
    );
    sync_matching(&posterior, &mut observe, "world.");
    let mut readback = Readback::new(Arc::clone(&gpu));
    let mut rng = StdRng::seed_from_u64(31);
    let mut deter = vec![0.1; batch * size.deter];
    let mut stoch = vec![0.0; batch * latent_width];
    for row in stoch.chunks_exact_mut(size.classes) {
        row[0] = 1.0;
    }
    posterior.set_input("initial_deter", &deter);
    posterior.set_input("initial_stoch", &stoch);
    let mut deters = Vec::new();
    let mut stochs = Vec::new();
    for time in 0..length {
        let observation = (0..batch * config.observation_dim())
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let action = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0].to_vec();
        let keep = [f32::from(time != 0), f32::from(time != 2)];
        let mask = |width| {
            keep.iter()
                .flat_map(|&v| vec![v; width])
                .collect::<Vec<_>>()
        };
        let uniforms = (0..batch * latent_width)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        observe.set_input("previous_deter", &deter);
        observe.set_input("previous_stoch", &stoch);
        for (name, data) in [
            ("observation", observation),
            ("previous_action", action),
            ("keep_deter", mask(size.deter)),
            ("keep_stoch", mask(latent_width)),
            ("keep_action", mask(config.action_count)),
        ] {
            observe.set_input(name, &data);
            posterior.set_input(&format!("{name}_{time}"), &data);
        }
        posterior.set_input(&format!("uniforms_{time}"), &uniforms);
        observe.step();
        deter = output(&observe, 0, batch * size.deter, &mut readback);
        let logits = output(&observe, 1, batch * latent_width, &mut readback);
        stoch = sample(&logits, &uniforms, size.classes, config.unimix);
        deters.extend_from_slice(&deter);
        stochs.extend_from_slice(&stoch);
    }
    posterior.step();
    close(&output(&posterior, 0, deters.len(), &mut readback), &deters);
    assert_eq!(output(&posterior, 1, stochs.len(), &mut readback), stochs);

    let rows = batch * length;
    let horizon = config.imagination_length;
    let graph = build_imagination_graph(&config);
    let mut imagination = build_session(&graph, &gpu, Mode::Inference, false);
    initialize_d3(&mut imagination, &graph, 37);
    // Exercise nonzero reward/value predictions, unlike D3's zero head initialization.
    for name in ["world.reward.out.weight", "behavior.value.out.weight"] {
        let weights = (0..imagination.param_size(name).unwrap())
            .map(|_| rng.random::<f32>() * 0.1)
            .collect::<Vec<_>>();
        imagination.set_parameter(name, &weights);
    }
    let mut transition = build_session(
        &build_transition_graph(&config, rows),
        &gpu,
        Mode::Inference,
        false,
    );
    let mut heads = build_session(
        &build_imagination_head_graph(&config, rows),
        &gpu,
        Mode::Inference,
        false,
    );
    let mut behavior = build_session(
        &behavior::build_inference_graph(&config, rows),
        &gpu,
        Mode::Inference,
        false,
    );
    sync_matching(&imagination, &mut transition, "world.");
    sync_matching(&imagination, &mut heads, "world.");
    sync_matching(&imagination, &mut behavior, "behavior.");
    imagination.set_input("deter", &deters);
    imagination.set_input("stoch", &stochs);
    let mut features = Vec::new();
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    let mut continuations = Vec::new();
    let mut values = Vec::new();
    for time in 0..=horizon {
        heads.set_input("deter", &deters);
        heads.set_input("stoch", &stochs);
        heads.step();
        let feature = output(&heads, 2, rows * config.feature_dim(), &mut readback);
        features.extend_from_slice(&feature);
        rewards.extend(output(&heads, 0, rows * config.value_bins, &mut readback));
        continuations.extend(output(&heads, 1, rows, &mut readback));
        behavior.set_input("feature", &feature);
        behavior.step();
        values.extend(output(
            &behavior,
            1,
            rows * config.value_bins,
            &mut readback,
        ));
        if time == horizon {
            break;
        }
        let action_draws = (0..rows * config.action_count)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let latent_draws = (0..rows * latent_width)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        imagination.set_input(&format!("action_uniforms_{time}"), &action_draws);
        imagination.set_input(&format!("latent_uniforms_{time}"), &latent_draws);
        let logits = output(&behavior, 0, rows * config.action_count, &mut readback);
        let action = sample(
            &logits,
            &action_draws,
            config.action_count,
            config.actor_unimix,
        );
        actions.extend_from_slice(&action);
        transition.set_input("deter", &deters);
        transition.set_input("stoch", &stochs);
        transition.set_input("action", &action);
        transition.step();
        deters = output(&transition, 0, rows * size.deter, &mut readback);
        let logits = output(&transition, 1, rows * latent_width, &mut readback);
        stochs = sample(&logits, &latent_draws, size.classes, config.unimix);
    }
    imagination.step();
    for (index, expected) in [
        (
            IMAGINATION_FEATURE,
            &features[..horizon * rows * config.feature_dim()],
        ),
        (IMAGINATION_ALL_FEATURES, features.as_slice()),
        (IMAGINATION_ACTION, actions.as_slice()),
        (IMAGINATION_REWARD, rewards.as_slice()),
        (IMAGINATION_CONTINUATION, continuations.as_slice()),
        (IMAGINATION_VALUE, values.as_slice()),
    ] {
        close(
            &output(&imagination, index, expected.len(), &mut readback),
            expected,
        );
    }
}
