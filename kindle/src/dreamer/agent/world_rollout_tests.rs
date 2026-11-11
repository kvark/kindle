use super::*;
use crate::dreamer::ObservationKind;
use crate::dreamer::world_learner::PixelLoss;
use meganeura::Graph;
use meganeura::graph::Op;

fn plain() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(3);
    c.observation_kind = ObservationKind::Rgb32;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.skip_full_optimize = true;
    c
}

fn with_rollout() -> DreamerConfig {
    let mut c = plain();
    c.world_pixel_loss = PixelLoss::MotionCrossEntropy;
    c.world_prediction_length = 2;
    c
}

fn with_context_frame() -> DreamerConfig {
    let mut c = with_rollout();
    c.rgb_spatial_decoder = true;
    c.rgb_context_lag = Some(2);
    c
}

#[test]
fn agent_world_accepts_the_context_frame_decoder_but_not_image_history() {
    let c = with_context_frame();
    assert!(c.check().is_ok(), "{:?}", c.check());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    let graph = world::build_training_graph(&world_training_config(&c), 4);
    assert!(graph.outputs().len() > world::LOSS_WORLD_ROLLOUT);
    let inputs = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!inputs.iter().any(|n| n.starts_with("decoder_history")));
    assert!(inputs.iter().any(|n| n == "keep_action_0"));
    let mut history_only = with_context_frame();
    history_only.rgb_context_lag = None;
    assert!(history_only.check().is_err());
    // The autoregressive history stays available to standalone worlds, which
    // use the learner's own pixel objective rather than the agent fields.
    let mut standalone = history_only.clone();
    standalone.loss_scales.policy = 0.0;
    standalone.loss_scales.value = 0.0;
    standalone.loss_scales.replay_value = 0.0;
    standalone.actor_critic_gradient = false;
    standalone.world_prediction_length = 0;
    standalone.world_pixel_loss = PixelLoss::SquaredError;
    assert!(standalone.check().is_ok(), "{:?}", standalone.check());
}

fn cdp_with_pixels() -> DreamerConfig {
    let mut c = with_rollout();
    c.loss_scales.reconstruction = 0.0;
    c.loss_scales.future_prediction = 1.0;
    c.cdp_centered = true;
    c.world_pixel_scale = 1.0;
    c
}

#[test]
fn cdp_world_trains_an_auxiliary_pixel_decoder_and_rollout() {
    let c = cdp_with_pixels();
    assert!(c.check().is_ok(), "{:?}", c.check());
    assert!(c.is_cdp() && c.trains_pixels() && c.pixel_scale() == 1.0);
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    let names = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let graph = world::build_training_graph(&world_training_config(&c), 4);
    let parameters = names(&graph);
    assert!(parameters.iter().any(|n| n.starts_with("world.decoder.")));
    assert!(parameters.iter().any(|n| n.starts_with("world.future_predictor.")));
    assert!(graph.outputs().len() > world::LOSS_WORLD_ROLLOUT);

    let mut without = cdp_with_pixels();
    without.world_pixel_scale = 0.0;
    assert!(without.check().is_err());
    without.world_prediction_length = 0;
    without.world_pixel_loss = PixelLoss::SquaredError;
    assert!(without.check().is_ok());
    assert!(!without.trains_pixels());
    let graph = world::build_training_graph(&world_training_config(&without), 4);
    assert!(!names(&graph).iter().any(|n| n.starts_with("world.decoder.")));

    let mut plain = with_rollout();
    plain.world_pixel_scale = 1.0;
    assert!(plain.check().is_err());
    let mut negative = cdp_with_pixels();
    negative.world_pixel_scale = -1.0;
    assert!(negative.check().is_err());
}

fn with_rollout_and_bonus() -> DreamerConfig {
    let mut c = with_rollout();
    c.disagreement_bonus = true;
    c.intrinsic_reward_scale = 1.0;
    c
}

#[test]
fn agent_world_rollout_accepts_the_action_effects_bonus_without_cdp() {
    let c = with_rollout_and_bonus();
    assert!(c.check().is_ok());
    assert!(!c.is_cdp() && c.uses_disagreement());
    let graph = world::build_training_graph(&world_training_config(&c), 4);
    assert!(graph.outputs().len() > world::LOSS_WORLD_ROLLOUT);
    let heads = graph
        .nodes()
        .iter()
        .filter(|n| matches!(&n.op, Op::Parameter { name } if name.starts_with("world.exploration.")))
        .count();
    assert!(heads > 0);
    let imagination = world::build_imagination_graph(&c);
    assert_eq!(imagination.outputs().len(), world::IMAGINATION_BONUS + 1);
}

#[test]
fn agent_world_rollout_requires_a_plain_rgb_world_with_full_bptt() {
    let c = with_rollout();
    assert!(c.check().is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
    let defaults = serde_json::to_value(plain()).unwrap();
    assert!(
        defaults.get("world_pixel_loss").is_none()
            && defaults.get("world_prediction_length").is_none()
    );
    let mut bad = c.clone();
    bad.world_backprop_length = 2;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.world_prediction_length = 4;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.loss_scales.reconstruction = 0.0;
    bad.loss_scales.future_prediction = 500.0;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.rgb_spatial_decoder = true;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.world_pixel_loss = PixelLoss::MotionEventRollout;
    assert!(bad.check().is_err());
    let mut bad = c.clone();
    bad.observation_kind = ObservationKind::Features;
    assert!(bad.check().is_err());
}

#[test]
fn agent_world_graph_adds_the_rollout_output_without_new_parameters() {
    let parameters = |g: &Graph| {
        let mut names = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    let inputs = |g: &Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let plain_graph = world::build_training_graph(&world_training_config(&plain()), 4);
    let rollout_graph = world::build_training_graph(&world_training_config(&with_rollout()), 4);
    assert_eq!(parameters(&plain_graph), parameters(&rollout_graph));
    assert!(plain_graph.outputs().len() <= world::LOSS_WORLD_ROLLOUT);
    assert!(rollout_graph.outputs().len() > world::LOSS_WORLD_ROLLOUT);
    let names = inputs(&rollout_graph);
    for name in [
        "pixel_rollout_uniforms_0",
        "pixel_rollout_uniforms_1",
        "pixel_rollout_weight_0",
        "pixel_rollout_weight_1",
    ] {
        assert!(names.iter().any(|n| n == name), "{name}");
    }
    assert!(
        !inputs(&plain_graph)
            .iter()
            .any(|n| n.starts_with("pixel_rollout"))
    );
}

fn fixture(config: &DreamerConfig, rng: &mut StdRng) -> SequenceBatch {
    let (rows, length, network) = (config.batch_size, config.batch_length, config.network());
    let mut batch = SequenceBatch::default();
    batch.initial_deter = (0..rows * network.deter)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();
    batch.initial_stoch = vec![0.0; rows * network.stoch * network.classes];
    for state in batch.initial_stoch.chunks_exact_mut(network.classes) {
        state[rng.random_range(0..network.classes)] = 1.0;
    }
    for time in 0..length {
        batch.observations.push(
            (0..rows * config.observation_dim())
                .map(|_| f32::from(rng.random::<u8>()) / 255.0 - 0.5)
                .collect(),
        );
        let flags = (0..rows)
            .map(|row| FrameFlags {
                is_first: (time, row) == (0, 0) || (time, row) == (3, 1),
                is_last: (time, row) == (2, 1),
                is_terminal: (time, row) == (2, 1),
            })
            .collect::<Vec<_>>();
        let mut actions = vec![0.0; rows * config.action_count];
        for (row, flag) in flags.iter().enumerate() {
            if !flag.is_first {
                actions[row * config.action_count + rng.random_range(0..config.action_count)] = 1.0;
            }
        }
        batch.previous_actions.push(actions);
        batch.rewards.push(
            (0..rows)
                .map(|row| ((time + row) % 3) as f32 - 1.0)
                .collect(),
        );
        batch.flags.push(flags);
    }
    batch
}

#[test]
#[ignore = "builds and runs all Dreamer GPU sessions"]
fn tiny_rgb_agent_trains_a_prior_image_rollout() {
    tiny_rgb_agent_trains(with_rollout());
}

#[test]
#[ignore = "builds and runs all Dreamer GPU sessions"]
fn tiny_rgb_agent_trains_a_prior_image_rollout_with_the_bonus() {
    tiny_rgb_agent_trains(with_rollout_and_bonus());
}

#[test]
#[ignore = "builds and runs all Dreamer GPU sessions"]
fn tiny_rgb_agent_trains_with_the_context_frame_decoder() {
    tiny_rgb_agent_trains(with_context_frame());
}

#[test]
#[ignore = "builds and runs all Dreamer GPU sessions"]
fn tiny_cdp_agent_trains_the_auxiliary_pixel_rollout() {
    tiny_cdp_agent_trains(cdp_with_pixels());
}

fn tiny_cdp_agent_trains(config: DreamerConfig) {
    let world = tiny_rgb_agent_trains(config);
    assert!(world.future_prediction_loss > 0.0);
}

fn tiny_rgb_agent_trains(config: DreamerConfig) -> WorldMetrics {
    let mut last = None;
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
    let mut rng = StdRng::seed_from_u64(701);
    for _ in 0..2 {
        let batch = fixture(&config, &mut rng);
        let posterior = core.sample_posterior_batch(&batch);
        let mut targets = core.imagine_and_target(&batch, &posterior);
        let world = core.train_world(&batch, &posterior, &mut targets);
        core.sync_world_inference();
        let rollout = world.prior_rollout_loss.expect("rollout loss reported");
        assert!(world.total_loss.is_finite() && rollout.is_finite() && rollout > 0.0);
        assert!(world.reconstruction_loss > 0.0);
        core.learner_step += 1;
        last = Some(world);
    }
    last.unwrap()
}
