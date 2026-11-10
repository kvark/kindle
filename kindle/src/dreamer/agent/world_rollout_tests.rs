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
    let config = with_rollout();
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
        core.learner_step += 1;
    }
}
