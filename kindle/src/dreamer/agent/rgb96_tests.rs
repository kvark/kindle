use super::*;
use crate::dreamer::{ModelSize, ObservationKind};
use meganeura::{Graph, graph::Op};
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb96;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.cdp_centered = true;
    config.world_microbatch_size = Some(config.batch_size);
    config.validate();
    config
}

fn shapes(graph: &Graph) -> BTreeMap<String, Vec<usize>> {
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
fn rgb96_observations_roundtrip_and_preserve_legacy_geometry() {
    assert_eq!(ObservationKind::default(), ObservationKind::Features);
    let mut config = config();
    for (kind, side) in [
        (ObservationKind::Rgb32, 32),
        (ObservationKind::Rgb48, 48),
        (ObservationKind::Rgb64, 64),
        (ObservationKind::Rgb96, 96),
    ] {
        config.observation_kind = kind;
        assert_eq!(config.observation_shape(2), [2, 3 * side * side]);
        assert_eq!(config.encoded_observation_dim(), 16 * (side / 16).pow(2));
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["observation_kind"], format!("rgb{side}"));
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(json).unwrap(),
            config
        );
        assert_eq!(
            Observation::rgb(vec![0.25; 3 * side * side], side)
                .as_slice()
                .len(),
            config.observation_dim()
        );
    }
    assert_eq!(
        Observation::rgb_pair(vec![0.0; 6 * 96 * 96], 96)
            .as_slice()
            .len(),
        6 * 96 * 96
    );
    assert!(std::panic::catch_unwind(|| Observation::rgb(vec![0.0; 3 * 64 * 64], 96)).is_err());
    assert!(
        std::panic::catch_unwind(|| Observation::rgb(vec![f32::NAN; 3 * 96 * 96], 96)).is_err()
    );
    assert!(serde_json::from_str::<ObservationKind>("\"rgb95\"").is_err());
}

#[test]
fn rgb96_changes_visual_projection_shapes_and_keeps_control_graphs() {
    for size in [ModelSize::Tiny, ModelSize::Size12M] {
        let mut candidate = config();
        candidate.model_size = size;
        let mut plain = candidate.clone();
        plain.observation_kind = ObservationKind::Rgb64;
        assert_eq!(candidate.feature_dim(), plain.feature_dim());
        assert_eq!(
            candidate.encoded_observation_dim() * 4,
            plain.encoded_observation_dim() * 9
        );
        if size == ModelSize::Size12M {
            assert_eq!(candidate.encoded_observation_dim(), 2304);
        }
        for build in [
            world::build_imagination_graph as fn(&DreamerConfig) -> Graph,
            world::build_posterior_graph,
        ] {
            let (a, b) = (shapes(&build(&plain)), shapes(&build(&candidate)));
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (name, shape) in &a {
                if !name.starts_with("world.representation.posterior.") {
                    assert_eq!(shape, &b[name], "{name}");
                }
            }
        }
        let a = behavior::build_training_graph(&plain, 6, 2);
        let b = behavior::build_training_graph(&candidate, 6, 2);
        assert_eq!(format!("{:?}", a.nodes()), format!("{:?}", b.nodes()));
        assert_eq!(a.outputs(), b.outputs());
        let a = shapes(&world::build_training_graph(&plain, 4));
        let b = shapes(&world::build_training_graph(&candidate, 4));
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        assert!(a.iter().any(|(name, shape)| shape != &b[name]));
        for (name, shape) in &a {
            if name.starts_with("world.representation.encoder.")
                || name.starts_with("world.dynamics.")
            {
                assert_eq!(shape, &b[name], "{name}");
            }
        }
    }
}

#[test]
fn rgb96_cdp_and_reconstruction_compile_with_world_gradient_owners() {
    let mut config = config();
    for cdp in [true, false] {
        config.loss_scales.future_prediction = if cdp { 500.0 } else { 0.0 };
        config.loss_scales.reconstruction = if cdp { 0.0 } else { 1.0 };
        config.cdp_centered = cdp;
        config.validate();
        let graph = world::build_training_graph(&config, 4);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        let mut encoder_gradients = 0;
        for (name, buffer) in &plan.param_buffers {
            if plan
                .param_grad_pairs
                .iter()
                .any(|(parameter, _)| parameter == buffer)
            {
                assert!(
                    name.starts_with("world."),
                    "unexpected gradient owner {name}"
                );
                encoder_gradients += usize::from(name.starts_with("world.representation.encoder."));
            }
        }
        assert_eq!(encoder_gradients, 12);
    }
}

#[test]
#[ignore = "requires separately guarded RGB96 two-update and exact-restore qualification"]
fn rgb96_learns_and_restores_parameters_moments_and_geometry() {
    let config = config();
    let mut core = DreamerCore::new(config.clone()).unwrap();
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!core.gpu_device().is_software_emulated);
    let observation = |time: usize| {
        Observation::rgb(
            (0..config.observation_dim())
                .map(|i| ((i * 37 + i / 43 + time * 11) % 256) as f32 / 255.0 - 0.5)
                .collect(),
            96,
        )
    };
    let encoder_name = "world.representation.encoder.cnn0.weight";
    let before = core.world_train.read_params(&[encoder_name]);
    core.begin_episode(observation(0));
    for time in 1..=16 {
        core.act(ActionMode::Sample, None);
        core.observe(
            observation(time),
            Reward {
                extrinsic: if time % 3 == 0 { 1.0 } else { 0.0 },
                intrinsic: 0.0,
            },
            FrameFlags::default(),
        );
    }
    for step in 1..=2 {
        let report = core.learn().expect("ready synthetic replay");
        assert_eq!(report.learner_step, step);
        assert!(report.world.total_loss.is_finite() && report.behavior.policy_loss.is_finite());
        assert_eq!(u64::from(core.world_train.adam_step_count()), step);
        assert_eq!(u64::from(core.behavior_train.adam_step_count()), step);
    }
    assert_ne!(before, core.world_train.read_params(&[encoder_name]));
    let path = std::path::PathBuf::from(std::env::var_os("KINDLE_RGB96_CHECKPOINT").unwrap());
    assert!(!path.exists());
    core.save_checkpoint(&path).unwrap();
    let restored = DreamerCore::restore_with_gpu(
        &path,
        Arc::clone(&core.gpu),
        read_checkpoint_metadata(&path).unwrap(),
    )
    .unwrap();
    assert_eq!(restored.config, config);
    assert_eq!(restored.replay_len(), 0);
    for (source, target) in [
        (&core.world_train, &restored.world_train),
        (&core.behavior_train, &restored.behavior_train),
        (&core.behavior_slow, &restored.behavior_slow),
    ] {
        assert_eq!(source.adam_step_count(), target.adam_step_count());
        let names = source.param_names();
        let values = source.read_params(&names);
        assert!(values.iter().flatten().all(|x| x.is_finite()));
        assert_eq!(values, target.read_params(&names));
        for name in names.into_iter().filter(|name| source.has_param_grad(name)) {
            let mut a = vec![0.0; source.param_size(name).unwrap()];
            let mut b = vec![0.0; a.len()];
            source.read_adam_m(name, &mut a);
            target.read_adam_m(name, &mut b);
            assert_eq!(a, b, "first moment {name}");
            assert!(a.iter().all(|x| x.is_finite()));
            source.read_adam_v(name, &mut a);
            target.read_adam_v(name, &mut b);
            assert_eq!(a, b, "second moment {name}");
            assert!(a.iter().all(|x| x.is_finite()));
        }
    }
    eprintln!(
        "RGB96: two synthetic updates and exact parameter/moment/geometry restore; zero Atari actions"
    );
}
