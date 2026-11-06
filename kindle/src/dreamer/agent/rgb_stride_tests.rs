use super::*;
use crate::dreamer::{ModelSize, ObservationKind, VideoEncoder};
use meganeura::{Graph, graph::Op};
use std::collections::BTreeMap;

fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb64;
    config.rgb_stride2 = true;
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
fn rgb_stride2_roundtrip_and_legacy_defaults() {
    let legacy = DreamerConfig::tiny(3);
    let encoded = serde_json::to_value(&legacy).unwrap();
    assert!(encoded.get("rgb_stride2").is_none());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(encoded).unwrap(),
        legacy
    );
    let mut invalid = legacy;
    invalid.rgb_stride2 = true;
    assert!(
        invalid
            .check()
            .unwrap_err()
            .contains("require the learned RGB encoder")
    );
    let mut config = config();
    for (kind, side) in [
        (ObservationKind::Rgb32, 32),
        (ObservationKind::Rgb48, 48),
        (ObservationKind::Rgb64, 64),
        (ObservationKind::Rgb96, 96),
    ] {
        config.observation_kind = kind;
        config.validate();
        let mut plain = config.clone();
        plain.rgb_stride2 = false;
        assert_eq!(config.observation_dim(), 3 * side * side);
        assert_eq!(config.observation_shape(2), plain.observation_shape(2));
        assert_eq!(
            config.encoded_observation_dim(),
            plain.encoded_observation_dim()
        );
        assert_eq!(config.feature_dim(), plain.feature_dim());
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["rgb_stride2"], true);
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(json).unwrap(),
            config
        );
    }
    config.video_encoder = Some(VideoEncoder::Frozen);
    assert!(
        config
            .check()
            .unwrap_err()
            .contains("require the learned RGB encoder")
    );
}

#[test]
fn rgb_stride2_preserves_parameters_and_control_graphs() {
    for size in [ModelSize::Tiny, ModelSize::Size12M] {
        let mut candidate = config();
        candidate.model_size = size;
        let mut plain = candidate.clone();
        plain.rgb_stride2 = false;
        let a = world::build_imagination_graph(&plain);
        let b = world::build_imagination_graph(&candidate);
        assert_eq!(format!("{:?}", a.nodes()), format!("{:?}", b.nodes()));
        assert_eq!(a.outputs(), b.outputs());
        let a = world::build_posterior_graph(&plain);
        let b = world::build_posterior_graph(&candidate);
        assert_eq!(shapes(&a), shapes(&b));
        assert_eq!(
            a.outputs()
                .iter()
                .map(|id| &a.node(*id).ty.shape)
                .collect::<Vec<_>>(),
            b.outputs()
                .iter()
                .map(|id| &b.node(*id).ty.shape)
                .collect::<Vec<_>>()
        );
        let a = behavior::build_training_graph(&plain, 6, 2);
        let b = behavior::build_training_graph(&candidate, 6, 2);
        assert_eq!(format!("{:?}", a.nodes()), format!("{:?}", b.nodes()));
        assert_eq!(a.outputs(), b.outputs());
        assert_eq!(
            shapes(&world::build_training_graph(&plain, 4)),
            shapes(&world::build_training_graph(&candidate, 4))
        );
    }
}

#[test]
fn rgb_stride2_world_graphs_keep_encoder_gradient_owners() {
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
#[ignore = "requires separately guarded RGB stride-two two-update and exact-restore qualification"]
fn rgb_stride2_learns_and_restores_parameters_moments_and_geometry() {
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
            64,
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
    let path = std::path::PathBuf::from(std::env::var_os("KINDLE_RGB_STRIDE2_CHECKPOINT").unwrap());
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
        "RGB stride-two: two synthetic updates and exact parameter/moment/geometry restore; zero Atari actions"
    );
}
