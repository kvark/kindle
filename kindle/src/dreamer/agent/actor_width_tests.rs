use super::*;
use meganeura::{Graph, graph::Op};
use std::collections::BTreeMap;

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

fn graphs(config: &DreamerConfig) -> Vec<Graph> {
    vec![
        behavior::build_training_graph(
            config,
            config.batch_size * config.batch_length * config.imagination_length,
            config.batch_size * (config.batch_length - 1),
        ),
        behavior::build_inference_graph(config, 2),
        behavior::build_actor_inference_graph(config, 2),
        world::build_training_graph(config, config.world_backprop_length),
        world::build_imagination_graph(config),
    ]
}

#[test]
fn width_override_preserves_legacy_graphs_and_serialization() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_critic_gradient = true;
    assert!(
        serde_json::to_value(&config)
            .unwrap()
            .get("actor_hidden_units")
            .is_none()
    );
    let original = graphs(&config);
    config.actor_hidden_units = Some(config.network().units);
    for (a, b) in original.iter().zip(graphs(&config)) {
        assert_eq!(format!("{:?}", a.nodes()), format!("{:?}", b.nodes()));
        assert_eq!(a.outputs(), b.outputs());
    }
    let encoded = serde_json::to_value(&config).unwrap();
    let restored: DreamerConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored.actor_hidden_units, config.actor_hidden_units);
    config.actor_hidden_units = Some(0);
    assert!(config.check().unwrap_err().contains("actor_hidden_units"));
}

#[test]
fn width_changes_only_actor_shapes_in_all_graph_owners() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_critic_gradient = true;
    let original = graphs(&config);
    let units = config.network().units * 2;
    config.actor_hidden_units = Some(units);
    for (a, b) in original.iter().zip(graphs(&config)) {
        let (a, b) = (shapes(a), shapes(&b));
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (name, shape) in &a {
            if !name.starts_with("behavior.actor.") {
                assert_eq!(shape, &b[name], "{name}");
            }
        }
        assert_eq!(
            b["behavior.actor.layer0.weight"],
            [config.feature_dim(), units]
        );
        assert_eq!(b["behavior.actor.layer1.weight"], [units, units]);
        assert_eq!(b["behavior.actor.out.weight"], [units, config.action_count]);
    }
}

#[test]
#[ignore = "requires separately guarded actor-width learning and restore qualification"]
fn widened_actor_learns_shares_parameters_and_restores() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_critic_gradient = true;
    config.actor_hidden_units = Some(config.network().units * 2);
    let mut core = DreamerCore::new(config.clone()).unwrap();
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!core.gpu_device().is_software_emulated);
    let actor_name = "behavior.actor.layer0.weight";
    let critic_name = "behavior.value.layer0.weight";
    assert_eq!(
        core.behavior_train.param_size(actor_name),
        Some(config.feature_dim() * config.actor_hidden_units())
    );
    assert_eq!(
        core.behavior_train.param_size(critic_name),
        Some(config.feature_dim() * config.network().units)
    );
    let initial = core.behavior_train.read_params(&[actor_name, critic_name]);
    core.begin_episode(Observation::from_vec(vec![0.2; Observation::LEN]));
    for time in 0..16 {
        core.act(ActionMode::Sample, None);
        core.observe(
            Observation::from_vec(vec![0.1 + time as f32 * 0.01; Observation::LEN]),
            Reward {
                extrinsic: if time % 3 == 0 { 1.0 } else { 0.0 },
                intrinsic: 0.0,
            },
            FrameFlags::default(),
        );
    }
    for update in 1..=2 {
        let report = core.learn().expect("synthetic replay ready");
        assert_eq!(report.learner_step, update);
        assert!(report.behavior.policy_loss.is_finite());
        let actor = core.behavior_train.read_params(&[actor_name]);
        for session in [&core.policy_live, &core.imagination, &core.world_train] {
            assert_eq!(session.read_params(&[actor_name]), actor);
        }
        assert_eq!(u64::from(core.behavior_train.adam_step_count()), update);
    }
    let trained = core.behavior_train.read_params(&[actor_name, critic_name]);
    assert_ne!(trained[0], initial[0]);
    assert_ne!(trained[1], initial[1]);
    let path = std::path::PathBuf::from(std::env::var_os("KINDLE_ACTOR_WIDTH_CHECKPOINT").unwrap());
    assert!(!path.exists());
    core.save_checkpoint(&path).unwrap();
    let restored = DreamerCore::restore_with_gpu(
        &path,
        Arc::clone(&core.gpu),
        read_checkpoint_metadata(&path).unwrap(),
    )
    .unwrap();
    assert_eq!(
        restored.config.actor_hidden_units,
        config.actor_hidden_units
    );
    assert_eq!(restored.replay_len(), 0);
    for (source, target) in [
        (&core.world_train, &restored.world_train),
        (&core.behavior_train, &restored.behavior_train),
        (&core.behavior_slow, &restored.behavior_slow),
    ] {
        assert_eq!(source.adam_step_count(), target.adam_step_count());
        let names = source.param_names();
        assert_eq!(source.read_params(&names), target.read_params(&names));
        for name in names.into_iter().filter(|name| source.has_param_grad(name)) {
            let mut a = vec![0.0; source.param_size(name).unwrap()];
            let mut b = vec![0.0; a.len()];
            source.read_adam_m(name, &mut a);
            target.read_adam_m(name, &mut b);
            assert_eq!(a, b, "first moment {name}");
            source.read_adam_v(name, &mut a);
            target.read_adam_v(name, &mut b);
            assert_eq!(a, b, "second moment {name}");
        }
    }
    eprintln!(
        "Widened actor: two finite synthetic updates, shared weights, unchanged critic width and exact optimizer restore; zero Atari actions"
    );
}
