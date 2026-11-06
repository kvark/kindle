use super::*;

#[test]
#[ignore = "requires separately declared guarded residual actor qualification"]
fn residual_native_learning_sharing_and_restore() {
    let config = super::super::residual_actor_tests::config();
    let core = DreamerCore::new(config.clone()).unwrap();
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!core.gpu_device().is_software_emulated);
    let mut vector = VectorCore::new(core, 2);
    let names = [
        "behavior.actor.out.weight",
        "behavior.actor.residual.weight",
        "behavior.value.out.weight",
    ];
    let initial = vector.learner.behavior_train.read_params(&names);
    assert!(initial[1].iter().all(|&x| x == 0.0));
    let observation = |stream: usize, time: usize| {
        Observation::from_vec(
            (0..config.observation_dim())
                .map(|i| ((i + stream * 11 + time * 7) % 31) as f32 / 31.0)
                .collect(),
        )
    };
    vector.ingest(
        (0..2)
            .map(|stream| {
                (
                    stream,
                    observation(stream, 0),
                    FrameFlags {
                        is_first: true,
                        ..Default::default()
                    },
                    Reward::default(),
                )
            })
            .collect(),
    );
    for time in 1..=16 {
        vector
            .act_with_expert_labels(
                ActionMode::Sample,
                &[Some(time % 3), Some((time + 1) % 3)],
                None,
            )
            .unwrap();
        vector.ingest(
            (0..2)
                .map(|stream| {
                    (
                        stream,
                        observation(stream, time),
                        FrameFlags::default(),
                        Reward {
                            extrinsic: if time % 3 == 0 { 1.0 } else { 0.0 },
                            intrinsic: 0.0,
                        },
                    )
                })
                .collect(),
        );
    }
    for update in 1..=2 {
        let report = vector.learn().expect("synthetic labeled replay ready");
        assert_eq!(report.learner_step, update);
        assert!(report.behavior.policy_loss.is_finite());
        assert!(
            report
                .behavior
                .expert_imitation
                .as_ref()
                .unwrap()
                .loss
                .is_finite()
        );
        let core = &vector.learner;
        for name in [
            "behavior.actor.out.weight",
            "behavior.actor.residual.weight",
            "behavior.actor.residual.bias",
        ] {
            let source = core.behavior_train.read_params(&[name]);
            for target in [&core.policy_live, &core.imagination, &vector.policy] {
                assert_eq!(target.read_params(&[name]), source, "shared {name}");
            }
        }
        assert!(
            core.world_train
                .param_size("behavior.actor.residual.weight")
                .is_none()
        );
        assert_eq!(
            core.world_train.read_params(&[names[0]]),
            core.behavior_train.read_params(&[names[0]])
        );
        assert_eq!(u64::from(core.behavior_train.adam_step_count()), update);
    }
    let core = &mut vector.learner;
    for (name, (before, after)) in names
        .iter()
        .zip(initial.iter().zip(core.behavior_train.read_params(&names)))
    {
        assert_ne!(before, &after, "parameter changed: {name}");
        assert!(after.iter().all(|x| x.is_finite()));
    }
    let checkpoint =
        Path::new(&std::env::var_os("KINDLE_RESIDUAL_ACTOR_CHECKPOINT").unwrap()).to_owned();
    assert!(!checkpoint.exists());
    core.save_checkpoint(&checkpoint).unwrap();
    let restored = DreamerCore::restore_with_gpu(
        &checkpoint,
        Arc::clone(&core.gpu),
        read_checkpoint_metadata(&checkpoint).unwrap(),
    )
    .unwrap();
    assert!(restored.config.actor_residual);
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
        "Residual actor: two finite labeled synthetic updates, base/residual/critic changes, shared acting/imagination weights and exact restore; zero Atari actions"
    );
}
