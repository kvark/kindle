use super::*;

#[test]
#[ignore = "requires separately guarded actor dynamics snapshot and restore qualification"]
fn dynamics_actor_uses_preupdate_snapshot_and_restores_optimizer_owner() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_dynamics_scale = Some(0.5);
    config.disagreement_bonus = true;
    config.intrinsic_reward_scale = 1.0;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 0.25;
    let mut core = DreamerCore::new(config.clone()).unwrap();
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let names = core
        .behavior_train
        .plan()
        .input_buffers
        .iter()
        .filter_map(|(name, _)| name.strip_prefix("dynamics_parameter_").map(str::to_owned))
        .collect::<Vec<_>>();
    assert!(!names.is_empty());
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    for name in &names {
        assert!(core.behavior_train.param_size(name).is_none());
    }
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
        let before = core.world_train.read_params(&names);
        let report = core.learn().expect("ready synthetic replay");
        assert_eq!(report.learner_step, update);
        assert!(report.behavior.dynamics_actor_loss.unwrap().is_finite());
        assert!(report.timing.dynamics_snapshot_seconds.unwrap() > 0.0);
        for (name, expected) in names.iter().zip(&before) {
            let mut actual = vec![0.0; expected.len()];
            let buffer = core
                .behavior_train
                .input_buffer(&format!("dynamics_parameter_{name}"))
                .unwrap();
            core.readback.read_regions(&mut [(buffer, &mut actual)]);
            assert_eq!(&actual, expected, "private pre-update snapshot {name}");
        }
        assert_ne!(core.world_train.read_params(&names), before);
        assert_eq!(u64::from(core.behavior_train.adam_step_count()), update);
        assert_eq!(u64::from(core.world_train.adam_step_count()), update);
        check_production_return(&mut core, &report);
        for name in [
            "deter".to_owned(),
            "stoch".to_owned(),
            "action_uniforms_0".to_owned(),
            format!("latent_uniforms_{}", config.imagination_length - 1),
        ] {
            let target = format!("dynamics_{name}");
            let mut source_values = vec![
                0.0;
                core.imagination
                    .slot_size(ExternalSlot::Input(&name))
                    .unwrap()
                    / 4
            ];
            let mut target_values = vec![0.0; source_values.len()];
            core.readback.read_regions(&mut [
                (
                    core.imagination.input_buffer(&name).unwrap(),
                    &mut source_values,
                ),
                (
                    core.behavior_train.input_buffer(&target).unwrap(),
                    &mut target_values,
                ),
            ]);
            assert_eq!(source_values, target_values, "snapshot input {name}");
        }
    }
    let checkpoint =
        std::path::PathBuf::from(std::env::var("KINDLE_ACTOR_DYNAMICS_CHECKPOINT").unwrap());
    assert!(!checkpoint.exists());
    core.save_checkpoint(&checkpoint).unwrap();
    let restored = DreamerCore::restore_with_gpu(
        &checkpoint,
        Arc::clone(&core.gpu),
        read_checkpoint_metadata(&checkpoint).unwrap(),
    )
    .unwrap();
    assert_eq!(restored.config.actor_dynamics_scale, Some(0.5));
    assert_eq!(restored.replay_len(), 0);
    for (source, target) in [
        (&core.world_train, &restored.world_train),
        (&core.behavior_train, &restored.behavior_train),
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
        "Two synthetic updates: pre-update private snapshots, shared draws and optimizer restore pass; zero Atari actions"
    );
}

fn check_production_return(core: &mut DreamerCore, report: &LearnReport) {
    let config = &core.config;
    let starts = config.batch_size * config.batch_length;
    let horizon = config.imagination_length;
    let rows = starts * (horizon + 1);
    let mut reward_logits = vec![0.0; rows * config.value_bins];
    let mut value_logits = vec![0.0; reward_logits.len()];
    let mut continuation = vec![0.0; rows];
    let mut bonus = vec![0.0; starts * horizon];
    core.readback.read(
        &core.imagination,
        &mut [
            (world::IMAGINATION_REWARD, &mut reward_logits),
            (world::IMAGINATION_VALUE, &mut value_logits),
            (world::IMAGINATION_CONTINUATION, &mut continuation),
            (world::IMAGINATION_BONUS, &mut bonus),
        ],
    );
    let rewards = decode_rows(&reward_logits, rows, &core.bins);
    let values = decode_rows(&value_logits, rows, &core.bins);
    let mut expected = 0.0f64;
    for start in 0..starts {
        let reward = (0..=horizon)
            .map(|t| {
                rewards[t * starts + start]
                    + if t > 0 {
                        config.intrinsic_reward_scale * bonus[(t - 1) * starts + start]
                    } else {
                        0.0
                    }
            })
            .collect::<Vec<_>>();
        let cont = (0..=horizon)
            .map(|t| continuation[t * starts + start])
            .collect::<Vec<_>>();
        let terminal = cont.iter().map(|c| 1.0 - c).collect::<Vec<_>>();
        let value = (0..=horizon)
            .map(|t| values[t * starts + start])
            .collect::<Vec<_>>();
        let returns = lambda_returns(
            &vec![0.0; horizon + 1],
            &terminal,
            &reward,
            &value,
            &value,
            1.0,
            config.lambda,
        );
        let weights = continuation_weights(&cont[..horizon]);
        expected -= returns
            .iter()
            .zip(weights)
            .map(|(&r, w)| f64::from(r) * f64::from(w))
            .sum::<f64>();
    }
    expected /= (starts * horizon) as f64 * f64::from(report.behavior.return_scale);
    let actual = f64::from(report.behavior.dynamics_actor_loss.unwrap());
    assert!(
        (actual - expected).abs() < 5e-3 * (1.0 + expected.abs()),
        "production return alignment: {actual} != {expected}"
    );
    eprintln!(
        "production normalized return loss: actual={actual} existing-target-reference={expected}"
    );
}
