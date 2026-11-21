use super::*;

#[test]
#[ignore = "requires separately guarded completed replay-value target and restore qualification"]
fn completed_values_reach_native_loss_and_restore_without_new_parameters() {
    let mut config = DreamerConfig::tiny(3);
    config.replay_value_completed = true;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 0.25;
    let mut candidate = DreamerCore::new(config.clone()).unwrap();
    assert_eq!(
        candidate.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let mut control_config = config.clone();
    control_config.replay_value_completed = false;
    let mut control = DreamerCore::with_gpu(control_config, Arc::clone(&candidate.gpu));
    for (left, right) in [
        (&candidate.world_train, &control.world_train),
        (&candidate.behavior_train, &control.behavior_train),
    ] {
        let names = left.param_names();
        assert_eq!(names, right.param_names());
        assert_eq!(left.read_params(&names), right.read_params(&names));
    }
    for core in [&mut candidate, &mut control] {
        for episode in 0..3 {
            core.begin_episode(Observation::from_vec(vec![0.2; Observation::LEN]));
            for step in 1..=10 {
                core.act(ActionMode::Greedy, Some(&[true, false, false]));
                core.observe(
                    Observation::from_vec(vec![step as f32 * 0.02; Observation::LEN]),
                    Reward {
                        extrinsic: if step % 3 == 0 { 2.0 } else { 0.0 },
                        intrinsic: 0.0,
                    },
                    FrameFlags {
                        is_last: step == 10,
                        is_terminal: step == 10 && episode != 1,
                        ..FrameFlags::default()
                    },
                );
            }
        }
    }
    let mut batch = candidate
        .replay
        .sample(&config, &mut candidate.rngs.replay)
        .unwrap();
    let same = control
        .replay
        .sample(&control.config, &mut control.rngs.replay)
        .unwrap();
    assert_eq!(batch.observations, same.observations);
    assert_eq!(batch.flags, same.flags);
    // Include a deliberate unavailable label so native fallback is exercised
    // independently of which completed/timeout stream sections were sampled.
    batch.completed_returns[0][0] = None;
    let posterior = candidate.sample_posterior_batch(&batch);
    let control_posterior = control.sample_posterior_batch(&batch);
    let target = candidate.imagine_and_target(&batch, &posterior);
    let baseline = control.imagine_and_target(&batch, &control_posterior);
    assert_eq!(target.action_target, baseline.action_target);
    assert_eq!(target.imagined_value_target, baseline.imagined_value_target);
    assert_eq!(target.replay_weight, baseline.replay_weight);
    assert_eq!(target.replay_slow_target, baseline.replay_slow_target);
    let mut replaced = 0;
    let mut different = 0;
    for time in 0..config.batch_length - 1 {
        for lane in 0..config.batch_size {
            let row = time * config.batch_size + lane;
            let range = row * config.value_bins..(row + 1) * config.value_bins;
            let value = batch.completed_value_target(&config, time, lane);
            if let Some(value) = value {
                let mut encoded = vec![0.0; config.value_bins];
                candidate.bins.encode(value, &mut encoded);
                assert_eq!(target.replay_value_target[range.clone()], encoded);
                assert_eq!(target.replay_weight[row], 1.0);
                replaced += 1;
                different += usize::from(
                    target.replay_value_target[range.clone()]
                        != baseline.replay_value_target[range],
                );
            } else {
                assert_eq!(
                    target.replay_value_target[range.clone()],
                    baseline.replay_value_target[range]
                );
            }
        }
    }
    assert!(replaced > 0 && different > 0);
    assert_eq!(target.completed_value_rows, replaced);
    assert_eq!(baseline.completed_value_rows, 0);
    let metrics = candidate.train_behavior(&target);
    let baseline_metrics = control.train_behavior(&baseline);
    assert_eq!(metrics.completed_value_rows, Some(replaced));
    assert_eq!(baseline_metrics.completed_value_rows, None);
    assert!(metrics.total_loss.is_finite() && baseline_metrics.total_loss.is_finite());
    let name = "behavior.value.out.weight";
    assert_ne!(
        candidate.behavior_train.read_params(&[name]),
        control.behavior_train.read_params(&[name])
    );
    candidate.sync_behavior_inference();
    for update in 1..=2 {
        let report = candidate.learn().expect("ready synthetic replay");
        assert_eq!(candidate.learner_step, update);
        assert!(report.world.total_loss.is_finite() && report.behavior.total_loss.is_finite());
        assert!(report.behavior.completed_value_rows.is_some());
    }
    let checkpoint =
        std::path::PathBuf::from(std::env::var("KINDLE_COMPLETED_VALUE_CHECKPOINT_DIR").unwrap());
    assert!(!checkpoint.exists());
    candidate.save_checkpoint(&checkpoint).unwrap();
    let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
    let restored =
        DreamerCore::restore_with_gpu(&checkpoint, Arc::clone(&candidate.gpu), metadata).unwrap();
    assert_eq!(restored.config(), &config);
    assert_eq!(restored.learner_step, candidate.learner_step);
    assert_eq!(restored.replay.len(), 0);
    for (left, right) in [
        (&candidate.world_train, &restored.world_train),
        (&candidate.behavior_train, &restored.behavior_train),
    ] {
        let names = left.param_names();
        assert_eq!(names, right.param_names());
        assert_eq!(left.read_params(&names), right.read_params(&names));
    }
}
