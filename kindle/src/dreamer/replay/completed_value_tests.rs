use super::{tests::frame, *};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn completed_values_match_forward_sums_without_actor_and_preserve_sampling() {
    for context in [1, 3] {
        let mut config = DreamerConfig::tiny(3);
        config.replay_value_completed = true;
        config.replay_context = context;
        config.horizon = 2;
        config.extrinsic_reward_scale = 2.0;
        config.intrinsic_reward_scale = 3.0;
        config.check().unwrap();
        let mut legacy = config.clone();
        legacy.replay_value_completed = false;
        let mut replay = SequenceReplay::with_streams(64, 2);
        let mut control = SequenceReplay::with_streams(64, 2);
        for index in 0..20 {
            for lane in 0..2 {
                let mut arrival = frame(index, &config);
                arrival.reward = Reward {
                    extrinsic: index as f32 - 5.0,
                    intrinsic: 1.0,
                };
                arrival.flags.is_last = index == 19 && lane == 0;
                arrival.flags.is_terminal = arrival.flags.is_last;
                replay.push_stream(lane, arrival.clone(), &config);
                control.push_stream(lane, arrival, &legacy);
            }
        }
        let mut rng = StdRng::seed_from_u64(81);
        let mut control_rng = rng.clone();
        let mut completed = 0;
        let mut fallback = 0;
        for _ in 0..20 {
            let batch = replay.sample(&config, &mut rng).unwrap();
            let same = control.sample(&legacy, &mut control_rng).unwrap();
            assert_eq!(batch.observations, same.observations);
            assert_eq!(batch.previous_actions, same.previous_actions);
            assert_eq!(batch.flags, same.flags);
            assert_eq!(batch.frame_indices, same.frame_indices);
            for time in 0..config.batch_length - 1 {
                for row in 0..config.batch_size {
                    let (stream, _) = batch.frame_indices[time][row];
                    let index = batch.observations[time][row * config.observation_dim()] as usize;
                    let target = batch.completed_value_target(&config, time, row);
                    assert!(same.completed_value_target(&legacy, time, row).is_none());
                    if stream == 0 && index < 19 {
                        let expected = (index + 1..20)
                            .enumerate()
                            .map(|(offset, i)| {
                                (2.0 * (i as f64 - 5.0) + 3.0) * 0.5_f64.powi(offset as i32)
                            })
                            .sum::<f64>() as f32;
                        assert_eq!(target, Some(expected));
                        completed += 1;
                    } else {
                        assert_eq!(target, None);
                        fallback += 1;
                    }
                }
            }
        }
        assert!(completed > 0 && fallback > 0);
    }
}

#[test]
fn completed_values_keep_evicted_suffix_and_do_not_complete_timeouts_or_cross_resets() {
    let mut config = DreamerConfig::tiny(3);
    config.replay_value_completed = true;
    config.horizon = 2;
    let mut replay = SequenceReplay::new(3);
    for index in 0..7 {
        let mut arrival = frame(index, &config);
        arrival.flags.is_last = index == 6;
        arrival.flags.is_terminal = index == 6;
        replay.push(arrival, &config);
    }
    let values = |r: &SequenceReplay| {
        r.streams[0]
            .frames
            .iter()
            .map(|f| f.completed_return)
            .collect::<Vec<_>>()
    };
    assert_eq!(values(&replay), [Some(8.0), Some(6.0), Some(0.0)]);
    let mut replay = SequenceReplay::new(16);
    replay.push(frame(0, &config), &config);
    let mut timeout = frame(1, &config);
    timeout.flags.is_last = true;
    replay.push(timeout, &config);
    replay.push(frame(0, &config), &config);
    let mut terminal = frame(2, &config);
    terminal.flags.is_last = true;
    terminal.flags.is_terminal = true;
    replay.push(terminal, &config);
    assert_eq!(values(&replay), [None, None, Some(2.0), Some(0.0)]);
    replay.push(frame(0, &config), &config);
    replay.push(frame(3, &config), &config);
    replay.push(frame(0, &config), &config);
    let mut terminal = frame(4, &config);
    terminal.flags.is_last = true;
    terminal.flags.is_terminal = true;
    replay.push(terminal, &config);
    assert_eq!(&values(&replay)[4..], &[None, None, Some(4.0), Some(0.0)]);
}

#[test]
fn completed_value_selection_masks_last_reset_and_unfinished_rows() {
    let mut config = DreamerConfig::tiny(3);
    config.replay_value_completed = true;
    let mut batch = SequenceBatch {
        flags: vec![vec![FrameFlags::default(); 2]; 4],
        completed_returns: vec![
            vec![Some(-3.0), None],
            vec![Some(4.0), Some(2.0)],
            vec![Some(5.0), Some(0.0)],
            vec![None, None],
        ],
        ..Default::default()
    };
    batch.flags[1][0].is_last = true;
    batch.flags[3][0].is_first = true;
    let targets = (0..3)
        .flat_map(|t| (0..2).map(move |lane| (t, lane)))
        .map(|(t, lane)| batch.completed_value_target(&config, t, lane))
        .collect::<Vec<_>>();
    assert_eq!(
        targets,
        [Some(-3.0), None, None, Some(2.0), None, Some(0.0)]
    );
    config.replay_value_completed = false;
    assert!(
        (0..3).all(|t| (0..2).all(|lane| batch.completed_value_target(&config, t, lane).is_none()))
    );
}
