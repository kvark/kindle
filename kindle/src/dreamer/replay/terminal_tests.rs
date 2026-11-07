use super::{tests::frame, *};
use rand::{SeedableRng, rngs::StdRng};

fn append(replay: &mut SequenceReplay, config: &DreamerConfig, stream: usize, index: usize) {
    let mut arrival = frame(index, config);
    arrival.flags.is_terminal = matches!(index % 23, 9 | 10 | 19);
    arrival.flags.is_last = arrival.flags.is_terminal || index % 23 == 5;
    replay.push_stream(stream, arrival, config);
}

#[test]
fn terminal_ranges_match_exhaustive_windows_through_eviction_and_context() {
    for context in [1, 2, 5] {
        let mut config = DreamerConfig::tiny(3);
        config.replay_context = context;
        let mut replay = SequenceReplay::with_streams(43, 3);
        for index in 0..100 {
            for stream in 0..3 {
                append(&mut replay, &config, stream, index);
                let actual: Vec<_> = replay
                    .terminal_start_ranges(&config)
                    .iter()
                    .flat_map(|&(lane, first, last)| (first..=last).map(move |start| (lane, start)))
                    .collect();
                let mut expected = Vec::new();
                for (lane, history) in replay.streams.iter().enumerate() {
                    let required = context + config.batch_length;
                    for (start, window) in history
                        .frames
                        .iter()
                        .collect::<Vec<_>>()
                        .windows(required)
                        .enumerate()
                    {
                        if window[context..].iter().any(|f| f.flags.is_terminal) {
                            expected.push((lane, history.oldest_frame() + start as u64));
                        }
                    }
                    let terminal_indices: Vec<_> = history
                        .frames
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.flags.is_terminal)
                        .map(|(offset, _)| history.oldest_frame() + offset as u64)
                        .collect();
                    assert_eq!(
                        history.terminal_frames.iter().copied().collect::<Vec<_>>(),
                        terminal_indices
                    );
                }
                assert_eq!(
                    actual, expected,
                    "context={context}, arrival={index}, stream={stream}"
                );
            }
        }
    }
}

#[test]
fn terminal_sampling_keeps_fresh_priority_then_samples_every_eligible_window() {
    let mut config = DreamerConfig::tiny(3);
    config.terminal_replay_fraction = Some(1.0);
    config.batch_size = 1;
    config.replay_context = 3;
    let mut replay = SequenceReplay::with_streams(100, 2);
    for index in 0..24 {
        for stream in 0..2 {
            append(&mut replay, &config, stream, index);
        }
    }
    let fresh = replay.fresh_starts.clone();
    let mut rng = StdRng::seed_from_u64(293);
    for (stream, start) in fresh {
        let batch = replay.sample(&config, &mut rng).unwrap();
        let report = batch.sampling.unwrap();
        assert_eq!(
            (
                report.fresh_sequences,
                report.terminal_sequences,
                report.uniform_sequences
            ),
            (1, 0, 0)
        );
        assert_eq!(
            batch.frame_indices[0][0],
            (stream, start as usize + config.replay_context)
        );
    }
    let expected: std::collections::BTreeSet<_> = replay
        .terminal_start_ranges(&config)
        .iter()
        .flat_map(|&(lane, first, last)| (first..=last).map(move |start| (lane, start as usize)))
        .collect();
    let mut counts = std::collections::BTreeMap::new();
    for _ in 0..6000 {
        let batch = replay.sample(&config, &mut rng).unwrap();
        let report = batch.sampling.unwrap();
        assert_eq!(
            (
                report.fresh_sequences,
                report.terminal_sequences,
                report.uniform_sequences
            ),
            (0, 1, 0)
        );
        assert_eq!(report.eligible_terminal_sequences, expected.len());
        let terminals = batch.flags.iter().filter(|f| f[0].is_terminal).count();
        assert!(terminals > 0);
        assert_eq!(terminals, report.terminal_rows);
        let (lane, index) = batch.frame_indices[0][0];
        let start = (lane, index - config.replay_context);
        assert!(expected.contains(&start));
        *counts.entry(start).or_insert(0_usize) += 1;
    }
    assert_eq!(counts.len(), expected.len());
    let mean = 6000.0 / expected.len() as f64;
    // Overlapping terminal windows must not receive multiplicity weighting.
    for &count in counts.values() {
        assert!((count as f64 - mean).abs() < 6.0 * mean.sqrt());
    }
}

#[test]
fn disabled_terminal_sampling_preserves_samples_and_rng() {
    let config = DreamerConfig::tiny(3);
    let mut explicit_zero = config.clone();
    explicit_zero.terminal_replay_fraction = Some(0.0);
    let (mut left, mut right) = (SequenceReplay::new(60), SequenceReplay::new(60));
    for index in 0..100 {
        append(&mut left, &config, 0, index);
        append(&mut right, &explicit_zero, 0, index);
    }
    let (mut a, mut b) = (StdRng::seed_from_u64(101), StdRng::seed_from_u64(101));
    for _ in 0..120 {
        let x = left.sample(&config, &mut a).unwrap();
        let y = right.sample(&explicit_zero, &mut b).unwrap();
        assert!(x.sampling.is_none());
        assert_eq!(y.sampling.unwrap().terminal_sequences, 0);
        assert_eq!(x.frame_indices, y.frame_indices);
        assert_eq!(x.flags, y.flags);
        assert_eq!(x.observations, y.observations);
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
}

#[test]
fn absent_terminal_candidates_preserve_uniform_rng_and_ignore_timeouts() {
    let config = DreamerConfig::tiny(3);
    let mut targeted = config.clone();
    targeted.terminal_replay_fraction = Some(0.25);
    let (mut left, mut right) = (SequenceReplay::new(60), SequenceReplay::new(60));
    for index in 0..30 {
        let mut arrival = frame(index, &config);
        arrival.flags.is_last = index % 5 == 4;
        left.push(arrival.clone(), &config);
        right.push(arrival, &targeted);
    }
    let (mut a, mut b) = (StdRng::seed_from_u64(99), StdRng::seed_from_u64(99));
    for _ in 0..50 {
        let x = left.sample(&config, &mut a).unwrap();
        let y = right.sample(&targeted, &mut b).unwrap();
        let report = y.sampling.unwrap();
        assert_eq!(
            (
                report.terminal_sequences,
                report.terminal_rows,
                report.eligible_terminal_sequences
            ),
            (0, 0, 0)
        );
        assert_eq!(x.frame_indices, y.frame_indices);
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
}

#[test]
fn terminal_mixture_reports_routes_and_training_rows_with_prefix() {
    let mut config = DreamerConfig::tiny(3);
    config.replay_context = 5;
    config.terminal_replay_fraction = Some(0.25);
    let mut replay = SequenceReplay::new(100);
    for index in 0..80 {
        append(&mut replay, &config, 0, index);
    }
    replay.fresh_starts.clear();
    let mut rng = StdRng::seed_from_u64(52);
    let mut selected = 0;
    for _ in 0..1000 {
        let batch = replay.sample(&config, &mut rng).unwrap();
        let report = batch.sampling.unwrap();
        assert_eq!(report.fresh_sequences, 0);
        assert_eq!(
            report.terminal_sequences + report.uniform_sequences,
            config.batch_size
        );
        assert_eq!(
            report.terminal_rows,
            batch
                .flags
                .iter()
                .flatten()
                .filter(|f| f.is_terminal)
                .count()
        );
        assert!(report.terminal_rows >= report.terminal_sequences);
        selected += report.terminal_sequences;
    }
    assert!(
        (400..600).contains(&selected),
        "{selected} terminal choices out of 2000"
    );
}

#[test]
fn terminal_config_validates_and_roundtrips_without_changing_legacy_json() {
    let mut config = DreamerConfig::tiny(3);
    let legacy = serde_json::to_value(&config).unwrap();
    assert!(legacy.get("terminal_replay_fraction").is_none());
    assert!(
        serde_json::from_value::<DreamerConfig>(legacy)
            .unwrap()
            .terminal_replay_fraction
            .is_none()
    );
    for fraction in [0.0, 0.25, 1.0] {
        config.terminal_replay_fraction = Some(fraction);
        config.check().unwrap();
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serde_json::from_value::<DreamerConfig>(value)
                .unwrap()
                .terminal_replay_fraction,
            Some(fraction)
        );
    }
    for fraction in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        config.terminal_replay_fraction = Some(fraction);
        assert!(
            config
                .check()
                .unwrap_err()
                .contains("terminal_replay_fraction")
        );
    }
}
