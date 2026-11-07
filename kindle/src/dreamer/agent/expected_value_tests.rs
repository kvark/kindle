use super::*;

#[test]
#[ignore = "requires separately guarded expected-value branch and restore qualification"]
fn expected_value_actor_matches_branches_and_restores_optimizer_owner() {
    let mut config = DreamerConfig::tiny(3);
    config.actor_expected_value = true;
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
        .filter_map(|(name, _)| name.strip_prefix("expected_parameter_").map(str::to_owned))
        .collect::<Vec<_>>();
    assert!(!names.is_empty());
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    for name in &names {
        assert!(core.behavior_train.param_size(name).is_none());
    }
    for (session, name) in [
        (&mut core.world_train, "world.reward.out.weight"),
        (&mut core.behavior_train, "behavior.value.out.weight"),
    ] {
        let values = (0..session.param_size(name).unwrap())
            .map(|i| ((i % 13) as f32 - 6.0) * 0.003)
            .collect::<Vec<_>>();
        session.set_parameter(name, &values);
    }
    core.sync_world_inference();
    core.sync_behavior_inference();
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
        let mut reference = Reference::new(&core);
        let report = core.learn().expect("ready synthetic replay");
        assert_eq!(report.learner_step, update);
        assert!(
            report
                .behavior
                .expected_value_actor_loss
                .unwrap()
                .is_finite()
        );
        assert!(report.timing.expected_value_snapshot_seconds.unwrap() > 0.0);
        for (name, expected) in names.iter().zip(&before) {
            let mut actual = vec![0.0; expected.len()];
            let buffer = core
                .behavior_train
                .input_buffer(&format!("expected_parameter_{name}"))
                .unwrap();
            core.readback.read_regions(&mut [(buffer, &mut actual)]);
            assert_eq!(&actual, expected, "private pre-update snapshot {name}");
        }
        assert_ne!(core.world_train.read_params(&names), before);
        assert_eq!(u64::from(core.behavior_train.adam_step_count()), update);
        assert_eq!(u64::from(core.world_train.adam_step_count()), update);
        reference.check(&mut core, &report);
    }
    let checkpoint =
        std::path::PathBuf::from(std::env::var("KINDLE_ACTOR_EXPECTED_VALUE_CHECKPOINT").unwrap());
    assert!(!checkpoint.exists());
    core.save_checkpoint(&checkpoint).unwrap();
    let restored = DreamerCore::restore_with_gpu(
        &checkpoint,
        Arc::clone(&core.gpu),
        read_checkpoint_metadata(&checkpoint).unwrap(),
    )
    .unwrap();
    assert!(restored.config.actor_expected_value);
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

struct Reference {
    values: Session,
    transition: Session,
    heads: Session,
    critic: Session,
    actor: Session,
    bonus: Session,
}

impl Reference {
    fn new(core: &DreamerCore) -> Self {
        let config = &core.config;
        let rows = config.batch_size * config.batch_length;
        let mut bonus = meganeura::Graph::new();
        let ensemble = super::super::exploration::Disagreement::new(&mut bonus, config);
        let state = bonus.input("state", &[rows, config.feature_dim()]);
        let action = bonus.input("action", &[rows, config.action_count]);
        let result = ensemble.bonus(&mut bonus, state, action);
        bonus.set_outputs(vec![result]);
        let build = |graph: meganeura::Graph| {
            let mut session = build_session(&graph, &core.gpu, Mode::Inference, false);
            let names = session
                .param_names()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            for name in names {
                let source = if name.starts_with("world.") {
                    &core.world_train
                } else {
                    &core.behavior_train
                };
                let values = source.read_params(&[&name]);
                session.set_parameter(&name, &values[0]);
            }
            session
        };
        Self {
            values: build(behavior::build_expected_values_graph(config)),
            transition: build(world::build_transition_graph(config, rows)),
            heads: build(world::build_head_graph(config, rows)),
            critic: build(behavior::build_value_inference_graph(config, rows)),
            actor: build(behavior::build_actor_inference_graph(config, rows)),
            bonus: build(bonus),
        }
    }

    fn check(&mut self, core: &mut DreamerCore, report: &LearnReport) {
        let config = core.config.clone();
        let rows = config.batch_size * config.batch_length;
        let size = config.network();
        let actions = config.action_count;
        let mut inputs = Vec::new();
        for (source, target) in [
            ("deter", "expected_deter"),
            ("stoch", "expected_stoch"),
            ("latent_uniforms_0", "expected_latent_uniforms"),
        ] {
            let mut original = vec![
                0.0;
                core.imagination
                    .slot_size(ExternalSlot::Input(source))
                    .unwrap()
                    / 4
            ];
            let mut actual = vec![0.0; original.len()];
            core.readback.read_regions(&mut [
                (
                    core.imagination.input_buffer(source).unwrap(),
                    &mut original,
                ),
                (
                    core.behavior_train.input_buffer(target).unwrap(),
                    &mut actual,
                ),
            ]);
            assert_eq!(original, actual, "shared input {source}");
            self.values.set_input(target, &actual);
            inputs.push(actual);
        }
        self.values.step();
        self.values.wait();
        let mut start = vec![0.0; rows * config.feature_dim()];
        let mut branch_values = vec![0.0; rows * actions];
        self.values.read_output_by_index(0, &mut start);
        self.values.read_output_by_index(1, &mut branch_values);
        assert_eq!(start, join_features(&inputs[0], &inputs[1], rows, &config));
        assert!(
            branch_values
                .chunks(actions)
                .any(|q| q.iter().any(|v| (*v - q[0]).abs() > 1e-4))
        );
        let bins = TwoHotBins::new(config.value_bins);
        let mut reference_values = vec![0.0; rows * actions];
        for action in 0..actions {
            let mut one_hot = vec![0.0; rows * actions];
            for row in 0..rows {
                one_hot[row * actions + action] = 1.0;
            }
            self.transition.set_input("deter", &inputs[0]);
            self.transition.set_input("stoch", &inputs[1]);
            self.transition.set_input("action", &one_hot);
            self.transition.step();
            self.transition.wait();
            let mut deter = vec![0.0; rows * size.deter];
            let mut logits = vec![0.0; rows * size.stoch * size.classes];
            self.transition.read_output_by_index(0, &mut deter);
            self.transition.read_output_by_index(1, &mut logits);
            let mut stoch = vec![0.0; logits.len()];
            for group in 0..rows * size.stoch {
                let offset = group * size.classes;
                let mut probabilities = vec![0.0; size.classes];
                softmax_unimix(
                    &logits[offset..offset + size.classes],
                    config.unimix,
                    &mut probabilities,
                );
                let selected = (0..size.classes)
                    .max_by(|&a, &b| {
                        let score = |i: usize| probabilities[i].ln() - (-inputs[2][offset + i].ln()).ln();
                        score(a).total_cmp(&score(b))
                    })
                    .unwrap();
                stoch[offset + selected] = 1.0;
            }
            self.heads.set_input("deter", &deter);
            self.heads.set_input("stoch", &stoch);
            self.heads.step();
            self.heads.wait();
            self.critic
                .set_input("feature", &join_features(&deter, &stoch, rows, &config));
            self.critic.step();
            self.critic.wait();
            self.bonus.set_input("state", &start);
            self.bonus.set_input("action", &one_hot);
            self.bonus.step();
            self.bonus.wait();
            let mut reward = vec![0.0; rows * config.value_bins];
            let mut value = vec![0.0; reward.len()];
            let mut continuation = vec![0.0; rows];
            let mut bonus = vec![0.0; rows];
            self.heads.read_output_by_index(0, &mut reward);
            self.heads.read_output_by_index(1, &mut continuation);
            self.critic.read_output_by_index(0, &mut value);
            self.bonus.read_output_by_index(0, &mut bonus);
            let reward = decode_rows(&reward, rows, &bins);
            let value = decode_rows(&value, rows, &bins);
            for row in 0..rows {
                let expected = reward[row]
                    + config.intrinsic_reward_scale * bonus[row]
                    + continuation[row] * value[row];
                let index = row * actions + action;
                assert!(
                    (branch_values[index] - expected).abs() < 1e-3 * (1.0 + expected.abs()),
                    "branch {row}/{action}"
                );
                reference_values[index] = expected;
            }
        }
        self.actor.set_input("feature", &start);
        self.actor.step();
        self.actor.wait();
        let mut logits = vec![0.0; rows * actions];
        self.actor.read_output_by_index(0, &mut logits);
        let mut weights = vec![0.0; rows * config.imagination_length];
        core.readback.read_regions(&mut [(
            core.behavior_train.input_buffer("imagined_weight").unwrap(),
            &mut weights,
        )]);
        let mut expected_loss = 0.0f64;
        for row in 0..rows {
            let mut probabilities = vec![0.0; actions];
            softmax_unimix(
                &logits[row * actions..(row + 1) * actions],
                config.actor_unimix,
                &mut probabilities,
            );
            let q = &reference_values[row * actions..(row + 1) * actions];
            let mean = q.iter().map(|&x| f64::from(x)).sum::<f64>() / actions as f64;
            expected_loss -= f64::from(weights[row])
                * probabilities
                    .iter()
                    .zip(q)
                    .map(|(&p, &q)| f64::from(p) * (f64::from(q) - mean))
                    .sum::<f64>()
                / (rows as f64 * f64::from(report.behavior.return_scale));
        }
        let actual = f64::from(report.behavior.expected_value_actor_loss.unwrap());
        assert!((actual - expected_loss).abs() < 1e-3 * (1.0 + expected_loss.abs()));
        check_selected_branches(core, &reference_values);
    }
}

fn check_selected_branches(core: &mut DreamerCore, branches: &[f32]) {
    let config = &core.config;
    let starts = config.batch_size * config.batch_length;
    let rows = starts * (config.imagination_length + 1);
    let mut reward = vec![0.0; rows * config.value_bins];
    let mut value = vec![0.0; reward.len()];
    let mut continuation = vec![0.0; rows];
    let mut bonus = vec![0.0; starts * config.imagination_length];
    let mut actions = vec![0.0; starts * config.imagination_length * config.action_count];
    core.readback.read(
        &core.imagination,
        &mut [
            (world::IMAGINATION_REWARD, &mut reward),
            (world::IMAGINATION_VALUE, &mut value),
            (world::IMAGINATION_CONTINUATION, &mut continuation),
            (world::IMAGINATION_BONUS, &mut bonus),
            (world::IMAGINATION_ACTION, &mut actions),
        ],
    );
    let reward = decode_rows(&reward, rows, &core.bins);
    let value = decode_rows(&value, rows, &core.bins);
    for row in 0..starts {
        let action = actions[row * config.action_count..(row + 1) * config.action_count]
            .iter()
            .position(|&x| x == 1.0)
            .unwrap();
        let expected = reward[starts + row]
            + config.intrinsic_reward_scale * bonus[row]
            + continuation[starts + row] * value[starts + row];
        let actual = branches[row * config.action_count + action];
        assert!((actual - expected).abs() < 1e-3 * (1.0 + expected.abs()));
    }
}
