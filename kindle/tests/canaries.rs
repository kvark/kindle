//! Tier 3: Training convergence canaries.
//!
//! Small, well-understood problems with known convergence behavior.
//! These verify that the full training pipeline (graph → autodiff →
//! compile → GPU execution → optimizer) is wired correctly.
//!
//! These tests require a GPU.

/// Canary 3: next-step prediction on a deterministic sequence.
///
/// A trivial world model task: given an observation `x`, predict `x + delta`.
/// The world model should drive prediction error below a threshold within
/// a bounded number of steps.
///
/// Expected: MSE loss < 0.01 within 2000 steps.
/// Failure signal: broken world model training, optimizer, or gradient flow.
#[test]
#[ignore] // requires GPU
fn canary_next_step_prediction() {
    use meganeura::nn;
    use meganeura::{Graph, build_session};

    let batch = 1;
    let dim = 4;
    let hidden = 8;
    let steps = 2000;

    // Build a simple prediction graph: input -> MLP -> output, MSE loss
    let mut g = Graph::new();
    let x = g.input("x", &[batch, dim]);
    let target = g.input("target", &[batch, dim]);

    let fc1 = nn::Linear::new(&mut g, "fc1", dim, hidden);
    let fc2 = nn::Linear::new(&mut g, "fc2", hidden, dim);

    let h = fc1.forward(&mut g, x);
    let h = g.relu(h);
    let y = fc2.forward(&mut g, h);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);

    // Xavier init
    let scale1 = (2.0 / (dim + hidden) as f32).sqrt();
    let w1: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i as f32 * 0.618034).fract() - 0.5) * 2.0 * scale1)
        .collect();
    session.set_parameter("fc1.weight", &w1);
    session.set_parameter("fc1.bias", &vec![0.0f32; hidden]);

    let scale2 = (2.0 / (hidden + dim) as f32).sqrt();
    let w2: Vec<f32> = (0..hidden * dim)
        .map(|i| ((i as f32 * 0.381966).fract() - 0.5) * 2.0 * scale2)
        .collect();
    session.set_parameter("fc2.weight", &w2);
    session.set_parameter("fc2.bias", &vec![0.0f32; dim]);

    // Training data: predict x + 0.1
    let inputs: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect())
        .collect();
    let targets: Vec<Vec<f32>> = inputs
        .iter()
        .map(|x| x.iter().map(|v| v + 0.1).collect())
        .collect();

    let mut final_loss = f32::MAX;
    for step in 0..steps {
        let idx = step % inputs.len();
        session.set_input("x", &inputs[idx]);
        session.set_input("target", &targets[idx]);
        // Must set LR before each step — meganeura clears it after use
        session.set_learning_rate(1e-3);
        session.step();
        session.wait();
        final_loss = session.read_loss();
    }

    assert!(
        final_loss < 0.01,
        "canary 3 failed: next-step prediction loss {final_loss:.4} > 0.01 after {steps} steps"
    );
}

/// Canary: kindle encoder + world model convergence.
///
/// Verifies that the full kindle training graph (encoder → world model → MSE)
/// reduces prediction error on a deterministic observation sequence.
#[test]
#[ignore] // requires GPU
fn canary_kindle_world_model() {
    use kindle::env::{
        Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
    };
    use kindle::{Agent, AgentConfig};

    struct ConstantEnv {
        state: usize,
        patterns: Vec<Vec<f32>>,
    }

    impl ConstantEnv {
        fn new() -> Self {
            // 4 deterministic observations that cycle
            Self {
                state: 0,
                patterns: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                ],
            }
        }
    }

    impl HomeostaticProvider for ConstantEnv {
        fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
            &[]
        }
    }

    impl Environment for ConstantEnv {
        fn observation_dim(&self) -> usize {
            4
        }
        fn num_actions(&self) -> usize {
            2
        }
        fn observe(&self) -> Observation {
            Observation::new(self.patterns[self.state].clone())
        }
        fn step(&mut self, _action: &Action) -> StepResult {
            self.state = (self.state + 1) % self.patterns.len();
            StepResult {
                observation: self.observe(),
                homeostatic: vec![],
                reward: 0.0,
                task_event: false,
                terminated: false,
                truncated: false,
            }
        }
        fn reset(&mut self) {
            self.state = 0;
        }
    }

    let mut env = ConstantEnv::new();
    let adapter = Box::new(kindle::GenericAdapter::discrete(0, 4, 2));
    let config = AgentConfig {
        latent_dim: 4,
        hidden_dim: 16,
        buffer_capacity: 500,
        batch_size: 1,
        learning_rate: 3e-3,
        // AgentConfig's default reconstruction objective is required here:
        // the stopped WM target cannot train the encoder on its own.
        ..AgentConfig::default()
    };

    let mut agent = Agent::new(config, vec![adapter]);
    let encoder_before = agent.dump_encoder_params();
    assert!(!encoder_before.is_empty(), "encoder parameters must exist");
    let mut rng = rand::rng();

    // Collect early loss
    let warmup = 50;
    for _ in 0..warmup {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        env.step(&action);
        // Post-action convention: observe() receives the observation
        // RESULTING from the action (see Agent::observe docs).
        let next_obs = env.observe();
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&next_obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
    }
    let early_loss = agent.diagnostics()[0].loss_world_model;

    // Train more. 2500 steps: with the encoder shaped by recon (its
    // latents keep moving early on) the WM chases a moving target and
    // the loss transiently RISES from its collapsed-init low before
    // converging — 500 steps measured the rise, not the convergence.
    for _ in warmup..2500 {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        env.step(&action);
        // Post-action convention: observe() receives the observation
        // RESULTING from the action (see Agent::observe docs).
        let next_obs = env.observe();
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&next_obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
    }
    let late_loss = agent.diagnostics()[0].loss_world_model;
    let encoder_after = agent.dump_encoder_params();
    let encoder_change: f32 = encoder_before
        .iter()
        .zip(encoder_after.iter())
        .map(|((name_before, before), (name_after, after))| {
            assert_eq!(name_before, name_after);
            before
                .iter()
                .zip(after.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        })
        .sum();

    // The early loss is near-zero for the WRONG reason (collapsed
    // encoder init makes every target identical), so `late < early`
    // is not a meaningful convergence signal any more — assert on the
    // absolute level instead.
    assert!(
        late_loss < 0.15,
        "kindle world model didn't converge: early={early_loss:.4}, late={late_loss:.4}"
    );
    assert!(
        encoder_change > 1e-4,
        "default representation objective did not update encoder parameters: Δ={encoder_change}"
    );
}

/// SIL must dispatch independently of a live on-policy advantage. The call used
/// to live only at the tail of selected policy-update branches, so default
/// n_step=1 agents could accumulate event samples forever without replaying
/// them.
#[test]
#[ignore] // requires GPU
fn canary_sil_runs_with_default_n_step() {
    use kindle::env::{Observation, StepResult};
    use kindle::{Agent, AgentConfig, GenericAdapter};
    use rand::SeedableRng;

    let config = AgentConfig {
        batch_size: 1,
        latent_dim: 4,
        hidden_dim: 8,
        warmup_steps: 0,
        extrinsic_reward_alpha: 1.0,
        use_sil: true,
        sil_event_filter: true,
        sil_event_horizon: 1,
        value_loss_coef: 0.0,
        ..AgentConfig::default()
    };
    let adapter = Box::new(GenericAdapter::discrete(0, 4, 2));
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    for (step, reward) in [0.0, 1.0].into_iter().enumerate() {
        let obs = Observation::new(vec![step as f32, 0.0, 0.0, 0.0]);
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let result = StepResult {
            observation: Observation::new(vec![step as f32 + 1.0, 0.0, 0.0, 0.0]),
            homeostatic: vec![],
            reward,
            task_event: reward > 0.0,
            terminated: false,
            truncated: false,
        };
        agent.observe_step_results(
            std::slice::from_ref(&result),
            std::slice::from_ref(&action),
            &mut rng,
        );
    }

    assert_eq!(agent.sil_buffer_size(), 1);
    assert_eq!(agent.sil_updates_attempted_count(), 2);
    assert!(agent.sil_updates_fired_count() >= 1);
}

/// Dynamic masks must remain aligned with the acted-from state across a
/// delayed KL-PPO rollout update. This catches missing graph inputs and the
/// stale-live-mask bug where every historical row used the newest mask.
#[test]
#[ignore] // requires GPU
fn canary_kl_ppo_dynamic_action_masks() {
    use kindle::{
        Action, Agent, AgentConfig, GenericAdapter, HomeostaticVariable, Observation,
        RewardWeights, StepResult,
    };
    use rand::SeedableRng;

    const LANES: usize = 4;
    let adapters = (0..LANES)
        .map(|_| Box::new(GenericAdapter::discrete(0, 4, 2)) as Box<dyn kindle::EnvAdapter>)
        .collect::<Vec<_>>();
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 16,
        batch_size: LANES,
        learning_rate: 3e-4,
        lr_policy: 3e-4,
        reward_weights: RewardWeights {
            surprise: 0.0,
            novelty: 0.0,
            homeostatic: 0.0,
            order: 0.0,
        },
        extrinsic_reward_alpha: 1.0,
        end_to_end_encoder: true,
        n_step: 2,
        rollout_length: 4,
        use_kl_ppo: true,
        kl_beta: 0.05,
        ppo_n_epochs: 2,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, adapters);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut observations = vec![Observation::new(vec![1.0, 0.0, 0.0, 0.0]); LANES];

    for step in 0..64 {
        let mut masks = vec![0.0; LANES * kindle::MAX_ACTION_DIM];
        let expected = (0..LANES).map(|lane| (step + lane) % 2).collect::<Vec<_>>();
        for (lane, &action) in expected.iter().enumerate() {
            masks[lane * kindle::MAX_ACTION_DIM + action] = 1.0;
        }
        agent.set_action_masks(&masks);
        let actions = agent.act(&observations, &mut rng);
        for (actual, &want) in actions.iter().zip(&expected) {
            assert!(matches!(actual, Action::Discrete(value) if *value == want));
        }

        let results = expected
            .iter()
            .enumerate()
            .map(|(lane, &action)| {
                let mut data = vec![0.0; 4];
                data[(step + lane + action + 1) % 4] = 1.0;
                StepResult {
                    observation: Observation::new(data),
                    homeostatic: Vec::<HomeostaticVariable>::new(),
                    reward: 1.0,
                    task_event: false,
                    terminated: (step + 1) % 11 == 0,
                    truncated: false,
                }
            })
            .collect::<Vec<_>>();
        agent.observe_step_results(&results, &actions, &mut rng);
        observations = results
            .iter()
            .map(|result| result.observation.clone())
            .collect();

        let diagnostics = &agent.diagnostics()[0];
        assert!(diagnostics.loss_policy.is_finite());
        assert!(diagnostics.loss_world_model.is_finite());
    }
}

/// Plain advantage-weighted CE uses the same dynamic action-mask contract as
/// PPO. In particular, delayed rollout rows must use their collection-time
/// mask and the graph must not allocate probability to padded action heads.
#[test]
#[ignore] // requires GPU
fn canary_plain_policy_dynamic_action_masks() {
    use kindle::{
        Action, Agent, AgentConfig, GenericAdapter, Observation, RewardWeights, StepResult,
    };
    use rand::SeedableRng;

    const LANES: usize = 4;
    let adapters = (0..LANES)
        .map(|_| Box::new(GenericAdapter::discrete(0, 4, 2)) as Box<dyn kindle::EnvAdapter>)
        .collect::<Vec<_>>();
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 16,
        batch_size: LANES,
        learning_rate: 3e-4,
        lr_policy: 3e-4,
        reward_weights: RewardWeights {
            surprise: 0.0,
            novelty: 0.0,
            homeostatic: 0.0,
            order: 0.0,
        },
        extrinsic_reward_alpha: 1.0,
        n_step: 2,
        rollout_length: 4,
        advantage_normalize: true,
        use_grpo: true,
        entropy_floor: 0.5,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, adapters);
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let mut observations = vec![Observation::new(vec![1.0, 0.0, 0.0, 0.0]); LANES];

    for step in 0..48 {
        let mut masks = vec![0.0; LANES * kindle::MAX_ACTION_DIM];
        let expected = (0..LANES).map(|lane| (step + lane) % 2).collect::<Vec<_>>();
        for (lane, &action) in expected.iter().enumerate() {
            masks[lane * kindle::MAX_ACTION_DIM + action] = 1.0;
        }
        agent.set_action_masks(&masks);
        let actions = agent.act(&observations, &mut rng);
        for (actual, &want) in actions.iter().zip(&expected) {
            assert!(matches!(actual, Action::Discrete(value) if *value == want));
        }

        let results = expected
            .iter()
            .enumerate()
            .map(|(lane, &action)| {
                let mut data = vec![0.0; 4];
                data[(step + lane + action + 1) % 4] = 1.0;
                StepResult {
                    observation: Observation::new(data),
                    homeostatic: vec![],
                    reward: if action == lane % 2 { 1.0 } else { -1.0 },
                    task_event: false,
                    terminated: (step + 1) % 13 == 0,
                    truncated: false,
                }
            })
            .collect::<Vec<_>>();
        agent.observe_step_results(&results, &actions, &mut rng);
        observations = results
            .iter()
            .map(|result| result.observation.clone())
            .collect();

        let diagnostics = &agent.diagnostics()[0];
        assert!(diagnostics.loss_policy.is_finite());
        assert!(diagnostics.loss_world_model.is_finite());
    }
}
