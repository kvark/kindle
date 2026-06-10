//! WM time-direction test.
//!
//! Trains the agent on a deterministic 4-state ring where action 0
//! ADVANCES the ring (s → s+1) and action 1 stays put. After training,
//! probes the dynamics model via `Agent::wm_predict`: for each state's
//! latent z(s), the prediction WM(z(s), advance) must be closer to
//! z(s+1) than to z(s-1).
//!
//! This is the regression test for the 2026-06-10 fix: before it, the
//! online WM loss trained `WM(z_{t+1}, a_t) → z_t` — a BACKWARD model
//! that every consumer (planner rollouts, MCTS, k-step training)
//! assumed was forward.
//!
//! Requires a GPU (full Agent construction).

use kindle::env::{
    Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
};
use kindle::{Agent, AgentConfig, GenericAdapter};
use rand::SeedableRng;

const RING: usize = 4;
const OBS_DIM: usize = RING;

struct RingEnv {
    state: usize,
}

impl RingEnv {
    fn obs_of(state: usize) -> Observation {
        let mut v = vec![0.0f32; OBS_DIM];
        v[state] = 1.0;
        Observation::new(v)
    }
}

impl HomeostaticProvider for RingEnv {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &[]
    }
}

impl Environment for RingEnv {
    fn observation_dim(&self) -> usize {
        OBS_DIM
    }
    fn num_actions(&self) -> usize {
        2
    }
    fn observe(&self) -> Observation {
        Self::obs_of(self.state)
    }
    fn step(&mut self, action: &Action) -> StepResult {
        if let Action::Discrete(0) = action {
            self.state = (self.state + 1) % RING;
        }
        StepResult {
            observation: self.observe(),
            homeostatic: vec![],
        }
    }
    fn reset(&mut self) {
        self.state = 0;
    }
}

#[test]
#[ignore] // requires GPU
fn wm_rolls_forward_in_time() {
    let mut env = RingEnv { state: 0 };
    let adapter = Box::new(GenericAdapter::discrete(0, OBS_DIM, 2));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        buffer_capacity: 4000,
        batch_size: 1,
        learning_rate: 2e-3,
        // Recon is the anti-collapse term: without it the constant
        // task-embedding contribution dominates the encoder output and
        // all 4 states' latents come out near-collinear (verified at
        // init), making direction probing meaningless. This mirrors
        // the real operating config, where recon is always on.
        recon_loss_coef: 1.0,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    // Train with the post-action convention. Mostly advance (the pair
    // (z(s), advance) → z(s+1) is what we probe), some stays mixed in
    // so the WM sees action contrast.
    let steps = 3000usize;
    for _step in 0..steps {
        let obs = env.observe();
        // act() keeps the lane state machine honest; the action we
        // EXECUTE is scripted for coverage. Random 75/25 advance/stay —
        // a deterministic pattern with period 4 would alias against
        // the ring period and starve some (state, action) pairs.
        let _ = agent.act(std::slice::from_ref(&obs), &mut rng);
        let action = if rand::Rng::random_range(&mut rng, 0..4) == 3 {
            Action::Discrete(1)
        } else {
            Action::Discrete(0)
        };
        env.step(&action);
        let next_obs = env.observe();
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&next_obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
    }

    // Read back the converged latent for each ring state by encoding
    // one more lap (the latent of the transition that LANDED on state
    // s is enc(o_s); grab it from the diagnostics-visible buffer via
    // one observe per state with a scripted advance).
    let mut z_of_state = vec![vec![0.0f32; 8]; RING];
    for _ in 0..RING {
        let obs = env.observe();
        let _ = agent.act(std::slice::from_ref(&obs), &mut rng);
        let action = Action::Discrete(0);
        env.step(&action);
        let next_obs = env.observe();
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&next_obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
        // env.state is now the state whose encoding was just computed.
        z_of_state[env.state] = agent.latents()[0].clone();
    }

    // Probe: WM(z(s), advance) must land nearer z(s+1) than z(s-1).
    let dist = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
    };
    for a in 0..RING {
        let row: Vec<f32> = (0..RING)
            .map(|b| dist(&z_of_state[a], &z_of_state[b]))
            .collect();
        eprintln!("z-dist row {a}: {row:?}");
        eprintln!("  z[{a}] = {:?}", z_of_state[a]);
    }
    for s in 0..RING {
        let mut stay_row = vec![0.0f32; kindle::MAX_ACTION_DIM];
        stay_row[1] = 1.0;
        let pred_stay = agent.wm_predict(&z_of_state[s], &stay_row);
        let ds: Vec<f32> = (0..RING)
            .map(|k| dist(&pred_stay, &z_of_state[k]))
            .collect();
        eprintln!("STAY  state {s}: all = {ds:?}");
    }
    let mut forward_wins = 0usize;
    for s in 0..RING {
        let mut action_row = vec![0.0f32; kindle::MAX_ACTION_DIM];
        action_row[0] = 1.0; // advance
        let pred = agent.wm_predict(&z_of_state[s], &action_row);
        let ds: Vec<f32> = (0..RING).map(|k| dist(&pred, &z_of_state[k])).collect();
        let d_next = ds[(s + 1) % RING];
        let d_prev = ds[(s + RING - 1) % RING];
        eprintln!(
            "state {s}: d(pred, z_next) = {d_next:.4}, d(pred, z_prev) = {d_prev:.4}, all = {ds:?}"
        );
        if d_next < d_prev {
            forward_wins += 1;
        }
    }
    eprintln!("final wm loss: {}", agent.diagnostics()[0].loss_world_model);
    assert!(
        forward_wins >= 3,
        "dynamics model is not forward in time: prediction landed nearer \
         z(s+1) for only {forward_wins}/{RING} states (a backward model \
         scores 0-1 here)"
    );
}
