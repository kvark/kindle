## v0.2 (TBD)

- Parameterized actions now use separate policy and dynamics representations:
  the discrete policy label remains an 18-way one-hot, while the world model,
  replay paths, surprise replay, and k-step training receive a 20-value token
  with normalized `(x, y)` appended. ARC single, joint, validation, and GRPO
  harnesses stage the coordinates that were actually executed. Parameter
  staging is per-lane and one-shot, so ordinary actions cannot inherit a stale
  click. This changes the world-model `a_proj` checkpoint shape; older full WM
  checkpoints are not architecture-compatible. A GPU canary trains one
  discrete action to reach different states for `x=-1` and `x=+1`, proving the
  learned dynamics actually use the parameter tail.
- Shooting MPC now searches and queues `(discrete action, x, y)` tuples for
  action slots declared parameterized by the host. Planner parameters are
  consumed exactly once after `act()`, and ARC single/GRPO runners execute and
  report the selected coordinate. Parameterized spaces bypass MCTS because its
  current one-child-per-action tree cannot compare coordinates. A GPU canary
  learns that negative x changes state and verifies the planner selects
  `x=-0.927` with the predicted changed state closer than the no-op state.
- Planner queues now reject actions invalidated by a newer dynamic mask. Fixed
  `planner_action_repeat`, which previously resampled a different action on
  every supposed repeat; it now repeats the same identity and parameters.
- ARC-AGI-3 adapters now keep action identities stable while masking dynamic
  availability, expose seeded random baselines and executable progress gates,
  and report per-attempt level gains instead of treating cumulative level state
  retained across resets as repeated success. Five-seed CD82 and 25-game probes
  show that the historical ARC recipes do not beat random; the old “100% L1”
  interpretation is retracted in the integration notes.
- ARC object-click proposals no longer depend on SciPy, infer the modal
  background, choose coordinates that lie on concave components, and rank
  pieces nested inside panels ahead of their containers. FT09's eight
  immediately state-changing cells consequently fit in slots 0–7 instead of
  ranks 21–33. Three 5k-step seeds now reach one L1 event each (uniform
  coordinates: two events total); no run reaches L2. Object slots remain
  stable discrete WM actions rather than planner-overridable coordinate aliases.
- Fixed the GRPO harness's missing-ARC error path (`sys` was accidentally
  shadowed), and marked the ARC optional dependencies with their Python 3.12
  minimum without raising Kindle core's Python requirement.
- Added a bounded object-state search harness for cloneable local ARC games.
  It deduplicates visual action outcomes, detects commuting involutions, and
  regenerates actions/object coordinates at each non-commutative state instead
  of globally discarding root no-ops. Invalid simulator branches are skipped;
  state and transition budgets bound clone cost. A curated gate reaches L1 on
  8/8 games in 246 expanded states, adding SP80, CD82, LF52, and SU15 to the
  earlier four; the broader 19-game complex-action probe reaches 8 at L1. FT09
  still reaches L2 in 11 actions/2,978 states and VC33 in 10 actions/45 states.
  This remains simulator-backed planning rather than policy learning.
- ARC demonstrations now retain availability masks and hybrid object/pixel
  tokens, and train both action identity and click coordinates. The new narrow
  `train_policy_supervised` path applies positive masked cross-entropy directly
  instead of fabricating rewards/value advantages. The coordinate head now
  consumes the raw observation token plus a fixed task embedding, removing an
  unnecessary GPU encode and cross-batch dependency while preventing cross-game
  click interference. Compatible demonstration files merge with exact-row
  deduplication and conflicting-label rejection. Learned clicks can be
  projected onto the visible object representatives used by search; a stable-
  motion prior activates after two matching click displacements and resets at
  level transitions. A fresh one-lane reload retains L1 on 8/8 searched games
  and L2 on FT09/VC33 (53 unique samples, 100% action-label accuracy,
  coordinate MSE 0.001792). Nearest-object snapping alone remains documented
  as an SU15 failure. Online ARC runners expose projection as an opt-in mode,
  leaving random and unconstrained pixel clicks unchanged. The coordinate
  checkpoint input width changed from 64 to 72, so older `coord.bin` files are
  not compatible.
- A deeper bounded search reaches VC33 L3 in 33 total actions and 1,169
  expansions. Search demonstrations record deterministic object rank, count,
  and 12 normalized features for every current candidate. Coordinate regression
  (including task-balanced and VC33-only fits) and a temporary widened
  ranked-object policy did not retain L3; the 64-way universal head experiment
  was removed and `MAX_ACTION_DIM` remains 18. A persisted variable-set pointer
  now fits all 67 complex labels in the 76-row joint corpus. A separate
  zero-update checkpoint reload reaches L1 on 8/8 games, FT09 L2, and VC33 L3
  without cloning, coordinate projection, or motion extrapolation.
- Coordinate-head REINFORCE can train only lanes whose executed action consumed
  coordinates, preventing unrelated discrete-action rewards from updating click
  locations. Baselines are per lane rather than shared across unrelated games,
  and callers may supply an explicit task-reward vector instead of implicitly
  using the combined intrinsic reward. ARC uses positive level deltas for this
  credit. The joint trainer supports the path via `--coord-alpha`.
- Fixed `arc_agi3_grpo.py --help`; a literal percent sign in one help string
  was interpreted as an argparse formatting directive.
- Atari's harness now uses the spatial Nature-DQN encoder by default, keeps
  intrinsic reward weights explicitly zero in bare extrinsic mode, handles
  Gymnasium terminal/reset observations without transition misalignment, and
  exposes return/episode/latent outcome gates plus an `extrinsic_grpo` preset.
  The seeded Breakout progress gate reaches 1.94 recent return after 120k
  environment steps (random baseline 1.225, required 1.5); sparse-score Pong
  remains open.
  Frame skip and sticky-action probability are configurable, an optional
  foreground residual emphasizes small sprites, and controller-only baselines
  avoid constructing an irrelevant GPU agent. At frame skip 1, the seeded
  pixel-only Pong controller reaches -11.50 over 120k decisions versus -20.03
  random. Reward-trained Pong remains random-level, and failed demonstration
  replay was removed rather than exposed as a recipe. A separate supervised
  harness balances unique controller states, trains Kindle with direct masked
  cross-entropy, and performs a zero-update checkpoint reload. It reaches
  98.8% held-out action accuracy and -12.60 over 120k decisions (random -20.03,
  controller -11.50), establishing retained imitation but not autonomous
  reward learning. A follow-on harness freezes that learned motion encoder and
  continues the policy head using raw signed Pong rewards with 24-lane
  same-task GRPO. Across three fresh 120k-decision evaluations it reduces the
  distilled source's point deficit 368→237 (35.6%), improves every seed, raises
  positive-point fraction 27.2%→32.9%, and moves aggregate completed-game
  return -14.06→-13.43. A separate load-only process reproduces the result with
  encoder maximum change 0; sparse-score-only learning from scratch remains
  open.
- Added a random-initialized Pong shaping harness that learns through
  class-balanced signed tracking-consistency rewards with raw Pong points still
  active. It executes no controller actions and calls no supervised objective;
  a fixed balanced validation set restores the best snapshot before frozen
  evaluation. A separate load-only process reproduces returns
  -13.20/-13.60/-10.75 across three 120k-decision seeds (aggregate -12.64,
  +246/-503 points; active tracking at least 98.7% per seed). The reward rule
  encodes the desired ball-tracking decision, so this is documented as a
  task-specific shaped-reward result rather than autonomous reward discovery or
  sparse-score-only learning.
- Added deterministic policy-mode actions in Rust and Python for evaluation;
  sampling remains the training default.
- Discrete policy training now applies collection-time action masks in the
  plain CE graph as well as PPO/KL-PPO, initializes padded action heads as
  invalid, smooths entropy-recovery targets only over valid actions, and uses
  the documented entropy coefficient instead of dividing it by the universal
  action width. A delayed plain-policy dynamic-mask GPU canary covers this.
- Gymnasium vector examples use same-step autoreset and feed the true terminal
  observation to Kindle before staging the reset observation for the next act.
- Python CI now applies Rustfmt and Clippy with warnings denied to the binding
  crate, which is outside the root Cargo workspace.
- Native `Environment::step` results now carry task reward plus explicit
  `terminated`/`truncated` episode semantics. `Agent::observe_step_results`
  forwards rewards, uses the pre-reset terminal observation/homeostatic
  state, and marks boundaries in one call; all native examples use it. A
  separate `task_event` bit distinguishes positive achievements from terminal
  mechanics: MountainCar's -1 goal step is an event, whereas CartPole falling
  is not. Explicit false events also suppress the legacy inference that every
  positive reward is an achievement.
  The seeded eight-lane `native_learning` CartPole gate reaches a final
  recent mean return of 263.38 after 800k environment steps (required 195).
  Its intrinsic-only gate uses per-step GRPO over the generic homeostatic
  deviation penalty, with task reward/events and SIL disabled. A single
  ordered multi-seed invocation trains for exactly 50k decisions per seed,
  freezes learning, and returns 500/500/215 on seeds 42–44 (random
  22.11/22.74/22.42; required 195). Score-triggered learning-rate changes are
  now disabled whenever the extrinsic channel is zero, eliminating an external
  metric leak from intrinsic runs. The harness reports training and evaluation
  budgets separately and rejects empty, invalid, or duplicate seed lists.
  One task-conditioned 32/64 policy now trains on eight CartPole and eight Taxi
  lanes simultaneously, then freezes for an independent 160k-decision
  evaluation per task. Across seeds 42–44, sampled-policy CartPole returns are
  485.40/482.37/402.43 and Taxi completion is 92.3%/95.6%/90.3%, passing every
  task/seed gate. A current 16/32 model reaches only 78.7% Taxi completion, and
  greedy evaluation of the wider seed-42 model reaches 87.1%, so the retained gate
  claims simultaneous sampled-policy competence rather than deterministic
  mastery or cross-game transfer.
  A sequential mode trains the same eight-lane agent on
  CartPole→Taxi→CartPole with frozen evaluations around each switch. Across
  seeds 42–44, CartPole starts at 488.18/493.80/486.31, Taxi reaches
  93.9%/92.8%/93.3%, and retained CartPole returns
  253.50/500.00/229.98. Reverse Taxi→CartPole retention passes only two of
  three seeds (seed 43 Taxi: 89.6%), so order-independent continual learning
  remains open.
  Event-filtered SIL now ranks successful episodes against other successes and
  gives every retained event transition a minimum imitation weight, so
  negative step costs cannot erase successful trajectory prefixes. With
  shaped/masked Taxi this raises final 200-episode completion to 90%, 95%, and
  93% across seeds 42–44 (required 90%) over the same 800k-step budget.
  SIL admission baselines are now keyed by task id, preventing positive-return
  CartPole episodes from rejecting successful negative-return Taxi episodes in
  a shared agent. Capacity uses per-task water-filling so a later curriculum
  cannot erase inactive tasks, while minibatches round-robin only across task
  ids active in current lanes. Long episodes therefore cannot crowd shorter
  tasks out, and stale action semantics cannot block adaptation after a switch.
  Agents may now opt into orthogonal one-hot codes for the first eight task ids,
  making the trainable task-projection columns genuine isolated adapter biases.
  The native multitask gate enables this mode; overflow ids use deterministic
  normalized hashes. The default keeps historical dense codes so existing
  Atari/ARC/Python checkpoints retain their input semantics.
  Extrinsic-only MountainCar combines eight-step action persistence until its
  first success, four-step persistence afterward, and a 50k SIL buffer; its
  final 200-episode completion is 97.5%, 91.0%, and 99.5% across seeds 42–44
  after 400k environment steps (required 90%).
  Acrobot uses four-step persistence for its first 50 successes and two-step
  persistence afterward; event SIL then reaches 95.0%, 93.0%, and 97.5%
  completion across seeds 42–44 after 240k environment steps (required 90%).
  The harness exposes the repeat-switch success count and entropy coefficient
  so these exploration schedules can be calibrated without source edits.
- Native environments can expose state-dependent legal actions through
  `Environment::action_mask`; `Agent::set_action_masks_from_envs` stages
  them before sampling. Transitions and SIL samples now retain their
  collection-time masks, so delayed rollout PPO and SIL updates no longer
  train historical actions under the newest state mask. KL-PPO now applies
  the same mask to its policy and KL distributions; a GPU canary alternates
  masks across a delayed rollout to enforce the contract.
- Fixed continuous policy-gradient weighting. The old Gaussian path scaled the
  regression target (`advantage * action`) before MSE, which is not equivalent
  to weighting the loss and actively trained zero-advantage rows toward zero.
  Continuous graphs now take an explicit signed advantage, retain raw action
  targets, exclude padded universal action heads from the loss, and support the
  end-to-end policy encoder. Unsupported continuous PPO/KL-PPO/SIL combinations
  fail at construction. The native harness now has matched random/expert/
  reward-only Pendulum modes, a frozen greedy evaluation phase, and a
  harness-local DAgger gate. DAgger reaches the expert return on two of three
  seeds (-359.02/-497.27/-362.44). Correctly masked reward-only learning scores
  -1860.34 versus random -1724.57 and remains open.
- PPO clipped-surrogate + end-to-end encoder restored
  (`build_ppo_policy_graph_e2e`). Removed in v0.1 era after never
  converging; revalidated post-autodiff-bug-fix and now trains stably
  on LunarLander (29.7% landings post-drop, vs 0% before fix). The
  earlier failure was likely an autodiff-bug-induced symptom rather
  than a clipped-surrogate pathology.
- GRPO (Group-Relative Policy Optimization) added via `use_grpo`
  config flag. Per-step variant uses n-step return with no V
  bootstrap, normalized across the policy batch (cross-lane mean/std).
  After the bootstrap-noise fix (early version had random V_n
  contaminating the n-step return), per-step GRPO reaches 23% landings
  on LunarLander vs A2C 42.8% and KL-PPO 45.2%.

  Optional `use_grpo_episode` enables per-episode GRPO: each transition
  in a completed episode is annotated retroactively with the episode's
  total return, then advantage is normalized cross-batch. Composes
  with `use_ppo` (PPO clip + GRPO advantage = canonical DeepSeek-R1
  GRPO) and `use_kl_ppo`. Per-episode mode is degenerate at small
  batch sizes (8-32 lanes): clustered episode returns yield zero
  advantage after std normalization. The std-skip fix
  (`use_grpo_episode` only mean-centers) helps but doesn't unlock
  meaningful learning at our scale.

- Self-Imitation Learning (Oh et al. 2018) added via `use_sil`. The
  agent maintains a buffer of `(obs, action, R_to_go, V_at_collect)`
  from "successful" episodes (return > EMA baseline). At each policy-update
  cadence, an independent supervised CE step runs on a sampled
  batch with per-sample weight = `sil_loss_coef × max(0, R_to_go - V)`.
  The advantage is clamped to `advantage_clamp` to keep the SIL
  gradient comparable in magnitude to the regular policy gradient
  (without the clamp, raw R_to_go on dense-reward envs produces
  100× stronger gradients that destabilize training).
  Validated end-to-end via diagnostics (`sil_buffer_size`,
  `sil_counters`, `sil_last_param_change`); confirmed SIL changes
  policy parameters and policy entropy.

  The independent cadence is important: the previous placement after selected
  successful on-policy branches made default `n_step=1` silently skip SIL and
  made multi-epoch PPO replay once per epoch. Continuing tasks may optionally
  capture a bounded backward window at a positive event without fabricating an
  episode boundary; the knob is not part of the Atari preset because its Pong
  ablation was negative.

  Empirical finding on LunarLander: SIL eliminates the hover/timeout
  failure mode (13.7% → 0-4%) but at the cost of more crashes (42% →
  66-88%). Hover-state and landing-state require genuinely different
  optimal actions (mid-air vs on-ground); SIL imports the landed-noop
  action into hover-state where it causes freefall and crash. Adding
  Gaussian noise to obs (encoder smoothness regularization, harness-
  side) moderates this somewhat but doesn't beat the no-SIL baseline
  total non-crash rate (44+14=58% non-crash for baseline vs ~27-34%
  for SIL variants).

  The deeper conclusion: pure on-policy A2C with SIL cannot bridge
  state distributions that genuinely require different optimal
  actions. Off-policy methods (Q-learning, SAC) would be needed to
  evaluate counterfactual actions in those states.

## v0.1 (14 Apr 2026)

- Cold-start continually-self-training RL agent on meganeura: encoder,
  world model, credit assigner, policy+value head, all trained from first
  contact with no pretraining.
- Frozen four-primitive reward circuit: surprise (WM prediction error),
  novelty (1/√N visit counts), homeostatic balance (env-provided target
  variables), and **order** (causal entropy reduction — frozen random
  digest of the universal obs token, sliding recent/reference windows,
  `r_order = H_reference − H_recent`).
- Universal obs/action tokens + per-env `EnvAdapter`; one compiled graph
  handles discrete and continuous action spaces. `Agent::switch_lane`
  hops between environments with no graph rebuild; per-env deterministic
  task embeddings keep representations disambiguated.
- Continual-learning mechanics: experience replay mixing, representation
  drift monitor with reactive encoder-LR scaling, policy entropy floor.
- Numerical confidence tooling: e-graph opt-level parity tests,
  CartPole / random-walk / world-model convergence canaries, long-run
  stability probes. (Direct finite-difference gradient checks are
  planned, not yet implemented — gradients are currently covered
  indirectly by the convergence canaries.)
- Seven built-in environments in `kindle-gym`: grid_world, cart_pole,
  mountain_car, pendulum, acrobot, taxi, random_walk.
- Optional Python bindings (`kindle-py`, pyo3 + maturin): `kindle.Agent`
  with `run(gym_env, steps)` and `diagnostics()`, drop-in for any
  gymnasium loop.
- Cargo workspace layout: `kindle` (core), `kindle-gym`
  (envs + runnable examples), `python` (maturin-built extension, out of
  the default workspace build).
