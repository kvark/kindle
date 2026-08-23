# Failed experiments — kept for the diagnostic trail

This document collects experiments whose code was removed because it
didn't work, but whose lessons are useful for future architectural
work. Each entry: what was tried, what symptoms appeared, what the
hypothesis was, why we stopped.

## PPO + end-to-end encoder (build_ppo_policy_graph_e2e)

> **Update 2026-05-01 — RESTORED.** `build_ppo_policy_graph_e2e` is
> back in `kindle/src/policy.rs` and is dispatched again from
> `Agent::new` when `use_ppo && end_to_end_encoder` (the
> `assert!(!(use_ppo && end_to_end_encoder))` guard listed under
> "Files removed" below is gone). The failure documented in this
> entry was re-diagnosed as a symptom of a meganeura autodiff bug
> (SumAll/MeanAll dropped `grad_output`), not a clipped-surrogate
> pathology; post-fix the graph trains stably on LunarLander (29.7%
> landings post-drop vs 0% before the fix — see CHANGELOG v0.2).
> Everything below is kept unchanged as the historical record of the
> pre-fix behavior.

**Removed** 2026-04-26. The function was in `kindle/src/policy.rs`
and dispatched in `agent.rs` when both `use_ppo=true` and
`end_to_end_encoder=true`. Across four debug rounds it never
converged on CartPole.

### Architecture
PPO clipped surrogate (`-mean(min(r·A, clip(r,1±ε)·A))`) on the same
encoder + policy + value graph as `build_policy_graph_e2e`. Entropy
term used `stop_gradient(z)` to keep its gradient out of the encoder
(same fix as plain e2e).

### Symptoms
With `--advantage-normalize` (the standard PPO setup), entropy
collapsed to 0 within ~5k env-steps regardless of LR. Without
advantage_normalize, entropy stayed healthy but mean return was
stuck at ~+19 (random baseline). Compare plain-PG e2e: solves
CartPole sustained at +218.

### Debug attempts (all unsuccessful)
| Variation | Mean return | Notes |
|-----------|-------------|-------|
| baseline | +9.6 | ent collapse step 5k |
| ppo_n_epochs=1 | +9.5 | no help |
| ppo_n_epochs=4 | +9.6 | no help |
| no advantage_normalize | +19.5 | stuck at random, ent stable |
| larger clip_eps=0.5 | +14.7 | wider clip, no help |
| stop_gradient on value head's z | +11 | mostly random, V doesn't shape encoder |
| + entropy_beta=0.1 | +16.8 | NO collapse, but no convergence |
| + entropy 0.05→0 anneal | +16.5 | no change from constant |
| + auxiliary CE loss | +11.6 | overfits on n_epochs=4, hurts |
| vlc=0 (value coef = 0, isolate policy) | +9.5 | confirms policy graph is broken |

### Hypothesis (unverified, would need meganeura instrumentation)
The PPO surrogate at small normalized advantages (mean(A) ≈ 0 by
construction) produces a per-element gradient with magnitude
proportional to A, but the gradient signal-to-noise ratio is too
low at kindle's per-step / small-batch update cadence. Plain
cross-entropy loss `-A·log_softmax(action)` has stronger
well-conditioned gradients on the same advantages, which is why the
plain-PG e2e graph converges where the PPO surrogate does not.
The clip mechanism only kicks in when ratio drifts from 1, but
ratio stays near 1 across a single rollout's epochs in our setup
because policy updates are tiny → clip never engages → no
trust-region behavior.

### Why we stopped
Every variation tested still failed to converge. Real diagnosis
requires per-op gradient norm inspection in meganeura (an
instrumentation feature that doesn't exist). Possibly the right
fix is a complete rewrite using KL-penalty PPO instead of the
clipped surrogate, or matching the surrogate to plain-PG's
log-prob form (essentially making it a soft-clipped CE loss).
Either is multi-day work and would need empirical comparison to
the working plain-PG e2e baseline.

### What replaces it in practice
The working sustained-solve path is **plain-PG e2e + `--lr-drop-on-solve`**.
The dynamic LR mutator (`Agent::set_learning_rate` /
`Agent::set_lr_policy`) provides the trust-region effect that PPO
would provide if it worked: once a sustained solve is detected, drop
the LR by 10×–1000× to prevent the post-solve crash. Empirically
this gets CartPole-v1 to mean +218 over 1.6M env-steps (officially
solved) and Acrobot-v1 to mean -128 (sustained improvement).

### Files removed
- `kindle/src/policy.rs::build_ppo_policy_graph_e2e` (~120 lines)
- The PPO-and-e2e dispatch branch in
  `kindle/src/agent.rs::Agent::new` (replaced with an explicit
  `assert!(!(use_ppo && end_to_end_encoder))`)

The PPO surrogate's gradient pathology in our setup is documented
here as a known issue that future PPO work would need to address.

## Frozen π_old snapshot for KL-PPO

**Removed** 2026-04-26 (implemented + reverted same day). Designed to
address the empirical finding that kindle's KL-PPO has near-zero KL
between π_new and π_old — the per-transition `logits_at_action` stored
at action-time is too close to the current policy (only a few env-steps
of drift), so the KL trust region never activates.

### Architecture
A `capture_kl_snapshot_logits` method that ran a forward-only pass
(LR=0) on the current rollout's obs/task using the current policy
weights, capturing the resulting logits as a "frozen snapshot." The
K subsequent training-epoch calls then used this snapshot as the
old_logits input, so KL would grow monotonically as the policy
drifted from the captured point — what standard PPO does.

### Implementation sketch
- New method `capture_kl_snapshot_logits(n_step, rollout_length)` that
  set a `kl_snapshot_capture_pending` flag and re-entered
  `policy_step_rollout_batch` with the flag on.
- Inside `policy_step_rollout_batch`, when the flag was set: LR forced
  to 0, weights didn't move, but the forward pass produced logits.
- After the call: `read_output_by_index(1, &mut kl_old_logits_scratch)`
  to capture the snapshot.
- Inside the K-epoch dispatch, called the snapshot capture once before
  the K-loop. Removed the per-row refill of kl_old_logits_scratch from
  ripe.logits_at_action (the snapshot replaced it).

### Symptoms
Tested on CartPole seed=42, KL-PPO config (vlc=0.1, β=0.05,
rollout=5, epochs=4):
- Steps 5k-25k: V learns normally (V[+0.18, +27]), ent oscillates
  0.01-0.69, KL stays at 0.0001-0.004 (still tiny — snapshot didn't
  produce expected KL growth).
- Step 35k+: V drops to exactly 0.00, ent locks at 0.69 (uniform max),
  pi loss has small spikes. Mean +17.

V going to exactly 0.00 indicates either the value head's output
saturates at the scaled_tanh's zero crossing, OR the encoder produces
z=0 (degenerate). Couldn't quickly diagnose which.

### Hypotheses for the bug
1. The forward-only "snapshot" pass still triggers the SGD update at
   LR=0; meganeura's update path does the dispatch even with
   learning_rate=0.0. Adam's m/v moments may still update (need to
   verify). If m/v drift during snapshot passes, then on real training
   passes Adam's effective LR is wrong.
2. The early-return after snapshot capture skips loss bookkeeping
   (read_loss, last_policy_loss EMA update, watchdog). The watchdog
   tracks policy_loss_ema and re-inits the policy if loss explodes.
   Skipping the read + EMA update could mask a bad loss.
3. The kl_old_logits_scratch buffer was being read mid-step in the
   capture pass; possible race with the GPU-side scratch.

### Why we stopped
The bug needs grad-inspection iteration that adds another 1-2 hours.
Plus, even if the snapshot mechanism worked, the deeper architectural
issue is that kindle's training cadence (per-step or 5-step rollout
with 1-4 epochs) is fundamentally different from standard PPO's
"collect 2048 steps with frozen policy, then train 10 epochs of
mini-batches." The snapshot would help, but a full PPO refactor
(separate collect phase from train phase) is the principled fix.

### Update 2026-04-27 — bug confirmed as NaN cascade in encoder
With the snapshot mechanism re-enabled and `--grad-debug-every 5000`,
the failure mode is precisely localized: between step 40k and 45k
on CartPole, ALL `policy_encoder.*` parameters become NaN (verified
via grad-inspection's bulk weight-norm read). The KL gradient flowing
through stop_gradient(z) → policy_encoder pushes the weights into
overflow within a few hundred K-cycles.

Tried adding NaN/Inf guard + soft clamp (±20) on captured logits in
`capture_kl_snapshot_logits`; result was V collapse moved EARLIER
to step 15k (the clamp itself caused gradient pathology — clamping
old_logits to ±20 made KL gradient larger for non-clamped new_logits).

Real fix likely needs one of:
- Stop-gradient on z for the KL term too (currently only for entropy)
- Numerically-stable KL formulation that doesn't blow up on
  near-deterministic π_old
- Frozen-weights snapshot in a SEPARATE session (not a shared-weights
  forward pass), so capturing doesn't perturb gradients

All require multi-day diagnostic work. Documented and parked.

### What replaces it in practice
Nothing — KL-PPO without the snapshot remains in the codebase
(behaves like plain PG with a tiny KL nudge). The frozen-snapshot
mechanism is documented here for future attempt.

### Files removed
- `Agent::capture_kl_snapshot_logits` (~30 lines)
- `Agent::kl_snapshot_capture_pending` field
- The conditional snapshot capture call in the K-epoch dispatch in
  `policy_step_batched`

## Auxiliary CE loss alongside PPO surrogate

Tried adding `cross_entropy_loss(logits, A·one_hot)` to the PPO
surrogate to provide a strong policy gradient even when the surrogate
gradient is weak. With ppo_n_epochs=4, the unclipped CE part overfits
the rollout batch (same advantage targets repeated 4×), and the policy
overshoots. Mean +11.6 — same as PPO baseline. Not worth keeping.

Reverted same day. The intuition (PPO needs a non-degenerate gradient
fallback) might still be right, but the right form is probably
matching the surrogate to plain-PG's log-prob structure rather than
adding CE on top.

## Standard-PPO scaling on kindle's K-epoch loop

**Tested** 2026-04-27 with the now-fixed KL-PPO snapshot mechanism
(`stop_gradient(z)` on KL term, encoder no longer NaN-cascades).
The hypothesis was that running KL-PPO at *standard* PPO scales —
larger rollouts, more epochs per rollout — would let the trust region
engage productively and outperform plain-PG e2e on CartPole.

### Configurations tried (CartPole-v1, 240k env-steps)
| rollout | history_len | epochs | lr_pi | mean | symptom |
|---------|-------------|--------|-------|------|---------|
| 64 | 32 (default) | 4 | 3e-4 | ~+9 | V=0 throughout (buffer too small) |
| 64 | 32 | 10 | 1e-4 | ~+9 | V=0 throughout |
| 16 | 64 | 10 | 1e-4 | +10.75 | ent collapses by step 10k, KL=0 |

### Diagnosis
The K-epoch loop in `policy_step_batched` calls
`policy_step_rollout_batch` K times on the *same* `lanes ×
rollout_length` batch — no mini-batching. Standard PPO mini-batches
within each epoch precisely to prevent overfitting to a single batch.
With K=10 and tiny rollouts, the policy overfits and entropy collapses
to 0 within a few hundred rollouts. Both π_old and π_new collapse to
the same deterministic policy → KL stays at 0 → trust region never
engages anyway.

The architectural difference between kindle's per-step / small-rollout
cadence and standard PPO's "collect 2048 steps with frozen policy,
train 10 epochs of mini-batches" is fundamental. A real PPO refactor
would require:
1. A separate collect phase that snapshots obs/action/return arrays
   into a fixed buffer (decoupled from the lane buffers)
2. A train phase that mini-batches over that buffer (e.g., M=64
   mini-batch over B=2048 collected transitions, K=10 epochs)
3. Snapshot capture happens once at the start of the train phase
   (already implemented), but the K-loop iterates over mini-batches,
   not the full batch

This is multi-day work and the empirical evidence so far suggests
trust-region PPO doesn't outperform plain-PG e2e + LR-drop on these
toy tasks. Parked until a task surfaces where the trust region's
benefit outweighs the implementation cost.

### LunarLander head-to-head (2026-04-27)
Tested KL-PPO snapshot vs plain-PG e2e at identical config and
budget on LunarLander-v3 (the original motivation for trust region —
its local-optimum plateau).

| Approach | Mean | LR drop event |
|----------|------|---------------|
| Plain-PG e2e | -179 | step 34k @ -151 (well-timed; sustained) |
| KL-PPO snapshot | -380 | step 8k @ -140 (premature; crash to -900s) |

Same flags except `--use-kl-ppo --kl-use-snapshot --kl-beta 0.05
--kl-target 0.02`. 80k policy steps × 8 lanes = 640k env-steps each.

KL stayed at 0.0000–0.0019 throughout the snapshot run; adaptive β
collapsed to floor 0.0001 immediately. Diagnosis: lr_policy=5e-5
across a 32-row batch × 2 epochs produces weight drift too small to
register meaningful KL(π_new ‖ snapshot). Trust region never engages.

To get a non-trivial KL drift signal in the snapshot setup, kindle
would need either (a) much higher lr_policy with the trust region
constraining commit (but raises the entropy-collapse risk seen on
CartPole), or (b) a real PPO refactor that snapshots over a much
larger collect buffer with mini-batched K-epoch training. Neither is
in scope for the current investigation; the snapshot mechanism stays
in the codebase as a research artifact behind `--kl-use-snapshot`.

### Bug fixed in passing
Found and fixed: `entropy_beta_input_present` was true for
`use_kl_ppo + e2e + entropy_beta > 0`, but the KL graph variant
doesn't include the entropy regularization branch (no `entropy_beta`
input). Set_input call panicked with "unknown input: entropy_beta".
Gate now also checks `!use_kl_ppo`. Note: this means KL-PPO has no
entropy regularization on its policy loss — adding one would require
extending build_kl_policy_graph_e2e.

## DIAYN-options head-to-head on LunarLander

**Tested** 2026-04-27. Same context as the KL-PPO failure: testing
whether DIAYN (mutual-info skill diversity, kindle `5b7fbc7`) breaks
the LunarLander -132 ceiling that plain-PG e2e + LR-drop established.

### Configurations (LunarLander-v3, 80k policy steps × 8 lanes)
| Config | Mean | Notes |
|--------|------|-------|
| Plain-PG e2e | **-179** | baseline (no options, no DIAYN) |
| Options=4, DIAYN α=0.0 | -221 | options alone hurt |
| Options=4, DIAYN α=0.1 | -214 | tiny lift over α=0 |
| Options=4, DIAYN α=0.3 | -243 | DIAYN reward overwhelms extrinsic |

### Diagnosis
- Options alone (4 per-option heads, no DIAYN) hurt by 42 points vs
  plain-PG. The per-option-head gradient dilution outweighs whatever
  benefit option diversity might provide, at this budget.
- DIAYN at α=0.1 partially compensates (small lift over α=0) but
  doesn't recover plain-PG's level — entropy stays near max (1.30+)
  throughout, indicating the policy never commits.
- DIAYN at α=0.3 makes things worse — the intrinsic reward signal
  becomes large enough to drown out extrinsic, and the policy is
  rewarded more for trajectory diversity than for landing.

This aligns with the structural finding from MountainCar+DIAYN
(memory `project_kindle_structural_cap.md`): "DIAYN gives the agent
diverse skills to try, but if NONE of those skills stochastically
reaches the goal frequently enough to bootstrap policy gradient, the
agent never converges." LunarLander's positive-reward (+100 for
successful landing) is rare enough at random behavior that even
8 lanes × 6300 episodes = ~50k episodes don't see enough successful
landings to outweigh the DIAYN bonus.

### Conclusion across all 2026-04-27 LunarLander head-to-heads
Plain-PG e2e + LR-drop dominates at 80k×8 budget. Trust region (KL-PPO
snapshot) and skill diversity (DIAYN-options) both fail to break
through.

### LR-drop strength sweep — 100x is the sweet spot
Subsequently tested whether the prior reported "-132 ceiling" was a
sustained mean or a transient peak window. Extended budget to 200k
policy steps (1.6M env-steps) at three LR-drop strengths:

| `--lr-drop-on-solve` | Mean / 1.6M env-steps |
|----------------------|-----------------------|
| 10 (the previous default in scripts) | -221 (peak window -110, decays after step 100k to -480) |
| **100** | **-160** (sustained, no decay; range -130..-170 across 60k+ steps) |
| 1000 | -163 (similar to 100; freezing too aggressively loses minor learning) |

The prior reported ceiling of "-132 over 1.2M env-steps" was the
PEAK WINDOW from a 10x LR-drop run that was stopped before the
post-decay phase. With 10x drop, the policy reaches that peak
around step 80-100k but decays afterward as accumulated noise
pushes it into worse basins.

Setting `--lr-drop-on-solve 100` (instead of 10) preserves the policy
at the peak window indefinitely. This is the new recommended config
for kindle on negative-reward envs where the LR-drop strategy applies.
1000x freezes too tightly and the policy can't adapt to lane-batch
variance; 100x is the empirical sweet spot.

### Three-seed reproducibility (100x, LunarLander 200k×8)
| Seed | Mean / 1.6M env-steps |
|------|----------------------|
| 42   | -159.96 |
| 0    | -166.53 |
| 1    | -178.73 |
| **3-seed mean** | **-168.4** |

Robustly better than the 10x-baseline -221 across all three seeds.

### Generalization caveat
Quick CartPole and Acrobot tests at 80k×8/rollout=4/100x didn't fire
the LR-drop trigger at this budget — agents stayed near random. The
prior memos that report CartPole +218 (with 10x) and Acrobot -128
used different configs (longer budget, `--rollout-length` ≥5, etc.).
The 100x finding is currently validated on LunarLander only; verifying
on CartPole/Acrobot would require running each env at its own
working baseline config and substituting 10x→100x in the LR-drop arg.

## Reward-prediction-from-z auxiliary loss (`reward_pred_loss_coef`)

**Tested 2026-04-27, kept off by default.** Added a `z → r̂` MLP head to
the e2e policy graph with MSE loss against per-row single-step reward.
Rationale: prevent encoder collapse by forcing z to retain reward-
predictive features (vy, angle, leg contact for LunarLander). Anti-
collapse signal independent of policy/V/WM training.

### Symptoms
LunarLander 200k×8, 3 seeds, `reward_pred_loss_coef=0.05`:

| Metric (mean) | baseline | rpred only | recon only | both |
|---------------|----------|------------|------------|------|
| best_ep        | +14.7    | +7.5       | **+29.9**  | -3.6 |
| V↔R corr (post-training) | +0.36 | **+0.09**  | +0.47      | +0.35 |
| max R range    | +48      | +77        | +82        | +72  |

The reward-pred head **destroys the V correlation** (+0.36 → +0.09):
predicting noisy per-step reward and predicting smooth discounted
return ask the encoder for incompatible features, and the per-step
reward signal wins the gradient race because it has a tighter target.
Combining recon + rpred (both heads on) regresses below baseline as
the two anti-collapse signals interfere with each other.

### Why kept (not removed)
Code path is gated `reward_pred_loss_coef > 0.0 && e2e && !KL/PPO`,
default 0.0 (byte-parity with plain-PG e2e). Provides the negative
result for future experimenters considering reward-from-z as an
encoder regularizer. Reconstruction decoder (`recon_loss_coef`)
ships enabled in our LunarLander recipe instead — it tracks V at
+0.47 corr while improving best-episode return 2× over baseline.

## Pong short-budget replay and controller distillation

**Tested and removed 2026-08-23.** The calibrated pixel controller uses the
84x84 grayscale stack, connected components, two-frame velocity, and a
wall-reflected paddle intercept. With frame skip 1 and sticky probability 0.1,
it scored -11.50 across six completed games at 120k decisions; seeded uniform
actions scored -20.03 across 29 games. The observation stream is therefore
controllable without RAM access.

The following learned variants did not retain that competence:

| Variant | Held-out result |
|---|---:|
| Standard rollout-1 learner, frame skip 4, 128k decisions | -20.39 mean return |
| Event-balanced + 64-step positive-event SIL, 128k decisions | -20.54 mean return |
| Foreground residual, 128k decisions | -20.39 mean return; higher latent std only |
| Positive-event controller-window replay | 5 won / 566 lost points |
| All-action controller distillation, replay frozen at reset | 4 won / 105 lost points |
| Same distilled policy, greedy evaluation | 0 won / 128 lost points |

The first controller-window result also exposed two lifecycle defects that are
fixed independently: SIL had no standalone dispatch for default `n_step=1`,
and repeated PPO epochs could multiply SIL updates. SIL now runs once at its
policy cadence. Bounded positive-event capture remains as a general opt-in
primitive for continuing tasks, but it is not an Atari preset.

The warm-start CLI and external-demonstration API were removed. Keeping them
would imply a working imitation path when the matched evaluation shows none.
The replacement is a purpose-built offline/behavior-cloning objective with a
held-out dataset and state-balanced sampling, rather than another replay-weight
sweep. It is intentionally separate from the failed reward-shaped replay path.

### Direct supervised CE replacement

The replacement succeeds on the narrow retention question. The controller and
learner share a 64-value motion token derived from the model-visible frame
stack. Collection deduplicates states and balances actions 0–3; direct masked
cross-entropy reaches 100% on 768 training rows and 98.8% on 256 held-out rows
after 1,175 epochs. A separate process loads the saved Kindle weights without
updates or controller access and scores -12.60 over five games/120k decisions
(gate -15; matched random -20.03; controller -11.50).

This does not rehabilitate the removed demonstration replay API: it shows that
a correctly conditioned, purpose-built supervised objective retains the
controller. Autonomous sparse-reward Pong and million-decision evaluation
remain open.

### Reward continuation: full-policy forgetting and retained replacement

Continuing the distilled checkpoint for 120k decisions with raw signed Pong
rewards initially regressed the separate deterministic evaluation from -12.60
to -17.23. The policy-side encoder receives substantially larger gradients than
the action head; updating it erased the compact motion representation. Freezing
`policy_encoder.*` while training the head avoids that failure. A 24-lane
same-task GRPO batch is more robust than the first eight-lane probe: over seeds
42, 7, and 123, the source/fine-tuned point deficits are 109→80, 116→75,
and 143→82. The aggregate deficit falls 368→237 (35.6%), and completed-
game return moves -14.06→-13.43. Positive-point fraction rises 27.2%→32.9%
while retaining 86.0% of the source event count, ruling out a false win from
merely slowing rallies. Parameter inspection reports encoder maximum change 0
and policy-head L1 change 12.426; a separate load-only process reproduces the
gate.

Multiplying positive Pong rewards by three looked better on the training seed
but was rejected after matched evaluation: its aggregate point ratio was worse
and seed 7 regressed. The retained recipe uses raw, unbalanced environment
rewards. It demonstrates reward-driven improvement after supervised
initialization, not from-scratch reward learning.

### From-random Pong shaping ablations

Starting from random weights with the same structured motion token, 320k
decisions of raw signed Pong points collapsed to FIRE and evaluated at -21.00
with 0 won/824 lost points. A ball-progress potential also collapsed to FIRE.
Paddle-alignment and serve-bonus variants collapsed to NOOP or DOWN because
sticky actions and detector flicker made the apparent displacement a poor
causal reward target. A synthetic contextual bandit reached 100%, separating
these reward-definition failures from a broken online optimizer path.

A scalar tracking-consistency reward (+1 for matching the token-derived
NOOP/FIRE/UP/DOWN decision, -1 otherwise) still failed when classes were
imbalanced or entropy recovery enforced a high entropy floor. Inverse-frequency
class balancing plus disabled entropy recovery learned the mapping, but the
320k-decision final policy could regress after an earlier successful phase. The
retained harness evaluates a fixed balanced token set every 100 training steps
and restores the best policy snapshot. Its best snapshot occurs at step 2,600
and a separate zero-update reload scores -13.20/-13.60/-10.75 over three seeds,
with 98.7% or better active tracking on each.

This retained result is a task-specific shaping benchmark, not a discovery
benchmark: the optimizer receives scalar rewards rather than controller actions
or cross-entropy labels, but the reward function explicitly encodes the desired
tracking decision. Sparse-score-only Pong from random initialization therefore
remains open.

## Intrinsic-only CartPole credit and budget boundary

**Tested 2026-08-23.** Applying episode-return GRPO and SIL directly to
CartPole's negative-only homeostatic deviation penalty creates a length bias:
long balancing episodes accumulate more penalty than short failures. After
800k decisions that configuration collapsed to a 9.39-step recent return,
below the matched random baseline of 22.87. No new reward module was needed;
per-step GRPO removes the episode-length comparison, and SIL is disabled.

A frozen three-seed gate after 40k training decisions was close but not
reliable: returns were 226/334/185, so seed 44 missed the conventional 195
threshold. Using the full original milestone allowance—6,250 vector steps
across eight lanes, exactly 50k training decisions—raises the frozen results to
500/500/215. Matched random actions score 22.11/22.74/22.42. The retained gate
also disables extrinsic reward/events and prevents the externally measured
CartPole return from triggering a learning-rate change.

## Joint CartPole + Taxi interference and retained capacity

**Tested 2026-08-23.** The first shared 16-lane agent kept CartPole at 426 but
froze Taxi at only 17.6% completion. This exposed two core SIL assumptions that
were valid only for one task: a global positive CartPole return baseline
rejected successful negative-return Taxi episodes, and uniform transition
sampling let 500-step CartPole trajectories dominate shorter Taxi paths.
Per-task admission baselines and task-balanced replay raise the same narrow
model's greedy Taxi completion to 74.2% while restoring CartPole to 500.

Replay competition was not the remaining simultaneous limit. Reserving SIL
only for Taxi reduced its completion to 70.2%, and doubling narrow-model
training from 100k to 200k vector steps reached only 76.7%. The original dense
task-code model reached 81.8% under frozen stochastic evaluation. After task
codes became orthogonal, the current 16/32 rerun still reaches only 78.7%.
The retained 32/64 model passes all three seeds at CartPole
485.40/482.37/402.43 and Taxi 92.3%/95.6%/90.3%. Greedy seed-42 Taxi reaches
87.1% (CartPole 500), so the gate explicitly measures the frozen sampled policy
rather than claiming deterministic mastery.

Sequential switching exposed a different alias. With dense task codes and
global balanced rehearsal, CartPole→Taxi retained CartPole at 499.54 but Taxi
stopped at 83.4%; 150k steps worsened Taxi to 65.6%. In the reverse order Taxi
reached and retained 96.5%/96.2%, but CartPole never exceeded a 10-step return.
Filtering SIL updates to active tasks removed stale labels, yet dense codes
still shared every trainable task-projection column: wide/narrow Taxi reached
only 59.5%/68.8%, and halving SIL weight fell to 26.3%.

Orthogonal codes for the first eight task ids close the three-seed
CartPole→Taxi→CartPole gate: pre-switch CartPole is
488.18/493.80/486.31, Taxi is 93.9%/92.8%/93.3%, and retained CartPole is
253.50/500.00/229.98. Reverse order remains slightly unreliable: seed 43
retains Taxi at 89.6%, while seeds 42 and 44 pass. No threshold was relaxed to
hide that miss.

## Pendulum categorical imitation and continuous stabilizers

**Tested 2026-08-23.** A temporary 7-bucket Pendulum wrapper plus successful-
episode SIL did not retain a deterministic swing-up controller. Uniform replay
collapsed to the trajectory's 69% zero-torque majority action. Sampling action
classes uniformly raised controller agreement but still evaluated at -1325
with unit SIL weight and -1193 with weight 10 (expert -358; random about -1725).
Three buckets simplified the labels but still evaluated at -1148. The wrapper,
expert-step warm-start flags, and action-balanced SIL knob were removed rather
than adding an unproven categorical path.

That probe exposed the core continuous bug: Kindle multiplied the MSE target by
advantage instead of multiplying the per-row Gaussian loss. After correcting
the objective and adding an end-to-end continuous encoder, a second defect was
found: all 18 universal output heads contributed to the loss even though
Pendulum uses only one. The 17 padded heads dominated the objective. Once they
were masked, reward-only seed 42 scored -1860.34 versus random -1724.57; the
earlier -1498.87 result is not evidence of progress.
Online expert correction (DAgger) is retained in the native harness because it
does reach the expert regime on two of three seeds; it is not a generic
demonstration API and does not establish autonomous reward learning.

Two attempted stability changes were rejected. Bounding the Gaussian mean with
`tanh` degraded seed 7's 816k-step greedy evaluation to -750.10. Lowering the
policy-head multiplier from 5x to 1x made seed 7/123 pass but moved the failure
to seed 42 (-735.79); 2x left seed 7 at -622.30. With the active-dimension mask,
the retained 5x recipe evaluates at -359.02/-497.27/-362.44 across seeds
42/7/123 and remains a 2/3 result, not a reliable solve.

Several follow-up stabilizers were also removed after matched greedy
evaluation. Per-step credit and action repeat 8 scored -1657.55 and -1555.14;
episode credit with repeat 8 scored -1498.87 before the padded-head fix. A
temporary fixed-variance Gaussian KL penalty scored -1838.30 with episode
credit and converged to the -1498.87 constant-torque policy with per-step
credit. A Pendulum-specific height/velocity progress reward scored -1876.48
(repeat 1) and -1621.90 (repeat 8). A conventional GAE/bootstrapped A2C preset,
with and without stronger KL penalties, also converged to constant torque.
None supplied evidence for a general mechanism, so their flags and graph paths
were removed rather than retained as inactive surface area.

## ARC multi-game distillation aliases and optimizer shortcuts

**Tested 2026-08-23.** Extending the focused FT09/VC33 coordinate dataset to
eight searched L1 paths exposed four failures hidden by in-training MSE. The
pure object token mapped four distinct LF52 states to the same 64 values despite
different required clicks; only CD82 reached L1 in the first joint evaluation.
Hybrid object/pixel tokens remove all exact conflicts in the 39-sample set.

The coordinate head's former `[world-model latent, raw token]` input predicted
correctly when reloaded at the 39-lane training batch but saturated to unrelated
coordinates at batch 1. The latent was redundant with the adapter token and
forced a GPU encoder pass, so it was removed rather than patched with a batch-
specific checkpoint recipe.

Training action identity by passing searched labels through a synthetic-reward
DAgger loop reached 97.4% and later collapsed to the majority ACTION6 label;
including SU15/LF52's valid ACTION7 made the collapse reproducible at 41%.
A dedicated masked positive-CE update replaces that workaround and reaches
100%. Coordinate rates 0.0005–0.002 and widths 128/256 all bottomed out around
0.0021–0.0027 training MSE; widening was removed. The retained low-rate/freeze
recipe initially reached L1 on 7/8 games. SU15's fourth raw prediction was
`(31,33)` instead of `(28,35)`. Naively snapping it to the nearest visible
object still failed because the grid contains two-pixel-spaced candidates and
selected `(30,33)`. A generic motion prior succeeds: after two identical
executed click displacements it extrapolates the next target before applying
the same object projection. This preserves the irregular paths and closes the
fresh-checkpoint gate at 8/8; nearest-object snapping alone remains a recorded
failed shortcut.

### Shared coordinate head and aggregate early stopping

**Tested 2026-08-23.** Merging the eight L1 paths with FT09/VC33 L2 paths
produced 53 unique rows. A token-only shared coordinate head reached 100%
action accuracy and aggregate MSE 0.002194, and both deeper games replayed to
L2, but R11L regressed to the wrong second-click object. The same failure
remained after adding task context for only 20k updates (MSE 0.003330), so
neither aggregate training loss nor deeper-game success was a sufficient
retention gate.

The coordinate head now receives Kindle's fixed task embedding in addition to
the raw token. Continued fitting showed that the prior 0.0022 freeze threshold
was itself unsafe: MSE 0.002194 failed R11L while 0.002185 happened to pass.
The retained one-stage recipe uses a 0.0015 threshold and a 35k ceiling. With
equal-per-task gradients it reaches the ceiling at MSE 0.001792 and passes the
mixed-depth environment gate; the earlier row-weighted run was 0.001758. The
failed token-only and loose-MSE shortcuts remain removed from the recipe.

### VC33 L3 coordinate regression and ranked-object classification

**Tested and removed 2026-08-23.** Dynamic object search reaches VC33 L3, but
adding its path expands the joint corpus to 76 unique rows and exposes a wrong
abstraction for the learned executor. A shared task-conditioned coordinate
head reached 100% action identity and MSE 0.004448 but regressed R11L and failed
VC33 L3. Giving every task equal total coordinate gradient restored the prior
L1/L2 gates (unweighted MSE 0.005759), yet VC33 still stopped at L2. A
VC33-only 30k fit also stopped at L2 despite MSE 0.003614, ruling out only
cross-game interference. Continuous `(x, y)` regression averages nearby valid
objects when the program needs a categorical choice.

A second experiment encoded the target's deterministic object rank as a
discrete policy action and temporarily widened `MAX_ACTION_DIM` from 18 to 64.
This also failed to memorize the demonstrations: the 64/128 model reached
78.9%, a higher-rate continuation fell to 75.0%, a 128/256 model reached 72.4%,
and a conservative 100k-update fit reached only 86.8%. The hybrid token stores
features for six objects while required ranks reach 50, so a fixed rank head
cannot infer the candidate it is supposed to point at reliably. The wider
universal head and semantic-action path were removed. Search exports retain
object rank/count labels and the features of every current candidate. Their
replacement is now implemented: a variable-set pointer scores each candidate
after the stable ACTION6 identity is selected. It fits all 67 complex labels
and retains VC33 L3 after a separate checkpoint reload, without penalizing
native and Atari policies with dozens of padded logits.
