# Kindle: live learning with Dreamer and predictive representations

Updated 2026-09-06. This is the single project plan, replacing both earlier
Dreamer plans. Detailed logs stay in `runs/`; decisions and their strongest
evidence belong here. Working rules are in [AGENTS.md](../AGENTS.md).

## Goal and current decision

Build an agent that keeps learning while it acts. Begin with one game, pixels,
a compact action vocabulary, and game rewards or explicit human guidance.
Progress toward intrinsic motivation, retention across tasks, and independent
Kindles sharing useful experience. Keep the implementation small, native Rust,
and efficient on Meganeura and Blade.

The next architecture is Dreamer's recurrent belief and imagined actor/critic
with an action-conditioned prediction objective over frozen perceptual features.
Keep the measured DINO feature-reconstruction agent as the control. Establish
whether predicting unseen observations improves dynamics and control before
changing the visual encoder or building pretraining infrastructure.

A 100-hour game exposure is a later engineering gate. First prove that acting
and learning can coexist at the declared control rate and that a shorter run
produces useful learning. Atari's step-driven laboratory protocol and a real-time
lifetime are different experiments; neither clock substitutes for the other.

Guidance should enter through ordinary, recorded interactions: action overrides,
demonstrations or explicitly labeled feedback. Preserve the actual executed action
and distinguish game reward from human feedback in experiment records. Report
unguided evaluation separately; do not attribute assisted behavior to the agent.
Build a small adapter for the first real guidance source, not a preference-learning
framework in advance.

## What exists

| Component | Implementation and purpose | Current limitation |
| --- | --- | --- |
| Pixel boundary | `kindle/src/env.rs`: RGB8, categorical actions, reward, terminal/truncation flags | Synchronous trait; no desktop-game capture/input adapter |
| Perception | `kindle/src/vision/`: frozen DINOv3 ViT-S/16, deterministic letterboxing, fixed projection/pooling | Single frame; fixed 7×7×64 observations; GPU readback |
| Recurrent model | `dreamer/networks.rs`, `world.rs`: block RSSM, categorical posterior/prior, reward, continuation, optional reconstruction and causal forecast | Causal learning survives frozen native evaluation across three seeds; matched Pong control completes, Pong replication pending |
| Learning | `dreamer/agent.rs`, `behavior.rs`, `distributions.rs`: posterior replay, imagination, two-hot returns, actor/critic, slow value | Host/GPU round trips; acting and learning share one mutable core |
| Replay | `dreamer/replay.rs`: bounded FIFO, fresh-sequence queue then uniform sampling, cached context | f32 storage; no lifetime sample or persistence |
| Runtime | `dreamer/runtime.rs`: initialization, LaProp with leaf-wise AGC, inference synchronization | Parameter synchronization passes through host memory |
| Checkpoints | Model, optimizer, counters, return normalizer, backend/head/hash identity, encoder/tensor-file fingerprints and required-tensor checks | Replay/live state absent; non-atomic generation; resume is a new data segment |
| Native tests | `kindle-gym`: episodic/persistent visual GridWorld and representation/dynamics probes | Small deterministic integration task, not general lifetime evidence |
| Game experiments | `python/examples/atari.py`, strict summaries and optional frozen-evaluation video | Atari only; no lifetime archive or real-time deadline scheduler |

Python handles adapters and analysis; inference and learning remain Rust.
The 12M preset has about 10.4M trainable parameters with this frontend; its
preset name is not an exact parameter-count match to upstream Dreamer.

The `exp/dreamerv3-baseline-12m` branch contained work missing from the starting
checkout. Its landed backend pins, full-recurrence row microbatching,
full-precision gradient safeguards, and score/provenance checks are recovered. Do not restore
its chronological experiment diary as a second plan.

## Evidence to preserve

| Completed experiment | Result | What it establishes |
| --- | --- | --- |
| Native sparse-food GridWorld, DINO reconstruction scale 0.25 | Online learning and 2,500 rewards in 10,000 frozen greedy actions | Complete loop works on a simple visual task; diagnostic rates differ from Atari |
| Seed-split Pong perception probes | Coordinate R² 0.9867 pooled versus 0.9890 unpooled; pooled reward-event AUC 0.9929 | Static task cues survive compression; this is not a temporal test |
| Local pinned upstream D3, size1m, Pong seed 0 | −20.462 in the declared final window | A local control already exists; JAX 0.6.2, ALE 0.9.0, bfloat16, learned RGB perception |
| Local pinned upstream D3, size12m, Pong seed 0 | −10.667 across six final-window episodes; 28.6 minutes | Current executable capacity control; one seed and runtime/perception differences limit the claim |
| Kindle 1M, full BPTT, Pong seed 0 | −17.8 in the same nominal window | Useful small-preset comparison; perception, runtime and reset accounting differ |
| Kindle 12M, full BPTT, row microbatch 4, Pong seeds 0/1/2 | −4 / +6 / −2; seed mean 0; 2/3/4 complete episodes | Three independent uninterrupted learning runs, above matched random −20.623 |

The 12M runs each execute 100,000 actions and 24,729 learner updates in about
55 hours on the RTX 5080: about 0.5 actions/s at train ratio 256. These are online
training-window scores, not frozen endpoint evaluations. Few final-window
episodes and three seeds leave substantial uncertainty. Historical published
200M and third-party 12M scores are context, not matched acceptance thresholds.

Reproduce the three-seed calculation:

```bash
PYTHONPATH=python python python/examples/summarize_atari_scores.py \
  runs/atari-pong-12m-bptt64-micro4-ac3e4f6-published-seed0-100k.jsonl \
  runs/atari-pong-12m-bptt64-micro4-ac3e4f6-published-seed1-100k.jsonl \
  runs/atari-pong-12m-bptt64-micro4-ac3e4f6-published-seed2-100k.jsonl \
  --minimum-kindle-seeds 3 --require-uninterrupted-kindle
```

The local upstream directory is
`runs/upstream-dreamerv3-e3f02248-pong-size1m-published-seed0-100k`.
It contains a completion marker, exact config and runtime manifest. The requested
JAX 0.4.33 could not run on Blackwell; record the actual 0.6.2 runtime. Upstream
counts action-free reset records in its step budget; Kindle counts executed
actions. ALE versions differ. The new 12M capacity control is in
`runs/kickoff-upstream-pong-12m-seed0-100k-v2`; do not repeat the completed 1M
control. Upstream still trains much faster than the optimized Rust loop.

## Architecture and causal objective

For an observation arriving after action `a[t-1]`:

```text
previous belief + a[t-1] ──► h[t] ──► predict u[t]
                              │
RGB[t] ──► frozen E ──► u[t] ──┤
                              ▼
                         posterior z[t]
                              │
                 reward / continuation / imagination
                              │
                         actor / critic
```

`h[t] = f(h[t-1], z[t-1], a[t-1])` precedes the new observation.
`u[t] = E(frame[t])` is a fixed perceptual target. The predictive head reads
`h[t]`; it must not read `z[t]`, the current adapter output, or the target frame.
The posterior then incorporates `u[t]` for action selection and replay.

Compare three objectives with perception and behavior fixed:

1. Reconstruction control: predict `u[t]` from `(h[t], z[t])`, as today.
2. Predictive auxiliary: retain reconstruction and add `g(h[t]) → u[t]`.
3. Prediction only: remove the posterior decoder; retain the predictive head,
   reward, continuation, balanced KL and replay value.

Use the same spatial decoder structure and matched downstream initialization for
all heads; only the predictive trunk lacks the current posterior's stochastic
input. The first kickoff predictor used a global MLP, confounding the objective
with a different spatial parameterization despite similar parameter counts.
Keep those pilot results labeled; do not interpret them as a causal-loss ablation.

The hypothesis is that a deterministic target scored before observation gives
the recurrent model a more direct signal than posterior categorical prediction
alone. [Dreamer-CDP](https://arxiv.org/abs/2603.07083v2) motivates this connection;
its learned encoder, cosine objective and optimizer schedule differ from a
frozen-DINO experiment. Name those differences explicitly. Renaming the existing
posterior decoder is not the pivot.

Frozen targets exclude joint encoder/predictor collapse without online EMA
encoders or SIGReg. A constant predictor can still fit background statistics:
compare persistence, an unconditional mean, unrelated actions, and action-relevant
reward/state probes. Lower average feature loss alone is not a control result.

Mask unpredictable reset observations out of the future loss. Preserve terminal
and truncation alignment, gradients through preceding observations, and row
microbatch equivalence. Prefer full-sequence BPTT for this comparison; shorter
chunks cut the temporal credit path being tested. Measure loss normalization
and gradient magnitudes: the earlier 1/3,136 feature reduction suppressed a
decisive representation-learning signal.

Keep categorical RSSM sampling, balanced KL, two-hot reward/value heads, learned
continuation, short imagination, REINFORCE actor, critic and slow value. No
planner, critic replacement or action-space redesign is needed for this gate.

## Execution gates: status and next work

### 1. Recover and measure the working control

Recovery is complete, both lockfiles and revision metadata agree, and native,
Python and serialized GPU tests pass. Pixel-agent restores verify the actual DINO
file's SHA-256 before GPU construction. Legacy checkpoints without a fingerprint
require the pinned reference file, not an arbitrary same-shaped encoder. Preserve
older checkpoints under their original backend provenance.

Measure replay sampling, posterior inference, imagination/targets, world update,
behavior update, synchronization and total time. Obtain GPU timestamps or a trace
for dominant stages. Report unprofiled wall time separately: kernel profiling
changes launch overhead. Compare the same workload before/after optimization and
verify parameter-update equivalence.

Uncached host reads dominated the initial profile. Batched cached parameter and
output reads, followed by removal of three forced graph materializations, reduce
the 12M canary from 7.42 to about 1.05 s/update (**7.06×**),
preserving all 241 parameter/optimizer/slow-value tensors by name and eight
world/behavior reports exactly. These are medians after two warmup updates, not
end-to-end game throughput. A 10k-action native confirmation also preserves all
2,230 update reports and 241 tensors, then retains 2,500 frozen food rewards.

Complete dispatch capture in an isolated instrumentation build identified 11,264
forced copies writing 7.82 GB per microbatch. Removing them lowers a world B4×T64
gradient pass from 76,016 to 64,752 dispatches and from 154/221 ms GPU/wall to
119/175 ms, without optimizer/accumulation. Device allocation falls from 13.63 to
6.11 GB. Production dependency pins stay unchanged. Full instrumentation costs
2.39× wall time, so lower-perturbation windowed profiling remains useful.

A full 16-row batch now fits at 6.30 GB and measures 0.54 s/update, but changes
floating-point reductions and is not byte-exact after eight updates. Validate its
learning behavior before replacing the measured microbatch-4 Atari control. Next
optimize grouped RSSM/layout operations and batch non-recurrent heads across B×T.
Preserve full recurrence, gradient range and measured learning; launch gaps and
arithmetic need different fixes.

### 2. Test prediction on controlled sequences

The optional deterministic future-feature head is implemented. GPU tests verify
causality, action sensitivity, reset masking and gradients through earlier
observations; a zero reconstruction weight removes that decoder's parameters.
Checkpoint round trips and predictive microbatch equivalence pass for the
matched spatial head. After identical 10k random-trajectory native training,
reconstruction gives 2,500 frozen food rewards; prediction-only and auxiliary
give 2,499 each. The auxiliary reduces terminal-inclusive held-out feature MSE
by about 5% with 29% more parameters, and strongly beats unrelated actions.
Prediction-only's error is slightly higher than the control's. The earlier
global-MLP pilot was confounded by head structure and is retained only in the report.
Reconstruction remains the default. Subsequent prediction-only, agent-controlled
persistent seeds 0/1/2 collect 1,137/1,237/1,278 food in 10k actions and retain
2,495/2,498/2,373 food with 0/0/5 deaths in 10k frozen greedy actions. Every seed
passes the declared training and frozen gates, without forced coverage or an
intrinsic bonus. Causal Pong seed 0 completes 100k agent-selected actions and 24,729
updates: final-window mean +18.1429 over seven games. The final checkpoint wins
all 28 completed games in 50k frozen sampled actions, mean +19.0714, versus
−20.3455 for the zero-update baseline. All training reports and checkpoint
tensors are finite. This is first positive control on both games, not proof of
superiority to reconstruction or multi-seed Pong reliability. The same-build matched
reconstruction run also succeeds: final-window mean +17.6667 over six games and
frozen mean +19.2143, with 28 wins in 28 games. Both complete the same 100k-action,
24,729-update budget in about 3.83 hours, excluding construction. Finish the
independent Pong seed before a broader claim. The
[self-learning report](experiments/2026-09-05-self-learning.md)
records the acceptance criteria, raw artifacts and pending comparisons.

The matched native reconstruction control exposes a frozen-policy failure:
1,100 training food and 247 in each final 1k window, but only 2 food / 98 deaths
in 10k frozen greedy actions. Tensor/configuration/counter checks pass. Diagnose
the action-mode and fresh-recurrent-state changes separately; do not replace the
failed declared endpoint with a sampled result. Strong online training alone
does not establish retained behavior.

Native single-variable AGC-off and BPTT-8 tests also retain 2,500 rewards on that
same trajectory. The task does not separate their control quality. Keep the
clipped full-BPTT Atari control; these toy ablations do not justify changing it.

Advance if action-sensitive held-out prediction and native control are retained.
If the auxiliary works and prediction-only fails, keep the auxiliary and diagnose
missing posterior information. Removing a decoder is not itself the objective.

### 3. Exercise sparse reward and continuity

Keep Pong as the dense Atari reference. Use Private Eye under the same pinned
wrapper for the first sparse Atari probe, with random and extrinsic-only controls.
Record reward-event counts; a short rare-reward curve cannot establish generality.

The first 20k-action seed-0 check is complete: random scores 700, extrinsic-only
−2,900 and mixed novelty 1,251. Both learners make 4,729 full-BPTT updates. The
mixed result comes from one 4,069-reward episode, followed by 8,000 actions with
no reward events. Replicate this discovery before treating it as sustained sparse
competence or extending the run budget; no Atari-100k score is claimed.

The native `--persistent` world handles exhaustion and respawn through ordinary
dynamics, retains food progress and requires no harness time-limit resets.
Measure food, deaths, action diversity and return per fixed interaction window.
This is a continuity integration test, not broad lifelong skill. Privileged
environment state must not enter observations or intrinsic reward.
The first agent-controlled extrinsic-only 10k run collects 1,100 food with 37
deaths and no harness resets; its final windows approach 250 food/1,000 actions.

### 4. Add one bounded intrinsic-reward experiment

The optional `--visitation-bonus` uses a fixed 16-bit random-hyperplane hash of
frozen visual features and 65,536 saturating counters. Subtract a fixed,
checkpointed first-observation reference: an uncentered native smoke collapsed
2,001 observations into five buckets; centering occupied 637 on the same stream.
Versioned counts survive episodes and checkpoints. Score each arrival before updating its count and
preserve the intrinsic scalar in replay. Both runners report intrinsic and
extrinsic returns separately. Validate its perceptual discrimination empirically.

This is a novelty control, not epistemic uncertainty. Compare extrinsic-only,
intrinsic-only and mixed reward with a fixed scale from the outset. Inspect
whether the bonus responds to controllable states or irrelevant visual noise.
Disagreement or learning progress follows a specific observed failure; raw
world-model prediction error is not the default reward.

The first continuing native comparison gives food/deaths of 1,100/37 for
extrinsic-only, 1,116/32 for mixed reward and 150/60 for intrinsic-only over 10k
actions (one seed each). Intrinsic-only is roughly random overall and its game
reward stalls late, despite more evenly distributed visual visitation;
the small mixed/control difference is inconclusive. Retain this bounded bonus
off by default. Its single-episode Private Eye discovery is worth replication,
but neither experiment demonstrates a reliable general exploration solution.

This baseline combines historical intrinsic and extrinsic scalars when making
replay targets, so the world reward head learns a decaying visitation signal too.
That signal is nonstationary, unlike a fixed game reward. A next intrinsic design
should separate environment-reward prediction from current intrinsic valuation
in imagination, and evaluate held-out dynamics plus later guided adaptation,
not only bucket coverage or unguided game score. [Plan2Explore](https://arxiv.org/abs/2005.05960v2)
provides the relevant precedent: an exploration policy maximizes imagined latent
disagreement, then the learned world model supports downstream task policies.
Test a small uncertainty proxy only against the existing bounded novelty and
extrinsic controls; disagreement is not automatically calibrated uncertainty.

### 5. Make real-time learning feasible

For action rate `f`, batch `B×T`, ratio `R`, and update duration `U`, learner
demand is `f × R / (B×T)` updates/s. Acting, perception and learning must fit
available device time with headroom. At 10 actions/s, B16×T64 and R256 require
2.5 updates/s, roughly 2.6 times the optimized microbatch-4 12M canary throughput,
before perception and acting costs. The admitted full-batch candidate reduces
that gap to about 1.35× but still requires learning validation and runtime headroom.
The original read path was roughly 20 times short.

Report p50/p95/p99 action latency, missed deadlines, effective ratio and training
credit. Bound backlog explicitly; dropping credit is an experimental policy to
report. When concurrency is necessary, use separately owned acting state and
versioned parameter snapshots. Threads around the current mutable core are
insufficient. Preserve arrival order and explicitly mark any observation gaps;
bounded queues must not turn missing transitions into fictitious adjacent frames.

Start that separation with an inference-only frozen actor. Current evaluation
restores build all training graphs: native full-BPTT construction takes about
90–100 seconds versus five for BPTT 8, even when no update will run. Match the
new actor's logits, recurrent states and sampled actions against the frozen
monolithic core before adding versioned learner snapshots or concurrent queues.
Do not shorten training's credit horizon merely to make evaluation cheap.

Choose the full-run rate and ratio from measurements. If smaller models or
lower ratios are required, measure their learning curves. Do not silently slow
a native real-time game to accommodate Atari's compute budget.

## Perception and pretraining ladder

DINO remains the first control: native inference is validated and static Pong
probes are strong. A four-seed motion probe improves mean displacement R² from
0.247 to 0.676 by appending causal feature differences, but horizontal ball motion
remains weak (0.173). The trained seed-0 Pong RSSM reaches mean motion R² 0.768
and horizontal-ball R² 0.608 on the same held-out seeds, without extra input
channels or learner updates. Keep the current frontend: there is no demonstrated
temporal-capacity hole in this trained belief. Early sample efficiency and other
games remain open questions; measure downstream learning and latency before a
temporal-input or encoder replacement.

[LeVJEPA](https://arxiv.org/abs/2608.27395v1) is a later frontend candidate. Its
[released checkpoint](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)
is a 303M-parameter ViT-L, much larger than the current ViT-S. Pin actual code and
weight hashes, inspect temporal attention, and benchmark before a native port.
Use causal patch tokens and bounded history. Check that special tokens cannot
leak future frames into earlier patches; reset history at declared boundaries.
A cache needs bounded size and an explicit positional policy.

Build an encoder interface when there is a second measured frontend. Matching
7×7×64 shapes does not make spaces or checkpoints compatible: replay must carry
encoder/projection identity, not merely dimensions.

Offline visual adaptation and dynamics pretraining are optional.
[Pixels to Play](https://arxiv.org/abs/2508.14295) supplies the motivating aligned
video/action data. Exclude the eventual held-out title from action training and
model selection; disclose passive-video overlap. Keyboard macros are game-specific
and need common action meanings before dynamics transfer. Reset actor, critics,
game reward/continuation heads, their optimizer state, return normalizers, replay
and recurrence when evaluating transferred perception/dynamics.

[LeWorldModel](https://arxiv.org/abs/2603.19312v3) motivates compact prediction;
its controller and end-to-end regularization are separate choices, not prerequisites.

## Lifetime storage and recovery

100k replay frames cover 2.8 hours at 10 actions/s. Size storage by measured
retention needs. Current 12M f32 observations/context use about 2.3 GB before
container overhead; a million such entries do not fit comfortably on this host.

First pack frozen features to f16 and categorical context to indices, measuring
error and update effects. Then compare FIFO against a bounded reservoir of
contiguous chunks, retaining context, flags and representation provenance. Use
one simple mixture and report sample ages. Separate world-model forgetting from
actor forgetting.

For a full lifetime, archive timestamped actions, both rewards, flags and
compressed video in append-only chunks. Measure bytes/hour before choosing disk
capacity. Checkpoint parameters/optimizer, normalizers, RNG streams, replay and
scheduler together with an atomic generation manifest before claiming fault-tolerant
continuation. Mark model-only restores as recovery with replay loss. No game
rewind or branching is permitted.

Current saves detect damaged/mixed tensor files through metadata fingerprints;
restores also require all model and optimizer tensors instead of accepting the
backend's partial-load fallback. This detects a torn overwrite but does not retain
the preceding generation. Legacy files without hashes cannot prove generation
consistency. Keep a separate known-good checkpoint in the meantime.

## The eventual 100-hour contract

Select a locally controlled immutable game with a native objective, benign compact
actions and trustworthy reward/terminal extraction. An offline Doom-like game is
a candidate after the native and sparse gates. Keep the eventual held-out title
out of architecture selection.

Predeclare version, control/observation rates, action repeat, reward mapping,
seeds, compute, initialization, evaluation windows and primary score. Record wall
exposure and actual action/frame counts. Natural deaths, restarts, menus and
evaluation count. No cloned environments, save-state search or hidden target-game
demonstrations enter training.

Collect random and hour-zero controls. A novice-human baseline is useful when
available, but does not block engineering. Fix an absolute competency threshold
and an improvement-over-random threshold. At fixed times, freeze learning and
evaluate the declared sampled policy on natural episodes; retain videos and
count all interactions. Never replace the final policy with the best checkpoint.

Advance through systems smoke, short learning/kill tests, a stability pilot,
one 100-hour engineering lifetime and at least three independent lifetimes with
unchanged settings for a scientific claim. Gates are stable computation, bounded
training debt, improving real behavior and calibrated predictions, not elapsed
time or falling scalar loss.

## After one reliable learner

Test retention across two games with compatible actions. Then share immutable
experience chunks between two independent Kindles, carrying environment version,
action meanings, encoder identity and behavior-policy provenance. Measure useful
transfer and harm from mismatched experience before swarm services or shared
optimizer state. Recompute the receiving agent's recurrent context rather than
sharing learned latent states as if their coordinates agreed. Another Kindle's
visitation bonus is not the receiver's novelty; keep reward-generator identity
and ownership explicit.

Goal conditioning, learned preferences, uncertainty-aware exploration, selective
planning and hierarchy return when a measured failure calls for them. No parallel
legacy planner stack, speculative universal framework or premature pretraining
pipeline belongs in the current core.

## Reproducibility pins

- Dreamer behavior: `danijar/dreamerv3` at
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- Recovered code: `exp/dreamerv3-baseline-12m` at `26a34aa`.
- Meganeura: `bd6be0882c53b94f65f164f88464cc6b24e9df4d`.
- Blade: `b208f3b1f97196c2971436b5726e61e71b149c37`.
- DINOv3 ViT-S/16 snapshot: `114c1379950215c8b35dfcd4e90a5c251dde0d32`;
  measured SHA-256:
  `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`.

Record actual source/configuration for every run and preserve older backend
results under their original provenance. See the
[kickoff experiment report](experiments/2026-09-05-kickoff.md) for measurements,
commands and remaining experimental checks.
