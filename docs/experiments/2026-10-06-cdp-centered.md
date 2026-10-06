# Centered CDP: make visual variation matter to the world model

The [Pong probes](../results/2026-10-06-cdp-pong-world.md) find useful CNN
coordinates even in the failed learners. Their recurrent states largely miss
the ball, and a constant target mean nearly matches their tiny cosine errors.
Test that specific loss weakness before more capacity or training budget.

## One mechanism

Control: qualified CDP with unchanged raw cosine distance. Candidate:
`--cdp --cdp-centered`. For the full B*T replay-target matrix U, form a
**detached per-feature mean** m=mean(U, rows). The candidate loss is
mean(1−cos(prediction−m, stop_gradient(U)−m)), retaining weight500 and the
existing squared-norm floor1e-8. All targets, including episode starts, remain.
The same mean applies to both operands; it is not estimated separately from
predictions. It cannot backpropagate through targets or become a policy input.

The mean is a training-loss statistic, not an encoder transform. Acting,
posterior inputs and raw forecast outputs remain unchanged. No extra parameter,
EMA state, reward shaping, RAM target, RGB decoder or encoder. This is a Kindle
ablation of the [published CDP objective](https://arxiv.org/html/2603.07083v2),
not an unchanged paper reproduction or guaranteed prevention of collapse.
Require full-batch BPTT and no row microbatch split so the statistic has one
unambiguous population. The CLI default and `--cdp` alone remain unchanged.

## Qualification

Current Meganeurab684ffd9 / Bladee349cddf, driver580.178.04; rebuild only for this
new loss path. CPU configuration/serialization/refusal and runner/audit tests,
formatting and strict Clippy. An independent F64 test checks production-width
centered cosine values and32,768 prediction derivatives over128 rows, including
a large common component and detached targets/mean. Retain the original cosine
test and one unchanged-CDP four-update upstream check to protect the default.

One1024-action/195-update production Pong smoke per arm, followed by frozen
restore/1024 actions/zero updates. Save actual zero-experience weights before
learning, verify their identity and the current finite final checkpoint,
encoder movement, and unchanged346 tensors during frozen evaluation. Initial
weights must match across arms at the same seed. These4096 actions are excluded
diagnostics, not learning evidence. Each GPU qualification process is bounded
to120s and any failure is reviewed before follow-up.

## Fixed learning allocation, only after reviewed qualification

- **Six fresh200k-action learners**,49,939 updates/run: control/centered1009,
  centered/control2017, control/centered3019.1.2M training actions total.
  Both arms use the same newly built native library. Old500k and200k results
  remain historical context, not controls for a changed backend.
- Keep Size1M/N8/B8/T16/H15/R32, microbatch8, replay100000, full18/sticky.25/
  repeat4/no reset no-ops,100000-frame cap, native input and one GPU RGB64
  resize. Encoder6e-6/dynamics4e-4/base4e-5, warmup1000, AGC.3,
  ac_grads=false, action-effects intrinsic coefficient1. Centering is the only
  changed learning mechanism; no coefficient sweep or seed replacement.
- Preserve each run's actual initial checkpoint before its first action or
  update. After training, audit counters/debt/finite checkpoint, then frozen
  final and initial policies. Sampled action policy, first3 natural episodes
  per each of8 streams, cap200k actions and30min/model. New held-out seed
  base3,500,000,000 plus learner seed, stream offset1,000,003 modulo2^32.
  Same evaluation seeds across both arms/initial controls. Retain all excess
  episodes, cutoffs and unfinished tails; capped cohorts are not complete.
- Exact saved tensors, full trajectory replay and whole stream-zero videos
  before advancing. Report all score/action/time curves, online last50 means,
  final frozen/initial scores and learner-seed bootstrap paired uncertainty.
  Compare observed reward-event separation/KL, not raw loss magnitudes across
  the two different objectives. No validation-selected world checkpoint.
- Expected about3h;5h service bound,60min/training process. Serialized GPU
  guard, Restart=no, KillMode=control-group, expected native device and>=2GiB
  sampled Vulkan estimated headroom. Record standalone allocation warnings;
  stop API/numerical/hard faults/deadlines for review. No NVML polling, recovery
  or blind retry. Heavy CPU audits: one CPU,2GiB, zero swap.

## Decision

Promote only if every centered seed improves over its own initial control and
the paired frozen-score improvement over unchanged CDP has a positive95%
learner-seed bootstrap lower bound. With three seeds that is an early screen,
not proof of generality or mastery. Retain wall time; no speed claim from lower
loss or smaller model. A promising change still needs the remaining original
games and sufficient budget to reach the unchanged DreamerV3 targets.

If the candidate does not meet this screen, retain all failures and reassess
the objective/temporal credit evidence. No automatic extension, seed rerun or
five-game restart. The full goal remains DreamerV3 quality on Boxing, Pong,
Freeway, Breakout and Qbert with CDP.

Artifacts: `runs/cdp-centered-20261006.n6XuQXmE`.

## Reviewed follow-up, October6

The [completed learning comparison](../results/2026-10-06-cdp-centered-learning.md)
passes its declared screen, but only4/72 centered matches are wins. Before a
larger allocation, apply the unchanged [frozen Pong probe](2026-10-06-cdp-pong-world.md)
to all six200k checkpoints, ordered control/centered for1009/2017/3019.
Same native/runner and fixed development trajectories:196,608 extra diagnostic
actions,30 GPU readouts/61,440 readout updates, zero actor updates. Keep every
control, event count, matched visible cohort, pixel check and exact346 tensors.
These reuse development probe trajectories, not a new unseen-game test. No
duplicate smoke or qualification for unchanged native.900s/model and45min
service bound; audit before any follow-up, no retries. Artifacts:
`runs/cdp-centered-world-20261006.EYV6C9au`.

One existing synthetic fixed-batch profiler has also completed under a separate
120s guard, using centered1009 without mutating its source checkpoint. It
measures session/kernel costs, not GPU utilization or gameplay. Artifacts:
`runs/cdp-centered-profile-20261006.EvJjduZZ`. Imagination has3,100 dispatches;
ordinary GPU timestamp median10.91ms versus13.28ms synchronized wall time.
Per-dispatch instrumentation inflates execution substantially, so its family
shares are diagnostic, not uninstrumented whole-agent fractions. No larger
learning allocation or optimization candidate is launched by this follow-up.
