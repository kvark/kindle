# Freeway zeros: reward discovery fails before the comparison is informative

The user stopped the comparison for investigation at 20:42 UTC. Seven pairs
finished; CDP seed3019 was interrupted at 126,408 actions/31,541 updates.
The partial trajectory ledger and all250 saved tensors pass CPU audits, but it
is not a completed run. The controller and GPU workers are stopped. No GPU
wedge or recovery caused this stop. The five-game goal remains unfinished.

**The evidence points to failed exploration, not a disconnected reward or
learner path.** Continuing the unchanged Freeway matrix was the wrong next
diagnostic. We should have checked reward discovery after the first matched
zero-score group rather than repeating it across more seeds.

## What happened

- Across1,526,408 actual training actions and381,114 updates, all eight complete
  or partial Freeway traces contain **zero nonzero real rewards**. Raw rewards
  match both stored reward channels exactly; no clipping/missing-reward bug is
  found by this check. All18 actions occur; the agents are not stuck issuing NOOP.
- Every recorded Freeway update has zero imagined reward, return and absolute
  policy advantage. Entropy stays about2.89035–2.89037 nats, near the uniform
  18-action maximum, log(18)=2.890371758. The reward-directed policy term has no
  signal; entropy regularization favors remaining random.
- The world model is still being optimized: checkpoint optimizer state is
  nonzero and its training losses fall substantially. This is not proof that
  its representation is sufficient for control. Saved optimizer moments are
  not raw gradients, and nonzero entropy updates are not reward-driven learning.
- The longest consecutive UP-family streak is11 actions. A scripted crossing
  first obtains reward after43–44 actions. This is evidence for insufficient
  temporally coherent exploration, not a proof that every crossing requires an
  uninterrupted43-action sequence.

Freeway rewards reaching the far side, not partial progress through the lanes.
See the [official ALE description](https://ale.farama.org/environments/freeway/).
No discovered reward -> zero predicted return/advantage -> nearly random policy
is the observed failure loop. Faster world-model updates cannot supply a missing
task signal by themselves.

## Discriminating checks

1. **Independent adapter/reward control.** Same full18 actions, sticky.25,
   repeat4, native pixels, seeds1009/2017/3019. For6,144 actions per policy/seed,
   uniform random actions score0/0/0 episodes, while always UP scores21/21/23
   on each seed. Every wrapper step's reward, termination, RAM state and frame
   count matches a separate raw-ALE reference. The adapter can execute a crossing
   and report its reward. These36,864 policy actions plus36,864 raw-reference
   actions are excluded CPU diagnostics; no learned policy is constructed.
2. **Positive-reward path on GPU.** Fresh, unchanged qualified CDP, same config
   and seed1009. Exactly1,024 actions/195 updates: UP overrides for512 actions,
   then sampled policy for512. Eight real rewards reach replay; the learner
   samples296 positive-reward transitions, including repeated samples. First
   real reward arrives at aggregate action344; first nonzero imagined advantage
   at376. Maximum imagined reward mean=.0004542 and mean absolute advantage=
   .0009246. All250 checkpoint tensors are finite, debt is zero, and the
   120-second host-guard declaration passes with nine sealed source checks and
   no new allocation warning. The guarded job finishes in about12s including
   initialization; the loop/checkpoint interval is5.63s. This confirms signal
   propagation, **not autonomous competence or a new assisted comparison arm**.
3. **Existing reward-bearing learning control.** Seaquest CDP/RGB seed1009 uses
   the exact same native library and respective config dictionaries as Freeway.
   Their existing200k-action logs contain3,168/2,467 positive-reward events,
   first seen at aggregate action696; raw/stored rewards match. Advantages become
   nonzero and policy entropy falls. Final online scores are561.2/278.0.
   No Seaquest rerun was needed. This argues against a globally disconnected
   learner; it does not prove every Freeway-specific learning property correct.
4. **Authors' data.** The pinned official
   [DreamerV3 Atari-100k score file](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/scores/atari100k-dreamerv3.json.gz)
   contains five Freeway seeds, each with48 recorded scores, all zero. The local
   file matches the repository blob exactly. This is contextual published
   evidence, not our matched small-model protocol; zero Freeway reward is not
   unique to Kindle or CDP.

## Decision

Keep the matrix stopped. Do not increase model size, retry unchanged runs or
discard negative results. No production learner code changed in this diagnosis.
The next research decision is how to achieve first reward through generic,
temporally coherent exploration. A reward-bearing game such as Boxing can test
representation/learning improvements; Freeway is an exploration test, not an
informative representation ranking while every method receives zero reward.

Before restarting any larger comparison, declare a small experiment that
measures time to first reward, reward-event coverage in replay, nonzero policy
advantages and subsequent unassisted return. Prefer the roadmap's one generic
GPU-compatible exploration mechanism over a Freeway-specific UP/action aid.
The scripted pulse above is only a diagnosis. The remaining requested games
and methods are still owed; this finding is not a completed five-game evaluation.

[Compact measurements/configs](2026-10-04-freeway-zero-reward-diagnosis.json) ·
[Full CPU diagnosis](../../runs/freeway-diagnosis-20261004.zrnrKBca/result-reviewed.json) ·
[GPU pulse declaration](../../runs/freeway-diagnosis-20261004.zrnrKBca/reward-pulse-spec.json) ·
[GPU pulse trace](../../runs/freeway-diagnosis-20261004.zrnrKBca/reward-pulse.jsonl) ·
[Seven completed pairs, curves and rollout links](2026-10-04-cdp-atari-comparison.md)

The initial scratch diagnosis incorrectly compared a scalar raw reward against
the stored `[extrinsic,intrinsic]` pair and counted false mismatches. The reviewed
version checks both components and finds zero mismatches; the existing production
auditor already did this correctly. The original diagnostic output is retained.
This correction did not change native code, study evidence or the ALE controls.
