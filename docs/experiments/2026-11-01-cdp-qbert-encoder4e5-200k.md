# Fresh200k Qbert comparison at encoder4e-5

The [bounded check](../results/2026-11-01-cdp-qbert-encoder4e5.md) passes all
execution, replay and frozen-invariance checks, with conservative200k cost
28m43s below35 minutes. Select one fresh learner to measure whether encoder4e-5
improves the cheaper Qbert recipe. The retained2e-5 control reaches greedy
13,785.417/sampled9,987.500 in30m16s. Both means miss15,000; the stronger H5/R32
recipe passes on one of two learners at about40 minutes. This comparison aims
to improve the cheaper point, not to establish a universal learning rate.

Change only encoder2e-5→4e-5. Keep centered CDP/12M48/Auto/N8/B16/T16/H15/R16/
expert100, base/actor4e-5, dynamics4e-4, warmup1000, action-effects coefficient1,
actor-unimix0, full18 actions, microbatch16 and replay100k. Keep all-state
corrective labels, all action weights1, world imitation gradients,20% episode
continuation guidance/cap1920, native max-pooled frames and one GPU resize to48.
Native447fe50, the exact retained runner and teacher stay fixed. Use the qualified
runtime`/x/Temp/kindle-breakout-corrective-20261028/runtime`; no build is needed.

Train200,000 aggregate actions/12,477 updates under2100s. Require first update376,
residual credit.5 and a verified100k/6227 recovery checkpoint. No control or check
weights, replay, games, belief, RNG or experience enter this learner. Seed4013
and evaluation4000004013 are reused development inputs, not independent
confirmation. No checkpoint selection, extension, retry, cap increase, rate sweep
or automatic successor follows this declaration.

Assess fixed final and actual-initial checkpoints with sampled and greedy
actions: first3 natural episodes per8 lanes=24 per arm, guidance/labels off,
zero updates and truncations. Greedy remains primary; retain sampled separately.
The original mean15,000/90%-first-pyramid gate is unchanged. Each arm has300s
and the existing600k action ceiling. Preserve excess episodes and partial tails.
Report frozen gameplay, own-initial-subtracted gain per training minute, both
action modes, first-pyramid counts and total cost. Training curves include aid
and cannot establish unaided competence.

Require357 control/check-matching initial tensors, matched pre-update training
semantics,357 finite/invariant tensors per frozen arm, matching actual-initial
sampled/greedy traces, exact sources/config/counters, finite reports and changed
CNN/reward/actor/value. Verify raw expert world/actor losses, labeled-row/weight
accounting and actual override exposure. Complete training pixel/action/label
replay plus frozen outcome replays/videos, host guards, curves, coverage and
comparison. Ordinary frozen logs lack live pixel hashes; their replay does not
claim equality to unavailable live images.

Training replay cap1200s, enclosing audit1800s, whole main service5400s.
Existing progress, recovery, postprocessing and terminal-collection observers
retain their deadlines and never start or restart GPU stages. Preserve failures;
CPU-only repairs must retain original failures and avoid repeating completed
training, evaluation, replay or plot stages. Check live state before launch.

CPU preflight1.372s passes five actual CLI paths,24 config cases,49 path checks,
15 script compilations,eight startup-header cases,four observer-parser cases,
seven recovery-config cases,357 schema tensors and257 bounded/235 control pins,
with zero native/environment construction. The existing encoder flag is replaced,
not duplicated; both old rates are rejected by startup and recovery checks.
Original control pins match without historical source substitutions.

Root`/x/Temp/kindle-qbert-encoder4e5-20261101/full`,trial
`qbert-12m-48-b16-r16-encoder4e5-expert100-200k-seed4013`,service
`kindle-qbert-encoder4e5-200k-20261101`. The five-game goal remains unfinished.
