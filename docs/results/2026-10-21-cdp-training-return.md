# CDP training resumes faster, but the first Freeway policy fails

Fresh Freeway2017 completes500k actions in **28m14s**, but its final sampled
policy scores **0 in all24 natural frozen episodes**, equal to its actual
untrained control. This is a failed learning result, not a GPU failure or a
five-game success. Backend optimization remains parked.

## Evidence

- Eight batched environments,12M/B32/T16/BF16, ratio32, centered CDP and
  action-effects coefficient1; full18 actions, sticky.25, repeat4, native
  frames with one GPU64 resize. No action hints or extrinsic reward shaping.
- 500,000 actions,1,999,960 emulator frames,31,211 updates; training debt.5.
  Whole-run throughput295.08 actions/s,19.67× aggregate real time,2.46× per
  stream. Mean reported update45.707ms; learning takes1,430.24 of1,694.43s.
  GPU utilization is not measured.
- Only **one training crossing**, at aggregate action43,664, across240
  completed rounds. The reward enters45 sampled learner updates, last seen
  at141,976. This contradicts a completely disconnected reward pipeline;
  it does not prove correct learning or adequate exploration.
- Final online last50 rounds average0. Final policy and actual-initial
  control each complete24 natural evaluation rounds with zero updates,
  zero score and zero25-crossing task successes.
- All three GPU jobs pass their host guards, with zero new warnings. Full
  CPU action replay checks every reward/boundary/reset/frame count and
  produces whole stream0 videos. Checkpoint audits find346 finite tensors
  per model. Full frozen runs do not export a second tensor copy; exact
  before/after tensor equality is established only for the live smoke.

The historical same-seed12M/B8 run took3h54m12s and scored17.25 frozen,
with120 positive training rounds. Current wall time is8.29× lower, but
learning is worse. Batch/microbatch8→32, historical default precision→BF16
and backend changes are confounded; this is **not** an isolated optimization
benefit. B32 performs about one quarter as many optimizer steps at the same
replay-position ratio. Its intrinsic reward remains nonzero; finite losses
and lower cosine loss do not establish a useful policy representation.

## Follow-up learning test (now complete)

**Update:** [B8 scores30.58 with24/24 crossing successes](2026-10-21-cdp-freeway-b8.md)
in37m36s. The failed B32 result below is retained unchanged.

A fresh500k-action **B8 seed2017** run changes only batch/microbatch size
relative to the completed current-backend B32 recipe. Same native binary,
BF16, rates, rewards, environment seeds and budgets. All346 actual initial
tensors match B32 exactly. Expected124,939 updates, followed by final and
actual-initial24-episode controls and full videos. This tests the batch change
as a package, including optimizer/EMA/warmup cadence; it does not isolate
gradient variance or prove three-seed reliability. No budget extension or
new backend work follows automatically.

## Artifacts and retained failures

- [Completed B32 review](../../runs/cdp-training-return-20261021.LaTWMxXh/learning-review.json)
- [Final-policy whole video](../../runs/cdp-training-return-20261021.LaTWMxXh/freeway-2017-final.mp4)
- [Actual-initial whole video](../../runs/cdp-training-return-20261021.LaTWMxXh/freeway-2017-initial.mp4)
- [B8 declaration](../../runs/cdp-freeway-b8-20261021.JEzySupH/declaration.json)

The first build found an unintended outer workspace; the separate package
workspace fixes it. The smoke's first audit compared serialized files rather
than tensor values; all346 values match exactly under the corrected CPU
comparison, without another GPU run. The full-run controller's old
extrinsic-only Pong/Boxing validator rejects the CDP recipe after all three
jobs finish. Its failure remains; the separately prepared CDP-compatible
review passes at16:37:21UTC without repeating training/evaluation.

The completed runner's rebuildable Cargo cache was removed; sources, lock,
native package, checkpoints and all evidence remain. No historical queue or
checkpoint lifetime was resumed. Five-game supremacy remains unachieved.
