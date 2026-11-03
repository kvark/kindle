# Frozen halfway assessment of the retained Breakout learner

Assess the already retained100,000-action/6,227-update checkpoint from the
[replay-value1 learner](../results/2026-11-02-cdp-breakout-replay-value1.md).
Its declared final200k result is70.750 sampled/143.667 greedy, both0/24
two-wall successes. The latest context9 and H3 interventions regress. Before
selecting a larger training budget, measure whether the second half of the best
retained life improved frozen play. This is a retrospective development
diagnostic, not a change to that completed experiment's selection rule.

Run exactly two new frozen arms: sampled and greedy actions from recovery100k.
Use the original native447fe50/SHA35ad0db, exact runnercf4ea238 and its runtime
at `/x/Temp/kindle-breakout-corrective-20261028/runtime`. Restore the checkpoint
without configuration overrides. Retain12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4, expert100 and critic1/1. No guidance,
corrective labels, learning, new training, prior stage repeat or native build.

Keep the original evaluation cohort4000001009, full18/sticky.25/repeat4/noops0,
native pooled RGB/one GPU64 resize and sampled posteriors. Each arm selects the
first three natural episodes per eight lanes=24, with no truncations anywhere
in the run, zero updates and357 invariant finite tensors. Preserve the original
100,000-frame episode cap,600k aggregate-action cap and300s per-arm wall cap.
Do not retry, enlarge caps, change action modes or pick another checkpoint after
seeing outcomes. The original final and actual-initial evaluations are reused
from their audited artifacts without another model execution or replay.

Report means, all lane means and paired200k-minus100k lane differences for both
modes; sampled remains primary. Compute a descriptive95% paired-lane bootstrap
interval using2,000 resamples, fixed seed20261103, indices49/1949. The unchanged
task gate is90% success at864 points. Report both endpoints and the actual
initial scores; no checkpoint promotion follows this diagnostic.

Bound halfway training time by the adjacent progress records surrounding the
100k checkpoint. They use the training timer; do not invent an exact time by
interpolating or treating checkpoint Unix time as the same clock. Report the
resulting interval for initial-subtracted score per training minute and the
known200k wall time. Preserve the original training/co-host costs. No claim of
a learning plateau, optimal budget, isolated cause or future gain follows from
two checkpoints of one reused learner/cohort. A new training allocation would
require a separate explicit selection.

Before launch, verify the original338 pins, completed reference artifacts,
100k metadata/retention and357-tensor schema. Qualify both actual CLI paths
before native/emulator construction, exact command/header constraints,
wrong-checkpoint/mode/update rejection, cohort selection and paired arithmetic.
Pin clean source, this fixed protocol, private controller, imported audit code,
runtime, checkpoints and original comparison inputs. Keep all failure records.

Run one service `kindle-breakout-halfway-20261103` with1800s total cap. Use the
existing serial host-guarded queue. After the two GPU arms, perform CPU-only
full recorded-action/outcome/frame/boundary replays with videos, frozen export
and source invariance checks, score/gate reconstruction and paired comparison.
Ordinary frozen logs have no live pixel hashes; do not claim unavailable image
equality. Verify terminal state and original invocation from the journal, then
collect once. Successful GPU/replay stages must not repeat if CPU processing
needs a separately preserved repair.

Root `/x/Temp/kindle-breakout-halfway-20261103`. The context9 allocation is
terminal and no Kindle service is running at selection; recheck before launch.
No learner, extension, sweep or automatic successor is selected. Breakout and
the five-game goal remain unfinished.
