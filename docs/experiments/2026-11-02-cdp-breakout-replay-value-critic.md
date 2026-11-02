# Breakout: critic calibration after the replay-value gain

Select one frozen diagnostic within the existing five-game goal after the
H30/R12 learner and its separate CPU coverage repair close. No Kindle service
is active at selection review. Qualify the adapted CPU controller before
launch, then require the same idle-host check immediately before GPU work.

Replay-value0.3→1 improves the retained behavior8e-5 recipe from59.000/117.292
to70.750/143.667 sampled/greedy, at nearly unchanged cost. The earlier critic
diagnostic used behavior4e-5/value0.3 checkpoints, so it does not measure the
critics behind this gain. Existing frozen transition logs contain no value or
intrinsic-bonus fields. This capture measures those missing quantities; it
does not repeat old evaluations merely to obtain another score.

Keep both completed models unchanged:12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4/expert100, sampled training collection,
native447fe50/SHA35ad0db, original teacher and checkpoint configuration.
The0.3 model retains exact runner6c1b69fb; the1.0 model retains runnercf4ea238.
Runtime `/x/Temp/kindle-breakout-corrective-20261028/runtime`. No native build,
learning, objective, policy, teacher or configuration change is selected.

Use the qualified `--probe-behavior` path: after each original act, read the
live posterior's value and action-conditioned intrinsic bonus before the
environment transition, without another posterior sample. First run1,024
greedy actions on the0.3 model, cap120s; require128 original semantic events,
357 invariant tensors, finite values/nonnegative bonuses, zero updates and
passing host guard. No retry. Then capture these four fixed trained-policy arms:

| Replay-value weight | Mode | Original actions | Retained mean |
| --- | --- | ---: | ---: |
| 0.3 | Sampled | 31,848 | 59.000000 |
| 0.3 | Greedy | 40,560 | 117.291667 |
| 1.0 | Sampled | 35,552 | 70.750000 |
| 1.0 | Greedy | 43,424 | 143.666667 |

Keep learner1009/evaluation4000001009, original lane offsets, sampled categorical
posteriors, full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU64
resize. Guidance and labels are off. Each arm uses the original first three
natural episodes per eight lanes, 24 total, with600k-action/300s caps and zero
updates/truncations. No initial-policy capture is needed for this trained-critic
calibration. Original training times, scores and failed two-wall gates stay fixed.

Require exact equality with every original transition/episode/reset, original
checkpoint counters/config/native/runner identity,357 invariant tensors per
arm, finite values/bonuses and host guards. Reuse original source replays only
after their log hashes and complete semantic trajectories match. Do not repeat
an ordinary evaluation or emulator replay as a validation stage. Ordinary
source logs lack live pixel hashes; do not claim unavailable image equality.

Reuse the qualified return arithmetic unchanged: F32 discount1−1/333, actual
arrival rewards plus the frozen intrinsic bonus, complete natural episodes,
zero terminal bootstrap. First episode per lane fits a constant; the next two
per lane supply16 test episodes. Exclude excess episodes and partial tails.
Report critic bias, MAE, RMSE, correlation and R², alongside earlier-episode
constant/zero baselines, episode-start values and bonus magnitudes. Only the
overall metric group applies; no Qbert phase labels or inferred first-wall
boundaries. Retain final50k training reward/bonus/entropy/advantage summaries
as descriptive context, not gradient attribution.

Sampled arms are primary for calibration. Greedy changes the target policy.
The two models visit different states despite shared seeds, so differences
are not matched-state causal effects of the loss weight. Realized returns are
noisy samples of expected value. This diagnostic cannot establish whether
critic changes caused the gameplay gain or justify a new loss weight by itself.

CPU preparation checks the actual five CLI paths, both357-tensor checkpoint
schemas, all282/338 source inputs, exact runner/native/wrapper bytes and the
terminal H30/R12 evidence. Native/emulator construction is forbidden. Reuse the
nine passing value-target/ledger tests only after checking unchanged test and
helper sources; do not rerun unrelated qualification. Preserve any failure.

Root `/x/Temp/kindle-breakout-replay-value-critic-20261102`; main
`kindle-breakout-replay-value-critic-20261102` and original collector
`kindle-breakout-replay-value-critic-collection-20261102`, caps2400/2700s.
After successful original completion, run the read-only terminal reader once.
No fitting, new learner, extension, cap increase, coefficient sweep or automatic
successor is selected. The five-game goal remains unfinished.

## CPU preparation

Preparation passes on November2/sourcefe139dd in5.834s (5.879s wrapper): all
five actual CLI paths, both357-tensor schemas, all282/338 source inputs and422
preceding H30/R12 inputs. Both completed checkpoints retain200k actions/12,477
updates. Their configurations differ only in replay-value0.3→1. Exact runner,
native and wrapper identities match; no native or emulator is constructed.

The nine existing value-target/ledger tests are retained without repetition:
the test source matches revision1b6bed9 and the arithmetic helper matches the
original successful diagnostic's pin. Source adaptation completes in.063s,
with no preparation failure, build or repeated stage. The two models' final50k
summaries each include3,125 reports; these descriptive means remain separate
from the forthcoming complete-return calibration.
