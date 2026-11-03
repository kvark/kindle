# Optional Breakout life-loss feedback

The completed fresh 200k comparison lowers frozen scores to **42.542 sampled /
90.875 greedy**, from retained **70.750 / 143.667**: declines of **39.87% /
36.75%**. Training takes **21m39.349s**, 0.89% longer than the control. Both
final arms have **0/24 two-wall successes**. Retain the raw-reward control;
no penalty sweep, extension, confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

| Measurement | Raw-reward control | One-point life-loss aid |
| --- | ---: | ---: |
| Sampled frozen mean |70.750|42.542|
| Greedy frozen mean |143.667|90.875|
| Own initial sampled / greedy |1.417 / 2.208|1.417 / 2.208|
| Final sampled / greedy two-wall successes |0/24 / 0/24|0/24 / 0/24|
| Training seconds |1,287.910|1,299.349|
| Initial-subtracted sampled return/min |3.230038|1.899028|
| Initial-subtracted greedy return/min |6.590132|4.094359|

Initial-subtracted return per training minute falls **41.21% / 37.87%**.
Both learners use 200,000 actions / 12,477 updates; candidate raw frames total
799,584 versus control 799,595. Four candidate frozen captures total 77,592
actions / 310,203 frames / 67.331s, with zero updates or truncations. Scores
use the original first three natural episodes per eight lanes, 24 per arm;
extra episodes and partial tails remain separately recorded. Guidance, labels
and reward shaping are off in every frozen arm.

## Full comparison and interpretation

The [fixed full protocol](../experiments/2026-11-03-cdp-breakout-life-penalty-200k.md)
adds one training point penalty per net observed lost life. Raw game rewards,
scores, actual episode boundaries and policy observations remain unchanged.
Independent ALE replay verifies **1,290 lost lives and 1,290 negative training
rewards**, alongside every pixel, action and teacher label. Native447fe50/
SHA35ad0db and the teacher remain exact; qualified runner9ff2552b adds the
declared reward handling to retained runnercf4ea238. This is an explicit
training aid, not an extra actor input or a repaired native loss.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16/lambda.95,
encoder/base4e-5, behavior8e-5, dynamics4e-4, replay-value1, expert100/unit
weights/world gradients/all-state corrective labels, sampled collection and
posteriors, replay100k and the existing guidance schedule. Seed1009 and
evaluation4000001009 are reused development choices. No weights, replay,
belief, games, RNG or bounded-check experience transfer occurs. The outcome
rejects this aid for this recipe; it does not establish an isolated mechanism,
independent seed reliability or that all life-loss feedback is harmful.

The raw online last50 mean rises **67.78→94.72** while frozen scores fall.
Actual overrides rise **86,117 / 43.0585%→87,960 / 43.98%**. Final50k replay
expert cross-entropy falls slightly, **.560514→.551947**, on different visited
and replayed states. Guided online scores and lower replay loss do not establish
better unaided control.

Fixed raw-score coverage counts only candidate logs and reuses the completed
control analysis. Training actions at pre-action scores216–431 rise5,947→7,530,
but overrides rise4,685→6,233, leaving native actions **1,262→1,297**
(.631%→.6485% of all training). Neither learner reaches432+ during training.
Candidate training maximum is431; selected frozen maxima are83 sampled and253
greedy, with only one greedy episode reaching216. These are descriptive score
bins, not verified wall boundaries or causal evidence about missing experience.
Training penalties never enter the raw-score bins.

## Full validation and closure

All **453 pins**, 100k/6,227 recovery, 357 control/bounded-matching initial and
invariant frozen tensors, 47 paired pre-update events, 800 sampled/795 greedy
actual-initial semantic events, finite supervised reports, changed CNN/reward/
actor/value weights, full guided and frozen replays/videos, host guards,
curves, exposure and comparison pass. The comparison plot is reviewed.
All five original services succeed: main **06:42:06.048**, postprocessing
**06:42:24.088**, collection **06:42:57.559 UTC November3**, sourceb80194e.
Original main983163/invocation0191b9d7c0fe47acbe15c6db9845f519 is never restarted.

Recovery confirmation already passed06:27:40.709. Independent terminal
confirmation completes once **06:56:45.055**, in1.304s (1.323s wrapper).
Fixed coverage completes once **06:56:53.724**, in2.995s (3.016s wrapper),
with50 diagnostic/453 full pins and277,592 actions/1,109,787 frames reconciled.
No full-run preparation or original stage fails, and no GPU, replay, original
collector or completed reader repeats. The three earlier bounded CPU failures
and corrections below remain preserved. Keep CPU co-host timings without
correcting training wall time.

Root `/x/Temp/kindle-breakout-life-penalty-20261103/full` is closed. All original
services and recovery/terminal/coverage readers are complete; do not repeat
them. No Kindle learner or GPU allocation is active at closure. Check live
state before separately selected work.

| Full artifact | SHA256 |
| --- | --- |
| Declaration |`028a12503e0ebaef1fdd3e75889c179aa736e930ed30033f52b5ba36ec1828e9`|
| Learning review |`3b4e8c21ac253e7589fcd2341b9ab7145d10d20441a3cbe58a89c0747641c5bb`|
| Original completion |`4d2762686bfba2a73261e8d7e1e89cd00d8c356ca552200cd636fb654d3a0d9f`|
| Terminal summary |`4e848210411fba17552b2051866e681d093f936fd9ad33cb2bd3e78aedae2f75`|
| Fixed coverage review |`96930bc4d7fbd6a98142f54ad434c820322fcfbb89ab0ac3e1a8fbea03eb754e`|

## Historical bounded check

The bounded native check passes **8,192 actions / 32,730 frames / 489 updates
in 47.317s**. All **118 observed lost lives** produce negative training feedback;
independent ALE replay verifies life counts and training rewards while raw game
scores remain unchanged. The **1,024-action / 4,096-frame frozen check takes
.891s**, makes zero updates and preserves all **357 tensors**. It completes no
natural episodes, so this is not evidence of gameplay benefit.

All **408 pins**, 357 control-matching initial tensors, 47 matching pre-update
semantic events, finite learning reports, changed CNN/reward/actor/value weights,
complete pixel/action/teacher-label/life/reward replay and host guards pass.
Actual training overrides are 3,937 / 48.059%; guided returns are not unaided
competence. The conservative 200k projection is **21m10.713s**, below the fixed
30-minute training cap. Native447fe50 and the teacher remain unchanged.

Main and original collector succeed **06:06:19.889 / 06:06:24.949 UTC November 3**,
launch source `2594ab5`; main982587/invocationd62fb9aff31b447bae4b1033709deff8,
collector982590/invocationab4606f96627424e9b39906fc0804d30. Independent terminal
confirmation completes **06:07:15.464**, in 1.620s (1.638s wrapper), with both
original journal identities and all pins verified. Its first attempt wrongly
expected systemd to retain InvocationID after termination; that failed reader
is preserved, and the corrected reader verifies the journal. No training,
frozen stage, replay or original collector repeats. The earlier CPU fixture and
schema-path failures below also remain preserved.

The bounded root is closed; do not repeat its stages or readers. No Kindle
allocation was active at that closure. The separately selected fresh 200k
comparison is now complete above. Retained 70.750 sampled / 143.667 greedy
and zero two-wall successes stay unchanged.

Completion SHA `251c82e094f6cdd5874d8fa597a11ac02243f29993cbc338273154be069efee3`;
terminal summary SHA `dc938474eddcdf5c06c39124b04e6b4aa3ce99c05aca823d7bc51c5c0b90e752`.

## CPU qualification preceding the check

The separate Python overlay passes CPU qualification. It subtracts one training
point per observed net lost life while preserving raw rewards, game scores,
actual termination/truncation flags and policy observations. Frozen evaluation
refuses shaping. The aid defaults off; its gameplay benefit is unmeasured.
The [bounded protocol](../experiments/2026-11-03-cdp-breakout-life-penalty.md)
selects one fresh 8,192-action check after qualification, not a full learner.

The runner derives from retained `cf4ea238`; only the declared reward handling
changes. Native `447fe50` / SHA `35ad0db` is copied byte-for-byte, with no build
or native implementation change. The retained runtime, teacher and all 338
control pins remain unchanged. The new runtime is
`/x/Temp/kindle-breakout-life-penalty-20261103/runtime`; its source and packaging
receipt are retained beside it. Do not edit the retained control runtime.

All **57 CPU tests** pass at **06:00:51.313 UTC November 3**, source `c6341db`,
in 2.102s including pin checks (2.145s wrapper). They exercise the actual runner
with synthetic environments and an agent stub, including separate raw/training
reward accounting, loss/gain/multiple-loss/game-over/truncation/reset cases,
F32 conversion, malformed input rejection, actual CLI refusals, and exact
default-off semantic agreement with the retained runner. The changed auditor
rejects corrupt reward contracts and stored rewards. The independent replay
reader queries environment life counts and separately computes penalties;
its synthetic tests reject altered initial/transition/reset counts and rewards.
All 365 qualification pins, including the 338 control pins, remain unchanged.
No native module, actual emulator or GPU is used in these tests.

Bounded preflight passes at **06:03:31.852 UTC**, in 1.972s (2.005s wrapper):
both actual commands, nine native-config assertions, five pre-update comparison
cases, expected 376-action first update / 489 updates / .5 residual credit, and
the 357-tensor initial schema. Pre-update comparison preserves raw environment,
action, teacher and intrinsic-reward fields while normalizing only the declared
extrinsic training change. The full auditor checks that training change.

Two CPU failures remain preserved. The first synthetic fixture retained expert
imitation without declaring guidance; the ordinary auditor correctly rejected
it. The corrected fixture disables imitation in that synthetic case. The first
bounded preflight omitted `.safetensors` from a schema-component filename; only
that reader was repaired. Production reward, runner, auditor and replay sources
are identical before and after both repairs. No GPU stage has failed or repeated.

At that CPU qualification milestone, no native check or full learner had run. The
next bounded check keeps 12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/
behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled collection and replay100k.
Require real ALE life/reward replay, finite training, unchanged frozen weights
and the fixed cost gate before any full-run selection. Retained frozen scores
70.750 sampled / 143.667 greedy and zero two-wall successes remain unchanged.

Evidence root `/x/Temp/kindle-breakout-life-penalty-20261103`:

| Artifact | SHA256 |
| --- | --- |
| CPU review | `b97fdbb00091ed394fc06493809616c50058c338beafd5178be96450d02a3a90` |
| Runtime receipt | `cc257a731477f337fbada8f4e6bc380e29efc26b9cf25e6b053aa3e83784d662` |
| Bounded preflight | `ce0818216b92f76d97883f6413014b98974a14642c21ce18fbeaa60eddd1873c` |
| Runner | `9ff2552b1c59fe7e45db156d9bba571737d3a3eb34d2ac073136f2bf92ce244a` |
| Reward helper | `e61a067c8ca6d9df400788a2ae25ec5d2bac69eadc5199f31186d98e2dc5b76a` |
| Vector auditor | `47a3a4ee206eba485ae8e6999b50204615fe9a45ee2bb6f0df93eac45cee0d20` |
| Independent replay reader | `cac4c9a91f34c23791d343ef67fc8acada0df879b08fbef52668555cd2677ae0` |
