# Smaller replay lowers Breakout scores at nearly unchanged training cost

The fresh200k replay16k comparison scores **34.000 sampled /57.625 greedy**,
below retained replay100k's **70.750 /143.667** by51.94%/59.89%. Training takes
**21m44.359s**,1.28% longer than21m27.910s. Both final arms retain **0/24 two-wall
successes**. Retain the100,000-frame replay buffer with the existing recipe;
no capacity sweep,extension,confirmation orautomatic successor isselected.
The five-game goal remains unfinished.

| Measurement | Replay100k control | Replay16,384 candidate |
| --- | ---: | ---: |
| Sampled frozen mean |70.750|34.000|
| Greedy frozen mean |143.667|57.625|
| Own initial sampled /greedy |1.417 /2.208|1.417 /2.208|
| Final sampled /greedy two-wall successes |0/24 /0/24|0/24 /0/24|
| Training seconds |1,287.910|1,304.359|
| Initial-subtracted sampled return/min |3.230038|1.498820|
| Initial-subtracted greedy return/min |6.590132|2.549144|

Initial-subtracted return per training minute falls53.60% sampled/61.32% greedy.
Both learners use200,000 actions/12,477 updates; candidate executes799,573 raw
frames versus799,595. Four candidate frozen captures total70,920 actions/
283,512 frames/61.452s with zero updates/truncations. Scores use the fixed first
three natural episodes per eight lanes,24 per arm, with guidance andlabels off.
Extra completed episodes andpartial tails remain recorded separately.

## Change and interpretation

The [fixed full protocol](../experiments/2026-11-03-cdp-breakout-replay16k-200k.md)
changes only shared replay capacity100,000→16,384 stored frames. Keep centered
CDP12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16,lambda.95,encoder/base4e-5,
behavior8e-5,dynamics4e-4,replay-value1,expert100 with unit weights/world gradients/
all-state corrective labels,sampled collection/posteriors,actor-unimix.01,
action-effects1,warmup1000 and20% episode guidance/cap1920. Native447fe50/
SHA35ad0db,byte-identical runnercf4ea238 andteacher remain unchanged. There is
no build orimplementation change andno transfer ofweights,replay,belief,games,
RNG orbounded-check experience. Seed1009/evaluation4000001009 are reused
development choices,not independent reliability evidence.

The full pre-eviction16,000-action prefix matches2,078 semantic events and2,000
vector transitions against control,excluding only unix_time andelapsed_seconds.
Final replay length is16,384. Exact source confirms cached recurrent context
andsampled refresh,but cache age remains unmeasured. Capacity changes recency,
diversity,reuse andcoverage together. This result rejects this smaller-buffer
recipe locally; it does not identify which change caused the regression or
prove cached context harmless.

Reported expert cross-entropy over the final50k actions falls.560514→.381669,
despite worse frozen play. Each loss uses its own replay distribution; this is
not acommon held-out imitation test orproof ofoverfitting. Actual teacher
overrides rise86,117/43.0585%→90,825/45.4125%. Candidate online last50 mean54.42
isguided andisnot unaided competence.

The fixed CPU coverage analysis counts only new candidate logs and reuses
completed control counts. Native actions atpre-action scores216–431 fall
1,262→878 (.631%→.439% ofalltraining); total actions there fall5,947→5,606,
while actual overrides rise4,685→4,728. Neither learner visits432+ during
training. No selected candidate final episode reaches216 (max85 sampled/167
greedy). These are descriptive score bins,not verified wall boundaries,visual
state coverage orcausal evidence that missing experience explains the result.

## Validation and closure

All **445 full pins**,100k/6,227 recovery,357 control/bounded-matching initial
and invariant frozen tensors,47 pre-update transitions,800 sampled/795 greedy
paired initial semantic events,finite supervised reports,changed CNN/reward/
actor/value weights,full guided pixel/action/label replay,frozen replays/videos,
host guards,curves,exposure andcomparison pass. The original main succeeds
**05:34:16.487 UTC November3**,postprocessing **05:34:44.611** andcollection
**05:35:19.248**. Allfive original services succeed. Source9cfd8cc; main977721/
invocation84334254aa564f5ab49a4907f2d9e950 isnever restarted.

Read-only terminal confirmation completes05:35:33.562 in3.362s (3.382s wrapper).
Coverage completes05:35:39.907 in3.355s (3.377s wrapper),with50 diagnostic/445
full pins and270,920 actions/1,083,085 frames reconciled. Recovery confirmation
already passed05:19:54.953. No full-run preparation failure,GPU stage repeat,
successful replay repeat ororiginal collector repeat occurs. Keep allCPU
co-host timings without correcting training wall time.

The earlier bounded main/collector remain failed; their separate CPU audit
repair andpartial-line status-reader failure are preserved below. Full selection
required the repaired completion andallits408 pins without rerunning anybounded
stage. That earlier failure does not relabel this original full run as repaired.

Root `/x/Temp/kindle-breakout-replay16k-20261103/full` isclosed. Recovery,terminal
andcoverage readers are complete; do not repeat them orcompleted qualification,
training,evaluations,replays orcollectors. All Kindle services areterminal at
closure; no GPU allocation isactive. Check live state before new work.

| Artifact | SHA256 |
| --- | --- |
| Full declaration |`75d022883e179433719403cc039faf86506ce535e5c35d7ff92c46c71be972fd`|
| Full learning review |`475ce1b5803a2973461e6e353ffc1927775a06de54a3b09298440c76656d3703`|
| Original full completion |`2e48b8b095e1576591f0385154292289d33e0684615f03b37dc3d62657862248`|
| Full terminal summary |`ba7fd117c96192228f5c156faaca88c7580420adc5893cba0bc679fc9f3aa0e8`|
| Fixed coverage review |`25a8eb8052142667e5c9c3589975c0e6ed220dd17b5097c5ff952f957370e7d6`|

## Historical bounded check and CPU audit repair

The fresh check completes **24,576 actions / 98,167 frames / 1,513 updates in
158.358s**, including **515 reports with replay saturated at16,384 frames**.
The conservative200k projection is **23m37.590s**, below the30-minute limit.
A1,024-action frozen check takes.903s with zero updates and no natural episodes.
Execution and early cost are established; gameplay benefit remains unmeasured.
At this bounded closure, no full learner, capacity sweep, extension or automatic promotion was selected.
The five-game goal remains unfinished.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-replay16k.md) changes
only shared replay capacity100,000→16,384 from the retained lambda0.95 control.
Native447fe50/SHA35ad0db, exact runnercf4ea238 and teacher remain unchanged;
there is no source implementation change or build. Keep12M64/Auto/N8/B16/T16/
H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled
collection/posteriors, full18/sticky.25 and the existing guidance schedule.
Seed1009 and evaluation4000001009 are reused development choices. No prior
weights, replay, belief, games, RNG or check experience transfer occurs.

| Bounded stage | Actions | Raw frames | Updates | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Fresh training | 24,576 | 98,167 | 1,513 | 158.358 |
| Frozen greedy | 1,024 | 4,096 | 0 | .903 |

All **378 original / 408 repair pins**,357 control-matching initial and
invariant frozen tensors,47 pre-update events,finite supervised reports,
changed CNN/reward/actor/value weights,full guided pixel/action/label replay
and host guards pass in the corrected audit. The pre-eviction16,000 actions
match **2,078 semantic events / 2,000 vector transitions** against the retained
control. Final replay length is16,384. Actual overrides number12,760 (51.921%);
guided returns and this short frozen capture are not competence measurements.

## Original failure and separate CPU closure

Main976085/invocationaf346c90f72b4c26bbee0fa9333738bf launches04:51:39.595 UTC
November3/source1e277a5. Both guarded GPU stages finish successfully. The new
long-prefix CPU comparison then incorrectly includes episode wall-clock time,
and the original main fails **04:54:31.441**. The original collector976088/
invocationc96a1d9b2ced4483b1ed81f4aa0b7615 fails **04:54:39.720** on that state.
Neither original service is restarted or relabeled as passing. The original
audit has no review/completion artifact. A separate partial-line status-read
error also remains recorded; it did not affect either service.

Saved-prefix diagnosis finds exactly40 differing `elapsed_seconds` fields,
first at action1,224. Every other recorded semantic field matches. The repair
excludes that wall-clock field along with `unix_time`; actions,rewards,pixel
hashes,labels,overrides and boundaries remain compared. Eight tests of the
actual prefix reader verify timing tolerance, semantic differences and the
fixed16,000-action boundary. Exact original sources and service journals remain
pinned. Native, runner, learning and evaluation are unchanged.

The corrected audit writes only under
`/x/Temp/kindle-breakout-replay16k-20261103/cpu-audit-repair`. It performs the
previously unrun guided replay in12.906s and completes **04:59:11.269 UTC** in
**17.639s**, after2.741s CPU preparation. Closure is
`complete_with_cpu_audit_repair`; original audit/collector failure remains.
No GPU stage, successful replay or original collector repeats. The original
success-only terminal reader was prepared but never run; do not run it now.

The exact native source confirms cached recurrent context and sampled refresh
from a pre-update posterior. This does not measure cache age or establish a
stale-state defect. Smaller replay changes recency,diversity,reuse and coverage
together; its effect on frozen gameplay requires a separate full comparison.
The original90%-two-wall gate and retained70.750/143.667 scores are unchanged.

Both the native-check and repair roots are closed. No Kindle allocation is
active at closure; check live state before new work. Do not repeat completed
qualification, GPU stages, replay, repair or collectors. Keep the failed
original services and all receipts intact.

| Artifact | SHA256 |
| --- | --- |
| Original declaration | `c70548faa4622c1154babe39a7c8a6fe99bb96a39d2322dcdc957a283cef3178` |
| Repair declaration | `038260a0cf0e1aa8a14c38415172453c1c81ed699d5d40d48af0f43a57f8a184` |
| Prefix diagnosis | `63e3d246a0c75c7521a7cf7f201f32f1c3192c7f7ddd1a79847f7822d64cef11` |
| Corrected review | `de05ac47f67f206aae213b4cad0b270742621cb15f77b5f1007ccbb26c4f2cdf` |
| Guided replay | `ff3aee7a70d0567e1b8463cedd8477107e32b42ee56df0c39d9d8b2c62a1a04b` |
| Corrected closure | `fa2de49f7a177449919f934d28808c294d3bb44cf147cc11ad41b2104e0808de` |
