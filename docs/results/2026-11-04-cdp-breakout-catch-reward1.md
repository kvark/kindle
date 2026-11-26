# One-point pixel catch reward lowers primary Breakout performance

The fresh200k comparison completes200,000 actions/799,557 emulator frames/
12,477 updates in **21m51.733s**,1.85% more training time than the retained
control. One added training point per causal pixel-detected catch lowers the
primary greedy raw score **143.667→86.417**. Sampled score also falls slightly.
All frozen arms have zero of24 two-wall successes. The fixed preference rule
fails;retain the unshaped recipe,with no extension or coefficient sweep.

| Measure | Retained control | Catch reward +1 |
|---|---:|---:|
| Training time |21m27.910s|21m51.733s|
| Greedy raw return,24 episodes |143.667|86.417|
| Sampled raw return,24 episodes |70.750|68.458|
| Actual-initial greedy /sampled |2.208 /1.417|2.208 /1.417|
| Greedy initial-subtracted return/min |6.590132|3.851776|
| Sampled initial-subtracted return/min |3.230038|3.066555|
| Two-wall successes,each final mode |0/24|0/24|

Greedy return falls39.85%,and greedy gain/min falls41.55%. The paired eight-lane
bootstrap difference in gain/min is−2.738356,with95% interval
[−4.825834,−.559765]. The raw greedy difference is−57.250,interval
[−102.291667,−10.041667]. Sampled difference−2.291667 has interval
[−27.417708,26.750000],crossing zero. These are10,000 seed0 resamples of one
reused development cohort,not independent training-seed reliability estimates.
The original mean864/90%-two-wall gate and five-game goal remain unfinished.

All original services and terminal collection succeed;no GPU allocation remains
active at review. All required execution,checkpoint,replay,frozen,comparison and
source checks pass. The signal can be delivered correctly without improving
control. This result does not establish why learning worsens or rule out every
form of credit-assignment aid. No automatic successor is selected.

## Bounded execution and cost check

The optional catch aid completes **8,192 actions /32,725 emulator frames /
489 updates in47.580 seconds**. Independent replay verifies139 detected catches
and exactly139 added training points: raw reward189 versus training reward328.
Raw game scores,policy frames and episode boundaries stay unchanged. The frozen
check completes1,024 actions/4,096 frames in.897s with zero updates,357 invariant
tensors and no natural completed episodes. This establishes execution and early
cost,not gameplay benefit.

The fixed padded projection is **21m17.777s for200k actions**,below the30-minute
cap. The retained bounded control takes47.690s; the small timing difference does
not establish a speed improvement. A separately declared
[fresh200k comparison](../experiments/2026-11-04-cdp-breakout-catch-reward1-200k.md)
measures raw frozen gameplay and fails its preference rule,as reported above.
Retain70.750 sampled/143.667 greedy.
The original864/90%-two-wall gate and the five-game goal remain unfinished.

## Intervention

The [bounded protocol](../experiments/2026-11-04-cdp-breakout-catch-reward1-check.md)
adds exactly+1 to the extrinsic arrival reward when the fixed causal pooled-pixel
detector emits a catch. The detector and pixel-extraction source segments are
identical to the preceding
[signal](2026-11-04-cdp-breakout-catch-signal.md) and
[raw-motion validation](2026-11-04-cdp-breakout-catch-validation.md).
No lives,RAM,future frames,teacher action or raw-motion witness enters detection.
The arrival after actionj credits actionj. Reset frames initialize per-lane
detectors;episode resets clear their state.

The private qualified runner exposes `--breakout-catch-reward`,default off,
for fresh native-RGB Breakout training only. Evaluation,restore,other games and
unsupported image geometry reject it. Each transition records pooled observations,
emitted events,raw game rewards and shaped training rewards;the native stored
extrinsic reward must equal the shaped value. The replay audit recomputes
observations from actual pixels and events with the preserved reference detector,
then independently verifies reward addition and reset alignment. Frozen runs
omit the aid,guidance and labels entirely.

Native447fe50/SHA35ad0db,teacher and all native settings remain unchanged:
CDP12M64/Auto/N8/B16/T16/H15/R16/context1,encoder/base4e-5,dynamics4e-4,
behavior8e-5,replay-value1,representation.1,expert100/unit weights/world gradients,
sampled collection/posteriors,full18/sticky.25/repeat4/noops0,native pooled frames
and one GPU64 resize. Episode guidance remains20% with cap1920 and all-state
corrective labels. Bounded-check overrides are3,733/8,192,or45.569%;guided
online returns are not unaided competence.

The overlay changes only the Python reward adapter,runner and vector audit;
the native binary is copied unchanged,with no build. Runner SHA256 is
`25327009ea08befee646534a62e7e47e65f1b8dfacbca658fc204264a699e8aa`.
The qualified package remains under the evidence root's `runtime/`;do not mix it
with other optional overlays or treat this private experiment flag as a new
canonical CLI default.

## Qualification and bounded validation

All18 adapter fixtures pass. The saved-observation qualification verifies43,392
available arrivals and1,362 catches across32 episode segments,including all1,132
selected-cohort catches. The32 final segment arrivals have no following saved
pre-action observation and are explicitly excluded from that saved-data check;
reset/terminal fixtures and the subsequent complete training replay cover the
integration boundaries. Default-off accounting exactly matches the retained
200,000-action log. Both actual queued commands reach mocked construction
boundaries with the intended configuration,and both replay entry points pass.

All405 current/338 retained source pins,357 control-matching initial tensors,
47 shared pre-update action/raw-outcome events,489 finite weighted-loss reports,
changed CNN/reward/continuation/actor/value weights,full pixel/action/teacher-label/
catch/reward replay,complete frozen outcome replay and host guards pass. Three
catch events occur before the first update;their declared training-reward fields
and stored extrinsic components differ,while shared raw behavior agrees. Frozen
replay has no original live pixel hashes;the training replay verifies the logged
guidance pixel hashes. No broader frozen pixel-equality claim follows.

The independent terminal review recounts every training reward,confirms native
stored extrinsics,checks source/weight hashes and verifies both original process
identities and terminal success. Main PID1163956/invocation
`3a127082c06e4cd89673c69fc031bfeb` and collectorPID1163959/invocation
`653f3607bdfb4de59a9752acb473629f` succeed16:35:56.085/16:36:07.631 UTC
November4/source`69b14f45`. Separate3.713s review closes16:36:31.546 with47
artifacts. No GPU,training,replay or collector stage fails or repeats.

Two earlier CPU preparation failures remain preserved. The first test invocation
imports the canonical runner ahead of the private runner and fails before any
fixture runs. Repairing test import order leaves production unchanged. A later
preflight looks for `guidance_config` in an older bounded report after successfully
checking both queue commands,both replay entry points and489 weighted reports.
The corrected lookup uses `training.header.exploration`;a separate suffix repair
completes only unfinished checks. Original sources and failure receipts remain.

Training includes all actual host work without time correction. Its wrapper
breakdown is40.630s learning,2.482s acting,2.537s environments,1.109s observing,
.092s resets and.266s checkpointing;construction is separately2.390s. These
timers do not isolate GPU kernels. The short check cannot establish long-run
cost or competence,and a positive catch reward could favor rallies without
better brick clearance. The full comparison keeps original raw-score gates.

## Full 200k comparison

The selected fresh learner starts at16:51:43 UTC November4/source318591c9.
Main PID1164587/invocation`873fae9edd444c15be0db8fcffa5812c` finishes successfully
at17:17:51.351 UTC. CollectorPID1164590/invocation`b1ac122f67744881a163af11b53956ff`
finishes17:18:02.038 after a3.661s terminal audit. Recovery observerPID1164593/
invocation`6e074f4b87304631833165b2ddc2f8df` succeeds earlier. All three original
services are terminal;no stage restarts or repeats. The evidence root is the
bounded root's `full/` directory;service`kindle-breakout-catch-reward1-200k-20261104`.

CPU qualification passes full audit, exposure, comparison and terminal-state
helpers in5.582s. Additional catch accounting checks reject seven corrupted
catch/header/native-reward cases and independently recount all8,192 bounded
actions:139 catches,raw189/shaped328. The actual plot environment reads the
bounded raw-score curve successfully. All five exact queued commands reach
mocked construction with the intended configuration; no native agent, emulator
or GPU descriptor is created during preflight. Three altered training commands,
four replay-path cases and11 comparison-header cases are rejected. Queue
preflight takes2.301s,with no full-preparation failure or repeated GPU stage.

Startup verifies all357 initial tensors against both controls and both47-event
pre-update comparisons,permitting only the declared catch fields and extrinsic
reward differences against the unshaped control. First update is376. A live
publication reader at16:52:28 UTC verifies the original PID/invocation and414
finite reports through7,000 actions. Startup observer .310s and publication
reader .022s are CPU co-host work; no wall-time correction is applied.

The100k/6,227 checkpoint passes the runner's .329s check at17:02:40.534 UTC.
Independent verification finishes17:02:44.724 in1.379s: all357 tensors are finite,
all473 declaration pins are unchanged, and checkpoint files/directories are
read-only. The recovery service exits successfully under its original invocation;
main PID1164587/invocation873fae9edd444c15be0db8fcffa5812c continues unchanged
at that check.
This retains weights and optimizer moments,not replay,games,beliefs or full RNG.

A supplemental publication reader initially asserts that recovery parentPID1164593
must write journal output. The parent waits silently and launches verifier child
PID1165576,which writes the report. The failed assertion is preserved in
`publication-recovery-reader-error.json`; a separate .015s reader verifies the
child command,original service invocation,exact report,successful terminal state
and still-active original learner. Original observer/verifier work does not
repeat or change. All co-host costs remain in actual training time;the first
ad hoc reader's duration is unmeasured. Recovery review SHA256:
`f84109f604f3853656fabffda8ad8fc250df22b3d5f9246c0c3ada9f0ef69e39`.

Training completes200k actions/12,477 finite reports within its1,800s cap.
All357 initial tensors match both controls;CNN,reward,continuation,actor and value
parameters change. The final checkpoints remain invariant across each frozen
arm,with zero updates/truncations. Actual-initial sampled/greedy traces reproduce
799/794 retained semantic events,allowing only episode elapsed-time differences.
Full pixel/action/label/catch/reward replay takes115.314s and verifies5,231 events:
raw reward13,151 versus native training reward18,382. The four frozen arms total
86,288 actions/344,967 frames/55.011s,each with24 selected natural episodes.
Full outcome replays/videos,guards,raw-score curves,coverage,exposure and paired
comparison pass. Frozen logs do not contain original live pixel hashes;no frozen
pixel-equality claim is made.

Catch counts by50k-action window are1,234/1,272/1,336/1,389. Actual training
overrides are85,864/200,000,or42.932%,versus control43.0585%. Late reported expert
loss is.561892 versus control.559936;this is repeated replay fit,not held-out
accuracy or an identified cause. There are974 native actions in the216–431
pre-action score band and none above it;maximum raw trajectory score is418.
These exposure counts do not establish wall clearance or unaided competence.

Training construction is separately2.516s. Wrapper timers assign1,151.679s to
learning,62.865s acting,62.963s environments,27.534s observing,1.176s resets and
.551s checkpointing. These do not isolate GPU kernels. All CPU co-host work
remains in actual training time without correction.

The original473 declaration pins and338 retained pins pass terminal verification.
A later publication reader looks for `behavior_metrics` in the older retained
learning review and fails after printing the candidate metrics and score changes.
Its error is preserved in`publication-retained-reader-error.json`;the unfinished
lookup is resolved from the retained `expert-exposure-review.json`,and remaining
publication hashes are collected in.000536s. No original audit,learner,replay or
assessment repeats. This read occurs after training and affects no timing result.

The locked greedy score/gain-per-minute/paired-interval rule rejects this candidate,
and the original864/90%-two-wall gate still fails. No bounded experience transfer,
extension,coefficient sweep,second learner or automatic successor occurs. The GPU
is idle at terminal review;check live state before any future allocation.

Full declaration SHA256:
`abc92ca53cea3f8de41e09f5a52cdc4938fac38c21cbf7d4f4bb5658b504f1ee`.
Queue preflight SHA256:
`c77fcfb9b4ca6ee1b83b14bcd5abb533c3074212684a56e9177eed07f603cd16`.
Startup,launch,qualification and observer receipts remain under `full/`.

## Evidence

Root `/x/Temp/kindle-breakout-catch-reward1-20261104` contains source-segment
provenance,private runtime/diffs,CPU failures and repairs,commands,declaration,
initial/final/frozen weights,logs,replays,guards and terminal reviews. Both
bounded services are terminal;no GPU allocation was active at bounded closure.
The separate full comparison above is also closed. Do not repeat either run.

| Artifact | SHA256 |
|---|---|
| Declaration |`9fa30dc762e8ec1093ebdbc849b2748308ef325a23df9c68aeb16e17776506a0`|
| CPU review |`05c2599d6b717360d53cf0ab5670613c631c3003efa8e56bf75b98b3fb44e3aa`|
| Bounded completion |`4c53c952b66067e6c5a4ddf78e49485fe7e1e0c6175a9430bf5778876c5bae26`|
| Original terminal collector |`dde3818630bd7ff508d69e3f96641164f398640abd848fd4468b46f8f00f7e57`|
| Independent terminal summary |`7eb5d39eba7b75b3e4535f5a8e65287a5a4bc416b6256de9b04d22f12987c198`|

Full-run completion artifacts:

| Artifact | SHA256 |
|---|---|
| Learning review |`95a1d8ae662ef5df38b147ded8737ac8eff6aaac26e43868b84c30798e0305d0`|
| Paired comparison |`af8457d08a28ea8d2bec64ac186edb13daf7b4c271463b29eb874b23d3d6864e`|
| Main completion |`def54d997c3eb1cd51249895ae18491bcd562ea0dbba5fe1da869242bd6f06e6`|
| Original terminal collector |`25210ee428d6eb7790943f07adf729d27d422f8b547f374a25546e68ae1606d7`|
| Full training replay |`0c94e531045a0bd42abd4ef402ee99fde32eb80f44e5170c5abfff0ab920a492`|
