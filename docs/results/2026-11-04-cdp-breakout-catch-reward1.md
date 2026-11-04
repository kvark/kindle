# One-point pixel catch feedback passes bounded execution and cost checks

The optional catch aid completes **8,192 actions /32,725 emulator frames /
489 updates in47.580 seconds**. Independent replay verifies139 detected catches
and exactly139 added training points: raw reward189 versus training reward328.
Raw game scores,policy frames and episode boundaries stay unchanged. The frozen
check completes1,024 actions/4,096 frames in.897s with zero updates,357 invariant
tensors and no natural completed episodes. This establishes execution and early
cost,not gameplay benefit.

The fixed padded projection is **21m17.777s for200k actions**,below the30-minute
cap. The retained bounded control takes47.690s; the small timing difference does
not establish a speed improvement. Select one separately declared
[fresh200k comparison](../experiments/2026-11-04-cdp-breakout-catch-reward1-200k.md)
to measure raw frozen gameplay. Its preparation is pending;no full learner is
active. Retain70.750 sampled/143.667 greedy until that comparison passes its rule.
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
corrective labels. Actual overrides are3,733/8,192,or45.569%;guided online returns
are not unaided competence.

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

## Evidence

Root `/x/Temp/kindle-breakout-catch-reward1-20261104` contains source-segment
provenance,private runtime/diffs,CPU failures and repairs,commands,declaration,
initial/final/frozen weights,logs,replays,guards and terminal reviews. Both
services are terminal;no GPU allocation is active at closure. Check live state
before the separately selected full comparison. Do not repeat this check.

| Artifact | SHA256 |
|---|---|
| Declaration |`9fa30dc762e8ec1093ebdbc849b2748308ef325a23df9c68aeb16e17776506a0`|
| CPU review |`05c2599d6b717360d53cf0ab5670613c631c3003efa8e56bf75b98b3fb44e3aa`|
| Bounded completion |`4c53c952b66067e6c5a4ddf78e49485fe7e1e0c6175a9430bf5778876c5bae26`|
| Original terminal collector |`dde3818630bd7ff508d69e3f96641164f398640abd848fd4468b46f8f00f7e57`|
| Independent terminal summary |`7eb5d39eba7b75b3e4535f5a8e65287a5a4bc416b6256de9b04d22f12987c198`|
