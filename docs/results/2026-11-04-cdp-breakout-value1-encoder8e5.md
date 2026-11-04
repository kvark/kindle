# Encoder8e-5 with the retained Breakout critic passes its bounded check

The fresh check completes8192 actions/32,722 frames/489 updates in **48.178s**,1.02% above the
retained47.690s check. Padded200k projection **21m33.842s** passes30min.
Frozen1024 actions take.920s withzero updates/no natural episodes. This
establishes execution and early cost; no gameplay benefit is yet measured.

Only encoder4e-5→8e-5 changes from the retained single-image12M64/Auto/N8/
B16/T16/H15/R16/expert100/behavior8e-5/replay-value1 recipe. Native447fe50,
runnercf4ea238 and teacher stay fixed. Earlier8e-5 helped greedy play with
behavior4e-5/value.3, while sampled play stayed flat; its benefit may not
transfer to this combination. Retain143.667 greedy/70.750 sampled for now.

All56 current/338 retained inputs,357 finite matching initial/invariant frozen
tensors,47 pre-update events, cadence, finite aligned imitation reports,
changed CNN/reward/actor/critic, full pixel/action/label replay and guards pass.
The21-episode training replay takes5.556s. All489 reports are supervised;
117,058 repeated labeled rows equal their weight sum. Maximum raw world/actor
imitation-loss difference is8e-7. Actual3,733 overrides are45.5688% of8192;
these and guided returns are exposure, not unaided competence.

CPU qualification takes5.511s (5.670s wrapper): both actual commands,33 rejection
cases,357 control shapes and an actual collector fixture pass without a native
agent, emulator or GPU descriptor. Native metadata is loaded only. The one
historical replay source is resolved from exact retained bytes; old pins stay
unchanged. MainPID1090844/invocation12f3c96f1574401cb1f7f1a50a72124c launches
00:14:01 UTC November4/source507b773d. Main/collector complete00:15:09/00:15:13;
both original services succeed, with no stage failure or repeat. The collector
takes1.437s. Training construction takes2.467s separately; no wall-time correction.

Root `/x/Temp/kindle-breakout-value1-encoder8e5-20261104/check` is closed.
Review SHA`fc58e44990f2f3a3ea412b4c501ee0f3a2a38e93ce81829211e1636557f51868`;
completion SHA`bb3e6de3acf5859ff1ecef437a126c5af6aa4d23391c37fad6571496981587ff`.
No GPU allocation remains at bounded closure; do not restart its stages or
collector. [Bounded protocol](../experiments/2026-11-04-cdp-breakout-value1-encoder8e5.md).

After that closure, a [separate fresh200k comparison](../experiments/2026-11-04-cdp-breakout-value1-encoder8e5-200k.md)
is selected with a100k recovery and four24-episode frozen arms. Primary is
greedy initial-subtracted return per training minute, with sampled outcomes
and the original864/90%-two-wall gate also reported. No bounded experience
transfers. Reused1009/cohort cannot establish independent reliability; the
five-game goal remains unfinished.

## Fresh full comparison is running

The separate200k comparison launches00:19:16.636 UTC November4/source6543be61,
mainPID1091232/invocation`0dd5518ac0c94b879959da7818f66c36`, with120 fixed inputs.
Startup confirms357 matching control/bounded initial tensors,47 paired
pre-update semantic events and first update376. Its.375s CPU check co-hosts
with learning and remains in measured wall time. Original recovery and terminal
observers are active. No full frozen result or gameplay improvement is yet known.
Root`/x/Temp/kindle-breakout-value1-encoder8e5-20261104/full` must not restart.

Full helper qualification passes3.245s. The original queue preflight fails
before GPU work: its wrong-rate CLI fixture correctly rejects at the native
configuration boundary after constructing a second mocked guidance object,
while the copied assertion expects one. The original preflight source,
declaration, stderr and exit1 remain unchanged. Separate`preflight_repaired.py`
expects two matching mocked guidance objects; all five actual commands and
three wrong-command/four replay-input cases pass in.525s (.620s wrapper),
withzero real native agents/emulators/GPU descriptors. No successful helper
stage repeats. `preflight-repair-declaration.json` and
`preflight-repair-review.json` pin original failure and corrected CPU source;
`launch_qualified.py` verifies them before the sole launch. Original120 runtime
pins stay unchanged. This CPU fixture correction changes no production source,
command, model, cap or evaluation rule.

## The original 100k recovery passes while learning continues

The main observer saves and verifies100k actions/6227 updates at00:30:02.503 UTC
November4 in.319s. The independent verifier passes00:30:17.292 in.455s
(.514s wrapper), preserving all120 declared inputs and the original active
PID1091232/invocation0dd5518ac0c94b879959da7818f66c36. Its original observer
service is terminal success; the learner and terminal collector continue.

All357 tensors are finite:280 world/66 behavior/11 slow-value, with45,710,865/
4,938,291/853,503 elements including optimizer state. Exact metadata and file
hashes, counters, configuration and0555 directory/0444 files pass. Verification
makeszero model calls/updates; recovery omits replay, games, live beliefs and
complete RNG state. These CPU co-host times remain in measured training time.

A separate.013s read-only observation at00:31:12.137 confirms replay capacity
100k and11.159GiB GPU headroom, above the2GiB floor, while the same learner is
active. This single observation does not establish the full-run minimum.
`recovery-verification.json` SHA
`4a49d2554b5259c36b6068b13b7b1ca726aba2e824c0a85bf3849f4001980a9f` and
`post-recovery-observation.json` retain the evidence. No experiment stage
repeats and no full frozen result is yet available.
