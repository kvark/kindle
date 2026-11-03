# Fresh400k Breakout comparison at the retained replay ratio

Select one fresh400,000-action learner after the [halfway assessment](../results/2026-11-03-cdp-breakout-halfway.md)
finds22.125→70.750 sampled and33.250→143.667 greedy from100k to200k, with all
eight lane means improving in both modes. Initial-subtracted score per minute
also rises. This motivates testing more learning work at the same recipe; it
does not predict a gain beyond200k or establish a learning plateau/slope there.
The earlier400k/R8 comparison held update count roughly fixed and regressed in
greedy play on an older recipe. This comparison keepsR16 and doubles both
experience and updates. Existing warm-start failures remain retained.

Use the exact native447fe50/SHA35ad0db, runnercf4ea238 and teacher from the
retained200k replay-value1 control. No native or runner change, build or repeated
bounded training is needed. Reuse the already qualified matching8,192-action
check and full200k execution evidence. Twice the measured200k training time
with10% margin projects47m13.403s. Select a **50-minute training cap**,400,000
actions/24,977 updates, with no cap increase or outcome-conditioned extension.
This is a larger budget point; it is not a claim of faster wall-clock training.

Keep centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,behavior8e-5,dynamics4e-4,policy1,imagined/replay critic1/1,
lambda.95/horizon333,constant actor/world expert100,unit action weights,
all-state corrective labels,actor/critic representation gradients off,
sampled collection/posteriors,unimix.01/entropy.0003,action-effects1,
replay100k,warmup1000,20% episode guidance/cap1920,seed1009/eval4000001009,
full18/sticky.25/repeat4/noops0,native pooled RGB/one GPU64 resize/raw rewards.
H3,centroid,crop and imitation decay stay off. No prior weights, replay, belief,
RNG, games or experience transfer. Do not restart or resume a completed service.

Retain actual initial and100k/6,227,200k/12,477,300k/18,727 checkpoints. Preserve
all four files and357 finite tensors at each milestone. First update376 and
residual credit.5 remain. Compare all357 initial tensors against the retained
control and verify common pre-update semantics. At200k, compare all357 tensors
and the entire common training semantic prefix against the completed control,
excluding only timing fields and declared total-budget/output identities.
This tests the expected common prefix without claiming full replay/RNG state
equivalence. Preserve any mismatch and stop for diagnosis; do not silently relax
the assertion or repeat a successful GPU stage.

After the fixed endpoint, freeze final and actual initial, sampled and greedy,
first three natural episodes per eight lanes=24 per arm. Guidance/labels off,
zero updates/truncations,357 invariant tensors. Keep300s/600k actions per arm
and the100,000-frame episode cap. Sampled remains primary. Compare both modes,
actual-initial gain and gain per training minute against retained70.750/143.667
in21m27.910s. The original90%-two-wall864 gate is unchanged. No intermediate
frozen checkpoint selection or deployment promotion is allowed.

Require finite supervised reports,unchanged configuration/runtime/teacher,
changed CNN/reward/actor/value,full training pixel/action/label replay,frozen
action/outcome/frame/boundary replays/videos,host guards,curves,exposure and
comparison. Describe training score-range coverage without calling score
fractions verified wall boundaries. Guided returns and replay CE are not
unaided competence. Keep CPU co-host costs without time correction.

Before launch, qualify actual train/frozen commands,400k/24,977 accounting,
all three recovery milestones,retention,common-prefix checks,observer wiring,
comparison and terminal collection. Pin clean source,exact scripts/runtime,
inputs and this private fixed protocol. The full controller must preserve
original identities and stop on failures. Repair CPU processing separately;
never repeat completed GPU stages because an observer or audit failed.

Root `/x/Temp/kindle-breakout-budget400k-20261103/full`, intended service
`kindle-breakout-budget400k-20261103`. The original learner is active; the CPU launch-wrapper failure and separate
serialization-identity repair below remain recorded. Check live state before work; no overlapping GPU jobs.
No extension,coefficient sweep,fresh-seed confirmation or automatic successor
is selected. This reused seed/cohort cannot establish independent reliability.
Breakout and the five-game goal remain unfinished.


## CPU qualification and fixed service wiring

The first CPU preflight passes62 cases and all five actual CLI paths in4.757s
(4.814s wrapper). No native agent or emulator is constructed. It validates the
400k accounting, three retained checkpoints, common-prefix failure cases,
score coverage, frozen scorer, journal identities, transient poll retry and
plot-environment imports. The retained200k semantic reader covers38,050 events,
including25,000 transitions and12,477 learner reports;357 initial tensors are
finite. All338 control/413 halfway pins pass. Sources, original generation
receipt and the fixed terminal reader are retained under the private root.
No failed preparation or GPU work occurs in this qualification.

The main service runs training first, then requires all recovery checks before
starting the four frozen arms. A mismatch stops that original learner for
diagnosis; it cannot start a replacement. Recovery observation expires at3,900s
without changing the learner. Transient polling errors are retried. Service
limits are7,800s main,4,000s recovery,8,100s progress/postprocessing and8,400s
collection; training itself remains capped at3,000s. Original process identities
are recorded at launch. Startup checks all initial tensors,47 common pre-update
transitions, first six update counters and finite reports. Final collection and
the separately fixed terminal reader validate outputs and original journals;
CPU processing failure never authorizes repeating a completed GPU stage.


## Original launch and CPU identity repair

The original learner launches14:31:26.282 UTC November3/source4bb0d101, with446
pins,PID1034231/invocation13c5fb9a8be543deb2c279522c8498a2. Its launch wrapper
fails14:31:31 on raw initial-checkpoint file-hash equality after starting the
main,progress and postprocessing services. The learner keeps running. Inspection
finds all357 initial tensors equal, every safetensors payload byte equal and
parsed headers equal; only JSON header ordering changes serialized file hashes.
Metadata differs only in those file hashes. This is a CPU comparison defect,
not a measured learning mismatch. Preserve the original failure/log/receipt.

A separate CPU repair qualifies six cases and the38,050-event reference reader
in4.257s (4.478s wrapper). It verifies each recorded raw hash against its actual
retained file, then compares sorted-header plus exact payload identities and
canonical metadata. No payload,shape,value,configuration or semantic field is
ignored. All357 tensors remain exactly compared at initial/100k/200k. This
explicit amendment replaces serialized-byte identity with content identity;
the original private protocol and scripts stay unchanged. A content or training
semantic mismatch still stops the original learner, without a GPU retry.

The recovery and collection observers start once14:34:38 UTC; original main,
progress and postprocessing identities are preserved. Startup passes14:34:41
through30,008 actions/1,853 finite reports,357 matching/finite initial tensors,
47 common pre-update transitions,first update376 and residual credit.5. Recovery,
postprocessing,progress and collection are active. No completed GPU stage repeats
and native/runner/training commands are unchanged. CPU co-host time remains
uncorrected. No final frozen result or successor is available.

Repair evidence: `/x/Temp/kindle-breakout-budget400k-20261103/cpu-repair`.
At terminal closure, verify its fixed script/qualification and preserved original
launch failure alongside the original collector and independent result reader.


## First retained checkpoint

The original recovery observer retains100,000 actions/6,227 updates, all four
files/206,067,162 bytes and357 finite tensors. A separate read of those existing
artifacts passes14:42:59.826 UTC in1.385s, checking446 original pins,11 repair
pins and all five original service identities. Retained metadataSHA60c3210e,
reviewSHA48dde4b6,retentionSHA1b0b634a. No retention or GPU stage repeats and no
frozen result is produced. The full content/semantic comparison stays scheduled
at200k;100k is not independently promoted. CPU co-host time is uncorrected.
Observation: `/x/Temp/kindle-breakout-budget400k-20261103/recovery100k-observation.json`.


## Common 200k prefix passes

The original recovery observer retains200,000 actions/12,477 updates and passes
the common-prefix comparison at14:53:46.448 UTC. All357 tensors match at initial,
100k and200k, including optimizer state. Raw-file integrity and canonical header/
payload identities pass. All38,050 common events match:25,000 transitions,12,477
learner reports,263 episodes,259 resets,48 progress records,two checkpoints and
one header. The declared timing,budget/path and serialization normalization is
unchanged. This verifies the recorded prefix and checkpoint contents,not full
live replay/RNG/belief state equality. No stop or repeated GPU stage is needed.

The fixed read-only observation passes once14:54:06.583 in1.602s (1.630s wrapper),
checking both recoveries,446 original/11 repair pins and all five original service
identities. Prefix receiptSHA`c346f20bbaf519d48549b6e62b06f0bad629d1b11b9d331e31202cf7371a20af`;
normalized semanticSHA`225cbe82ad4b403ca15a730506462c8e81a6ab9044dae6590f49df8767335b24`.
The observer's full comparison,retention and GPU stages are not repeated.
Evidence: `/x/Temp/kindle-breakout-budget400k-20261103/recovery200k-observation.json`.
The learner continues toward300k recovery and the fixed400k endpoint. No frozen
result or successor is available; CPU co-host costs remain uncorrected.


## All recovery checkpoints pass

The original recovery observer retains300,000 actions/18,727 updates and exits
successfully15:04:28.160 UTC. All100k/6,227,200k/12,477 and300k/18,727 retained
checkpoints pass their four-file integrity and357-finite-tensor requirements;
the200k common-prefix receipt remains unchanged. The terminal service has been
unloaded, so its original invocation is verified from its journal, not an empty
live InvocationID field. No service restart occurs.

A single read-only final recovery review passes15:05:24.225 in2.999s (3.018s
wrapper),checking all three checkpoint metadata/configurations and files,
446 original/11 repair pins,original recovery identity and unchanged active
main/progress/postprocessing/collection identities. Recovery completionSHA
`27dd453c407d7e85ce05459a7f880bacf555d9ab970ef9596c75603bf2f006e6`;
300k reviewSHA`f50e87e64907e0ec27f1329bb314af53943bb4ce231f81b657b136600aa2d996`.
Evidence: `/x/Temp/kindle-breakout-budget400k-20261103/recovery300k-observation.json`.
No full tensor/semantic comparison,retention or GPU stage repeats. CPU co-host
cost remains uncorrected. Main training continues to the declared400k endpoint;
final frozen outcomes and terminal collection remain pending.
