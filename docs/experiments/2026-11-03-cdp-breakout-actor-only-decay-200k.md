# Fresh200k Breakout comparison with late actor-only imitation decay

Select one fresh comparison after the [numerical and bounded qualification](../results/2026-11-03-cdp-actor-only-imitation-decay.md)
passes8,192 actions/489 updates in47.329s,599 pins,357 matching initial/invariant
frozen tensors,legacy action reproduction and full guided replay. The conservative
200k estimate21m11s passes30 minutes. Bounded experience is not reused.

The intended setting change from the retained replay-value1 control is actor
imitation100 held through update6227 (100k actions), linearly reaching zero at9352
(150k), then staying zero through12477. Include the endpoint:3,126 zero-weight
reports. **World imitation stays100 throughout.** Retain teacher guidance,
corrective labels and optimizer history; zero current actor imitation does not
remove earlier momentum or constitute unaided collection. Current actor gradient
attribution motivates this test but does not establish a causal gameplay defect.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/discount333,
base-encoder4e-5/behavior8e-5/dynamics4e-4, policy1/imagined-value1/replay-value1,
expert unit weights/world gradients/all-state corrective labels, actor/critic
representation gradients off, sampled collection/posteriors, actor-unimix.01/
entropy.0003, action-effects1, replay100k/warmup1000,20% episode guidance/cap1920,
seed1009/eval4000001009, full18/sticky.25/repeat4/noops0/native pooled RGB/one
GPU64 resize/raw rewards. No crop,centroid or open-loop loss is enabled.

Use qualified nativec2d627f3/SHA0015dff7 and the matching runtime/runner under
`/x/Temp/kindle-actor-only-decay-20261103`. Native and runner differ from retained
447fe50; record the explicit `replay_value_bootstrap=imagined` default addition
when comparing configs. Teacher and other learner settings stay fixed. Matched
initial values and legacy traces do not establish isolated causality or broad
runtime equivalence. Seed/cohort reuse is development, not independent confirmation.

Fresh training reuses no weights,replay,belief,RNG,games or experience. Stop at
200,000 actions/12,477 updates or1800s. Preserve actual initial and100k/6227
recovery with all four retained files and357 finite tensors. First update376,
residual.5,finite reports,exact actor schedule and constant world weight must pass.
Reconstruct behavior totals with the scheduled actor term and require raw
world/actor expert CE alignment. Record actual overrides separately from competence.

Fixed frozen arms are final/actual-initial, each sampled and greedy, first three
natural episodes per8 lanes=24 per arm. Sampled remains primary. Guidance/labels
off; zero updates/truncations and357 invariant tensors. Keep300s/600k actions
per arm. No outcome-based checkpoint/mode selection,retry or episode-cap increase.
Compare against retained sampled70.750/greedy143.667 and own initials1.417/2.208,
all0/24 two-wall. The90% two-wall gate remains unchanged. Report both modes and
own-initial-subtracted frozen learning per training minute.

Require357 initial values matching control and bounded check,47 pre-update events,
both initial frozen semantic traces,changed CNN/reward/actor/value,full training
pixel/action/label replay,frozen action/outcome/frame/boundary replays and videos,
guards,curves,exposure and comparison. Ordinary frozen logs have no live pixel
hashes; do not claim unavailable image equality. Keep CPU co-host timing without
wall-time corrections.

Before launch,qualify all five actual CLI paths,audit/comparison assertions and
negative cases,schedule accounting,header readiness,observer wiring,retention and
checkpoint schema on CPU. Pin source,runtime,scripts,inputs and protocol from a
clean commit. Retain one original main/progress/recovery/postprocessing/collection
service. Main cap4500s,audit1200s and fixed observer deadlines stay unchanged.
Preserve failures and repair CPU-only processing separately; never repeat a
successful GPU stage. Independently collect startup,recovery and terminal evidence
once. Any additional saved-data reader must be declared before reading outcomes.

No extension,coefficient sweep,confirmation learner or automatic successor is
selected. Root `/x/Temp/kindle-actor-only-decay-20261103/full`; service
`kindle-breakout-actor-only-decay-200k-20261103`. This is a deliberate comparison
within the unfinished five-game goal; execution success alone cannot finish it.

## CPU preparation

All five actual CLI paths,48 audit/comparison config cases,14 prelaunch command
cases,19 header checks,four observer-parser cases,14 recovery-body cases,seven
schedule cases,six retention cases and two collector corruption cases pass.
All19 controllers compile and44 embedded paths resolve;599 bounded and338
control pins remain unchanged. CPU preflight completes11:00:35.722 UTC
November3 in2.635s wrapper/2.522s body,with no native/environment construction.

The original.096s preflight import failure is retained: the F32 converter is
local to the general audit function,so the repaired controllers import the
existing independent decay-schedule converter. Before that first test,a source
review corrects one endpoint-neighbor fixture to the retained native report's
F32 value. Production code,runtime and schedule arithmetic remain unchanged;
no GPU stage occurs during preparation. Exact prior sources and receipts remain
under the private package.
