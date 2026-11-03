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
`kindle-breakout-budget400k-20261103`. Selection only: CPU harness qualification
and launch are pending. Check live state before work; no overlapping GPU jobs.
No extension,coefficient sweep,fresh-seed confirmation or automatic successor
is selected. This reused seed/cohort cannot establish independent reliability.
Breakout and the five-game goal remain unfinished.
