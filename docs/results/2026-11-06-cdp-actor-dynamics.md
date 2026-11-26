# Optional dynamics-gradient actor: implementation and qualification

The [declared development](../experiments/2026-11-06-cdp-actor-dynamics.md) adds
`actor_dynamics_scale`, default absent, to train the actor through imagined
RSSM transitions and decoded reward/continuation/critic predictions. The existing
score-function, entropy and imitation losses remain. Hard actions and latents
use mixed-softmax straight-through derivatives; this is a biased estimator.
Disagreement values, weights and normalization are detached. No gameplay benefit
or efficiency gain is established.

World snapshots are private graph inputs copied directly from GPU parameters
before posterior sampling and world training. They add no optimizer slots,
trainable weights or checkpoint tensors. Actor parameters and moments have one
owner; the auxiliary critic path freezes weights while differentiating inputs.
The runtime reuses original imagination starts and random draws, and reports
snapshot time and the unscaled auxiliary loss separately.

CPU checks pass **188 Kindle + 5 gym tests**, **16 CLI/report tests**, and strict
workspace and binding Clippy. The initial compile catches missing action-mask and counter
type errors in the new fixture. Subsequent graph construction catches scalar-only
scaling applied to row tensors. Gradient-ownership checks then find that the
compiler exposes a frozen scalar's zero-gradient placeholder as an optimizer
slot; using explicit snapshot inputs avoids ownership entirely. A compile check
also catches confusion between logical and physical parameter buffer views.
The CLI restore check initially runs only under an encoder override; its failing
test is preserved and the guard now applies to all restores before construction.
These failures and repaired executions remain at
`/x/Temp/kindle-actor-dynamics-20261106`; all occurred before GPU qualification.

Packaging and native qualification now pass on source744dd793, native SHA256
`876ed0c55a9dae95b414aea650d31bfa35544f866c71236da82f937dd50c1a23`.
The installed package passes1,108 Python tests (10 deselected), with zero native
construction attempts. The packaged wheel and test executable remain retained;
a separate verified receipt releases1,008,970,169 bytes of disposable build files
without changing the original packaging record.

The two guarded native fixtures finish in3.379s at05:29:54.539 UTC November6,
with241 unchanged pins, three synthetic forward/backward steps, two Tiny-agent
updates and zero Atari actions. The independent F64 recurrent surrogate checks
six actor derivatives, frozen model/critic inputs and zero-weight masks. A
255-bin fixture checks uniform zero, localized means and all510 derivatives.
The production normalized loss differs from the existing rollout-return formula
by at most4.05e-9 in the two Tiny updates; this is a bounded numerical result,
not a general derivative-accuracy claim. Pre-update world snapshots, shared
starts/draws, single optimizer ownership and parameter/moment restore pass.

Default-off legacy qualification finishes05:31:48.196 UTC:1,024 frozen greedy
actions from each retained final/actual-initial checkpoint,2,048 total and zero
updates. Both reproduce all128 saved transition rows exactly (excluding finite
wall-clock timestamps) and preserve all357 checkpoint tensors. All39 pins and
both guards pass. Neither native fixture nor frozen arm was repeated; no GPU
fault or nonfinite result occurred.

## Completed bounded cost and numerics check

The separate [bounded protocol](../experiments/2026-11-06-cdp-actor-dynamics-bounded.md)
passes on documentation source7609cd4a and native744dd793/876ed0c. One fresh
seed1009 learner with coefficient.5 and the retained recipe completes
**8,192 actions /32,725 emulator frames /489 updates in58.368809s**.
Construction takes3.510119s. A subsequent1,024-action/4,096-frame greedy restore
takes.863110s with zero updates and357 invariant tensors. Both guards pass;
no fault, nonfinite value, failed stage or stage repeat occurs.

Training takes **22.393% longer** than the retained bounded47.689684s. The
padded200k projection is **1,567.522s (26m07.522s)**, below the declared30-minute
cap. Minimum observed native headroom is17,399,513,088 bytes; after subtracting
the entire5,939,200,000-byte full replay footprint,11,460,313,088 bytes (10.67GiB)
remain. These are short-run projections, not measured full-learner cost.

Summed learner timings locate the added cost in behavior training:
**2.502150→13.941962s** across489 updates. Original imagination remains
16.647413→16.637848s and world training13.612795→13.546420s. Snapshot submission
adds.142025s; this measures wrapper/submission time and is not an isolated GPU
copy benchmark. Total reported learner time rises40.438189→51.877643s. The
new backward pass is affordable here, but must improve learning enough to
compensate for its cost. Historical timing and changed experience limit causal
performance attribution; no paired timing repetition was allocated.

All357 actual-initial tensors and47 pre-update transitions match the retained
control exactly. Both weighted-loss checks pass on all489 reports: maximum
absolute errors9.825e-5 world and3.714e-5 behavior. Auxiliary loss ranges
−.371540 to−.012000 (mean−.140876), with finite snapshot timings. Encoder,
reward, continuation, actor and critic parameters change. Full CPU replay
reconstructs every training action, reward, frame, boundary and pixel label;
3,746 overrides are45.728% of actions. Guided training returns are not frozen
competence.

The actual queued command/configuration paths pass CPU preflight in.167s with
zero native/emulator construction. All45 allocation pins and237 original source
pins remain unchanged. Controller and audit finish **05:40:12.457 UTC November6**
in76.369s; the outer execution receipt succeeds in76.404s. Root
`/x/Temp/kindle-actor-dynamics-20261106/bounded` is closed. No Atari cost check,
synthetic fixture, frozen arm or collector should repeat.

At bounded closure, no full learner, extension, sweep or automatic successor
was selected. This
establishes native execution and cost feasibility, not useful policy derivatives,
gameplay improvement or a SuperMimic-like efficiency gain. The next substantive
learning comparison would need a separately declared fresh budget and frozen
actual-initial controls, judged by return gained per training minute. Keep
coordinates off and Breakout143.667 greedy/70.750 sampled; original864/90%-two-wall
and five-game gates remain unfinished.

## Full 200k comparison

The [separately declared full comparison](../experiments/2026-11-06-cdp-breakout-actor-dynamics-200k.md)
is complete and negative. **Retain `actor_dynamics_scale=None`**, coordinates off
and Breakout143.667 greedy/70.750 sampled. Adding the dynamics-gradient actor
term at coefficient.5 lowers both frozen scores and increases training time.
This does not demonstrate a SuperMimic-like learning-efficiency benefit here.

| Measure | Retained control | Dynamics coefficient.5 |
|---|---:|---:|
| Training wall time |21m27.910s|26m38.517s|
| Greedy return |143.666667|85.708333|
| Sampled return |70.750000|51.625000|
| Greedy own-initial gain/min |6.590132|3.134155|
| Sampled own-initial gain/min |3.230038|1.884559|
| Two-wall success, either final mode |0/24|0/24|

The fresh seed1009 learner completes **200,000 actions /799,565 emulator frames /
12,477 updates in1,598.517122s**,24.117% longer than retained. Primary gain/min
falls **52.44%**. Its paired eight-lane difference is−3.455977,95% interval
[−5.380121,−1.816569]; raw greedy difference is−57.958333,[−100.917708,−20.750000].
Sampled gain/min difference is−1.345479,[−1.909585,−.749736]. These10,000-resample
intervals describe the reused evaluation4000001009 cohort, not training-seed
reliability. The declared favor-candidate criterion fails; the original864/90%
two-wall gate and five-game goal remain unfinished.

Each final/actual-initial sampled/greedy arm uses24 natural episodes with no
truncation, guidance, labels or updates. All357 tensors remain invariant per
arm. Actual-initial means match retained1.416667 sampled/2.208333 greedy; all799
sampled and794 greedy semantic events also match. All357 starting tensors and47
pre-update events match both bounded and retained controls. Sampled final uses
34,488 actions/137,923 frames; greedy final uses35,384 actions/141,505 frames.
All frozen action/outcome/frame/boundary replays and videos pass.

The complete training audit passes all12,477 finite reports, weighted world and
behavior losses (including.5 times the auxiliary loss), snapshot timings,
configuration and changed learning parameters. Full CPU replay reconstructs all
training actions, rewards, frames, boundaries and corrective labels in105.831s.
Actual overrides are88,937/44.4685% of actions; the20% setting selects episodes,
not a fixed action fraction. Guided online returns are not frozen competence.

Learning-stage wall time rises1,133.927→1,452.328s. Native component sums are
371.415s behavior training,356.949s world training,453.607s ordinary imagination
and3.548s snapshot submission. The latter is wrapper/submission time, not an
isolated GPU-copy measurement. Late own-replay imitation cross-entropy improves
.560514→.545267 despite worse gameplay. The final50k auxiliary-loss mean is
−.433429; finite or decreasing losses do not establish useful model derivatives.
State distributions, optimizer updates and subsequent experience differ. This
one coefficient and development seed do not isolate a gradient-bias mechanism,
prove all dynamics-gradient recipes fail or establish broad reliability.

CPU helper checks pass in3.674s and exact five-queue command qualification in
.380s, with zero native/emulator construction. Sourcef3be4bd4 launches05:49:07.298
UTC November6, original PID1397920/invocation96e0f92ec2784b0f84cc8c37c45ae0d7.
The100k/6,227 checkpoint is checked06:02:19.684 and independently verified
06:02:37.883 in.357s:357 finite tensors/moments,345 unchanged pins and0555/0444
recovery directory/file modes. The verifier performs no model forward or update;
original learner identity remains unchanged. The recovery observer succeeds.
Replay, live games, beliefs and full RNG state are not included in recovery.

Training finishes06:15:50.771 UTC; original main and terminal collector finish
**06:19:43.764 /06:19:55.079 UTC**. All345 current/338 retained pins, guards,
curves, exposure, score coverage and terminal collection pass. Main, recovery
and collector all terminate successfully. The separate publication reader
passes in.024s. No learner, GPU stage or collector is repeated, and no production
stage fails. Earlier implementation preparation failures remain preserved.

Root `/x/Temp/kindle-actor-dynamics-20261106/full` is closed, with no running
Kindle service or render-device holder at closure. Retain the optional code
default off and the existing control. No extension, sweep, confirmation, second
learner or automatic successor is selected. Any further derivative-quality
assessment needs a separately justified protocol; do not restart this comparison.
