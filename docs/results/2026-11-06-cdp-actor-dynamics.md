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

## Selected fresh200k learning comparison

The separate [full protocol](../experiments/2026-11-06-cdp-breakout-actor-dynamics-200k.md)
selects one fresh seed1009 learner at coefficient.5 and the retained recipe,
with no bounded weights or experience transfer. The22.39% bounded cost increase
is within budget and motivates a direct test of frozen learning per minute;
no useful-gradient or gameplay benefit is assumed. Greedy is primary, using
actual-initial controls and the retained143.667/6.590132gain/min control.

CPU helper qualification passes in3.674s with no native-module import, emulator
construction or GPU work. It checks the real comparison entry, gain/min decision,
new weighted actor loss and snapshot timing, corruptions, retained exposure and
plot inputs. Exact five-queue command qualification passes in.380s with zero
native/emulator construction, including coefficient corruption and restore
configuration paths. Root `/x/Temp/kindle-actor-dynamics-20261106/full`. One200k/12,477 learner with1,800s cap,
100k recovery and four frozen24-episode arms is selected. No extension, sweep,
second learner or automatic successor is selected. Previous failures remain
preserved; no completed old stage is repeated.

The original learner launches **05:49:07.298 UTC November6**, sourcef3be4bd4,
PID1397920/invocation96e0f92ec2784b0f84cc8c37c45ae0d7. All345 allocation pins,
357 initial tensors versus both bounded/retained controls,47 pre-update semantic
events and first update at action376 pass. Startup observer takes.314s.
At05:49:50.823, original main, recovery and terminal observer process identities
are verified live; no stage failed or repeated. No frozen learning score is
available yet. These process identities, not stale logs or observer timeouts,
identify the running allocation.

The100k/6,227 recovery checkpoint is saved and checked at06:02:19.684 UTC;
independent CPU verification passes **06:02:37.883 UTC** in.357s. All357 tensors
and optimizer-state entries are finite, all345 allocation pins match, and the
recovery directory/files retain0555/0444 modes. It contains45,710,865 world,
4,938,291 behavior and853,503 slow-value elements including optimizer storage.
The verifier performs zero model forwards and zero learner updates. The original
learner process/invocation remains active throughout; the recovery observer
finishes successfully in.414s. No stage or collector is repeated. Replay, live
games, beliefs and full RNG state are not restored by this retained checkpoint.
The200k learner and terminal collector remain active; no frozen score is available.
