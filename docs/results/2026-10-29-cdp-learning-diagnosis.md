# Learning diagnosis: useful forecasts and better replay fitting do not guarantee control

The [fresh200k actor-width512 comparison](../experiments/2026-11-06-cdp-breakout-actor-width-200k.md)
is selected after [bounded qualification](2026-11-06-cdp-actor-width.md). Actor-only capacity is untested;
whole25M changed the world/critic too. Keep12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/dynamics4e-4/behavior8e-5,critic1/1,expert100/world gradients and
20% guided episodes. Width512 adds1,054,720 trainable actor parameters; other
widths/losses stay fixed. Coordinates and actor dynamics stay off.

One fresh seed1009 learner receives200k actions/12,477 updates and100k recovery,
with no bounded experience/weight transfer. Four frozen final/actual-initial
sampled/greedy arms select24 natural episodes each. Judge raw greedy return
and own-initial gain/min against143.667/6.590132, with the predeclared paired
lane interval. Initial actor values/actions differ from retained; require313
non-actor initial matches and exact357 tensors/47 events against the bounded
run. CPU helper/comparison/plot checks pass in3.541s; all five queued command
paths pass in.414s without native/emulator construction. The original learner
starts08:27:14.431 UTC November6/source91647453/PID1432519/invocation
`6e0453060296495b8d1062b0dadc9b37`, with347 pins and recovery/terminal observers.
Startup passes08:27:20.152:357 bounded initial tensors/47 events and313 retained
non-actor tensors match;11 retained initial events differ as allowed. Root
`/x/Temp/kindle-actor-width-20261106/full`, service
`kindle-breakout-actor-width-200k-20261106`, is active. The100k/6,227 checkpoint
passes its original audit08:39:19.911 UTC and independent CPU verification
08:39:45.022 in.372s:357 finite tensors,347 unchanged pins and read-only files.
The verifier performs no model forward/update; original learner identity stays
unchanged and the recovery observer succeeds. Recovery omits replay, games,
live beliefs and full RNG. Frozen scores remain pending. Check original live
identity; no extension, sweep, second learner or automatic successor. Original
864/90%-two-wall and five-game gates remain.

The completed bounded run takes53.270s for8,192 actions/32,715 frames/489 updates,
11.702% longer than retained; padded200k projection23m50.597s and11.214GiB
conservative replay headroom pass. Native/CPU/restore/replay/guard checks pass
without repeated or failed stages. Packaged runtime remains after4.807GB logical
cache release. No bounded stage or old collector should repeat.

The [completed actor-update diagnostic](2026-11-06-cdp-dynamics-actor-update.md) finds dynamics.5 changes
the full estimated update by.01136%/.01702% at two retained200k batches;
cosines exceed.99999998. Conditional imitation exceeds the auxiliary term
4782×/3282×. Critic gradients and estimated updates are exactly unchanged.
All actions, features and targets reconstruct exactly;419 native pins,22
unchanged behavior parameters,32 world inputs and guards pass. Native.369s,
controller3.986s, complete07:38:44.292 UTC November6, sourcec041afea/SHA51c5332c.
Zero Atari actions or updates. Original CPU launcher collision remains failed;
separate unchanged-arithmetic repair completes in.698s with454 pins. Closure
07:43:01.701 seals92 files. Packaged executable retained,4.688GB logical build
cache released. No active allocation or successor selected. These late fixed
batches do not explain the candidate's early trajectory or justify a coefficient
sweep. Keep actor dynamics off and the retained control; five-game goal unfinished.

The [frozen action-path probe](2026-11-06-cdp-breakout-action-path-derivatives.md)
is closed. Recorded-plan return gradients favor81/92 known short rescues for
control and80/92 for candidate; full-plan predictions favor71/92 and73/92.
Survival gradients favor12/18 and8/18 differing game-over rescues, with local/
finite agreement10/18 and7/18. Actual-terminal plans still receive.970267/.967395
discounted survival. These findings do not show globally reversed return gradients
or isolate the failed learner's cause; fixed plans omit actor feedback/optimizer
interactions and short rescues lack complete alternative returns.

All775 production pins,43 invariant frozen GPU inputs per model,checkpoint hashes,
19 identical-plan zero checks and guards pass. Native1.676s/.949s,controller6.029s,
complete07:06:43.521 UTC November6; CPU analysis.417s/closure07:11:32.503. Zero Atari
actions or learner/optimizer updates; no failed/repeated production stage. The
original toy-shape qualification failure and separate passing repair remain.
Root `/x/Temp/kindle-breakout-action-path-20261106/shape-repair` is closed; both
binaries retained,5,610,733,853 logical cache bytes released. No active allocation
or successor selected. Keep actor dynamics off and the retained baseline.

The [matched action-derivative context capture](2026-11-06-cdp-breakout-action-derivative-contexts.md)
is complete:120 latent origins per model,43,424 actions/173,660 frames each,
50.591s control/50.498s candidate. All16 CPU tests,537 pins,357 invariant tensors
per model,exact transitions and guards pass. Zero updates or multi-step forecasts;
no failed/repeated stage. Controller completes06:42:29.897 UTC November6, with
no active GPU allocation at closure. Root
`/x/Temp/kindle-breakout-action-derivatives-20261106` retains the states alongside
references to92 known short rescues/18 differing game-over outcomes. This supplies
inputs for a separately qualified native derivative probe; derivative quality
remains unmeasured, and no derivative GPU stage or learner is selected here.

The [full dynamics-gradient actor comparison](2026-11-06-cdp-actor-dynamics.md#full-200k-comparison) is complete and negative:
200k actions/799,565 frames/12,477 updates in **26m38.517s**, with **85.708 greedy /
51.625 sampled** versus143.667/70.750. Training costs24.117% more; primary
own-initial gain/min falls52.44%,6.590132→3.134155. Its paired eight-lane95%
interval[−5.380121,−1.816569] describes the reused cohort, not seed reliability.
All0/24 two-wall. Retain `actor_dynamics_scale=None` and coordinates off.

All345 current/338 retained pins,100k recovery,357 matching initial/invariant
frozen tensors,47 pre-update events,weighted losses,full replays/videos,guards,
curves,exposure and terminal collection pass. Actual-initial1.416667/2.208333 and
all799/794 initial semantic events match retained. Original main/collector finish
06:19:43.764/06:19:55.079 UTC November6/sourcef3be4bd4; all three services succeed.
The.024s publication reader passes. No failed/repeated production stage; original
implementation preparation failures remain. Root
`/x/Temp/kindle-actor-dynamics-20261106/full` is closed, with no allocation active
at closure. No extension,sweep,confirmation or automatic successor is selected.

Learning-stage time rises1,133.927→1,452.328s. Late own-replay imitation CE improves
.560514→.545267 despite worse gameplay. Correct derivatives and finite losses
have not produced useful extra learning at coefficient.5 in this development
seed. This does not isolate a categorical-gradient bias mechanism or rule out
all dynamics-gradient recipes. Original864/90% and five-game gates remain
unfinished; further derivative-quality work needs a separate justified protocol.

The [optional dynamics-gradient actor](../experiments/2026-11-06-cdp-actor-dynamics.md) is implemented and natively qualified. A private pre-update world snapshot receives the same
initial states/draws as imagination; actor gradients traverse dynamics and frozen
reward/continuation/critic heads. Categorical straight-through derivatives are
biased; disagreement values, trajectory weights and normalization stay detached.
The existing actor losses remain, with one optimizer and no added trainable
weights or checkpoint tensors. Default off; its bounded cost check is now complete.

[Qualification](2026-11-06-cdp-actor-dynamics.md) passes188 Kindle+5 gym tests,1,108 packaged Python tests, strict
workspace/binding Clippy and guarded F64/production/snapshot/restore checks.
Default-off final/initial policies reproduce2,048 frozen actions and preserve357
tensors each. Original preparation failures remain; no GPU stage repeats.
The [bounded result](2026-11-06-cdp-actor-dynamics.md#completed-bounded-cost-and-numerics-check) completes8,192 actions/32,725 frames/489 updates in58.369s,
22.39% longer than retained47.690s. Behavior training rises2.502→13.942s;
imagination/world time is essentially unchanged. The26m07.522s padded200k
projection and10.67GiB conservative replay headroom pass. All357 initial tensors,
47 pre-update events,weighted losses,full pixel/action replay,357 frozen tensors,
45 allocation/237 source pins and guards pass. No failed or repeated stage.
Controller closes05:40:12.457 UTC November6/source7609cd4a/native744dd793.
Root `/x/Temp/kindle-actor-dynamics-20261106/bounded` is closed; no successor was
selected by that check. The separate full comparison above is now closed. Numerical correctness and cost
feasibility do not establish gameplay improvement or useful model derivatives.
Original Breakout864/90% and five-game gates remain unfinished.

The [completed coordinate-channel comparison](2026-11-06-cdp-rgb-coordinates.md#full-200k-comparison)
finishes 200k actions / 799,523 frames / 12,477 updates in **20m58.414s**.
Greedy **66.333** and sampled **35.917** regress retained **143.667 / 70.750**;
all 0/24 two-wall. Primary own-initial gain/min falls 53.36%, despite 2.29%
shorter training time. Retain `rgb_coordinates=false`. The paired eight-lane
95% gain/min interval [−4.914387, −1.999408] describes the reused cohort,
not seed reliability or an isolated perceptual cause.

All 124 current / 338 retained pins, 100k recovery, full replays, frozen controls,
357 invariant tensors, guards and curves pass. Original main/collector finish
04:55:15.624 / 04:56:00.819 UTC November6; all three services succeed. Earlier
bounded import, packaging and fixture failures and the pause timing remain in
the result. No completed GPU stage repeats. Root
`/x/Temp/kindle-rgb-coordinates-20261104/full` is closed; no allocation is active
at closure. Check live state before work. No extension, sweep, confirmation or
automatic successor is selected.

The [SuperMimic source review](2026-11-06-supermimic-architecture-review.md) identifies direct
actor gradients through imagined dynamics as a separate architectural proposal
within the same goal. It is not implemented or allocated by this comparison.
The original 864/90%-two-wall gate and five-game goal remain unfinished.

The [completed categorical-mixing comparison](2026-11-04-cdp-breakout-latent-unimix001.md#full-200k-comparison) finishes 200k actions /
799,612 frames /12,477 updates in **21m19.046s**. Lower latent mixing.001 scores
**67.208 greedy /51.583 sampled**, below retained.01's143.667/70.750; all0/24
two-wall. Primary own-initial gain/min falls53.46%; paired difference−3.523393
has95% lane interval[−5.303444,−1.906890]. Retain.01. The.69% shorter wall time
comes with essentially unchanged learning-stage time1133.956/1133.927s.
Late own-replay imitation CE improves.560514→.535996 despite worse gameplay;
raw KL rises11.815355→12.406710. Sampling,KL,derivatives and experience change
together, so no isolated noise mechanism or seed reliability is established.

All154 current/338 retained pins,100k/6,227 recovery,357 matching initial/
invariant frozen tensors,47 bounded initial events,finite reports,full replays/
videos,guards,curves,exposure and comparison pass. Actual-initial trajectories
differ as allowed; candidate greedy initial1.833 is used. Actual overrides43.823%.
Main/collector succeed20:45:02.471/20:45:24.864 UTC November4/sourceb3a49b5b;
all three original services succeed. A separate2.669s publication reader repair
closes20:46:39.251 after a stale manifest filename. Original failure and bounded
CLI-validation/stdout-manifest repairs remain; no production stage repeats.
Root`/x/Temp/kindle-breakout-latent-unimix001-20261104/full` is closed, with no
active Kindle allocation at closure. No sweep,extension,confirmation or automatic
successor is selected. Original864/90% and the five-game goal remain unfinished.

The [saved corrective-label timing audit](2026-11-04-cdp-breakout-label-timing.md) finds the retained learner's
late sampled actions align best with current labels: **52.73%**, versus43.02%
with preceding labels, on24,998 identical unaided directional windows. All
nonzero offsets−4…+4 lower agreement in every lane; current-label switches show
the same pattern. The exact native447fe50/runnercf4ea238 source review finds no
off-by-one mismatch in the inspected path. This does not support a simple fixed
delay; representation precision, recurrent state, optimization and teacher
quality remain unresolved. Evolving training policies and feedback-dependent
labels limit causal interpretation; no frozen score or loss changes.

Eight CPU fixtures,18 pins, all200k actions/799,595 frames and the independent
52,477-anchor/54-group-offset recount pass. Analysis1.067s completes20:02:57.209
UTC November4/source185e3180; recount1.198s closes20:03:15.242. No failed or
repeated stage, native/emulator import, GPU work or update. Root
`/x/Temp/kindle-breakout-label-timing-20261104` is closed; no successor selected.
Retain143.667/70.750 and B16/T16; original864/90% and the five-game goal remain
unfinished. No Kindle service is active at closure; check live state before work.

The [completed B32/T16 comparison](2026-11-04-cdp-breakout-batch32.md#full-200k-comparison) finishes200k actions/799,541 frames/
6,231 updates in **19m20.785s**,9.87% less training time than retained. Greedy
143.667→73.333 and sampled70.750→47.750 both fall; all0/24 two-wall. Primary
own-initial gain/min falls44.21%; paired difference−2.913742 has95% lane interval
[−5.157081,−.748597]. Retain B16/T16 and143.667/70.750. No batch sweep,rate
compensation,extension,second learner or automatic successor is selected.

Full sampled positions fall only.12%,while learning-stage time falls1133.927→
1023.134s. Batch/world microbatch16→32 changes centering,diversity,cadence and
update-based warmup together; no isolated mechanism or reliable speed claim.
Late labeled-row imitation CE is.613392 versus.560514. Actual overrides52.277%;
own-replay fitting and guided returns are not frozen competence. Native447fe50/
SHA35ad0db,runnercf4ea238,teacher,T16/R16 and rates stay fixed.

All137 current/338 retained pins,100k/3,106 recovery,357 matching initial/invariant
frozen tensors,79 bounded/47 retained initial events,799/794 frozen initial
semantics,6,231 finite reports,full replays/videos,guards,curves,coverage,exposure
and comparison pass. Main/collector succeed19:48:24.902/19:49:09.145 UTC November4,
source00e548d3; recovery succeeds earlier. Separate2.529s publication review
closes19:49:35.865 with55 completion artifacts. Pre-execution fixture correction
and ancillary progress-reader syntax error remain preserved; no original
pipeline stage fails or repeats. CPU co-host time stays uncorrected.

Root`/x/Temp/kindle-breakout-batch32-20261104/full` is closed; all three services
succeed and no GPU allocation is active at closure. Do not restart the learner
or collectors; check live state before work. Original864/90%-two-wall and the
five-game goal remain unfinished.

The [completed discount100 comparison](2026-11-04-cdp-breakout-discount100.md#full-200k-comparison) finishes
200k actions /799,569 frames /12,477 updates in **21m23.435s**, 0.35% less time
than retained. Shortening return discount horizon333→100 lowers greedy
143.667→70.250 and sampled70.750→56.083; all0/24 two-wall successes. Primary
own-initial gain/min falls51.73%; paired difference−3.409216 has95% lane interval
[−4.516728,−2.234156]. All eight greedy lane means are lower. Retain horizon333
and143.667/70.750; no discount sweep,extension,second learner or automatic successor.

Late labeled-row imitation CE improves.560514→.546030 despite worse frozen play.
Own-replay fitting is not competence; new critic calibration is unmeasured.
Continuation targets,imagined/replay credit and experience change together,
without an isolated causal mechanism or seed-reliability claim. Imagination15,
native447fe50/SHA35ad0db and teacher stay fixed; runner112be6c4 exposes horizon.

All151 current/338 retained pins,100k/6,227 recovery,357 matched initial/invariant
frozen tensors,47 pre-update and799/794 actual-initial semantic events,full
replays/videos,guards,curves,coverage,exposure and comparison pass. Actual
overrides44.156%; guided returns are not frozen competence. CPU co-host time
stays uncorrected. The earlier CPU source-name assertion and repair remain;
no full-run stage fails or repeats.

Main/collector complete19:08:30.521/19:08:42.614 UTC November4/source11477bc7;
all three original services succeed. Separate2.919s publication review closes
19:09:01.626 with55 completion artifacts. Root
`/x/Temp/kindle-breakout-discount100-20261104/full` is closed, with no active GPU
allocation at review. Do not restart its learner or collectors;check live state
before work. Original864/90%-two-wall and the five-game goal remain unfinished.

The [completed dynamics-scale3 comparison](2026-11-04-cdp-breakout-dynamics-scale3.md#full-200k-comparison)
finishes 200k actions /799,607 frames /12,477 updates in **21m33.324s**, 0.42%
above the retained control. Increasing prior-fitting KL1→3 lowers greedy
143.667→105.667 and sampled70.750→62.208; all0/24 two-wall successes.
Primary own-initial gain/min falls27.17%; its paired difference−1.790483 has95%
lane interval[−3.243235,−.399936]. Retain dynamics scale1 and the143.667/70.750
baseline. No extension, coefficient sweep, second learner or automatic successor.

Late raw KL rises11.815355→13.872131. These own-replay means, shared recurrent
gradients and changing experience do not identify a causal bottleneck. Native
447fe50/SHA35ad0db and teacher remain unchanged; runner52593379 exposes the
coefficient. Other retained12M64/Auto/N8/B16/T16/H15/R16/expert100 settings stay
fixed. Reused1009/cohort does not establish seed reliability.

All145 current/338 retained pins,100k/6,227 recovery,357 matched initial/invariant
frozen tensors,47 pre-update and799/794 initial semantic events,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Actual overrides42.418%;
guided returns are not frozen competence. CPU co-host time stays uncorrected.
Earlier exploratory JSON-reader and canonical import failures/CPU repairs remain;
no full-run stage fails or repeats.

Main/collector complete18:27:07.476/18:27:46.942 UTC November4/sourceb222c0c8;
all three original services succeed. Independent2.635s publication review closes
18:28:08.346 with55 completion artifacts. Root
`/x/Temp/kindle-breakout-dynamics-scale3-20261104/full` is closed, with no active
GPU allocation at review. Do not restart its learner or collectors; check live
state before work. The original864/90%-two-wall gate and five-game goal remain
unfinished.

The [completed B32/T8 check](2026-11-04-cdp-breakout-b32-t8.md) takes46.818s for8,192 actions/
32,723 frames/493 updates:1.83% below retained47.690s, missing the required5%.
The padded200k projection20m57.311s passes30min, but both cost conditions were
required. Retain B16/T16; no full learner,retry,sweep or automatic successor.
Learning40.471s is almost unchanged from40.495s; sustained speed/gameplay remain
unmeasured. Frozen1,024/.875s haszero updates,357 invariant tensors,no episodes.
All67 current/338 retained pins,357 matching initials,39 initial events,493 finite
reports,weighted losses,full replay and guards pass. Original CPU argparse-fixture
failure and repair remain;no native/runner changes or repeated GPU stages.
Main/collector succeed17:43:44.620/17:43:52.229 UTC November4/sourcec6ec7a95;
independent2.425s closure17:44:20.894 retains80 artifact hashes. Root
`/x/Temp/kindle-breakout-b32-t8-20261104` is closed;no GPU allocation is active at
review. Check live state. Original Breakout gate and five-game goal unfinished.

The [completed catch-reward comparison](2026-11-04-cdp-breakout-catch-reward1.md#full-200k-comparison)
finishes 200k actions /799,557 frames /12,477 updates in **21m51.733s**,
1.85% above the retained control. Adding one training point per causal pixel
catch lowers greedy143.667→86.417 and sampled70.750→68.458; all0/24 two-wall.
Primary initial-subtracted gain/min falls41.55%; its paired difference−2.738356
has95% lane interval[−4.825834,−.559765]. The locked preference rule fails.
Retain the unshaped recipe. This reused development learner/cohort does not
identify the learning mechanism or establish broad reliability.

All473 current/338 retained pins,100k/6,227 recovery,357 matched initial/invariant
frozen tensors,47 pre-update and799/794 initial semantic events,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Independent replay verifies
5,231 catches:raw reward13,151,training reward18,382. Actual overrides42.932%;
guided returns and reward exposure are not frozen competence.

Main/collector succeed17:17:51.351/17:18:02.038 UTC November4/source318591c9;
terminal audit takes3.661s. Recovery succeeds earlier. All original services
finish successfully with no repeated GPU stage. Supplemental publication readers'
parent-versus-child journal and historical-field lookup errors remain preserved,
with separate read-only corrections;the original pipeline is unaffected.

Root `/x/Temp/kindle-breakout-catch-reward1-20261104/full` is closed. No GPU
allocation is active at review;check live state before work. No extension,
coefficient sweep,second learner or automatic successor is selected. The original
864/90%-two-wall gate and five-game goal remain unfinished.

The [bounded one-point catch-reward check](2026-11-04-cdp-breakout-catch-reward1.md)
passes8,192 actions/32,725 frames/489 updates in47.580s;frozen1,024 actions/
4,096 frames/.897s has zero updates,357 invariant tensors and no natural
completed episodes. Independent replay verifies139 catches:raw reward189,
training reward328. The padded200k projection21m17.777s passes30min;execution
and early cost do not establish gameplay benefit or a speed improvement.

All405 current/338 retained pins,357 matching initial tensors,47 shared pre-update
events,finite losses,changed CNN/reward/continuation/actor/value,full training
pixel/action/label/catch/reward replay,frozen outcome replay and guards pass.
All18 CPU fixtures and43,392 saved arrivals pass;default-off accounting exactly
matches the retained200k log. Original test import-path and old-report field
lookup failures remain preserved;separate repairs repeat no completed checks.
Native447fe50/SHA35ad0db and teacher unchanged;private runner25327009.

Main/collector succeed16:35:56.085/16:36:07.631 UTC November4/source69b14f45.
Separate3.713s review closes16:36:31.546 with47 artifacts. Both original services
are terminal;no GPU stage or original collector fails/repeats. Root
`/x/Temp/kindle-breakout-catch-reward1-20261104` is closed;do not repeat the check.

Select one [fresh200k catch-reward comparison](../experiments/2026-11-04-cdp-breakout-catch-reward1-200k.md) after this cost pass.
Preparation is pending;no full learner or GPU allocation is active yet. Use the
exact qualified overlay,30-minute cap,100k/6,227 recovery and four actual-initial/
final sampled/greedy24-episode assessments. Primary greedy raw score and
initial-subtracted gain/min must improve with a positive paired interval.
No bounded experience transfer,coefficient sweep,extension or automatic successor.
Check live state before launch. Retain143.667 greedy/70.750 sampled;original
864/90%-two-wall and the five-game goal remain unfinished.

The [completed raw-motion catch validation](2026-11-04-cdp-breakout-catch-validation.md)
finds unique near-paddle reversals for all1,476 teacher and1,132 retained-policy
signals,including all700 previously unresolved teacher detections. All arrivals
have zero raw reward;neither cohort has a signal within16 raw frames of its120
selected life losses or in the first eight episode decisions. Spatial support
is1,294/1,476 and961/1,132. A separate post-hoc reader finds all182/171 exclusions
involve crop-edge paddle components;the fixed matching rule is unchanged.
Pixel witnesses do not establish exact collision truth or gameplay benefit.

One instrumented CPU replay reproduces43,424 retained-policy actions/173,660
frames,including all prior pixel/action/reward/boundary records,in44.848s.
Analysis/recount take1.405/.481s;controller46.947s completes16:19:04.097 UTC
November4/sourcee012286e. All12 new fixtures,16 retained detector fixtures,
58 unchanged pins and14 completion artifacts pass. The separate.099s exclusion
reader completes16:20:33.403. No failure,repair,repeated completed stage,native
import,learner update or GPU work occurs. Root
`/x/Temp/kindle-breakout-catch-validation-20261104` is closed;check live state.

Next selected development is the [bounded one-point catch-reward check](../experiments/2026-11-04-cdp-breakout-catch-reward1-check.md).
Implementation and CPU qualification are pending,followed by one fresh8,192-action
check and1,024 frozen actions. Default off;training-only explicit pixel reward;
raw game scores and all frozen assessments remain unaided. Native447fe50 and
teacher unchanged. No full learner,extension,coefficient sweep or automatic
successor. No GPU allocation is active at selection. Retain143.667 greedy/
70.750 sampled;original864/90%-two-wall and the five-game goal remain unfinished.

The [CPU-only catch-signal diagnostic](2026-11-04-cdp-breakout-catch-signal.md)
finds a fixed causal pooled-pixel detector matches all776 fitted contact
candidates,including674 accepted-error fits,in24 saved teacher episodes.
It emits1,476 signals;700 have no neighbor in the filtered,incomplete reference
and remain unresolved. Mean detection delay is5.550 raw frames;all1,476 arrivals
have zero game reward. This is reference coverage,not detector precision,
current-policy validation or evidence that a catch reward improves gameplay.

All16 CPU fixtures,23 unchanged pins,prior evidence links and independent output
recount pass. Analysis/recount take.435/.143s;controller.615s completes16:07:08.888
UTC November4/sourceed155372,with18 completion artifacts. No failure,repair,
repeated stage,ALE/native import,learner update or GPU work occurs. Root
`/x/Temp/kindle-breakout-catch-signal-20261104` is closed;no running Kindle service
is observed before or afterward. Check live state before any GPU work.

No reward coefficient,training change or automatic successor is selected.
Unmatched signals and current-policy misses/serves need validation before a
training aid. Retain Breakout143.667 greedy/70.750 sampled and representation
scale0.1 after the [lower-KL regression](2026-11-04-cdp-breakout-representation001.md#full200k-comparison). Original864/90%-two-wall
and the five-game goal remain unfinished.

The [completed representation-KL comparison](2026-11-04-cdp-breakout-representation001.md#full200k-comparison)
finishes200k actions/799,591 frames/12,477 updates in21m23.007s,close to retained
21m27.910s. Reducing posterior KL scale0.1→0.01 lowers greedy143.667→93.208
and sampled70.750→35.792;all0/24 two-wall. Primary gain/min falls35.42%; its
paired difference−2.334505 has95% lane interval[−3.921731,−.481545]. The locked
preference rule fails. Retain representation scale0.1; no mechanism or broad
reliability claim follows from one reused development learner/cohort.

All774 current/338 retained pins,100k/6,227 recovery,357 matching initial/
invariant frozen tensors,47 pre-update and799/794 initial semantic events,
full replays/videos,guards,curves,coverage,exposure and comparison pass. Late
raw KL rises11.815355→16.080293 while reported expert loss falls.559936→.541684;
own-replay fitting does not establish gameplay benefit. Actual overrides42.074%.

Main/collector succeed09:39:32/09:39:42 UTC November4; separate4.155s saved-result
review completes09:41:35,verifying57 artifacts. All three original services
succeed. Original prelaunch exact-float CPU assertion and separate suffix repair
remain preserved,along with the earlier bounded flag-spelling fixture failure.
No training,evaluation,replay or collector stage fails or repeats. CPU co-host
costs remain included. Sourceca33c9fd/native447fe50/runtime unchanged.

Root`/x/Temp/kindle-breakout-representation001-20261104/full` is closed.
No GPU allocation is active at closure; check live state. No extension,coefficient
sweep,second learner or automatic successor is selected. Retain143.667 greedy/
70.750 sampled. Original864/90%-two-wall and the five-game goal remain unfinished.

The [bounded representation-KL check](2026-11-04-cdp-breakout-representation001.md)
changes only posterior KL scale0.1→0.01 and passes8,192 actions/32,719 frames/
489 updates in47.300s, versus retained47.690s. Conservative200k projection
21m10.262s passes30min; gameplay benefit is unmeasured. Frozen1,024 actions/.872s
has zero updates,357 invariant tensors and no completed natural episodes.

All706 current/338 retained pins,357 matching initial tensors,47 pre-update
semantic events,489 finite weighted-loss reports,full pixel/action/label replay
and guards pass. Original canonical CPU fixture has34 passes/one unsupported
flag-spelling failure; its source/receipt remain preserved. Only that failed
fixture is repaired; all35 packaged cases and preflight pass. Production source
is unchanged by the repair; no GPU or completed audit stage repeats.

Main/collector succeed09:04:06/09:04:18 UTC November4/sourceae529140. Independent
6.424s CPU closure completes09:04:41, verifying89 artifacts; the saved-report
summary takes.025s. Raw KL mean rises2.687→3.942 while early expert CE is nearly
unchanged; own-replay fitting is not gameplay evidence. Native447fe50/runtime
unchanged. Root`/x/Temp/kindle-breakout-representation001-20261104` is closed.
No GPU allocation is active at closure; check live state before work.

No full learner,extension,sweep or automatic successor is selected. Retain
143.667 greedy/70.750 sampled,all0/24 two-wall. Any full comparison requires
fresh experience,100k recovery and four24-episode frozen arms. Original864/90%
and the full five-game goal remain unfinished.

The [matched prior-continuation arrival diagnostic](2026-11-04-cdp-breakout-prior-continuation-arrivals.md)
finds true-ending prior MSE.911006→.162000, all24 errors lower, but intermediate-life
MSE.001530→.178824 and false below-.5 predictions0→29/96. Prior AUC.791667→.807726
has difference+.016059 with paired lane95% interval[−.107216,.132389]; no improved
discrimination or equivalence is established. Posterior errors show the same
tradeoff. The fixed control-policy life-loss cohort does not establish population
calibration or explain the sampled/greedy gameplay gap causally.

One50.803s capture re-observes43,424 actions/173,660 frames with zero updates and357
invariant tensors. All19 CPU checks,1,108 pins,120 joins/repeated queries,full
pixel/action/reward/frame/reset replay and guards pass. F64/native prior error
is at most1.697e−7. Analysis takes.329s; independent4.353s CPU closure verifies
head arithmetic,4,608 ranking pairs and intervals. Original service succeeds
08:40:44.594 UTC November4/sourcece299c0e; closure completes08:41:29.381.
A subsequent outer display KeyError is preserved; separate.107s saved-result
reader completes08:42:31.075, verifies55 artifacts and repeats no capture,
analysis or closure stage. Original closure stays passed.

Root`/x/Temp/kindle-prior-continuation-arrivals-20261104` is closed; native07873e7c,
runtime and source checkpoint remain unchanged. Retain143.667 greedy/70.750
sampled over prior-loss147.625/49.167, whose locked improvement rule failed;
all0/24 two-wall. No GPU allocation is active at closure; check live state.
No new learner,extension,sweep or automatic successor is selected.
Original864/90%-two-wall and the full five-game goal remain unfinished.

The [completed prior-continuation comparison](2026-11-04-cdp-breakout-prior-continuation.md#full200k-comparison)
finishes 200k actions / 799,588 frames / 12,477 updates in21m35.191s, .57% longer
than control. Greedy147.625 versus143.667 rises2.76%; sampled49.167 versus70.750
falls30.51%. All0/24 two-wall. Primary gain/min rises2.22%, but its paired difference
+.146325 has95% lane interval[−2.613242,2.672747]. The locked improvement rule fails;
retain the control with this optional loss off. No equivalence or broad reliability
claim follows from one reused development learner/cohort.

All389 current/338 retained pins,100k/6,227 recovery,357 matching initial/invariant
frozen tensors,47 pre-update and799/794 initial semantic events,full replays/videos,
guards,curves,coverage and comparison pass. Auxiliary loss mean.028900 and5,333
native-reported masked rows reproduce; sampled row identities are not logged.
Main/collector succeed08:28:41/08:29:18 UTC November4; independent3.342s CPU closure
completes08:29:38, verifying55 artifacts. All three original services succeed;
no full-run failure,repair or repeated stage. Earlier bounded CPU fixture failure
remains preserved. Sourceae754547/native07873e7c/runtime unchanged.

Root`/x/Temp/kindle-prior-continuation-20261104/full` is closed. No GPU allocation
is active at closure; check live state before work. No extension,second learner,
coefficient sweep or automatic successor is selected. Native/runner/wrapper
differences,changed experience and CPU co-hosting limit interpretation.
The original864/90%-two-wall gate and five-game goal remain unfinished.

The [one-step prior continuation bounded check](2026-11-04-cdp-breakout-prior-continuation.md)
passes8,192 actions/32,728 frames/489 updates in47.016s. Conservative200k projection
21m32.945s passes30min; frozen1,024 actions/.892s haszero updates/no natural episodes.
This default-off scale1 native loss trains the shared head on sampled one-step
priors with actual terminal targets, equal row weights and reset masks. No new
parameters or change to acting, forecasts, sampler or posterior loss. At bounded
closure, gameplay benefit was unmeasured; the separate full comparison is above.

All184 Rust/5 gym CPU tests,1,064 packaged Python checks,strict Clippy,two guarded
F64/gradient fixtures,314 current/338 retained pins,357 initial/invariant frozen
tensors,47 initial/128 disabled semantic events,full training replay and guards
pass. Auxiliary mean.090883,valid fraction.997276,341 reported masked rows;
sampled row identities are not independently logged. All489 totals pass, maximum
error.0001106. Only B16/micro16 is GPU-qualified here. Original CPU centered-CDP/
BPTT fixture failure is preserved; corrected fixture and all GPU stages pass.

Source07873e7c/native SHAaeecf9e7; runtime`/x/Temp/kindle-prior-continuation-20261104/runtime`.
Main/collector succeed07:51:09/07:51:19 UTC November4. Independent1.881s closure
completes07:51:43, verifying66 artifacts; separate.018s auxiliary arithmetic passes.
No bounded failure or GPU repeat. Closure`complete_with_cpu_fixture_repair`.
Root`/x/Temp/kindle-prior-continuation-20261104` is closed through the bounded check;
cache/runtime/evidence retained. No learner or GPU allocation was active at that
closure. Retain143.667 greedy/70.750 sampled,all0/24 two-wall.
Original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal-weight ranking diagnostic](2026-11-04-cdp-breakout-terminal-weight-ranking.md)
finds posterior AUC.952691→.946615, difference−.006076 with lane95% interval
[−.040365,.028646]; no demonstrated improvement or equivalence. One-step prior
AUC.791667→.696615 has difference−.095052[−.188379,−.013455]. Posterior162/prior639
pair orders change; this is not a pure monotone shift of control outputs.

The fixed analytic inverse of weight10 preserves all rankings and reduces false
below-.5 outputs from11→3/96 posterior and23→2/96 prior. It still leaves higher
intermediate-life MSE than control (.026632 versus.003178 posterior;.022044
versus.001530 prior). Terminal error rises versus raw weight10 but remains below
control. No calibration fit or threshold search occurs. The outcome-conditioned
control-policy cohort cannot establish population calibration or a causal gameplay
bottleneck; transforming saved outputs does not change the learned actor.

All24 CPU tests and1,125 pins pass. One2.060s single-core CPU analysis completes
07:27:01 UTC November4/source db4840bd/PID1144889. Independent4.073s closure
completes07:27:22, verifying16 artifacts,13,824 pair orders,240 transformed outputs
and all arithmetic. No failure, repair, repeat, model/emulator execution or GPU
work occurs. Root`/x/Temp/kindle-terminal-weight-ranking-20261104` is closed.

Retain143.667 greedy/70.750 sampled over weight10's120.542/66.375, all0/24 two-wall.
No correction, loss, learner or automatic successor is selected. No learner or
GPU allocation is active at closure; check live state. Do not repeat completed
captures or learners. Original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal-weight arrival diagnostic](2026-11-04-cdp-breakout-terminal-weight-arrivals.md) finds terminal
posterior MSE.644558→.054219, paired difference−.590339 with lane interval
[−.718137,−.451811]. All24 terminal errors improve; below-.5 outputs rise5/24→22/24.
Terminal prior MSE.911006→.423453 also improves. But intermediate-life posterior
MSE.003178→.099410 and prior.001530→.153795 worsen, with false below-.5 outputs
0→11/96 posterior and0→23/96 prior. All eight lane means show both tradeoffs.

The targeted loss changes predictions on the intended terminal states, while
increasing false ending indications at other life losses. The outcome-conditioned
control-policy cohort does not establish population calibration or a causal
explanation for gameplay. Remaining-life information, discrimination, probability
levels and use of continuation in imagined behavior remain unresolved. Retain
143.667 greedy/70.750 sampled over weight10's120.542/66.375, all0/24 two-wall.
No new loss, correction, weight sweep, learner or automatic successor is selected.

One50.596s frozen capture re-observes43,424 actions/173,660 frames,zero updates,
357 invariant tensors. All120 source/repeated queries,pixels/actions/rewards/
frames/resets and guards pass. F64/native prior error≤1.606e−7; deterministic
features match exactly. Posterior outputs are offline estimates.16 CPU tests,
1,085 pins and.272s CPU analysis pass. Sourceba1ae856/native3672a21a unchanged;
main succeeds07:13:01 UTC November4. Independent4.290s CPU closure completes
07:14:59, verifying44 artifacts and all arithmetic. No failure, repair or repeat.

Root`/x/Temp/kindle-terminal-weight-arrivals-20261104` is closed. No learner or
GPU allocation is active at closure; check live state before work. Do not repeat
completed captures, analyses or learners. Original864/90%-two-wall and five-game
goal remain unfinished.

The [completed terminal-weight10 comparison](2026-11-04-cdp-breakout-terminal-weight.md) scores120.542 greedy/66.375
sampled versus143.667/70.750 retained, all0/24 two-wall. It finishes200k actions/
799,593 frames/12,477 updates in21m37.140s,+.72% time. Greedy gain/min falls16.94%;
paired difference−1.116550 has interval[−3.490321,1.371544], so reliable harm or
equality is not established. The locked improvement rule fails. Retain existing
sampler/continuation1 with terminal weighting off; no sweep, second learner,
extension or automatic successor. Original864/90%-two-wall and five-game goal
remain unfinished.

All370 current/338 retained pins,100k/6,227 recovery,357 initial/invariant frozen
tensors,47 pre-update events,799/794 actual-initial semantic events, finite losses,
full pixel/action/label and frozen replays/videos,guards,coverage,curves and
comparison pass. Terminal exposure5,257/3,194,112 rows versus5,376 retained;
8,250 versus8,105 updates contain no ending. Late expert CE.546991 versus.560514
on own replay does not yield a gameplay gain. Actual overrides43.4255% and guided
194.642 returns are not frozen competence. The separate matched-arrival diagnostic finds lower ending error but worse intermediate-life predictions.

Source8ee590c5/native3672a21a/SHAa8b483a3; same12M64/Auto/N8/B16/T16/H15/R16/
replay-value1/expert100 and retained rates/guidance. Only true terminal continuation
rows receive10; continuing rows1, unchanged sampler/global continuation1. Enabled
stable-logit BCE,native/runner/wrapper versions and later experience differ from
retained447fe50; no isolated coefficient, calibration or reliability claim.
Main succeeds07:02:09 UTC November4,collector07:02:43; independent3.346s CPU
closure completes07:03:46, verifying53 artifacts and the decision. All services
succeed; no full-stage failure/repair or GPU repeat. Earlier bounded CPU failures
remain preserved. CPU co-host costs stay included without correction.

Root`/x/Temp/kindle-terminal-weight-20261104/full` is closed. No learner or GPU
allocation is active at closure; check live state before future work. Do not
restart completed stages. Retain runtime, checkpoints, logs, videos and build cache.

The [terminal-only continuation check](2026-11-04-cdp-breakout-terminal-weight.md) passes8,192 actions/32,722 frames/489 updates in46.934s.
Conservative200k projection21m00.438s passes30min; gameplay benefit is unmeasured.
Weight10 applies only to true terminal continuation rows, with continuing weight1,
ordinary row denominator, existing sampler and global continuation1. Keep12M64/
Auto/N8/B16/T16/H15/R16/replay-value1/expert100 and retained guidance/rates.
Mixture and open-loop supervision stay off. The stable-logit path also differs
from older probability BCE; no isolated coefficient or calibration claim.

Native3672a21a/SHAa8b483a3 passes181 Rust/5 gym CPU checks,1,046 packaged Python
checks, strict Clippy and10 guarded F64 loss/gradient cases. Maximum loss error
3.94e−7/derivative1.08e−8; continuing derivatives stay unchanged by the weight.
Default-off1,024-action compatibility matches128 events/357 invariant tensors.
Fresh357 initial tensors/47 events, finite reports, cadence, changed parameters,
full pixel/action/label replay and guards pass. Frozen1,024 actions take.882s,
zero updates/no natural episodes. Terminal exposure345/125,184 rows versus322
retained; no replay mixture. Actual overrides45.4956% are not frozen competence.

Main/collector succeed06:29:25/06:29:36 UTC November4; separate1.782s CPU closure
completes06:30:07, verifying66 artifacts and298 current/338 retained pins. Original
Python import, numerical-reader and invalid-argument-helper failures remain;
qualified path/helper repairs pass, with no GPU repeats. Closure is
`complete_with_cpu_preflight_repairs`. Root`/x/Temp/kindle-terminal-weight-20261104`;
matching runtime/wheel/test executable and build cache remain. Do not rerun stages.
No full learner, sweep, extension or automatic successor is selected. No learner
or GPU allocation is active at closure; check live state before work. Retain70.750
sampled/143.667 greedy; original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal ranking check](2026-11-04-cdp-breakout-terminal-ranking.md) finds posterior AUC
.952691→.897569, paired difference−.055122 with lane interval[−.141927,.008257].
Prior AUC falls .791667→.734809, interval[−.205306,.085938]. Both intervals cross
zero; neither decline nor equality is established on this reused cohort. There
are261 posterior/707 prior reversed class-pair orders, with no ties. A strict
monotone output shift alone cannot fully describe the change; no correction or
new training setting is selected.

The prior arrival result remains: better game-ending predictions, but more false
ending indications at intermediate life losses. Retain the existing sampler,
continuation1 and70.750 sampled/143.667 greedy; mixture65.708/85.250 remains
weaker, all0/24 two-wall. Remaining-life information and use of continuation in
imagined behavior remain unresolved, without a causal gameplay claim.

One2.082s CPU-only analysis,11 tests and1,097 pins pass. Source9deeeb19/process
1118658 completes06:03:28 UTC November4. The independent verifier recounts9,216
pair orders and saves its summary, then exits1 on a missing import in final output.
The original failure remains; separate1.992s CPU repair completes06:04:44, verifying
17 artifacts without repeating analysis or arithmetic. Closure is
`complete_with_cpu_output_repair`.

Root`/x/Temp/kindle-terminal-ranking-20261104` is closed. No model/emulator/GPU
execution or learner update occurs. Do not rerun analysis, closure or repair;
check live state before future work. No learner or GPU allocation is active.
The original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal-mixture arrival diagnostic](2026-11-04-cdp-breakout-terminal-mixture-arrivals.md) finds a tradeoff:
terminal posterior MSE falls .644558→.252775, paired interval[−.506700,−.274697],
and below-.5 predictions rise5/24→17/24. Terminal prior mean falls .952850→.512441.
But intermediate-life prior MSE rises .001530→.170491, interval[.106436,.241616],
with21/96 below.5 versus0/96; posterior error also worsens. This selected cohort
shows better terminal recognition alongside more false ending indications, not
population calibration or a causal explanation for weaker gameplay.

Retain the existing sampler/continuation1 and70.750 sampled/143.667 greedy;
the completed mixture remains65.708/85.250, all0/24 two-wall. Distinguishing lost
lives from game endings is the next diagnostic question. No new learner, loss,
threshold, mixture setting or automatic successor is selected.

The51.327s frozen candidate-only capture completes43,424 actions/173,660 frames,
zero updates. Sixteen CPU tests,1,052 pins,357 invariant tensors,120 exact endpoint/
repeated-query checks, full pixel/transition parity and guards pass. F64/native
prior error≤2.104e-7; posterior outputs remain reference estimates. Source6dda986d;
main succeeds05:53:02 UTC November4, independent4.265s closure05:53:33 with52
artifacts. No failure, repair or completed-stage repeat occurs.

Root`/x/Temp/kindle-terminal-mixture-arrivals-20261104` is closed. No learner or
GPU allocation is active. Do not repeat its capture, analysis or closure; check
live state before future work. Native2b5a3eea differs from control447fe50 and the
cohort is reused. Original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal-replay comparison](2026-11-04-cdp-breakout-terminal-mixture.md) increases ending exposure
9.43× but lowers frozen Breakout: sampled70.750→65.708 and primary greedy
143.667→85.250, all0/24 two-wall. The fresh200k/12,477-update learner takes
21m19.292s versus retained21m27.910s. Greedy gain/min falls40.90%, with paired
difference interval[−4.987,−.700]. Retain the existing sampler and continuation1;
the optional mixture stays off. Original864/90%-two-wall and five-game goal remain
unfinished. No extension, mixture sweep, second learner or automatic successor
is selected.

Terminal rows rise5,376→50,706; updates without endings fall8,105→161. Late
expert CE falls.560514→.480124 without gameplay gain. All objectives receive the
changed replay distribution without importance correction. The subsequent arrival
diagnostic above measures a selected-cohort tradeoff; population calibration remains
unmeasured. Native, runner and guidance-wrapper sources differ
from retained447fe50. Scoped policy-AST/CPU fixtures support the active full18 path,
not byte equality or broad trajectory parity; reused1009/cohort limits reliability.

All343 current/338 retained pins,100k/6,227 read-only recovery,357 matching initial/
invariant frozen tensors,799/794 actual-initial semantic events, full training and
frozen replays/videos, guards, exposure, coverage, curves and comparison pass.
Main succeeds05:43:42 UTC November4/sourcef5540278; original collector succeeds
05:43:58 with52 artifacts and an independent exposure recount. All three services
are terminal success; no full-stage failure, repair or repeat. Earlier bounded
failures/repair remain preserved. CPU co-host time remains included without correction.

Root`/x/Temp/kindle-terminal-replay-20261104/full` is closed. No learner or GPU
allocation remains active. Check live state before future work; do not restart
completed training, assessments, replays or collectors.

The [completed fixed terminal-state readout](2026-11-04-cdp-breakout-terminal-readout.md) finds retained posterior
ending information: held-out balanced MSE.131485 versus saved head.323868 and
balanced constant.248501, paired interval[−.251523,−.130600]. It detects20/24
endings but falsely ends10/96 continuing life losses at gamma/2. The saved head
has higher posterior AUC.952691 versus.917752, with fewer detections andzero
false positives. All four fixed views pass the diagnostic screen; primary stays
retained posterior. This supports usable information on the selected cohort,
not calibrated acting probabilities, a replacement head or gameplay gain.

One4.368s CPU-only analysis,17 tests and887 pins pass. Independent4.242s closure
verifies32 saved folds without refitting, maximum normal-equation residual2.010e-14,
and all19 final artifacts. Sourceb8f670c8; analysis completes04:36:18 UTC November4,
closure04:37:28. No failure, repair or completed-stage repeat. Root
`/x/Temp/kindle-breakout-terminal-readout-20261104` is closed; no native model,
emulator, GPU work or learner update occurs. Do not repeat its fit or closure.

Retain scale1 and70.750 sampled/143.667 greedy. No new learner, loss, sampler,
inference correction or automatic successor is selected. Check live state before
work. Original864/90%-two-wall and five-game goal remain unfinished.

The [completed continuation10 arrival comparison](2026-11-04-cdp-breakout-continuation10-arrivals.md) finds observed-terminal
continuation.759636→.730320, both5/24 below.5, against target0. Primary paired
MSE difference−.043530 has interval[−.212505,.123448]; no improvement is established.
Prior terminal.952850→.938323 also has an interval spanning zero. Intermediate-life
prior MSE.001530→.009264 worsens, paired interval[.002367,.013707]. This fixed
outcome-conditioned cohort does not establish population calibration or explain
the gameplay regression. Retain scale1 and70.750 sampled/143.667 greedy.

The candidate-only capture takes50.752s for43,424 actions/173,660 frames,zero
updates. All841 pins,357 invariant tensors,120 source/repeated-query checks,full
pixel/transition parity and guards pass. F64 prior/native error≤1.398e-7; posterior
outputs remain reference estimates. Fifteen CPU tests and independent4.224s
closure pass; all52 final artifacts verify. Source91924e08; main succeeds04:26:05
UTC November4, closure04:27:01. No failure, repair or completed-stage repeat.

Root `/x/Temp/kindle-breakout-continuation10-arrivals-20261104` is closed. No learner
or GPU allocation is active; check live state before work. Do not repeat its
capture, analysis or closure. No new learner, coefficient sweep, loss, sampler or
automatic successor is selected. Original864/90%-two-wall and five-game goal remain
unfinished.

The [completed200k continuation10 comparison](2026-11-04-cdp-breakout-continuation10.md#full200k-comparison)
lowers primary greedy143.667→55.833 and sampled70.750→35.250, all0/24 two-wall.
Training200k actions/799,592 frames/12,477 updates takes21m28.116s, only+.016%
time. Greedy gain/min falls62.10%; paired return interval[−125.171,−49.498] and
gain/min interval[−5.832,−2.306] are negative. Retain continuation scale1 and the
existing native447fe50/12M64/B16/T16/H15/R16/replay-value1/expert100 recipe.

Late own-replay continuation BCE.022457→.022325 and expert CE.560514→.542827
fall slightly without gameplay benefit. These changing-state summaries do not
establish terminal-specific calibration or explain the regression. Actual
overrides44.825%; native score216–431 exposure689 versus1262, neither learner
reaches432. No isolated mechanism or seed-reliability claim on reused1009/cohort.

All761 current/338 retained pins,100k/6227 recovery,357 matched initial/invariant
frozen tensors,47 pre-update and799/794 initial semantic events, full replays/
videos, guards, curves, exposure and comparison pass. Source6a05eb02; original
main/collector succeed04:13:52/04:13:59 UTC November4. All three services and56
completion artifacts pass; no experiment-stage failure, repair or repeat occurs.
Separate3.910s read-only report review completes04:15:30. CPU co-host costs remain.

Root `/x/Temp/kindle-breakout-continuation10-20261104/full` is closed. No learner
or GPU allocation is active; check live state before work. Do not restart its
stages or collectors. No extension, coefficient sweep, second learner or automatic
successor is selected. Original864/90%-two-wall and five-game goal remain unfinished.

The [completed Breakout arrival-head diagnostic](2026-11-04-cdp-breakout-arrival-head.md) lowers mean terminal
continuation from.952850 before observation to.759636 after the actual arrival,
against target zero. Only5/24 observed endings fall below.5;20 errors decrease
and4 increase. Paired MSE difference−.266448 has lane interval[−.401298,−.154655].
All eight lane means improve, but substantial observed-terminal error remains.
Intermediate-life MSE rises.001530→.003178, with mean.979477→.979466.

Posterior outputs are offline F64 head estimates. All120 prior estimates match
native within1.178e-7; deterministic states match exactly, while categorical draws
and observation conditioning differ. This outcome-conditioned cohort does not
establish calibration or isolate representation,rare-example fitting or forecast
optimization. No new loss,sampler,coefficient or learner is selected.

The50.457s frozen capture completes43,424 actions/173,660 frames withzero updates.
All634 pins,357 invariant tensors,120 saved/repeated predictions,full pixel and
transition parity,13 CPU tests andhost guards pass. Source8754f65d,
main1102754/invocationc114e5e2e92442ebbf08136138683b22 succeeds03:28:26 UTC
November4. Independent3.691s closure completes03:28:57;all52 final artifacts
verify,without failures,repairs or repeated stages.

Root `/x/Temp/kindle-breakout-arrival-head-20261104` is closed. No learner or GPU
allocation is active;do not restart its capture or collectors. Check live state
before work. Terminal-state fitting is a concrete learning question; no successor
allocation is selected. Retain70.750 sampled/143.667 greedy and20% episode
guidance. The original864/90%-two-wall andfive-game goal remain unfinished.

The [completed Breakout terminal replay audit](2026-11-04-cdp-breakout-terminal-replay.md) finds5,376 terminal
exposures among3,194,112 replay rows (.1683%), from263 observed game endings.
Terminal targets are present in the existing continuation loss. In the final50k
window,76.352% of updates contain no terminal rows;205 terminal-containing
batches have a conservative mean prediction bound below.5,10 above.5 and524
remain unresolved. The final-window aggregate terminal mean lies in
[.106423,.669123], not an exact estimate or confidence interval. Some replay
fitting is established; its changing posterior states differ from the frozen
one-step prior diagnostic, whose mean terminal prediction remains.952850.

The CPU-only analysis takes.493s;16 tests, all606 pins,12,477-report independent
Decimal recount,263 terminal/episode joins and22 final artifact hashes pass.
Sourcebf2011ee,analysisPID1102416;3.710s closure completes03:17:31 UTC November4
without failures,repairs or repeated stages. Root
`/x/Temp/kindle-breakout-terminal-replay-20261104` is closed. Native,emulator,
checkpoint and training remain unchanged; no GPU work or new learning occurs.

A paired final-model prediction before/after observing the same arrival is the
next diagnostic question; no new allocation,loss,sampler or learner is selected.
No learner or GPU allocation is active. Do not repeat completed stages; check
live state before work. Retain70.750 sampled/143.667 greedy and20% episode
guidance. The original864/90%-two-wall and five-game goal remain unfinished.

The [completed Breakout ending-feedback diagnostic](2026-11-04-cdp-breakout-ending-feedback.md) lowers mean
terminal-transition continuation from .996255 at nine steps to .952850 at one
step after actual observations, while the target is zero. All 24 terminal errors
fall; paired MSE difference −.081526 has lane interval [−.112610,−.053525]. The
96 intermediate-life predictions also fall, .997175→.979477 against target
.996997, so this is not a general forecast improvement. Weak ending prediction
persists at short range. Continuation on actual posterior arrivals and its
class-specific training fit remain unmeasured; no causal defect is identified.

The unchanged native447fe50/checkpoint completes 43,424 actions /173,660 frames /
1,080 queries in52.878s with zero updates. All 554 pins, 357 invariant tensors,
full pixel/transition/trajectory parity, 120 saved-prefix/actor-probability
matches, 120 repeated endpoint queries and host guards pass. Eighteen CPU checks
pass. Main1101025/invocationc782fcb701a646e7991130fabef5e24e/source714bb790
succeeds02:54:31 UTC November4; independent3.434s closure completes02:56:00.
All 39 final artifacts verify, with zero failures, repairs or repeated stages.

Root `/x/Temp/kindle-breakout-ending-feedback-20261104` is closed. No learner or
GPU allocation is active; do not restart its capture or collectors. Check live
state before work. Observations, horizon and later diagnostic draws change
together; this outcome-conditioned reused cohort does not establish population
calibration. No native change, loss, learner or automatic successor is selected.
Retain70.750 sampled/143.667 greedy and20% episode guidance. The original
864/90%-two-wall and five-game goal remain unfinished.

The [completed Breakout rescue forecasts](2026-11-04-cdp-breakout-rescue-forecasts.md) favor the surviving sequence
in 12 of 18 game-over comparisons, but predict mean continuation mass .971526
on 24 recorded endings, close to discount-only .973295 rather than actual zero.
The paired lane interval [33.33%,100%] is wide. Reward plus endpoint value favors
64 of 92 short rescues; all 240 nine-action branches have zero external reward.
Partial discrimination and weak ending forecasts do not identify a causal defect.

The unchanged native447fe50/checkpoint completes 43,424 actions /173,660 frames
in 52.471s with zero updates. All 506 pins, 357 invariant tensors, full input/output
pixels and transitions, 120 repeated forecasts, 19 identical-plan pairs and host
guards pass. Scalar posterior draws differ from the original vector policy;
continuation means game-over, not intermediate life loss. Twelve CPU checks pass.
Main1100615/invocationcdcca768df77488db2db2ec90726a4a2/sourcee40b7acd succeeds
02:37:57 UTC November4. Independent recount passes; its own-stdout manifest
error is preserved and separately repaired at02:40:46, with all 43 final artifacts
verified. Closure`complete_with_cpu_manifest_repair`; no GPU stage, analysis or
recount repeats.

Root `/x/Temp/kindle-breakout-rescue-forecasts-20261104` is closed. No learner or
GPU allocation is active; do not restart the capture, oracle or collectors. Check
live state before work. The next diagnostic question is whether short forecasts
with intervening real observations recognize the same endings; no new allocation,
loss or learner is selected. Retain70.750 sampled/143.667 greedy and20% episode
guidance. The original864/90%-two-wall and five-game goal remain unfinished.

The [completed200k block-guidance comparison](2026-11-04-cdp-breakout-block32.md) scores74.875 greedy/41.250
sampled, below retained143.667/70.750. Training200k actions/799,610 frames/
12,477 updates takes21m11.408s (−1.28% time), but primary greedy gain/min falls
47.96%. Paired greedy return interval[−104.542,−38.625] and gain/min interval
[−4.826,−1.759] are negative; all0/24 two-wall. Retain20% whole-episode guidance.
No extension, second learner, probability/block-length sweep or automatic successor.

The candidate chooses50% of32-action blocks; actual overrides51.493%. Native
score216–431 exposure is468 actions versus1262 retained, and neither reaches432.
Late own-replay expert CE.552 versus.561 does not yield frozen improvement.
Probability and temporal distribution change together, with different experience;
no isolated mechanism or training-seed reliability claim. Teacher/native447fe50,
12M64/Auto/N8/B16/T16/H15/R16/replay-value1/expert100 remain fixed.

All126 current/338 retained pins,100k/6227 recovery,357 matched initial/invariant
frozen arrays,47 bounded pre-update and799/794 initial semantic events, full
replays/videos, guards, curves, block accounting, exposure and comparison pass.
Sourceb07ebbcc; original main/collector succeed02:15:09/02:15:35 UTC November4.
All three services succeed, with zero failures or repeated GPU stages. Bounded
CPU qualification failures remain preserved in the report. Root
`/x/Temp/kindle-breakout-block32-20261104/full` isclosed. No learner or GPU allocation
is active; check live state before work. Do not restart this learner or collectors.
Original864/90%-two-wall and five-game goal remain unfinished.

The [completed CPU teacher-takeover comparison](2026-11-04-cdp-breakout-command-takeovers.md) finds32-action survival
84/120 with the existing teacher versus79/120 with the saved command-dependent
predictor. Net−5 rescues (−4.17 percentage points), onlytwo improved lanes, fails
the fixed12-net/five-lane screen. Retain the existing teacher; no integration,
coefficient/window retry or learner isselected. Its older fresh full-game screen
remains failed. This outcome-conditioned student-state test is not a full-game
score, seed-reliability claim or causal architecture diagnosis.

The existing teacher avoids the original loss endpoint in92/120 cases, candidate
89/120; both survive32 decisions in67, existing-only17, candidate-only12, neither24.
All120 origins,240 teacher/120 recorded branches, eight reverse-order checks,
full43,424-action/173,660-frame source replays and independent recount pass.
Policies receive pixels/executed commands only; no life-count input or GPU work.

Annotation20.830s andbranch47.276s complete under the fixed caps. Corrected
controller succeeds01:24:46 UTC November4/source16ddeae1; separate1.406s recount
andclosure complete01:25:18. All470 original/490 repair pins pass. The original
pre-replay import failure remains; exact cohort extraction passesfour repair
checks and changes no policy, method or cap. Closure`complete_with_cpu_import_repair`;
original controller stays failed, withno completed-stage repeat.

Root`/x/Temp/kindle-breakout-command-takeovers-20261104/cpu-import-repair` isclosed.
No learner or GPU allocation isactive. Existing supervision still rescues many
current-policy failures; how to turn it into stable learned control remains open.
No automatic successor isselected. Check live state before future work. The
original864/90%-two-wall and five-game goal remain unfinished.

The [completed trained-policy posterior comparison](2026-11-04-cdp-breakout-value1-modal.md) scores modal175.708
versus sampled143.667 with greedy actions in both arms (+22.30%). Paired lane
return difference+32.042 has95% interval[−25.750,78.959]; all0/24 two-wall.
The preset improvement rule fails: retain sampled posteriors. This reused model/
cohort does not establish a reliable modal gain, a global inference default or
an isolated mechanism. No learner or initial-model GPU assessment runs; original
200k/12,477-update training stays21m27.910s, withno learning-gain/min claim.

All431 current/69 qualification/338 retained pins,357 finite/invariant tensors
per arm,5,478 exact default semantic events and final counters/tails, full replays/
videos and host guards pass. Two arms total87,968 actions/351,799 frames/56.610s,
zero updates/truncations. Minimum14.526GiB headroom across33,050 snapshots passes.
Original main/collector succeed01:06:38/01:06:43 UTC November4/source2afd0dc5;
all25 completion artifacts verify, withno GPU or completed-stage repeat.

Root`/x/Temp/kindle-breakout-value1-modal-20261104` isclosed. Both services are
terminal success; no GPU allocation remains active. The older four-arm modal
assessment remains protocol-failed; its saved initial tensors match on CPU,
without assigning an accepted modal initial score. No retry, initial assessment,
inference sweep or automatic successor isselected. Check live state before
future GPU work. The original864/90%-two-wall and five-game goal remain unfinished.

The [completed encoder8e-5/current-critic comparison](2026-11-04-cdp-breakout-value1-encoder8e5.md) finishes200k
actions/799,555 frames/12,477 updates in21m13.443s (−1.12%). Greedy103.583
falls27.90% below retained143.667; sampled74.000 rises4.59% from70.750.
All0/24 two-wall. Greedy gain/min falls27.52%, paired difference−1.814
[−4.055,1.152]; sampled difference+.190[−.534,1.026]. Both intervals span zero.
The fixed improvement rule fails: retain encoder4e-5 and the existing recipe.
Only encoder rate changes; native447fe50/runnercf4ea238/teacher stay fixed.
Earlier neighboring-recipe benefit does not carry over in this development
comparison; no isolated mechanism, reliable harm or seed-reliability claim.

All120 original/12 CPU-repair/338 retained inputs,100k recovery,357 matched
initial/invariant frozen tensors,47 pre-update and799/794 initial evaluation
semantic events, full replays/videos, guards, curves, exposure and comparison
pass. Minimum11.159GiB headroom across75,262 snapshots clears2GiB. Late expert
CE improves.560514→.548159 without primary gameplay gain. Original main/collector
succeed00:45:07/00:45:34 UTC November4/source6543be61; all services are terminal
success, withno GPU or completed-stage repeat. The pre-launch CPU fixture
failure and separate correction remain preserved; original receipts stay intact.

Root`/x/Temp/kindle-breakout-value1-encoder8e5-20261104/full` is closed; no GPU
allocation remains active. Do not restart completed stages. No extension,
confirmation, rate queue or automatic successor isselected. Check live state
before future work. The original864/90%-two-wall and five-game goal remain unfinished.

The [completed RGB frame-pair comparison](2026-11-03-cdp-rgb-frame-pair.md#full-200k-comparison-retains-single-image-input)
finishes200k actions /799,579 frames /12,477 updates in24m17.046s (+13.13%).
Greedy114.000/sample39.208 fall20.65%/44.58% below retained143.667/70.750;
all0/24 two-wall. Greedy gain/min falls29.68%, paired difference−1.956
[−3.884,.191]; the primary interval spans zero. Retain single-image input and
the existing recipe; no extension, confirmation orautomatic successor isselected.
Late labelCE improves.560514→.539096 without a frozen gameplay gain.

All114 current/338 retained pins,100k/6227 recovery,357 bounded-matching initial/
invariant frozen tensors,47 pre-update events, full replays/videos, guards,
curves, exposure and comparison pass. Own initials1.542 sampled/1.458 greedy
differ from the three-channel control. Actual minimum GPU headroom5.019GiB passes
2GiB across75,250 snapshots. Main/collector succeed00:02:43/00:03:10 UTC November4,
source8ea32681; all services are terminal success, with no full-stage or GPU repeat.
Original numerical and CPU qualification/status/report failures remain preserved.

Geometry/initialization, native, runner and teacher/schedule source differ from
control; full18 teacher defaults agree, but no isolated effect or seed-reliability
claim. Root`/x/Temp/kindle-rgb-frame-pair-20261103/full` isclosed. No GPU allocation
remains active; check live state before work. The original864/90%-two-wall gate
and five-game goal remain unfinished.

The [completed50% episode-guidance Breakout comparison](2026-11-03-cdp-breakout-guidance50.md)
finishes200k actions /799,731 frames /12,477 updates in21m37.887s (+.77%).
Greedy50.583/sample34.417 fall64.79%/51.35% from143.667/70.750; all0/24 two-wall.
Greedy gain/min falls66.07%, paired difference−4.354[−6.127,−2.718]; sampled
falls52.77%. Own initials2.208/1.417 match. Retain20% guidance andthe existing
B16/T16 recipe. Actual overrides rise43.0585%→73.15%; late labelCE improves
.560514→.541246, without gameplay benefit. Only1,774 native actions occur at
scores216–431; none reaches432. This reused1009/cohort result does not isolate
a mechanism or establish seed reliability.

All117 original/338 retained pins,100k/6,227 recovery,357 matching initial/
invariant frozen tensors,47 bounded/799–794 initial events, full replays/videos,
guards, curves, exposure andcoverage pass. Original main/collector fail
22:45:58/22:46:28 UTC November3/source2e1e70de on a redundant guidance-header
assertion. Separate4.137s CPU correction completes22:47:34.594, preserving98
artifacts and20 repair pins; an earlier null-journal-message wrapper failure
isretained. No GPU orsuccessful CPU stage repeats. Closure is
`complete_with_cpu_audit_repair`; original services remain failed. Full root
`/x/Temp/kindle-breakout-guidance50-20261103/full`, separate correction under
`cpu-comparison-repair`. No extension, probability sweep, confirmation orautomatic
successor isselected. No Kindle allocation isactive atclosure; check live state.
The original864/90%-two-wall gate andfive-game goal remain unfinished.

The [50% episode-guidance Breakout check](2026-11-03-cdp-breakout-guidance50.md) passes
8,192 actions /32,746 frames /489 updates in **48.271s**,1.22% above the retained
20% guidance check. Padded200k projection **21m36.346s** clears30min. Frozen1024/
.891s has zero updates, no natural episodes and357 invariant tensors. All56
current/338 retained pins,357 matching initial tensors, first-episode selection,
finite reports, full pixel/action/label replay and both guards pass. Actual
interventions are6,272/8,192 (76.5625%); guided episode probability is.50, not
that action fraction. Teacher,1920-action cap, labels, B16/T16 recipe and
native447fe50/runnercf4ea238 stay fixed. No gameplay benefit is established.

CPU qualification passes2.236s with30 rejection cases and zero native agent/
emulator constructions. Main/collector succeed22:11:15.870/22:11:22.558 UTC
November3/source7f711b49; no stage fails or repeats. Root
`/x/Temp/kindle-breakout-guidance50-20261103/check` is closed; do not rerun it.
No full learner, probability sweep, extension or automatic successor is selected.
No Kindle allocation remains active at closure; check live state before work.
Retain70.750 sampled/143.667 greedy; original864/90%-two-wall andfive-game goal
remain unfinished.

The [completed current-critic B8/T32 Breakout comparison](2026-11-03-cdp-breakout-value1-context32.md#full-200k-comparison)
finishes 200k actions /799,554 frames /12,469 updates in **23m06.548s**,7.66% more
training time. Primary greedy **116.667** falls 18.79% from 143.667; sampled
**94.583** rises 33.69% from 70.750. All 0/24 two-wall; own initials 2.208/1.417 match.
Greedy gain/min falls 24.84%, paired difference −1.637 [−3.928,.686]; sampled
rises 24.82%, difference +.802 [−.517,2.496]. Both intervals span zero. The fixed
primary improvement rule fails; retain B16/T16 and the existing recipe. No
reliable harm, sampled advantage, isolated memory effect or seed reliability
is established.

All 122 current/338 retained pins, 100k/6,219 recovery, 357 matched initial/invariant
frozen tensors, 63 bounded/47 shared pre-update and 799/794 actual-initial semantic
events, full replays/videos, guards, curves, exposure and comparison pass.
Native 447fe50/runnercf4ea238 stay unchanged. Actual overrides 46.948%; late expert
CE .610334 versus .560514 and 1,598 native actions at scores 216–431 are descriptive,
not causal evidence. No training action reaches 432. Main succeeds 21:58:08.786 UTC
November 3/source06c873c2; terminal collector succeeds 21:58:53.098 in 2.466s.
All three services succeed; no full stage fails or repeats. Bounded CPU failures
remain preserved. Root `/x/Temp/kindle-breakout-value1-context32-20261103/full`
is closed; do not restart or repeat observers. No extension, confirmation,
context sweep or automatic successor is selected. No Kindle allocation remains
active at closure; check live state before work. The original 864/90%-two-wall
and five-game goal remain unfinished.

The [current-critic Breakout B8/T32 check](2026-11-03-cdp-breakout-value1-context32.md) passes8192 actions /32,722 frames /
481 updates in51.909s; padded200k projection23m14.028s clears30min. Frozen1024/
.875s haszero updates/no natural episodes. All357 initial/invariant tensors,
47 shared prefix events,full pixel/action/label replay andboth guards pass.
Native447fe50/runnercf4ea238 remain unchanged; behavior8e-5/replay-value1 fixed.
Original main/collector fail21:16:10 UTC November3/source9de712d1 on the replay
manifest path. Separate9.605s CPU repair closes21:19:14.395,readback21:20:20.243,
with67 original/7 repair pins and39 outputs unchanged; no GPU orsuccessful replay
repeats. Earlier CPU schema-fixture failure remains preserved. Closure is
`complete_with_cpu_audit_repair` under`/x/Temp/kindle-breakout-value1-context32-20261103/cpu-repair`.
No full learner orautomatic successor isselected; no allocation remains active.
Retain70.750/143.667; gameplay benefit andthe five-game goal remain unfinished.

The [completed four-action Breakout comparison](2026-11-03-cdp-breakout-minimal-guided.md#full-200k-comparison) finishes
200k actions /799,580 frames /12,477 updates in **18m36.686s**,13.29% less training
time than full18. Sampled **55.958** andgreedy **91.500** fall20.91%/36.31% from
retained70.750/143.667; all0/24 two-wall. Own initials1.250/4.667 differ fromfull18.
Greedy initial-subtracted gain/min falls29.20%,with paired difference−1.925
[−3.907,−.192]. Both primary replacement andcheaper-budget selection rules fail.
**Retain full18 andthe existing200k recipe.** This is one reused development
seed/cohort,not a training-seed reliability orisolated mechanism result.

All169 current/338 retained pins,read-only100k/6,227 recovery,357 bounded-matching
initial/invariant frozen tensors,47 pre-update events,full pixel/action/label
replays/videos,guards,curves,exposure andcomparison pass. Actual overrides42.116%;
late expert CE.567343 andonly1,222 native actions at scores216–431 do not establish
a causal limitation. No training action reaches432. Main succeeds20:59:55.332 UTC
November3/source1917a4de; terminal collector succeeds21:00:26.026 in2.408s.
All three services succeed; no full-run stage fails orrepeats. Earlier bounded
preflight andCPU collector failures remain preserved. Native447fe50 unchanged;
root`/x/Temp/kindle-breakout-minimal-guided-20261103/full` isclosed. Do not restart
it orrepeat its observers. No extension,confirmation orautomatic successor is
selected. No Kindle allocation remains active; check live state before work.
The original864/90%-two-wall gate andfive-game goal remain unfinished.

The [four-action guided Breakout check](2026-11-03-cdp-breakout-minimal-guided.md) completes8192 actions /
32,728 frames /489 updates in **41.133s versus47.690s**,13.75% less early training
time. Including separate construction gives44.072s versus50.094s (−12.02%).
All118 inputs,357 finite/invariant tensors,correct cadence,full training pixel/
action/label replay andboth guards pass. Frozen1024/.918s haszero updates and
five natural episodes; no24-episode gameplay gain is established. Native447fe50
is unchanged; explicit adapter mapping preserves the full18 teacher internally.

Main completes20:21:17.038 UTC November3/source11023a19; terminal closure at
20:22:44.473 is `complete_with_cpu_terminal_repair`. The original preflight
accidentally initializes a native agent on synthetic pixels before failing with
zero emulator steps/updates; its unguarded work andcheckpoint remain preserved.
Repaired CPU qualification passes161 tests/zero agent constructions; source
qualification passes258 tests. A later CPU collector failure on cleared systemd
InvocationID is preserved and repaired using the original journal identity.
No declared learner/frozen/replay stage repeats. Root
`/x/Temp/kindle-breakout-minimal-guided-20261103/bounded-check` is closed; do not
rerun it. No active allocation orfull learner isselected;18m24.645s projected
cost clears30min,but competence and the original864/90%-two-wall gate remain
unmeasured/unfinished. Retain70.750/143.667 andthe existing five-game goal.

The [completed life-bounded actor comparison](2026-11-03-cdp-life-return-actor.md#full-200k-comparison-does-not-improve-the-retained-recipe)
finishes 200k actions / 799,589 frames / 12,477 updates in **21m41.269s** (+1.04%).
Sampled **62.417** and greedy **137.542** fall from retained70.750/143.667 by
11.78%/4.26%; all0/24 two-wall. Own initials1.417/2.208 match. Greedy gain/min
falls5.31%, with paired difference−.350 [−2.534,2.012]. The declared selection
rule fails; the interval spans zero and does not establish reliable harm.
**Retain the existing200k recipe.** Against the contextual whole-game actor,
sampled rises26.95% but primary greedy falls4.98%; that comparison also fails.

All330 current/338 retained pins, 100k/6,227 recovery, 357 initial/frozen tensors,
47 pre-update and799/794 actual-initial semantic events, full replays/videos,
guards, curves, exposure and comparisons pass. Independent replay verifies
1,288 raw life losses, including1,032 nonterminal boundaries. The loss has
613,108 positive repeated rows on11,932 updates. Late expert CE.543307 improves
without a frozen gain; native exposure at scores216–431 is only1,201 actions,
with none at432+. These are descriptive counts, not causal architecture evidence.

Original main completes **19:50:54.540 UTC November3**, source`e6e5f4ba`, and
terminal collector completes **19:51:02.459** in3.231s. Main, recovery and collector
all succeed; no full-run failure, repair, GPU/replay repeat or cap change occurs.
The preceding bounded CPU collection failure/repair remain preserved. Root
`/x/Temp/kindle-life-return-actor-20261103/full` is closed; do not restart the
learner or repeat its observers. The optional mode remains default-off in
qualified native`ec12d41e`.
No extension, coefficient sweep, confirmation or automatic successor is selected.
No Kindle allocation remains active at closure; check live state before work.
The original864/90%-two-wall gate and five-game goal remain unfinished.

The [optional life-bounded actor qualification](2026-11-03-cdp-life-return-actor.md)
passes 8,192 actions / 32,725 frames / 489 updates in **49.039s**, versus the
retained whole-game short check's 50.754s. Frozen 1,024 actions take .903s with
zero updates, no completed natural episodes and all 357 tensors unchanged.
All 123 raw-frame life-loss boundaries reproduce in independent pixel/action/
label replay. Targets become positive at action 568; 34,261 positive repeated
rows appear on 469 updates. These are execution results, not gameplay gains.

Default-off native `ec12d41e` passes 174 Rust / 1,044 packaged Python CPU tests,
strict Clippy and 12 F64 derivative cases. All 263 current / 338 retained pins,
214 preserved sources, 357 matched initial tensors and 47 pre-update events pass.
The original main service succeeds 19:15:26.762 UTC November 3. Its CPU collector
fails on a list/dictionary manifest mismatch; a separate 1.213s repair closes
19:16:56.676 as `complete_with_cpu_collection_repair`, preserving that failure.
No GPU or replay repeats. Runtime and evidence remain under
`/x/Temp/kindle-life-return-actor-20261103`; bounded subdirectory `check` is closed.

The padded 200k projection is 21m56.964s, below 30 minutes. No full learner,
extension or coefficient sweep is selected by this qualification. Keep retained
sampled70.750/greedy143.667 and the original 864/90%-two-wall gate. No Kindle
allocation remains active at closure; check live state before new work.
The five-game goal remains unfinished.

The [completed return/life audit](2026-11-03-cdp-breakout-return-life-audit.md)
finds390→143 positive targets among480 fixed retained states when returns stop
at the next life loss;70.74% of clipped weight disappears. Within16 actions of
loss,30 positives becomezero,but that subset is only3.49% of original weight.
Full-game returns remain valid for game-total reward; this arithmetic does not
identify causal errors or prove that a life-bounded target would help. These are
retained frozen states,not the conservative learner's online batches. At that
audit's closure, no native change or learner was selected; the later optional
prototype is reported above.

All252 pins,16 CPU fixtures,35,552 actions/142,169-frame annotation replay,
eight pixel-sequence hashes,512 selected pixels,480 original returns/weights
and independent recount pass.142 raw life losses,zero gains; one historical
file resolves exactly fromGit. Replay15.018s/analysis.639s/recount.273s close
18:52:33.129 UTC November3/source0dc11ba4. No failure,GPU work,model forward,
update or prior-stage repeat. Root`/x/Temp/kindle-breakout-return-life-audit-20261103`
is closed; do not rerun. Retain70.750/143.667 and the original task gate;
the five-game goal remains unfinished. Check live state before new work.

The [completed200k conservative-actor comparison](2026-11-03-cdp-conservative-actor.md)
finishes200k actions /799,546 frames /12,477 updates in21m46.724s (+1.46%).
Greedy144.750 versus143.667 is nearly flat; sampled49.167 versus70.750 falls30.51%.
All0/24 two-wall; own initials1.417/2.208 match. Greedy initial-subtracted score/min
falls.68%,paired lane difference−.045 [−2.574,2.589]. The declared selection rule
fails: retain the existing200k recipe. One reused seed/cohort and different
native/runner revisions do not establish an isolated effect or reliability.
The original864/90%-two-wall gate and five-game goal remain unfinished.

The loss has1,010,324 positive repeated rows on12,403 updates, with bonuses active.
Final50k teacher CE.559403 versus.560514 does not yield better frozen efficiency.
Training max377 and only939 native actions at216–431 show limited later-score
exposure; no432+ action occurs. These are score bands,not verified wall events
or causal evidence. Actual overrides46.429%; guided returns are not competence.

All308 current/338 retained pins,100k/6,227 recovery,357 initial/frozen tensors,
47 pre-update and799/794 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main completes18:38:09.815 UTC November3/sourcecf8c53fc;
original collector completes18:38:40.403. Both services succeed,all five GPU
stages run once; no full-pipeline failure or repair. Earlier prelaunch copy-script
failure remains. Root`/x/Temp/kindle-conservative-actor-20261103/full` is closed;
no restart,extension,confirmation or automatic successor. No Kindle allocation
is active at closure; check live state before separately selected work.

The [completed bounded check](2026-11-03-cdp-conservative-actor.md) passes8192/489 in47.402s control/50.754s
candidate (+7.07%),with33,588 positive targets and the bonus active. Padded200k
projection22m43s passes30min; frozen1024 has zero updates/no natural episodes.
Native3f6c6648 passes1018 Python/169 Rust CPU checks,Clippy and12 F64 cases.
All258 current/338 retained pins,215 preserved sources,357 initial/frozen tensors,
replays and five GPU stages pass. A CPU comparison matches357 final control
arrays/1065 events to the old short control. Closed17:58:41.844 UTC November3;
CPU fixture/path failures remain,no GPU repeats. Bounded root`check` is closed.
Actual full CPU postprocessing and queued-config preflights pass before launch;
one private copy-script assertion is preserved. See the result for evidence.

The [completed-return actor pair](2026-11-03-cdp-episodic-actor.md) closes with a useful bonus-off
result:sampled46.458→56.542 (+21.70%),greedy65.208→95.375 (+46.26%),all0/24
two-wall,in17m05.486s→17m12.959s (+.729%) for200k actions/12,477 updates each.
Greedy lane-bootstrap gain30.167 [2.167,60.001],gain/min1.726 [.095,3.462];sampled
intervals also exceedzero. One reused seed/cohort,not training-seed reliability.
Retained action-effects1 remains70.750/143.667 in21m27.910s;candidate costs19.80%
less but has lower score and gain/min point estimates. Keep the retained primary
recipe and this bonus-off development result; no extension,third learner or
automatic confirmation. Original864/90%-two-wall gate remains unmet; the
five-game goal is unfinished.

Both100k/6,227 recoveries,261 matching initial/invariant frozen tensors,47
pre-update events,799/794 initial semantic events,full pixel/action/label and
frozen replays/videos,guards,curves,exposure and comparison pass. Candidate
positive targets appear on12406 updates;1167144 positive replay rows are repeated
exposures. Neither learner reaches432 training points;maxima399/415. Guided
returns and replay fit are not competence. Same nativee3b02163/runner/teacher;
only actor scale100 differs within the bonus-off pair.

All307 original/22 first-repair/10 closure pins,69 control/154 candidate files
and338 retained pins pass. Original main fails16:45:24 UTC November3 on CPU
exposure count/schema; its collector also fails. Separate3.458s repair completes
16:49:21.889. Unstarted candidate launches16:51:24,finishes training17:08:43,then
its pipeline fails17:12:50 on CPU initial elapsed-time equality; its collector
also fails. Both pipeline failures remain;
allten GPU phases succeed,zero learner/frozen/replay repeats. Separate CPU
comparison/closure completes17:19:25.280; independent2.865s verification completes
**17:19:38.264**,source28b03632,status`complete_with_cpu_postprocess_repair`.
All four original services remain failed; no original passing completion exists.
Root `/x/Temp/kindle-episodic-actor-20261103/full` is closed; do not rerun it.
No Kindle GPU allocation or successor is active. Check live state before work.

The [closed bounded pair](2026-11-03-cdp-episodic-actor.md)
passed8192 actions/489 updates in36.677s control/38.277s candidate (+4.36%),
with33105 positive replay rows and no natural frozen episodes. All native/CPU,
legacy, tensor, replay and terminal checks pass; no failed or repeated stage.
Native e3b02163/SHA d406629f adds no parameters; disabling disagreement removes96
ensemble tensors, leaving261. Runtime,wheels,validation binary and1.008GB cache
remain. Bounded root `check` is closed; never reuse its experience or rerun it.

The [CPU replay-actor target audit](2026-11-03-cdp-breakout-replay-actor-targets.md) confirms no direct
return-weighted actor update on executed replay actions. Reusing existing critic
targets gives241 positive states of480; removing short-window external rewards
leaves200 and83.32% of normalized positive weight. Imagined bootstraps and
intrinsic bonuses stay fixed; this is arithmetic dependence,not a causal learning
fraction. No gameplay benefit or new loss is selected. Completed real-outcome
target accounting remains the next design question.

All17 fixtures,20 current/158 original pins,512 incoming and480 teacher alignments
pass. Approximate value recovery matches final native normalizer state to2.339e-7;
fixed sensitivity leaves the broad result unchanged. Analysis completes15:42:20.331
UTC November3/source934053ae in.436s CPU; independent recount15:43:34.993 in.202s.
No failure,model forward,emulator,GPU work,update or prior-stage repeat. Root
`/x/Temp/kindle-breakout-replay-actor-targets-20261103` is closed; do not rerun.
No Kindle service is active at closure. Keep the retained200k/400k budget points;
Breakout and the five-game goal remain unfinished. Check live state before work.

The [completed400k/R16 Breakout comparison](2026-11-03-cdp-breakout-budget400k.md)
raises sampled70.750→92.792 (+31.15%) andgreedy143.667→173.708 (+20.91%),all0/24
two-wall,in43m36.578s (+103.16%) for400k actions/1,599,381 frames/24,977 updates.
Initial-subtracted score/min falls35.13%/40.33%. Retain200k as the cheaper baseline
and400k as a higher-score budget point; no extension,confirmation or automatic
successor is selected. Final50k teacher CE falls.560514→.411859; guided116.0
is not unaided competence. Training maximum408/zero432+ and only2,397 native
actions at216–431 (0.59925% of all actions) show limited later-score exposure,
not verified wall boundaries or a causal diagnosis.

All446 pins,three recoveries,357 matching initial/invariant frozen tensors,
38,050 exact common200k semantic events,initial800/795 traces,full replays/videos,
guards,curves,exposure,coverage and comparison pass. Initial/100k/200k tensors
and canonical content match the retained control; full live replay/RNG equality
is not asserted. Same12M64/Auto/N8/B16/T16/H15/R16/context1,rates/teacher/expert100/
critic1/1,native447fe50/runnercf4ea238; fresh life but reused1009/cohort.

Original mainPID1034231/invocation13c5fb9a8be543deb2c279522c8498a2 and all observers
succeed; main/post/collection finish15:21:22.175/15:21:33.374/15:22:02.467 UTC
November3/source4bb0d101. Independent terminal reading and repair closure pass
once15:22:18.072 in2.689s (2.708s wrapper). Closure`complete_with_cpu_launch_repair`
preserves the original serialized-file-hash launch-wrapper failure; six-case CPU
repair retains raw integrity and exact header/payload checks. All11 repair pins
pass. No learner restart,native change or successful GPU/reader/collector repeat.
CPU co-host costs remain uncorrected. Root`/x/Temp/kindle-breakout-budget400k-20261103`
is closed; no Kindle GPU allocation remains active. Check live state before
separately selected work. Breakout and the five-game goal remain unfinished.

The [completed frozen halfway assessment](2026-11-03-cdp-breakout-halfway.md)
finds **22.125→70.750 sampled /33.250→143.667 greedy** from100k to200k; all eight
lane means improve in both modes. Paired lane intervals are[36.333,61.042] and
[80.292,145.875]. Initial-subtracted score/min rises1.894–1.974→3.230 sampled
and2.839–2.959→6.590 greedy. Halfway time is bracketed10m29.382s–10m55.956s;
final21m27.910s remains unchanged. All0/24 two-wall. Two checkpoints establish
progress over that interval,not a slope at200k,optimal budget or future success.

Both new frozen arms total51,048 actions/204,112 frames/52.534s,zero updates.
All413 pins,357 finite/invariant tensors per arm,full replays/videos,guards and
paired comparison pass; original final/initial evaluations are reused without
another execution.17 fixtures/two actual CLI paths pass in3.369s preflight.
ServicePID1032697/invocationc47bd45edffe4af5871bef32e1714dac succeeds
14:03:53.486 UTC November3/sourceca6bc06e; fixed terminal collection succeeds
14:04:28.611 in1.015s. No failure,build,training or prior-stage repeat.
Root `/x/Temp/kindle-breakout-halfway-20261103` is closed; do not repeat its
stages. The original200k evaluation rule is unchanged. Reused learner/cohort
limits reliability; Breakout and the five-game goal remain unfinished.

The [completed context9 comparison](2026-11-03-cdp-breakout-context9.md)
lowers sampled **70.750→46.583** and greedy **143.667→91.167**, all **0/24
two-wall**, in **23m30.959s (+9.55%)** for200k actions/799,606 frames/12,473
updates. Initial-subtracted score/min falls40.54%/42.60%. Retain context1 and
constant expert100/critic1/1. Prefix recomputation totals101.886s,8.169ms/update
and7.22% of training wall. Late CE.560514→.559628 barely changes; guided online
67.78→72.46 is not unaided competence. No improved state accuracy is established.

All667 pins,100k/6,223 recovery,357 matching initial/invariant frozen tensors,
47 common pre-update and800/795 initial frozen events,full replays/videos,guards,
curves,exposure and comparison pass. All five original services succeed;
main/post/collection finish **13:48:19.732 /13:48:49.527 /13:48:54.418 UTC
November3**, source55f0a46a. Fixed independent terminal reading passes once
**13:49:17.338** in1.766s wrapper. No full-run stage fails or repeats. An ad-hoc
poll rejects the unloaded recovery unit's empty invocation; original journal
identity and success pass. Earlier bounded failures/repairs remain preserved.

Context9 adds eight forward-only observed steps before16 learning steps with
12M64/Auto/N8/B16/T16/H15/R16,retained rates/teacher/constant expert100/critic1/1.
Sampling,warmup,total work and native/runner differ from447fe50; no isolated
cache-age cause or independent reliability claim. Native65dd69b4/runtime and
bounded runner are reused without a build. Keep CPU co-host times uncorrected.
Root `/x/Temp/kindle-replay-context-20261103/full` and all readers are closed;
do not repeat stages or readers. No Kindle allocation remains active at closure.
Check live state before GPU work. No extension,sweep,confirmation or automatic
successor is selected. Breakout and the five-game goal remain unfinished.

The [optional replay-state recomputation check](../results/2026-11-03-cdp-replay-context.md)
passes8,192 actions/32,730 frames/485 updates in50.564s. Context9 adds eight
forward-only steps before16 learning steps; no new parameters or prefix targets.
Prefix time is3.234s/6.668ms per update. Conservative200k projection22m37.921s
passes30min; gameplay benefit remains unmeasured. First update440,all485 finite
reports,357 matching initials/invariant frozen tensors,47 common pre-update
and128 legacy events,full pixel/action/label replay and guards pass. Frozen1024
in.914s haszero updates/no natural episodes. Keep the70.750/143.667 control.

Native65dd69b4/SHA7b15cb69 passes155+5 Rust/983 packaged Python CPU checks and
strict Clippy. CPU target-shape/unused-helper failures and a failed RGB fixture
remain. Test-only694c64bd passes156+5 CPU checks and the repaired device fixture;
production/runtime unchanged,passing posterior fixture not repeated. All235
native pins/guards pass. Original bounded service fails13:08:10 UTC November3
only on its stale489-vs-declared485 queue assertion; the unstarted frozen stage
continues once. A separate CPU repair handles omitted decay-only scale fields.
All601 original/631 repair pins pass; collection completes13:11:47.928 in2.590s,
closure`complete_with_cpu_controller_repair`. Original service remains failed;
no training/frozen/replay stage repeats. Runtime/source differs447fe50,not an
isolated cause. Root `/x/Temp/kindle-replay-context-20261103` and readers are
closed. No full learner was selected at that bounded closure; the separately
selected full comparison is completed above. The five-game goal remains unfinished.

The [CPU replay-context age reconstruction](../results/2026-11-03-cdp-breakout-replay-age.md)
matches all12,477 logged updates and199,632 sampled starts in.387s CPU.
Final50k unmasked contexts have median246/mean365.099/p99 1,776 updates since
their last computation;98.66% of labeled actor rows can depend on initial
context before a reset. This measures age,not latent error or a gameplay cause.
All15 fixtures and27 diagnostic/338 control pins pass; review completes
12:35:19.800 UTC November3/sourceb729ab23. No GPU work,learner,emulator,failure
or prior stage repeat. Root `/x/Temp/kindle-breakout-replay-age-20261103` is
closed. Earlier replay16k and B8/T32 results remain; this age audit alone
did not select a new buffer,prefix,loss or learner. Retain the70.750/143.667 control; goal unfinished.

The [completed Breakout H3/scale1 comparison](../results/2026-11-03-cdp-breakout-open-loop.md)
lowers frozen sampled **70.750→60.875** and greedy **143.667→104.625**, all
**0/24 two-wall**, in **22m06.821s (+3.02%)** for 200k actions / 799,552 frames /
12,477 updates. Initial-subtracted score/min falls **16.76% / 29.72%**.
Retain the control without H3, constant actor/world imitation100 and critic1/1.
Late replay expert CE rises .560514→.648492; guided online67.78→84.74 is not
unaided competence. No improved forecasts or isolated loss mechanism is proven.

All644 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial frozen events,finite weighted auxiliary
reports,full replays/videos,guards,curves,exposure and comparison pass. All five
original services succeed; main/post/collection finish **12:16:22.850 /
12:16:49.658 /12:16:54.736 UTC November3**, sourceffc4a26d. Independent terminal
reading passes once **12:17:22.865** in1.404s wrapper. No full-run failure or
GPU/replay/collector/reader repeat. Earlier bounded CPU collection failure and
repair remain preserved. Keep CPU co-host timings without correction.

Nativec2d627f3/runner8946f5f3 differ from retained447fe50/cf4ea238; the explicit
imagined-bootstrap default is normalized. Reused1009/cohort does not establish
an isolated cause or independent reliability. Root
`/x/Temp/kindle-breakout-open-loop-20261103/full` and all its readers are closed.
No allocation remains active at closure; check live state before GPU work.
No extension,sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [Breakout H3 world-supervision check](../results/2026-11-03-cdp-breakout-open-loop.md)
passes8,192 actions/32,717 frames/489 updates in48.603s; conservative200k
projection21m45.264s passes30min. Frozen1,024 actions/.893s haszero updates
and no natural episodes. All602 pins,357 matching initial/invariant frozen
tensors,47 pre-update events,finite weighted auxiliary reports,full guided
replay and guards pass. Keep actor/world imitation100; only H3/scale1 world
supervision is added. Nativec2d627f3/runner8946f5f3 are reused without a build
or repeated numerical/legacy capture; gameplay benefit remains unmeasured.

Original mainPID1019894/invocation7b942f6a912842eea2e3c49f11e24cc4 succeeds
11:41:09.152 UTC November3/source63c347f9. The original collector's parent-PID
logging assertion fails; separate CPU repair passes11:43:00.131 in1.065s with
six identity fixtures/seven repair pins. Original failure and all602 pins remain;
closure`complete_with_cpu_collection_repair`. No GPU,frozen or replay repeat.
Root `/x/Temp/kindle-breakout-open-loop-20261103/native-check` and its readers
are closed. No allocation is active at closure; check live state before work.
No full learner,extension,sweep or automatic successor is selected by this check.
The five-game goal remains unfinished.

The [completed actor-only imitation-decay comparison](../results/2026-11-03-cdp-breakout-actor-only-decay.md)
lowers frozen sampled **70.750→51.833** and greedy **143.667→104.750**, all0/24
two-wall, in **21m42.074s** (+1.10%). Initial-subtracted score/min falls28.07%/
28.30%. Retain constant actor/world imitation100 and the70.750/143.667 control.
The actor holds100 through100k, fades to0 by150k and stays0 for3,126 reports;
world stays100. Guided online67.78→74.68 is not unaided competence.

All649 pins,100k recovery,357 matching initial/invariant frozen tensors,paired
initial traces,full replays/videos,guards,curves,exposure and comparison pass.
The fixed pre-fade checks match12,500 transitions,6,227 reports,170 episodes,
167 resets and all357 tensors/51,502,659 elements exactly at100k. Nativec2d627f3/
runner8946f5f3 still differ from447fe50; no full-state, broad runtime or isolated
causal claim. All five original services succeed; main/post/collection finish
11:28:15.248/11:28:35.619/11:28:40.720 UTC November3, source18d493e6. Independent
terminal reading passes once11:29:04.321 in1.512s wrapper. No full-run failure,
GPU repeat or collector/reader repeat. Earlier CPU preparation failures and
co-host timings remain preserved without time correction.

Root `/x/Temp/kindle-actor-only-decay-20261103/full` and all its readers are
closed; do not repeat them. No allocation remains active at closure; check live
state before GPU work. No extension,sweep,confirmation or automatic successor
is selected. Breakout and the five-game goal remain unfinished.

The [actor-only imitation-decay qualification](../results/2026-11-03-cdp-actor-only-imitation-decay.md)
passes F64 loss/derivatives and fresh Breakout8,192 actions/32,725 frames/489
updates in47.329s. Actor scale100 reaches zero at256 (234 zero-weight reports);
world stays100. All599 pins,357 matching initial/invariant frozen tensors,47
pre-update events,full pixel/action/label replay and guards pass. Legacy1,024
frozen actions reproduce128 control events with357 invariant tensors; candidate
frozen1,024 actions take.879s,zero updates/no natural episodes. Conservative200k
projection21m11s passes30min. Execution/cost are qualified; gameplay is unmeasured.

Nativec2d627f3/SHA0015dff7 at `/x/Temp/kindle-actor-only-decay-20261103/runtime`
passes149 Kindle+5 gym CPU tests,971 packaged Python checks and strict Clippy.
An original Boolean-error-message fixture failure is preserved;5c6526fc repairs
only its expectation,with no rebuild/production change. Packaging closure is
`complete_with_cpu_test_repair`. Before GPU work,349b53ea clarifies the explicit
legacy `replay_value_bootstrap=imagined` default addition. Numerical.74s/228 pins
passes10:50:21.863 UTC; original bounded service succeeds10:52:27 November3,
PID1015127/invocationcc77dbd06adf48fc9a01aec6a179b5db/source5c6526fc.
Terminal10:52:49.664 takes1.144s. No GPU stage fails/repeats. All stages/readers
are closed; do not rerun them. ReviewSHAf9b8ca51/completionSHA5097159b.

The preceding [actor optimizer attribution](../results/2026-11-03-cdp-breakout-actor-optimizer.md)
finds conditional imitation contributions7,493×/10,928× imagined score-function
on two fixed final-checkpoint batches; momentum also affects the full estimate.
All13 fixtures,210 current/158 prior pins pass in1.027s CPU,zero updates/GPUwork.
F64 estimates are not measured native movement or proof of a gameplay cause.
Its root and the earlier raw-gradient capture/backward readers are closed.

Retain policy1/expert100/critic1/1 and the70.750/143.667 gameplay control,all0/24
two-wall. The optional actor-only fade preserves world imitation; the earlier
shared fade weakened both on an older recipe and regressed. No full learner,
sweep,extension or automatic successor is selected by this qualification.
No Kindle GPU allocation is active at closure; check live state before work.
The five-game goal remains unfinished.

The [completed policy-weight100 comparison](../results/2026-11-03-cdp-breakout-policy100.md)
lowers sampled **70.750→38.250** and greedy **143.667→92.125**, all0/24 two-wall,
in **21m49.372s** (+1.67% training time). Initial-subtracted score/min falls
47.75%/37.48%. Retain policy1, expert100, critic1/1 and actor/critic representation
gradients off. Only policy1→100 changes, scaling score-function and entropy
together; native447fe50 and teacher stay fixed, runner5a86e224 adds12 CLI lines.
Current actor-gradient dominance is measured in the diagnostics above; reused
seed/cohort limits reliability. Slightly lower replay expert CE does not establish gameplay benefit.

All436 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed **10:00:29.787 /
10:00:57.364 /10:01:01.652 UTC November3**, source0c40e657; allfive original services
succeed. Independent terminal/coverage readers complete once10:02:01.134/
10:05:11.888. All52 diagnostic/436 full pins reconcile273,544 actions/1,093,592
frames. Native training actions at raw scores216–431 fall1,262→1,196,none432+;
these descriptive bins do not identify wall boundaries or a causal limitation.
Actual guidance overrides are44.545%; guided online67.68 is not frozen competence.

Root `/x/Temp/kindle-breakout-policy100-20261103/full` is closed. Bounded,recovery,
terminal and coverage readers are complete; do not repeat stages or observers.
No full-run failure or GPU repeat occurs. Two earlier bounded CPU fixture failures
and the coverage-reader provenance-path failure/CPU repair remain preserved.
CompletionSHA38776056/terminalSHA26e0da3d/coverageSHAab72765d. Keep CPU co-host
costs uncorrected. No Kindle allocation remains active at closure; check live state.
No extension,coefficient sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed contact-timing CPU teacher comparison](../results/2026-11-03-cdp-breakout-contact-timing-policy.md)
lowers mean185.375→125.875, with only2/8 improved lanes and zero two-wall successes
in both arms. The fixed10%/five-lane screen fails. Retain the original teacher;
no24-episode extension, integration, coefficient search or learner is selected.
All16 CPU fixture cases,211 pins, exact pixel/action/debug replay and independent
ledger recount pass. Collection13,352 actions/53,396 frames takes8.838s, replay
10.789s; controller completes09:20:01.737 UTC November3/source334972c8 in19.811s.
Recount09:20:24.966 verifies2,552 corrected near-contact decisions, zero time clamps.
No failure, repeated emulator stage, GPU work or learner update occurs. Root
`/x/Temp/kindle-breakout-contact-timing-policy-20261103` is closed; do not repeat
stages or readers. ReviewSHA17202d2d/completionSHAb915684a. No Kindle allocation is
active at closure; check live state. Breakout and the five-game goal remain open.

The preceding [saved contact-timing diagnostic](../results/2026-11-03-cdp-breakout-paddle-contact-timing.md)
passes its fixed prediction screen: episode2 loss-contact MAE6.101→5.757 pixels,
successful-contact MAE2.376→2.283, with lower RMSE and seven improved lane means.
Some tails worsen. All24 pins, five new/eight reused fixtures and739-row/875-metric
recount pass; sourceb7bafe64, completion09:10:22.591 UTC November3. Its root is
closed, with no failed/repeated stage, emulator/native import, GPU work or update.
The later gameplay failure shows why this aggregate error gain was insufficient
for teacher integration. It does not diagnose the native learner's causal limit.

The [current-position](../results/2026-11-03-cdp-breakout-paddle-timing.md)
and [next-position](../results/2026-11-03-cdp-breakout-response-timing.md)
saved-data screens also pass; their roots are closed. All timing diagnostics
reuse development episodes and teacher states. The original runtime teacher
and retained native learner remain unchanged.

The [completed imagined-value0.3 comparison](../results/2026-11-03-cdp-breakout-imagined-value03.md)
lowers sampled **70.750→35.708** and greedy **143.667→102.000**, all 0/24
two-wall, in **21m44.356s** (+1.28% training time). Initial-subtracted score/min
falls 51.16%/30.34%. Retain imagined-value1/replay-value1 and actor/critic
representation gradients off, with the retained12M64/Auto/N8/B16/T16/H15/R16/
lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/expert100/replay100k recipe.
Only imagined return fitting and its slow-value regularizer are rescaled together;
replay targets still bootstrap from imagined returns. Native447fe50 and teacher
stay fixed; runner76440d32 adds12 CLI lines. Reused seed/cohort limits reliability.

All448 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed **08:44:22.585 /
08:44:33.979 /08:45:10.209 UTC November3**, source44f2cb48. All five original
services succeed. Terminal/coverage readers complete once08:45:43.886/08:45:54.789;
48 diagnostic/448 full pins reconcile275,720 actions/1,102,301 frames. Native
training actions at raw scores216–431 fall1,262→404, none432+; these are descriptive
bins, not verified walls or causal coverage. Slightly better replay CE and guided
online78.46 do not outweigh worse frozen play. Actual overrides are43.987%.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103/full` is closed. Recovery,
terminal and coverage readers are complete; do not repeat stages or observers.
No full-run failure or GPU repeat occurs. Five prelaunch CPU corrections and the
bounded fixture's unintended native initialization remain preserved. Completion
SHA90fcf03d/terminalSHA314d5237/coverageSHA788a17f1. Keep CPU co-host time uncorrected.
No Kindle allocation remains active at closure; check live state before work.
No extension,coefficient sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed actor/critic-gradient comparison](2026-11-03-cdp-breakout-acgrads-value1.md) lowers sampled
70.750→54.958 and greedy143.667→117.833 in21m56.980s (+2.26% training time),
all0/24 two-wall. Initial-subtracted score/min falls24.48%/20.07%. Retain the
raw-reward control with actor_critic_gradient off and12M64/Auto/N8/B16/T16/
H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100/
replay100k. Only initial imagined policy/value representation gradients change;
native447fe50, runnercf4ea238 and teacher stay exact. Active losses and slightly
lower replay expert CE do not establish gameplay benefit or an isolated cause.

All428 pins,100k/6,227 recovery,357 matched initial/invariant frozen tensors,
47 pre-update events,800/795 paired initial traces,full replays/videos,guards,
curves,exposure and comparison pass. Main/post/collection succeed07:54:00.102/
07:54:18.504/07:54:52.283 UTC November3/source286c54b. Terminal and coverage
readers complete once07:55:21.963/07:55:30.341;48 diagnostic/428 full pins
reconcile285,024 actions/1,139,517 frames. Native training actions at raw scores
216–431 fall1,262→828,none432+; bins are descriptive,not verified walls or
causal evidence. Actual44.8745% overrides and online64.28 are not competence.

Root`/x/Temp/kindle-breakout-acgrads-value1-20261103/full` is closed. Allfive
original services and recovery/terminal/coverage readers are complete; do not
repeat them. No full-run failure or GPU repeat;two pre-launch CPU corrections
remain preserved. CompletionSHAb029cb5e/terminalSHA074d990e/coverageSHAf8b8ac82.
No allocation is active at closure; check live state. No extension,coefficient
sweep,confirmation or automatic successor. Keep CPU co-host timings uncorrected.
Reused1009/cohort limits reliability. Breakout and the five-game goal remain
unfinished.

The [fixed subaction diagnostic](2026-11-03-cdp-breakout-subaction.md) finds no extra rescues from one
mid-action command switch: three constant commands and nine two-frame pairs
both avoid34/120 recorded losses, zero gain in every lane. Retain four-frame
control; no shorter-interval learner or automatic successor is selected. This
uses late loss states of the older104.667 greedy policy and a shared recorded
continuation, not a repeated-feedback policy or the latest143.667 control.
All302 pins,360 wrapper parity checks,120 recorded controls,eight reversed orders
and37,016-action/148,038-frame source replay pass. Eight CPU tests;35.736s analysis
completes07:10:16.312 UTC November3/sourceba52b20, controller07:10:17.145 and
independent recount07:11:01.186. No failure,GPU work or updates. Root
`/x/Temp/kindle-breakout-subaction-20261103` is closed; do not repeat stages or
readers. ReviewSHA8b92a0c2/completionSHA7e72eafb. No allocation active at closure;
check live state. Breakout and the five-game goal remain unfinished.

The [completed one-point life-loss comparison](2026-11-03-cdp-breakout-life-penalty.md)
lowers sampled70.750→42.542 and greedy143.667→90.875 at nearly unchanged
training cost21m39.349s (+0.89%), all0/24 two-wall. Retain the raw-reward
control. Independent replay verifies all1,290 observed losses and negative
training rewards; raw scores, episode boundaries and native configuration
remain unchanged. All453 pins, recovery, initial/frozen tensors, full replays
and original services pass; terminal and fixed coverage readers are complete.
No new penalty sweep, extension or automatic successor is selected.

The earlier [saved-data diagnostic](2026-11-03-cdp-breakout-life-feedback.md)
found zero immediate reward at120 lost lives,96 nonterminal, using an older
policy. Lost future rewards and game-over targets already provide indirect
consequences. The new training result rejects this particular extra aid locally;
it neither establishes a missing standard RL loss nor proves all such feedback
harmful. Reused development seed/cohort limits reliability and causal claims.

Guided raw online last50 improves67.78→94.72 and replay expert CE falls
.560514→.551947 despite worse unaided play. Native training actions at raw
scores216–431 barely rise1,262→1,297, none432+. Neither guided returns, replay
fit nor these descriptive score bins substitute for frozen competence or
identify the cause. Original main/post/collection succeed06:42:06.048/
06:42:24.088/06:42:57.559 UTC November3; independent terminal06:56:45.055 and
coverage06:56:53.724 pass once. All stages/readers are closed, with no GPU
repeat or active allocation. Keep CPU co-host caveats and the three earlier
bounded CPU corrections. Breakout and the five-game goal remain unfinished.

The [completed replay16k comparison](2026-11-03-cdp-breakout-replay16k.md) lowers sampled70.750→34.000
andgreedy143.667→57.625 in21m44.359s (+1.28% training time),all0/24 two-wall.
Initial-subtracted return/min falls53.60%/61.32%. Retain shared replay100k with
12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100 andsampled collection. Onlycapacity100k→16,384 changes;
native447fe50,runnercf4ea238 andteacher remainexact. Recency,diversity,reuse and
coverage change together; cache age andisolated cause remainunmeasured.

All445 pins,100k/6,227 recovery,357 matchedinitial/invariant frozen tensors,
16k/2,078-event semantic prefix,paired initial traces,full replays/videos,guards,
curves,exposure andcomparison pass. Main/post/collection succeed05:34:16.487/
05:34:44.611/05:35:19.248 UTC November3/source9cfd8cc; allfive originals succeed.
Terminal05:35:33.562 andcoverage05:35:39.907 pass once;50 diagnostic/445 full
pins,270,920 actions/1,083,085 frames reconciled. Native216–431 actions fall
1,262→878,none432+; score bins are not verified walls orcausal coverage. Late
replay expert CE falls.560514→.381669 onchanged replay despite worsegameplay.
Actual45.4125% overrides/online54.42 are not unaided competence.

Root`/x/Temp/kindle-breakout-replay16k-20261103/full` isclosed. Allreaders and
original stages arecomplete; do not repeat them. No full-run failure orGPU repeat.
Earlier bounded main/collector failures andseparate CPU repair remainpreserved;
original bounded services stayfailed. CompletionSHA2e48b8b0/summarySHAba7fd117/
coverageSHA25a8eb80. No allocation active atclosure; check live state. No capacity
sweep,extension,confirmation orautomatic successor. Keep CPU co-host caveats.
The five-game goal remains unfinished.

The [completed lambda1 comparison](2026-11-03-cdp-breakout-lambda1.md) lowers
sampled70.750→45.667 andgreedy143.667→90.292 in21m52.277s (+1.89%), allzero
of24 two-wall successes. Retain lambda0.95. Using the entire available trace
before endpoint bootstrap does not improve this recipe; both imagined/replay
targets change, so it does not isolate intermediate-value bias, variance or
imagined-reward error as the cause. One reused development seed/cohort limits
reliability. Expert CE improves.560514→.536404 andnative216–431 exposure rises
1,262→1,434 (.631%→.717%), withnone432+. These descriptive measures do not
establish the mechanism anddo not substitute for frozen gameplay.

All448 pins,recovery,357 matched/invariant tensors,paired initial traces,full
replays/videos,guards,curves,exposure andcomparison pass. Main/post/collection
succeed04:39:49.589/04:40:12.443/04:40:16.846 UTC November3/source958d622.
Terminal andcoverage readers complete04:40:46.018/04:40:51.786; allfive original
services succeed,no full-run failure orstage repeat. Earlier CPU import and
prelaunch date-wiring failures remain. Root`/x/Temp/kindle-breakout-lambda1-20261103/full`
isclosed; do not repeat completed stages/readers. Native447fe50/teacher remain
unchanged; runner9eb21cca adds10 CLI lines. No new allocation orautomatic
successor isselected; the five-game goal remains unfinished. Check live state.

The [return-lambda1 check](2026-11-03-cdp-breakout-lambda1.md) passes8,192
actions/489 updates in50.121s and1,024 frozen/.868s withzero updates/no natural
episodes. Conservative200k projection22m26.028s clears30min. All385 pins,357
matched initial/invariant tensors,47 pre-update events,replay andguards pass;
original services succeed04:07:39.518/04:07:43.412 UTC November3/sourcef466942.
Four exact-source scalar cases confirm continuing,terminal andtimeout semantics.
Lambda.95→1 changesboth imagined andreal-replay targets,retaining endpoint
bootstraps; neither gameplay benefit norcause isestablished. Native unchanged,
no full learner selected. Completed stages must not repeat; initial CPU test
import failure remains preserved. No GPU allocation active atclosure.

The [fixed pixel-motion readout](2026-11-03-cdp-breakout-pixel-motion-readout.md)
fails its screen: learned-plus-motion teacher agreement55.91% overall/55.88%
near descending trails actor62.56%/66.91%. Pixel/motion alone improves over
current pixels alone54.08% versus48.25%, but the augmented readout falls below
its saved current-pixel baseline57.40%. This does not support the proposed
temporal-input prototype or establish a general limitation of temporal inputs.
One new CPU predecessor capture matches20,960 actions/83,799 frames and2,064
saved current positions; old features/fits are reused. Six fixtures/75 pins pass
in13.330s, complete03:56:34.670 UTC November3/sourcec91dcfa. No failure,GPU work
or updates. Root`/x/Temp/kindle-breakout-pixel-motion-20261103` is complete;
do not repeat stages. No new learner or automatic successor is selected.

The [completed Breakout crop comparison](2026-11-02-cdp-breakout-playfield-crop.md)
scores47.042 sampled/114.833 greedy, below full-frame70.750/143.667 by33.51%/
20.07%, in21m49.159s (+1.65%). Own initials1.375/1.917; all0/24 two-wall.
Initial-subtracted return/min falls35.20%/21.47%. Retain full-frame12M64/Auto/N8/
B16/T16/H15/R16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100
and sampled collection. Crop0/32/160/168 removes HUD information; nativeacb175f/
runner046aa5bc differ from447fe50/cf4ea238. No isolated cause or reliability claim.

All413 pins,100k/6,227 retained recovery,357 matched initial/invariant frozen
tensors,47 bounded-crop transitions,full replays/videos,guards,curves,exposure
and comparison pass. Main/postprocess/collection succeed12:52:52.604/12:53:23.251/
12:53:27.613 UTC November2/source53a75d7; allfive original services succeed.
Terminal reader completes12:54:38.573 in2.282s; coverage12:55:05.372 in3.053s,
with54 diagnostic/413 full pins and283,824 actions/1,134,745 frames reconciled.
CompletionSHA701b9246/summarySHA26d91aa1/coverageSHA92d30ffb. No full-run failure
or stage repeat. Earlier bounded compile/packaging failures remain preserved.

Native216–431 actions fall1,262→1,057, none432+; score bins are not verified walls
or causal coverage. Late expert CE improves.560514→.543409 despite worse play.
Actual guidance41.9085% and online80.18 are not competence. Keep crop default-off;
its153 Rust/642 Python CPU checks,geometry/default compatibility and47.995s
bounded training remain qualified. CPU co-host times stay uncorrected.
Root `/x/Temp/kindle-playfield-crop-20261102/full` is closed. Recovery,terminal
and coverage readers are complete; do not repeat completed stages or collectors.
No Kindle allocation is active at closure. No extension,confirmation,crop sweep
or automatic successor is selected. Check live state before new GPU work;
Breakout and the five-game goal remain unfinished.

The [completed Breakout 25M comparison](2026-11-02-cdp-breakout-capacity25m.md)
scores **49.958 sampled /104.292 greedy**, below12M70.750/143.667 by29.39%/
27.41%, in **36m08.795s** (+68.40% training time). Own initials1.458/1.583;
all0/24 two-wall. Initial-subtracted return/min falls58.46%/56.88%. Retain12M64/
Auto/N8/B16/T16/H15/R16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/
expert100 and sampled collection. Preset12M→25M expands several components
with different initial values on reused1009/cohort; no isolated capacity cause.

All420 pins,100k/6,227 retained recovery,357 matching bounded initial/invariant
frozen tensors,47 pre-update transitions,full replays/videos,guards,curves,
exposure and comparison pass. Allfive original services succeed: main/post/
collection **11:28:52.929/11:29:26.592/11:29:35.005 UTC November2**, launchsource
0e69b83. Terminal confirmation completes11:30:31.187 in3.019s; completionSHA
1988e621/summarySHAb80dc809. No original learner,observer or GPU stage repeats.

The separately corrected CPU coverage reader completes11:30:39.622 in3.214s,
with55 diagnostic/420 full pins and290,216 actions/1,160,239 frames reconciled.
Native216–431 exposure falls1,262→307,with none at432+. Score fractions are
not verified walls or causal coverage evidence. Final50k expert CE improves
.561→.499 despite lower frozen scores; actual overrides41.812%,guided online
79.7 is not competence. Original coverage fixture failure remains preserved;
no preflight,trajectory or replay repeats. CoverageSHA7f2efd01.

Root `/x/Temp/kindle-breakout-capacity25m-20261102/full` is closed. Recovery,
terminal and repaired coverage readers are complete; do not repeat them.
Native447fe50,runnercf4ea238 and teacher remain unchanged. No Kindle allocation
is active at closure; check live state before new work. No extension,confirmation,
capacity sweep or automatic successor is selected. CPU co-host times remain
uncorrected. Breakout and the five-game goal remain unfinished.

The [Breakout 25M bounded check](2026-11-02-cdp-breakout-capacity25m.md)
passes8,192 actions/32,722 frames/489 updates in **84.864s**;1,024 frozen actions
in1.343s have zero updates/no natural episodes. All380 pins,357 actual schema/
finite/invariant frozen tensors,full guided replay and guards pass. Parameters
rise16,334,353→36,608,401; only the preset changes at64/Auto/N8/B16/T16/H15/R16/
encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100. Native447fe50,
runnercf4ea238 and teacher stay fixed. Different architectures have different
initial values; no12M initial equality is claimed. Conservative200k projection
**37m59s** passes45min, but short training costs **77.95% more**. Gameplay benefit
is unmeasured. Main/collection succeed **10:39:11.653/10:39:18.624 UTC November2**,
sourcea8144ec; terminal confirmation10:40:52.368. No failure or stage repeat.
Root `/x/Temp/kindle-breakout-capacity25m-20261102` is closed; no full learner or
GPU allocation is active at closure. Do not repeat checks,collectors or readers.
No automatic promotion,extension,retry or capacity sweep is selected. Check live
state before further work. The five-game goal remains unfinished.

The [saved Breakout crop-gap diagnostic](2026-11-02-cdp-breakout-crop-gap.md)
finds a visible ball at the preceding decision for all 103 losses with an
available inferred paddle crossing; 17 crossings remain unavailable. All 98
nearby missing-ball gap starts occur afterward. Six of 776 fitted successful
contacts have a one-decision gap just before crossing. This does not support
changing the teacher's fallback; fitted timing and differing event filters
limit causal interpretation. Retain the teacher and current learning recipe.
All 14 CPU fixture cases and 17 pins pass; the saved 52,072 actions / 208,254
frames reconcile. Analysis completes **10:28:57.759 UTC November 2**, source
`04c30c6`, in **0.387s**, with no new trajectory, emulator/native import, GPU
work, learner update, failure or repeated stage. Evidence is closed under
`/x/Temp/kindle-breakout-crop-gap-20261102`; do not rerun. No Kindle allocation
is active at closure. The five-game goal remains unfinished.

The [posterior-bootstrap comparison](2026-11-02-cdp-breakout-posterior-bootstrap.md) is **closed_protocol_failed**.
Training completes200k actions/799,562 frames/12,477 updates in **21m37.556s**
(+.75%). Accepted sampled53.708 falls24.09% below70.750; initial-subtracted
return/min falls25.14%. Own initial1.417 matches. Greedy has two25,000-action
timeouts; natural-only117.917 is censored diagnostic,not an accepted score.
Own initial-greedy2.208 matches; all natural cohorts have0/24 two-wall successes.
Retain imagined bootstrap with12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100 and sampled collection.
Native11373c1/runner8952f81f differ from retained447fe50/cf4ea238; reused1009/cohort
supports no isolated causal or seed-reliability claim.

All412 original/553 repair pins,100k/6,227 retained recovery,357 matching initial
and invariant frozen tensors,800/795 actual-initial semantic events,full
replays/videos,guards,curves and exposure checks pass. Four frozen arms total
285,280 actions/1,140,815 frames/261.765s,zero updates. Actual overrides87,981
(43.9905%); guided online returns are not frozen competence.

Original main928259/7aac1e71b2de47f9934cb67d2f7ee1ef and collection fail
**09:54:47.194/09:55:09.359 UTC November2**, after allfive GPU stages finish.
The separate CPU closure fails10:06:17.630 on an overly strict ALE-import
assertion after its three missing replays finish. Preserve every failure.
A separate10.403s CPU repair completes **10:11:27.827** and terminal confirmation
passes10:12:01.748; no GPU stage,successful replay or collector repeats. Original
main/collection/CPU-closure services remain failed. No original passing review
or completion is created; closureSHAde4c7aad/summarySHA21184753.

Root `/x/Temp/kindle-replay-bootstrap-20261102`, closure in`cpu-closure-repair`.
All services are terminal; no Kindle allocation is active at confirmation.
Do not repeat training,frozen captures,replays,collectors,readers or qualification.
No extension,cap increase,retry,confirmation or automatic successor is selected.
Check live state before new work. The five-game goal remains unfinished.

The earlier bounded qualification remains valid:8,192 actions/489 updates in
46.900s; native11373c1 passes150 Rust/628 Python CPU checks and both strict
Clippy checks. Default restore matches128 events/357 tensors. Its2.989GiB cache
is released; wheel/native/test binary/evidence remain. Optional posterior mode
is not retained as the learning recipe; default imagined bootstrap is unchanged.

The [completed scalar value calibration diagnostic](2026-11-02-cdp-breakout-value-calibration.md)
finds a small recoverable signal in the stronger Breakout critic. An affine
correction fitted on eight earlier episodes reduces sampled RMSE9.231→7.698
on sixteen later episodes, versus constant7.855; offset alone scores8.313.
Six of eight lanes improve; the conditional lane MSE interval is[−5.359,−.471].
Correlation stays.231, so most return variation remains unexplained. No
inference correction, loss change or new learner is selected.

All20 diagnostic/74 original pins and seven fixture groups pass. CPU analysis
completes **08:44:53.508 UTC November2/source6a90267** in.918s, with zero GPU
work, native/ALE imports, updates, failures or repeated stages. Root
`/x/Temp/kindle-breakout-value-calibration-20261102` is closed; do not rerun.
No Kindle allocation is active at closure. Prioritize actual target construction
and fitting; the five-game goal remains unfinished.

The [completed replay-value critic diagnostic](2026-11-02-cdp-breakout-replay-value-critic.md) finds that the
stronger Breakout model still has weak sampled-return calibration: correlation
.231/R²−.439, criticRMSE9.231 versus earlier-episode constant7.855. The0.3 model
also trails its constant (9.891 versus8.384); its correlation is.218/R²−.421.
Weight1 sampled mean prediction12.282 underestimates realized16.527; greedy
16.420 versus29.685 has larger bias but changes the target policy. Different
own-policy states prevent causal attribution. Keep the measured gameplay gain;
no new loss weight,target,architecture orlearner is selected.

All74 pins,357 invariant tensors perarm,original trajectories with4,030/5,122/
4,496/5,478 semantic events,finite values/bonuses,guards andreturn decomposition
pass. Four captures151,384 actions/605,381 frames/171.911s havezero updates.
Original main916432/628c655e44284a00ab7ecd46b019125d andcollector916435/
4fe85dbc943542098062105aa841e1d4 succeed **08:23:41.562/08:23:43.655 UTC
November2**, launch sourced982167. Reader passes once08:24:19.771 in1.074s;
completionSHA7d26970f/summarySHA4f127fcd. No preparation,capture orcollector
fails orrepeats. Native447fe50/exact runners6c1b69fb/cf4ea238 stayunchanged.

Root `/x/Temp/kindle-breakout-replay-value-critic-20261102` is closed. Do not
repeat captures,qualification,collector orterminal reader. No Kindle GPU
allocation isactive atclosure; checklive state. No automatic successor is
selected. Breakout andthe five-game goal remain unfinished.

The [completed Breakout H30/R12 comparison](2026-11-02-cdp-breakout-h30-r12.md) scores **58.917 sampled /
124.708 greedy**, below70.750/143.667 by16.73%/13.20%, in **23m15.641s**
(+8.36%). Initial-subtracted return/min falls23.47%/20.09%; all0/24 two-wall.
Retain H15/R16 with12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100 and sampled collection. Horizon and ratio change rollout
length,cadence,sampling,warmup and relative work together; reused1009/cohort
provides no isolated mechanism or independent reliability claim.

All422 pins,100k/4,670 recovery,357 matching initial/invariant frozen tensors,
47 pre-update and800/795 actual-initial semantic events,replays/videos,guards,
curves,exposure and comparison pass. Allfive original services succeed: main/
post/collection **07:58:54.411/07:59:16.112/07:59:20.722 UTC November2**,
launch source538ef64. Terminal reader passes once08:00:10.140 in2.623s;
completionSHAac50672c/summarySHA5d7058c4. No original stage fails or repeats.

The separate CPU coverage reader fails on its obsolete12,477-update assertion;
preserve that failure. A separately qualified9,358-update repair completes
08:03:41.018 in2.918s, with52 original/11 repair/422 full pins. All288,800 actions
and1,154,554 frames reconcile; completed control analysis is reused. Native
216–431 exposure falls1,262→1,157 (.631%→.5785%), with none at432+. Score
fractions do not establish walls or a causal mechanism. CoverageSHA41792f03.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/full` is closed. Recovery,terminal
and repaired coverage readers are complete; do not repeat stages,collectors,
qualification or readers. No learner or GPU allocation is active at closure;
check live state. No extension,confirmation,sweep or automatic successor is
selected. CPU co-host costs remain uncorrected. Breakout and the five-game goal
remain unfinished. Detailed evidence stays in the linked result and protocol.

The [completed lower-dynamics/replay-value1 comparison](2026-11-02-cdp-breakout-value1-dynamics1e4.md)
scores **42.917 sampled /112.125 greedy**, below70.750/143.667 by39.34%/21.95%,
in21m36.378s (+0.66%). Initial-subtracted return/min falls40.54%/22.81%; all0/24
two-wall. Retain dynamics4e-4 withbehavior8e-5/replay-value1 andsampled collection.
Onlydynamics4e-4→1e-4 changes at12M64/Auto/N8/B16/T16/H15/R16/expert100,
native447fe50/runner ea2bf8c6/teacher fixed. Reused1009/cohort isdevelopment
evidence,not independent reliability or an isolated architecture cause.

All422 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,
curves,exposure andcomparison pass. Allfive original services succeed; main/
post/collection finish07:10:06.890/07:10:34.218/07:11:08.882 UTC November2,
launch sourcec691416. Terminal reader completes once07:11:52.749 in2.877s;
completionSHA5f2cd92c/summarySHA8ca4e947. Original bounded preparation failure
andseparate correction remain; no full stage fails orrepeats.

The saved-log coverage follow-up completes07:12:03.388 in2.969s with48 diagnostic/
422 full pins: native actions at216–431 fall1,262→927 (.631%→.4635% oftraining),
none at432+. All283,208 new actions/1,132,245 frames reconcile; the completed
control isreused without recount. Score fractions are not verified walls or
an isolated causal test. Coverage reviewSHA65b130da. Actual overrides48.841%,
late expert CE.662474 andguided71.18 are separate from frozen competence.

Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/full` isclosed. Recovery,
terminal andcoverage readers arecomplete; do not repeat stages,collectors or
readers. No Kindle allocation isactive atclosure; no extension,confirmation,
rate sweep orautomatic successor isselected. CPU co-host costs remain uncorrected.
Check live state; Breakout andthe five-game goal remain unfinished.

The [completed Qbert R20 fresh-seed confirmation](2026-11-02-cdp-qbert-r20-seed18013.md)
scores15,931.250 greedy/24 first pyramids on18013 in26m23.114s,versus development
16001's15,279.167/24 in25m57.740s. Both pass the original15k/90% gate within30min:
one passing development learner andone fresh confirmation,not broad reliability.
Sampled14,976.042/24 and12,972.917/24 both fail15k; own fresh initials142.708
greedy/198.958 sampled clear none. Retain240k/R20 withgreedy evaluation.
Earlier R24 confirmations remain failed; Breakout andthe five-game goal remain
unfinished. No Qbert extension,third learner orautomatic successor isselected.

Fresh240k actions/959,784 frames/18,721 updates keep12M64/Auto/N8/B16/T16/H5/R20/
dynamics1e-4/replay-value1/expert100/greedy collection,native447fe50,runnere304af89
andteacher fixed; onlylearner/evaluation seeds change. All1,046 pins,both100k/7,784
and200k/15,596 recoveries,357 finite/invariant frozen tensors,48 recorded initial
differences,full replays/videos,guards,curves,exposure,coverage andcomparison pass.
Allfive original services succeed; main/post/collection finish06:25:18.233 /
06:25:49.125 /06:26:21.455 UTC November2/source0a4892a. Original main901816 /
976a1ed5007f4ce8b343732f91c38cf9 isterminal. Reader completes once06:27:04.671 in
9.806s; completionSHA3fb594a7/summarySHAc41ca062. No full-run stage fails orrepeats;
original CPU preparation failure andseparate correction remain preserved.

Root `/x/Temp/kindle-qbert-r20-seed18013-20261102/full` isclosed. Both recovery
readers andterminal reader arecomplete; do not repeat stages,collectors orreaders.
Actual overrides25.3096% andguided16,043.5 are not frozen competence. CPU co-host
cost staysuncorrected. No Kindle allocation isactive atclosure; checklive state.

The [Breakout ascending-centering reference](2026-11-02-cdp-breakout-ascending-center.md) fails its expanded24-episode
screen:233.208 versus225.083 (+3.61%),four ofeight lane means improve,all0/24
two-wall. The earlier eight-episode gain46.93%/six lanes remains recorded;
all16,425 first-episode semantic events reproduce. Retain the original teacher;
no fresh cohort,integration,learner or sweep is selected. All31 diagnostic/1,046
active Qbert pins,full pixel/action/debug replay and original learner identity
pass. CPU collection35.990s/replay45.402s/controller87.435s completes06:06:24.338
UTC November2/sourcef8e4931,with no GPU work or updates. Root
`/x/Temp/kindle-breakout-ascending-center-20261102/standard24` is complete;
do not repeat stages. CPU co-host cost remains uncorrected.

The [completed Breakout greedy-collection comparison](2026-11-02-cdp-breakout-greedy-training.md)
finishes200k actions /799,661 frames /12,477 updates in21m44.645s. Primary
sampled78.542 improves70.750 by11.01%, with9.81% more initial-subtracted return
per training minute. Greedy90.042 falls37.33% from143.667; all0/24 two-wall.
Retain this sampled development point and the stronger greedy control. Only live
collection changes at12M64/H15/R16/replay-value1; native447fe50,runner6e03d739,
rates,teacher and other settings stay qualified. Reused1009/cohort provides no
independent reliability claim; guided131.780 is not frozen competence.

All427 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
paired traces,replays/videos,guards,curves,exposure and comparison pass. Main/
post/collection succeed05:36:00.508/05:36:21.599/05:36:57.105 UTC November2,
source9247ffa. Allfive original services succeed; no GPU stage repeats. Terminal
reader completes once05:39:32.077 in1.844s; completionSHA8ceba70a, summarySHAf9ba2112.
The original bounded CPU wrapper failure and separate qualification remain.

The [coverage follow-up](2026-11-02-cdp-breakout-score-coverage.md#greedy-collection-follow-up)
finds native actions at216–431 rising1,262→2,382 (0.631%→1.191% of training),
with zero actions at432+. Three completed unguided episodes reach216 versus
none incontrol. Training maxima408→424 still miss later score ranges. These
are not verified wall boundaries or an isolated cause. All43 diagnostic/427
full pins and292,528 new actions reconcile; the saved control is reused.
CPU reader completes once05:39:43.053 in2.967s, reviewSHA19787de1; no GPU work.

Root `/x/Temp/kindle-breakout-greedy-training-20261102/full` is closed. No learner
or GPU allocation remains active at closure. Do not repeat completed stages,
collectors or readers. No extension, confirmation, cap increase or automatic
successor is selected. Check live state; the five-game goal remains unfinished.

The earlier [greedy-collection prerequisite](2026-11-02-cdp-breakout-greedy-training.md#earlier-bounded-prerequisite)
passes8,192 actions/489 updates in47.418s;1,024 frozen actions/.869s havezero
updates/no natural episodes. All387 pins,tensors,replay,guards andoriginal
services pass. It precedes the separately selected full comparison above; its
experience is not reused. The45 CPU tests and separate wrapper qualification
remain retained. Native-check completionSHAfc5ef001/summarySHA86262978;
do not repeat the bounded check or its readers.

The [Qbert 240k/R20 comparison](2026-11-02-cdp-qbert-r20-240k.md) completes 240,000 actions,
959,778 frames and 18,721 updates in 25m57.740s. Greedy 15,279.167 with 24/24
first pyramids passes the 15k/90% gate, versus retained 14,913.542/24. Sampled
12,972.917/24 falls from 14,745.833/24 and fails 15k. Greedy score rises 2.45%
and training time rises 4.85%, so initial-subtracted return/min falls 2.27%
(sampled: −16.21%). Retain this passing development budget point with its measured efficiency cost;
independent confirmation remains open. Seed 16001 and its original
cohort are reused. R24→20 and 200k→240k change experience, cadence and sampling
together. Native 447fe50, runner e304af89, teacher and other settings stay fixed.
The earlier R24 confirmation remains failed.

All 991 pins, both recoveries, 357 initial/invariant frozen tensors, paired
traces, replays/videos, guards, curves, coverage and original services pass.
Main/post/collection succeed at 04:48:55.980/04:49:14.933/04:49:46.075 UTC
November 2, source 55ead06. The terminal reader completes at 04:50:26.168;
completion SHA7423739f, summary SHA7f721710. Original CPU preparation failures
and corrections remain preserved; no GPU stage repeats. Root
`/x/Temp/kindle-qbert-r20-240k-20261102/full` is closed. No allocation is active,
and no extension, confirmation, sweep or successor is selected. Do not repeat
completed stages/readers. Check live state; the five-game goal remains unfinished.

The [saved Breakout score-coverage diagnostic](2026-11-02-cdp-breakout-score-coverage.md)
finds97.03% ofcandidate training actions below216 points andzero at432+,with
maximum408 versus control418. Of5947 candidate actions at216–431,4685 areteacher
overrides; native exposure there is1262 actions (0.63% oftraining). Frozen greedy
means117.292→143.667 still havezero432+ episodes. Score fractions are not verified
wall boundaries; this locates scarce experience,not an isolated cause orselected
change. All400k training/175144 frozen actions,seven fixtures,31 original/six
repair/991 active pins reconcile. Original cumulative-counter failure remains;
separate9.051s CPU analysis completes04:37:06.430/sourceb608d35 withno GPU work.
Root`/x/Temp/kindle-breakout-score-coverage-20261102` iscomplete; do not rerun.

The [November2 R20 bounded check](2026-11-02-cdp-qbert-r20-240k.md) passes
execution andearly cost:8192 actions/611 updates in50.566s, conservative240k
projection27m09.558s. Learning uses81.49%/81.23% ofretained full Qbert training
time;240k/R20 would keep update count nearlyfixed against200k/R24. This is a
budget hypothesis for later-play improvement,not an isolated architecture cause
or measured gameplay benefit. All940 pins/tensors/replay/guards andoriginal
services pass. This prerequisite preceded the completed full comparison above;
its projection did not establish gameplay benefit.

The [completed Qbert replay-value1 crossed assessment](2026-11-02-cdp-qbert-replay-value1-cross-cohort.md) keeps14009
ahead onboth cohorts:16525.000/15906.250 versus16001's15163.542/14913.542,all
24/24 first pyramids. Cohort choice changes16001's near-threshold gate,while
learner gaps1361.458/992.708 remain. Original fresh16001 confirmation staysfailed;
no favorable-cohort selection or new independent learner. The original recipe
still passes primary greedy onone of two learners.

All957 pins,357 finite/invariant tensors,new-arm replays/videos,guards,matrix
andoriginal services pass. Four new arms138952 actions/555639 frames/115.406s,
zero updates; four completed diagonal GPU arms are reused. Main/collector
succeed03:46:51.221/03:47:07.244 UTC November2/sourceb883162. Separate3.310s
reader passes03:47:46.833; completionSHAcd59f0ff/summarySHAf92533f6. No stage
fails orrepeats. Native447fe50/35ad0db,runnere304af89,configs,training times,
24 natural episodes perarm and15k/90% gates stayfixed. No isolated weight,
policy or population reliability claim.

Root`/x/Temp/kindle-qbert-replay-value1-cross-cohort-20261102` isclosed. Do not
repeat crossed arms,diagonals,preparation,collector orreader. No learner,loss
change,extension,retry orautomatic successor isselected. No allocation remains
atclosure;check live state. The five-game goal remains unfinished.

The [completed Breakout replay-value1 comparison](2026-11-02-cdp-breakout-replay-value1.md) scores
70.750 sampled/143.667 greedy versus59.000/117.292, improving19.92%/22.49% in
21m27.910s (1.27% less training time). Initial-subtracted return/min improves
21.96%/24.50%. Retain scale1 as the stronger development point; allzero of24
two-wall successes, reused1009/cohort and no independent reliability claim.

All338 pins,100k/6227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,full replays/videos,guards,
curves,exposure andcomparison pass. Main/post/collection succeed03:39:28.406/
03:39:34.988/03:39:37.809 UTC November2/source45cd1ab. Allfive original services
succeed; no GPU,replay,audit orcollector stage fails orrepeats. Separate1.749s
terminal reader passes03:40:38.280; completionSHA6254f729/summarySHAb55b2016.
Original CPU preparation failures andseparate repairs remain preserved.

Only replay-value0.3→1 changes at12M64/Auto/N8/B16/T16/H15/R16/base-encoder4e-5/
behavior8e-5/dynamics4e-4/expert100,sampled collection. Native447fe50/35ad0db and
teacher fixed; runnercf4ea238 adds ten fresh-CLI lines tocontrol6c1b69fb. Critic
andrepresentation gradients change together; no isolated mechanism. Actual
overrides43.0585%; late CE.560514 andguided67.78 are not frozen competence.
CPU co-host costs remain uncorrected. Root`/x/Temp/kindle-breakout-replay-value1-20261102/full`
isclosed. Do not repeat stages orreaders. No extension,sweep,confirmation or
automatic successor isselected; no allocation remains atclosure,check live state.
The five-game goal remains unfinished.

The [Qbert replay-value1 phase diagnosis](2026-11-02-cdp-qbert-replay-value1-phases.md) locates the entire
frozen score gap after identical550-point first-pyramid prefixes inall96 episodes.
Both learners collect about76% later actions; late native movement agreement is
77.50%/77.07%. Greedy median gap150 is smaller than mean1611.458; fresh scores
range11175–15650 versus14800–22025. These are descriptive cohort differences,
not an isolated cause or changed gate. All27 diagnostic/338 active pins and
fixtures pass in4.038s CPU at03:19:44.467/source305d145; original Breakout identity
stays unchanged. No GPU work,replay repeat,loss change or new learner. Do not rerun.

The [completed Qbert replay-value1 confirmation](2026-11-02-cdp-qbert-replay-value1-seed16001.md) scores
**14,913.542 greedy /14,745.833 sampled**, each24/24 first pyramids. Both means
miss15,000. Fresh16001 completes200k actions/799,817 frames/18,715 updates in
**24m45.754s**; own initials86.458/148.958 clear none. Development14009 remains
16,525/14,882.292 in25m36.935s. The fixed recipe passes primary greedy onone of
two learners; it has no successful fresh confirmation or broad reliability.

All912 pins,100k/9,340 recovery,357 finite/invariant tensors,48 initial tensor
and47 pre-update differences,full replays/videos,guards,curves,coverage,exposure
andcomparison pass. Main/post/collection succeed02:49:22.758/02:49:37.668/
02:50:08.343 UTC November2/sourcebff6eb4. Allfive original services succeed;
no training,frozen,replay,audit orcollector stage fails orrepeats. Separate
5.166s terminal reader passes02:50:48.666;completionSHA587f591e/summarySHA8d825bdf.
CPU preparation failures andseparate repairs remain preserved.

Only learner/evaluation seeds change. Keep12M64/Auto/N8/B16/T16/H5/R24/dynamics
1e-4/base-encoder-behavior4e-5/replay-value1/greedy/expert100,native447fe50/35ad0db,
runnere304af89 andteacherfa261c53. Actual overrides24.0885%;online15,187 and
late replay CE.068976 are not frozen competence. CPU co-host costs remain
uncorrected. Root`/x/Temp/kindle-qbert-replay-value1-seed16001-20261102` isclosed.
No extension,retry,third learner,tuning,seed substitution orautomatic successor
is selected. Do not repeat completed stages orreaders. No active allocation
remains atclosure;check live state. The five-game goal remains unfinished.

The [completed Breakout real-trajectory value diagnostic](2026-11-02-cdp-breakout-value-consistency.md) finds
sampled one-step residual RMSE.732/.941 and15-step2.546/3.121,while complete-return
RMSE remains6.682/6.691. Short targets bootstrap with the same critic; they do
not prove calibrated value or native-target fitting. Last15 sampled mean values
14.782/19.343 exceed realized.377/.592,an outcome-conditioned slice,not causal
proof. All32 diagnostic/73 source/912 active pins pass;6.366s CPU-only analysis
completes02:27:46.302/source9010df1. Original import failure/repair remain;
Qbert's original identity is unchanged. No new loss orlearner is selected.

The [completed Qbert replay-value scale1 comparison](2026-11-02-cdp-qbert-replay-value1.md) finishes200k actions /
799,786 frames /18,715 updates in**25m36.935s**. Greedy**16,525.000** improves
14,770.833 by11.88% andpasses15k/90% with24/24 first pyramids. Sampled14,882.292
improves14,254.167 by4.41%,also24/24 pyramids,but still misses15k. Training+1.79%;
initial-subtracted return/min+10.03% greedy/+2.63% sampled. Own initials147.917/
205.208 andzero pyramids match. Retain scale1 asstronger primary development;
reused14009 isnot independent confirmation. The five-game goal remains unfinished.

All859 pins,100k/9,340 recovery,357 control/bounded initial andinvariant frozen
tensors,47 pre-update events,1306/1338 paired initial semantic events,full replays/
videos,guards,curves,coverage,exposure,comparison andoriginal collector pass.
Main/post/collection succeed01:54:44.234/01:55:18.718/01:55:48.439 UTC November2,
source3a6655a. Allfive original services areterminal success; no GPU,replay or
collection stage fails orrepeats. Separate5.947s terminal reader passes01:56:36.021;
completionSHA048abefc,summarySHAf359e215. Preparation failures/repairs remain.

Only replay-value0.3→1 changes at12M64/Auto/N8/B16/T16/H5/R24/dynamics1e-4/base-
encoder-behavior4e-5/greedy/expert100. Critic andrepresentation gradients change;
real rewards/boundaries andH5 imagined-return bootstraps already supply targets.
No isolated path ormissing-loss defect isestablished. Native447fe50/35ad0db,
teacherfa261c53 andrunnere304af89 fixed. Actual overrides28.263%;online scores
andlower late imitation CE are not causal explanations. CPU co-host costs remain
uncorrected. Root`/x/Temp/kindle-qbert-replay-value1-20261102/full` isclosed.
No confirmation,extension,sweep,cap increase orautomatic successor isselected.
Do not repeat learner,observers,collector orfinal reader. No active allocation
remains atclosure; checklive state before GPU work.

The [pinned replay-target source review](2026-11-02-cdp-qbert-replay-value1.md#what-the-existing-replay-value-term-already-uses)
confirms240 real-replay targets/update already use observed rewards,intrinsic
bonus andactual last/terminal flags, withH5 imagined-return bootstraps andlambda.95.
Scale0.3→1 changes loss weighting,not targets; no missing-loss defect,target
accuracy or gameplay cause is established. Source-only.014s/.033s wrapper,
original process unchanged; no new model orGPU work. The source review preceded the now-completed comparison.

The [completed Qbert replay-value scale1 check](2026-11-02-cdp-qbert-replay-value1.md) passes8192 actions /
32,755 frames /733 updates in57.202s;1024 frozen/.855s haszero updates andno
natural completed episodes. Conservative200k projection25m36.196s clears30min.
All815 pins,357 control-matching initial/invariant frozen tensors,47 pre-update
semantic events,finite reports,full pixel/action/label replay andhost guards pass.
Main/collector succeed01:15:11.906/01:15:18.703 UTC November2/source921ad96;
originalPID877568/invocation7245ee2b08064c578610357b76b6fe60. Both services are
terminal success; no original GPU/replay/collector stage fails orrepeats.

Only existing replay-value scale0.3→1 changes at12M64/Auto/N8/B16/T16/H5/R24,
dynamics1e-4/base-encoder-behavior4e-5/greedy/expert100 on reused14009. Critic and
representation objectives both change; bootstrapped targets remain. Native
447fe50/35ad0db,teacherfa261c53 fixed; runner e304af89 adds only the fresh CLI.
No isolated critic cause,independent confirmation orgameplay gain is established.
All27 production/private CPU tests pass; original fixture/anchor failures and
separate repairs remain. Terminal readout2.769s passes01:15:58.242.

Root `/x/Temp/kindle-qbert-replay-value1-20261102/native-check` isclosed;
completionSHA91b0870d. No full learner,extension,sweep orautomatic successor is
selected. Do not repeat this check,collector orreader. No active allocation
remains; check live state before GPU work. The five-game goal isunfinished.

The [completed Breakout actor/critic-rate comparison](../results/2026-11-02-cdp-breakout-behavior8e5.md) finishes
200k actions /799,577 frames /12,477 updates in **21m44.517s**. Sampled **59.000**
and greedy **117.292** improve56.292/104.667 by4.81%/12.06%, for1.31% more
training time. Own initials1.417/2.208 match; all0/24 two-wall. Initial-subtracted
return/min improves3.58%/10.87%. Retain8e-5 as an incremental development option
and higher sampled point, not seed reliability or an isolated actor/critic cause.
The separate dynamics1e-4/behavior4e-5 point remains stronger greedy138.750.

All282 pins,100k/6,227 recovery,357 matched initial/invariant frozen tensors,
800/795 initial semantic events, full training/frozen replays/videos, guards,
curves, exposure, comparison and original collector pass. Main/post/collection
succeed **01:00:07.889/01:00:31.461/01:01:01.857 UTC November2**, source3099ff7.
All five original services are terminal success; no GPU or collection stage
repeats. Original CPU prelaunch failure and separate repair remain preserved.
Terminal reader passes1.979s; completionSHA7e0389c2, summarySHA76dfd6e3.

Only behavior4e-5→8e-5 changes; base/encoder4e-5, dynamics4e-4,12M64/Auto/N8/
B16/T16/H15/R16/expert100, sampled collection, native447fe50/35ad0db, teacher and
runner6c1b69fb stay fixed. Both actor and critic change; their frozen copies also
supply representation gradients. Actual overrides40.959%; late expert CE.551145
and aided online66.0 are not frozen competence. CPU co-host costs are uncorrected.
Root `/x/Temp/kindle-breakout-behavior8e5-20261102/full` is closed. Do not rerun
this learner, observers or terminal reader. No extension, rate sweep,
confirmation or automatic successor is selected. No Kindle GPU allocation remains
active at closure; check live state before work. The five-game goal is unfinished.

The [fresh Qbert confirmation](2026-11-01-cdp-qbert-dynamics1e4-seed14009.md)
finishes25m09.906s with14,770.833 greedy/14,254.167 sampled and24/24 first pyramids
each. Both miss15k. Dynamics1e-4 therefore passes greedy onone of two learners,
with no extension or substitute seed selected. Late imitation CE is lower than
the passing development seed (.064864 versus.069685), reinforcing that replay
fit alone does not rank frozen competence. All original services and audits pass;
no original stage fails or repeats. The five-game goal remains unfinished.

The [completed Breakout action-ranking comparison](../results/2026-11-02-cdp-breakout-current-ranking.md)
finds only8/128 origins with distinct15-step rewards. Pairwise agreement63.24%/
66.67% for dynamics4e-4/1e-4 doesnot translate into selected-return gain:−.027775/
−.035356 versus uniform, withlane intervals crossingzero. Paired difference
+.007581[−.015094,.030325] establishesno advantage. Short external-reward ranking
omits intrinsic reward andcritic bootstrap; long-term control remains unresolved.
No new learner,loss,inference planner,origin replacement orsuccessor isselected.

Both full captures total74,032 actions/296,076 source frames/115.848s,zero updates.
All71 pins,357 invariant tensors percapture,eight shared pixel hashes,128 origins/
2,432 candidate-control forecasts permodel,repeated-plan/live-state checks and
host guards pass. Original main succeeds00:18:33 UTC November2 andcollector
completion is00:18:46; both services areterminal success. No original stage fails
orrepeats. Native447fe50/35ad0db unchanged;66 CPU tests andsingle21.987s oracle
pass,withtwo earlier fixture failures preserved. Root
`/x/Temp/kindle-breakout-current-ranking-20261102` isclosed; no active allocation.
The five-game goal remains unfinished. Checklive state before GPU work.

The [completed Breakout critic diagnostic](../results/2026-11-02-cdp-breakout-current-critic.md)
finds weak sampled return prediction: correlation .105/.097 and R²−.031/−.267
for dynamics4e-4/1e-4. Both beat an earlier-episode constant, but neither beats
the hindsight test mean in squared error. Intrinsic rewards contribute13.51%/
11.50% of mean sampled discounted return. Greedy changes the target policy;
its bias−11.11/−12.94 is not a matched-policy training diagnosis. Different
visited states and noisy realized returns limit causal claims. No loss or new
learner is selected by this result.

Four frozen arms total145,632 actions/582,380 frames/154.699s,zero updates.
All73 pins,357 invariant tensors perarm,complete source semantics,source replay
hashes andhost guards pass. Original main succeeds00:06:14 UTC November2 and
collector completion is00:06:33; both services are terminal success. No stage
fails or repeats. Native447fe50/35ad0db andexact source runners stayfixed.
Root `/x/Temp/kindle-breakout-current-critic-20261101` isclosed; no active Kindle
allocation remains. No successor isselected; checklive state before GPU work.

The [current Qbert dynamics-group comparison](2026-11-01-cdp-qbert-dynamics1e4.md)
raises greedy12,678.125→15,471.875 andsampled12,093.750→14,862.500,each24/24
first pyramids,for.83% more time. Greedy passes15k; sampled does not. Late
imitation CE worsens.044040→.069685,so lower training loss would have ranked
these outcomes incorrectly. The4e-4→1e-4 rate change jointly affects recurrent,
posterior,prediction andexploration groups;no isolated mechanism follows.
Reused12011 is development evidence,not independent confirmation. All audits
complete through a separately retained CPU repair after the original controller
NameError; original main/collector remain failed,withzero repeated GPU stages.


The [updated current-run timing summary](2026-11-01-cdp-current-training-cost.md)
places81% of Qbert and88% of Breakout training wall time inside learning,versus
4–5% in emulator stepping. Within current Qbert H5/R24 updates,world training
is44%,posterior preparation24% andimagination20%;Breakout H15/R16 still spends
40% on imagination. This replaces older timings for describing current cost,
without claiming isolated GPU costs or a useful new optimization. The CPU-only
review preserves all606 active Qbert inputs and original process;its original
failed exact-timer-sum assertion and separate residual-accounting repair remain
recorded. No GPU stage repeats or new learner is selected.


The [fresh Qbert dynamics-group1e-4 comparison](../experiments/2026-11-01-cdp-qbert-dynamics1e4-200k.md)
launches22:32:25 UTC November1/source0d8ec52 with606 pins. Original mainPID
857991/invocation1e44edc5d824407b8a21cd62657f6ee2 and progress/recovery/post/
collection observers are active. Startup22:33:29 passes357 bounded/control
initial tensors,47 matching pre-update transitions, correct cadence and723 finite
reports through8088 actions. No full frozen result is available. Preserve the
original processes; do not restart or overlap another GPU allocation.

Fixed200k actions/18,715 updates/30min,100k/9,340 four-file recovery andfour
final/actual-initial sampled/greedy24-episode assessments remain. Only dynamics
4e-4→1e-4 changes recurrent/posterior/prediction/exploration rates together on
reused12011 at12M64/Auto/N8/B16/T16/H5/R24/encoder-base4e-5/greedy/expert100.
Native447fe50/35ad0db and teacherfa261c53 are unchanged; runnere161bf38 adds only
12 qualified CLI lines. Compare primary greedy12,678.125 andsampled12,093.750;
original15k/90% gate and failed confirmation remain. No reliability claim.

Full CPU qualification passes4.011s after preserving/correcting one retained-
control path failure: five CLI commands,32 config/five teacher/eight mode/17
header/four observer/12 recovery cases,six retention/two collector fixtures,
49 paths/17 scripts andall616 bounded/562 control pins. No GPU stage repeats.
The1.711s startup CPU co-host cost stays uncorrected. Root
`/x/Temp/kindle-qbert-dynamics1e4-20261101/full`; service
`kindle-qbert-dynamics1e4-200k-20261101`. No retry, cap increase, rate sweep,
extension or automatic successor. The five-game goal remains unfinished.

The [completed bounded Qbert dynamics1e-4 check](2026-11-01-cdp-qbert-dynamics1e4.md)
passes8192 actions/32,756 frames/733 updates in57.461s;1024 frozen/.853s haszero
updates/no natural episodes. Conservative200k projection25m43.153s clears30min.
All616 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
finite supervised reports, full pixel/action/label replay andhost guards pass.
Main/collector succeed22:27:02/22:27:09 UTC November1/source1a1eceb, original
PID857526/invocation655eec1976bd483e9f5b780a773a69bc. Both services are terminal
success; no original stage fails or repeats. No GPU allocation remains active.
Root `/x/Temp/kindle-qbert-dynamics1e4-20261101/native-check` iscomplete; do not
rerun or reuse its training experience. Gameplay benefit remains unmeasured.

The [completed Qbert RGB64 action-ranking comparison](2026-11-01-cdp-qbert-rgb64-ranking.md)
finds10007/12011 overall gain over uniform+6.726/−1.367 on shared10007 histories;
paired difference+8.092 has lane95% interval[−.562,19.772]. Later gains+2.227/
+1.842 differ+.385[−1.519,2.293], withonly13/64 informative later origins.
12011 has lower reward MAE21.721 versus38.696 overall,26.914 versus56.163 later.
Prediction accuracy and action quality differ; no established paired advantage,
causal gameplay explanation, new loss or learner follows from this diagnostic.

All64 pins,128 origins/2,432 candidate-control forecasts per model, exact source
transitions, eight shared CPU/native pixel hashes,357 invariant tensors per
capture, repeated-plan/live-state checks, host guards and original collector
pass. Full pair156,736 actions/626,854 source frames/181.731s,zero updates.
Both128-action checks and53 CPU tests pass; the single30.410s CPU oracle checks
78,368 actions/154,591 branch frames andeight reverse orders. Native447fe50/
35ad0db stays fixed; the explicit probe flag permits only seed12011 to differ.

Main/collection succeed22:17:28/22:17:35 UTC November 1/source4aa6ac7, original
PID856594/invocation315da71b11fb44e49522fc7bf72708fa. Both services are terminal
success; no failure, repeated stage or build. Scalar posterior draws differ
across models and from the original vector actor;15-step held plans exceedH5
training and do not measure adaptive policies or independent learner reliability.
Root `/x/Temp/kindle-qbert-rgb64-ranking-20261101` is complete; do not rerun.
No GPU allocation remains active. Original training costs and one-of-two gate
outcome stay unchanged. No retry, cap increase, extension or automatic successor.
The five-game goal remains unfinished.

The [completed Qbert RGB64 critic diagnostic](2026-11-01-cdp-qbert-rgb64-critic.md)
finds useful predictions in both learners: sampled R² .689/.558 for 10007/12011,
with RMSE 813.017/896.689 versus constant 1,460.945/1,351.585. Later biases differ:
−99.585/+211.312. Both underestimate before the first pyramid. These own-policy
states and noisy returns do not establish a causal control limitation or correct
alternative-action ranking. Mean intrinsic returns are .041–.051% of total;
this is not gradient attribution or a selected coefficient change.

Four captures exactly reproduce original trajectories/scores: 252,312 actions /
1,009,070 frames / 263.020s, zero updates/truncations. All 64 pins, 31,743 semantic
events, 357 invariant tensors per arm, finite live values/bonuses, host guards
and original collector pass. The preceding 1,024-action check and nine CPU tests
pass. Native447fe50/35ad0db and runneraba4300d stay fixed; source replays are
reused only after exact semantics and log hashes match. No live-image equality
is claimed where source logs lack pixel hashes.

Main/collection succeed 22:06:42/22:06:52 UTC November 1, sourcee6340ac, original
PID852818/invocation8b849586cc3241a2a52e5317e2f10c88. Both services are terminal
success; no failure, repeated stage, learner or build. Original training times
25m09.797s/25m27.366s and one-of-two gate outcome remain unchanged. Root
`/x/Temp/kindle-qbert-rgb64-critic-20261101` is complete; do not rerun. No GPU
allocation remains active at closure. No loss change, retry, extension, cap
increase or automatic successor is selected. The five-game goal remains open.

The [completed Qbert RGB64 cross-cohort assessment](2026-11-01-cdp-qbert-rgb64-cross-cohort.md) keeps10007
ahead on both cohorts:15,710.417/16,218.750 versus12011's11,982.292/12,678.125.
Alltrained cells have24/24 first pyramids;10007 passes15k on both,12011 on
neither. Initial-subtracted gaps3,883.333/3,705.208 persist. The original12011
confirmation failure remains; this adds no independent learner or reliability.

All611 pins,357 finite invariant tensors per arm, full outcome replays/videos,
host guards andoriginal collector pass. Four new arms total140,488 actions/
561,791 frames/117.899s,zero updates/truncations; completed diagonal GPU stages
do not repeat. Main/collection succeed21:51:01/21:51:22 UTC November1/source
63dd44a,originalPID850983/invocationef4177111f444dea8f2d905f7f800544. Both
services are terminal success; no stage fails or repeats. Native447fe50/35ad0db,
runneraba4300d andcheckpoint configs remain fixed. Original training times
25m09.797s/25m27.366s and CPU co-host caveats remain unchanged.

Root`/x/Temp/kindle-qbert-rgb64-cross-cohort-20261101` is complete; do not rerun.
No GPU allocation remains at closure. No learner, loss, weight sweep, extension,
retry or automatic successor is selected. The gap persists across cohorts;
current phase evidence does not yet distinguish representation, optimization,
teacher quality or coverage of difficult decisions. The five-game goal remains
unfinished; check live state before further work.

The [completed Breakout dynamics-group comparison](2026-11-01-cdp-breakout-dynamics1e4.md) is mixed:
sampled50.625 falls10.07% from56.292, while greedy138.750 rises32.56% from104.667.
Allfour candidate arms havezero of24 two-wall successes; own initials1.417/2.208
match. Training200k actions/799,574 frames/12,477 updates takes21m48.095s,
1.58% longer. Retain4e-4 for the sampled-primary recipe;1e-4 is a distinct greedy
development point, not an overall winner or reliable task success.

All272 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
800/795 initial semantic traces,100k/6,227 four-file recovery,full replays/videos,
guards,curves,exposure,comparison andoriginal collector pass. Main/post/collection
succeed21:42:21/21:42:32/21:43:05 UTC November1/sourceb9ce97f. Allfive services
are terminal success. Two extra read-only summary queries fail on exposure-field
and observer-journal assumptions; corrected summary passes without changing
original completion or repeating any original stage. Actual overrides51.8185%;
late replay expert CE worsens.558097→.670023. Guided scores are not competence.

Only dynamics-group4e-4→1e-4 changes recurrent/posterior/prediction/exploration
rates together;encoder/base/actor4e-5 and12M64/H15/R16/expert100 stayfixed.
Native447fe50/35ad0db andteacher remain unchanged;runnerfe5cf347 has the qualified
12-line CLI addition. Reused1009/cohort and CPU co-host caveats remain. No native
build,extension,cap change,confirmation orautomatic successor. Root
`/x/Temp/kindle-breakout-dynamics1e4-20261101/full` is complete; do not rerun.
No GPU allocation remains at closure. The five-game goal remains unfinished;
the separately selected Qbert assessment above is now also complete.

The [current Qbert RGB64 phase analysis](2026-11-01-cdp-qbert-rgb64-phases.md) finds weaker12011 has
higher final50k native movement agreement75.22% versus10007's70.35%, despite
lower frozen greedy12,678.125 versus15,710.417. Both collect about78% of
training actions after the first pyramid. All96 frozen prefixes score550;
the gap develops afterward. Duration and reward rate differ, without an
identified cause. No new loss, learner or coverage intervention is selected.

All400k training actions/1,599,654 frames,235 completed episodes/16 tails and96
frozen episodes reconcile. All26 diagnostic/272 active inputs, two sets of six
boundary cases and hand-counted trace/partial/override fixtures pass. CPU3.601s
(3.622s wrapper) completes21:27:09 UTC November1/source3219fb7, preserving the
original Breakout PID/invocation. No native/emulator import, GPU work, replay,
fitting, update, failure or repeat. Keep co-host time without correction.
Root`/x/Temp/kindle-qbert-rgb64-phases-20261101` is complete; do not rerun.

The [bounded Breakout dynamics-group1e-4 check](2026-11-01-cdp-breakout-dynamics1e4.md) passes8192 actions/
32,733frames/489 updates in47.413s;frozen1024/.856s haszero updates/no natural
episodes. Conservative200k projection21m13.289s passes30min;gameplay benefit
remains unmeasured. Dynamics-group4e-4→1e-4 changes recurrent,posterior,prediction
andexploration rates together;encoder/base/actor4e-5 andother12M64/H15/R16/
expert100 settings stayfixed. Native447fe50/35ad0db isunchanged,no build.

All250 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
489 finite reports,full guided replay andguards pass. Original service succeeds
21:08:45/collector21:08:56 UTC November1/sourcee333fa0,PID847195/invocation
03896c77109f470ea5f1097dc8b92477. An extra post-closure parent-PID journal assertion
fails because children log;separate invocation review passes without changing
original completion orrepeating anyGPU/replay/collector stage. All471 runner tests
and6.533s CPU qualification pass. Root
`/x/Temp/kindle-breakout-dynamics1e4-20261101/native-check` iscomplete;do not rerun.
No full learner was selected by the bounded check; the separate full life above
is now complete.
The five-game goal remains unfinished.

The [fresh Qbert RGB64 seed12011 confirmation](2026-11-01-cdp-qbert-rgb64-seed12011.md) completes but misses the
unchanged15k gate: greedy12,678.125 andsampled12,093.750,each24/24 first pyramids.
Own initials168.750/167.708 clear none. The fixed recipe passes onone of its two
learners (10007greedy15,710.417); no reliable Qbert success or third learner is
selected. Only learner/evaluation seeds change;this is not a paired geometry test.
Training200k actions/799,831frames/18,715 updates takes25m27.366s,within30min.

All562 pins,357 own-bounded initial/invariant frozen tensors,47 pre-update events,
18,715 finite reports,100k/9,340 retained four-file recovery,full replays/videos,
guards,curves,coverage,exposure,comparison andoriginal terminal collector pass.
Four frozen arms131,048 actions/524,038frames/111.169s havezero updates/truncations.
Main/post/collection succeed20:55:58/20:56:18/20:56:51 UTC November1/source34c6a88.
No failure,extension,cap change,GPU/replay/collector repeat or build. Actual
44,539 overrides (22.2695%) and online scores are not frozen competence.
CPU co-host timings remain without correction. Root
`/x/Temp/kindle-qbert-rgb64-seed12011-20261101/full` is complete;do not restart it.
No Kindle service or GPU allocation is active at closure. The five-game goal
remains unfinished;check live state before the separately selected diagnostic.

The [completed exact Breakout policy capture](2026-11-01-cdp-breakout-policy-readback.md) reproducesall37,016 actions/
148,038frames in30.672s,zero updates. Across12,763 disagreements on31,161
selected decisions,mean chosen/teacher probabilities are.634/.228;2,713 teacher
commands getbelow.1. The120 life-clipped windows/3,816 rows show similar.634/.224
on1,378 disagreements. Sustained earlier teacher rescue108/120 does not prove
individual labels optimal or identify a loss change. Final window actions areall
FIRE;do not interpret late aggregate agreement as improved paddle control.

All268 pins,357 invariant tensors perstage,115 repeated readbacks,128 F64/native
probabilities (max9.22e-7),exact smoke/full prefixes,pixels/outcomes/lives andguards
pass. Eight final output images remain unrecorded. Six CPU fixtures pass;analysis
.074s. Original service succeeds20:58:35/collection20:58:53 UTC November1,
source8481d9c,PID845612/invocation831e211938b14d1da284cdefb14e46c6. No failure,
build,retry,learning or repeated stage. Root`/x/Temp/kindle-breakout-policy-readback-20261101`
is complete;do not rerun. No successor selected orGPU allocation active atclosure.
The five-game goal remains unfinished;check live state before further work.

The [fixed current-pixel action readout](2026-11-01-cdp-breakout-pixel-action-readout.md) fails its screen:
learned+pixel kernel57.40% overall/55.15% near-descending teacher agreement,
versus learned-only56.91%/58.82% and actor62.56%/66.91%. Pixel-only48.25%/43.38%.
Keep the actor;no native spatial path,head,loss,teacher or learner selected.
All2064 aligned rows,51 diagnostic/562 active Qbert pins andsix CPU fixtures pass.
Analysis.760s/controller3.410s completes20:30:08 UTC November1/source6a6ce27,
preserving originalPID842010/invocationd37840f258104f9caf8b6bc5c0316899.
No replay,native/emulator/GPU work,updates,failure or repeated fit. The additive
kernel and reused teacher/cohort limit interpretation;no broad spatial-input
claim. Retain CPU co-host timing. Root`/x/Temp/kindle-breakout-pixel-action-readout-20261101`
is complete;do not rerun. Qbert confirmation continues;the five-game goal is open.

The [completed Qbert12M/RGB64 comparison](2026-11-01-cdp-qbert-rgb64.md) reaches **15,710.417 greedy /24 first pyramids**,
passing the unchanged15k/90% gate and improving RGB48's14,339.583 by9.56%.
Sampled14,250/24 rises3.55% from13,761.458 but still misses15k. Own initials
1.042 greedy/195.833 sampled clear none;cross-geometry traces differ. Retain this
single-learner development point;reused10007/cohort does not establish reliability.
No extension,automatic confirmation or successor is selected by this result.

Training completes200k actions/799,823frames/18,715 updates in**25m09.797s**,
within30min. Total time falls11.90% versus the original slower-teacher control;
initial-subtracted gain/min rises24.35% greedy/17.38% sampled. Recorded learning
work rises16.73%;the faster teacher also changes acting cost,so no isolated
resolution/native speed claim. Keep12M64/H5/R24/encoder4e-5/base behavior4e-5/
greedy collection/expert100/native447fe50/runneraba4300d/teacherfa261c53.
Geometry/head shapes/initialization change together;older4M64 negative remains.
Late imitation CE.051060→.049491;44,600 actual overrides (22.3%) are not competence.

All521 pins,357 bounded-matching initial/invariant frozen tensors,47 pre-update
transitions,100k/9,340 retained four-file recovery,full replays/videos,guards,
curves,coverage,exposure,comparison and original collector pass. All five services
succeed:main20:11:19/post20:11:52/collection20:12:24 UTC November1/sourceb8de34f.
Four frozen arms156,312 actions/625,086frames/146.320s,zero updates/truncations.
No failure,cap change,GPU/replay/collector repeat or build. CPU co-host caveats
remain without time correction. Root `/x/Temp/kindle-qbert-rgb64-20261101/full`
is complete;do not restart it. No GPU allocation remains active at closure.
Check live state before future work;the five-game goal remains unfinished.

The [fixed Breakout one-step risk controller](2026-11-01-cdp-breakout-one-step-risk.md) lowers mean return
225.083→184.417 (−18.07%), with three of eight lanes improved and both0/24
two-wall successes. Retain the original teacher;no fresh cohort,penalty retry,
integration or learner. The penalty changes39 of1,355 eligible decisions on
38,223 selected candidate states;this is not original-trajectory disagreement.
All36 diagnostic/521 active Qbert inputs,43,056-action/172,188-frame exact
action/debug/live-pixel/outcome replay and video pass. Six CPU fixtures pass;
nonfatal test ResourceWarning retained. Collection26.617s/replay32.202s;
controller61.624s completes19:52:07 UTC November1/source316a857, preserving
originalQbertPID836919/invocation425aae9cee2541fe9623b753225af57f.
No failure,repeat,native/GPU work or update;retain CPU co-host timing.
Root`/x/Temp/kindle-breakout-one-step-risk-20261101` is complete,do not rerun.
The error-prediction result remains;this use of it did not improve gameplay.
Qbert RGB64 continues;the five-game goal remains unfinished.

The [fixed Breakout response-error check](2026-11-01-cdp-breakout-response-uncertainty.md) passes its
prediction screen: command-conditioned error-forecast MSE falls 5.04% overall
and 4.49% near descending balls, with lower Gaussian NLL and both scores better
in all eight lanes. Stationary-edge MSE worsens and predicted error²4.809 remains
far below observed18.383. No controller or gameplay gain is established.
All37,448 rows/control metrics,16 diagnostic and521 active Qbert inputs pass;
.772s analysis/3.262s controller completes19:46:15 UTC November1/source1bbd75e,
preserving originalPID836919/invocation425aae9cee2541fe9623b753225af57f.
Four CPU fixture groups pass;no failure,repeat,native/emulator/GPU work or update.
Keep CPU co-host timing. Root`/x/Temp/kindle-breakout-response-uncertainty-20261101`
is complete;no controller,integration,sweep or learner is selected.
Qbert RGB64 continues;the five-game goal remains unfinished.

The [command-dependent Breakout teacher comparison](2026-11-01-cdp-breakout-command-response.md) is complete.
Fresh mean return 250.875 → 273.125 rises 8.87%, with five of eight lane means
improved, but misses the declared 10% requirement. Both arms have zero of 24
two-wall successes. Retain the current teacher; no integration, retry, further
cohort or learner is selected. Development 225.083 → 295.333 (+31.21%, six
lanes) remains a separate reused-cohort result. All 41 fresh inputs, unchanged
policy/response bytes, exact action/debug/live-pixel/outcome replays and both
videos pass. Control55,280 actions/221,090 frames/32.304s; candidate64,560/
258,195/35.377s. Full controller144.403s completes 19:26:13 UTC November 1,
sourcef24cd1a. No failure, refit, cap change, repeated stage, native/GPU work or
update. The .470s fixed prediction fit and 72.478s development comparison also
remain complete. Do not rerun these stages. Root
`/x/Temp/kindle-breakout-command-fresh-20261101`; no active allocation.
The five-game goal remains unfinished; check live state before future work.

The [completed Qbert encoder8e-5 comparison](2026-11-01-cdp-qbert-encoder8e5.md) lowers both frozen
scores: greedy 14,339.583 → 12,435.417 (−13.28%) and sampled 13,761.458 →
11,213.542 (−18.51%), with 23/24 and 24/24 first pyramids. Both means miss
15,000. Actual-initial scores 1.042/171.875 reproduce, with zero pyramids.
Retain encoder4e-5 and the faster teacher. No rate extension, sweep,
confirmation or successor is selected. Training completes 200k actions /
799,800 frames / 18,715 updates in 20m36.093s. Time falls 27.87%; initial-
subtracted gain/minute rises 20.22% greedy and 12.64% sampled, but teacher CPU
cost also changes. This does not establish a rate-induced speed benefit.
Late imitation CE rises .051060 → .052894. Actual 44,160 overrides (22.08%)
and online guided scores are not frozen competence.

All 528 inputs, 357 matching/invariant tensors, 47 pre-update events,
100k/9,340 four-file recovery, full replays/videos, guards, curves, coverage,
exposure, comparison and the original collector pass. All five services
succeed: main 19:01:36, postprocessing 19:01:43 and collection 19:02:15 UTC on
November 1, source ea34785. Pre-launch preparation and CPU selector failures
remain preserved; one repaired CPU preflight rerun passes. No full-run or
GPU/replay/collector failure or repeat occurs. Native447fe50, runneraba4300d,
12M48/H5/R24/base behavior4e-5 remain fixed. The reused10007 cohort and teacher
implementation change limit interpretation. Retain CPU co-host timing without
correction. Root `/x/Temp/kindle-qbert-encoder8e5-20261101/full` is complete;
do not rerun it. No Kindle service or GPU allocation is active. The five-game
goal remains unfinished; check live state before future work.

The [Breakout full-width paddle diagnostic](2026-11-01-cdp-breakout-paddle-crop.md) acceptszero of42,303
selected decisions:24 initial/reset,37,960 extra-red,4,285 image-edge and34
nonstandard exclusions. Thus old-crop bias remains unquantified;do not widen
the teacher crop on this evidence. All52,072 recorded actions/208,254frames,
original actions/debug/pixels/outcomes and208,254 prior raw annotations match.
The source stays225.083/zero two-wall;no new score,policy,loss or learner.
Eight CPU fixtures/47 diagnostic pins/528 active Qbert pins pass. Replay53.457s/
controller56.159s completes18:41:41 UTC November1/source7433689,preserving
originalPID831942/invocationbae3fe2dd4a24070ab2845a7828145fa. Retain CPU co-host
cost without correction. No GPU work,failure or repeated stage;no relaxed filters.
Root`/x/Temp/kindle-breakout-paddle-crop-20261101`;complete,do not rerun.

The [completed Qbert300k comparison](2026-11-01-cdp-qbert-300k.md)
finishes300,000 actions/1,199,763frames/28,090 updates in31m18.803s,+9.64% time
versus200k with the faster teacher. Greedy14,339.583→13,861.458 falls3.33%;
sampled13,761.458→8,668.750 falls37.01%. First pyramids are24/24 greedy and
18/24 sampled;both means miss15k. Own initials1.042/171.875 reproduce,zero
pyramids. Final50k-action imitation CE.051060→.037849 improves without frozen
benefit. Retain200k/12M48/H5/R24/encoder4e-5/base behavior4e-5/greedy collection/
expert100 and the faster teacher;no budget extension,confirmation or successor.
All524 inputs,357 initial/invariant tensors,47 pre-update events,both100k/9,340
and200k/18,715 recoveries,full replays/videos,guards,curves,coverage and comparison
pass. Main succeeds18:22:35 UTC November1/source7bd5001. Original postprocessor
fails18:22:46 on stale200k exposure assertions;collection records failure18:22:47.
Separate6.491s CPU repair completes18:24:23,all9 repair pins/95 original artifacts
unchanged,zero GPU or completed CPU-stage repeats. Original failed services and
ancillary CPU reporting failures remain;closure`complete_with_cpu_postprocess_repair`.
Actual56,103 overrides (18.701%) and online guided scores are not frozen competence.
Extra experience and updates remain coupled on reused10007;no reliability claim.
CPU co-host caveats stay without correction. Root`/x/Temp/kindle-qbert-300k-20261101`,
authoritative`postprocess-repair/completion-review.json`. All services terminal;
no GPU allocation active,no restart. The five-game goal remains unfinished.

The [completed CPU prefix comparison](../results/2026-11-01-cdp-qbert-budget-prefix.md)
finds all25,000 transitions,18,715 non-timing reports and118 episode records
identical through200k actions/799,841frames,with357 elementwise-equal200k arrays.
The100k control has metadata only;serialized hashes differ at both milestones.
Original bare-key failure is preserved;the group-qualified repair completes
18:06:05 UTC in4.883s/4.921s wrapper. A separate.902s cumulative-frame-count
correction completes18:08:25 without repeating the comparison. Final closure
18:09:12 is `complete_with_cpu_reporting_repairs`. All25 repair/524 active pins
and originalPID826399/invocation stay. No native/GPU work or updates;retain CPU
co-host time without correction. Root `/x/Temp/kindle-qbert-budget-prefix-20261101`;
complete,do not rerun. Extra experience and updates remain coupled;no new frozen
score or independent reliability follows.

The [completed Qbert25M/R16 comparison](2026-11-01-cdp-qbert-25m-r16.md)
finishes200k actions/799,786frames/12,477 updates in23m40.953s,17.08% less time
than12M/H5/R24. Primary greedy14,339.583→10,696.875 falls25.40%;sampled
13,761.458→11,967.708 falls13.03%;all24/24 first pyramids,both candidate means
miss15k. Own initials39.583/173.958 clear none and differ across capacities.
Own-initial-subtracted return/minute falls10.36% greedy and rises4.66% sampled.
Retain12M/H5/R24 and the faster teacher;no larger-recipe promotion or extension.
All521 inputs,357 bounded-matching/invariant tensors,retained100k/6,227 recovery,
full replays/videos,guards,curves,coverage,exposure,comparison and original
collector pass. All five services succeed:main17:25:24/post17:25:48/collection
17:25:55 UTC November1/source65f4cee. Original CPU selection/blocked-launch
failures and one-basename repair remain;no GPU or completed-stage repeat.
Capacity,initialization,ratio,total work and teacher CPU cost change together;
no isolated cause or reliability. Actual48,158 overrides (24.079%) are not
frozen competence. CPU co-host caveats remain without correction. Root
`/x/Temp/kindle-qbert-25m-r16-20261101/full`;complete,do not rerun. No Kindle
service or GPU allocation remains active. No retry,confirmation or automatic
successor selected;the five-game goal remains unfinished. Check live state.


The [existing centroid-target CPU reference](2026-11-01-cdp-centroid-target-precision.md)
has ball x/y MAE.058/.225 pixels on555 shared test rows,versus saved-head error
5.461/5.206 against that target. Both target and earlier pixel decoder miss the
same2/557 reference balls;target haszero extra detections versus4. The crop
difference does not explain most of this old encoder6e-6 head error;GPU readback,
current-model accuracy and gameplay causality remain unmeasured. All20,960
recorded actions/83,799frames,2,064 prior pixel predictions,27 inputs and521 active
Qbert inputs pass;originalPID821599/invocation stays. CPU-only11.769s completes
17:00:05 UTC November1/source79701cc,zero GPU work,updates,failures or repeats.
Retain CPU co-host time;no crop,loss or learner selected. Root
`/x/Temp/kindle-centroid-target-precision-20261101`;complete,do not rerun.


The [saved Qbert training-phase diagnostic](../results/2026-11-01-cdp-qbert-training-phases.md)
finds76–77% of collected actions after the first pyramid across three H5/R24
learners,rising to87–89% in the final50k. Late native movement agreement is100%
before the milestone but70–72% afterward;overrides are excluded. Broad absence
of later-phase exposure is unsupported,while difficult-state coverage,fitting,
representation and teacher limitations remain unresolved. All600k actions,
357 completed episodes,24 partial tails and17 inputs reconcile. CPU-only3.440s
completes16:34:52 UTC November1/source2365121,zero replay,GPU work,updates or
failures. No learner/intervention selected;host idle. Root
`/x/Temp/kindle-qbert-training-phases-20261101`;complete,do not rerun.

The [completed Breakout B8/T32 comparison](../results/2026-11-01-cdp-breakout-context32.md)
finishes200k actions/799,549frames/12,469updates in23m12.128s,8.11% longer than
B16/T16. Primary sampled56.292→48.583 falls13.69%;greedy104.667→130.417 rises
24.60%,all0/24 two-wall. Retain B16/T16 for primary sampled performance and
B8/T32 as a stronger greedy development point;no overall win or reliability.
All292 inputs,357 matching/invariant tensors,retained100k/6,219 recovery,paired
initial traces,full replays/videos,guards,curves,exposure and original collector
pass. All five services succeed:main16:26:03/post16:26:09/collection16:26:47 UTC
November1/source4f7b4c4. Zero failures,GPU repeats or repairs. Native447fe50,
runner6c1b69fb and teacher stay fixed;batch,temporal gradients,diversity,target
rows and warmup change together. CPU co-host caveats remain without correction.
Root`/x/Temp/kindle-breakout-context32-20261101/full`;complete,do not rerun.
No Kindle service or GPU allocation remains active. No extension,context retry,
confirmation or automatic successor selected;the five-game goal is unfinished.

The [saved centroid-head CPU diagnostic](../results/2026-11-01-cdp-centroid-head-readout.md)
finds5.460/5.275-pixel ball x/y MAE versus9.267/8.064 for the earlier linear
recurrent probe on555 shared test rows. The existing head recovers more precise
position than those probes;substantial error remains. This is the old encoder6e-6
centroid checkpoint,not the current learner;its original gameplay protocol
failure stays. CPU F64 reconstruction has no independent native-forward check.
Native target and pixel-reference crops differ by one boundary row;actual target
accuracy is unmeasured. All29 diagnostic/292 then-active Breakout inputs and
original process identity pass. Analysis2.640s (2.696s wrapper) completes16:21:52
UTC November1/source5318dc9,zero new replay,fitting,GPU work,updates or failures.
Retain co-host time;no new head,loss or learner selected. Root
`/x/Temp/kindle-centroid-head-readout-20261101`;complete,do not rerun.

The [saved Qbert behavior-rate outcome diagnostic](2026-11-01-cdp-qbert-behavior-phases.md) retains the425-point
first-pyramid failure:its slot loses15,275 points while the other23 gain15,725,
leaving only+18.75 official greedy mean. Twelve other slots improve,11 worsen;
conditional successful mean14,964.130 still misses15k. Late replay CE rises
.051060→.055250 (+8.21%). Five successful candidate episodes share the failure's
first197 commands;no simple command-level cause is established. All96 rows,
22 diagnostic/292 active Breakout inputs and originalPID812861/invocation stay.
Analysis1.719s completes16:06:19 UTC November1/sourcec197078,zero replay/native/
GPU work,updates,failures or repeats. Retain CPU co-host time;no new rate,loss
or learner selected. Root`/x/Temp/kindle-qbert-behavior-phases-20261101`;complete,
do not rerun. The Breakout comparison above subsequently completes.

The [completed Qbert behavior-rate comparison](2026-11-01-cdp-qbert-behavior8e5.md)
reduces observed training28m34s→20m52s (26.92%),with nearly unchanged greedy mean
and3.00% higher sampled mean,both below15k. Acting/feedback566.324→184.264s accounts
for82.83% of the time reduction. Native and runner stay fixed;equivalent teacher
bytes and behavior rate change together,so this is not an isolated teacher,rate
or backend-speed estimate. Retain the faster teacher,not a higher-rate default.

The [saved Breakout takeover-state join](2026-11-01-cdp-breakout-takeover-states.md) finds onlytwo of21 earlier-only
rescues have projected contact withinfive actions at delay16;14 already match
the teacher command. Timing estimates and this imperfect teacher do not isolate
the cause. All480 origins/478 saved-branch survival recounts,nine declared inputs
and524 active Qbert inputs pass;originalPID809245/invocation stays. Analysis2.766s
completes15:23:52 UTC November1/source0e77ede,recount15:24:44. Zero emulator/native/
GPU work,updates,failures or repeats;retain CPU co-host timing. No new loss,teacher
or learner selected. Root`/x/Temp/kindle-breakout-takeover-states-20261101`;complete,
do not rerun. The Qbert behavior-rate comparison subsequently completes;see above.

The [Qbert teacher CPU diagnostic and equivalent predicate change](2026-11-01-cdp-qbert-teacher-cost.md)
identify avoidable adapter work:RGB equality reductions consume53.78% of profiled
snapshot time. Direct channel conjunctions preserve all472 snapshot decisions/
states and200k recorded labels/trajectory outcomes,with CPU replay633.898→230.331s.
The passing130-test/25-input qualification and unchanged native447fe50 runtime
are packaged for future Qbert trials. This addresses compute cost;the subsequent whole-training comparison above
measures lower cost with similar frozen quality. Original harness failure and
timing limits remain;no isolated teacher-speed or behavior-rate claim follows.

The [completed Qbert H3/R24 comparison](2026-11-01-cdp-qbert-h3-r24.md) reduces
training time6.56% at fixed200k/18,715 updates,but lowers greedy14,339.583→13,823.958
and sampled13,761.458→11,277.083,each24/24 first pyramids. Both still miss15k.
Initial-subtracted greedy return/minute rises3.18%,while sampled falls12.54%;
retain H5 for stronger absolute primary score and H3 as a cheaper point. This
narrows the short-imagination tradeoff without identifying a model-error or
credit-assignment cause. All506 inputs,recovery,frozen/replay audits and original
services pass;complete14:53:44 UTC November1,zero GPU repeats or successor.

The [fresh fixed Breakout centering cohort](2026-11-01-cdp-breakout-unseen-center-fresh.md)
scores209.833→212.042 (+1.05%),withfour of eight lane means better andzero two-wall
for both controllers. It fails the10%/five-lane screen;the old cohort's19.88%
gain does not carry over at the declared threshold. Keep the qualified teacher;
no integration,parameter retry or cohort sweep follows. Both full exact replays,
31 declared/506 active learner inputs and original Qbert identity pass. CPU-only
135.931s completes14:26:31 UTC November1/sourceb7edcf8,zero updates/failures/repeats.
This tests a fixed reference change,not learned-policy improvement or a general
claim against earlier positioning. Retain CPU co-host time without correction.

The [fixed H5/R24 Qbert temperature0.5 assessment](2026-11-01-cdp-qbert-h5-r24-temperature.md)
also lowers mean return:12,840.625 versus greedy14,339.583 and sampled13,761.458,
each24/24 first pyramids. Exact default compatibility and all frozen/replay audits
pass. Retain greedy;this inference change does not close the later-play score
gap or establish an architectural cause. No learning or repeated stage occurs.

The [completed400k/R8 comparison](2026-11-01-cdp-breakout-r8.md) doubles Breakout
experience at almost fixed updates but adds12.06% time:sampled56.292→56.708,
greedy104.667→97.583,allzero two-wall. Learning time stays near19min;acting and
environment costs roughly double. Retain200k/R16;more experience at this cadence
does not improve learning per minute. Changed sampling,warmup and exposure
prevent an isolated coverage conclusion.

The [CPU unseen-ball centering reference](2026-11-01-cdp-breakout-unseen-center.md)
raises expanded24 mean225.083→269.833,but the gain comes from the original eight
screening episodes. The added16 average244.938→238.250,andboth controllers still
clearzero two-wall tasks. This identifies a useful earlier-positioning lead,
not independent evidence for changing the training teacher. Exact replay passes;
no native/GPU work or learner updates. Keep the qualified teacher unchanged.

The [fixed Breakout upper-crop reference](2026-11-01-cdp-breakout-crop70.md)
leaves all eight returns unchanged:mean185.375,zero two-wall successes. The
10%/five-lane screen fails;retain the teacher,no crop retry or24-episode expansion.
All16,280 actions/65,107 frames replay exactly;26 declared/490 active Qbert pins
and original process identity pass. Seven fixtures,2.499s preflight,10.639s
collection/13.219s replay/26.484s controller;complete12:41:57 UTC November1,
source7b12a46. Separate.139s log comparison finds identical actions/pixel hashes/
rewards/boundaries and zero accepted detections above y90;96 debug rows differ.
No native/GPU work,updates,failures or repeated stages. Keep co-host timing;
root`/x/Temp/kindle-breakout-crop70-20261101`. Do not rerun.

The [saved-frame pooling diagnostic](2026-11-01-cdp-breakout-pool-timing.md)
finds exact two-frame midpoint paddle measurements on35,193 accepted pairs.
Latest-frame discrepancy averages1.456px;the response model still has1.965px
MAE against its calibrated pooled target on31,817 matched rows,versus2.875px
against latest raw detections. This separates temporal averaging from the
remaining prediction residual without identifying a causal gameplay limitation.
All exclusions remain;no current-frame correction or teacher change is selected.
The1.516s CPU analysis adds no emulator replay,GPU work or learning.

The [fixed Breakout response-history fit](2026-11-01-cdp-breakout-response-history.md)
also fails its prediction screen:adding one earlier displacement/command lowers
RMSE but raises MAE in both held-out-all and near-descending subsets. Keep the
current teacher;no gameplay test follows. This rejects one linear mean-response
model,not the prior ambiguity finding or all history/uncertainty approaches.
All56 inputs and seven tests pass,CPU-only,withno active learner at completion.

The [completed Breakout position-loss comparison](2026-11-01-cdp-centroid-encoder4e5.md)
learns its auxiliary training target but lowers sampled-primary return56.292→35.542,
while greedy rises104.667→112.750;all0/24 two-wall. Training costs21m47s,+1.50%.
Retain the no-centroid control. Low auxiliary loss does not establish held-out
position accuracy or better control;native/runner differences also prevent an
isolated loss-effect claim. All original stages and541 inputs pass;the separate
terminal verifier's journal-format failure is preserved and repaired CPU-only.
No learner or allocation is active at closure;the five-game goal is unfinished.

The [latest retained-log Qbert phase check](2026-11-01-cdp-qbert-h5-r24-phases.md)
places the H5/R24 greedy seed score gap entirely after the first-pyramid crossing
transition:both learners have identical1219-frame milestones and550-point
prefixes inall24 episodes. Later duration and reward rate both vary;neither
is established as the cause. On an older4013 failure-feature corpus,a
[fixed lane-held-out linear readout](2026-11-01-cdp-qbert-failure-linear-readout.md)
trails the actor on movement decisions41.02% versus52.34% and on every lane
overall. This does not support that head replacement or establish information
absence. Both checks were CPU-only and preserved the then-active Breakout centroid4e-5
comparison. See their reports for supervision,selection and co-host
limits;the newer H5/R24 representations are not measured by the old-feature probe.

The [retained Breakout planning-cost analysis](2026-11-01-cdp-breakout-plan-costs.md)
finds56.38% of12,521 horizontal disagreements add at most1 teacher-model cost
unit. Near-descending disagreements have a larger median/tail,while three of six
teacher-only surviving first-command interventions also have gaps below1.
Exact imitation agreement mixes weak and strong model preferences,but cost does
not reliably identify harmless commands. Do not soften/drop labels solely from
this proxy. All34,429 planning rows reproduce exactly;4.606s CPU,331 diagnostic/
401 active Qbert pins,zero emulator/native/GPU work or updates. Original Qbert
H5/R24 remains active;no new loss or successor is selected. See the report for
correlated-row,imperfect-model and CPU co-host limits.

The [completed frozen Qbert cross-cohort diagnostic](../results/2026-11-01-cdp-qbert-cross-cohort.md) keeps4013
stronger on both cohorts:15,515.625/14,370.833 versus8039's11,578.125/11,376.042.
Only original4013/4013 clears15k;all first-pyramid counts24/24 except retained
8039/8039's23/24. Own-initial subtraction preserves the ranking. Cohort alone
does not explain the agent gap,but4013's passing margin does not hold on the
alternate cohort. Original gate outcomes unchanged;no independent reliability or
isolated-weight claim. All413 pins,357 finite invariant tensors per arm,full
replays/videos,guards and original invocation pass. Four new arms131,080 actions/
524,174 frames/97.911s,zero updates/truncations;four diagonal arms reused without
GPU repeats. Main/collector succeed08:28:02/08:28:16 UTC November1/source63fce09.
No assessment stage fails or repeats;two earlier CPU checker failures/co-host
timing remain preserved. Root`/x/Temp/kindle-qbert-cross-cohort-20261101`;both
services terminal,no active allocation at closure. Do not rerun assessment,
collector,diagonal arms or preflights. No new learner,extension,cap increase or
automatic successor. Check live state;the five-game goal remains unfinished.

The [completed Breakout greedy-collection comparison](../results/2026-11-01-cdp-breakout-greedy-training.md) finishes200k/
799,639 frames/12,477 updates in21m46s. Sampled39.042/greedy99.083 fall30.64%/
5.33% below retained56.292/104.667;all0/24 two-wall. Own initials1.417/2.208 match.
Training takes1.41% longer;retain sampled collection. All272 pins,100k/6227
recovery,357 matched initial/invariant frozen tensors,paired initial semantic
traces,full training pixel/action/label and frozen replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed08:20:54/08:21:10/
08:21:42 UTC November1/sourcebd51be9. Independent2.048s terminal confirmation
and its observer succeed08:22:04. No original/GPU stage or confirmation repeats;
prior CPU preparation path failure and co-host timing remain. Actual overrides
40.8625%;higher aided online125.64 is not frozen competence. Reused1009/cohort;
no isolated coverage cause or reliability claim. Root
`/x/Temp/kindle-breakout-greedy-training-20261101/full`;all services terminal,
no allocation active at closure. Do not rerun learner,collectors or confirmations.
No extension,cap increase,confirmation learner or automatic successor. Check live
state before GPU work;the five-game goal remains unfinished.

The [completed bounded Breakout check](2026-11-01-cdp-breakout-greedy-training.md) passes8192 actions/32,740 frames/
489 updates in46.633s;1024 frozen/4076 frames/.874s haszero updates and eight
zero-return natural episodes. All252 pins,357 matching initial/invariant frozen
tensors,47 expected initial semantic differences,full replay and guards pass.
Conservative200k projection20m52s passes30min;no competence inferred. Original
service succeeds07:51:36 UTC November1/sourceccb4387,collection07:51:45. No
original stage fails or repeats. Root
`/x/Temp/kindle-breakout-greedy-training-20261101/native-check`;terminal at closure,
do not rerun. The full learner is a separate selection and uses no check experience.

The [completed delayed-takeover diagnostic](2026-11-01-cdp-breakout-delayed-takeover.md)
finds105/120 losses remain avoidable after eight recorded actions and91/120 after
sixteen,versus108 immediate. Between8 and16,the pairs are84 both/21 earlier-only/
seven later-only/eight neither. Most early sequences remain recoverable;later
control remains a useful focus,but the nonmonotonic teacher outcomes identify
no architectural cause. All478 new branches,37,016 source actions,334 diagnostic/
359 active Qbert pins,21 tests and source/branch invariants pass in55.638s CPU,
closing07:12:23 UTC/sourcec9fbf9f.240 baseline branches do not repeat;zero GPU
work or updates. Original Qbert8039 identity stays unchanged;no successor selected.

The [completed Qbert seed8039 check](2026-11-01-cdp-qbert-greedy-seed8039.md) finishes200k/799,796 frames/
12,477 updates in27m34s. Greedy11,376.042/23 first pyramids and sampled8,292.708/24
miss15k;own initials251.042/128.125 clear none. Keep4013's15,515.625 passing point,
but this same recipe passes primary greedy on one of two learners;reliability
remains unresolved. Only learner/evaluation seeds change,no sampled-training
control for8039 or isolated collection-mode effect. All359 pins,100k/6227 recovery,
357 finite initial/invariant frozen tensors,48 initial differences,full guided
pixel/action/label and frozen replays/videos,guards,curves,coverage,exposure and
comparison pass. All five original services succeed;main/post/collection finish
07:44:45/07:45:00/07:45:02 UTC November1/sourcecbee206. Independent.878s terminal
confirmation passes07:45:29;zero failures or repeated original/GPU stages.
Actual overrides32.0105%;lower late replay CE/higher aided online returns are not
frozen competence. Root`/x/Temp/kindle-qbert-greedy-seed8039-20261101/full`;
all services terminal,no active allocation at closure. Do not rerun this learner,
collectors or confirmations. No extension,cap increase,third Qbert learner or
automatic successor. Check live state before work;the five-game goal is unfinished.

The completed8039 CPU exposure recount finds later wholly unguided actions
81,530→80,031 versus4013,but unguided age1920+5525→2977;last50k31,625→26,900
and4541→2165. Released-guidance exposure increases. This localizes a difference
to long autonomous continuations without proving a cause;both learner and frozen
cohort differ. All381 diagnostic/272 active Breakout pins,original process identity,
200k actions/799,796 frames,boundaries,labels and overrides pass. Analysis.847s /
command5.579s completes08:00:47 UTC November1/sourceafa6ae6. Prior4013 counts are
reused;no native/emulator/GPU work,updates or failures. See the result's exposure
section. Root`/x/Temp/kindle-qbert-seed8039-coverage-20261101`;do not rerun. Retain
CPU co-host timing;no new learner,loss or automatic successor is selected.

The [completed Breakout first-disagreement intervention](2026-11-01-cdp-breakout-first-disagreement.md)
finds102 both/six teacher-only/six policy-only/six neither survive at the original
loss endpoint;both arms108/120 when followed by the same teacher. Thus the first
disagreement usually remains recoverable and does not explain the full teacher-
takeover benefit. Examine sustained subsequent control without claiming an
architectural cause. Separate64-action pairs86/10/11/13 use a new origin endpoint.
All37,016 actions,6,905 old teacher-suffix rows,eight reverse checks,320 repair/
317 active Qbert pins and22 tests pass. Analysis39.271s completes06:24:01 UTC
November1/sourced139b43. Original pre-replay origin-order failure is preserved;
separateCPU closure`complete_with_cpu_order_repair`,zero completed replay/GPU
repeats. Qbert identity was preserved; no new loss or learner followed this diagnostic.

The [completed greedy-training Qbert comparison](2026-11-01-cdp-qbert-greedy-training.md)
finishes 200k actions / 799,803 frames / 12,477 updates in 27m33s. Primary frozen
greedy 15,515.625 / 24 first pyramids passes the 15k/90% gate on this learner;
sampled 11,989.583 / 24 improves but misses the score gate. Own initials
80.208 / 177.083 clear none. Relative to sampled training, greedy improves
10.60% and sampled 4.05%, at .51% less observed training time. Keep CPU co-host
timing caveats; no causal speedup or coverage mechanism is established.
Only live collection changes sampled to greedy; native447fe50, teacher, losses
and 12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 remain unchanged.
All 317 pins, 100k/6227 recovery, 357 matching initial/invariant frozen tensors,
full guided pixel/action/label replay, frozen replays/videos, guards, curves,
coverage, exposure and comparison pass. All five original services succeed;
main/post/collection finish 06:56:22/06:56:26/06:56:28 UTC November 1, sourcec659423.
Separate .821s terminal confirmation passes 06:57:01. Preserve the ancillary
CPU observer-accounting preflight failure; zero original pipeline/GPU repeats.
Actual overrides 30.113%; guided online scores are not frozen competence.
Reused4013/cohort gives one passing learner, not seed reliability. Retain this
recipe for a separately declared fresh-seed check. Root
`/x/Temp/kindle-qbert-greedy-training-20261101/full`; all services terminal,
no allocation active at closure. Do not rerun completed stages or confirmations.
The five-game goal remains unfinished; check live state before new work.

Its [post-hoc exposure recount](2026-11-01-cdp-qbert-greedy-training.md#greedy-collection-reaches-more-later-autonomous-training-states)
confirms later unguided actions58,487→81,530 and unguided age1920+159→5,525;
last50k counts are20,829→31,625 and159→4,541. Released-guidance actions only rise
12,782→13,428 overall and fall4,791→3,954 late. This locates the additional deep
experience mainly in wholly unguided episodes. All actions already have labels;
these are experience counts,not sampled replay counts. Exploration,collection
and learning change together,so improved coverage is not established as the
mediator of the score gain. Actual late movement agreement63.84%→73.63% is also
not frozen-policy accuracy. The1.017s CPU analysis,332 diagnostic/359 active pins
and exact boundary/label reconciliations pass07:21:44 UTC/source234ca3a;zero
native/emulator/GPU work or updates. No new learner or loss follows.


The [greedy-training execution check](2026-11-01-cdp-qbert-greedy-training.md)
qualifies one collection-policy experiment:8192 actions/489 updates in66.241s,
with357 matching initial/invariant frozen tensors,full replay and guards.
Conservative200k projection29m39s passes35min. It changes action collection only;
no new loss or model. Task benefit and later-state exposure remain unmeasured,
and no full learner or automatic promotion is selected.

The [completed frozen Qbert training readout](2026-11-01-cdp-qbert-training-readout.md)
replays200k actions/799,739 frames in128.729s,zero updates. Last50k later movement
agreement is96.92% forced,85.56% in unguided episodes and75.10% after guidance
releases;all passive83.79% versus59.31% original changing-policy agreement.
These are different model times/contexts,not a causal sampling or coverage test.
Only159 wholly unguided training actions reach age1920+,versus12,782 released
actions;all200k actions already have labels. All562 pins,357 invariant tensors,
128 F64 comparisons,98 repeated readbacks,exact smoke/full prefix,four tests,
phase counts and guards pass. Service succeeds05:56:14 UTC November1/source5d53dac,
original collector passes;zero failures/repeats. Roots
`/x/Temp/kindle-qbert-training-coverage-20261101` and
`/x/Temp/kindle-qbert-training-readout-20261101`. No active allocation or automatic
successor;greedy training actions are a candidate requiring a separate bounded
protocol. Five-game goal unfinished;check live state before work.

The [first-disagreement intervention](2026-11-01-cdp-qbert-first-disagreement.md)
is a local causal check under the fixed feedback-aware teacher. At25 selected
origins,teacher-first survives the original loss endpoint in25 cases versus20
with policy-first;both then follow the same teacher. Four primary differences
are between movements and one is movement versus NOOP. The other20 first
choices remain recoverable under subsequent teacher control. This narrows the
problem to some costly individual choices plus subsequent control,without
identifying representation,training coverage or actor optimization as its cause.
All source/branch/suffix checks,11 tests and552 pins pass in65.045s CPU;zero
native/GPU work or updates. No new learner or loss is selected.

The [exact batched Qbert actor readout](2026-11-01-cdp-qbert-policy-readback.md)
reproduces20,584 original greedy actions and saved pixel trajectories,withzero
updates and357 invariant tensors. On1,024 fixed failure-window decisions,
teacher agreement86.33% hides52.34% movement agreement versus97.66% NOOP;on
teacher-rescued windows movement agreement is38.07%. Of122 movement disagreements,
34 chooseNOOP and88 another action. Origins already agree31/32,and all224
commands in unrescued windows agree. This identifies a later sequence/preference
gap,not its architecture or a single causal command. The prior lower-NOOP-weight
learner was worse;no weight retry or new loss follows. All525 pins and original
GPU stages pass;only separate terminal CPU collection needs a preserved
null-journal-message repair. The qualified read-only API preserves the original
batched path;captured features remain available without fitting a new head.

The [completed Breakout encoder8e-5 comparison](2026-11-01-cdp-breakout-encoder8e5.md)
adds a mixed result:sampled56.292→55.875 (−.74%) and greedy104.667→125.375
(+19.79%) at nearly unchanged21m24s training time. All0/24 two-wall;all265 pins,
recovery,initial/frozen checks,replays,guards and original services pass. Retain
4e-5 as sampled point and8e-5 as a separate greedy option;sampled remains primary,
so no overall win or automatic higher-rate successor follows. Lower late replay
CE.5581→.5252 does not identify the cause or establish reliable transfer.

The completed [Breakout encoder4e-5 comparison](2026-11-01-cdp-breakout-encoder4e5.md)
further improves sampled25.292→56.292 and greedy42.625→104.667,with training
21m28s versus22m27s. Only encoder2e-5→4e-5 changes as a learner setting;
native,exact runner and teacher stay fixed. Both modes improve substantially
per training minute,with all audits passing,but remain0/24 two-wall successes.
Retain4e-5 for this development recipe. The
[fixed-state follow-up](2026-11-01-cdp-encoder4e5-representation.md) also improves
combined-kernel ball x/y MAE7.554/6.515→6.028/4.192 source pixels and motion
readouts. Near-descending teacher agreement rises62.50%→66.91%, but teacher
probability falls.582→.573 and opposite-direction choices rise7→11 of89
directional labels. All six replacement readouts still trail the learned actor.
This strengthens the case for encoder optimization without identifying the
remaining architectural cause or establishing a universal rate. The earlier
4M/expert10 rate4e-5 failure and reused
seed/cohort limits remain. All473 diagnostic pins,357 invariant tensors per
stage and full replay/host checks pass, with zero updates or repeated GPU work.

The subsequent [own-policy replay](2026-11-01-cdp-encoder4e5-representation.md#own-policy-trajectories-improve-productivity-and-show-a-smaller-agreement-gain)
measures the4e-5 trajectory distribution: selected episodes last25.36% longer
and earn95.88% more return per action. Near-descending teacher agreement rises
57.59%→59.86%, with fewer opposite-direction and more NOOP choices on directional
labels. All120 lost lives have a detected near-descending ball within eight
actions;86 last such decisions match the teacher. This motivates examining the
preceding control sequence and recoverability, without identifying a causal
last-action error. All265 analysis/276 active Qbert pins and original process
identity pass; the23.375s CPU replay performs no native inference or learning.

The [teacher-takeover follow-up](2026-11-01-cdp-breakout-recoverability.md)
avoids108/120 of those recorded losses when control begins at the early window
origin, versus26/120 at the last near-descending state. Both are measured at
the original loss action endpoint. Paired outcomes are85 early-only,three
late-only,23 both and nine neither. This supports investigating earlier control
sequences and anticipation; it does not prove late states optimally irrecoverable
or identify representation, forecasting, actor fitting or timing as the cause.
All293 analysis/276 active Qbert pins and branch/source checks pass in36.589s
of CPU-only analysis, with no native policy inference or GPU work. No new learner is selected.

The completed [Qbert4e-5 follow-up](2026-11-01-cdp-qbert-encoder4e5.md) improves
primary greedy13,785.417→14,028.125 (+1.76%) and sampled9,987.500→11,522.917
(+15.37%),each24/24 first pyramids,in27m41s versus30m16s. Both means still miss
15,000;the primary gain is modest and varies by lane. Late replay CE worsens
.0814→.0893 and guided online returns fall,despite better frozen means. All
original audits and services pass;an ancillary terminal identity assertion is
separately corrected without repeating any original stage. Retain this cheaper
development point,not a universal rate or independent reliability claim.

The [CPU Qbert life replay](2026-11-01-cdp-qbert-life-replay.md) locates all96
losses per retained2e-5/4e-5 greedy policy after the first pyramid,withzero gains
in each fixed24-episode cohort. The newer policy lasts21.13% longer but scores
only1.76% higher,so reward/action falls15.99%. Losses at episode age1920+ rise
5→33 of96;this is not matched training-state coverage or a causal guidance-cap
effect. Complete recorded trajectories reproduce in49.272s CPU,with293 inputs
and the active Breakout process/265 inputs unchanged. Raw cube-color histories
are retained without claiming a general level detector. This narrows the next
question to later-play decisions and progress;no new loss or learner is selected.

The [Qbert teacher-takeover diagnostic](2026-11-01-cdp-qbert-recoverability.md)
avoids25/32 selected later-play losses from learner states when control begins
up to32 actions earlier;23 survive64 actions. Exact16,613-action source replay,
recorded branches,eight reverse-order checks,eight tests and309 pins pass in
63.910s CPU,withzero native/GPU work or updates. This establishes useful teacher
continuations on many failure states;the whole-sequence intervention does not
identify representation,forecasting,actor fitting or coverage as the cause.
The first episode per lane is a reused development subset,not a new game score
or reliability result. No new learner,loss or successor is selected.

The completed [Qbert encoder-rate comparison](2026-10-30-cdp-qbert-encoder-moderate.md)
also improves frozen gameplay at nearly unchanged training cost: greedy
11,984.375→13,785.417 (+15.03%) and sampled6,759.375→9,987.500 (+47.76%) in
30m16s versus30m27s. Greedy clears24/24 first pyramids and sampled23/24; both
means still miss15,000. All235 pins,initial/frozen/recovery checks,replays and
original services pass. This adds development evidence that encoder optimization
can matter to useful learning across these two recipes,not a causal mechanism
or a universal rate. Seed4013/cohort are reused and runner plumbing differs as
disclosed. The subsequent [H5/R32 combination](2026-10-30-cdp-qbert-encoder-h5.md)
does not improve its primary greedy score:14,050→13,889.583,while sampled
13,331.250→13,738.542 improves slightly. Both means miss15,000. Retain6e-6 for
the primary H5/R32 point and2e-5 for the cheaper H15/R16 point; their separate
improvements do not establish a useful combined recipe.

The completed [moderate encoder-rate comparison](2026-10-30-cdp-breakout-encoder-moderate.md)
now supplies a useful training improvement: sampled13.208→25.292 and
greedy17.875→42.625 for0.61% more training time,with no two-wall successes.
The [fixed-state follow-up](2026-10-30-cdp-encoder-representation.md) also improves
ball/motion readouts and near-paddle scalar-actor teacher agreement50%→62.5%.
This motivates further representation investigation without establishing a
causal mechanism or resolving the earlier full optimizer-protocol failure.

The completed [late-imitation-decay comparison](2026-11-01-cdp-breakout-expert-decay.md)
also narrows the objective question. Holding imitation through100k actions and
fading it tozero by150k lowers sampled25.292→17.167 and greedy42.625→25.167 at
the200k endpoint. The first100k recorded semantics match the constant control;
online guidance and optimizer history remain. Retain constant expert100. Locally
imperfect teacher actions and large imitation gradients did not imply that
removing late imitation pressure would improve frozen control. The online
last50-episode mean rises while frozen scores fall,so online returns remain
unsuitable for selecting this aided learner. Native/runner differences and the
reused development seed/cohort limit causal and reliability claims.

The [own-trajectory CPU replay](2026-10-30-cdp-encoder-representation.md#own-policy-trajectories-show-a-smaller-agreement-gain)
finds a smaller near-descending teacher-agreement change54.18%→57.59%,with
opposite-direction fractions rising and directional NOOP fractions falling.
The policies encounter different states,so this is not a paired comparison.
Substantial action error and position-readout error remain; a simple replacement
head still has no supporting result. The18.108s CPU analysis preserves all196
inputs and the active Qbert learner's235 inputs/process,with no GPU work.

The [native optimizer replay](2026-10-30-cdp-breakout-native-optimizer.md) directly
measures the small encoder movement estimated below. All clipped gradients,
moments and group movements agree, but the declared full comparison fails six
scalar F64 rounding allowances under momentum cancellation. The failed status
and all bounds remain preserved; no GPU work repeats. This supplies descriptive
native movement measurements, not an accepted full qualification or a gameplay
fix. The source corpus is unaided final-greedy evaluation with teacher labels
added offline; prior prose calling it teacher-guided is corrected.

The [CPU optimizer follow-up](2026-10-29-cdp-breakout-optimizer-attribution.md)
finds that AGC clips every encoder tensor on both saved batches. The F64 estimate
of update/weight norm is .00212–.00234% for the encoder versus .1449–.1490% for
recurrent dynamics; saved momentum substantially changes the total direction.
All191 pins, both complete parameter sets and12 CPU tests pass in4.150s, with
zero GPU work or learner updates. This makes the optimizer's transformation a
concrete target for native qualification. It does not establish a beneficial
learning-rate/clipping change or measured native parameter movement.

The [qualified fixed-checkpoint gradient diagnostic](2026-10-29-cdp-breakout-gradient-attribution.md)
shows that imitation already supplies larger raw gradients than CDP to the CNN,
posterior and recurrent dynamics on both fixed batches. Total–expert alignment
is.925–.997. This does not support simple CDP dominance on these states; AGC,
LaProp history, training-state coverage and autonomous transfer remain unresolved.
All139 pins,357 invariant tensors and production/component agreement pass, with
zero updates. The original staged-path startup failure remains preserved; the
corrected Auto diagnostic succeeds23:40:08 UTC. No new learner is selected.

The [completed FIRE-weight4 comparison](2026-10-29-cdp-breakout-fire-weight.md#full-comparison-completes-without-timeouts-but-does-not-improve-score)
finishes200k actions in22m07s, with all196 pins, full replays, frozen invariance
and terminal collection passing. Sampled12.375 / greedy17.792 do not improve
accepted control13.208 /17.875; all0/24 two-wall successes. No evaluation times
out. The longest recorded FIRE-free command intervals are123/167 actions,
compared with the prior verified12,828-action stall; command gaps also include
normal play and are not verified serving waits. The absence of that failure
in this rollout does not close the gameplay gap or establish an isolated cause.
Retain the control, with no extension, sweep or confirmation selected.

The saved training audit found labels present:23,317 FIRE targets and99.805%
eligible replay coverage in the previous centroid run. The new full run verifies
399,577 repeated FIRE replay targets with weight4 and2,988,487 total labeled
rows. Its weight-normalized expert CE is not directly comparable with the old
unweighted-label CE. The bounded preflight's accidental integrated-GPU
initialization remains preserved; no declared full-run GPU stage repeats.

The [CPU timeout replay](2026-10-29-cdp-centroid-timeout.md) identifies a concrete
Breakout serving failure:12,828 actions without FIRE after a life loss, with two
lives remaining. FIRE launches a ball at two preselected earlier origins; at a
third, late color-changing origin, eight FIRE actions have no visible effect.
All26,804 lane actions /107,212 raw frames match the retained replay, and24
recorded branch rows reproduce exactly. CPU cost19.309s, zero updates. This
supports examining serving-state coverage and actor targets before another
undirected training increase; it does not establish a training fix or explain
the late ROM state. The full learner remains protocol-failed.

The [native-checked actor comparison](2026-10-29-cdp-centroid-actor.md) now
measures the learned policy on those saved states. Overall teacher agreement
changes54.08%→55.41%, and near-paddle agreement50.00%→58.82%; simple linear/RBF
readouts still trail the actor. FIRE agreement worsens42/44→34/44, and56/136
near-paddle decisions still disagree. This is uneven imitation improvement,
not a closed-loop or causal result. It does not explain the candidate's timeout,
which is absent from the reused control corpus. All289 pins, native probability
checks and frozen tensors pass; no training or full gameplay evaluation repeats.

The subsequent [position-supervision experiment](2026-10-29-cdp-color-centroid.md#full-200k-comparison-fails-the-evaluation-protocol)
learns its direct target: mean raw loss falls.05547→.00349 across fixed50k-action
windows. Fresh200k training takes21m45s. Sampled14.375 versus13.208 and selected-
natural greedy20.125 versus17.875 are modest descriptive changes; all0/24 two-wall
successes. One final-greedy25,000-action timeout fails the declared four-arm
protocol. Complete CPU closure preserves the failure and all saved evidence,
without repeating GPU work. Retain the control. The subsequent
[frozen state readouts](2026-10-29-cdp-centroid-representation.md) show better
recurrent linear ball precision,12.878/10.918→9.267/8.064 source pixels on555
shared test rows, but much smaller response-target gains. Near-paddle linear
target error falls14.848→13.486; nonlinear target readouts do not improve there.
The new checkpoint carries more accessible position information, while actor
use and a gameplay benefit remain unmeasured. Native/runner and learning-history
confounds remain. The five-game goal is unfinished, with no retry or automatic
successor selected.

## More Breakout fitting updates lower imitation error without improving play

The [completed H5/R32 comparison](2026-10-29-cdp-breakout-h5-r32.md) doubles
world/expert updates to24,954 while reducing total scheduled imagined
transitions by one third. Its final50k-action mask-corrected expert CE falls
.696→.391,or43.80%,but sampled13.125/greedy14.333 do not improve the retained
H15/R16 control13.208/17.875. All0/24 two-wall successes;training takes33m30s
versus22m19s,50.12% more time. All audits pass,including357 matched initial
tensors and identical actual-initial semantic traces. Retain the cheaper control.

This separates improved fitting on each learner's sampled replay from improved
frozen control. It does not isolate the cause: horizon,cadence,sampling,warmup,
total work and encountered trajectories differ. Together with the pixel
precision and readout findings below,it leaves precise state use,training-state
coverage and objective effects unresolved. It supplies no evidence for an
unchanged training extension or a chosen architecture/loss replacement.

## H5/R32 reaches the Qbert gate on one of two learners

The [completed fresh6029 check](2026-10-29-cdp-qbert-h5-seed6029.md) scores
15,792.708 greedy and15,420.833 sampled,each24/24 first pyramids,in40m22s.
Its own initial checkpoints clear none;all audits pass. Seed4013 remains
14,050/13,331.250 with23/24 each,in40m24s. This establishes useful frozen
success for the fixed recipe on one learner,while retaining variation across
the two learners. It does not identify which earlier recipe change caused the
gain,replicate the H15/R16 comparison on6029,or establish broad reliability.

## Breakout still has substantial training imitation error

A CPU-only summary of completed lives checks the exact native447fe50 expert
loss and all reports from the retained12M Breakout and two Qbert H5/R32 learners.
All use expert scale100,world gradients and unit action weights. The actor loss
is mean cross-entropy over all240 replay positions,including zero-weight boundary
rows. Correcting for the number of valid labels gives:

| Completed learner | First50k-action CE | Final50k-action CE | Final1000-update CE | Final50k geometric mean teacher probability |
| --- | ---: | ---: | ---: | ---: |
| Breakout1009,12M64/H15/R16 |1.364|.696|.670|.499|
| Qbert4013,12M48/H5/R32 |.384|.0381|.0333|.963|
| Qbert6029,12M48/H5/R32 |.376|.0387|.0375|.962|

Breakout improves throughout all four fixed50k windows;it has not reached a
measured plateau. Its final1000 updates still assign the teacher action only
.512 geometric-mean probability on sampled replay states. This identifies
substantial remaining training-label error alongside the previously observed
frozen urgent-action disagreements. It does not distinguish missing precise
state information,ambiguous teacher labels,insufficient optimization,or loss
interference. The reconstruction control selected at this point tests one
representation objective change; its completed result appears below. This
diagnostic does not select another learner or intervention.

These probabilities are`exp(-cross_entropy)`,not greedy agreement,sampled
accuracy or calibrated action quality. Games,states,class mix,horizon,resolution
and update counts differ:Qbert uses24,954 updates versus Breakout12,477. Its
whole-training labels are74.6–75.1% NOOP,versus32.1% for Breakout;those online
counts do not identify the replay-minibatch class distribution. Do not interpret
the cross-game difference as an isolated architectural or optimization effect.

In Breakout's final50k actions,the coefficient-weighted expert scalar averages
69.483,while the mean absolute imagined-policy scalar is.003946. These scalar
values do **not** measure gradient magnitudes,parameter updates or causal
interference;the objectives use different state distributions and adaptive
optimization. They do not establish that increasing the coefficient is a fix.
The corresponding imagined-state entropy is.827,not replay-state entropy.

Six mask-normalization checks pass;report counts,valid-label totals and aligned
world/actor raw expert losses match the completed audits. All18 diagnostic pins
and212 active RGB inputs remain unchanged. The original active PID324938/
invocation`23da92f687b74243812a8007db61ac5a` is unchanged before/after. No agent,
game or native module is constructed;no GPU work occurs. The2.402s analysis
starts18:21:37 UTC October29/source`dd3fb58`,on one low-priority CPU core while
RGB trains. Retain this co-hosting caveat without subtracting it from training
time. No live RGB loss trajectory is used to select this analysis.

Evidence:`/x/Temp/kindle-expert-fit-20261029`,including fixed protocol,exact
447fe50 behavior/replay sources,script,declaration and full four-window/tail
results. Declaration SHA256
`59095424e582115e51c11d6aa9b27c59ddb232d57bb9cab24845903713e98009`;
review`008b4feff22b3034dcab88bfbd329e0c4edba904d56014375e4f199e04eeb89b`.

The subsequently completed [matched reconstruction control](2026-10-29-cdp-breakout-rgb-control.md) has final50k-action expert CE.704287 versus CDP.695816,with lower
frozen sampled/greedy scores10.625/16.750 versus13.208/17.875 and23.63% more
training time. Its matched initial tensors/traces and all audits pass after
a CPU-only plotting import repair. This supplies no evidence that this pixel
objective fixes Breakout's training-fit/control gap. It does not isolate the
cause or rule out other representation objectives;retain CDP.

The subsequent [frozen posterior assessment](2026-10-29-cdp-greedy-posterior.md#breakout-trained-score-rises-slightly-the-paired-assessment-fails-its-protocol)
raises trained greedy17.875→18.833 with modal states, but both remain0/24
two-wall successes. Its actual-initial modal arm has five time-limit truncations,
so the fixed four-arm protocol fails; no paired modal gain is accepted. All
exports remain finite/invariant and saved trajectories replay. This supplies
no demonstrated fix for the control gap and does not isolate sampling noise
as its cause. Retain sampled states; no training or automatic successor follows.

## RGB64 preserves precise ball positions on the reused Breakout cohort

The [fixed pixel diagnostic](2026-10-29-cdp-breakout-representation.md#rgb64-pixels-retain-much-more-precise-ball-position-than-the-linear-feature-readouts-recover)
uses the same resized geometry and teacher targets as the saved-feature readouts.
On555 shared test rows,ball x/y errors are.059/.140 source pixels,versus
CNN13.387/13.861 and recurrent12.878/10.918. The pixel rule misses two of557
reference balls and produces four extra detections;these failures remain in the
report. Its explicit color/crop priors make this a diagnostic of available pixel
precision,not a learned policy or proof of information lost in the features.
Only position is tested. Resize alone is a weak explanation for the much larger
linear readout errors;the cause of weak closed-loop control remains unresolved.

The complete20,960-action/83,799-frame CPU replay takes9.701s,with.161s for the
exact common-row comparison. All18 diagnostic and169 active H5/R32 inputs stay
unchanged,as does the original learner PID/invocation. No native module,agent,
GPU work or update occurs. Both low-priority CPU operations co-host with that
learner;no training-time correction is made. No architecture,loss or additional
learner is selected from this reused development cohort.

The [subsequent fixed nonlinear state readout](2026-10-29-cdp-breakout-representation.md#nonlinear-state-readouts-still-leave-substantial-position-error)
does not close that gap:trained combined ball x/y errors are12.053/10.211 on
the same555 pixel-supported rows. Recurrent motion R² improves to.539/.637
versus linear.473/.614,but paddle decoding worsens. All93 pins and35 CPU tests
pass;analysis takes5.368s with zero native/GPU work. This narrows the tested
decoder explanation,without proving information absent or a causal policy fix.

## Breakout action readouts do not outperform its retained policy

The [completed linear and nonlinear readouts](2026-10-29-cdp-breakout-representation.md#a-fixed-nonlinear-readout-also-trails-the-recorded-policy)
use the same saved CNN/recurrent features and held-out lanes. Trained
concatenated linear agreement is50.58% overall/49.26% near the paddle;
the fixed RBF readout scores49.25%/47.06%, versus recorded native greedy
54.58%/53.68%. Both improve over their initial-feature readouts, but neither
supports a simple actor replacement as an established fix. Other nonlinear
decoders and inadequate feature precision remain unresolved. Teacher agreement
is not closed-loop competence, and the original vector policy and scalar
feature capture use different categorical draws. No native change follows.

## Shorter imagination with more replay improves one learner

The [completed H5/R32 comparison](2026-10-29-cdp-qbert-h5-r32.md) reallocates
compute from longer imagined rollouts to twice as many world/expert updates.
At200k actions, primary greedy improves11984.375→14050 and sampled6759.375→
13331.250, with23/24 first pyramids each, for32.67% more training time. Both
still miss mean15000. All audits pass. This supports retaining a different
allocation of learning work in the current architecture. Horizon,replay ratio,
cadence,sampling and warmup change together;it does not identify which change
caused the gain or establish better quality per GPU-hour in general. The reused
seed/cohort remain development evidence;no extension or automatic successor.


## Recorded learning work dominates Breakout's training cost

A CPU summary of the completed Breakout H15/R16 control and both Qbert H5/R32
learners reconciles their wrapper stage timers with the previously audited
per-update timing aggregates. These are recorded wall times, not kernel-exclusive
GPU measurements. The denominators differ explicitly in the table:

| Completed learner | Learning / training wall time | Acting / training wall time | World training / recorded update time | Imagination / recorded update time |
| --- | ---: | ---: | ---: | ---: |
| Breakout12M64/H15/R16 |89.15%|4.69%|30.92%|40.68%|
| Qbert4013,12M48/H5/R32 |73.57%|23.05%|41.95%|26.10%|
| Qbert6029,12M48/H5/R32 |73.49%|23.12%|41.85%|26.06%|

Breakout spends1,193.879 of1,339.198 training seconds inside the learning call.
Its recorded update totals are1,180.487s, including365.036s world training and
480.262s imagination/target work. Environment stepping takes62.714s (4.68%).
This supports examining the allocation of learning work rather than expecting
emulator throughput alone to remove the measured cost. It does not identify a
particular shader, prove wasted computation, or establish that shorter horizons
preserve policy quality. The active H5/R32 experiment remains the selected test.

Qbert's acting stage is also substantial, about559–560s. In the exact recorded
runner this timer includes guidance generation, the native policy call, override
checks and teacher feedback. It therefore does **not** isolate Python teacher
cost from native action selection. Different games, resolutions, trajectories
and recipes prevent an isolated cross-game speed claim. No backend change or
new GPU allocation follows this summary.

The.736s analysis uses one low-priority CPU core while the selected Breakout
learner runs. All13 inputs and169 active-learning pins remain unchanged, with
the same PID/invocation before and after. No active trajectory is read to select
an intervention; no agent, game, learning update or GPU work occurs. Keep the
CPU co-hosting caveat without correcting training wall time. Evidence:
`/x/Temp/kindle-training-cost-summary-20261029`, review SHA256
`bdf199e7f9fb508ce7d90ac5e636026f71f8e971fd3e1607644acebc71fa0921`.

## Reviewed expert-label path has no apparent one-step offset

A source review of the exact native447fe50 runtime follows corrective labels
from the current input image through replay and both supervised losses. The
teacher proposes an outgoing action before the environment advances. Native
acting retains that label separately from the executed action; observation
stores both with the next arrival. `SequenceBatch::expert_targets` therefore
takes the label stored at t+1 for state t. It excludes outgoing terminal/reset
boundaries and the sampled sequence tail, while retaining actions that lead
into a terminal state.

Behavior targets and posterior features use the same time-major row order.
The active GPU handoff copies the initial imagination features, which are the
posterior replay states before an imagined transition, excluding the final
replay time. The world loss uses the corresponding posterior state and label,
with T/(T−1) normalization to match behavior's averaging. Executed actions
remain the dynamics inputs even when they disagree with the teacher.

No one-step offset is identified in this reviewed path. This is static source
evidence, supported by the already completed8192-action T32 adapter pixel/label
replay; it is not a new device-index capture, gradient test, proof of teacher
quality or explanation for weak control. No code or training setting changes.

Exact source snapshots, runtime/receipt hashes and source locations remain in
`/x/Temp/kindle-expert-alignment-review-20261029`. The review takes.788s on one
affined low-priority CPU core at14:40:56 UTC October29/source27cd30c, co-hosted
with the declared T32 comparison. All229 active pins and its original
PID/invocation remain unchanged before/after. No native agent, GPU work or new
updates are created by the review. Retain co-hosting as a timing caveat without
correcting training wall time. Review SHA256
`81cbc0d674fbc35bec7c5141e8d5abc7dc0bfa5ba25411b7946216811a0c4918`.

## Late latent prediction loss remains above its floor

A retrospective CPU-only check of the retained Qbert12M and Breakout12M control
logs finds no evidence that the entire latent prediction term becomes inactive
late in training. Each completed 200k-action learner has 12,477 finite reports.
The last 50k actions contain 3,125 reports per learner:

| Final 50k actions | Qbert | Breakout |
| --- | ---: | ---: |
| Mean raw posterior/prior KL | 5.476490 | 11.357183 |
| Mean KL after the floor | 5.490822 | 11.357186 |
| Mean addition from the floor | .014332 | .000003 |
| Reports whose averaged loss is at the floor | 0/3,125 | 0/3,125 |

Both use a one-nat floor. Inspection of the exact native447fe50 source confirms
that it applies separately to each state's KL before averaging over batch and
time. Logged averages therefore cannot recover the fraction of clipped states
or the gradient magnitudes. This is not evidence that every state receives a
useful dynamics gradient, nor that removing the floor would have no effect.
It does not support treating a broadly inactive late dynamics term as an
established explanation for weak control; no floor-change learner is selected.

Across the whole runs, raw/floored means are 4.811732/4.840013 for Qbert and
8.331334/8.333059 for Breakout. Raw report means below one occur 51 and 148
times, respectively, all in the first 50k actions. Report means at the floor
occur zero and four times, respectively, using a 1e-6 numerical tolerance.
Quarter summaries and exact source snapshots remain in the evidence package.

The script verifies completed-review and curve/log hashes, exact headers,
sequential report counts, finite values and equal dynamics/representation KL
metrics. All nine analysis inputs and 273 active Qbert comparison pins remain
unchanged, as does the active PID/invocation. The 2.290s check runs on one affined
low-priority CPU core at 13:17:13 UTC October29/source `30bff33`; no native agent,
GPU work or updates occur. It is co-hosted with the 25M/R16 comparison, whose
training wall time receives no correction. These are reused development logs,
not independent learners or a causal experiment.

Evidence: `/x/Temp/kindle-kl-floor-diagnosis-20261029`. Review SHA256
`5a31c432c0ef89b46a3831d5546774d3cdafa916011208f9baf1b8586105e0da`;
analysis `c7e637f7c7657a66ccbd6a6d8b63239db4e3a09c99952aae8c9beace2676062a`.

## Tested three-step loss does not improve the retained recipe

The subsequent fresh200k-action test adds direct H3/scale1 world supervision.
It completes in30m15s but lowers fixed-final greedy mean11984.375→9377.083 and
sampled6759.375→5613.542. Both modes clear23/24 first pyramids versus control's
24/24, and still miss mean15000. All numerical,checkpoint,replay,host and
comparison checks pass; [full result](2026-10-29-cdp-open-loop-supervision.md).
Retain the control. This tests one concrete loss on a reused development seed;
it does not invalidate the forecast diagnosis or establish why the loss hurts.
The subsequent [frozen forecast comparison](2026-10-29-cdp-open-loop-supervision.md#frozen-forecasts-worsen-overall-with-a-small-early-game-improvement)
finds H3 also raises15-step reward-sum MAE37.499→40.793 on identical control
trajectories. Early-game error improves2.347→2.035,but later error worsens
45.259→49.350; corrected one-step predictions also worsen overall. All152
control arrays reproduce exactly under the newer runtime. This does not show
better aggregate reward prediction paired with worse policy performance,or
identify the causal mechanism. No seed confirmation or new learner is selected.

## Actual action-block rankings weaken after the first pyramid

The subsequent frozen branch comparison completes128 origins /2432 modeled
15-step plans while re-observing48256 recorded actions/192980 frames, in95.811s.
No updates occur. A128-action/19-plan check takes3.360s; full service runs
06:51:13–06:52:55 UTC on October29 (1m42.029s). All46 pins,357 invariant tensors,
128 exact recorded-plan matches with the earlier forecasts, eight CPU/native
pixel-hash trajectories and host guards pass. Native447fe50 remains unchanged.

At each origin, each of18 candidate controls holds one action for four policy
steps, then follows the recorded continuation through step15. A19th recorded
plan provides replay/prediction parity. The128 fixed origins contain four
positions per phase per test episode:64 before/64 after the first pyramid,
across the same16 reused episodes. They are balanced positions rather than a
frequency-weighted sample of all visited states. Both emulator futures and
model latent draws are paired across candidates, with one realization per plan.

| Actual branch comparison | Before first pyramid | After first pyramid | All |
| --- | ---: | ---: | ---: |
| Origins with unequal candidate outcomes | 19/64 | 21/64 | 40/128 |
| Pairwise ranking agreement on unequal outcomes | 72.96% | 57.21% | 64.50% |
| Mean actual discounted return of model-selected block | 120.687 | 339.478 | 230.082 |
| Uniform-choice actual return | 102.429 | 337.877 | 220.153 |
| Best candidate actual return | 144.159 | 344.077 | 244.118 |
| Model gain over uniform | 18.258 | 1.601 | 9.929 |
| Lane-bootstrap95% interval for that gain | [.249,37.313] | [−.131,3.443] | [.996,19.692] |

Ties receive fair average-outcome and half-pairwise credit. The88 origins with
identical actual outcomes do not count as successful discrimination. Overall
model-selected return230.082 is below the recorded-plan return233.863 and close
to the scalar actor's preferred held block231.016. These are short open-loop
plan scores, not the full adaptive policy's return; there is no evidence here
that deploying this planner would improve the retained greedy score.

This supplies actual counterfactual outcomes for the declared control plans,
unlike the earlier unrelated-action sensitivity control. It supports useful
action-conditioned learning, with weaker evidence of useful ranking later in
play. Later-phase gain over uniform is not resolved above zero, and many
origins are insensitive to the candidate controls. Fixed continuations, short
horizon, uncommon held actions, a single model/emulator draw and reused lanes
limit causal conclusions. This does not establish that an additional world loss
will improve policy learning.

CPU oracle generation takes26.878s for the full recorded source and154607 extra
branch emulator frames, including eight reverse-order checks. All128 recorded
plans reproduce their actual outcomes; the full source trajectory is unchanged.
The first ALE snapshot prototype failed source replay at stream0/index3879 after
15 origins. Its serialized bytes matched, but ALE0.12.1 keeps previous sticky
actions outside `ALEState`. Isolated CPU process copies preserve that memory;
candidate-order reversal and full replay pass. Original failure/source/log are
retained; no GPU stage repeats and the pinned emulator runtime is unchanged.
See the [protocol and upstream source references](../experiments/2026-10-29-cdp-action-ranking.md).

The helper's nine new CPU tests cover ranking ties, ineffective controls,
invalid scores and reward-independent/reset-safe origin selection; together
with the24 existing forecast tests, all33 pass. Sourcef3da9ff, evidence root
`/x/Temp/kindle-action-ranking-20261029`. Declaration SHA256
`8e08ad1bdc3cd72b1fa5e64de5514496f9b97162cd62826c9a52bcb1f371865f`;
comparison `b1e1115715dabb9467473d68756feee224b8adfc907d0235d69aa0c487dddfaf`;
CPU oracle `b44a1cd0523e7c2da1aab8f6dea1cc1784daf1b367b934b5bed477a29372a274`;
final review `971b7b29bafae8b92a929e148cc1dec9819d13b89bbcebc41f23a9a2cadf68e9`.
The service is terminal success; no GPU allocation is active.

These three diagnostics motivate a separately selected bounded implementation
of short open-loop world supervision. That next experiment must establish
correct gradients and acceptable cost before selecting a full learner; the
current frozen evidence is not a measured training improvement.

## Live critic learns useful values but underestimates realized returns

The subsequent four-arm live-state diagnostic completes98664 actions /394520
frames in69.704s of evaluation, with zero updates. Main service
`kindle-critic-calibration-20261029` runs06:28:28–06:30:01 UTC on October29
(1m33.689s). A preceding1024-action/4096-frame check takes.882s. All44 pins,
357 invariant tensors per arm,12522 matching full semantic events, counters and
guards pass. The instrumented actions, rewards, episode/reset boundaries and
frame counters exactly reproduce all four retained evaluations. Original
verified replays supply task phases; no new pixel recordings are claimed.

The sampled final policy is primary, because the actor samples during
imagination. Returns include actual rewards and frozen intrinsic bonuses,
discounted with the model's F32 gamma, with no critic bootstrap. First episodes
per lane calibrate a constant baseline; subsequent16 natural episodes are tested.

| Live critic measurement | Sampled final | Greedy final |
| --- | ---: | ---: |
| Test state-action observations | 19574 | 27102 |
| Mean realized discounted return | 1475.689 | 1936.698 |
| Mean predicted value | 1218.534 | 1545.437 |
| Bias | −257.154 | −391.262 |
| Critic MAE / constant-baseline MAE | 790.664 /1038.765 | 955.480 /1148.890 |
| R² / correlation | .184 /.542 | .287 /.613 |
| Mean intrinsic discounted return | .710 | .805 |

Sampled before/after-first-pyramid correlations are.618/.358, with biases
−649.367/−102.482. Within those phases R² is−.388/−.104 despite overall positive
R²: some aggregate explanatory power comes from distinguishing phases. The
critic still beats the declared global calibration constant in phasewise MAE.
Greedy before-pyramid correlation is.941 with bias−1135.742; greedy behavior
changes the return being estimated, so this is not evidence that the sampled
policy's critic should predict the greedy return exactly.

Actual-initial critics predict zero; their sampled/greedy mean target returns
are70.477/34.404, including intrinsic returns9.092/8.538. These actors visit
different states from the trained actor. The controls therefore do not establish
a matched-state improvement in critic accuracy. All four unchanged task scores
are6759.375/177.083 sampled final/initial and11984.375/80.208 greedy final/initial.

This is evidence of useful but imperfect value learning. Underestimation alone
does not identify an incorrect action ranking or prove why policy improvement
falls short: shared value bias can leave action preferences unchanged, realized
returns are stochastic samples, and the critic uses symlog-distribution decoding.
The small trained intrinsic contribution does not explain the observed value
gap. Proceed toward actual alternative-action outcomes before selecting a
critic/loss intervention. No new training or changed loss is selected here.

Native447fe50 and the pinned runner remain unchanged; all9 value-analysis CPU
tests pass. [Protocol](../experiments/2026-10-29-cdp-critic-calibration.md), source
1227d1e, evidence `/x/Temp/kindle-critic-calibration-20261029`. The controller
validates each completed arm before starting the next. No GPU stage repeats,
failures or host recovery occur; no GPU allocation is active at completion.
Declaration SHA256
`9ea9be0f32dec18cf1737ba5b29c07518aacbaa6a47224dca076b33137087cbc`;
comparison `95c90655aaefad74be9f3066bf4acc92ff5452bd5ad2ae28d9f41a58d7ff17eb`;
bounded review `0db34d65e9c8e72559aed69a85e9b4394d59a684f5f571c8f7d68fc7ca719f17`.

## Forecast interpretation

The first investigation within the existing five-game goal finds useful
action-conditioned15-step reward forecasts in the strongest retained Qbert
learner. This does not support treating a wholesale collapse of imagination as
the established cause of its11984.375 greedy score. Errors are substantially
larger after the first pyramid, and prediction quality on recorded trajectories
does not establish correct ranking of alternative actions.

## Frozen forecast result

Both the retained12M48/B16/R16/expert100 final model and its actual initial model
re-observe the same48256 recorded actions /192980 emulator frames. Captures take
207.599s and207.409s respectively, with zero updates. The128-action check takes
3.700s. The complete service starts06:11:50 UTC and succeeds06:18:53 on October29,
using native447fe50, Auto precision and the RX7900XT. No training is resumed.

The first natural episode in each of eight lanes supplies calibration baselines;
the next two supply16 evaluation episodes. One-step analysis covers27102 actions.
The common-horizon comparison uses1681 origins with valid15-action futures within
one episode. These are reused development trajectories from the retained greedy
evaluation, not independent learner confirmation.

| Measurement | Actual initial | Trained, recorded actions | Trained, unrelated actions |
| --- | ---: | ---: | ---: |
| One-step positive-reward AUC, all test actions | .500 | .978 | .972 |
| 15-step endpoint positive-reward AUC | .500 | .938 | .673 |
| 15-step cumulative reward MAE | 107.570 | 37.499 | 69.538 |

Repeating the previous reward15 times gives cumulative MAE100.119; the
calibration-mean baseline gives153.956. On common valid origins, cumulative MAE
at horizons1/3/5/10/15 is2.569/7.096/11.583/24.234/37.499. Endpoint AUC is
.976/.966/.954/.945/.938. Absolute cumulative error naturally grows with horizon;
these measurements retain useful information through15 actions.

The phase split identifies a more specific weakness:

| Origin phase | One-step reward MAE | 15-step cumulative MAE | Mean actual15-step reward sum |
| --- | ---: | ---: | ---: |
| Before first pyramid | .176 | 2.347 | 50.822 |
| After first pyramid | 3.123 | 45.259 | 120.098 |

The15-step samples number304 before and1377 after. The reward distribution also
changes, so this is not an isolated effect of reaching another level. Correcting
the state with the actual target observation lowers15-step endpoint reward MAE
from.345 to.254 before, and4.579 to2.603 after. Both head/state accuracy and
free-running prediction warrant further inspection in later play.

An exploratory magnitude grouping finds1584 test transitions with reward100:
one-step prior and corrected-posterior predictions average85.855 and86.134.
For1289 reward25 transitions they average21.089 and21.235. This localizes some
reward underestimation even after observing the frame; it does not identify the
mechanism or prove that this bias changes the preferred action.

Terminal evidence is sparse:16 one-step terminations and only2 at the sampled
15-step endpoints. One-step Brier score.000584 is only slightly below the
always-continuing baseline.000590. AUC alone does not establish calibrated death
risk, and the full-game terminal target does not label every lost life.

## What is already in the loss

Source inspection confirms that CDP predicts the detached current CNN encoding
from the recurrent state before that observation is consumed. Successive
training states then receive real observations. Reward and continuation heads
train on these corrected posterior states. The imagined actor instead uses
prior rollouts without those corrections. The retained recipe has no direct
multi-step free-running supervised loss against recorded future targets. The
optional three-step loss tested above is implemented but remains disabled after
its negative gameplay and forecast comparisons.

The critic already combines imagined and real-replay losses; replay scale is.3.
The latter bootstraps with imagined returns from replay states, including in
the active GPU kernel `targets_returns.wgsl::replayed`. Consequently it is not
an independent Monte Carlo validation of model-generated targets. Intrinsic
disagreement rewards also enter these targets. No missing-loss bug or causal
policy failure is established by this audit.

## Measured update cost

Summing the12477 saved learner timing records for each latest12M learner gives:

| Recorded update phase | Qbert | Breakout |
| --- | ---: | ---: |
| Imagination and target generation | 46.62% | 40.68% |
| World training | 27.34% | 30.92% |
| Posterior-state preparation | 16.78% | 16.53% |
| Behavior training | 5.80% | 5.62% |
| Other recorded update work | 3.46% | 6.24% |

Recorded update totals are1175.345s and1180.487s, within original training walls
1826.985s and1339.198s. This is phase accounting, not GPU utilization or a
breakdown of the remaining acting, adapter, checkpoint and logging time.
Imagination is a major cost, making its contribution to policy improvement an
important learning-efficiency question. Shortening it is not automatically a
win; the previous Pong H5 comparison lost score.

## Initial interpretation and subsequent diagnostic direction

The following interpretation preceded the completed critic-calibration,
branched-action and three-step-loss tests recorded above. Those follow-ups now
provide the counterfactual evidence and negative loss comparison; this is the
historical rationale for them,not a request to repeat completed GPU stages.

Prioritize live-belief critic calibration and actual branched action outcomes,
especially after the first pyramid. Good forecasts along recorded paths do not
prove that selecting a different action improves actual returns. The unrelated
actions above are a sensitivity control evaluated against the original outcomes;
their true counterfactual outcomes were not collected.

A multi-step world loss remains a reasonable candidate, but this result does not
establish that adding it is the best next training allocation. A paired real/model
update experiment must control behavior distribution and intrinsic targets; naive
off-policy actor updates would confound the comparison. No new loss, extension,
fresh learner or automatic successor allocation is selected by this result.

## Validation and retained evidence

All39 declared input pins remain unchanged. Both full captures preserve all357
tensors byte-for-byte (280 world,66 behavior,11 slow-value), and all predictions
are finite. Eight per-stream prefix checks, counters, recorded rewards/boundaries,
frame counts and host guards pass. Final/initial input trajectories and pixel
hashes match across all eight streams; stream0 also matches the retained replay
video hash. Minimum sampled device-budget headroom is14.59GiB. No GPU stage is
repeated and no host recovery occurs. The forecast helper's historical346 count
assumption is removed while full tensor-key/shape/dtype/byte checks remain; all24
forecast CPU tests pass. Native code is unchanged.

Scalar re-observation uses different categorical draws from the original vector
actor. The diagnostic does not reproduce its live beliefs, measure new frozen
competence or improve the five-game result. The optional greedy-posterior draft
is unused and remains separate; its130 Rust/612 Python CPU tests pass, but its
new native implementation has not been built or GPU-tested.

Protocol: [selected diagnosis](../experiments/2026-10-29-cdp-learning-diagnosis.md).
Source8295d15. Evidence root `/x/Temp/kindle-learning-diagnosis-20261029` contains
the immutable queues/controller, all captures, guard reviews, `comparison.json`,
`training-timing-review.json`, exploratory `reward-magnitude-review.json`, and
the inspected `forecast-comparison.png`/SVG. Main service
`kindle-learning-diagnosis-20261029` is terminal success; no GPU allocation is active.

SHA256:

- declaration: `a1393fbc26caf730131291dbc09d9ed9ef8812772b5f135c38eda98f11dc87c9`
- comparison: `8eac4db249557527b40154d24a1890f233f475e98d8f9197b4eb5bd49e792681`
- timing review: `153b2baed2b0b4a5d73bd601fe09105e440bc30a9a754477fd031067e84fbaf9`
- bounded review: `276b7282c3efb9f9de026b6471e728cfc1f7d88dd8bc0174d4b11af05ce73926`
