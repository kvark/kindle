# Kindle: CDP learning roadmap

## Selected direction: Breakout world model in isolation

The goal **prove the world model training on Breakout** remains active and unproven.
Actor/critic, CDP comparison and the five-game campaign stay parked. Use native
pixel supervision and H0 as the broad reference; no pretrained encoder or
object/RAM training labels.

The [clean-launch fitting diagnostic](experiments/2026-11-08-breakout-world-launch-fitting.md) is selected. The corrective model's new fitting branches
were not included in its original16-stream fitting exports. Query only its final
20,480 state onall288 retained launch branches:36 eight-lane groups,four fixed
latent draws,history16/endpoints1/4/8/16/32. No training,gradient,new gameplay,
other model query or old assessment repeat. Check source joins,finite arrays,
all240 frozen tensors/ages andthe existing host guard. Root
`/mnt/data/kindle-breakout-world-launch-fitting-20261108`. No learner is selected.
The question is fitting versus different-context failure,not a new world gate;
all draws/missing objects remain andfitting success cannot repair the failed trial.

The [corrective trial](results/2026-11-08-breakout-world-launch-corrective.md) stays closed and failed:ball energy.172366→.229385;
natural eight-step53/107→46/107;234/304 false alarms. Retain earlier small64.
Native455.523s/loop527.794s;new final84 groups170.763s,closure1,785 pins/three
guards/27 files/two terminal services. Report-only generator correction preserved,
sealed original restored;no measured files changed. The [new collection](results/2026-11-08-breakout-world-clean-launch.md),
[conditional audit](results/2026-11-08-breakout-world-transition-support.md),[survival sensitivity](results/2026-11-08-breakout-world-small64-survival.md) and[small64 trial](results/2026-11-08-breakout-world-small64.md) remain closed. Do not repeat them.

All original physics/event gates stand. Precise physics, event timing/calibration,
persistent brick memory, late/tunnel/second-wall coverage, untouched families and
fresh confirmation remain required. Existing oracle has no above-top-brick
endpoint. Intermediate scores cannot complete the world-model or five-game goal.
No unchanged extension, sweep, completed-stage repeat or automatic successor.

### What would establish the world model

Finite training losses and backend qualification establish that an experiment
runs; they do not establish game dynamics. Keep the complete goal distinct from
the completed H32 comparison:

| Capability | Retained evidence | What is still missing |
|---|---|---|
| Read observed images | [H0 coverage](results/2026-11-07-breakout-world-coverage.md):108/112 balls within one RGB64 pixel | Consistent precision across the full game,including late and occluded states |
| Forecast under submitted actions | [Local-action world](results/2026-11-08-breakout-world-local-action.md):43/107 eight-step and11/95 thirty-two-step balls match;ordinary primary.240566 trails H0.169502 | Accurate free flight and collision trajectories,including fitting data; existing precision/distribution gates |
| Predict rewards and endings | [Local-action world](results/2026-11-08-breakout-world-local-action.md):RIGHT16 false ending risk.507002 and244/304 nonending alarms;[survival diagnostic](results/2026-11-08-breakout-world-survival-decomposition.md) changes scores with oracle labels only | Correct event timing and natural-frequency calibration; current outcome-selected slices alone cannot establish calibration |
| Remember changed world state | Existing reports retain brick-cell scores and regressions | Persistent removed-brick memory through absence/occlusion; low image loss or a static wall reconstruction is insufficient |
| Cover the game and generalize | Current fixed-action challenge has33 distinct contexts across seven reused development groups; repetitions share histories | Late/tunnel/second-wall states,above-top-brick endpoints,untouched families and fresh confirmation |

Pixel supervision makes these errors observable. CDP remains a later comparison,
not an assumed solution to the forecast gap. A better forecast score would be
intermediate evidence only; it cannot discharge the untested rows above. Preserve
all per-action/horizon regressions and all protocol failures when deciding the
next separately justified experiment.

## Retained campaign evidence

The [prior-reward comparison](results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison) is complete and fails its fixed retention rule.
Fresh22003 scores79.167 greedy/47.000 sampled versus matched75.250/44.708,
in21m33.955s versus21m30.822s. Greedy own-initial gain/min rises5.113%
(3.391251→3.564653), but the paired95% difference[−1.322212,1.717042] spans zero.
Sampled gain/min rises5.095% with its interval also crossing zero. All0/24 two-wall.
Keep prior reward off and posterior reward1. Small point gains on one reused
seed/cohort establish neither a useful improvement nor a reward-model mechanism.

All200k actions/799,622 frames/12,477 updates,100k recovery,357 matched initial/
invariant frozen tensors,47 initial events, full replays,854 allocation/338 original
retained pins and original services pass. Independent closure recounts192 frozen
episodes, paired intervals, all learner reports,53 completion artifacts and28
checkpoint files in5.507s, complete23:13:49.902 UTC November6, source`dda5bf6f`.
The bounded check remains closed and passed. Earlier qualification failures,
prelaunch recovery-unit correction and the later publication reader's non-text
journal-entry failure remain preserved with separate CPU repairs; no GPU stage
repeats. Root `/mnt/data/kindle-breakout-prior-reward-20261106/full` is closed,
with no active allocation or selected successor. Original864/90%-two-wall and
full five-game goals remain unfinished.

The [completed reward-supervision comparison](results/2026-11-06-cdp-breakout-reward10.md#full200k-comparison) fails retention. Only
reward loss1→10 changes against retained22003:greedy52.167/sample42.625 versus
75.250/44.708,21m20.465s versus21m30.822s. Primary own-initial gain/min falls
31.086%, paired95% difference[−1.936771,−.046998]; sampled falls4.094% with its
interval crossing zero. All0/24 two-wall. Retain reward1. One reused seed/cohort
and changed subsequent experience do not identify a causal mechanism.

All200k actions/799,583 frames/12,477 updates,100k recovery, initial/frozen tensors,
full replays,263 allocation/338 original retained pins and original services pass.
Independent closure recounts192 episodes, both paired intervals,55 completion
artifacts and28 checkpoint files; complete22:03:43.920 UTC November6, source3be21d90.
Original CPU preparation/import failures remain preserved with separate repairs;
no GPU stage repeats. Root `/mnt/data/kindle-breakout-reward10-20261106/full` is
closed. No extension, sweep, second learner or automatic successor is selected.
Original864/90% two-wall and full five-game goals remain unfinished.

The [pre-contact shot predictor](results/2026-11-06-cdp-breakout-causal-shot.md) is complete and fails its fixed
screen. Adding planned contact offset lowers holdout slope MAE.691547→.601837
(12.972%) versus incoming direction alone, below the required20%; direction
accuracy76.761% misses80%. All eight lanes improve MAE, but this does not qualify
a steering controller. Inputs precede the incoming-fit window; reconstructed
actual contact offset and future geometry are excluded as features. Keep the
original teacher. This is observational prediction, not a causal action test.

All776 contacts qualify:492 fitting/284 reused holdout rows. Six CPU fixture
groups,64 pins,47 historical sealed files,18 completion artifacts and an
independent source-join/augmented-least-squares recount of54 metric groups pass.
The controller completes21:17:03.571 UTC November6 in1.058103s, source569d34bf;
separate terminal closure passes21:18:12.524. No failed or repeated stage.

Root `/mnt/data/kindle-breakout-causal-shot-20261106` is closed. No collector,
emulator/native construction, new game action, learner update or GPU work.
No controller, fit retry, integration or automatic successor is selected.
Original864/90% two-wall and the full five-game goal remain unfinished.

The [fresh-seed Breakout confirmation](results/2026-11-06-cdp-breakout-seed22003.md) is complete. Seed22003 learns
75.250 greedy/44.708 sampled in21m30.822s, versus its2.292/1.875 actual-initial
controls. Retained1009 on the same new cohort scores94.625/70.500; its historical
old-cohort point remains143.667/70.750. Fresh own-initial gains are positive on
all eight lanes in both modes. Greedy gain/min is21.091% lower than the matched
control, with its paired95% interval narrowly including zero; sampled is38.323%
lower. Both training-seed and evaluation-cohort variation matter. All0/24 two-wall;
the original864/90% gate and five-game goal remain unfinished.

All90 active/338 retained pins,69 completion artifacts,52 checkpoint files and
288 recounted episodes pass. Training executes200k actions/799,616 frames and
12,477 updates; recovery, all eight frozen arms, replays and original services
pass. Independent closure completes21:04:53.361 UTC November6. Native447fe50 /
SHA35ad0db, runnercf4ea238 and teacher are unchanged; execution sourcedcc4dd5d.
The initial archived-helper copy failure remains preserved with its separate
preparation repair. No GPU stage or learner update repeats.

Root `/x/Temp/kindle-breakout-seed22003-20261106/full` is closed; main/recovery/
terminal services are inactive, with no render holder or learner. Keep the
retained recipe with the measured variation disclosed. No new seed is promoted;
no extension, second learner, seed search or automatic successor is selected.
Consult the [fixed protocol](experiments/2026-11-06-cdp-breakout-seed22003.md) before any separately justified work.

The [completed replay-value comparison](results/2026-11-06-cdp-completed-value.md#full200k-comparison) stays closed and off:
89.958 greedy/50.500 sampled versus primary143.667/70.750, with21.906% lower
primary gain/min. Do not repeat completed stages. No extension, second learner,
seed search or automatic successor is selected. The five-game goal is unfinished.

The [sparse-wall brick-aiming teacher comparison](results/2026-11-06-cdp-breakout-brick-aim.md) is complete and
fails retention: 208.458 versus 225.083 (−7.386%), only 1/8 improved lanes and
0/24 two-wall for both. Keep the original teacher. The fixed RGB-cell rule is
eligible on 445 selected decisions and changes 191 of 39,749 first commands.
This does not calibrate outgoing angles or isolate the cause of the regression.

Six CPU fixture groups, 48 declared pins and 31 completion artifacts pass.
One candidate completes 43,176 actions / 172,667 frames in 33.187557s; full
38.043074s replay matches every decision/pixel/reward/frame/reset/outcome.
Controller 71.688512s, closed 19:16:42.795 UTC November 6, source e52a6844;
independent recount passes at 19:16:50.885. No failed or repeated stage.

Root `/x/Temp/kindle-breakout-brick-aim-20261106` is closed. No control rerun,
native learner or GPU work. No fresh cohort, integration, sweep or automatic
successor is selected. Retain CDP scale 500 and Breakout 143.667 greedy / 70.750
sampled; original 864/90% two-wall and the five-game goal remain unfinished.

The [completed CDP prediction-scale1000 comparison](results/2026-11-06-cdp-breakout-cdp1000.md#full200k-comparison) regresses to 77.167 greedy / 42.750 sampled in 21m30.371s, with 47.111% lower primary gain/min. Keep scale 500. All learning, frozen evaluation, recovery and independent closure checks pass; original CPU import and plot-fixture failures remain preserved with separate repairs. Root `/x/Temp/kindle-breakout-cdp1000-20261106/full` is closed. Do not repeat or extend its stages.

The [optional stride-two RGB encoder](results/2026-11-06-cdp-rgb-stride2.md) comparison is complete and negative.
Fresh200k scores16.458 greedy/13.000 sampled versus143.667/70.750 retained.
Training takes19m22.697s versus21m27.910s (9.722% less), but primary own-initial
gain/min falls88.580%; all eight lanes regress in both modes. Keep stride-two
off and retain the original convolution/max-pool encoder. Sampling/nonlinearity
change together; the result does not isolate a perceptual cause.

All12,477 updates,357 initial/frozen tensors, full action/label replays,100k
recovery,372 active/338 retained pins and original services pass. Independent
CPU closure recounts192 candidate/control episodes,55 completion artifacts and
28 checkpoint files. Original CPU fixture compile failure remains preserved;
no GPU stage repeats. Root `/x/Temp/kindle-rgb-stride2-20261106/full`, source
`8d4ab88d`, native `cde123e1`; closure17:57:14.858 UTC November6.
Main, recovery and terminal services are complete; no render holder or learner
is active. No successor is selected. Do not repeat completed stages. Keep
retained143.667 greedy/70.750 sampled; original864/90% two-wall and five-game
goal remain unfinished.

The [contact-horizon risk teacher comparison](results/2026-11-06-cdp-breakout-contact-risk.md) is complete and fails
retention:206.250 versus225.083 (−8.367%), only3/8 improved lanes and0/24 two-wall
for both. Keep the original teacher. The penalty changes159 first commands in
40,195 selected decisions, including110 at horizons above one action. The region
gate leaves no eligible3–5-action cases; this is not a calibrated uncertainty
model or an empirical five-action test.

One CPU candidate completes46,200 actions/184,763 frames in32.161830s; complete
35.692589s replay matches every decision/pixel/reward/frame/reset/outcome. Six
new fixture groups,45 declared pins and23 completion artifacts pass. Controller
68.250249s, closed16:53:43.976 UTC November6, source2ce19be9; independent recount
passes16:55:32.367. Root `/x/Temp/kindle-breakout-contact-risk-20261106` is closed.
No control rerun, failure, native learner or GPU work. No fresh cohort, retry,
integration or automatic successor is selected. Retain original guidance and
143.667 greedy/70.750 sampled; original864/90% and five-game gates remain open.

The [offline shot-outcome diagnostic](results/2026-11-06-cdp-breakout-shot-outcomes.md) is complete and fails its fixed
outer-region design screen. Post-catch reward/frame ratios are 1.173490 below
score216 and 1.271007 at216–431; only3/8 lanes favor projected outer entry.
All minimum counts pass. This does not justify a general edge-aiming rule or
reject every policy using remaining brick geometry.

All776 contact windows,18,749 distinct actions,33 output groups,30 historical
sealed artifacts and46 current pins pass. CPU controller.769346s, complete
16:39:37.438 UTC November6, source687788a0. Root
`/x/Temp/kindle-breakout-shot-outcomes-20261106` is closed; zero new actions,
emulator/native imports, updates or GPU work. Earlier three-pixel aiming and
initial-episode guidance failures remain retained. No successor is selected.
Keep original teacher/guidance and143.667 greedy/70.750 sampled; original864/90%
two-wall and the five-game goal remain unfinished.

The [initial-episode guidance comparison](results/2026-11-06-cdp-breakout-initial-guidance.md#full-200k-comparison) is complete
and fails retention: greedy 114.000 / sampled 49.125 versus 143.667 / 70.750.
Fresh 200k actions / 799,655 frames / 12,477 updates take 21m28.245s, just 0.026%
more training time. Primary own-initial gain/min falls 20.993%, 6.590132→5.206697.
Its paired 95% difference interval [-4.096855, 1.194018] spans zero; this reused
development seed/cohort does not establish a reliable population regression.
Both final modes remain 0/24 two-wall.

All 357 initial tensors match retained/own bounded, 47 own pre-update events and
both retained frozen-initial traces match. Finite/weighted reports, 100k recovery,
four 24-natural-episode frozen arms with 357 invariant tensors, full replays/videos,
curves and 374 current / 338 retained pins pass. All eight lanes leave the initial
phase at 22,232–30,344 aggregate actions. Initial-phase actions total 26,884;
overrides total 103,444/200,000 (51.722%). Timing, amount and state distribution
change together. Maximum training score is 422; no episode/tail reaches 432.

Original main/collector finish 16:20:30.454 / 16:20:37.393 UTC November 6.
Closure at 16:21:18.893 verifies terminal identities and no Kindle service/render
holder. Root /x/Temp/kindle-breakout-initial-guidance-20261106/full is closed,
execution source f797ad78, native 7704d25c. Two incomplete CLI mocks and prior
native fixture failures remain preserved; no full-stage failure or repeat.
Retain the original 20% schedule and 143.667 greedy / 70.750 sampled. No GPU
allocation, extension or automatic successor is selected. Original 864/90%
two-wall and the five-game goal remain unfinished.

The [RGB96 full comparison](results/2026-11-06-cdp-rgb96.md#full-200k-comparison) is complete and
fails retention:74.708 greedy/60.417 sampled versus143.667/70.750. Fresh200k
actions/799,574 frames/12,477 updates take30m52.756s,43.857% more training time.
Primary own-initial gain/min falls63.984%,6.590132→2.373492; its paired95%
difference interval[-5.582483,-2.893938] is negative. All final modes are0/24
two-wall. Geometry, capacity and initialization change together; no isolated
perceptual cause or broad reliability follows.

All357 own-bounded initial tensors/47 pre-update events, weighted/finite reports,
100k recovery, four24-natural-episode frozen arms with357 invariant tensors,
full replays/videos, curves,367 current/338 retained pins and56 completion
artifacts pass. Own-initial sampled1.333/greedy1.417 differ from retained controls.
Original main/collector complete15:19:49.164/15:20:13.484 UTC November6;
publication closure15:20:59.477 confirms all original services/children terminal,
no Kindle service/render holder and no full-stage failure/repeat.

Root /x/Temp/kindle-rgb96-20261106/full is closed, execution sourcecf007696,
native7704d25c. Original Python/two native fixture failures remain preserved with
separate repairs. Keep useful checkpoints/package artifacts; build cache was
already released. Retain RGB64,32×16,143.667 greedy/70.750 sampled and optional
actor paths off. No GPU allocation, extension or automatic successor is selected.
Original864/90%-two-wall and the five-game goal remain unfinished.

The [full enumerated one-step actor comparison](results/2026-11-06-cdp-actor-expected-value.md#full-200k-comparison)
is complete and fails retention. Fresh200k actions/12,477 updates take27m58.682s
versus retained21m27.910s, **30.341% more**. Greedy138.417 and sampled58.083 trail
143.667/70.750; all0/24 two-wall. Primary own-initial gain/min falls **26.126%**,
6.590132→4.868404. Its paired95% difference interval[-4.253377,.800655] spans zero;
this reused development cohort does not establish training-seed reliability.

All357 initial tensors and47 pre-update events match bounded/retained; actual-
initial sampled1.416667/greedy2.208333 and799/794 semantic events match retained.
All four frozen arms select24 natural episodes with zero updates/truncations and
357 invariant tensors. Weighted losses, required expected-value metrics/snapshot
timing,100k recovery, full replays/videos, curves, exposure,365 current/338 retained
pins and original terminal services pass. Main/collector complete13:46:32.617 /
13:46:51.144 UTC November6; publication reader and idle closure pass. Original
CPU compile failure remains preserved; no full GPU/audit stage fails or repeats.

Root `/x/Temp/kindle-actor-expected-value-20261106/full` is closed. No GPU
allocation, extension, sweep or successor is selected. Keep the option off and
retain ordinary inference,32×16 and143.667 greedy/70.750 sampled. Enumerating
one-step actions changes horizon/start distribution with the estimator; this
result does not isolate gradient bias or reject every differentiable-control
method. Original864/90%-two-wall and the five-game goal remain unfinished.

The [actor posterior averaging comparison](results/2026-11-06-cdp-actor-posterior-average.md#fixed-trained-policy-comparison-complete)
is complete and fails the declared retention criterion. Eight-sample greedy
return175.500 exceeds ordinary143.667 by22.158%, but the paired eight-lane95%
interval is[-15.417,83.792], only4/8 lanes improve and both have0/24 two-wall.
Keep ordinary inference; averaging stays frozen-only and default off. This
fixed-checkpoint serving result does not establish a training-efficiency gain,
a posterior-noise cause or training-seed reliability.

The fixed24-natural-episode cohort is selected from25 completed episodes, with
zero updates/truncations.45,128 actions/180,472 frames take29.068s, construction
2.936s. All357 tensors remain finite/invariant; full replay/video,681 pins and
the guard pass. Comparison completes12:33:57.992 UTC November6,
publication recount12:35:07.674; original child1479392 is terminal.
No Kindle service/render holder remains. The separate bounded candidate passed
at1.018472× observed ordinary per-action cost. Original pipeline/CPU-repair
failures remain failed; no completed stage was repeated.

Root `/x/Temp/kindle-actor-posterior-average-20261106/candidate-evaluation` is
closed; no GPU allocation, learner, count/seed sweep or successor is selected.
Retain32×16 and143.667 greedy/70.750 sampled; original864/90%-two-wall and
five-game gates remain unfinished. Consult a separately justified new protocol
before further GPU work; do not restart this comparison or its failed queue.

The [64×8 latent comparison](results/2026-11-06-cdp-latent-shape.md#completed-full-200k-comparison)
is complete and fails the primary retention criterion. Fresh 200k actions/
799,594 frames/12,477 updates take **21m25.849s**, 0.160% less observed time than
retained 21m27.910s. Greedy **136.958** trails 143.667; sampled **77.875** exceeds
70.750. All 0/24 two-wall. Primary own-initial gain/min falls **4.295%**,
6.590132→6.307119; paired difference −.283013 has 95% interval
[−2.896655, 2.688719]. Sampled gain/min rises 10.393%, but its paired interval
also spans zero. This reused cohort does not establish training-seed reliability.
Keep preset 32×16, actor width 256, coordinates/residual/dynamics gradients off.

All 357 initial tensors match bounded and retained, with 47 matching bounded
pre-update events. Actual-initial sampled 1.458333/greedy 1.791667 and subsequent
initial traces differ from retained; comparison uses own-initial scores.
Four frozen arms select 24 natural episodes each with zero updates/truncations
and 357 invariant tensors. All 356 current/338 retained pins, 100k recovery,
weighted losses, full replays/videos, guards, curves, exposure and original
services pass. Main/collector complete **11:42:01.706/11:42:12.947 UTC November 6**,
source eda94b80. Publication reader passes in .047s. No full preparation or
production stage fails or repeats; original native qualification failure and
separate successful test repair remain preserved.

Late own-replay labeled imitation CE improves .560514→.546367 while greedy
quality falls; experience differs. Training spends 96.182% of actions below
216 points and none at 432 or above. Sampling, KL and active one-hot counts
change together, so these results do not isolate a representation cause.
Root `/x/Temp/kindle-latent-shape-20261106/full` is closed; no GPU allocation,
extension or successor is selected. Retain 143.667/70.750; original 864/90%-two-wall
and five-game gates remain unfinished.

The [frozen residual serving ablation](results/2026-11-06-cdp-residual-actor-serving-ablation.md) is complete and inconclusive.
Zeroing two correction parameters raises greedy117.625→131.750 and sampled
49.333→60.000, but both paired intervals span zero; both remain below retained
143.667/70.750, all0/24 two-wall. Greedy difference+14.125 has95% interval
[−27.252083,56.626042] with6/8 improved lanes; sampled+10.666667 has interval
[−12.291667,43.083333] with4/8 improved. This serving intervention is not a
separately trained base-only control or a cause of the original regression.

All361 other tensors/moments and source files remain exact. Both new24-natural-
episode arms have zero updates/truncations and363 invariant tensors; full replays,
videos,guards,369 pins and70 sealed files pass. Aggregate evaluation74,048 actions/
296,118 frames; no learning. Original controller/wrapper complete **10:31:27.438
UTC November6** in86.268s, source80f4559c. Independent CPU recount closes
**10:33:13.169** in.519s. No experiment failure/repeat or successor; root
`/x/Temp/kindle-residual-serving-ablation-20261106` is closed. Keep residual off
and retained143.667/70.750; original864/90%-two-wall/five-game gates remain.

The [full residual actor comparison](results/2026-11-06-cdp-residual-actor.md#full-200k-comparison)
is complete and fails retention. Fresh200k actions/799,615 frames/12,477 updates
take **21m18.238s**,0.751% less observed time than retained21m27.910s. Greedy
**117.625** and sampled **49.333** regress the143.667/70.750 point estimates;
all0/24 two-wall. Own-initial greedy gain/min falls **17.792%**,6.590132→5.417615.
Paired eight-lane difference−1.172517 has95% interval[−3.413039,1.330529]; this
reused cohort does not establish seed reliability. Keep residual actor off,
preset256 base, coordinates and dynamics gradients off.

Own initials1.416667 sampled/2.208333 greedy and all799/794 initial semantic
events match retained. All363 bounded initial tensors/47 pre-update events match;
all357 retained tensors match plus six zero residual additions. Four frozen
arms select24 natural episodes each, with zero updates/truncations and363
invariant tensors. All351 current/338 retained pins,100k recovery,weighted
losses,full replays/videos,guards,curves,exposure and original services pass.
Main/terminal complete **10:16:34.650 /10:16:47.324 UTC November6**, sourcee22b94df.
Publication reader passes in.026s. No full preparation/production stage fails
or repeats; original native fixture failure and separate repair remain preserved.

Base-only late imitation CE is.557574 versus retained.560514 on different own
replay; it does not measure the combined residual policy's imitation. Training
spends96.094% of actions below216 points and none at432 or above. These results
do not isolate a gradient-conflict or representation cause. Root
`/x/Temp/kindle-residual-actor-20261106/full` is closed, with no successor selected.
Retain143.667/70.750; original864/90%-two-wall and five-game gates remain unfinished.

The [full actor-width512 comparison](results/2026-11-06-cdp-actor-width.md#full-200k-comparison)
is complete and negative. Fresh200k actions/799,552 frames/12,477 updates take
**24m14.355s**,12.924% longer than retained21m27.910s. Greedy **87.125** and
sampled **39.958** regress143.667/70.750; all0/24 two-wall. Primary own-initial
gain/min falls **46.293%**,6.590132→3.539370. Paired eight-lane difference
−3.050762 has95% interval[−5.029410,−.914726]; this reused cohort does not
establish seed reliability. Keep actor width absent/preset256, coordinates off
and actor dynamics off. No extension, sweep, confirmation or successor.

Actual-initial sampled1.666667/greedy1.333333 differ from retained1.416667/2.208333;
comparison uses own-initial scores. All357 initial tensors/47 pre-update events
match the bounded candidate, and313 non-actor initial tensors match retained.
Four frozen arms select24 natural episodes each, zero selected truncations or
updates,357 invariant tensors per arm. All347 current/338 retained pins,100k
recovery,weighted losses,full training/frozen replays/videos,guards,curves,
exposure and terminal collection pass. No preparation/production stage fails
or repeats. Main/collector complete **08:55:25.209 /08:55:47.274 UTC November6**,
source91647453; all three services succeed. Publication reader passes in.049s.

Late own-replay labeled imitation CE improves.560514→.553252 despite worse
frozen quality; experience differs. Training spends96.826% of actions below216
points and none at432 or above. These fitting/coverage measurements do not
isolate a capacity cause. Native/runner differ from the historical control;
scoped qualification does not establish broad runtime equivalence. Root
`/x/Temp/kindle-actor-width-20261106/full` is closed, with no allocation selected.
The bounded cost check remains passed; packaged runtime retained after prior
4.807GB logical cache release. Original864/90%-two-wall and five-game goal remain.

The [completed actor-update diagnostic](results/2026-11-06-cdp-dynamics-actor-update.md) finds dynamics.5 changes
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

The [frozen action-path probe](results/2026-11-06-cdp-breakout-action-path-derivatives.md)
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

The [matched action-derivative context capture](results/2026-11-06-cdp-breakout-action-derivative-contexts.md)
is complete:120 latent origins per model,43,424 actions/173,660 frames each,
50.591s control/50.498s candidate. All16 CPU tests,537 pins,357 invariant tensors
per model,exact transitions and guards pass. Zero updates or multi-step forecasts;
no failed/repeated stage. Controller completes06:42:29.897 UTC November6, with
no active GPU allocation at closure. Root
`/x/Temp/kindle-breakout-action-derivatives-20261106` retains the states alongside
references to92 known short rescues/18 differing game-over outcomes. This supplies
inputs for a separately qualified native derivative probe; derivative quality
remains unmeasured, and no derivative GPU stage or learner is selected here.

The [full dynamics-gradient actor comparison](results/2026-11-06-cdp-actor-dynamics.md#full-200k-comparison) is complete and negative:
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

The [optional dynamics-gradient actor](experiments/2026-11-06-cdp-actor-dynamics.md) is implemented and natively qualified. A private pre-update world snapshot receives the same
initial states/draws as imagination; actor gradients traverse dynamics and frozen
reward/continuation/critic heads. Categorical straight-through derivatives are
biased; disagreement values, trajectory weights and normalization stay detached.
The existing actor losses remain, with one optimizer and no added trainable
weights or checkpoint tensors. Default off; its bounded cost check is now complete.

[Qualification](results/2026-11-06-cdp-actor-dynamics.md) passes188 Kindle+5 gym tests,1,108 packaged Python tests, strict
workspace/binding Clippy and guarded F64/production/snapshot/restore checks.
Default-off final/initial policies reproduce2,048 frozen actions and preserve357
tensors each. Original preparation failures remain; no GPU stage repeats.
The [bounded result](results/2026-11-06-cdp-actor-dynamics.md#completed-bounded-cost-and-numerics-check) completes8,192 actions/32,725 frames/489 updates in58.369s,
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

The [completed coordinate-channel comparison](results/2026-11-06-cdp-rgb-coordinates.md#full-200k-comparison)
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

The [SuperMimic source review](results/2026-11-06-supermimic-architecture-review.md) identifies direct
actor gradients through imagined dynamics as a separate architectural proposal
within the same goal. It is not implemented or allocated by this comparison.
The original 864/90%-two-wall gate and five-game goal remain unfinished.

The [completed categorical-mixing comparison](results/2026-11-04-cdp-breakout-latent-unimix001.md#full-200k-comparison) finishes 200k actions /
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

The [saved corrective-label timing audit](results/2026-11-04-cdp-breakout-label-timing.md) finds the retained learner's
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

The [completed B32/T16 comparison](results/2026-11-04-cdp-breakout-batch32.md#full-200k-comparison) finishes200k actions/799,541 frames/
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

The [completed discount100 comparison](results/2026-11-04-cdp-breakout-discount100.md#full-200k-comparison) finishes
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

The [completed dynamics-scale3 comparison](results/2026-11-04-cdp-breakout-dynamics-scale3.md#full-200k-comparison)
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

The [completed B32/T8 check](results/2026-11-04-cdp-breakout-b32-t8.md) takes46.818s for8,192 actions/
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

The [completed catch-reward comparison](results/2026-11-04-cdp-breakout-catch-reward1.md#full-200k-comparison)
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

The [bounded one-point catch-reward check](results/2026-11-04-cdp-breakout-catch-reward1.md)
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

Select one [fresh200k catch-reward comparison](experiments/2026-11-04-cdp-breakout-catch-reward1-200k.md) after this cost pass.
Preparation is pending;no full learner or GPU allocation is active yet. Use the
exact qualified overlay,30-minute cap,100k/6,227 recovery and four actual-initial/
final sampled/greedy24-episode assessments. Primary greedy raw score and
initial-subtracted gain/min must improve with a positive paired interval.
No bounded experience transfer,coefficient sweep,extension or automatic successor.
Check live state before launch. Retain143.667 greedy/70.750 sampled;original
864/90%-two-wall and the five-game goal remain unfinished.

The [completed raw-motion catch validation](results/2026-11-04-cdp-breakout-catch-validation.md)
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

Next selected development is the [bounded one-point catch-reward check](experiments/2026-11-04-cdp-breakout-catch-reward1-check.md).
Implementation and CPU qualification are pending,followed by one fresh8,192-action
check and1,024 frozen actions. Default off;training-only explicit pixel reward;
raw game scores and all frozen assessments remain unaided. Native447fe50 and
teacher unchanged. No full learner,extension,coefficient sweep or automatic
successor. No GPU allocation is active at selection. Retain143.667 greedy/
70.750 sampled;original864/90%-two-wall and the five-game goal remain unfinished.

The [CPU-only catch-signal diagnostic](results/2026-11-04-cdp-breakout-catch-signal.md)
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
scale0.1 after the [lower-KL regression](results/2026-11-04-cdp-breakout-representation001.md#full200k-comparison). Original864/90%-two-wall
and the five-game goal remain unfinished.

The [completed representation-KL comparison](results/2026-11-04-cdp-breakout-representation001.md#full200k-comparison)
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

The [bounded representation-KL check](results/2026-11-04-cdp-breakout-representation001.md)
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

The [matched prior-continuation arrival diagnostic](results/2026-11-04-cdp-breakout-prior-continuation-arrivals.md)
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

The [completed prior-continuation comparison](results/2026-11-04-cdp-breakout-prior-continuation.md#full200k-comparison)
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

The [one-step prior continuation bounded check](results/2026-11-04-cdp-breakout-prior-continuation.md)
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

The [completed terminal-weight ranking diagnostic](results/2026-11-04-cdp-breakout-terminal-weight-ranking.md)
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

All 24 CPU tests and1,125 pins pass. One2.060s single-core CPU analysis completes
07:27:01 UTC November4/source db4840bd/PID1144889. Independent4.073s closure
completes07:27:22, verifying16 artifacts,13,824 pair orders,240 transformed outputs
and all arithmetic. No failure, repair, repeat, model/emulator execution or GPU
work occurs. Root`/x/Temp/kindle-terminal-weight-ranking-20261104` is closed.

Retain143.667 greedy/70.750 sampled over weight10's120.542/66.375, all0/24 two-wall.
No correction, loss, learner or automatic successor is selected. No learner or
GPU allocation is active at closure; check live state. Do not repeat completed
captures or learners. Original864/90%-two-wall and five-game goal remain unfinished.

The [completed terminal-weight arrival diagnostic](results/2026-11-04-cdp-breakout-terminal-weight-arrivals.md) finds terminal
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

The [completed terminal-weight10 comparison](results/2026-11-04-cdp-breakout-terminal-weight.md) scores120.542 greedy/66.375
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

The [terminal-only continuation check](results/2026-11-04-cdp-breakout-terminal-weight.md) passes8,192 actions/32,722 frames/489 updates in46.934s.
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

The [completed terminal ranking check](results/2026-11-04-cdp-breakout-terminal-ranking.md) finds posterior AUC
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

The [completed terminal-mixture arrival diagnostic](results/2026-11-04-cdp-breakout-terminal-mixture-arrivals.md) finds a tradeoff:
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

The [completed terminal-replay comparison](results/2026-11-04-cdp-breakout-terminal-mixture.md) increases ending exposure
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

The [completed fixed terminal-state readout](results/2026-11-04-cdp-breakout-terminal-readout.md) finds retained posterior
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

The [completed continuation10 arrival comparison](results/2026-11-04-cdp-breakout-continuation10-arrivals.md) finds observed-terminal
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

The [completed200k continuation10 comparison](results/2026-11-04-cdp-breakout-continuation10.md#full200k-comparison)
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

The [bounded continuation10 result](results/2026-11-04-cdp-breakout-continuation10.md)
passes46.764s training/.872s frozen,34 canonical/34 packaged CPU checks,699 pins
and independent5.398s closure at03:43:00 UTC November4. All84 final artifacts
verify, with no experiment-stage failure or repeat. Full selection is separate;
no bounded experience transfers and no gameplay benefit is established.

The [completed Breakout arrival-head diagnostic](results/2026-11-04-cdp-breakout-arrival-head.md) lowers mean terminal
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

The [completed Breakout terminal replay audit](results/2026-11-04-cdp-breakout-terminal-replay.md) finds5,376 terminal
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

The [completed Breakout ending-feedback diagnostic](results/2026-11-04-cdp-breakout-ending-feedback.md) lowers mean
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

The [completed Breakout rescue forecasts](results/2026-11-04-cdp-breakout-rescue-forecasts.md) favor the surviving sequence
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

The [completed200k block-guidance comparison](results/2026-11-04-cdp-breakout-block32.md) scores74.875 greedy/41.250
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

The [completed CPU teacher-takeover comparison](results/2026-11-04-cdp-breakout-command-takeovers.md) finds32-action survival
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

The [completed trained-policy posterior comparison](results/2026-11-04-cdp-breakout-value1-modal.md) scores modal175.708
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

The [completed encoder8e-5/current-critic comparison](results/2026-11-04-cdp-breakout-value1-encoder8e5.md) finishes200k
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

The [completed RGB frame-pair comparison](results/2026-11-03-cdp-rgb-frame-pair.md#full-200k-comparison-retains-single-image-input)
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

The [completed50% episode-guidance Breakout comparison](results/2026-11-03-cdp-breakout-guidance50.md)
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

The [50% episode-guidance Breakout check](results/2026-11-03-cdp-breakout-guidance50.md) passes
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

The [completed current-critic B8/T32 Breakout comparison](results/2026-11-03-cdp-breakout-value1-context32.md#full-200k-comparison)
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

The [current-critic Breakout B8/T32 check](results/2026-11-03-cdp-breakout-value1-context32.md) passes8192 actions /32,722 frames /
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

The [completed four-action Breakout comparison](results/2026-11-03-cdp-breakout-minimal-guided.md#full-200k-comparison) finishes
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

The [four-action guided Breakout check](results/2026-11-03-cdp-breakout-minimal-guided.md) completes8192 actions /
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

The [completed life-bounded actor comparison](results/2026-11-03-cdp-life-return-actor.md#full-200k-comparison-does-not-improve-the-retained-recipe)
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

The [optional life-bounded actor qualification](results/2026-11-03-cdp-life-return-actor.md)
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

The [completed return/life audit](results/2026-11-03-cdp-breakout-return-life-audit.md)
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

The [completed200k conservative-actor comparison](results/2026-11-03-cdp-conservative-actor.md)
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

The [completed bounded check](results/2026-11-03-cdp-conservative-actor.md) passes8192/489 in47.402s control/50.754s
candidate (+7.07%),with33,588 positive targets and the bonus active. Padded200k
projection22m43s passes30min; frozen1024 has zero updates/no natural episodes.
Native3f6c6648 passes1018 Python/169 Rust CPU checks,Clippy and12 F64 cases.
All258 current/338 retained pins,215 preserved sources,357 initial/frozen tensors,
replays and five GPU stages pass. A CPU comparison matches357 final control
arrays/1065 events to the old short control. Closed17:58:41.844 UTC November3;
CPU fixture/path failures remain,no GPU repeats. Bounded root`check` is closed.
Actual full CPU postprocessing and queued-config preflights pass before launch;
one private copy-script assertion is preserved. See the result for evidence.

The [completed-return actor pair](results/2026-11-03-cdp-episodic-actor.md) closes with a useful bonus-off
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

The [closed bounded pair](results/2026-11-03-cdp-episodic-actor.md)
passed8192 actions/489 updates in36.677s control/38.277s candidate (+4.36%),
with33105 positive replay rows and no natural frozen episodes. All native/CPU,
legacy, tensor, replay and terminal checks pass; no failed or repeated stage.
Native e3b02163/SHA d406629f adds no parameters; disabling disagreement removes96
ensemble tensors, leaving261. Runtime,wheels,validation binary and1.008GB cache
remain. Bounded root `check` is closed; never reuse its experience or rerun it.

The [CPU replay-actor target audit](results/2026-11-03-cdp-breakout-replay-actor-targets.md) confirms no direct
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

The [completed400k/R16 Breakout comparison](results/2026-11-03-cdp-breakout-budget400k.md)
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

The [completed frozen halfway assessment](results/2026-11-03-cdp-breakout-halfway.md)
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

The [completed context9 comparison](results/2026-11-03-cdp-breakout-context9.md)
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

The [optional replay-state recomputation check](results/2026-11-03-cdp-replay-context.md)
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

The [CPU replay-context age reconstruction](results/2026-11-03-cdp-breakout-replay-age.md)
matches all12,477 logged updates and199,632 sampled starts in.387s CPU.
Final50k unmasked contexts have median246/mean365.099/p99 1,776 updates since
their last computation;98.66% of labeled actor rows can depend on initial
context before a reset. This measures age,not latent error or a gameplay cause.
All15 fixtures and27 diagnostic/338 control pins pass; review completes
12:35:19.800 UTC November3/sourceb729ab23. No GPU work,learner,emulator,failure
or prior stage repeat. Root `/x/Temp/kindle-breakout-replay-age-20261103` is
closed. Earlier replay16k and B8/T32 results remain; this age audit alone
did not select a new buffer,prefix,loss or learner. Retain the70.750/143.667 control; goal unfinished.

The [completed Breakout H3/scale1 comparison](results/2026-11-03-cdp-breakout-open-loop.md)
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

The [Breakout H3 world-supervision check](results/2026-11-03-cdp-breakout-open-loop.md)
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

The [completed actor-only imitation-decay comparison](results/2026-11-03-cdp-breakout-actor-only-decay.md)
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

The [actor-only imitation-decay qualification](results/2026-11-03-cdp-actor-only-imitation-decay.md)
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

The preceding [actor optimizer attribution](results/2026-11-03-cdp-breakout-actor-optimizer.md)
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

The [completed policy-weight100 comparison](results/2026-11-03-cdp-breakout-policy100.md)
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

The [completed contact-timing CPU teacher comparison](results/2026-11-03-cdp-breakout-contact-timing-policy.md)
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

The preceding [saved contact-timing diagnostic](results/2026-11-03-cdp-breakout-paddle-contact-timing.md)
passes its fixed prediction screen: episode2 loss-contact MAE6.101→5.757 pixels,
successful-contact MAE2.376→2.283, with lower RMSE and seven improved lane means.
Some tails worsen. All24 pins, five new/eight reused fixtures and739-row/875-metric
recount pass; sourceb7bafe64, completion09:10:22.591 UTC November3. Its root is
closed, with no failed/repeated stage, emulator/native import, GPU work or update.
The later gameplay failure shows why this aggregate error gain was insufficient
for teacher integration. It does not diagnose the native learner's causal limit.

The [current-position](results/2026-11-03-cdp-breakout-paddle-timing.md)
and [next-position](results/2026-11-03-cdp-breakout-response-timing.md)
saved-data screens also pass; their roots are closed. All timing diagnostics
reuse development episodes and teacher states. The original runtime teacher
and retained native learner remain unchanged.

The [completed imagined-value0.3 comparison](results/2026-11-03-cdp-breakout-imagined-value03.md)
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

The [completed actor/critic-gradient comparison](results/2026-11-03-cdp-breakout-acgrads-value1.md) lowers sampled
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

The [fixed subaction diagnostic](results/2026-11-03-cdp-breakout-subaction.md) finds no extra rescues from one
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

The [completed life-loss comparison](results/2026-11-03-cdp-breakout-life-penalty.md) lowers sampled70.750→42.542
and greedy143.667→90.875 in21m39.349s (+0.89% training time), all0/24 two-wall.
Initial-subtracted return/min falls41.21%/37.87%. Retain raw training rewards
with12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100/replay100k and sampled collection. Only the optional
one-point life-loss aid changes; raw scores, native447fe50 and teacher stay fixed.
All1,290 losses/negative rewards independently replay; online94.72 is guided
and does not outweigh worse frozen play. Reused1009/cohort limits reliability.

All453 pins,100k/6,227 recovery,357 matched initial/invariant frozen tensors,
paired initial traces, full replays/videos, guards, curves, exposure and comparison
pass. Main/post/collection succeed06:42:06.048/06:42:24.088/06:42:57.559 UTC
November3/sourceb80194e. Independent terminal06:56:45.055 and coverage06:56:53.724
pass once;50 diagnostic/453 full pins reconcile277,592 actions/1,109,787 frames.
Native actions at raw scores216–431 barely rise1,262→1,297, with none432+.
These are descriptive score bins, not verified walls or a causal explanation.

Root`/x/Temp/kindle-breakout-life-penalty-20261103/full` is closed. All five
original services and recovery/terminal/coverage readers are complete; do not
repeat them or completed stages. No full-run failure or GPU repeat occurs.
Three earlier bounded CPU failures/corrections remain preserved. Completion
SHA4d276268/terminalSHA4e848210/coverageSHA96930bc4. No Kindle allocation is active
at closure; check live state before work. Keep CPU co-host timings uncorrected.
No penalty sweep, extension, confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed replay16k comparison](results/2026-11-03-cdp-breakout-replay16k.md) lowers sampled70.750→34.000
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

The [completed lambda1 comparison](results/2026-11-03-cdp-breakout-lambda1.md) scores **45.667 sampled /
90.292 greedy**, below retained lambda0.95's70.750/143.667 by35.45%/37.15%,
in **21m52.277s** (+1.89%). Both final arms have0/24 two-wall successes.
Retain lambda0.95 with12M64/Auto/N8/B16/T16/H15/R16/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100 and sampled collection. Lambda changes
both imagined/replay targets; no isolated mechanism or seed-reliability claim.

All448 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
paired initial traces,replays/videos,guards,curves,exposure andcomparison pass.
Main/postprocess/collection succeed04:39:49.589/04:40:12.443/04:40:16.846 UTC
November3/source958d622; allfive original services succeed. Terminal confirmation
04:40:46.018 andcoverage04:40:51.786 pass once;50 diagnostic/448 full pins,
282,720 actions/1,130,258 frames reconciled. Native216–431 actions rise1,262→1,434,
none432+; score bins are not verified walls or causal coverage. Expert CE falls
.560514→.536404 despite worseplay; actual overrides45.3985% are not competence.

Root`/x/Temp/kindle-breakout-lambda1-20261103/full` isclosed. Recovery,terminal
andcoverage readers arecomplete; do not repeat them orcompleted stages. Earlier
CPU import/date-wiring failures andrepair remain; no full-run failure orrepeat.
Native447fe50/teacher unchanged; qualifiedrunner9eb21cca adds10 CLI lines.
Keep CPU co-host caveats. No extension,confirmation,sweep orautomatic successor
selected. No Kindle allocation isactive atclosure; check live state beforework.
The five-game goal remains unfinished.

The [lambda1 bounded check](results/2026-11-03-cdp-breakout-lambda1.md)
passes8,192 actions/32,727 frames/489 updates in50.121s; frozen1,024/.868s has
zero updates/no natural episodes. Conservative200k projection22m26.028s clears
30min. All385 pins,357 matched initial/invariant tensors,47 pre-update events,
full replay andguards pass; main/collector succeed04:07:39.518/04:07:43.412 UTC
November3/sourcef466942. Terminal summary04:08:34.276, no bounded stage repeats.
Native447fe50/teacher fixed; runner adds10 CLI lines.515 current/22 private
Python checks,eight refusals,four exact-source scalar Rust cases pass; initial
import failure retained. Lambda changesboth imagined/replay traces; gameplay
is unmeasured. Root`/x/Temp/kindle-breakout-lambda1-20261103/native-check` is
closed, no full learner orallocation active; do not repeat completed stages.

The [fixed pixel-motion readout](results/2026-11-03-cdp-breakout-pixel-motion-readout.md)
fails its screen: learned-plus-motion teacher agreement55.91% overall/55.88%
near descending trails actor62.56%/66.91%. One new CPU predecessor capture
matches20,960 actions/83,799 frames and2,064 saved current pixel predictions;
existing features and baseline fits are reused. Six fixtures/75 pins pass;
capture12.177s+fit.950s, completion03:56:34.670 UTC November3/sourcec91dcfa.
No failure,GPU work or updates. No temporal-input prototype or learner selected;
reused older control states and readout geometry limit interpretation. Root
`/x/Temp/kindle-breakout-pixel-motion-20261103` is closed; completed stages must
not repeat. No service is active at closure; the five-game goal remains unfinished.

The [completed Breakout crop comparison](results/2026-11-02-cdp-breakout-playfield-crop.md)
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

The [completed Breakout 25M comparison](results/2026-11-02-cdp-breakout-capacity25m.md)
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

The [Breakout 25M bounded check](results/2026-11-02-cdp-breakout-capacity25m.md)
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

The [saved Breakout crop-gap diagnostic](results/2026-11-02-cdp-breakout-crop-gap.md)
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

The [posterior-bootstrap comparison](results/2026-11-02-cdp-breakout-posterior-bootstrap.md) is **closed_protocol_failed**.
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

The [completed scalar value calibration diagnostic](results/2026-11-02-cdp-breakout-value-calibration.md)
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

The [completed replay-value critic diagnostic](results/2026-11-02-cdp-breakout-replay-value-critic.md) finds that the
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

The [completed Breakout H30/R12 comparison](results/2026-11-02-cdp-breakout-h30-r12.md) scores **58.917 sampled /
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

The [completed lower-dynamics/replay-value1 comparison](results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
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

The [completed Qbert R20 fresh-seed confirmation](results/2026-11-02-cdp-qbert-r20-seed18013.md)
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

The [Breakout ascending-centering reference](results/2026-11-02-cdp-breakout-ascending-center.md) fails its expanded24-episode
screen:233.208 versus225.083 (+3.61%),four ofeight lane means improve,all0/24
two-wall. The earlier eight-episode gain46.93%/six lanes remains recorded;
all16,425 first-episode semantic events reproduce. Retain the original teacher;
no fresh cohort,integration,learner or sweep is selected. All31 diagnostic/1,046
active Qbert pins,full pixel/action/debug replay and original learner identity
pass. CPU collection35.990s/replay45.402s/controller87.435s completes06:06:24.338
UTC November2/sourcef8e4931,with no GPU work or updates. Root
`/x/Temp/kindle-breakout-ascending-center-20261102/standard24` is complete;
do not repeat stages. CPU co-host cost remains uncorrected.

The [completed Breakout greedy-collection comparison](results/2026-11-02-cdp-breakout-greedy-training.md)
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

The [coverage follow-up](results/2026-11-02-cdp-breakout-score-coverage.md#greedy-collection-follow-up)
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

The earlier [greedy-collection prerequisite](results/2026-11-02-cdp-breakout-greedy-training.md#earlier-bounded-prerequisite)
passes8,192 actions/489 updates in47.418s;1,024 frozen actions/.869s havezero
updates/no natural episodes. All387 pins,tensors,replay,guards andoriginal
services pass. It precedes the separately selected full comparison above; its
experience is not reused. The45 CPU tests and separate wrapper qualification
remain retained. Native-check completionSHAfc5ef001/summarySHA86262978;
do not repeat the bounded check or its readers.

The [Qbert 240k/R20 comparison](results/2026-11-02-cdp-qbert-r20-240k.md) completes 240,000 actions,
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

The [saved Breakout score-coverage diagnostic](results/2026-11-02-cdp-breakout-score-coverage.md)
finds97.03% ofcandidate training actions below216 points andzero at432+,with
maximum408 versus control418. Of5947 candidate actions at216–431,4685 areteacher
overrides; native exposure there is1262 actions (0.63% oftraining). Frozen greedy
means117.292→143.667 still havezero432+ episodes. Score fractions are not verified
wall boundaries; this locates scarce experience,not an isolated cause orselected
change. All400k training/175144 frozen actions,seven fixtures,31 original/six
repair/991 active pins reconcile. Original cumulative-counter failure remains;
separate9.051s CPU analysis completes04:37:06.430/sourceb608d35 withno GPU work.
Root`/x/Temp/kindle-breakout-score-coverage-20261102` iscomplete; do not rerun.

The earlier [Qbert R20 bounded check](results/2026-11-02-cdp-qbert-r20-240k.md)
completes8192 actions/32760 frames/611 updates in50.566s;1024 frozen/.868s has
zero updates/no natural episodes. Conservative240k projection27m09.558s passes
30min. All940 pins,357 matched initial/invariant frozen tensors,47 pre-update
semantic events,finite reports,replay andguards pass. Main/collector succeed
04:05:09.369/04:05:13.602 UTC/source8f7888a; terminal reader04:05:26.383.
No stage fails orrepeats. OnlyR24→20 changes onreused16001, with native447fe50,
runnere304af89,teacher andcurrent12M64/H5/dynamics1e-4/replay-value1 fixed.
This prerequisite preceded the separately selected full comparison above.
Its early cost projection did not establish gameplay benefit. Earlier
larger-budget regressions remain relevant. Root`/x/Temp/kindle-qbert-r20-240k-20261102/native-check` isclosed;
completionSHAdf02e12d/summarySHA15fb8ccd. No active allocation atclosure; check
live state. Do not repeat the check,collector orreader.

The [completed Qbert replay-value1 crossed assessment](results/2026-11-02-cdp-qbert-replay-value1-cross-cohort.md) keeps14009
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

The [completed Breakout replay-value1 comparison](results/2026-11-02-cdp-breakout-replay-value1.md) scores
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

The [Qbert replay-value1 phase diagnosis](results/2026-11-02-cdp-qbert-replay-value1-phases.md) locates the entire
frozen score gap after identical550-point first-pyramid prefixes inall96 episodes.
Both learners collect about76% later actions; late native movement agreement is
77.50%/77.07%. Greedy median gap150 is smaller than mean1611.458; fresh scores
range11175–15650 versus14800–22025. These are descriptive cohort differences,
not an isolated cause or changed gate. All27 diagnostic/338 active pins and
fixtures pass in4.038s CPU at03:19:44.467/source305d145; original Breakout identity
stays unchanged. No GPU work,replay repeat,loss change or new learner. Do not rerun.

The [completed Qbert replay-value1 confirmation](results/2026-11-02-cdp-qbert-replay-value1-seed16001.md) scores
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

The [completed Breakout real-trajectory value diagnostic](results/2026-11-02-cdp-breakout-value-consistency.md) finds
sampled one-step residual RMSE.732/.941 and15-step2.546/3.121,while complete-return
RMSE remains6.682/6.691. Short targets bootstrap with the same critic; they do
not prove calibrated value or native-target fitting. Last15 sampled mean values
14.782/19.343 exceed realized.377/.592,an outcome-conditioned slice,not causal
proof. All32 diagnostic/73 source/912 active pins pass;6.366s CPU-only analysis
completes02:27:46.302/source9010df1. Original import failure/repair remain;
Qbert's original identity is unchanged. No new loss orlearner is selected.

The [completed Qbert replay-value scale1 comparison](results/2026-11-02-cdp-qbert-replay-value1.md) finishes200k actions /
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

The [completed Qbert replay-value scale1 check](results/2026-11-02-cdp-qbert-replay-value1.md) passes8192 actions /
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

The [completed Breakout actor/critic-rate comparison](results/2026-11-02-cdp-breakout-behavior8e5.md) finishes
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

The [fresh Qbert confirmation](results/2026-11-01-cdp-qbert-dynamics1e4-seed14009.md)
completes 200,000 actions / 799,795 frames / 18,715 updates in **25m09.906s**.
Greedy **14,770.833** and sampled **14,254.167** each clear 24/24 first pyramids
but miss the fixed 15,000 mean gate. Actual initials score 147.917/205.208 and
clear none. The unchanged dynamics1e-4 recipe therefore passes greedy on **one
of two learners** (development12011 passes; fresh14009 fails), not two-seed
confirmation. No extension, seed substitution, retry or third learner is selected.

All 768 pins, 100k/9,340 recovery, 357 bounded-matching initial and invariant
frozen tensors, full training/frozen replays and videos, host guards, curves,
exposure and comparison pass. Main/postprocessing/collection succeed at
00:00:06/00:00:27/00:00:58 UTC November2 under the original identities; all five
services are terminal success. No original stage fails or repeats. Completion
SHA9a56dbda. Native447fe50/35ad0db, runner e161bf38 and teacher fa261c53 remain
fixed. Only learner/evaluation seeds change. Actual guidance is 59,478 actions
(29.739%); online scores are aided. CPU co-host costs remain uncorrected.

Root `/x/Temp/kindle-qbert-dynamics1e4-seed14009-20261101/full` is closed.
The earlier12011 original main/collector failures and separate CPU repair remain
preserved. Do not restart either learner or completed collectors. The five-game
goal is unfinished; no Qbert successor is selected.

The [completed Breakout action-ranking comparison](results/2026-11-02-cdp-breakout-current-ranking.md)
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

The [completed Breakout critic diagnostic](results/2026-11-02-cdp-breakout-current-critic.md)
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

The [completed Qbert dynamics-group1e-4 comparison](results/2026-11-01-cdp-qbert-dynamics1e4.md)
finishes200k actions/799,807 frames/18,715 updates in25m40.027s (+.83% time).
Greedy15,471.875/24 first pyramids passes15k/90%,up22.04% from12,678.125;
sampled14,862.5/24 rises22.89% from12,093.75 but misses15k. Own initials
168.750/167.708 match. Initial-subtracted return/min rises21.33% greedy/22.20%
sampled. Retain1e-4 as a useful development point; reused12011 is not independent
confirmation. Late imitation CE worsens.044040→.069685 despite better play.

All606 original/111 repair pins,100k/9,340 recovery,357 matching initial and
invariant frozen tensors,1,176/1,223 paired initial semantic events,full pixel/
action/label and frozen replays/videos,guards,curves,coverage andcomparison pass.
Original main fails23:00:53 UTC November1 on undefinedCONTROL in CPU audit,
after allfive GPU stages pass; original collector fails andboth remain failed.
Separate CPU servicePID862557/invocatione48562cf43cd4ba8927950760ef61922 completes
23:11:13 in297.912s with`complete_with_cpu_audit_repair`,SHA8f9d7c25. No original
passing completion exists. Two repair-preparation failures and one supplemental
journal-reader failure remain preserved; no GPU or completed replay/collector
stage repeats. Native447fe50/35ad0db,teacherfa261c53 andrunnere161bf38 stay fixed.

Only dynamics-group4e-4→1e-4 changes at12M64/Auto/N8/B16/T16/H5/R24/encoder-base
4e-5/greedy/expert100; no isolated-core mechanism or broad reliability claim.
Actual overrides47,700 (23.85%);online guided returns are not frozen competence.
All original services andCPU repair are terminal;no Kindle GPU allocation is
active at closure. Root`/x/Temp/kindle-qbert-dynamics1e4-20261101/full` iscomplete.
Do not rerun original services,completed GPU/replay stages or collectors. No
extension,rate sweep,confirmation or automatic successor is selected. The
five-game goal remains unfinished;check live state before further GPU work.

The [current saved-log cost summary](results/2026-11-01-cdp-current-training-cost.md)
finds learning81.14–81.31% of Qbert wall time and87.57–88.72% of Breakout;
emulator stepping3.99–4.89%. Within updates,Qbert world training44.21–44.53%,
posterior23.68–23.73%,imagination19.88–20.02%;Breakout imagination39.84–39.98%.
These are wrapper/learner wall timers,not exclusive GPU costs or a selected
optimization. All62,384 reports/39 completed receipt files,21 analysis pins and
606 active inputs pass with originalPID857991/invocation unchanged. CPU analysis
4.289s completes22:42:05 UTC/sourcea369f61 after preserving a1.894s failed wrapper
that incorrectly required independently measured timers to sum exactly. Separate
repair retains inter-stage residuals;no completed analysis or GPU stage repeats.
Keep co-host costs uncorrected. Original Qbert learner/observers continue;no
new allocation or successor. Root`/x/Temp/kindle-current-training-cost-20261101`.

The [completed bounded Qbert dynamics1e-4 check](results/2026-11-01-cdp-qbert-dynamics1e4.md)
passes8192 actions/32,756 frames/733 updates in57.461s;1024 frozen/.853s haszero
updates/no natural episodes. Conservative200k projection25m43.153s clears30min.
All616 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
finite supervised reports, full pixel/action/label replay andhost guards pass.
Main/collector succeed22:27:02/22:27:09 UTC November1/source1a1eceb, original
PID857526/invocation655eec1976bd483e9f5b780a773a69bc. Both services are terminal
success; no original stage fails or repeats. No GPU allocation remains active.
Root `/x/Temp/kindle-qbert-dynamics1e4-20261101/native-check` iscomplete; do not
rerun or reuse its training experience. Gameplay benefit remains unmeasured.

The [completed Qbert RGB64 action-ranking comparison](results/2026-11-01-cdp-qbert-rgb64-ranking.md)
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

The [completed Qbert RGB64 critic diagnostic](results/2026-11-01-cdp-qbert-rgb64-critic.md)
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

The [completed Qbert RGB64 cross-cohort assessment](results/2026-11-01-cdp-qbert-rgb64-cross-cohort.md) keeps10007
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

The [completed Breakout dynamics-group comparison](results/2026-11-01-cdp-breakout-dynamics1e4.md) is mixed:
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

The [current Qbert RGB64 phase analysis](results/2026-11-01-cdp-qbert-rgb64-phases.md) finds weaker12011 has
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

The [bounded Breakout dynamics-group1e-4 check](results/2026-11-01-cdp-breakout-dynamics1e4.md) passes8192 actions/
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

The [fresh Qbert RGB64 seed12011 confirmation](results/2026-11-01-cdp-qbert-rgb64-seed12011.md) completes but misses the
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

The [completed exact Breakout policy capture](results/2026-11-01-cdp-breakout-policy-readback.md) reproducesall37,016 actions/
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

The [fixed current-pixel action readout](results/2026-11-01-cdp-breakout-pixel-action-readout.md) fails its screen:
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

The [completed Qbert12M/RGB64 comparison](results/2026-11-01-cdp-qbert-rgb64.md) reaches **15,710.417 greedy /24 first pyramids**,
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

The [fixed Breakout one-step risk controller](results/2026-11-01-cdp-breakout-one-step-risk.md) lowers mean return
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

The [fixed Breakout response-error check](results/2026-11-01-cdp-breakout-response-uncertainty.md) passes its
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

The [command-dependent Breakout teacher comparison](results/2026-11-01-cdp-breakout-command-response.md) is complete.
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

The [completed Qbert encoder8e-5 comparison](results/2026-11-01-cdp-qbert-encoder8e5.md) lowers both frozen
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

The [Breakout full-width paddle diagnostic](results/2026-11-01-cdp-breakout-paddle-crop.md) acceptszero of42,303
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

The [completed Qbert300k comparison](results/2026-11-01-cdp-qbert-300k.md)
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

The [completed CPU prefix comparison](results/2026-11-01-cdp-qbert-budget-prefix.md)
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

The [completed Qbert25M/R16 comparison](results/2026-11-01-cdp-qbert-25m-r16.md)
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

The [existing centroid-target CPU reference](results/2026-11-01-cdp-centroid-target-precision.md)
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

The [bounded Qbert25M/R16 check](results/2026-11-01-cdp-qbert-25m-r16.md)
completes8,192 actions/32,762frames/489updates in56.153s;1,024 frozen/1.159s
haszero updates/no natural episodes. All501 inputs,357 derived/invariant tensors,
finite reports,full pixel/action/label replay and guards pass. Main16:44:28/
original collection16:44:45 UTC November1/source d211ad3;no failure,GPU repeat
or repair.31,636,273 trainable parameters;no cross-capacity initial equality.
Root`/x/Temp/kindle-qbert-25m-r16-20261101/native-check`;complete,do not rerun.

The [saved Qbert training-phase diagnostic](results/2026-11-01-cdp-qbert-training-phases.md)
finds76–77% of collected actions after the first pyramid across three H5/R24
learners,rising to87–89% in the final50k. Late native movement agreement is100%
before the milestone but70–72% afterward;overrides are excluded. Broad absence
of later-phase exposure is unsupported,while difficult-state coverage,fitting,
representation and teacher limitations remain unresolved. All600k actions,
357 completed episodes,24 partial tails and17 inputs reconcile. CPU-only3.440s
completes16:34:52 UTC November1/source2365121,zero replay,GPU work,updates or
failures. No learner/intervention selected;host idle. Root
`/x/Temp/kindle-qbert-training-phases-20261101`;complete,do not rerun.

The [completed Breakout B8/T32 comparison](results/2026-11-01-cdp-breakout-context32.md)
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

The [saved centroid-head CPU diagnostic](results/2026-11-01-cdp-centroid-head-readout.md)
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

The [saved Qbert behavior-rate outcome diagnostic](results/2026-11-01-cdp-qbert-behavior-phases.md) retains the425-point
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

The [completed Qbert behavior-rate comparison](results/2026-11-01-cdp-qbert-behavior8e5.md) finishes200k actions/
799,828frames/18,715updates in20m52.376s,26.92% less training time than control.
Greedy14,358.333/23 first pyramids versus14,339.583/24 is nearly unchanged;
sampled14,173.958/24 versus13,761.458/24 rises3.00%. Both means still miss15k.
Retain the qualified faster teacher;doubled actor/critic rate does not establish
clear quality improvement or a new default. Acting/feedback falls566.324→184.264s,
accounting for82.83% of total time reduction;this includes Python teacher work,
not an isolated native speed measurement. All524 inputs,357 matching/invariant
tensors,paired initial traces,retained100k/9,340 four-file recovery,full replays/
videos,guards,curves,coverage/exposure and original collector pass. All five
services succeed:main15:45:24/post15:45:34/collection15:46:07 UTC November1,
source6089e8b. No failed or repeated GPU stage;CPU co-host caveats remain.
Root`/x/Temp/kindle-qbert-behavior8e5-20261101/full`;complete,do not rerun.
No Qbert extension,rate sweep or confirmation selected;the goal is unfinished.

The [completed Breakout B8/T32 bounded check](results/2026-11-01-cdp-breakout-context32.md) passes8,192 actions/
32,730frames/481updates in53.799s;1,024 frozen/.870s haszero updates/no natural
episodes. Conservative200k projection24m04.805s passes30min. All261 inputs,357
matching initial/invariant frozen tensors,47 common-prefix events,481 finite
supervised reports,full pixel/action/label replay and both host guards pass.
Main15:52:52/original collection15:53:01 UTC November1/sourcec2f7abf;no failure,
GPU repeat,cap change or native edit. Longer gradients also change diversity,
target rows and warmup;cached recurrent history already exists. Native447fe50/
runner6c1b69fb/teacher stay fixed. Root`/x/Temp/kindle-breakout-context32-20261101/native-check`.
Complete,do not rerun. No full learner or automatic successor selected;the host
is idle at closure. Gameplay benefit remains unmeasured;the goal is unfinished.

The [saved Breakout takeover-state join](results/2026-11-01-cdp-breakout-takeover-states.md) finds onlytwo of21 earlier-only
rescues have projected contact withinfive actions at delay16;14 already match
the teacher command. Timing estimates and this imperfect teacher do not isolate
the cause. All480 origins/478 saved-branch survival recounts,nine declared inputs
and524 active Qbert inputs pass;originalPID809245/invocation stays. Analysis2.766s
completes15:23:52 UTC November1/source0e77ede,recount15:24:44. Zero emulator/native/
GPU work,updates,failures or repeats;retain CPU co-host timing. No new loss,teacher
or learner selected. Root`/x/Temp/kindle-breakout-takeover-states-20261101`;complete,
do not rerun. The Qbert behavior-rate comparison above subsequently completes.

The [qualified Python Qbert teacher overlay](results/2026-11-01-cdp-qbert-teacher-cost.md)
preserves472 snapshot decisions/states and the complete200k-action/799,841-frame
training-label replay while reducing observed CPU replay633.898→230.331s.
All130 CPU tests,25 declared/506 completed H3 inputs and16 package files pass;
only equivalent channel predicates and distribution RECORD change. Prefer
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`,same native447fe50/SHA35ad0db,
teacherSHAfa261c53,for future explicitly pinned Qbert trials. Replay15:01:16/
packaging15:03:33 UTC November1;old runtimes and CPU harness failure remain.
Whole-training timing is now reported in the completed behavior-rate comparison above.
Do not repeat completed qualification,replay or packaging. The goal is unfinished.

The [fresh fixed Breakout centering cohort](results/2026-11-01-cdp-breakout-unseen-center-fresh.md)
fails the10%/five-lane screen:209.833→212.042 (+1.05%),four of eight lane means
improve,both0/24 two-wall. Keep the native teacher;no integration,parameter retry
or cohort sweep follows. Both full exact replays/videos,31 declared/506 active
Qbert inputs and original Qbert identity pass. CPU-only135.931s completes14:26:31
UTC November1/sourceb7edcf8,zero updates/failures/repeats. Retain CPU co-host time.
Root`/x/Temp/kindle-breakout-unseen-center-fresh-20261101`;complete,do not rerun.
The Qbert H3 comparison below also closes;check live state before GPU work.

The [completed Qbert H3/R24 comparison](results/2026-11-01-cdp-qbert-h3-r24.md)
finishes200k/799,802frames/18,715updates in26m41s,6.56% less time than H5.
Greedy13,823.958/sample11,277.083 trail14,339.583/13,761.458;all trained arms
24/24 first pyramids but miss15k. Retain H5 for stronger absolute primary score;
H3 is a cheaper development point with slightly higher greedy return/minute.
All506 inputs,357 matching/invariant tensors,retained100k recovery,paired initial
traces,full replays/videos,guards,curves,coverage/exposure and original collector
pass. Main/post/collection succeed14:53:25/14:53:42/14:53:44 UTC November1/source
df3a75e;all five original services terminal,zero original/GPU failures or repeats.
Root`/x/Temp/kindle-qbert-h3-r24-20261101/full`. Do not rerun;no extension,horizon
sweep,confirmation or automatic successor. No GPU allocation remains active.

The [bounded H3 check](results/2026-11-01-cdp-qbert-h3-r24.md) completes8,192
acts/32,760frames/733updates in64.175s;1,024 frozen/.806s haszero updates/no
natural episodes. Conservative200k28m43s passes35min. All475 pins,357 matching/
invariant tensors,47 events,full guided replay/guards pass;main14:08:14/collector
14:08:26 UTC/source8dad414,no failure/repeat. Full CPU qualification3.225s covers
five CLI paths,24 config/eight mode/14 header/four observer/ten recovery cases,
six retention/two integration cases,50 paths and17 scripts. Plot import.044s
passes. The earlier preparer fails only an exact wording replacement before any
full output;its source/error remain. No native build or GPU repeats. Do not
repeat passed qualification or reuse bounded experience.

The [completed Qbert H5/R32 greedy comparison](results/2026-11-01-cdp-qbert-h5-r32-greedy.md)
finishes200k actions/799,822frames/24,954updates in34m59.741s,+22.53% versus R24.
Greedy14,057.292/23 first pyramids and sampled10,305.208/18 fall1.97%/25.12%
below controls14,339.583/24 and13,761.458/24. Actual initials1.042/171.875 match;
original15k/90% gate still fails. Retain R24;no extension,ratio retry or confirmation.
All490 pins,357 matching initial/invariant frozen tensors,1,238/916 matching
initial semantic events,100k/12,454 retained recovery,full pixel/action/label and
frozen replays/videos,guards,curves,coverage,exposure and comparison pass. All
five original services succeed:main12:58:49,post12:58:56,collection12:59:27 UTC
November1/source4a10fcc. Actual overrides20.308%;guided scores are not competence.
CPU co-host timings and earlier CPU preparer failures remain. No full/GPU failure
or repeat. Root`/x/Temp/kindle-qbert-h5-r32-greedy-20261101/full`;complete,do not
rerun stages. The five-game goal remains unfinished.

The [completed Breakout400k/R8 comparison](results/2026-11-01-cdp-breakout-r8.md)
finishes400k actions/1,599,156frames/12,489updates in24m03.031s,+12.06% versus
200k/R16. Sampled56.708 versus56.292 rises.74%;greedy97.583 versus104.667 falls
6.77%;all0/24 two-wall. Own initials1.417/2.208 match. Retain the cheaper200k/R16;
no extension,ratio retry or automatic successor. All315 pins,357 matching/invariant
tensors,800/795 initial semantic events,three audited/retained four-file recoveries,
full guided/frozen replays/videos,guards,curves,exposure and comparison pass.
Recoveries100k/3,114,200k/6,239,300k/9,364 complete13:20:35/13:26:36/13:33:07.
Main/post/collection succeed13:44:22/13:44:36/13:45:11 UTC November1/source566a2a5;
all five original services terminal,zero full/GPU failures or repeats. Actual
overrides176,191 (44.04775%) are guidance,not frozen competence. Learning time stays
near19min;acting/environment costs roughly double. Experience,cadence,sampling,
warmup and exposure change together on reused1009;no isolated coverage claim.
CPU co-host caveats and earlier preparation failures remain. Root
`/x/Temp/kindle-breakout-r8-20261101/full`;do not rerun learner,collectors or checks.
No GPU allocation remains at closure;the five-game goal is unfinished.

The [completed CPU unseen-ball centering reference](results/2026-11-01-cdp-breakout-unseen-center.md)
improves185.375→333.000 on eight screening episodes,then225.083→269.833 (+19.88%)
on the expanded24,five of eight lane means better;allzero two-wall. The original
eight reproduce;the additional16 average244.938→238.250,so the overall gain is
concentrated in the selected screen. This passes the development screen without
establishing independent improvement. Keep the native teacher;no integration,
learner,parameter retry or automatic successor selected. Exact controller bytes,
31 declared/315 completed learner inputs and full pixel/action/debug replay/video
pass. Expanded50,832actions/203,290frames take29.754s plus36.850s replay;68.302s
controller completes13:51:07 UTC November1/sourcebd3ce22,after R8 closure. CPU-only,
zero updates;no gameplay stage fails or repeats. Prior preflight lacks a success
receipt and its tool output is unavailable;exact sources/limitation remain.
Separate3.044s completed-service preflight passes without repeating eight fixtures.
Root`/x/Temp/kindle-breakout-unseen-center-20261101/standard24`;complete,do not rerun.

The [completed Qbert H5/R24 temperature assessment](results/2026-11-01-cdp-qbert-h5-r24-temperature.md)
scores12,840.625/24 first pyramids at0.5,10.45% below retained greedy14,339.583;
default sampled13,761.458/24 reproduces exactly. Own initial sampled/temperature
171.875/203.125 clearzero pyramids. Retain greedy;original15k/90% gate still fails,
original28m34s training stays. All558 pins,357 invariant tensors per arm,6,531/
1,237 matching default semantic events,full replays/videos and guards pass.
Four arms125,944actions/503,626frames/105.190s,zero updates/truncations. Qualified
e416696/runnerabc2c2b and CPU preflight are reused without build or repetition.
Main/collector succeed13:59:10/13:59:17 UTC November1/source1308ffb,original
PID801574/invocation6ce4cfed31e44e73aef685bb38ed64b7;both services terminal,no
failure or repeated stage. No sweep,retry,extension or automatic successor.
Root`/x/Temp/kindle-qbert-h5-r24-temperature-20261101`;complete,do not rerun.
No GPU allocation is active at closure;the five-game goal remains unfinished.

The [fixed Breakout upper-crop reference](results/2026-11-01-cdp-breakout-crop70.md)
leaves all eight returns unchanged:mean185.375,zero two-wall successes. The
10%/five-lane screen fails;retain the teacher,no crop retry or24-episode expansion.
All16,280 actions/65,107 frames replay exactly;26 declared/490 active Qbert pins
and original process identity pass. Seven fixtures,2.499s preflight,10.639s
collection/13.219s replay/26.484s controller;complete12:41:57 UTC November1,
source7b12a46. Separate.139s log comparison finds identical actions/pixel hashes/
rewards/boundaries and zero accepted detections above y90;96 debug rows differ.
No native/GPU work,updates,failures or repeated stages. Keep co-host timing;
root`/x/Temp/kindle-breakout-crop70-20261101`. Do not rerun.

The [completed Breakout pooling diagnostic](results/2026-11-01-cdp-breakout-pool-timing.md) finds35,193/35,193 accepted
pooled paddle centers exactly equal the last-two-raw-frame midpoint;latest-raw
MAE1.456 over35,610 decisions. Shared model errors remain1.965 pooled versus
2.875 latest-raw MAE (31,817 rows);near1.860/2.706. Raw targets change calibrated
semantics;no correction,teacher change or gameplay check selected. All52,072
actions/208,254 saved frames align;37,448 original pooled errors reproduce,all
exclusions remain. Six CPU fixtures,16 inputs/490 active Qbert pins and original
PID792387/invocation460e0220583948f4b088b5dc7c92956a pass. Analysis1.516s/controller
4.475s completes12:18:04 UTC November1/source8c45350. No replay,native/GPU work,
updates,failures or repeats;retain co-host time. Root
`/x/Temp/kindle-breakout-pool-timing-20261101`;complete,do not rerun.

The [completed six-feature Breakout response-history fit](results/2026-11-01-cdp-breakout-response-history.md)
fails its fixed screen:held-out MAE1.976923→1.997426 overall and1.948802→1.958316
near descending balls,despite lower RMSE. Six of eight lanes worsen MAE;recurrence
radius.165412 is stable. Retain current teacher;no gameplay check,integration or
learner follows. All37,448 original rows/splits and46,122 predictions reproduce;
one F64 fit uses19,146 training interiors and13,102 held-out rows. Seven CPU tests,
.138s preflight and56 input pins pass. Analysis.853s/controller1.087s completes
11:55:03 UTC November1/source885f9d1. No active learner,native/emulator/GPU work,
updates,failures or repeated fit. Root`/x/Temp/kindle-breakout-response-history-20261101`;
complete,do not rerun. No alternate fit,sweep or automatic successor selected.

The [completed centroid100/encoder4e-5 comparison](results/2026-11-01-cdp-centroid-encoder4e5.md)
finishes200k/799,590frames/12,477updates in21m47s,+1.50% observed training time.
Sampled35.542 versus56.292 falls36.86%;greedy112.750 versus104.667 rises7.72%.
Own initials1.417/2.208 match;all0/24 two-wall. Retain no-centroid control;sampled
remains primary,no overall win or gate success. Late centroid loss.044301→.000645
across first/last50k windows is training fit,not held-out accuracy. Expert CE
.559106 versus.558097 is nearly unchanged;actual overrides43.6295% are not
frozen competence. Native2974160/runner differ from control;no isolated cause.
All541 pins,372 matching/invariant tensors,357 shared values,47 pre-update and
800/795 initial semantic events,100k/6227 retained recovery,full training/frozen
replays/videos,guards,curves,exposure and comparison pass. Main/post/collection
finish11:41:19/11:41:26/11:42:02 UTC November1/sourcecd92f9f;all five original
services succeed,zero original/GPU failures or repeats. Separate terminal CPU
verifier failure on missing progress Consumed summary is preserved;1.553s
corrected confirmation passes11:43:44 without repeating any original stage.
Root`/x/Temp/kindle-centroid-encoder4e5-20261101/full`;terminal,do not rerun learner,
collectors or confirmations. CPU co-host caveats remain. No active allocation,
extension,coefficient sweep,cap increase,confirmation or automatic successor.
Check live state before work;the five-game goal remains unfinished.

The [completed CPU Qbert linear readout](results/2026-11-01-cdp-qbert-failure-linear-readout.md)
trails the original actor:held-out movement agreement41.02% versus52.34%,rescued
movement27.92% versus38.07%,overall66.70% versus86.33%. The actor wins all eight
held-out lanes;no replacement-head benefit or feature-information absence is
established. Fixed ridge1,training-fold normalization,all1024 older4013 failure
rows/eight folds;not the new H5/R24 representations. Three numerical fixtures,
finite fits/residuals,disjoint folds and11 hashes pass. Fits.357s/analysis.425s/
wrapper.514s complete11:32:26 UTC November1/source8d539db;original Breakout
PID787678/invocation709367befab94110b4afc489bd51d55c stays. Zero native/emulator/
GPU work,updates,failures or repeated fits;retain CPU co-host time. Root
`/x/Temp/kindle-qbert-failure-linear-readout-20261101`;complete,do not rerun.
No alternate head,sweep,loss,learner or automatic successor selected.

The [completed CPU Qbert phase diagnostic](results/2026-11-01-cdp-qbert-h5-r24-phases.md)
finds both8039/10007 greedy policies clear the first pyramid at exactly1219frames
with550 prefix points inall24 episodes. The full1090.625 mean gap occurs after
the crossing action:later actions1958.542→1899.542 and reward/100actions
759.760→725.943. Sampled10007 improves with shorter later duration/higher rate;
no single causal mechanism or paired seed effect is established. All96 fixed
selected episodes reconcile; six boundary fixtures and23 input pins pass.
Analysis.258s/wrapper.314s completes11:20:53 UTC November1/source9da3385; original
BreakoutPID787678/invocation709367befab94110b4afc489bd51d55c remains unchanged.
No emulator/native/GPU work,updates,failures or repeats. Retain CPU co-host time.
Root`/x/Temp/kindle-qbert-h5-r24-phases-20261101`;complete,do not rerun.
No recipe,confirmation learner or automatic successor is selected.

The [completed Qbert H5/R24 seed10007 confirmation](results/2026-11-01-cdp-qbert-h5-r24-seed10007.md)
finishes200k/799,841frames/18,715updates in28m34s. Greedy14,339.583/
24 first pyramids andsampled13,761.458/24 both miss15k;
own initials1.042/171.875 clear none. Identical recipe8039 passes,
10007 fails:one of two primary-greedy learners,not broad reliability. Same12M48/
Auto/N8/B16/T16/H5/R24/encoder4e-5/greedy collection/expert100/native447fe50;
only learner/cohort seeds change,no paired H15/R16 control or prior experience.
All442 pins,100k/9340 recovery,357 finite/invariant tensors,48 initial differences,
full training/frozen replays/videos,guards,curves,coverage,exposure and comparison
pass. Main/post/collection finish10:54:21/10:54:30/10:55:23 UTC November1/source0ebac01;
independent2.210s confirmation passes10:55:36. All six services succeed,
zero failures or repeated stages. Actual overrides23.303%,not frozen
competence. Retain CPU co-host timing. Root
`/x/Temp/kindle-qbert-h5-r24-seed10007-20261101/full`;terminal,do not rerun.
No extension,cap increase,third learner or automatic successor selected.
No active learner/GPU allocation at closure;check live state before work.
The five-game goal remains unfinished.

The [completed centroid100/encoder4e-5 bounded check](results/2026-11-01-cdp-centroid-encoder4e5.md)
passes8,192actions/32,722frames/489updates in47.291s;1024 frozen/.883s haszero
updates and no natural episodes. All302 pins,372 matching centroid initial and
invariant frozen tensors,357 shared control values,47 pre-update events,finite
reports,changed CNN/reward/actor/value/centroid weights,full replay and guards pass.
Main succeeds11:03:56 UTC November1/source77866fc;collector11:04:29,zero failures
or repeats. Conservative200k projection21m10s passes30min;competence unmeasured.
Native2974160/exact runner unchanged;no build. Root
`/x/Temp/kindle-centroid-encoder4e5-20261101/native-check`;complete,do not rerun.
The [separate fresh200k comparison](experiments/2026-11-01-cdp-centroid-encoder4e5-200k.md)
compares against retained4e-5 sampled56.292/greedy104.667,all0/24 two-wall. Full CPU
qualification passes11:14:22 UTC:five commands,18 config cases,seven startup
cases,four observer argument cases,five recovery cases,372 schema tensors,357
shared values and exact schedules in2.610s. Plot/import/exposure qualification
passes in.698s;all201 historical source hashes resolve. Before qualification,
stale collector service names were corrected and100k checkpoint retention added.
No native/emulator construction,build,failure or repeated check. The separate
fresh200k comparison is now complete;see the full result above. Do not rerun
passed qualification or the completed bounded check.
No retry,extension,cap increase or automatic successor.

The [completed CPU Breakout settling-cost check](results/2026-11-01-cdp-breakout-edge-settling.md) lowers the retained
edge-reset teacher185.375→80.375 on eight reused episodes,withonly2/8 lanes
improving andzero two-wall successes. Retain the existing teacher;the fixed
10%-mean/five-lane screen fails. No standard24 expansion,coefficient retry,
native change or learner follows. Collection12,880actions/51,509frames takes
24.952s;exact pixel/action/debug/outcome replay10.409s,controller40.337s completes
10:47:56 UTC November1/sourced105cee. Four cost fixtures,5.174s preflight,all24
inputs/442 active Qbert pins and scalar recount pass;original Qbert identity stays.
Zero native/GPU work,updates,failures or repeats;retain CPU co-host time. Root
`/x/Temp/kindle-breakout-edge-settling-20261101`;complete,do not rerun.

The [completed CPU Breakout response-ambiguity diagnostic](results/2026-11-01-cdp-breakout-response-ambiguity.md) finds4,597 of
13,102 held-out rows in exact-input groups with different next paddle positions;
near-descending514/2955,with2.512px RMS within-group pair differences. This shows
ambiguity in the teacher's four logged inputs,not irreducible noise,a learned-state
information limit or a causal mechanism. Interior states also vary;pair weighting,
sticky controls,pooling and measurement remain caveats.2.433s analysis/2.518s
controller completes10:42:08 UTC November1/source20b1208;four fixed fixtures,
seven diagnostic/442 active Qbert pins and.083s independent histogram recount pass.
Original Qbert identity unchanged;retain CPU co-host timing. Zero emulator/native/
GPU work,updates,failures or repeats. No teacher,loss,learner or successor selected.
Root`/x/Temp/kindle-breakout-response-ambiguity-20261101`;complete,do not rerun.

The [completed CPU Breakout response refit](results/2026-11-01-cdp-breakout-response-refit.md) fails its fixed screen:
held-out MAE rises1.976923→2.004710 overall and1.948802→1.977556 near descending
balls,despite lower RMSE. All eight lanes show that tradeoff;retain the existing
teacher. One four-feature F64 fit uses19,146 interior rows from episodes0/1;
13,102 episode2 rows stay held out. Fit.596s/controller2.880s completes10:34:29 UTC
November1/source2b54693;all44 diagnostic/442 active Qbert pins and original
PID781585/invocation539c22282c4e42c7a22bc150961ef963 remain unchanged. Eight CPU
tests,2.192s preflight and.186s independent scalar recount pass. No emulator,
native/GPU work,updates,failures or repeats. Retain CPU co-host timing. No
alternate fit,gameplay test,teacher change or automatic successor is selected.
Root`/x/Temp/kindle-breakout-response-refit-20261101`;complete,do not rerun.

The [completed CPU Breakout teacher contact diagnostic](results/2026-11-01-cdp-breakout-teacher-contact-errors.md)
records120 selected life losses;65 pass fixed geometry filters. Mean absolute
landing/paddle errors1.852/4.943 pixels versus.370/2.317 on674 filtered successful
contacts. In38 fitted losses,model offset is inside half the detected width but
reconstructed offset is outside. Retain55 rejected losses/102 rejected success
contacts;no unbiased failure probability or causal diagnosis. Prioritize response
prediction/uncertainty checks;no teacher,loss or learner change selected.
Full52,072-action/208,254-frame policy/debug/pixel/outcome and raw RGB-hash replay
passes.44.731s replay/1.420s analysis/48.646s controller completes10:25:39 UTC
November1/sourcef658f2a;16 CPU tests,34 diagnostic/442 active pins and.005s
independent recount pass. Original QbertPID781585/invocation539c22282c4e42c7a22bc150961ef963
stays unchanged;retain CPU co-host time. Zero native/GPU work,updates,failures or
repeated original stages. Root`/x/Temp/kindle-breakout-teacher-contact-errors-20261101`;
complete,do not rerun replay/analysis/qualification. No automatic successor.


The [completed Breakout H5/R24 comparison](results/2026-11-01-cdp-breakout-h5-r24.md)
finishes200k actions/799,605 frames/18,715 updates in22m50s,+6.42% training
time. Sampled42.333/greedy76.083 fall24.80%/27.31% below56.292/104.667;all0/24
two-wall. Own initials1.417/2.208 match. Late replay expert CE.558097→.380887
improves without gameplay gain;retain H15/R16. H15/R16→H5/R24 changes horizon,
cadence,sampling,warmup and work together on reused1009;no isolated cause or
reliability. All301 pins,100k/9340 recovery,357 matching/invariant tensors,paired
initial traces,full replays/videos,guards,curves,exposure and comparison pass.
Main/post/collection finish10:08:37/10:08:46/10:09:18 UTC November1;
independent2.171s confirmation passes10:09:38. All six services succeed,
zero failures or repeated stages. Sourcec4fa576,originalPID777532/invocation
acf63b2de9614a9dab625bcc6f6cbe52;actual overrides43.471%,not frozen competence.
Root`/x/Temp/kindle-breakout-h5-r24-20261101/full`;terminal,do not rerun learner,
collectors or confirmations. No extension,cap increase or automatic successor.
Retain CPU co-host timing. No GPU allocation at closure;check live state.
The five-game goal remains unfinished.

The [completed frozen Breakout temperature0.5 assessment](results/2026-11-01-cdp-breakout-temperature.md)
scores83.292 versus retained greedy104.667 (−20.42%) and default sampled56.292
(+47.96%);all0/24 two-wall successes. Own temperature initial1.333;default1.417
reproduces. Retain greedy for this checkpoint;original sampled-primary result
stays unchanged. Both default semantic sequences match exactly (4394/799 events).
All348 pins,357 finite/invariant tensors per arm,full replays/videos,guards and
original identity pass. Four arms81,016 actions/323,877 frames/72.156s,zero updates
or truncations;original21m28s training remains. Main/collector succeed09:27:46/
09:28:08 UTC November1/sourcea285376;zero failures or repeated stages. Qualified
e416696/runtime40872a3/runnerabc2c2b,actor-unimix.01 preserved. Root
`/x/Temp/kindle-breakout-temperature-20261101`;terminal,do not rerun arms,
preflight or collector. No new learner,sweep,cap increase or automatic successor.
No allocation active at closure;check live state. The five-game goal is unfinished.

The [completed Qbert H5/R24 comparison](results/2026-11-01-cdp-qbert-h5-r24.md)
finishes 200k actions / 799,812 frames / 18,715 updates in 28m47s.
Greedy 15,430.208 / 24 first pyramids passes the 15k/90% gate,
up 35.64% from 11,376.042 / 23; sampled 13,040.625 / 24 rises
57.25% from 8,292.708 / 24 but still misses 15k. Training takes 4.44% longer.
Own initials 251.042 / 128.125 match and clear none. Retain H5/R24 as a passing
8039 development point, not seed reliability or an isolated horizon effect.
H15/R16→H5/R24 changes cadence, sampling, warmup and work together; native447fe50,
runneraba4300d and teacher stay fixed. All 401 pins, 100k/9340 recovery, 357 matched
initial/invariant frozen tensors, paired initial traces, full training pixel/action/
label and frozen replays/videos, guards, curves, coverage, exposure and comparison
pass. Main/post/collection finish 09:23:12/09:23:20/09:23:21 UTC November1,
source64c2e89; independent 1.397s terminal confirmation passes 09:23:42.
All six services succeed; no failures or repeated original/GPU stages. Actual
training overrides 27.944% and aided online returns are not frozen
competence. CPU co-host timing remains without correction. Root
`/x/Temp/kindle-qbert-h5-r24-20261101/full`; terminal, do not rerun its learner,
collectors or confirmations. No extension, cap increase, confirmation learner or
automatic successor is selected. Check live state; the five-game goal is unfinished.

The [completed CPU Breakout planning-cost diagnostic](results/2026-11-01-cdp-breakout-plan-costs.md) finds56.38% of
12,521 horizontal disagreements have at most1 unit excess teacher-model cost;
near-descending median1.483,with17.64% above16. Three of six teacher-only surviving
interventions also have gaps below1,so model cost alone does not justify softened
or dropped labels. All34,429 planning rows reproduce exact plans/paths/costs/
labels;331 analysis/401 active pins and original Qbert identity pass.4.606s CPU/
6.342s controller completes08:50:43 UTC November1/source7d1ea69. Eight scalar
fixtures and independent JSON recount pass;zero emulator/native/GPU work,
updates,failures or retries. Root`/x/Temp/kindle-breakout-plan-costs-20261101`;
complete,do not rerun. Retain CPU co-host timing;no loss,teacher change or
successor selected. The Qbert H5/R24 learner has since completed.

The completed bounded check retains390 pins,357 matching initial/invariant
frozen tensors,47 matching pre-update events,733 finite reports,full replay and
guards.1024 frozen/.809s haszero updates/no natural episodes. Main succeeds
08:39:25 UTC November1/sourcea2b8387,collector08:39:58;zero failures/repeats.
Do not rerun bounded stages or collection;the full learner reuses no experience.

The [completed frozen Qbert cross-cohort diagnostic](results/2026-11-01-cdp-qbert-cross-cohort.md) keeps4013
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

The [completed Breakout greedy-collection comparison](results/2026-11-01-cdp-breakout-greedy-training.md) finishes200k/
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

The [completed bounded Breakout check](results/2026-11-01-cdp-breakout-greedy-training.md) passes8192 actions/32,740 frames/
489 updates in46.633s;1024 frozen/4076 frames/.874s haszero updates and eight
zero-return natural episodes. All252 pins,357 matching initial/invariant frozen
tensors,47 expected initial semantic differences,full replay and guards pass.
Conservative200k projection20m52s passes30min;no competence inferred. Original
service succeeds07:51:36 UTC November1/sourceccb4387,collection07:51:45. No
original stage fails or repeats. Root
`/x/Temp/kindle-breakout-greedy-training-20261101/native-check`;terminal at closure,
do not rerun. The full learner is a separate selection and uses no check experience.

The [completed CPU delayed Breakout takeover](results/2026-11-01-cdp-breakout-delayed-takeover.md) records survival
108/108/110/106/105/91 of120 at delays0/1/2/4/8/16 actions from first disagreement.
Most early sequences remain recoverable under the teacher;later takeover reduces
aggregate rescue,with nonmonotonic cases. No architectural cause or game score.
All478 new branches,37,016 source actions/148,038 frames,suffixes,eight reverse
checks,334 diagnostic/359 active Qbert pins and21 CPU tests pass.55.638s analysis /
57.311s controller closes07:12:23 UTC November1/sourcec9fbf9f.240 retained baseline
branches do not repeat;zero native/GPU work,updates,failures or retries. Original
QbertPID760870/invocationa0b219d7702b4c82a65e9b33498c3544 remains unchanged;
retain CPU timing. Root`/x/Temp/kindle-breakout-delayed-takeover-20261101`;
complete,do not rerun. No new learner,loss or successor is selected.

The [completed Qbert seed8039 check](results/2026-11-01-cdp-qbert-greedy-seed8039.md) finishes200k/799,796 frames/
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

The [completed CPU Breakout first-disagreement diagnostic](results/2026-11-01-cdp-breakout-first-disagreement.md)
finds102 both/six teacher-only/six policy-only/six neither survive at the original
loss endpoint,both arms108/120. First disagreements usually remain recoverable
under subsequent teacher control;no aggregate first-command survival benefit
or architecture cause. Analysis39.271s completes06:24:01 UTC November1/source
d139b43;all37,016 actions,6,905 old teacher-suffix rows,eight reverse checks,
320 repair/317 active Qbert pins and22 tests pass. Original pre-replay origin-order
failure is preserved;separateCPU closure`complete_with_cpu_order_repair`,zero
completed replay/GPU repeats. Qbert identity was preserved; no new learner/loss followed this diagnostic.

The [completed greedy-training Qbert comparison](results/2026-11-01-cdp-qbert-greedy-training.md)
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
The completed1.017s CPU coverage recount confirms later wholly unguided actions
58,487→81,530 and age1920+159→5,525,with332 diagnostic/359 active pins unchanged
at07:21:44 UTC/source234ca3a. This supports the coverage hypothesis without proving
causal mediation;all actions already have labels. See the result's coverage section.
No native/emulator/GPU work or new learner;retain CPU co-host timing.

The [completed greedy-training Qbert check](results/2026-11-01-cdp-qbert-greedy-training.md)
finishes8192 actions/32,761 frames/489 updates in66.241s;1024 frozen/.798s haszero
updates and no natural episodes. Conservative200k projection29m39s passes35min.
Only training collection changes sampled→greedy;native447fe50,teacher,losses and
12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 remain. Explicit fresh-training
CLI mode/default compatibility pass637 source/34 private CPU checks. All305 pins,
357 matching initial/invariant frozen tensors,47 expected initial trace differences,
full guided replay and guards pass. Main succeeds06:05:29 UTC November1/sourcec806de5,
original collector06:06:03. Supplemental silent-parent journal assertion fails;
retained.638s CPU correction closes06:06:50,zero original/GPU repeats. Test-path/
old-audit CPU failures also remain preserved. Root
`/x/Temp/kindle-qbert-greedy-training-20261101/native-check`. No active allocation,
full learner or automatic promotion;execution/cost only. Five-game goal unfinished.

The [completed frozen Qbert training readout](results/2026-11-01-cdp-qbert-training-readout.md)
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

The [completed first-disagreement Qbert intervention](results/2026-11-01-cdp-qbert-first-disagreement.md)
finds five of 25 first-command differences change survival through the original
loss endpoint: teacher-first survives 25, policy-first 20, with both then using
the same feedback-aware teacher. Four differences are movement versus movement;
one is movement versus NOOP. At the separate 64-action endpoint, paired outcomes
are 16 both, seven teacher-only and two neither. This identifies local action
effects under this controller, not an architectural cause or new game score.
All 16,613 source actions / 66,440 frames, 1,373 retained teacher suffix rows,
eight reverse-order checks, 11 tests and 552 input hashes pass. CPU analysis
65.045s / controller 65.779s completes 05:44:58 UTC November 1, source8a8b433.
No native/GPU work, learning, failures or retries; no new learner/loss or automatic
successor. Root `/x/Temp/kindle-qbert-first-disagreement-20261101`. The five-game
goal remains unfinished; check live state before new work.

The [completed Qbert vector-policy readout](results/2026-11-01-cdp-qbert-policy-readback.md)
reproduces20,584 original actions/82,324 frames in17.631s,zero updates. In1,024
failure-window decisions,teacher agreement is86.33% overall but52.34% for moves
and97.66% for NOOP;rescued-window movement agreement is38.07%. Of122 movement
disagreements,34 chooseNOOP and88 another action. Origins already agree31/32;
no single-action causal or architecture claim. All525 pins,357 invariant tensors,
128 F64 actor matches,exact smoke/full prefix,three guards and original service
pass05:31:49 UTC November1/sourcefd1c586. A null journal-message field fails the
separate CPU collector;original failure retained,.661s CPU repair completes
05:34:08,zero GPU/capture repeats. Additive native readout447fe50+patch/SHAf864fe4
passes130 Rust/585 Python CPU checks/Clippy and guarded state/RNG checks. Original
context-only patch failure is retained;857MB cache released,wheels/source/test
binary/runtime retained. Roots`/x/Temp/kindle-vector-policy-readback-20261101`
and`/x/Temp/kindle-qbert-policy-readback-20261101`. No learner/GPU allocation,
new loss or automatic successor;check live state. Five-game goal unfinished.

The [completed Breakout encoder8e-5 comparison](results/2026-11-01-cdp-breakout-encoder8e5.md)
finishes200k actions/799,597 frames/12,477 updates in21m24s. Sampled55.875 falls
.74% versus56.292;greedy125.375 improves19.79% versus104.667. All0/24 two-wall;
own initials1.417/2.208 match. Retain4e-5 as sampled development point and8e-5 as
separate greedy option;sampled remains primary,no overall win or reliability.
All265 pins,100k/6227 recovery,357 matching/invariant tensors,47 pre-update and
800/795 initial semantic events,replays/videos,guards,curves,exposure and comparison
pass. Main/post/collection succeed05:14:41/05:15:03/05:15:06 UTC November1;
all five original services succeed,zero failures/repeats. Separate.872s terminal
confirmation passes05:15:35. Source3897e74;only encoder4e-5→8e-5 changes in
12M64/Auto/N8/B16/T16/H15/R16/expert100,native447fe50/exact runner/teacher fixed.
Actual guidance44.4695%;late CE.5252 versus.5581 is not frozen competence.
Reused1009/cohort and CPU timing caveats remain. Root
`/x/Temp/kindle-breakout-encoder8e5-20261101/full`;no extension,rate queue,
confirmation or automatic successor. No learner/GPU allocation remains active
at closure;check live state. The five-game goal remains unfinished.

The [completed CPU Qbert teacher takeovers](results/2026-11-01-cdp-qbert-recoverability.md)
avoid25/32 selected recorded losses at their original endpoint;23 survive the
separate64-action follow-up. First episode per8 lanes,all32 losses;origins begin
up to32 actions earlier within the same life. This establishes useful teacher
continuations on many selected learner states,not an architectural cause or
new game score. All16,613 source actions/66,440 frames,recorded branches,eight
reverse-order checks,eight tests,309 diagnostic/265 protected Breakout pins and
original terminal identity pass. CPU63.910s/controller65.230s completes05:16:04
UTC November1/source1a4fffe,after Breakout training/main completion. No native/GPU
work,updates,failures or retries. Root`/x/Temp/kindle-qbert-recoverability-20261101`;
no new learner/loss or automatic successor. Preserve CPU overlap timing.

The [completed CPU Qbert life/color replay](results/2026-11-01-cdp-qbert-life-replay.md)
finds all96 losses per retained2e-5/4e-5 policy occur after the first pyramid,
withzero life gains in each fixed24-episode cohort. Candidate selected duration
rises21.13%,score1.76%,and reward/action falls15.99%. Losses at episode age1920+
rise5→33 of96;this is descriptive,not matched training coverage or a causal
cap effect. Complete44,512/60,648-action replays reproduce all recorded outcomes.
Five CPU tests,293 diagnostic/265 active Breakout pins and original process
identity pass. Analysis49.272s/controller50.619s completes05:07:23 UTC November1,
sourceaa80855,withzero native/GPU work or updates. Raw cube-color histories are
retained without a general level detector. Root
`/x/Temp/kindle-qbert-life-replay-20261101`;no new learner/loss or automatic
successor. Retain CPU co-host timing caveats;the original Breakout run continues.

The [completed Qbert encoder4e-5 comparison](results/2026-11-01-cdp-qbert-encoder4e5.md)
finishes200k actions/799,739 frames/12,477 updates in27m41s,8.56% less observed
training time than2e-5. Greedy14,028.125/sample11,522.917 improve1.76%/15.37%,
each24/24 first pyramids;own initials80.208/177.083 clear none. Both means still
miss15,000. Retain4e-5 as a development point,not independent reliability.
All276 pins,100k/6227 recovery,357 matching/invariant tensors,47 pre-update and
1139/1101 initial events,full replays/videos,guards,curves,coverage,exposure and
comparison pass. Main/post/collection succeed04:38:49/04:38:58/04:39:00 UTC
November1,source7f794b0;all five original services terminal success,zero repeats.
Separate.597s terminal confirmation corrects an overly strict watcher-PID check:
original collector output comes from a child in the same invocation. Preserve
that ancillary failure;the original pipeline passes unchanged. Actual overrides
38.4%;late replay CE.0893 versus.0814 and guided scores are not frozen competence.
Native447fe50,exact runner and teacher fixed;only encoder2e-5→4e-5 changes in
12M48/Auto/N8/B16/T16/H15/R16/expert100. Reused4013/cohort and CPU co-host caveats
remain. Root`/x/Temp/kindle-qbert-encoder4e5-20261101/full`. No Qbert extension,
confirmation or automatic successor;no active GPU allocation at closure.
Check live state before work. The five-game goal remains unfinished.

The [completed Breakout encoder8e-5 check](results/2026-11-01-cdp-breakout-encoder8e5.md)
finishes8192 actions/32,725 frames/489 updates in46.730s;1024 frozen actions/
4096 frames/.862s havezero updates and no natural episodes. All251 pins,357
matching initial/invariant frozen tensors,47 pre-update events,finite reports,
full input-pixel/action/label replay and both host guards pass. Conservative200k
projection20m55s passes30min. Main succeeds04:44:13 UTC November1/source5e71622,
original collection04:44:36;zero failures/GPU repeats. Only encoder4e-5→8e-5
changes,native447fe50,exact runner,teacher and other12M64 settings fixed.
Root`/x/Temp/kindle-breakout-encoder8e5-20261101/native-check`. Execution/cost
are qualified;no gameplay benefit,full learner or automatic promotion selected.
No GPU allocation remains active at closure;check live state before work.

The [completed CPU Breakout teacher-takeover diagnostic](results/2026-11-01-cdp-breakout-recoverability.md) avoids108/120
recorded losses from the early32-action-window origin versus26/120 from the
last near-descending state,measured at the original loss endpoint. Paired23 both/
85 early-only/three late-only/nine neither;separate64-action survival97/22.
This supports investigating earlier control,not an exact causal error or optimal
irrecoverability. All240 origins,37,016 source actions/148,038 frames,source/
branch replays,eight reverse-order checks,11 tests and293 analysis/276 active
Qbert pins pass. CPU36.589s completes04:14:23 UTC November1/source10846aa,
same QbertPID/invocation;zero native model/GPU work,updates or failures.
Root`/x/Temp/kindle-breakout-recoverability-20261101`;co-host timing retained,
no new learner/loss/successor selected. The original Qbert run continues.

The [completed CPU Breakout own-policy replay](results/2026-11-01-cdp-encoder4e5-representation.md#own-policy-trajectories-improve-productivity-and-show-a-smaller-agreement-gain)
reproduces37,016 actions/148,038 frames,with31,161 selected actions. Versus2e-5,
episodes last25.36% longer and earn95.88% more reward/action;near-descending
teacher agreement57.59%→59.86%,opposite-direction fractions fall,NOOP rises.
All120 observed lost lives have a recent near-descending detection;86 last
such decisions match the teacher. No causal last-action error or counterfactual
outcome is identified. Eight tests,265 analysis/276 active Qbert pins and
original process identity pass. CPU23.375s completes04:04:01 UTC November1/
source3012999;zero native/GPU work,updates,failures or replay repeats. Root
`/x/Temp/kindle-encoder4e5-policy-phase-20261101`;co-host timing caveats retained,
no learner/loss/successor selected. The original Qbert learner continues.

The [completed Qbert encoder4e-5 check](results/2026-11-01-cdp-qbert-encoder4e5.md) finishes8192 actions/32,754 frames/489
updates in64.165s;1024 frozen/.800s haszero updates/no natural episodes.
All257 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
finite reports,full replay and both guards pass. Conservative200k projection
28m43s passes35min. Main succeeds03:55:47 UTC November1/sourcee2887ef,
collection03:56:06;zero failures/GPU repeats. Native447fe50,runner and teacher
fixed;this establishes execution/cost,not gameplay benefit. Root
`/x/Temp/kindle-qbert-encoder4e5-20261101/native-check`. No allocation remains
active at bounded closure;the full learner is selected separately above.

The [completed encoder4e-5 representation diagnostic](results/2026-11-01-cdp-encoder4e5-representation.md) finds more precise
ball and motion readouts on the exact20,960-action/83,799-frame control corpus.
Combined-kernel ball x/y MAE7.554/6.515→6.028/4.192 on555 shared test rows;
pixel.059/.140 remains much more precise. Near-descending teacher agreement
rises62.50%→66.91%,but teacher probability falls.582→.573 and opposite-direction
choices rise7→11 of89 directional labels. All six replacement readouts trail
the learned actor;no single causal mechanism or own-policy coverage is measured.
All473 pins,357 invariant tensors per stage,128 native actor choices,16 prior
feature rows,full replay,matched splits,predictions and three host guards pass.
Stage times3.295/2.922/21.858s,CPU analyses3.333/1.539/.245s;service succeeds
03:44:02 UTC November1/sourcec17d723 after42.143s,collection03:44:41. The83 CPU
checks pass;native447fe50 and actor math stay fixed. Prior6e-6/2e-5 captures/fits
are reused,zero updates/failures/GPU repeats. Root/service
`kindle-encoder4e5-representation-20261101` under`/x/Temp`. No new learner,
loss,head replacement or gameplay evaluation;five-game goal remains unfinished.
No GPU allocation active at closure;check live state before work.

The [completed Breakout encoder4e-5 comparison](results/2026-11-01-cdp-breakout-encoder4e5.md) finishes200k actions/
799,581 frames/12,477 updates in21m28s,4.43% less training time than2e-5 control.
Sampled56.292/greedy104.667 improve25.292/42.625 by122.57%/145.55%;own initials
1.417/2.208 match,all0/24 two-wall. Initial-subtracted return/min rises1.063→2.557
and1.800→4.774. Retain4e-5 for12M64/Auto/N8/B16/T16/H15/R16/expert100;only the
encoder rate changes,native447fe50,exact runner and teacher fixed. All227 pins,
100k/6227 recovery,357 matching/invariant tensors,47 pre-update events,800/795
initial semantic events,full replays/videos,guards,curves,exposure and comparison
pass. Main/post/collection succeed03:32:40/03:32:44/03:32:47 UTC November1,
source2f9af49;all five original services terminal success,zero GPU repeats.
Actual overrides49.6715%,late replay CE.558 versus.591 and online103.80 versus
73.32 are not frozen competence/held-out accuracy. Earlier4M/expert10 rate4e-5
failure,reused1009/cohort and CPU co-host caveats remain;no universal rate or
reliability claim. Original CPU header-rate preparation failure is preserved.
Root`/x/Temp/kindle-breakout-encoder4e5-20261101/full`;no extension,sweep,
confirmation or automatic successor. No GPU allocation active at closure;
check live state before work. The five-game goal remains unfinished.

The current five-game goal includes architecture diagnosis. Prioritize the
remaining Breakout control gap and Qbert later-game decisions using retained
checkpoints and trajectories. Distinguish state representation,action ranking,
value estimates and on-policy coverage before selecting another loss. Existing
imitation and predictive losses already train the model;the completed auxiliary
loss comparisons have not established a better recipe. Compare any new learner
by frozen gameplay gained per training minute,with bounded checks before a full
allocation. Backend optimization and broad sweeps remain parked.

The [completed Breakout imitation-decay comparison](results/2026-11-01-cdp-breakout-expert-decay.md) finishes200k actions/
799,518 frames/12,477 updates in21m30s,4.23% less training time than constant100.
Sampled17.167/greedy25.167 fall32.13%/40.96% below25.292/42.625;own initials
1.417/2.208 match,all0/24 two-wall. Retain constant100;no extension,sweep,
confirmation or automatic successor. All236 pins,100k/6227 recovery,357 matching/
invariant tensors,exact12,477 scheduled reports (3,126 atzero),800/795 initial
semantic events,full replays/videos,guards,curves,exposure and comparison pass.
All first100k recorded semantics also match control before fading;hidden states/
gradients are not compared. Main/post/collection succeed02:52:26/02:52:59/02:53:02
UTC November1/source90b1c7e;all five services terminal success,zero GPU repeats.
Online last50 rises73.32→77.78 despite weaker frozen play;53.315% actual overrides
and late replay CE.824 versus.591 are not competence/held-out accuracy. Native
65ed45f/runner differ from447fe50,reused1009/cohort;no isolated causal/reliability
claim. Earlier CPU preparation failures,auxiliary-summary correction and co-host
timing caveats are preserved. Root`/x/Temp/kindle-expert-decay-20261101/full`.
No active GPU allocation at closure;check live state before work. Goal unfinished.

The [optional native imitation decay](results/2026-11-01-cdp-expert-imitation-decay.md) passes F64 loss/gradient checks at
weights100/50/0 and fresh Breakout8192 actions/32,724 frames/489 updates in46.991s.
Compressed decay128→256 gives234 zero-weight reports; raw expert losses remain
reported. Frozen1024/.902s haszero updates,357 invariant tensors and one natural
zero-return episode. All357 initial tensors,47 pre-update events,full guided replay,
416 pins and host guards pass. Service succeeds02:22:36 UTC November1/source65ed45f;
collection02:22:48,no GPU repeats. The148 Rust/804 packaged Python CPU checks and
strict Clippy pass; initial Python fixture failure remains preserved. Runtime
`/x/Temp/kindle-expert-decay-20261101/runtime`,nativeSHAcac21ef6,default off.
Online teacher guidance/labels and optimizer history stay fixed. Early200k cost
projection21m02s passes30min,but no gameplay benefit/full learner is established.
All services terminal,no active GPU allocation. Root`/x/Temp/kindle-expert-decay-20261101`;
check live state before work. Architecture diagnosis remains in the unfinished goal.

The [Breakout action interventions](results/2026-10-30-cdp-encoder-representation.md#local-action-interventions-limit-the-teacher-disagreement-diagnosis)
limit treating teacher disagreement as error. Of51 improved-policy disagreements,
the teacher saves a life in5 cases,the policy3,and43 tie;among18 confident
ones,the policy saves2,teacher1,and15 tie. Reward favors policy4/teacher1 with13
ties in that confident group. These are one-action interventions with pixel-
teacher continuation,not full-policy quality or optimal-action truth. All136
origins,234 distinct branches,5,240 source actions/20,950 frames,25 pins,32 CPU
tests and reverse-order checks pass. CPU17.081s,complete01:59:51 UTC November1
/source544c51e,collection02:01:35. Initial test-path failure retained;zero GPU,
model-forward or learner work. Root`/x/Temp/kindle-breakout-action-interventions-20261101`.
Service terminal success,no GPU allocation active;no learner/loss selected by
this diagnostic. Architecture diagnosis remains within the unfinished goal.

The [completed Qbert H5/R32 encoder2e-5 comparison](results/2026-10-30-cdp-qbert-encoder-h5.md) finishes200k actions/799,791 frames/24,954 updates
in40m35s,0.45% more training time than the H5/6e-6 control. Greedy13,889.583
falls1.14% below14,050,both23/24 first pyramids;sampled13,738.542 rises3.06%
from13,331.250,24/24 versus23/24. Own initials80.208/177.083 clear none;
both final means still miss15,000. Retain6e-6 for primary H5/R32 and2e-5 for
the cheaper H15/R16 point:the combination adds0.76% greedy score for34.04%
more training time. All325 pins,100k/12,454 recovery,357 matching/invariant
tensors,47 pre-update transitions,1139/1101 initial events,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Main succeeds03:37:08 UTC
October30/source254c1bd;original postprocess/collection fail on a copied
12,477-versus24,954 report-count assertion. Separate CPU closure completes
01:47:03 UTC November1 in2.020s,preserving325 original/43 repair pins and all
failed services,with zero GPU,replay or plot repeats. Status is
`complete_with_cpu_audit_repair`;no original passing completion is fabricated.
Actual overrides30.632%;guided returns/lower replay CE are not frozen competence.
Native447fe50/qualified runner/teacher fixed,reused4013/cohort,no reliability
claim. Root`/x/Temp/kindle-qbert-encoder-h5-20261030/full`,repair
`/x/Temp/kindle-qbert-encoder-h5-20261030/cpu-closure-20261101`. All original
services terminal,no GPU allocation active;do not restart the learner or failed
observers. No extension,sweep,confirmation or automatic successor selected.
Check live state before GPU work;the five-game goal remains unfinished.

The [Qbert H5/R32 encoder2e-5 check](results/2026-10-30-cdp-qbert-encoder-h5.md) passes8192 actions/32,751 frames/
978 updates in95.153s;1024 frozen/.798s haszero updates/no natural episodes.
All288 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
full replay and guards pass. Conservative200k projection42m35s passes45min.
Main succeeds02:32:47 UTC October30/source8b9936a;collection02:33:13,no failures
or GPU repeats. This qualifies execution/cost of the combined recipe,not gameplay
benefit or a full learner. Native447fe50/qualified runner/teacher unchanged,
12M48/Auto/N8/B16/T16/H5/R32/expert100;no experience reuse. Root
`/x/Temp/kindle-qbert-encoder-h5-20261030/native-check`. No allocation active at
closure;check live state before work. No automatic promotion;goal unfinished.

The [completed Qbert encoder2e-5 comparison](results/2026-10-30-cdp-qbert-encoder-moderate.md) finishes200k actions/799,728 frames/12,477 updates
in30m16s,.58% less training time than6e-6 control. Greedy13,785.417 and
sampled9,987.500 improve11,984.375/6,759.375 by15.03%/47.76%;first pyramids
24/24 and23/24,own initials80.208/177.083. Both means still miss15,000.
Retain2e-5 as the stronger H15/R16 development point,not broad reliability.
All235 pins,100k recovery,357 matched initial/invariant frozen tensors,47
pre-update events,1139/1101 actual-initial traces,full replays/videos,guards,
curves,coverage,exposure and comparison pass. Main/post/collection succeed
02:17:51/02:17:59/02:18:31 UTC October30/source54b1b5d;all five services succeed.
No GPU repeats. Earlier CPU preparation and ancillary observer/metadata errors
remain preserved;the passing pipeline is unchanged. Only encoder6e-6→2e-5
changes as a learner setting on12M48/Auto/N8/B16/T16/H15/R16/expert100;
native447fe50/Qbert teacher fixed,qualified optional-rate runner. Reused4013/cohort;
actual overrides39.667%,not frozen competence. CPU co-host timing caveats remain.
Root`/x/Temp/kindle-qbert-encoder-moderate-20261030/full`. No extension,sweep,
confirmation or automatic successor selected. No learner/GPU allocation active
at closure;check live state before work. The five-game goal remains unfinished.

The [completed Breakout own-trajectory replay](results/2026-10-30-cdp-encoder-representation.md#own-policy-trajectories-show-a-smaller-agreement-gain)
finds near-descending teacher agreement54.18%→57.59%,versus50%→62.5% on the
fixed control states. Different state distributions prevent causal subtraction;
near opposite-direction fractions rise11.04%→13.86%,directional NOOP falls
37.68%→23.24%. All28,808 actions/115,187 frames match retained outcomes;196
analysis/235 active Qbert pins and original process identity pass. CPU18.108s,
zero GPU work/updates,complete01:41:57 UTC October30/source d87b736. CPU co-host
time is retained without correction. No new learner/loss selected. Qbert was
active during this CPU diagnostic;its subsequent completion is recorded above.

The [completed encoder-rate diagnostic](results/2026-10-30-cdp-encoder-representation.md) finds better state and actor readouts on the exact control corpus: kernel
combined ball x/y MAE12.053/10.211→7.554/6.515, still above pixel.059/.140 on555
shared rows. Learned scalar actor agreement54.08%→60.90% overall and50%→62.5%
near descending; near opposite-direction choices19→7, overall81→88. Fixed
linear/kernel action heads still trail the learned actor. All358 pins,357
invariant tensors per stage,full20,960-action/83,799-frame replay,128 native actor
checks and guards pass;71 CPU checks. Capture32.097s,CPU3.445s+1.533s,zero updates.
Service succeeds01:18:25 UTC October30/sourcec7cb852;collection01:21:22. Original
CPU receipt-status failure preserved;no GPU repeat. Reused control states and
teacher estimates limit causal/reliability claims. No new learner or policy
selected;root`/x/Temp/kindle-encoder-representation-20261030`. No active GPU
allocation at closure;check live state. The five-game goal remains unfinished.

The [completed Breakout encoder2e-5 comparison](results/2026-10-30-cdp-breakout-encoder-moderate.md) finishes200k actions/799,511 frames/12,477 updates
in22m27s,0.61% more time than6e-6 control. Sampled25.292/greedy42.625 improve
13.208/17.875 by91.48%/138.46%;own initials1.417/2.208 match,all0/24 two-wall.
Retain2e-5 as stronger compact development point,not broad reliability. All173
pins,100k/6227 recovery,357 matched initial/invariant frozen tensors,47 pre-update
transitions,800/795 initial semantic events,full replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed00:59:07/00:59:27/
01:00:00 UTC October30/source24f69b5;all services terminal success,no GPU repeat.
Earlier CPU preflight-key failure preserved. Only encoder6e-6→2e-5 changes in
12M64/Auto/N8/B16/T16/H15/R16/expert100;native447fe50,teacher fixed,qualified
optional-rate runner. Reused1009/cohort;no isolated architecture/reliability claim.
Actual overrides53.644%,late replay CE.591 versus.696;neither is frozen competence.
Root`/x/Temp/kindle-breakout-encoder-moderate-20261030/full`. CPU co-host caveats
remain;no extension,sweep,confirmation or automatic successor. No active GPU
allocation at closure;check live state. The five-game goal remains unfinished.

The [native optimizer check](results/2026-10-30-cdp-breakout-native-optimizer.md) completes both disposable steps, with
exact raw gradients and all172 clipping/momentum/variance and8 group-movement
checks passing. Observed encoder update/weight .00211656/.00233922% versus
recurrent .149005/.144894%. The full comparison is protocol-failed: six scalar
parameter differences3.20e-12–9.04e-12 exceed the original cancellation allowance.
All29,376,512 parameter writes reproduce exactly from observed native momentum;
this conditional check does not replace the failed F64 prediction. Native loop
.808s/test1.57s; main fails00:10:08 UTC October30/source56d59fd. Separate CPU
closure checks238/11 pins and preserves every bound, ending00:13:13; collection
00:17:01. Original875718d graph-shape startup failure/219 pins also remain.
No completed GPU stage repeats;142 Rust CPU tests/Clippy and4 fixtures pass.
Corpus provenance corrected: unaided final-greedy trajectories, teacher labels
offline. Root `/x/Temp/kindle-optimizer-native-20261030/scalar-corrected`.
Both services terminal, no learner updates/new setting; check live state before
GPU work. The five-game goal remains unfinished.

The [completed CPU optimizer attribution](results/2026-10-29-cdp-breakout-optimizer-attribution.md) estimates all12 encoder
parameter tensors clipped on both fixed batches; encoder update/weight norm
.00212–.00234% versus recurrent .1449–.1490%. Saved momentum changes total
alignment: full-update/current-expert cosine .299–.784. These are F64 estimates,
not native parameter movement or a gameplay effect. Both86-parameter batches,
191 pins, prior raw matrices and conditional component sums pass in4.150s;
12 CPU tests and zero native-import attempts. Terminal review23:53:36 UTC
October29/source46534b7. No updates, GPU work, new learner or optimizer setting.
Root `/x/Temp/kindle-optimizer-attribution-20261029`; native qualification remains
unmeasured. Check live state before GPU work; the five-game goal is unfinished.

The [completed Breakout gradient diagnostic](results/2026-10-29-cdp-breakout-gradient-attribution.md) finds expert raw gradient
norms above CDP in encoder, posterior and dynamics on both fixed B16/T16 batches;
total–expert cosine.925–.997. This does not support CDP swamping imitation on
these states; actual clipped/LaProp updates and gameplay effects remain unmeasured.
Corrected Auto execution succeeds23:40:08 UTC October29/sourced3283ba, with2.868s
measured loop/7.31s native test, zero updates,139 pins,357 invariant tensors and
exact production-total gradient agreement. Both component sums/resize/host guard
and terminal collection pass23:41:12. Original523e67e startup assertion/guard
remain failed: a BF16-staged assumption rejected Auto before either batch;119
original pins were verified and the corrected run repeats no completed batch.
All141 CPU tests/Clippy pass; production learner/runtime unchanged. Root
`/x/Temp/kindle-gradient-attribution-20261029/auto-corrected`. No new learner,
coefficient change or automatic GPU successor. Both services terminal; check
live state before work. The five-game goal remains unfinished.

The [completed FIRE-weight4 comparison](results/2026-10-29-cdp-breakout-fire-weight.md#full-comparison-completes-without-timeouts-but-does-not-improve-score) finishes 200k actions /
799,434 frames / 12,477 updates in 22m07s. Sampled 12.375 / greedy 17.792 do not
improve retained 13.208 / 17.875; all 0/24 two-wall successes. Own initials
1.417 / 2.208 match. All four frozen arms have zero timeouts; longest recorded
FIRE-free command gaps are 123 sampled / 167 greedy, not verified serving waits.
All 196 pins, 100k/6227 recovery, 372 matched initial/invariant frozen tensors,
357 shared control tensors, paired initial traces, full replays/videos, guards,
curves, exposure and comparison pass. Main/post/collection succeed
23:08:19 / 23:08:40 / 23:08:43 UTC October29, source ce86d72. No full-run failure
or GPU repeat occurs. Actual overrides are 58.6715%; guided returns are not
unaided competence. Relative to the accepted control, centroid loss/head,
FIRE weight, native and runner differ; reused seed/cohort, no isolated cause or
reliability claim. Retain the control; no extension, weight sweep, confirmation
or automatic successor. Root `/x/Temp/kindle-breakout-fire-weight-20261029/full`.
All services are terminal; no GPU allocation is active. Check live state before
work. The five-game goal remains unfinished.

The [FIRE-weight4 bounded check](results/2026-10-29-cdp-breakout-fire-weight.md) completes8192 actions/32,736 frames/489 updates
in48.059s; frozen1024/4,076 frames takes.971s with zero updates and eight natural
zero-return episodes. All142 pins,372 matched initial/invariant frozen tensors,
357 shared control tensors,47 pre-update events, full guided replay and both
host guards pass. Conservative200k projection21m31s passes30min; this bounded
check establishes execution and cost only. The full result is recorded above.
Service succeeds22:27:25 UTC October29/sourcec4bb03b,
collection22:27:54. The699 CPU checks pass. An earlier preflight accidentally
initializes the integrated GPU before rejecting a fake frame, with no actions,
updates or checkpoint; source/receipt/kernel review are retained, no fault/reset
is found, and corrected preflight passes before guarded RX7900XT execution.
No declared training/frozen stage repeats. Native2974160/centroid100/expert100/
12M64/Auto/N8/B16/T16/H15/R16 retained; only FIRE weight1→4 changes. The separate
2.668s CPU audit finds23,317 FIRE labels,99.805% replay-label coverage and a longest
45-action passive FIRE-label run, unlike the frozen12,828-action wait. Evidence
`/x/Temp/kindle-breakout-fire-weight-20261029`. All services terminal; no automatic
successor. Check live state before work; the five-game goal remains unfinished.

The [CPU timeout diagnostic](results/2026-10-29-cdp-centroid-timeout.md) identifies a Breakout serving failure:
12,828 actions without FIRE after a life loss, with two lives remaining.
FIRE launches a moving ball at the preselected5,000/10,000-step origins; at20,000,
colors have changed and eight FIRE actions leave the image unchanged. The late
state remains unexplained. All26,804 lane actions /107,212 raw frames and24
recorded branch rows match exactly; all22 pins pass. CPU replay19.309s completes
22:10:19 UTC October29; collection22:12:16, source45f23ea plus pinned script.
Root `/x/Temp/kindle-centroid-timeout-20261029`. Zero native imports, GPU work or
learner updates. No new learner, evaluation override or automatic successor is
selected; the full learner remains protocol-failed. Check live state before GPU work.

The [native-checked actor diagnostic](results/2026-10-29-cdp-centroid-actor.md) finds teacher agreement
54.08%→55.41% overall and 50.00%→58.82% near the paddle for centroid100.
FIRE agreement falls 42/44→34/44; 56/136 urgent decisions still disagree.
Linear/RBF replacement readouts trail the learned actor. Two 128-action checks
match CPU/native probabilities within 9.45e-7, with exact greedy choices and
357/372 invariant tensors. All 289 pins, 26 CPU tests, host guards and collection
pass. Checks take 3.364s/2.624s; CPU analysis 1.630s, zero updates. Service succeeds
21:56:33 UTC October 29/source7e3b0f1; collection 21:56:51. Root
`/x/Temp/kindle-centroid-actor-20261029`. This measures action agreement on the
reused control corpus, not causal feature use, gameplay gain or the timeout's
cause. The full learner remains protocol-failed. No new learner or automatic
successor is selected; all services are terminal. Check live state before GPU work.

The [frozen centroid state readouts](results/2026-10-29-cdp-centroid-representation.md) complete with better recurrent
ball-position precision: linear x/y MAE 12.878/10.918→9.267/8.064 on 555 shared
rows, versus pixel .059/.140. Near-paddle target-x improves only 14.848→13.486;
nonlinear target readouts do not improve there. All 265 pins, 372 invariant
tensors, exact recorded trajectories and host guards pass. Full capture takes
32.435s and CPU fitting 3.368s, with zero updates. Service succeeds 21:44:05 UTC
October 29/source2418ac6; collection passes 21:44:19. The 47 scoped CPU checks
pass after preserved preflight/legacy-import failures; no GPU stage repeats.
Root `/x/Temp/kindle-centroid-representation-20261029`. Position accessibility
improves, but actor use and gameplay benefit remain unmeasured. The full learner
remains protocol-failed. No new learner or automatic successor is selected;
all services are terminal. Check live state before any later GPU work.

The [Breakout centroid100 comparison](results/2026-10-29-cdp-color-centroid.md#full-200k-comparison-fails-the-evaluation-protocol) finishes200k actions/799,500 frames/
12,477 updates in21m45s, but fails its zero-truncation protocol: the final greedy
arm has one25,000-action timeout. Sampled14.375 versus13.208 and selected-natural
greedy20.125 versus17.875 are descriptive; all0/24 two-wall successes. Own initials
1.417/2.208 match. Retain the control; no accepted full comparison or promotion.
All187 original/234 closure pins,100k/6227 recovery,372 bounded-matching initial/
invariant frozen tensors,357 shared control tensors,both initial semantic traces,
full replays/videos and five host guards pass. Main fails21:18:18 UTC October29,
sourceb58d3a2; CPU-only evidence closure finishes21:25:40,collection21:31:01.
Status`closed_protocol_failed`; no GPU stage repeats or original passing review
is created. Late raw position loss falls.05547→.00349 across50k windows,while
expert CE ends.684 versus control.696; held-out position is assessed above.
Actual overrides55.413%; guided returns are not unaided competence. Native2974160,
runner and added656386 parameters differ from447fe50; reused1009/cohort, no
isolated causal or speedup claim. Root`/x/Temp/kindle-color-centroid-20261029/full`.
All services are terminal; no GPU allocation remains active. No retry,extension,
confirmation or automatic successor. Check live state before any later GPU work.

The [optional native color-position loss](results/2026-10-29-cdp-color-centroid.md)
passes F64 targets/loss/derivatives and a fresh Breakout8192-action/32,735-frame/
489-update check in48.101s;1024 frozen actions take.875s with zero updates and
no natural episodes. Shared357 initial tensors and47 pre-update semantic events
match;all372 frozen tensors,full pixel/action/label replay and guards pass.
The head adds656386 parameters;scale100,12M64/Auto/N8/B16/T16/H15/R16/expert100,
other retained settings unchanged. Native2974160/SHAe7ba0d4 at
`/x/Temp/kindle-color-centroid-20261029/runtime`;default off. Its683 packaged
Python/144 Rust CPU checks and strict Clippy pass. Three isolated numerical
fixtures are retained:readback and exact-float test failures precede the pass,
with identical production source. The bounded service fails20:42:25 UTC only
because its CPU schema omitted ten Adam moments;separate CPU repair completes
20:43:45 preserving41 original/21 repair pins and zero repeated training/frozen
stages. Original service remains failed;closure`complete_with_cpu_audit_repair`.
The2.9GiB build cache is released;wheels/test binaries/evidence remain. The bounded
check establishes execution and early cost; the separately selected full learner
and its protocol-failed outcome are documented above.

The [fixed nonlinear Breakout state readout](results/2026-10-29-cdp-breakout-representation.md#nonlinear-state-readouts-still-leave-substantial-position-error)
also leaves substantial position error:combined-feature ball x/y MAE12.053/10.211
versus pixel.059/.140 on555 shared test rows. Motion decoding improves,but this
does not prove information absent or identify the policy's causal limitation.
All35 CPU tests/93 pins pass;5.676s CPU-only analysis completes20:14:49 UTC,
source378f4f8. No new learner or GPU work occurs in that diagnostic.

The [completed Breakout H5/R32 comparison](results/2026-10-29-cdp-breakout-h5-r32.md) finishes200k actions /799,475 frames /24,954 updates in33m30s,
50.12% more training time than H15/R16. Sampled13.125/greedy14.333 do not improve
control13.208/17.875; all0/24 two-wall successes. Final50k-action expert CE
falls.696→.391 without a frozen gameplay gain. All169 pins,100k/12,454 recovery,
357 matched initial/invariant frozen tensors,paired initial semantic traces,
full replays/videos,guards,curves,exposure and comparison pass. Source86eb300;
main/post/collection succeed20:05:58/20:06:03/20:06:35 UTC October29. Actual
overrides54.8335%;online guided returns are not unaided competence. H15→5/R16→32
changes horizon,cadence,sampling,warmup and total work together on reused1009;
no isolated cause or reliability claim. Retain H15/R16;no extension,confirmation
or automatic successor. Native447fe50,runner and teacher remain unchanged.
Root`/x/Temp/kindle-breakout-h5-r32-20261029/full`. CPU co-hosting caveats remain.
All services are terminal;no GPU allocation is active. Check live state before
GPU work;do not restart this learner or the failed posterior assessment.

The [fixed RGB64 pixel diagnostic](results/2026-10-29-cdp-breakout-representation.md#rgb64-pixels-retain-much-more-precise-ball-position-than-the-linear-feature-readouts-recover)
finds ball x/y MAE.059/.140 source pixels on555 shared test rows,versus learned
CNN13.387/13.861 and recurrent12.878/10.918. Two reference balls are missed and
four extra detections remain recorded. Color/crop priors and reused teacher
labels limit this to available pixel precision,not causal proof of feature
information loss or a selected architecture/loss change. Complete20,960-action/
83,799-frame CPU replay and paired rows pass;all18 diagnostic/169 active inputs
and original process identity remain unchanged.9.701s+.161s CPU co-hosting is
retained without time correction;no GPU work or learning occurs.

The [Breakout modal-posterior assessment](results/2026-10-29-cdp-greedy-posterior.md#breakout-trained-score-rises-slightly-the-paired-assessment-fails-its-protocol) is terminal and protocol-failed: trained sampled/modal means17.875/18.833 both
have0/24 two-wall successes, while the actual-initial modal arm records five
25,000-action time limits. Default final/initial traces reproduce; all357 frozen
tensors per arm,100 original pins,full recorded-action replays/videos and host
guards pass. Main fails19:07:21 UTC October29/sourcef676b27. Separate72.712s
CPU closure completes19:13:56 with192 pins; an earlier over-specific CPU
assertion remains preserved. No GPU stage repeats. Four arms total251,928
actions/1,007,559 frames/159.615s,zero updates; original22m19s training remains.
Retain sampled posteriors; no accepted paired modal gain,retry,cap increase or
successor. Closure status is `closed_protocol_failed`; no original passing
comparison/completion exists. No learner or GPU allocation is active at closure.
Check live state before GPU work; the five-game goal remains unfinished.

The [completed Qbert6029 H5/R32 learner](results/2026-10-29-cdp-qbert-h5-seed6029.md) finishes200k actions/799,761 frames/24,954 updates in40m22s.
Greedy15792.708/24 first pyramids and sampled15420.833/24 pass15k/90%,versus
own initials91.667/0 and103.125/0. Seed4013 remains14050/23 and13331.250/23;
one of two H5/R32 learners passes,not broad reliability. All263 pins,100k recovery,
357 finite/invariant tensors,48 recorded initial differences,full pixel/action/
label and frozen replays/videos,guards,curves,coverage,exposure and comparison pass.
Main succeeds17:52:35 UTC October29/source9920dee,postprocessing17:52:48,
collection17:53:23;all services succeed,no failures or GPU repeats. Same12M48/
Auto/N8/B16/T16/H5/R32/expert100/native447fe50,no experience reuse. Actual
overrides67,200 (33.6%);online guided scores are not frozen competence. No paired
H15/R16 control for6029. CPU co-hosting caveats remain;no time correction.
Root`/x/Temp/kindle-qbert-h5-seed6029-20261029/full`. No Qbert extension,third
learner or automatic successor selected. The five-game goal remains unfinished.

The [completed Breakout reconstruction comparison](results/2026-10-29-cdp-breakout-rgb-control.md) finishes200k actions/799,437 frames/12,477 updates in27m36s,23.63% slower
than retained CDP. Sampled10.625 versus13.208 andgreedy16.750 versus17.875
are lower;own initials1.417/2.208 match,bothzero of24 two-wall successes.
Retain CDP12M64/B16/T16/H15/R16/expert100;no RGB extension/confirmation.
All212 original/266 repair pins,100k recovery,399 finite/invariant tensors,
399 bounded-matching/342 shared initial values,800/795 matching initial semantic
events,full pixel/action/label and frozen replays/videos,guards,curves,exposure
and comparison pass. Main succeeds18:45:25 UTC October29/source531ce35.
Original postprocess fails18:45:46 on an eager ALE import in the plot environment;
separate4.655s CPU repair completes18:49:22,no GPU or curve repeat. Original
bounded-schema/collection-observer failures remain retained. Completion exists;
do not rerun collectors. All services terminal,no GPU allocation active.
Actual overrides114,085 (57.0425%);guided207.779 is not frozen competence.
Late expert CE.704 versus CDP.696 gives no training-fit improvement;not a
causal diagnosis. Reused1009/cohort,changed objective/head parameter count,
2.402s CPU co-host caveat retained. Root`/x/Temp/kindle-breakout-rgb-control-20261029/full`.
The five-game goal remains unfinished;check live state before GPU work.

The completed [Breakout kernel action readout](results/2026-10-29-cdp-breakout-representation.md#a-fixed-nonlinear-readout-also-trails-the-recorded-policy) takes1.824s CPU and scores49.25% overall/47.06% near-paddle teacher agreement,
below linear50.58%/49.26% and recorded policy54.58%/53.68%. No evidence for this
replacement head; other decoders/feature precision remain unresolved.31 CPU
tests,96 pins and263 active Qbert6029 pins pass; original PID/invocation stays.
Review17:16:31 UTC October29/source7b43296. The original historical-pin failure
is retained; two exact old sources are resolved fromGit before the first fit.
No GPU work or new learner. Tests1.103s/analysis wrapper2.621s co-host with6029;
retain as a timing caveat without wall-time correction. Evidence
`/x/Temp/kindle-breakout-kernel-readout-20261029`.

The [completed H5/R32 frozen temperature assessment](results/2026-10-29-cdp-qbert-h5-temperature.md) scores14025/22 first pyramids
at0.5 versus retained greedy14050/23;default sampled13331.250/23 reproduces
exactly. Both own initial arms177.083/0;all final means still miss15000. Retain
greedy forH5/R32;no sweep,extension,confirmation or automatic successor. All63
pins,357 invariant tensors per arm,6722/1138 default semantic events,replays/
videos and guards pass. Four arms135,136 actions/540,407 frames/81.325s,zero
updates;original40m24s training remains. Service succeeds16:45:24 UTC October29
/source81a0280 after2m43.706s;collection16:47:10. Qualifiede416696/runtime
`/x/Temp/kindle-action-temperature-20261029/runtime` unchanged,no build/retry.
Root`/x/Temp/kindle-qbert-h5-temperature-20261029`;no GPU allocation active at
completion. Reused development cohort;the five-game goal remains unfinished.

The [completed Qbert H5/R32 comparison](results/2026-10-29-cdp-qbert-h5-r32.md) finishes200k actions/799,762 frames/
24,954 updates in40m24s,+32.67% training time. Greedy14050/23 first pyramids
improves11984.375/24 by17.24%;sampled13331.250/23 improves6759.375/24 by97.23%.
Own initials80.208/177.083 clear none;both candidate means still miss15000.
RetainH5/R32 as strongest primary development score andH15/R16 as cheaper point;
no extension,confirmation or automatic successor selected. All229 pins,100k
recovery,357 matched initial/invariant frozen tensors,paired initial traces,full
replays/videos,guards,curves,coverage,exposure and comparison pass. Main succeeds
16:31:07 UTC October29/sourcee57dd2f,postprocessing16:31:24,collection16:32:04;
all services succeed,no GPU stage repeats. Actual overrides75,030 (37.515%);
online aided returns are not competence. H15→5/R16→32 changes horizon,cadence,
sampling,warmup and total work together;no isolated effect or seed confirmation.
Native447fe50,runner,teacher fixed. Evidence`/x/Temp/kindle-qbert-h5-r32-20261029`.
No GPU allocation remains active at completion;check live state before work.
The five-game goal remains unfinished.

The [completed Qbert B8/T32 comparison](results/2026-10-29-cdp-qbert-context32.md) finishes200k actions/799,710 frames/
12,469 updates in31m48s,+4.44% training time. Greedy11609.375/24 first pyramids
falls3.13% below B16/T16's11984.375/24;sampled6120.833/20 falls9.45% below
6759.375/24. Own initials80.208/177.083 clear none;both miss15000. RetainB16/T16;
no extension,confirmation or automatic successor. All229 pins,100k recovery,357
matched initial/invariant frozen tensors,paired initial traces,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Main succeeds15:15:59 UTC
October29/source07ce3dc,postprocessing15:16:17,collection15:17:05;all services
succeed. B16/T16→B8/T32 also changes diversity,warmup and target rows;no isolated
memory claim. Actual overrides42.950%;online aided returns are not competence.
The.788s CPU source review remains a timing caveat. No allocation active at completion;
check live state before work. The five-game goal remains unfinished.

A.788s [exact-runtime source review](results/2026-10-29-cdp-learning-diagnosis.md#reviewed-expert-label-path-has-no-apparent-one-step-offset)
finds no apparent one-step expert-label offset. Labels stored on arrival t+1
train posterior state t, with explicit boundary/tail masking and matching GPU
feature order. This is source evidence, not a new device-index/gradient test.
All229 active T32 pins and original process identity stay unchanged. The14:40:56
UTC CPU co-hosting is retained as a timing caveat; no GPU work or setting change.

The [Qbert B8/T32 check](results/2026-10-29-cdp-qbert-context32.md) passes8192
actions/32,756 frames/481 updates in72.797s;1024 frozen/.788s haszero updates
and no natural episodes. All41 pins,357 matching initial/invariant frozen tensors,
full guided replay and guards pass;service succeeds14:28:05 UTC October29,
collection14:28:28. Conservative200k projection32m35s passes45min. The separately
selected [fresh200k comparison](experiments/2026-10-29-cdp-qbert-context32-200k.md),
12M48/Auto/N8/B8/T32/H15/R16/expert100 completes with100k recovery and four frozen arms.
B16/T16→B8/T32 preserves256 positions/update but changes temporal gradients,
sequence diversity,warmup and targets;no isolated memory claim. Native447fe50,
runner,teacher and supervision fixed. No prior experience reused. Check live state.

The [completed action-temperature assessment](results/2026-10-29-cdp-qbert-action-temperature.md)
scores12389.583/24 first pyramids at0.5,3.38% above retained greedy11984.375/24;
default sampled6759.375/24 reproduces exactly. Own initials177.083/0;mean15000
still fails. Retain0.5 as a development option with a small point-estimate gain,
not broad superiority or a global default. All57 pins,357 invariant tensors per
arm,4204/1138 default semantic events,replays/videos and guards pass. Four arms
96,872 actions /387,366 frames /58.516s,zero updates;original30m27s training stays.
Service succeeds14:10:15 UTC October29/source4b24f75 after2m04.158s;collection
14:10:56. No active allocation,sweep,extension or successor. The goal is unfinished.

The [completed Qbert25M/R16 comparison](results/2026-10-29-cdp-qbert-25m-r16.md)
finishes200k actions /799,730 frames /12,477 updates in42m51s,40.72% more training
time than12M/R16. Primary greedy11984.375→9773.958 (−18.44%),both24/24 pyramids;
sampled6759.375→8678.125 (+28.39%),24→22 pyramids. Own initials9.375/177.083
clear none;both means still miss15000. Retain12M/R16;preserve sampled gain as
a separate budget point. No extension,confirmation or automatic successor.
All273 pins,100k recovery,357 matching25M bounded/R8 initial and invariant frozen
tensors,paired initial traces,full replays/videos,guards,curves,coverage,exposure
and three-arm comparison pass.25M/R16 also improves both modes over25M/R8.
Only model size changes from12M/R16;geometry/initialization differs,no12M initial
equality. Supervision/native447fe50/runner/teacher fixed. Actual guidance41.6855%.
Main succeeds13:38:47 UTC October29/source5d9d93e,postprocessing13:38:55,
collection13:39:32;all four services succeed. Original CPU preparation failure
and2.290s co-host diagnosis retained;no GPU repeat. Root
`/x/Temp/kindle-qbert-25m-r16-20261029/full`. Reused development seed/cohort.
No GPU allocation active at completion;the five-game goal remains unfinished.

A [CPU loss-floor diagnosis](results/2026-10-29-cdp-learning-diagnosis.md#late-latent-prediction-loss-remains-above-its-floor)
finds late mean dynamics KL5.491 Qbert/11.357 Breakout, above the one-nat floor.
Per-state clipping and derivatives remain unmeasured; no additional learner is
selected. The2.290s co-hosted check preserves all273 active pins and main invocation.

The [completed Qbert25M/R16 bounded check](results/2026-10-29-cdp-qbert-25m-r16.md)
passes8192 actions /32,757 frames /489 updates in100.299s,then1024 greedy frozen
/4096 frames in1.202s,zero updates/no natural episodes. All53 pins,357 finite/
invariant frozen tensors,357 matching25M/R8 initial tensors,derived geometry,
parameter changes,finite expert reports,full guided replay and guards pass.
Only exposed model size changes from12M/R16;update schedule/supervision/native
447fe50/control runner/teacher fixed. No12M initial equality. Conservative200k
projection44m54s narrowly passes45min;execution/early cost only,not competence.
Actual short-check overrides55.457%. Service succeeds12:33:10 UTC October29/
source `a25513b`,collection12:33:56. No failure/GPU repeat/full learner or automatic
promotion. Root `/x/Temp/kindle-qbert-25m-r16-20261029/native-check`. Check live state.

The [completed Qbert25M/R8 comparison](results/2026-10-29-cdp-qbert-25m-r8.md)
finishes200k actions /799,672 frames /6239 updates in26m24s,13.28% less training
time. Greedy11984.375→5817.708 (−51.46%),both24/24 first pyramids;sampled6759.375
→4533.333 (−32.93%),24→23 pyramids. Own initials9.375/177.083 clear none;mean15000
still fails. Retain12M/R16;no extension,confirmation or automatic successor.
All226 pins,100k/3114 recovery,357 matching25M bounded initial/invariant frozen
tensors,full replays/videos,guards,curves,coverage,exposure and comparison pass.
Capacity12M→25M and replay16→8 change together;no isolated effect or12M initial
equality. Same native447fe50/control runner/teacher/supervision. Actual guidance
45.751%;guided online mean15207.979 is not competence. Main succeeds12:21:32 UTC
October29,postprocessing12:22:03,collection12:22:22;all four services succeed,
no GPU repeat or failure. Root `/x/Temp/kindle-qbert-25m-r8-20261029/full`.
No GPU allocation active at completion;check live state before work.

The [completed Qbert25M/R8 bounded check](results/2026-10-29-cdp-qbert-25m-r8.md)
passes 8192 actions / 32,753 frames / 245 updates in 63.119s, then 1024 frozen
/ 4096 frames in 1.148s, zero updates and no completed natural episodes. All 44
pins, 357 finite/invariant frozen tensors, derived geometry/31,636,273 parameters,
update checks, full training pixel/action/label replay and host guards pass.
Capacity 12M→25M and replay16→8 change together; native447fe50/control runner/
teacher and supervision remain fixed. No cross-model initial equality is claimed.
Conservative 200k projection 28m15s passes the 35min cost gate; this is early cost,
not competence. Service succeeds 11:32:49 UTC October29/source `aeee122`, terminal
collection 11:33:20. Actual guidance 54.089% in this short check. No full learner
or automatic promotion selected. Root `/x/Temp/kindle-qbert-25m-r8-20261029/native-check`.
Check live state before GPU work.

The [completed Qbert NOOP-weight comparison](results/2026-10-29-cdp-qbert-noop-weight.md)
finishes200k actions/799,700 frames/12,477 updates in30m12s. Weight1→.25 lowers
greedy11984.375→9756.250 (−18.59%),both24/24 pyramids,and sampled6759.375→
6120.833 (−9.45%),24→22 pyramids. Own initials80.208/177.083 clear none;mean15000
still fails. Retain weight1;no extension,confirmation or further sweep selected.
All230 pins,100k recovery,357 initial/frozen tensors,paired initial traces,full
replays,guards,curves,coverage,exposure and comparison pass. Main succeeds11:18:14
UTC October29,postprocessing11:18:33;all services/collector succeed. Actual
training guidance43.631%;online aided scores are not competence. A1.000s CPU
recount finds later passive wait agreement95.044%→86.137%,hop55.968%→58.100%;
changing states prevent causal attribution. All10 recount pins match. Same native
447fe50/control runner/teacher;no GPU repeats or new allocation. The separate
empty-log status-read failure remains recorded. Check live state before work.

The [bounded Qbert NOOP-weight check](results/2026-10-29-cdp-qbert-noop-weight.md)
passes8192/489 in70.160s and1024 frozen/.783s,zero updates/no natural episodes.
All45 pins,357 matched initial/invariant frozen tensors,weighted loss counts,
full guided replay and guards pass;service succeeds10:29:24 UTC October29.
Only wait-label weight1→.25 changes under retained12M/B16/R16/expert100/native
447fe50;actor/world losses and total expert magnitude change together.
Conservative200k projection31m24s passes35min;task benefit remains untested.
Original CPU import assertion failure is preserved,corrected before GPU launch.
No full learner,extension,confirmation or active GPU allocation remains.

Updated October29,2026. [Handover](handover.md) contains the reproducible runtime
and machine-transfer instructions. [PR31](https://github.com/kvark/kindle/pull/31)
is the status dashboard; individual result reports preserve the evidence.

The [frozen posterior-mode assessment](results/2026-10-29-cdp-greedy-posterior.md)
lowers Qbert11984.375→10539.583 with modal categories,each24/24 first pyramids
and below mean15000. Own initials80.208/0.000 clear none. Retain sampled posterior
states with greedy actions. Four arms total110,112 actions/440,249 frames/67.014s,
zero updates;original training remains30m27s. All53 pins,357 invariant tensors,
6080/1100 exact default semantic events,full replays/videos and guards pass.
Service exits10:12:26 UTC October29;terminal collection passes,no GPU allocation
or successor selected. The optional native mode passes focused checks after a
test-only synchronization repair;original failure remains preserved. This is
one reused learner/cohort,not independent confirmation or a global default.

The actual action-block diagnostic completes128 origins/2432 forecasts while
re-observing48256 actions in95.811s,zero updates. Ranking agreement72.96% before/
57.21% after the first pyramid, with40/128 informative origins, supports useful
but weaker later-game action discrimination. Later gain1.601 over uniform has
interval[−.131,3.443]. All46 pins,357 invariant tensors,128 exact forecast matches,
pixel trajectories and guards pass; service exits06:52:55 UTC October29. The
original CPU snapshot failure is retained; isolated process branches preserve
sticky-action memory and pass full source replay. See the
[completed result](results/2026-10-29-cdp-learning-diagnosis.md).
The [optional three-step open-loop supervision](experiments/2026-10-29-cdp-open-loop-supervision.md)
is implemented default-off (`ab80486`):135 Rust/611 packaged Python CPU tests,
strict Clippy,two guarded gradient checks and legacy frozen compatibility pass.
Fresh8192-action/489-update Qbert12M check takes70.207s;all44 pins,357 initial/
frozen tensors,auxiliary reports,replay and guards pass. Conservative200k
projection31m25s passes35min,but no frozen episodes complete in the short check.
See the [completed result](results/2026-10-29-cdp-open-loop-supervision.md).
The [fresh200k comparison](results/2026-10-29-cdp-open-loop-supervision.md#full-comparison-retains-the-control)
completes799,709 frames/12,477 updates in30m15s. Greedy9377.083 and sampled
5613.542,each23/24 first pyramids,are lower than control11984.375/6759.375,
each24/24. Own initials80.208/177.083 havezero pyramids. The original15000 mean
gate remains unmet. Retain control; no extension,confirmation or retry.
All242 pins,100k recovery,357 initial/frozen tensors,initial semantic traces,
full replays/videos,guards,curves,coverage,exposure and comparison pass. Main
exits08:41:03 UTC October29,postprocessing08:41:29;allservices/collector succeed.
Actual training overrides87,421 actions(43.7105%);online aided scores are not
competence. No GPU allocation is active. The five-game goal remains unfinished.
The [subsequent frozen forecast comparison](results/2026-10-29-cdp-open-loop-supervision.md#frozen-forecasts-worsen-overall-with-a-small-early-game-improvement)
finds H3 raises15-step reward-sum MAE37.499→40.793 on the control trajectories:
early2.347→2.035 improves,but later45.259→49.350 worsens. Captures48,256 actions/
192,980 frames each take208.483s/207.348s,zero updates. All152 control arrays
reproduce exactly;65 current/39 resolved historical pins,357 invariant tensors,
eight frame-hash trajectories,guards and34 CPU tests pass. Service exits09:49:38
UTC October29;terminal collection passes. Retain control;no new loss or learner
selected,and no causal policy mechanism established.

The concurrent33.384s CPU-only Breakout replay finds near-
paddle teacher agreement55.19%/54.18% for retained4M/12M greedy policies,despite
their different overall scores. Canonical-only chosen buttons rule out aliases
as the explanation for those greedy errors. See the [phase diagnostic](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-frozen-policy-still-disagrees-on-urgent-paddle-commands);
all242 active Qbert pins and its process identity stay unchanged.
The subsequent [frozen Breakout representation readout](results/2026-10-29-cdp-breakout-representation.md)
completes20,960 actions/83,799 frames per model in32.301s trained/32.420s initial,
zero updates. Recurrent motion R² improves.282/.051→.473/.614. Trained paddle
MAE is4.334 CNN/9.643 recurrent pixels;near-paddle recurrent teacher-target MAE
remains14.848. Useful information is present,but this does not identify the
cause of weak control. All55 pins,357 tensors per capture,matched trajectories/
input hashes,readback/counter/host checks and terminal collector pass;19 CPU
tests pass. Service exits09:02:23 UTC October29. No native change,new learner
or active GPU allocation. The separately selected CPU teacher-action follow-up
now completes in1.333s,with zero GPU work/updates. Trained concatenated features
reach50.58% overall/49.26% near-paddle agreement,below the recorded native
policy54.58%/53.68%; initial concatenated38.60%/41.18%. All34 inputs and55
prior capture pins stay unchanged;26 CPU tests pass. Review completes09:23:24
UTC October29/source26a9961. No simple replacement-head or direct-CNN benefit
is established; no native change or new learner is selected. See the
[completed action readouts](results/2026-10-29-cdp-breakout-representation.md#teacher-action-readouts-improve-with-learning-but-trail-the-recorded-policy).

The live-critic follow-up completes98664 frozen actions/394520 frames in69.704s,
zero updates. Sampled value estimates beat a constant baseline but underpredict
mean return1218.534 versus1475.689;correlation.542/R².184 and later-phase
correlation.358 leave substantial uncertainty. Greedy changes the policy and
return target. All44 pins,357 invariant tensors per arm,12522 matching semantic
events and guards pass; service exits06:30:01 UTC October29. No new loss or
training is selected; next inspect actual alternative-action ranking. See the
[updated diagnosis](results/2026-10-29-cdp-learning-diagnosis.md).

The October29 architecture diagnosis completes two48256-action frozen captures
of the strongest Qbert12M model and its own initial weights, zero updates.
Fifteen-step reward-sum MAE107.570→37.499 and endpoint AUC.500→.938 show useful
predictive learning on recorded trajectories; unrelated actions give69.538/.673.
Before/after-first-pyramid MAE2.347/45.259 motivates later-play diagnosis, while
reward distributions differ. All39 pins,357 invariant tensors and guards pass;
service exits06:18:53 UTC. Existing replay critic targets already include real
rewards but bootstrap from imagination. Next inspect critic calibration and
actual alternative-action ranking; no changed loss or new training is selected.
Recorded imagination/target work consumes46.62%/40.68% of Qbert/Breakout update
time. See [full diagnostic evidence](results/2026-10-29-cdp-learning-diagnosis.md).
The five-game goal remains unfinished; no GPU allocation is active.

Breakout12M64/B16/R16/expert100 completes 200k actions /799,464 frames /12,477
updates in22m19s. Frozen sampled13.208/greedy17.875 improve4M/B8's9.458/12.208
by39.65%/46.42%, for25.72% more training time; own initials1.417/2.208.
All0/24 two-wall successes. All132 pins,100k recovery,357 bounded-matching
initial/invariant frozen tensors, full replays, guards, curves, exposure,
comparison and terminal collector pass. Main exits05:34:19 UTC,postprocessing
05:34:31 on October29. The original CPU path failure and106 pins remain preserved;
no GPU stage repeats. Retain both budget points; capacity and batch change together
on reused1009, without an isolated effect or reliability claim. Actual overrides
56.712%; online guided scores are not frozen competence. No extension,cap increase,
confirmation or automatic successor is selected. No GPU allocation is active.
See the [completed capacity result](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-capacity-improves-frozen-breakout-with-higher-cost); check live state before GPU work.

Breakout12M64/B16/R16/expert100 bounded check passes8192 actions/32,720 frames/
489 updates in47.623s; frozen1024/4096frames/.877s haszero updates/no natural
completed episodes. All56 pins,357 finite/invariant tensors,16,334,353 parameters,
finite supervised reports,changed CNN/reward/actor/value,3841 verified overrides,
full guided replay and guards pass. Initial values are not compared with the
4M architecture; actual12M64 shapes match the CPU derivation. Conservative200k
projection21m19s passes30min; task competence remains unmeasured. Service succeeds
04:52:02 UTC on October29/source d4200b2. No failure,GPU repeat,cap change,full
learner,extension or automatic promotion occurs. Use the matching Breakout-era
runtime; native447fe50 unchanged. No GPU allocation is active;check live state.
See [the bounded result](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-supervised-breakout-passes-execution-and-early-cost-checks).

Qbert guidance3840 completes200k actions/799,726 frames/12,477 updates in29m31s.
Primary greedy8187.500/24 first pyramids falls31.68% below retained1920's11984.375,
for3.04% less training time; sampled7051.042/23 rises4.31% from6759.375/24. Own
initials80.208greedy/177.083sampled,zero pyramids; both final means miss15000.
All227 pins,100k recovery,357 matching initial/invariant frozen tensors, paired
initial traces,finite reports,full replays,guards,curves,coverage,exposure,
comparison and repaired collector pass. Main succeeds04:39:46 UTC,postprocessing
04:39:55 on October29. Original CPU observer exit1 remains;replacement succeeds
with original deadlines. No GPU stage repeats or caps change. Actual overrides
59.6615%;45,727 actions at ages1920–3839 are all guided,none reaches3840,and no
cap-released native action occurs. These are exposure counts,not verified game
levels or competence. Retain1920;no extension,limit sweep,retry,confirmation or
automatic successor. The.630s co-host repair remains recorded. No GPU allocation
is active;check live state before work. See [the completed result](results/2026-10-25-cdp-cheap-qbert.md#longer-guidance-lowers-primary-greedy-quality-at-the-same-short-budget).

Breakout expert100 completes200k actions/799,351 frames/24,970 updates in
17m45s. Sampled9.458/greedy12.208 improve retained expert10's4.292/5.000 for
1.92% more training time; own initials1.375/2.583. All0/24 two-wall successes.
All69 pins,100k recovery,357 matching initial/invariant frozen tensors, paired
initial traces,finite reports,full replays,guards,curves,exposure,comparison and
original collector pass. Main succeeds03:47:13 UTC,postprocessing03:47:41 on
October29; all four services terminal success. Actual overrides58.7625%;guided
scores are not frozen competence. Retain expert100 as a useful development point
and the distinct stronger sampled controls. Only expert scale10→100 changes on
reused1009; no mechanism or seed-reliability claim. No observer/GPU failure,
GPU repeat,cap change,extension,coefficient sweep,confirmation or automatic
successor. The.894s co-host Qbert analysis remains recorded. GPU idle at completion;
check live state before work. See [the completed result](results/2026-10-26-cdp-breakout-pixel-reference.md#stronger-imitation-improves-frozen-breakout-at-nearly-unchanged-cost).

Qbert12M48/B16/R16/expert1000 completes200k actions/799,697 frames/12,477
updates in30m31s. Greedy10588.542/24of24 first pyramids and sampled6150/21
fall11.65%/9.02% below retained expert100, for0.21% more training time. Own
initials80.208/177.083, zero pyramids; both final means miss15000. All229 pins,
100k/6227 recovery,357 matching initial/invariant frozen tensors, paired initial
traces, finite reports, full replays, guards, curves, coverage, exposure,
comparison and original terminal collector pass. Main succeeds03:14:19 UTC,
postprocessing03:14:39 on October29; all four services are terminal success.
Actual overrides43.8025%; guided scores are not frozen competence. Only expert
scale100→1000 changes on reused4013; no mechanism or seed-reliability claim.
Retain expert100. No observer/GPU failure, repeated GPU stage, cap change,
extension, coefficient retry or automatic successor. The.668s co-host monitor
remains recorded. GPU idle at completion; check live state before work. See
[the completed result](results/2026-10-25-cdp-cheap-qbert.md#stronger-imitation-lowers-both-frozen-scores-at-the-same-short-budget).

Qbert12M48/B16/R8/expert100 completes400k actions/1,599,459 frames/12,489
updates in40m35s. Greedy10042.708/23of24 first pyramids is16.20% below retained
200k/R16's11984.375/24,for33.28% more training time;sampled7456.250/22 rises
from6759.375/24. Own initials80.208greedy/177.083sampled,zero pyramids. Both
modes still miss mean15000. All228 pins,three recoveries,357 matching initial/
invariant frozen tensors,paired initial semantic traces,finite reports,full
label/pixel/action and frozen replays,guards,curves,coverage,exposure and
comparison pass. Main succeeds02:16:04 UTC,postprocessing02:16:22 on October29.
Original CPU progress-observer exit2 is preserved;the private replacement and
repaired terminal collector pass. No GPU stage repeats or caps change. Retain
200k/R16;no extension,seed confirmation or automatic successor is selected.
The experience budget and ratio change together on reused4013;no isolated causal
or reliability claim. Actual training overrides40.40975%;guided scores are not
frozen competence. No learner or GPU allocation is active. See the completed
R8 result in cheap-Qbert and its protocol;check live state before GPU work.

The compatible RGB Qbert control completes200k actions/799,695 frames/24,970
updates in28m41s. Greedy6090.625/24 first pyramids and sampled4335.417/23 are
below retained CDP's9654.167/23 and7569.792/24 in26m07s; both miss mean15000.
Own initials263.542greedy/182.292sampled,zero pyramids. The locked paired lane
learning-per-minute decision favors this CDP recipe:mean CDP-minus-RGB156.359,
95% interval[110.311,206.365],with higher absolute CDP mean and all audits passing.
This is one reused development seed/cohort with strong supervision,not five-game
superiority or independent confirmation. All195 pins,100k recovery,342 shared
CDP/399 own initial tensors,399 invariant frozen tensors,paired initial semantic
traces,full replays,guards,curves,coverage,exposure and comparison pass. Main exits
00:57:02 UTC,postprocessing00:57:28 on October29;all services terminal success.
No GPU repeat,cap change or full-run failure. Actual RGB guidance45.0985%.
Retain the favorable comparison;no extension,confirmation or automatic successor
selected. GPU idle;check live state before work. See [the completed comparison](results/2026-10-29-cdp-qbert-rgb-control.md).

Qbert12M48/B16/R16/expert100 completes200k actions/799,742 frames/12,477
updates in30m27s. Greedy11984.375/24 first pyramids improves4M/B8's9654.167/23
by24.14%,for16.57% more training time;sampled6759.375/24 falls below7569.792/24.
Own initials80.208greedy/177.083sampled,zero pyramids;both final means fail15000.
All190 pins,100k recovery,357 reference/check initial and invariant frozen tensors,
finite reports,changed CNN/reward/actor/value,full label/pixel/action and frozen
replays,guards,curves,coverage,exposure and comparison pass. Main succeeds23:40:33
UTC,postprocessing23:40:50 on October28;all services terminal success. No failed
attempt,GPU repeat or cap/runtime change. Capacity and batch change together on
reused4013;no isolated capacity or seed-reliability claim. Actual overrides42.958%;
guided returns are not competence. Retain12M greedy as a promising development
budget point and4M as cheaper/stronger sampled control. No extension,confirmation
or automatic successor is selected. Use the matching Breakout-era runtime and
pinned Qbert runner;native447fe50 unchanged. See the expert100-capacity protocol
and completed cheap-Qbert result;check live state before GPU work.

Retained Breakout corrective greedy assessment completes:final5.000/initial2.583
versus sampled4.292/1.375,all0/24 two-wall successes. Original17m25s training cost
remains;zero new training/updates. All54 pins,357 invariant tensors per arm,source/
config/counters,full outcome replays/videos and guards pass. Total15,248 actions/
60,929 frames/7.601s evaluation;main succeeds23:45:31 UTC in23.812s,postprocessing
23:45:32 on October28. No failure,GPU repeat or cap change. Retain both weak modes
and stronger controls;no extension,confirmation,global default or automatic next
allocation. Both services are terminal and the host is idle;check live state. [Protocol](experiments/2026-10-28-cdp-breakout-corrective-greedy.md).

Breakout encoder-rate200k completes200k actions/799,152 frames/24,970 updates
in17m41s. Raising encoder6e-6→4e-5 lowers frozen4.292→1.458 versus own initial
1.375; both0/24 two-wall successes. Training costs1.52% more. All64 pins,100k
recovery,357 matched control/check initial and invariant frozen tensors,802 paired
initial semantic events,finite reports,full replays,guards,curves,exposure and
comparison pass. Main succeeds22:45:00 UTC,replacement CPU postprocessing22:45:15
on October28. The copied CPU preparation error and proactively replaced waiting
comparison observer are retained; no GPU stage repeats or caps change. Actual
overrides67.622%; online guided returns are not competence. Retain encoder6e-6;
no extension,seed confirmation or automatic successor. Native447fe50 unchanged;
optional CLI passes348 CPU tests. The device was idle at completion. See the encoder-rate
protocol and completed pixel-reference result; check live state before GPU work.

Breakout corrective200k completes200k actions/799,234 frames/24,970 updates in
17m25s,26.02% less training time than the old forced-only cap1920 control.
Frozen4.292 versus own initial1.375 improves the old guided2.625 but stays below
unaided full18's7.083; all0/24 two-wall successes. All64 pins,100k recovery,
346 shared old/357 bounded initial and357 invariant frozen tensors,paired initial
semantic traces,finite reports,full replays,guards,curves,exposure and comparison
pass. Main succeeds22:12:00 UTC,postprocessing22:12:22 on October28. Original
CPU observer startup failure remains; replacement succeeds,no GPU stage repeats.
Actual overrides67.135%; guided mean203.275 is not competence. Multiple settings
and native/runtime change together on reused1009. Retain stronger controls;
no extension,seed confirmation or automatic successor. The device was idle at completion.
See the corrective-200k protocol and completed pixel-reference result.

Optional Breakout corrective labels pass510 CPU tests,forced-reference parity
and181,224 recorded learner-action feedback checks. Applied source2854dd9 keeps
native447fe50 unchanged. The fresh8192-action/32,725-frame/994-update check takes
36.919s;1024 frozen actions/4096 frames take.555s,zero updates/no natural episodes.
All41 pins,346 shared initial/357 invariant frozen tensors,finite reports,changed
CNN/reward/actor/value,full label/pixel/action replay and guards pass. Main succeeds
21:43:38 UTC on October28. No task competence or cap-release coverage is measured.
Use`/x/Temp/kindle-breakout-corrective-20261028/runtime` for this adapter; original
sources remain archived. Early200k projection15m01s; no full learner was selected at bounded completion.
See the Breakout corrective-label protocol and pixel-reference result. The
separately selected full comparison is completed above.

Qbert expert100/200k/R16 completes200k actions/799,703 frames/24,970 updates in
26m07s. Greedy9654.167/23 first pyramids improves100k/R32's8365.625/24 by
15.40%,for37.63% more training time; sampled7569.792/24 versus7339.583/22.
Own initials263.542/182.292,zero pyramids. Both still miss mean15000. All138 pins,
100k recovery,357 matched initial/invariant frozen tensors,paired initial traces,
full replays,guards,curves,coverage,exposure and comparison pass. Main succeeds
21:41:05 UTC,postprocessing21:41:19 on October28; all services are inactive.
Actual overrides41.383%; guided returns are not competence. Budget and ratio
change together on reused4013; no isolated effect or seed confirmation. Retain
both budget points; no extension or automatic successor. See the expert100-r16
protocol and completed result in cheap-Qbert; check live state before GPU work.

Qbert expert100 completes100k actions/399,886 frames/24,939 updates in18m59s.
Greedy8365.625/24 first pyramids improves scale10's7700/24 by8.64%; sampled
7339.583/22 improves5572.917/24 in mean but loses two pyramids. Own initials
263.542/182.292,zero pyramids. Both still miss mean15000. All111 pins,50k recovery,
357 matched initial/invariant frozen tensors,paired initial traces,full replays,
guards,curves,coverage,exposure and comparison pass. Main succeeds20:55:18 UTC,
postprocessing20:55:29 on October28; all services inactive. Actual overrides
38.397%; guided returns are not competence. Only expert10→100 changes,affecting
actor/world supervision together,on reused4013. Modest unconfirmed same-budget
gain; no extension,second expert100 learner or automatic successor selected.
The bounded check's CPU preparation schema failure remains,no GPU repeat.
See the expert-scale100 protocol and completed result in cheap-Qbert; check
live state before GPU work.

**Current AMD development results:** all completed rows pass numerical,
checkpoint, accounting, frozen-tensor and full-replay audits. The learning gate
is separate. Each row describes one learner or warm-start lineage, not broad reliability.

| Game and recipe | Actions / updates | Training wall | Frozen final / starting policy | Task outcome |
|---|---:|---:|---:|---|
| [Boxing1M/RGB32/R16](results/2026-10-25-cdp-cheap-boxing.md) |200k /24,970|6m03s|−3.875 /0.00|10/24 wins; fail|
| Boxing1M/RGB32/R32, seed2017 |500k /124,939|28m22s|76.833 /2.875|24/24 wins; pass|
| Same recipe, confirmation4013 |500k /124,939|28m13s|79.167 /−0.625|24/24 wins; pass|
| [Freeway1M/RGB64/R16, no aid](results/2026-10-25-cdp-cheap-freeway.md) |200k /24,970|8m07s|0.00 /0.00|0/24 milestones|
| Same,5% persistent action aid in training |200k /24,970|8m11s|0.25 /0.00|0/24 milestones|
| [Freeway1M/RGB32/R32, same aid](results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget) |500k /124,939|28m17s|31.042 /0.00|24/24 milestones; pass|
| [Same recipe, confirmation4013](results/2026-10-25-cdp-cheap-freeway.md#fixed-recipe-seed4013-confirmation) |500k /124,939|28m50s|32.000 /0.00|24/24 milestones; pass|
| [Pong4M/RGB48/R16](results/2026-10-25-cdp-cheap-pong.md) |200k /24,970|10m22s|−17.125 /−20.375|0/24 wins|
| [Pong4M/RGB48/R32](results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions) |500k /124,939|48m53s|11.000 /−20.375|23/24 wins; mean15 gate fails|
| [Same recipe, confirmation4013](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation) |500k /124,939|50m14s|7.333 /−20.333|22/24 wins; mean15 gate fails|
| [Same, coefficient.01, seed1009](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result) |500k /124,939|49m55s|14.958 /−20.375|24/24 wins; mean15 narrowly fails|
| [Same, confirmation4013](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation) |500k /124,939|49m58s|14.000 /−20.333|24/24 wins; mean15 fails|
| [Pong12M48/B16/R16, coefficient.01](results/2026-10-25-cdp-cheap-pong.md#larger-capacity-with-less-replay-work-reaches-the-gate) |500k /31,227|41m28s|15.625 /−20.292|24/24 wins; pass on one learner|
| [Pong12M48, same recipe4013](results/2026-10-25-cdp-cheap-pong.md#capacity-confirmation-improves4013-but-narrowly-misses-the-gate) |500k /31,227|41m31s|14.833 /−20.542|24/24 wins; mean15 narrowly fails|
| [Same retained12M1009, greedy evaluation](results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-passes-the-gate-on-both-retained12m-learners) |500k /31,227 retained;0 new|41m28s retained|15.208 /−20.583|24/24 wins; greedy gate passes|
| [Same retained12M4013, greedy evaluation](results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-passes-the-gate-on-both-retained12m-learners) |500k /31,227 retained;0 new|41m31s retained|15.208 /−20.625|24/24 wins; greedy gate passes|
| [Qbert4M/RGB48/R32](results/2026-10-25-cdp-cheap-qbert.md) |200k /49,939|19m45s|809.375 /180.208|0/24 first pyramids|
| [Same, capped pixel guidance + expert imitation](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate) |200k /49,939|20m11s|1083.333 /180.208|2/24 first pyramids; gate still fails|
| [Same, fresh seed4013](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-transfers-on4013-with-substantial-seed-variation) |200k /49,939|20m07s|3153.125 /182.292|17/24 first pyramids; gate still fails|
| [Expert4013, more experience at R16](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-improves-first-pyramid-success-with-higher-cost) |500k /62,470|28m20s|3752.083 /182.292|21/24 first pyramids; gate still fails|
| [Continuation teacher/cap1920,4013/R32](results/2026-10-25-cdp-cheap-qbert.md#continuation-guidance-improves-frozen-learning-but-misses-the-full-gate) |200k /49,939|25m23s|4730.208 /182.292|23/24 first pyramids; mean15k still fails|
| [Same recipe, fresh1009](results/2026-10-25-cdp-cheap-qbert.md#continuation-improves-the-weaker-seed-but-both-miss-the-full-gate) |200k /49,939|26m38s|3137.500 /180.208|17/24 first pyramids; both gate criteria fail|
| [Retained continuation1009, greedy evaluation](results/2026-10-25-cdp-cheap-qbert.md#greedy-evaluation-lowers-both-retained-qbert-results) |200k /49,939 retained;0 new|26m38s retained|3058.333 /85.417|16/24 pyramids; below sampled3137.500/17|
| [Retained continuation4013, greedy evaluation](results/2026-10-25-cdp-cheap-qbert.md#greedy-evaluation-lowers-both-retained-qbert-results) |200k /49,939 retained;0 new|25m23s retained|3770.833 /263.542|19/24 pyramids; below sampled4730.208/23|
| [Continuation20%/R16,4013](results/2026-10-25-cdp-cheap-qbert.md#more-guidance-with-less-replay-saves-time-but-lowers-frozen-quality) |200k /24,970|21m59s|3333.333 /182.292|18/24 first pyramids; below5%/R32|
| [Same, expert world gradients](results/2026-10-25-cdp-cheap-qbert.md#expert-world-gradients-give-a-small-r16-gain-below-the-stronger-control) |200k /24,970|21m28s|3670.833 /182.292|19/24 first pyramids; small R16 gain, below5%/R32|
| [Same,12M/B16](results/2026-10-25-cdp-cheap-qbert.md#guided12m-b16-r16-lowers-quality-and-costs-more) |200k /12,477|24m51s|1792.708 /177.083|9/24 first pyramids; lower quality and higher cost|
| [Same4M, corrective labels and feedback](results/2026-10-25-cdp-cheap-qbert.md#corrective-labels-cost-more-without-improving-frozen-quality) |200k /24,970|25m57s|3633.333 /182.292|17/24 first pyramids;20.86% more training time without gain|
| [Corrective labels, equal wait weights/R32/100k](results/2026-10-25-cdp-cheap-qbert.md#equal-wait-weights-and100k-actions-save-time-but-lower-frozen-quality) |100k /24,939|18m58s|2260.417 /182.292|12/24 first pyramids;26.90% less time,37.79% lower mean|
| [Same retained100k learner, greedy evaluation](results/2026-10-25-cdp-cheap-qbert.md#greedy-actions-improve-the-retained-equal-weight-policy) |100k /24,939 retained;0 new|18m58s retained|4965.625 /263.542|21/24 first pyramids; both gate criteria still fail|
| [Expert scale10, fresh100k4013](results/2026-10-25-cdp-cheap-qbert.md#stronger-expert-supervision-improves-frozen-qbert-at-the-same100k-budget) |100k /24,939|18m55s|Greedy7700 /263.542; sampled5572.917 /182.292|24/24 first pyramids under both modes; mean15k fails|
| [Expert scale10, fixed-recipe1009](results/2026-10-25-cdp-cheap-qbert.md#expert10-reaches-the-first-pyramid-fraction-on-two-greedy-learners-with-large-score-variation) |100k /24,939|19m11s|Greedy3832.292 /85.417; sampled2667.708 /180.208|23/24 greedy,15/24 sampled pyramids; mean15k fails|
| [Expert scale100, fixed100k](results/2026-10-25-cdp-cheap-qbert.md#expert100-gives-a-smaller-further-gain-at-the-same100k-budget) |100k /24,939|18m59s|Greedy8365.625 /263.542; sampled7339.583 /182.292|24/24 greedy,22/24 sampled pyramids; mean15k fails,one reused learner|
| [Expert100,200k/R16](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-nearly-equal-updates-improves-qbert-with-higher-cost) |200k /24,970|26m07s|Greedy9654.167 /263.542; sampled7569.792 /182.292|23/24 greedy,24/24 sampled pyramids; mean15k fails,37.63% more time|
| [Expert100,12M48/B16/200k/R16](results/2026-10-25-cdp-cheap-qbert.md#larger-strongly-supervised-qbert-improves-greedy-score-with-higher-cost) |200k /12,477|30m27s|Greedy11984.375 /80.208; sampled6759.375 /177.083|24/24 pyramids both modes;mean15k fails,16.57% more time|
| [Expert100,12M48/B16/400k/R8](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r8-costs-more-and-lowers-primary-greedy-quality) |400k /12,489|40m35s|Greedy10042.708 /80.208; sampled7456.250 /177.083|23/24 greedy,22/24 sampled pyramids;mean15k fails,primary16.20% lower for33.28% more time|
| [Same, RGB64](results/2026-10-25-cdp-cheap-qbert.md#rgb64-costs-more-and-lowers-frozen-quality) |200k /49,939|22m20s|718.750 /180.208|0/24 pyramids; lower coverage|
| [Same, intrinsic coefficient100](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result) |200k /49,939|20m12s|809.375 /180.208|0/24 pyramids; lower coverage|
| [Same coefficient1, six actions](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage) |200k /49,939|18m21s|605.208 /148.958|0/24 pyramids; lower coverage|
| [Full18, more experience at R16](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result) |500k /62,470|26m53s|719.792 /180.208|0/24 pyramids;14.125/21 coverage|
| [Breakout4M/RGB64/R32](results/2026-10-26-cdp-cheap-breakout.md) |200k /49,939|22m30s|6.208 /1.375|0/24 two-wall successes|
| [Same, native actor mixture.01](results/2026-10-26-cdp-breakout-unimix.md) |200k /49,939|22m29s|7.083 /1.375|0/24 two-wall successes|
| [Full18, capped expert imitation](results/2026-10-26-cdp-breakout-pixel-reference.md#capped-native-imitation-lowers-frozen-breakout-quality) |200k /49,939|22m38s|4.000 /1.375|0/24; below full18 control|
| [Full18, longer expert cap1920](results/2026-10-26-cdp-breakout-pixel-reference.md#longer-guidance-lowers-frozen-breakout-quality) |200k /49,939|23m33s|2.625 /1.375|0/24; below cap640 control4.000|
| [Full18, corrective labels/expert10/20%/R16](results/2026-10-26-cdp-breakout-pixel-reference.md#corrective-labels-save-time-but-remain-below-unaided-breakout) |200k /24,970|17m25s|4.292 /1.375|0/24 two-wall;26.02% less time than old guidance,but below unaided7.083|
| [Retained corrective learner,greedy](results/2026-10-26-cdp-breakout-pixel-reference.md#greedy-corrective-breakout-gives-a-small-gain-but-remains-weak) |200k /24,970 retained;0 new|17m25s retained|5.000 /2.583|0/24 two-wall;small gain over sampled4.292|
| [Same, encoder rate4e-5](results/2026-10-26-cdp-breakout-pixel-reference.md#higher-encoder-rate-lowers-frozen-breakout-quality) |200k /24,970|17m41s|1.458 /1.375|0/24 two-wall;below original-rate4.292,1.52% more time|
| [Full18, corrective expert100/4M/B8/R16](results/2026-10-26-cdp-breakout-pixel-reference.md#stronger-imitation-improves-frozen-breakout-at-nearly-unchanged-cost) |200k /24,970|17m45s|Sampled9.458 /1.375; greedy12.208 /2.583|0/24 two-wall; one reused learner|
| [Same supervision,12M/B16/R16](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-capacity-improves-frozen-breakout-with-higher-cost) |200k /12,477|22m19s|Sampled13.208 /1.417; greedy17.875 /2.208|0/24 two-wall; higher scores for25.72% more time,one reused learner|
| [Same12M/B16,encoder2e-5](results/2026-10-30-cdp-breakout-encoder-moderate.md) |200k /12,477|22m27s|Sampled25.292 /1.417; greedy42.625 /2.208|0/24 two-wall;91.48%/138.46% higher scores for0.61% more training time;one reused learner|
| [Same12M/B16,encoder4e-5](results/2026-11-01-cdp-breakout-encoder4e5.md) |200k /12,477|21m28s|Sampled56.292 /1.417; greedy104.667 /2.208|0/24 two-wall;122.57%/145.55% higher scores than2e-5 for4.43% less training time;one reused learner|
| [Same, four actions](results/2026-10-26-cdp-breakout-minimal-actions.md) |200k /49,939|20m59s|9.542 /1.167|0/24 two-wall successes|
| [Four actions, coefficient1, actor/critic gradients](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain) |200k /49,939|20m41s|11.208 /1.167|0/24; modest unconfirmed gain|
| [Same, coefficient .01](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score) |200k /49,939|21m07s|5.042 /1.167|0/24; weaker frozen score|
| [Same, H5/500k target](results/2026-10-26-cdp-breakout-compact-h5-budget.md) |487,736 /121,873|45-minute cutoff|8.458 /1.375|500k incomplete;0/24 two-wall successes|
| [Retained Breakout12M/RGB64/H5 warm start](results/2026-10-26-cdp-breakout-short-imagination.md) |50k new /12,439 new|8m02s|27.208 /67.583 source|0/24 two-wall successes; regression|

Boxing's cheap AMD recipe now passes on two fresh learners, each within30 minutes;
this is useful confirmation, not broad seed reliability. The retained
same-seed1M/RGB64/500k/R32 control takes65m38s and scores71.625. Geometry, runtime
and GPU differ, so the2.31× wall-cost comparison does not isolate resolution.
Unaided Qbert's mean first-pyramid coverage improves5.83→16.63 of21 cubes, with a maximum
of20, but its task gate fails. Neither these results nor short-screen failures
establish a universal capacity or resolution rule.
Qbert's [completed coefficient100 test](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result)
keeps the200k budget and native model fixed. Frozen score stays809.375 with0/24
pyramids, while mean coverage falls16.625→13.708 of21 cubes. The measured late
bonus/extrinsic ratio rises0.146%→15.84%; that does not establish useful
exploration. All audits pass and service exits. No extension or broad sweep.

Breakout's smaller model gains modestly, with lower score per action than the
earlier T64 run but3.07× lower wall cost at the shared196,608-action point. One
selected frozen episode waits12061 actions after losing a life before choosing
FIRE. Training with native actor mixture.01 scores7.0833 and reduces the longest
selected no-FIRE gap12115→447. Adding the mixture only to the baseline frozen
weights scores6.0833, gap724, with zero updates. All task gates still fail;
shorter waits before FIRE are useful but not Breakout success. No unchanged extension.

Compact aided Freeway now passes in28m17s:31.0417 versus actual initial0.00,
24/24 versus0/24 crossing successes. Frozen evaluation disables the aid. The
same-protocol CPU always-UP reference scores21.6667,0/24. Resolution, ratio,
budget and native runtime differ from the short screen; no isolated attribution.
Focused seed4013 confirmation now passes at32.000 and24/24 successes in28m50s,
with the recipe and budget fixed. Both runs pass all audits and services exit.
Two useful learners do not establish broad seed reliability.
Pong's4M/RGB48/R32 test now completes500k actions and124939 updates in48m53s,
scoring11.0 with23/24 frozen wins versus initial−20.375 and0/24. The mean15 gate
fails despite useful learning. All audits pass. The larger update budget does
not isolate a replay-ratio effect or establish a hardware speedup; no extension.

The [retained Breakout2M policy on AMD Auto](results/2026-10-26-cdp-retained-breakout-auto.md)
now scores67.5833,0/24 two-wall successes, with zero updates. Its21,976-action
frozen check takes14.127s; all346 source/copied/frozen tensors match and all
audits pass. Only precision metadata changes. Historical98.2083 used another
GPU/precision/runtime, so the difference does not isolate precision. This is a
transferred-policy control for any separately declared warm start, not new
learning or a matched-budget comparison with the fresh compact200k policies.

The [completed H5 warm start](results/2026-10-26-cdp-breakout-short-imagination.md)
uses50k new actions/12,439 updates in8m02s but regresses67.5833→27.2083, both0/24
two-wall successes. The frozen source trace matches H15 before learning; all
audits pass and service exits. Replay/games/belief/RNG restart, so new experience
and replay restart confound isolated horizon attribution. Retain the source;
no extension. Counter-aware curves pass70 CPU checks without changing training.

The [Pong4013 confirmation](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation)
completes500k actions/124939 updates in50m14s: frozen7.333 and22/24 wins versus
own initial−20.333 and0/24. All audits and curves pass; service exits successfully.
Both compact learners win most matches but miss the unchanged mean15 gate.
Selected seeds are not held out; no unchanged extension or broad campaign.
Their [completed-report analysis](results/2026-10-25-cdp-cheap-pong.md#exploration-bonus-scale-in-the-completed-learners)
finds a late exploration-bonus/net-predicted-reward ratio of3.76/3.46. Signed
reward cancellation limits that comparison; it does not diagnose the score gap.
This motivates the selected smaller-bonus test below, without establishing that
the bonus caused either failed score gate.

The [six-action Qbert trial](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage)
completes200k/49939 updates in18m21s, but frozen605.208 and10.167/21 mean coverage
are below the full18 control's809.375 and16.625/21. Own initial148.958;0/24 pyramids.
All audits/curves pass and service exits. No extension. A separate
[CPU Breakout references](results/2026-10-26-cdp-breakout-pixel-reference.md)
score11.333 for position following and20.292 for fixed two-frame interception,
both0/24 two-wall successes. Full replays pass; neither supplies training guidance.

The [compact Breakout H5 result](results/2026-10-26-cdp-breakout-compact-h5-budget.md)
retains the45-minute cutoff at487,736 actions/121,873 updates. Frozen8.458 versus
own initial1.375,0/24 two-wall successes; the500k target is incomplete. Its shared
196,608-action/49,091-update prefix takes18m06s versus H15's22m06s,18.06% less wall
time on the same native/GPU/model. This cost saving and modest score gain over
the differently budgeted7.083 control do not establish task competence. All four
recovery checkpoints, numerical/accounting and frozen replay checks pass; the
original training guard/controller retain their time-budget failure. Both
services exit and the comparison is reviewed. No extension.

The [Pong coefficient.01 primary](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result)
completes500k/124939 updates in49m55s: frozen14.9583 and24/24 wins versus own
initial−20.375 and0/24. The coefficient1 control scores11.0/23 wins at the same
budget. All audits/curves pass, initial tensors/semantic evaluation traces match,
and service exits. The unchanged mean15 gate still fails; do not round it into
a pass. Training cost is similar. This result motivated the confirmation below.

The [smaller-bonus confirmation](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation)
completes500k/124939 updates in49m58s:14.0 and24/24 wins versus initial−20.333,
improving the same-seed coefficient1 control's7.333/22 wins. All audits and both
comparisons pass; service exits08:27:53 UTC. Both coefficient.01 learners still
miss mean15. These are reused development seeds, not broad reliability evidence.

The [four-action Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md)
finishes200k/49939 updates in20m59s,6.64% less training time than the full18
control. Frozen9.542 versus own initial1.167 improves on7.083/1.375, but still
0/24 two-wall successes. Changed dimensions/initialization require own controls;
no matched-weight or seed-reliability claim. All audits/comparison pass and
service exits09:01:55 UTC. Retain the modest gain; no extension selected.

The [Qbert500k/R16 result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result)
completes in26m53s: frozen719.792 versus initial180.208,0/24 pyramids and14.125/21
coverage, below the200k/R32 control809.375/16.625. Experience increases2.5× and
updates1.251× in1.36× training time, but this joint change does not improve the
recipe. All audits/comparison pass; service exits09:33:03 UTC. No extension.

The [bounded PongB16 check](results/2026-10-25-cdp-cheap-pong.md#b16-bounded-check-passes-with-lower-early-wall-cost)
passes16,384 actions/2,002 updates in72.224s versus B8's94.054s,23.21% less early
wall time. Full config, matched initial tensors, weight changes, ordinary restore
and346-tensor invariance pass; service exits09:38:03 UTC. This is not a full-budget
throughput measurement or natural-match competence result.

The [full-budget PongB16 result](results/2026-10-25-cdp-cheap-pong.md#b16-at-the-fixed500k-budget)
completes500k/62,454 updates in38m45s,22.37% less training time than B8. Frozen
11.875 versus initial−20.375 retains24/24 wins but is below B8's14.958 and misses
mean15. Full audits/curves/comparison pass; service exits10:28:51 UTC. The online
curve develops earlier in wall time, but fixed frozen quality is lower. No
extension or seed confirmation is selected for this recipe.

The [bounded Qbert12M/RGB48/B16 check](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-rgb48-bounded-check)
passes16,384 actions/2,002 updates in149.387s, versus93.813s for4M/B8. All checks
and ordinary restore/346-tensor invariance pass; service exits10:35:06 UTC. This
measures execution/early cost, not competence or a full-budget speed ratio.

The [completed Qbert capacity/batch screen](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-the-fixed200k-budget)
scores959.375 versus own initial165.625, with0/24 pyramids. At200k actions,
12M/B16 takes32m19s versus4M/B8's19m45s; coverage16.292/21 is below16.625/21.
All audits and completed comparison pass; service exits11:15:13 UTC. The score
increase does not justify63.70% more wall time for this compute-constrained
screen. No extension or confirmation selected.

The [completed Pong doubled-rate test](results/2026-10-25-cdp-cheap-pong.md#doubling-b16-learning-rates-lowers-the-frozen-score)
scores9.125 versus the B16 control's11.875, both24/24 wins and below mean15.
Training38m32s is essentially unchanged from38m45s. All audits, initial-tensor/
semantic-trace equality and completed comparison pass; service exits11:59:15 UTC.
Retain the original rates; no extension or seed confirmation selected.

The [completed Qbert training-aid test](results/2026-10-25-cdp-cheap-qbert.md#persistent-training-aid-does-not-improve-frozen-coverage)
scores678.125 versus809.375 without aid; mean cube coverage15.125 versus16.625,
both0/24 pyramids. Training20m17s versus19m45s, with10,039 verified overrides.
All audits and completed comparison pass; service exits12:24:16 UTC. Retain the
unaided recipe; no extension or confirmation selected.

The [completed PongH5 test](results/2026-10-25-cdp-cheap-pong.md#shorter-imagination-saves-time-but-lowers-the-frozen-score)
finishes500k/124,939 updates in38m07s,23.65% less training time than H15. Frozen
8.667 is below14.958, both24/24 wins and below mean15. All audits, matched initial
tensors/traces and comparison pass; service exits13:18:17 UTC. Retain H15; no
extension or conditional4013 confirmation is selected.

The [completed smaller-bonus Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score)
finishes 200k actions/49,939 updates in 21m07s, but lowers frozen score
9.542→5.042, both 0/24 two-wall successes. All audits and comparison pass after
copying the omitted, verified replay manifest and rerunning only the unchanged
CPU audit. The original service failure at 13:43:51 UTC and incomplete watcher
receipt are retained; no GPU work repeats. Retain coefficient 1; no extension.

The [Pong B16/R40 run is interrupted](results/2026-10-25-cdp-cheap-pong.md#host-restart-interrupts-b16r40)
after a host restart:265,440 complete logged actions/41,416 finite reports,
saved200k/31,192-update checkpoint. Its separate frozen assessment scores−9.708
versus initial−20.375, both0/24 wins. All frozen checks pass after a CPU-only
checker-import correction; original failures/unsealed tails remain. The500k
quality question is unresolved. No resumed training or extension.

The [completed Qbert H30 test](results/2026-10-25-cdp-cheap-qbert.md#longer-imagination-costs-more-and-lowers-frozen-coverage)
finishes200k actions/799,441 frames/49,939 updates in26m43s,35.34% more training
time than H15. Frozen582.292 versus809.375 and coverage9.583 versus16.625/21
are worse; both0/24 first pyramids. Retain H15. Final training/frozen/replay/
guard/comparison checks pass, including346 matched initial tensors and identical
semantic initial traces. The declared100k recovery was never written because the
pinned queue saved every200k. Original watcher failures remain; separate CPU
completion records `protocol_complete=false`. Service exits16:26:25 UTC. No GPU
repeat, extension or confirmation for H30.

The [completed six-action Pong test](results/2026-10-25-cdp-cheap-pong.md#six-actions-save-time-but-lower-frozen-quality)
finishes500k/124,939 updates in44m21s,11.17% less training time than full18.
Frozen10.375 versus own initial−20.042 is below full18's14.958; both24/24 wins
and below mean15. All numerical/config/counter/tensor/replay/guard checks,
four recoveries and curves/comparison pass; service exits17:28:27 UTC. Dimensions
and initial weights differ, and the control used the older kernel. Retain
full18; no extension or conditional4013 confirmation is selected.

The [bounded Breakout actor/critic check](experiments/2026-10-26-cdp-breakout-actor-critic-gradients.md)
passes4096 actions/963 updates in22.474s and1024 frozen actions in0.489s. All357
tensors preserve frozen invariance; all963 initial policy/value loss reports
are nonzero, CNN/reward weights change and346 shared initial tensors match the
control. Both guards pass; service exits17:34:23 UTC. This verifies execution,
not useful task learning or gradient correctness. Native unchanged; the exact
313-test CLI patch is applied as runner`fbdaee3`.

The [completed Breakout actor/critic test](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain)
finishes200k/799,187 frames/49,939 updates in20m41s, with frozen11.208 versus
9.542 and own initial1.167; both0/24 two-wall successes. All357-tensor/frozen,
100k recovery, numerical/replay/guard checks and matched346 shared initial
weights/semantic traces pass. Main service exits18:05:14 UTC. A separate CPU
comparison import failure is retained; corrected comparison passes without
repeating GPU work. This is a modest unconfirmed gain, with an older-kernel
control; no extension, seed confirmation or global default change is selected.

The [completed QbertRGB64 test](results/2026-10-25-cdp-cheap-qbert.md#rgb64-costs-more-and-lowers-frozen-quality)
finishes200k/799,444 frames/49,939 updates in22m20s,13.14% more training time
than RGB48. Frozen718.750 versus809.375 and coverage15.083 versus16.625/21 are
lower; both0/24 first pyramids. All numerical/config/counter/346-tensor/replay/
guard/100k recovery/curve/coverage/comparison checks pass; service exits18:39:49
UTC. Retain RGB48; no extension or confirmation. Geometry, parameter shapes,
initialization and old/new kernel/module limit attribution.

The [completed Pong greedy-action assessment](results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-does-not-improve-the-retained-pong-policy)
scores14.333 versus sampled14.958, both24/24 wins and below mean15. Own initial
greedy−20.833/0 wins. Final/initial77,640/20,288 actions and310,513/81,145 frames
in29.564/7.757s, zero learning,24 natural matches each. All346 frozen tensors,
source files, config/counters, complete replays/videos and both guards pass;
service exits18:44:25 UTC. Retain sampled evaluation; no greedy confirmation.
The retained learner's49m55s training cost remains. The five-game goal is unfinished.

The [bounded PongB16 actor/critic check](experiments/2026-10-26-cdp-pong-actor-critic-gradients.md)
passes4096/466 updates in16.126s and1024 frozen actions in0.436s. All357 tensors
remain unchanged,346 shared initial tensors match the control, all466 initial
policy/value reports are nonzero and both guards pass. Service exits18:51:00
UTC. Native/CLI unchanged; this is execution evidence, not useful learning.

The [completed PongB16 actor/critic comparison](results/2026-10-25-cdp-cheap-pong.md#b16-actorcritic-gradients-lower-frozen-quality)
finishes500k/62,454 updates in37m52s but lowers frozen11.875→−0.542 and wins
24/24→10/24. Own initial−20.375/0 wins. All357-tensor/replay/guard/recovery/
curve/comparison checks,346 shared initial tensors and semantic traces pass;
service exits19:36:47 UTC. Retain disabled gradients; no extension or confirmation.
The separate progress observer was reattached without restarting training.

The [CPU-only Qbert pixel rule and prepared adapter](results/2026-10-25-cdp-cheap-qbert.md#pixel-only-cpu-rule-clears-the-first-pyramid)
clear24/24 first pyramids, mean3764.583 and21/21 coverage. Collection and exact
replay each take about30s; no agent or learning. This hand-coded candidate
still misses mean15000 and does not establish unaided Kindle transfer.
The optional episode-guidance adapter passes CPU tests and full pixel/action
replay. The bounded native check completes4096/963 in22.541s and1024 frozen
restore in0.452s;512 verified overrides,346 matched initial/unchanged frozen
tensors, all finite reports and both guards pass. Its service fails19:40:14 UTC
only in the new CPU reader, which incorrectly rejects training intrinsic rewards.
A training-aware correction passes421 tests and the saved trajectory replay;
original failure retained, no GPU repeat.

The fresh Qbert pixel-guided comparison completes 200k actions/49,939 updates
in 20m13s, but lowers unaided frozen score 809.375→633.333 and first-pyramid
coverage 16.625→11.667/21; both have 0/24 pyramids. The 5% episode rule actually
overrides 23,428 actions (11.714%). All audits, 100k recovery, 346 matched initial
tensors/semantic traces, complete pixel/action replay and comparison pass;
service exits 20:07:01 UTC. Retain unaided; no extension or confirmation.
See [the result](results/2026-10-25-cdp-cheap-qbert.md#pixel-guided-training-lowers-unaided-frozen-quality).

Pong RGB32 completes500k/124,939 updates in42m19s,15.24% less training time
than RGB48, but lowers frozen14.958→2.292 and wins24/24→16/24. All numerical,
config/source/counter/346-tensor/replay/video/guard/recovery/comparison checks
pass; service exits21:02:34 UTC. Retain RGB48; no extension or confirmation.
Geometry/initialization, older control kernel and13.690s unintended integrated-
GPU inference during this run limit attribution. Full evidence is in cheap-Pong.

Qbert cap640 completes200k actions/799,403 frames/49,939 updates in20m00s.
Frozen750.000 improves whole-episode guidance633.333 but stays below unaided
809.375; all0/24 pyramids. Coverage16.208/21 versus11.667 and16.625. Actual
training overrides15,360 (7.680%);23 completed guided episodes clear a pyramid,
none of366 unguided episodes does. All audits,100k recovery,346 matched initial
and unchanged frozen tensors, semantic initial traces, full training pixel/
action replay and comparison pass; service exits21:30:21 UTC. Retain unaided.
No extension or confirmation follows capped Qbert. The applied
640-action option passes444 CPU tests; native unchanged. Use the matching
`runtime-qbert-guidance-cap` overlay. Prior isolation failures remain recorded.
See [the completed result](results/2026-10-25-cdp-cheap-qbert.md#capped-guidance-recovers-some-quality-but-remains-below-unaided-training).

Pong B16 behavior rate8e-5 completes500k actions/1,999,751 frames/62,454
updates in37m38s. Frozen12.375/24 wins improves original B16's11.875 slightly,
but trails B8's14.958; all miss mean15. All audits, four recoveries,346 matched
initial/unchanged frozen tensors, semantic initial traces, replays, guards and
comparison pass; service exits22:20:58 UTC. Retain B8 as the strongest compact
Pong recipe; no extension or seed confirmation follows this small gain.
The CPU startup import failure remains recorded; no GPU stage repeated.
CLI`a2c467c` passes460 CPU tests; native unchanged. See
[the completed comparison](results/2026-10-25-cdp-cheap-pong.md#behavior-rate-doubling-gives-a-small-gain-but-still-misses-the-gate).

QbertRGB48 frozen forecasts complete on15,376 recorded actions/61,462 frames
per model in50.330s trained/49.268s initial, with zero updates. One-step reward
AUC improves.500→.969; H15 reward-sum MAE22.487→12.400. All346 frozen tensors,
8 matched stream trajectories/pixel hashes, source/counter/support/guard checks
pass; service exits22:30:05 UTC. This shows predictive learning on the reused
cohort, not the cause of weak policy transfer. Retain4M48/H15; no new training
or automatic next allocation. The geometry helper passes24 CPU tests; native
unchanged. See [forecast results](results/2026-10-25-cdp-cheap-qbert.md#compact-qbert-learns-reward-forecasts-despite-weak-task-performance). Check live state before GPU work.


Pong12M48/Auto/N8/B16/T16/H15/R16 completes500k actions /31,227 updates in
41m28s,16.95% less training time than4M/B8/R32. Frozen15.625/24 wins versus
own initial−20.292/0 meets mean15/90% on one learner. All audits, four recoveries,
346 frozen tensors,42 pins, recorded-action replays, guards and comparison pass;
service exits01:51:06 UTC, CPU postprocessing01:51:35 on October27.
No live pixel hashes are available in ordinary unguided logs. See the protocol
scope correction and [completed result](results/2026-10-25-cdp-cheap-pong.md#larger-capacity-with-less-replay-work-reaches-the-gate).

Pong12M48 fresh4013 completes500k actions /1,999,720 frames /31,227 updates
in41m31s. Frozen14.833/24 wins versus own initial−20.542/0 narrowly misses
mean15; primary1009 scores15.625/24 wins in41m28s. Both improve their paired
4M/B8 controls14.000/14.958 with about17% less training time, but only one of
two learners passes the full gate. All numerical/config/counter checks, four
recoveries,346 frozen tensors,53 pins, replays, guards and comparison pass.
Service exits02:41:08 UTC, CPU postprocessing02:41:23 on October27. Original
progress-observer failure is retained; no GPU stage repeats. Both reused
seeds are development evidence. No extension or third seed is selected.
See [completed confirmation](results/2026-10-25-cdp-cheap-pong.md#capacity-confirmation-improves4013-but-narrowly-misses-the-gate).

Breakout expert200k completes798,964 frames /49,939 updates in22m38s,
frozen4.000 versus initial1.375, below full18 control7.083/minimal4 9.542;
all0/24 two-wall successes. All audits,100k recovery,346 matched initial/frozen
tensors,46 pins, replays, guards, comparison and exposure pass. Service exits
03:16:09 UTC; postprocessing03:16:14. Retain controls; no extension/confirmation.
20,463 trusted actions (10.2315%); guided/unguided completed means24.871/2.948
are not frozen competence. See [result](results/2026-10-26-cdp-breakout-pixel-reference.md#capped-native-imitation-lowers-frozen-breakout-quality).

[Breakout frozen movement analysis](results/2026-10-26-cdp-breakout-pixel-reference.md#frozen-movement-agreement-does-not-track-task-quality)
finds expert visible-ball teacher agreement40.32% versus initial32.87%, but
frozen4.000 remains below full18/minimal4 controls7.083/9.542. Near-impact
predicted landing distances36.60/26.88/15.42 pixels also differ across those
trajectories; teacher advice is not ground-truth action accuracy. Do not choose
an imitation-weight increase from falling loss alone. CPU-only replay passes
all recorded outcomes and56 active pins; no new Breakout allocation is selected.

[Longer saved Breakout prefixes](results/2026-10-26-cdp-breakout-pixel-reference.md#longer-saved-prefixes-expose-most-teacher-reward)
retain96.19% of full-reference reward at1920 actions versus14.52% at640.
This is a later-exposure hypothesis from reused CPU reference logs, not learned
transfer or a diagnosis of the failed4.000 native result. No Breakout allocation
is selected; the full teacher itself still has0/24 two-wall successes.

Qbert continuation1009 completes200k actions /799,538 frames /49,939 updates
in26m38s. Frozen3137.500/17of24 first pyramids versus own initial180.208/0
improves old-teacher1083.333/2; continuation4013 scores4730.208/23 in25m23s.
Both seeds improve, neither meets mean15000;1009 also misses90% pyramids.
All audits,100k recovery,346 matched initial/unchanged frozen tensors, paired
initial semantic traces,60 pins, replays, guards, curves, coverage, exposure and
four-arm comparison pass. Main service exits04:47:41 UTC; postprocessing04:47:57
on October27. Actual1009 guidance38,400 actions (19.2%); online guided returns
are not frozen competence. Both reused development seeds; no broad reliability,
extension or third seed. See the paired continuation result in cheap-Qbert.

The Breakout cap1920 bounded check completes16,384 actions/65,473 frames/
4035 updates in114.327s;1024 frozen/4096 frames/.558s, zero updates and no
natural frozen episodes. All32 pins,346 matched initial/unchanged frozen tensors,
finite reports, CNN/reward/actor/value changes,10,125 trusted actions/3829
supervised reports, exact guided replay and guards pass. Service exits05:38:24
UTC on October27. A copied4096-action CPU exposure assumption is repaired;
original failure retained, no GPU repeat. Natural ends/late starts leave0 released
actions in this check; do not claim cap-release coverage or competence.

Breakout cap1920 completes200k actions /798,815 frames /49,939 updates in
23m33s. Frozen2.625 versus own initial1.375 is below cap640's4.000; both0/24
two-wall successes. Training costs4.02% more. All audits,100k recovery,346
matched initial/unchanged frozen tensors, paired initial semantic traces,47 pins,
full replays/videos, guards, curves, comparison and exposure pass. Main exits
06:07:00 UTC; postprocessing06:07:30 on October27. Actual guidance46,833 actions
(23.4165%); guided online mean169.700 is not frozen competence. Retain controls;
no extension or confirmation. See the completed longer-guidance result in the
Breakout pixel-reference report.

Qbert continuation greedy evaluation completes on both retained learners.
1009 scores3058.333/16of24 first pyramids versus sampled3137.500/17;4013
scores3770.833/19 versus4730.208/23. Own greedy initials85.417/263.542,0pyramids.
Both greedy arms fail mean15000/90%; retain sampled actions for Qbert.
Four arms total65,104 actions /260,268 frames /30.277s evaluation, zero new
training/updates; entire service1m17.712s. Starts06:12:49 UTC/source`62482da`,
main exits06:14:07, postprocessing06:14:10 on October27. All60 pins, source
invariance,346 invariant tensors per arm, full outcome replays/videos and
host guards pass. Original26m38s/25m23s training costs and sampled results
remain. No retry, extension, third learner or automatic successor selected.
That frozen assessment is terminal. See the completed greedy comparison in
cheap-Qbert. The five-game goal remains unfinished.

Optional native expert-world gradients pass two guarded derivative/alignment
checks and a real Qbert8192-action/1987-update check in80.263s; frozen1024/.489s,
zero updates/no natural episodes. All357 finite/invariant tensors,346 shared
initial tensors,35 pins, full guided replay and guards pass.4520 trusted actions,
1978 supervised reports; max world/actor expert-loss difference1.5e−7.
Original service fails06:31:56 UTC only at the CPU schema validator; a separate
CPU-only repair completes audits without repeating GPU work or changing pins.
129 Rust CPU tests,474 Python tests and strict Clippy pass. Native0e157df,
SHA6a9c41c; default off. Use`/x/Temp/kindle-expert-world-20261027/runtime-audited`
for future work; its packaging receipt pins native/wheel/Python source. The2.6GiB
build cache is released; wheels/test executable and all evidence remain.
Qbert4013/4M48/R16/20% with expert world gradients completes200k actions /
799,662 frames /24,970 updates in21m28s. Frozen3670.833/19of24 first pyramids
versus own initial182.292/0 improves matched R16 control3333.333/18 by10.13%
with2.32% less measured training time, but remains below5%/R32's4730.208/23.
All audits,100k recovery,357 invariant tensors,346 shared initial matches,
paired initial semantic traces,52 pins, full replays, guards and comparison pass.
Main succeeds15:21:35 UTC on October28; original CPU postprocessing fails at
15:22:04 on an eager ALE import. A separate lazy-import repair completes at
15:24:27; original failure/pins remain, no GPU repeat. Actual guidance46.088%.
Retain5%/R32 as stronger; small reused-seed R16 gain gets no extension or
confirmation. New native/runner identities limit attribution. See the completed
expert-world result in cheap-Qbert and its October28 protocol. A separate
12M48/B16/R16/20%/world-gradient check completes8192 actions/32,756 frames/489
updates in60.333s;1024 frozen/.810s, zero updates/no natural episodes. All41
pins,357 finite/invariant tensors,346 shared12M initial tensors, parameter
changes,489 supervised reports, full guided replay and guards pass.4492 trusted
actions; max world/actor expert-loss difference1.4e−7. Service succeeds15:49:49
UTC on October28. Original prelaunch first384/0 assumption and unselected-service
failure remain; reviewed first376/residual.5 passes. No GPU stage repeats.
Use`/x/Temp/kindle-qbert-expert-capacity-20261028/reviewed`. Early200k cost
projects24.55min, not competence or measured full cost. Model/batch change
together. The separate fresh200k comparison completes799,693 frames/12,477
updates in24m51s. Frozen1792.708/9of24 versus own initial177.083/0 is51.16%
below same-native4M/B8 control3670.833/19, at15.74% more training cost. All63
pins,100k recovery,357 initial/frozen tensors,finite reports,parameter changes,
full replays,guards,curves,coverage,exposure and comparison pass. Main succeeds
16:40:13 UTC,postprocessing16:40:29 on October28. Actual guidance46.411%.
Retain4M; no extension or confirmation. Model/batch change together on reused4013.
Full root`/x/Temp/kindle-qbert-expert-capacity-20261028/full`; see the completed
expert-capacity result in cheap-Qbert. That life is terminal; the separately
selected corrective-label check is below. Check live state before GPU work.

Independent corrective labels are optional and separately validated: native
447fe50/SHA35ad0db,130 Rust CPU/491 packaged Python tests,strict Clippy,a5.87s
native diagnostic and8192/994 bounded Qbert check in61.976s. Runtime
`/x/Temp/kindle-corrective-labels-20261028/runtime`;2.6GiB cache released.
Fresh200k corrective training completes799,682 frames/24,970 updates in25m57s,
frozen3633.333/17of24 pyramids versus own182.292/0 and control3670.833/19.
Training costs20.86% more. All52 pins,100k recovery,357 matching initial/unchanged
frozen tensors,paired initial traces,full label/pixel/action replays,guards,
curves,coverage,exposure and comparison pass. Main exits18:06:30 UTC,CPU
postprocessing18:07:02 on October28; no GPU repeats. Display-only caption
correction preserves original data/pins. Retain control; no extension/confirmation.
200k labels include103,712 passive states;96,288 overrides are48.144%. Late
passive teacher/learner NOOP frequencies79.447%/46.077% suggest checking wait-label
weight, not a causal diagnosis. Root`/x/Temp/kindle-qbert-corrective-full-20261028`.
See the corrective-labels protocol/completed cheap-Qbert result.

The R32/wait-weight1/100k recipe completes100k actions/399,864 frames/24,939
updates in18m58s,26.90% less training time but37.79% lower frozen mean than the
corrective200k control:2260.417/12of24 first pyramids versus3633.333/17. Own
initial182.292/0; both original15k/90% criteria fail. All55 pins,50k recovery,
357 matched initial/invariant frozen tensors,identical paired initial traces,
finite reports,equal row/weight sums,full replays/videos,guards,curves,coverage,
exposure and comparison pass. Main succeeds18:53:59 UTC,postprocessing18:54:18
on October28. Actual overrides47,072 (47.072%); guided returns are not competence.
Original CPU observer argument failure is retained; no GPU stage repeats.
Retain controls,including stronger5%/R32/world-off4730.208/23. Weight,ratio and
budget change together on reused4013/cohort; no isolated attribution,extension,
confirmation or automatic successor. Root`/x/Temp/kindle-qbert-wait-full-20261028`.
The host is idle at this milestone. See the completed wait-labels result and
check live state before GPU work.

The retained100k/wait1 greedy assessment improves2260.417/12→4965.625/21of24
first pyramids, own initial263.542/0; both original15k/90% criteria still fail.
Zero new training/updates;41,976 actions/167,838 frames/19.462s evaluation,
whole service46.404s, exits19:03:28 UTC; postprocessing19:03:31 on October28.
All47 pins/55 source-training inputs,357 invariant tensors per arm,replays/videos
and guards pass. Retain greedy as a useful18m58s development option on this one
reused learner, not a global default or independent confirmation. No extension,
learner confirmation or automatic successor. See the completed wait-greedy result.

Qbert expert scale10 completes100k actions/399,860 frames/24,939 updates in
18m55s versus scale1's18m58s. Greedy7700/24of24 first pyramids improves
4965.625/21; sampled5572.917/24 improves2260.417/12. Own sampled/greedy initials
182.292/263.542,0pyramids. Both still fail mean15000. All84 pins,50k recovery,
357 matched initial/invariant frozen tensors,paired sampled/greedy initial traces,
finite reports,full training label/pixel/action and frozen outcome replays,
guards,curves,coverage,exposure and comparison pass. Main succeeds19:42:39 UTC,
postprocessing19:42:59 on October28; observers also succeed. Only expert scale
changes, affecting actor/world supervision together; actual overrides42.147%.
Retain scale10/greedy as the stronger100k development recipe on reused4013;
no extension, second learner or successor is selected. No allocation is active
at this milestone. See the completed expert-scale10 result in cheap-Qbert and
its protocol; check live state before GPU work.

Expert10 seed1009 completes100k actions/399,823 frames/24,939 updates in19m11s.
Greedy3832.292/23of24 first pyramids versus own85.417/0 trails4013's7700/24 in
18m55s; sampled2667.708/15 versus own180.208/0 trails5572.917/24. Both greedy
learners meet90% pyramids,neither mean15000. All115 pins,50k recovery,357 finite
initial/invariant frozen tensors,346 shared legacy1009 tensors,matching sampled
initial semantic trace,full replays/videos,guards,curves,coverage,exposure and
comparison pass. Main succeeds20:16:56 UTC,postprocessing20:17:25 on October28;
observers also succeed. Actual overrides40.745%; guided returns are not competence.
Both reused development seeds; no matched100k scale1 control on1009 or broad
reliability. No extension,third learner or automatic successor selected. Host idle
at completion; retain both results and check live state before GPU work.
See the expert10-seed1009 protocol; check live state before GPU work.


The fixed Breakout CPU edge correction scores225.083 versus166.167 on24 reused
natural episodes, both0/24 two-wall successes. All52,072 actions/208,254 frames,
policy/debug/input-output pixels/outcomes replay exactly;2051 earlier-eight
prefix rows match. Service succeeds19:21:19 UTC on October28, whole1m07.715s;
all11 inputs and84 active Qbert pins remain unchanged. This is reference
development, not native learning or independent confirmation; no training aid
is selected. See [the expanded assessment](results/2026-10-26-cdp-breakout-pixel-reference.md#fixed-edge-correction-improves-the-expanded24-episode-reference).





Qbert R16/20% completes200k actions /799,662 frames /24,970 updates in21m59s,
13.41% less training time than5%/R32, but frozen3333.333/18of24 first pyramids
is below4730.208/23. Own initial182.292/0; both miss mean15000. All audits,
100k recovery,346 matched initial/unchanged frozen tensors, paired initial
semantic traces,47 pins, full replays, guards, curves, coverage, exposure and
comparison pass. Main exits05:23:55 UTC; postprocessing05:24:19 on October27.
Actual guidance91,913 actions (45.9565%); online guided scores are not frozen
competence. Retain5%/R32; no extension or confirmation. Probability and ratio
change together on reused4013; no isolated causal claim. See the completed
higher-exposure comparison in cheap-Qbert.

Pong12M greedy evaluation completes on both retained learners:1009/4013 each
score15.208/24of24 wins, versus own initial−20.583/−20.625 and0wins. Both pass
the original mean15/90%-win gate under changed greedy inference; original
sampled15.625/14.833 and one-of-two outcome remain. Four arms total202,768
actions/810,927 frames/111.916s of evaluation, zero new training/updates;
main service3m33s, exits05:29:46 UTC, postprocessing05:29:51 on October27.
All58 pins,346 invariant tensors per arm, full replays/videos and guards pass.
Retain greedy as a useful12M development option, not a global default or new
independent learner evidence. Both means are only.208 above15; uncertainty
intervals cross15. Original training costs41m28s/41m31s remain. No extension,
retry, third learner or successor allocation selected. That assessment is terminal. See the paired greedy result in cheap-Pong;
check live state before GPU work.

The R16/20% bounded check completes8192 actions/32,752 frames/994 updates in
52.742s, frozen1024/.524s with346 invariant tensors and no completed natural
episodes. All33 pins,346 matched initial tensors, finite reports, changed
CNN/reward/actor/value,4527 trusted actions/990 supervised reports, exact guided
replay and guards pass; service exits04:52:23 UTC on October27. No GPU stage
repeats or check experience is reused. This is execution/early-cost evidence,
not competence. Protocol remains in the continuation experiment.

Qbert continuation200k completes799,570 frames /49,939 updates in25m23s.
Frozen4730.208/23of24 first pyramids versus own initial182.292/0 improves
old-teacher200k's3153.125/17 by50.02% for26.17% more training time. It also
beats500k/R16's3752.083/21 with10.45% less time. Mean15000 still fails.
All audits,100k recovery,346 matched initial/unchanged frozen tensors, identical
initial semantic traces,56 pins, replays, guards, curves, coverage and exposure
pass; main service exits04:07:44 UTC, CPU postprocessing04:08:08 on October27.
Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
5%episode continuation/cap1920/expert1/NOOPweight.05 as a promising development
recipe. Actual guidance34,468 actions (17.234%); guided online scores are not
unaided competence. Teacher and cap change together on reused4013; no isolated
causal effect or seed reliability claim. No extension of this completed life.
See [completed result](results/2026-10-25-cdp-cheap-qbert.md#continuation-guidance-improves-frozen-learning-but-misses-the-full-gate).
Check live state before GPU work.

Qbert CPU continuation reference scores22050/24 first pyramids, all24 above15k,
with exact input/output/reset pixel replay; not native learning or held-out
confirmation. The exact456-test optional adapter is applied after terminal
Breakout audits;24,733-action parity passes, original first-pyramid AST/native
unchanged. Native bounded8192/1987 check completes32,739 frames in70.493s,
frozen1024/.531s with346 invariant tensors. All32 pins,346 matched initial
weights, finite reports, parameter changes,2987 trusted labels/1968 nonzero
expert reports, full guided replay and guards pass; service exits03:23:09 UTC.
Configuration4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original
rates,4013/eval4000004013,25%episodes/cap640/expert1/NOOPweight.05.
No frozen natural episodes complete in the bounded check; do not infer task
competence from it. See [result](results/2026-10-25-cdp-cheap-qbert.md#native-continuation-check-passes)
and [protocol](experiments/2026-10-27-cdp-qbert-continuation.md).
Check live state before GPU work.


CPU Breakout predictive reference scores166.167 across24 natural episodes,
0/24 two-wall successes; full input/output pixel-hash replay passes. Selected
after one9001 and eight reused evaluation episodes; those8 are included in24.
This reference work preceded the separately selected bounded adapter check.
55.935s affined CPU work
co-hosted with Pong; all42 active pins unchanged. See [reference result](results/2026-10-26-cdp-breakout-pixel-reference.md#calibrated-paddle-response-improves-the-reference-but-misses-the-two-wall-gate).

Fresh Qbert4013 expert500k/R16 completes **500k actions /1,998,773 frames
/62,470 updates in28m20s**. Frozen **3752.083 /21 of24 first pyramids** versus
own initial182.292 /0 improves200k/R32's3153.125 /17, but costs40.88% more
training time for19.00% higher mean. Both miss the original15k/90% gate.
All audits, four recoveries,346 matched initial/unchanged frozen tensors,
identical initial semantic traces, full pixel/action replays, guards, curves,
coverage, exposure and comparison pass. All67 declared pins stay unchanged.
Service exits00:48:34 UTC on October27; postprocessing completes00:48:42.
Keep both measured budget points; no extension or seed confirmation is selected.
This completed comparison reuses4013 after its stronger result,
changes budget and ratio together, and is not independent seed confirmation.
CPU teacher prototypes remain external and unselected. See
[completed result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-improves-first-pyramid-success-with-higher-cost)
and [protocol](experiments/2026-10-27-cdp-qbert-expert-r16.md).
Check live state before GPU work.

The fresh Qbert4013 expert-imitation check completes 200k actions /799,532
frames /49,939 updates in **20m07s**. Frozen **3153.125 /17 of24 first pyramids**
versus own initial182.292 /0; coverage20.167 /21. Seed1009 scores1083.333 /2
in20m11s with the same recipe, so both improve but vary substantially. Retain
4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18, original rates,
5% episode guidance/cap640/expert1/NOOPweight.05. All audits, 100k recovery,
346 initial/frozen tensors, full pixel/action replay, guards, comparison and
CPU tile replay pass; all50 declaration pins stay unchanged. Training uses
11,520 trusted actions (5.760%); frozen evaluation has no guidance.
Service exits23:53:38 UTC on October26; postprocessing23:54:08, tile replay23:54:40.
Both learners miss the mean15000/90%-pyramid gate. No paired unaided Qbert4013
control exists. Both completed lives are terminal; do not extend either life.
See [completed seed check](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-transfers-on4013-with-substantial-seed-variation)
and [protocol](experiments/2026-10-26-cdp-qbert-expert-seed4013.md).
Check live state before GPU work.

Independent CPU replay of1009 finds17/24 frozen episodes peak at19–20 cubes;
every episode's ever-lit count equals its simultaneous maximum. On4013 only3
peak at19–20, and17 clear all21. The corner-omission pattern varies by learner.
An unchanged pixel teacher recovers all8 recorded200-action learner prefixes
within640 further actions, but all trajectories remain partial. No delayed-
guidance implementation or training is selected. See
[diagnostic evidence](results/2026-10-25-cdp-cheap-qbert.md#frozen-expert-policy-leaves-individual-cubes-untouched).

The fresh Qbert expert-imitation comparison completes200k actions/799,509
frames/49,939 updates in20m11s. Frozen1083.333/2of24 first pyramids improves
unaided809.375/0 and capped guidance750.000/0; coverage18.625/21 versus16.625
and16.208. Own initial180.208/0. All audits,100k recovery,346 initial/frozen
tensors, matched initial traces, full pixel replay, guards and comparison pass;
service exits23:23:23 UTC.5% episode guidance/cap640 actually labels14,080
actions (7.040%); expert scale1/NOOPweight.05. Native79c9a7f/runtime
`/x/Temp/kindle-expert-20261026/runtime`. The completed fresh-seed check above
also improves over its own initial; the mean15000/90% gate still fails.
The completed1009 service is terminal; do not restart or extend it.
See [completed result](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate).

Optional native expert imitation passes127 Rust CPU tests/strict Clippy,
477 packaged Python tests/binding Clippy, analytic GPU gradients and an actor-
only supervised update. Fresh Qbert4096 actions/16,378 frames/963 updates take
22.662s;1024 frozen actions take.452s,346 matched initial/unchanged frozen tensors,
512 verified expert actions (38 movement),611 nonzero auxiliary reports and full
pixel replay/guards pass. Service exits22:49:20; strengthened actor-only check
exits22:51:05 UTC. Native`79c9a7f`, isolated runtime
`/x/Temp/kindle-expert-20261026/runtime`; original native retained. Scale1,
NOOPweight.05,5% episode guidance/cap640; no natural-episode competence measured.
Build cache released after packaging. The bounded checks are complete; the
completed full comparison is described above.
See [bounded result](results/2026-10-25-cdp-cheap-qbert.md#native-expert-imitation-smoke-passes-task-benefit-remains-unmeasured).

New boot`a8648671-8577-4f42-9b46-a4345d3ac297`, kernel7.0.0-38, AMD module
`9BDBC866F55BF296F1B7FA2`; same RX7900XT/Mesa26.0.8/native binary. Host scan,
512-action restore/346-tensor invariance and natural frozen evaluation pass.
No host recovery performed. Qbert's refreshed queue uses this identity; its H15
control used7.0.0-34, so record the kernel change as a comparison confound.
Check live service state before GPU work. Whole-episode and capped Qbert
guidance, Pong RGB32, Pong behavior-rate learning and their postprocessing are
complete, including the fixed frozen Qbert forecast comparison. The bounded expert checks also complete as described above. Do not restart
completed services.

Earlier context comparisons are complete: [AMD B2/T64 F16](results/2026-10-25-cdp-amd-takeover.md)
finishes2M in11h40m, frozen500k/final/initial14.71/10.04/1.42. The
[Auto500k comparison](results/2026-10-25-cdp-amd-precision.md) takes2h53m and scores
15.67 versus initial1.42. All0/24 two-wall successes; all audits pass. Do not
extend either life. BF16 is unavailable on RX7900XT/RADV, and the early paired
precision update difference does not establish the cause of the historical gap.

**Compute priority, updated from the user's October25 direction:** finding useful
game-specific configurations matters more than maintaining one universal set of
hyperparameters. Measure frozen learning against GPU time and experience. A
working development recipe is useful evidence; reliable success across seeds is
still a separate requirement. The completed comparison retained its fixed budget.

## Goal and architecture

Learn to play games through experience, initially with sparse explicit rewards
and human guidance, eventually transferring skills between games. First make one
actor learn reliably. Video priors, native games, transfer and swarms follow.

The main path is Dreamer with **centered CDP**: a jointly learned small RGB CNN,
categorical recurrent state-space model, latent prediction instead of a pixel
reconstruction decoder, and imagined actor/critic learning. The action-effects
exploration ensemble currently has coefficient1. Native Rust/Meganeura/Blade
perform learning and inference; Python adapts games and analyzes results.
Frozen Large/Tiny JEPA and DINO are not the active frontend. Learned RGB with
reconstruction remains the control. CDP's five-game advantage over matched RGB
has not been established. The fixed Qbert RGB comparison favors one retained
CDP recipe under strong expert supervision. The completed matched Breakout RGB
control also scores lower in both modes for23.63% more training time;both
miss the two-wall gate. These and the earlier Seaquest comparison are narrower
evidence,not five-game superiority.

Use eight environments with batched inference and one shared learner, independent
stream histories, native max-pooled frames and one GPU resize (64 for the
reference;32/48 for declared game-specific tests). The historical
Atari reference was12M/B8/T16/H15/BF16/replay-ratio32. Current AMD recipes use
Auto and the per-game configurations below; BF16 is unavailable on this host.
Full18-action space, sticky actions.25 and action repeat4 remain the retained
protocol.500k actions are roughly2M emulator frames;
budgets count aggregate actions across streams, not per-stream actions.

## Verified position

| Game | Latest useful frozen result | What remains |
| --- | --- | --- |
| Boxing | AMD1M/RGB32/R32/500k seeds2017/4013 score76.83/79.17;24/24 wins each in28m22s/28m13s | Reference99.61; broad reliability remains open |
| Pong | AMD12M48/B16/R16/500k: retained1009/4013 greedy15.208/15.208 and24/24 wins each; sampled15.625/14.833. Training41m28s/41m31s; four-arm greedy service adds3m33s, no training | Reference20.45; both greedy point estimates pass narrowly, sampled one of two; broad reliability open |
| Freeway | AMD1M/RGB32/R32/aided500k seeds1009/4013:31.04/32.00,24/24 successes each in28m17s/28m50s; historical aided cohort32.10,72/72 | Reference33.40; broad seed reliability remains open |
| Breakout | Retain 12M64/Auto/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4, expert100, critic1/1: 200k scores70.750 sampled/143.667 greedy in21m27.910s. Fresh400k scores92.792/173.708 in43m36.578s, with lower gain/min. [Budget comparison](results/2026-11-03-cdp-breakout-budget400k.md). Current-critic B8/T32 scores94.583/116.667 in23m06.548s: greedy gain/min falls24.84%, while the sampled rise has an interval spanning zero. Retain B16/T16. [Sequence comparison](results/2026-11-03-cdp-breakout-value1-context32.md#full-200k-comparison). Increasing episode guidance20%→50% scores34.417/50.583 in21m37.887s; retain20%. [Guidance result](results/2026-11-03-cdp-breakout-guidance50.md). Intermittent50%32-action blocks also regress to41.250/74.875 in21m11.408s; retain20% episode guidance. [Block comparison](results/2026-11-04-cdp-breakout-block32.md#full200k-comparison). All final modes remain0/24 two-wall; earlier negative comparisons remain linked in the diagnosis. | Original mean864/90%-two-wall gate remains unfinished. Reference381.81 is a published long-run aspiration, not a matched-budget control; reliability remains open. |
| Qbert | Retain AMD 12M64/Auto/B16/T16/H5/R20, dynamics 1e-4, base/encoder/behavior 4e-5, replay-value 1, expert 100 and greedy collection/evaluation. Fixed 240k learners 16001/18013 score 15,279.167/15,931.250 greedy with 24/24 first pyramids each, in 25m57.740s/26m23.114s. Both own initial controls clear none; all frozen arms have zero updates/truncations and 357 invariant tensors. [Development result](results/2026-11-02-cdp-qbert-r20-240k.md), [fresh confirmation](results/2026-11-02-cdp-qbert-r20-seed18013.md). | Original 15k/90% gate passes on the development learner and one fresh confirmation. Sampled 12,972.917/14,976.042 both miss 15k. Broad reliability and reference 193,220.77 remain open; earlier failed recipes remain in the linked reports. |

These references are published long-run Dreamer scores, not equal-budget
controls. Five-game supremacy is **not achieved**. Report matched-frame and
wall-clock learning alongside any comparison to these endpoints.

Freeway's aid is5% persistent-uniform16-action blocks, disabled in evaluation.
Keep its cohort separate from unassisted31.83/30.58/0 (mean20.81,48/72 successes).
Do not hide seed3019's failed unassisted run or the intervention's selection on it.

Breakout warm-starting from2M to4M-total and8M-total yields80.17 and85.17,
below the98.21 source control. No unchanged extension follows. Greedy action
selection has inconsistent effects; current Breakout declarations use greedy
as primary and retain sampled secondary evaluations. Reward forecasts improve;
trained state contains ball information, but longer
warm-start training does not improve15-step ball-x prediction. This supports a
temporal-learning experiment, not an immediate encoder rewrite.

Evidence and videos:

- [Pong](results/2026-10-21-cdp-pong-budget.md)
- [Freeway replication](results/2026-10-23-cdp-freeway-replication.md)
- [Breakout budget](results/2026-10-22-cdp-breakout-budget.md),
  [warm start](results/2026-10-23-cdp-breakout-warmstart.md),
  [forecasts](results/2026-10-22-cdp-breakout-forecasts.md),
  [state diagnosis](results/2026-10-23-cdp-breakout-state.md)
- [Qbert](results/2026-10-11-cdp-qbert-budget.md)
- [Boxing](results/2026-10-09-cdp-boxing-capacity.md)

## Current execution order — strategy reset

1. **Transfer the actual runtime and unfinished checkpoint.** Follow the handover;
   do not rebuild the older PR source and assume it matches recent training.
   Backend optimization and upstream porting are parked. Native/JAX parity is
   not a prerequisite for learning.
2. **Qualify initialization and frozen restore on the target machine.** Original
   host PMU faults block GPU work there. A new machine needs its own bounded
   check, not the old host's boot/device pins. Preserve the867-update smoke
   checkpoint; compare all346 tensors before/after frozen evaluation.
3. **Completed: the declared Breakout context experiment.** B2/T64 versus retained
   B8/T16 keeps128 sampled positions/update and ratio32; it changes temporal
   gradient length and independent-sequence count together. One fresh seed1009,
   2M actions, fixed500k/final/actual-initial evaluations of24 natural rounds.
   No checkpoint selection or automatic extension. See the
   [exact declaration](experiments/2026-10-23-cdp-breakout-context64.md).
4. **Completed: matched Auto learning comparison.** Frozen15.67 versus14.71
   under F16 is a small development-seed difference with zero task successes;
   both remain weak and take similar wall time. Early update differences do
   not establish the cause of the T64 regression. Use protected Auto precision
   for subsequent cheap trials; do not extend this context experiment.
5. **Find cheaper recipes for the remaining gaps.** Use T16 as the development
   starting point, eight independent lanes and batched GPU inference. Try lower
   replay ratios and smaller capacity in bounded development runs; changing
   several settings together is allowed when seeking a working cheap recipe.
   Use the game-specific capacity evidence below instead of shrinking every
   game blindly. Confirm promising configurations across seeds before claiming
   reliability. Keep useful gains separate from published endpoints.

The original B2/T64 smoke completed4096 actions/867 updates in45.245s with finite
346 tensors and exact initial tensors. Normal restores faulted before run_start;
one traced restore passed. AMD restore now passes with all346 tensors unchanged
under the explicitly adapted precision. The fresh2M life and fixed evaluations
are complete. Hardware, precision and execution-path differences remain
comparison limitations.

[Incident and diagnostic boundaries](results/2026-10-23-cdp-context64-gpu-fault.md).

## Compute-constrained development

The historical500k-action capacity cohort supports specializing by game:

| Game | Size1M frozen mean | Size12M frozen mean | Cost-conscious direction |
| --- | ---: | ---: | --- |
| Boxing |77.69 |85.26 | Prefer1M initially; both sizes won72/72 selected matches |
| Freeway |30.26 |27.54 | Prefer1M initially; preserve separate reporting for action aids |
| Pong |4.54 |12.40 | Test intermediate capacity rather than assuming1M is sufficient |
| Breakout |5.14 |38.21 | Small capacity was weak; test an intermediate preset before a long run |
| Qbert |993.75 |2657.99 | Small capacity was weak; prioritize learning per GPU-hour |

[Archived capacity data](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-09-cdp-five-game-capacity.json),
blob`b600def5ce8bb55894eacf64e8a0b1c043251ad3`. These are historical development
cohorts, with some runtime differences; the roughly3.56x aggregate wall-cost
ratio is not an AMD speed prediction. They are not the later qualified-backend
results in the main status table. An intermediate native4M preset is now
implemented: deterministic1024, hidden128,32 stochastic variables with8 classes,
8 blocks, units128 and vision depth8. It is a Kindle development preset, not an
upstream benchmark size. Actual parameter count depends on image geometry;
RGB48 centered CDP/action-effects1 has3,578,225 trainable parameters.

The first cheap development configuration should use existing controls: centered
CDP, Auto precision, N8/B8/T16/H15, microbatch8, native frames/one GPU64 resize,
and replay ratio16 as a candidate instead of32. Choose capacity by the game;
1M is a justified starting point for Boxing/Freeway. Halving the ratio halves
scheduled replay-position work per action, not necessarily end-to-end time or
the experience needed to learn. Target100–200k actions and a30-minute training
cap for the first learning screen, with fixed frozen/actual-initial evaluation;
record a wall-cap cutoff explicitly. A short weak result does not prove that
the configuration cannot learn. No broad queue or automatic extension follows.
The firstR16 screens complete quickly but fail their gates. Fresh Boxing500k/R32
then passes within28m22s, with five times as many updates; the experience budget,
ratio and seed change together. Preserve enough learning work to test a recipe:
lower replay ratios are candidates, not a requirement for cheap training.

Native preprocessing, replay shapes and CNN geometry now support32/48/64, with
`--image-size` selecting fresh learned RGB/CDP geometry and restore retaining
checkpoint geometry. The default remains64. Do not simulate lower resolution
with CPU downscale/upscale.48x48 has56.25%
of the pixels and32x32 has25%, but RSSM and imagined actor/critic work do not
shrink proportionally. Check small-object visibility and actual learning before
using lower resolution, especially in ball games. Initial CPU visibility frames
support testing Boxing at32 and Pong at48; Breakout's tiny ball loses contrast
at32, so retain64 initially. Bounded native training/frozen smokes pass for
Boxing1M/RGB32 and Pong4M/RGB48, at578.11 and349.37 aggregate actions/s. They
verify execution and exact frozen tensors, not learning. The existing64x64
checkpoint restores unchanged; see the cheap-runtime result report.

## Runtime and performance

Retained NVIDIA Freeway B8 runs take about38 minutes per500k actions; Breakout2M
takes about2.5 hours. The AMD B2/T64 F16 run takes11h40m at47.59 actions/s, with
97.7% of wall time in learning; sampled device-wide busy is typically93–94%.
The failed B32 Freeway control ran faster (28m14s) but learned
worse. Batch size changes optimizer cadence at fixed replay ratio; synthetic
throughput alone is not the objective. AMD busy samples do not establish kernel
efficiency; the historical learning runs lack corresponding utilization data.
Do not recycle old65% utilization or0.51× realtime as current measurements.

The qualified c8aeba28 stack includes reduced-precision kernels, resident targets,
GPU sampling and convolution reuse. Its last retained RGB synthetic comparison
was1.525× JAX cost, with scope in the performance reports. No further kernel
sweeps or upstream porting are scheduled. Preserve the source; rebuild and check
on the new machine. Identical source does not guarantee identical binary bytes,
GPU behavior or speed across devices and drivers.

Long-term pipeline: GPU game → GPU capture → preprocessing → model → action,
with only actions read by CPU. Atari emulation currently remains on CPU. Playing
and updating can run faster than real time; no real-time throttle is required.
“Online learning” describes the coupled acting/training workflow sufficiently.

## After Atari

- **Video priors:** test whether video pretraining improves dynamics and learning
  at equal total experience/compute. Record data provenance and same-game exposure;
  video-pretraining support is not established by the Atari results.
- **Persistent/native games:** use `/x/Code/mind-games`, beginning with vkQuake2
  or TMNF, then suitable GOG/Wine games. Add sparse reward adapters and report
  end-to-end latency, reliable capture and real gameplay.
- **Transfer and retention:** train across related games, then measure adaptation
  to an unseen game, forgetting and dependence on per-game reward engineering.
- **Swarms:** only after strong single-actor native-game results and transfer.
  Experience sharing/concurrent learners are not prerequisites for this phase.

## Evidence and process

Develop cheaply, confirm promising changes across seeds. Keep actual-initial
controls, frozen evaluations with zero updates, full action accounting and
rollout videos. Disclose aids, truncations, selected seeds, warm starts and new
hardware. Checkpoint restore does not preserve replay/RNG/games/live belief.
Keep compact results in Git, large artifacts on shared storage, and only useful
checkpoint milestones. Do not build another status ledger into AGENTS.md.

[Strategy rationale](strategy_reset_plan.md), [historical reports](experiments/README.md)
and [performance history](backend_training_performance_plan.md) remain available.
Historical “next” paragraphs describe past states and do not authorize queues.
The previous full roadmap is preserved in Git at8d464e0.
