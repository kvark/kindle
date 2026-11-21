# Four generated training-history frames regress prediction

Reject this recipe. The fixed final checkpoint scores **27/29/19/31 of110**
ordinary rollout ball matches, versus the matched B8 control33/31/29/28.
Actual-history one-step falls to **139/139/140/146 of198** (70.2–73.7%), versus
178/174/176/173 (87.4–89.9%). Only **2/24 gates** pass:draw0 anddraw3 ordinary
horizon1. Both controls and the candidate remain far from the full physics goal.
Retain B32/reward32 as the ordinary reference and B8 as distinct fitting evidence.

| Fixed final measure | Clean B8 | Four generated history frames |
|---|---|---|
| Ordinary all16,110 supported per draw |33/31/29/28|27/29/19/31|
| One-step,198 supported per draw |178/174/176/173|139/139/140/146|
| Ordinary horizon1,support8 |7/7/6/7|8/7/7/8|
| Ordinary horizon4,support6 |4/3/3/2|3/3/1/3|
| Ordinary horizon8,support8 |2/1/0/1|1/2/0/2|
| Ordinary horizon16,support8 |1/1/1/1|0/0/0/0|
| Learn-call interval |218.921s|218.693s|
| Training/checkpoint/forecast loop |228.929s|270.376s|

Loop time rises 18.10%, including
40.020s ofcache generation/serialization andfirst cache snapshots.
These are shared-host wrapper intervals,not isolated kernel speed. No new native
build, backend primitive, emulator action/frame, actor orcritic is used.

## What was trained

Fresh B8/8,192 T5 updates,65,536 targets,RGB64,seed41001 andthe exact completed
B8 config. First4,096 updates use actual four-frame histories. Then half ofeligible
source repetitions replace allfour prefix frames with native prior forecasts.
For targetT>=9,generate four frames from actual observations beforeT−4 andthe
recorded commandsT−4..T−1,seed50001+32*age+T. Never use targetT orobserved noisy
reconstruction in generation. Refresh16 caches at4096,4352,...7936:368 queries,
11,776 imagined frames. Keep float32 RGB without CPU resizing/roundtrip.

1,678 calls/13,424 targets use generated prefixes;current RGB/actions/rewards/
terminals remain unchanged. Exact source exposure/native Gaussian order,
224 unique arrivals,588 positive and32,768 forced-max-noise targets match B8.
Each source has balanced forced/nonforced noise inactual andaugmented groups.
Arrivals4..8 andboth source positive events stay clean. Cache staleness isatmost
255 updates. This is detached error-correction training onfour generated frames,
not differentiable unrolling ortraining an uninterrupted16-action trajectory.
Generated pasts can depict a different state fromthe recorded target sequence;
this recipe doesnot supply an emulator-correct future for an arbitrary generated
state. Anchors,past-derived pixel importance andevent conditioning change jointly.

## What the caches and outputs show

Across each cache's665 supported prefix positions,strict ball matches range
14–384. For9,042 of12,122 supported augmented current targets,the last generated
history frame has no detectable ball (74.59%). These are exposure-weighted counts,
not independent episodes. Detector absence doesnot prove complete information
loss oridentify a causal optimization failure. Allcache source/command/seed/age
joins andall1,678 generated training assemblies verify. All11,776 cache endpoint
rows are retained. The full four-image/four-command exact-input audit finds no
duplicates/conflicting target images within each cache;this doesnot establish
observability orrule out near-aliases.

Final ordinary outputs have75/77/83/77 missing balls andno ambiguous masks of110
supported positions per draw. Teacher outputs miss57/55/58/50 balls of198,with
141/143/140/148 detected and139/139/140/146 correct. Conditional median teacher
position error .243–.302 pixels andmotion projection .983–.992 exclude missing
balls. The full all-horizon,paddle,brick,pixel,changed-pixel,false-detection and
event arrays/metrics remain;no best-age,draw,stage ortolerance replaces final.

Ordinary zero-reward RMSE slightly regresses .000005694→.000006841. Teacher
RMSE slightly improves .0348437→.0346195 andthe single causal positive isabout
.993815 for1. No positive/ending alarms arefalse onthis small slice;all endings
are negative. One distinct causal positive repeated across four draws andno
terminals cannot establish natural-frequency calibration orlife/game-over timing.

## Integrity and scope

Seven CPU preparation check groups pass. All135 initial andfirst-update tensors,
six first losses,and135 tensors/20 model arrays at2036 and4084 match clean B8
exactly.20 initial arrays exact aswell. The first cache preserves135 tensors;
allcache queries preserve model age.8,192 finite six-loss rows,one update/eight
targets each,total identity,source plan,all45 changed parameters,checkpoint
config/ages,135 own frozen tensors and20 restored model arrays pass.

Both original guarded stages complete successfully;two guards/18 sealed files
pass,withno stage/audit repeat orrepair. Native60b9b672/source8dbf9ab5 reused
unchanged. Declaration sourceabcd6ea6;root
`/mnt/data/kindle-breakout-full-causal-prefix-20261109`,invocation
`b0818d4b6b6f4764ab506bf4f4ee4da5`. Cached B8 isnever queried ortrained again.
All1,376 evaluated endpoints,complete curves/useful checkpoints and16 fixed
lane0/draw0 lossless frames arepreserved outside Git.

This closes one unsuccessful training-history comparison;it doesnot rule out
allgenerated-history objectives. The earlier latest-only result remains negative
andclosed. It also doesnot reverse the separate recurrent-world contact-fit
proof (110/110 across allfour draws),which uses a different architecture,loss
andtarget budget. Neither fitting proof establishes a general Breakout world.

Full physics/events,natural calibration,changed-brick memory,late/tunnel/second-
wall/above-top,untouched families andfresh confirmation remain required. The
full world-model goal staysactive andunproven. No automatic successor isselected;
actor/critic,CDP comparison andfive-game work remain parked. Fullres remains
protocol-failed andallprevious native stages stayclosed.

Closure verifies 247 pins,one successful terminal service and18 sealed guard files.
