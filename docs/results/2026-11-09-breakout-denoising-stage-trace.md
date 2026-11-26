# Frozen denoising stages: weak new-ball creation before the final step

The goal **prove the world model training on Breakout** remains unproven.
The [declared native trace](../experiments/2026-11-09-breakout-denoising-stage-trace.md)
reproduces all108 final batch arrays/864 RGB images and20 origin16 model arrays
exactly. All135 checkpoint tensors remain unchanged atage2,048. This is a new
internal-stage diagnostic ofthe retained half-max-noise checkpoint,not another
learner or a rerun ofthe closed assessment. No production code/API/model/loss/
sampler change;only a test-only native collector isadded.

The first clean denoising estimate isalready weak. The middle stage both loses
andcreates some next-position matches;the last stage changes none ofthose
matches. Thus the failure isnot confined tothe final sampler step. Intermediate
denoised estimates can be conditional means rather than samples:their point
scores are descriptive,not replacement forecasts orgates. No best-stage policy
is selected. Different stage inputs prevent unique causal attribution to sigma.

| Draw | Next-position matches at sigma5 /.283 /.002,outof198 | Detected atthose stages | Previous-position matches |
|---|---|---|---|
|0|17/11/11|79/48/48|59/34/34|
|1|17/17/17|69/49/49|49/30/30|
|2|13/16/16|67/48/48|53/30/30|
|3|15/14/14|67/51/51|52/32/32|

| Draw | Correct stage1 positions kept /lost /new atstage2 | Stage2→3 losses/gains |
|---|---|---|
|0|7/10/4|0/0|
|1|11/6/6|0/0|
|2|11/2/5|0/0|
|3|9/6/5|0/0|

The final clean estimate equals the served image bit-for-bit. Its11/17/16/14
matches of198 arethe unchanged failed fitting result;ordinary rollout remains
1/1/0/2 of110. No new competence orheld-out reliability isclaimed. Allfour draws
share a learner andthe reused eight clips. Missing/ambiguous balls remain
missing;18 unsupported ball endpoints per stage/draw stay explicit. Median
movement projection remains0 among detected/current/previous supported pairs
atall stages;copying andabsence still dominate.

Separate pixel measurements expose an asymmetry. For pixels occupied by the
next ball butnot the previous ball,measure projection ofthe predicted RGB
change onto the required RGB change;do the same for vacated old-ball pixels.
Copying gives0 andthe recorded target1. Masks/coordinates are analysis only;
no synthetic centroid ismanufactured fromfaint predictions. These RGB change
projections are not fractions ofphysical movement orloss decompositions.

| Stage | Median new-location RGB-change projection,range overdraws | Median old-location erasure projection |
|---|---|---|
|1|0.147–0.156|0.514–0.534|
|2|0.103–0.109|0.547–0.559|
|3|0.101–0.110|0.547–0.559|

The served image makes roughly10–11% ofthe desired color change atnew ball
pixels,while removing about55% atold ones. This supports weak creation ofthe
moved ball inthese predictions;it doesnot uniquely distinguish optimization,
capacity,conditioning or distributional uncertainty.

An additional CPU-only statistic onall224 unique source training arrivals
(not an exposure recount) finds internal whole-image standard deviation
0.47236984,but frame-difference standard deviation
0.03536981 andRMS0.03536982. The residual generator
still uses sigma_data.5 for EDM preconditioning,the value retained fromthe
absolute-image model. This suggests a data-scale optimization hypothesis;it
isnot proof that this coefficient causes the failure. History conditioning
also uses.5,but is a separate operation andmust not be silently rescaled in a
future test. No new coefficient orlearner isselected bythis diagnostic.

| Evidence | Result |
|---|---|
| Collector source | 35018f7d;production sourceccf09d7f unchanged |
| Native executable SHA256 | 3c8070f3a628e0d3e433f68a2fd1bfa1723db7ac68fdd29490852c2763ad0315 |
| CPU | Release collector compiles;16 focused checks pass;three GPU checks ignored until explicit selection |
| Native queries | 108 traces/324 denoiser calls;four production parity queries |
| Exact parity | 108 final batch arrays/864 images,20 origin16 arrays |
| Frozen state | 135 tensors andage2,048 unchanged;no learning |
| Native elapsed | 3.029s including restore/query/export;not isolated GPU kernels |
| CPU analysis | 0.671s;37 declared/25 CPU pins |
| Data | All27 origins5..31,eight lanes,four draws;no new emulator actions/frames |

Exact source snapshots,declaration,input bytes,case/checkpoint/cache hashes,
sealed host guard,raw float stage outputs andevery2,592 draw/origin/stage/lane
row remain. The trace helper receives only observed prefix images/actions and
one incoming command;expected outputs andtarget pixels are not arguments.
Only the selected current checkpoint'scached final RGB isdecoded for required
parity;no other historical model isqueried. Production inference andnoisemath
are unchanged. No sampler intervention,alternative-policy score ornew baseline
learner is introduced.

`lane0-stage-traces.png` has27 exact lossless panels ofseparate one-step
forecasts,not a continuous rollout. Fixed lane0/draw0 andposter origin16 prevent
best-case selection. `stage-endpoints.csv` and`review.json` retain all endpoints,
raw candidate counts,stage transitions,motion andRGB-change projections.
Root `/mnt/data/kindle-breakout-denoising-stage-trace-20261109`.

The full goal still requires original physical/event precision,natural
calibration,persistent brick memory,late/tunnel/second-wall/above-top coverage,
untouched families andfresh confirmation. The full oracle lacks above-top
endpoints. No trace orfitting score completes that goal. All earlier failures/
repairs andclosed learners stay retained;no extension,collector,broad learner
or automatic successor. Actor/critic,CDP comparison andfive-game work stay parked.

Closure verifies87 pins in0.517s,one sealed guard/nine
files andone terminal successful service. Own580,736,006-byte
build cache isreleased;the native executable,source andall evidence remain.
