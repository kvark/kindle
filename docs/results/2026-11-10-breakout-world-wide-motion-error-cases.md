# Wider-motion saved cases: drift, undetected balls and event disagreement

The saved wall/serve cases confirm functional image andevent errors. They do
not identify a causal capacity,data,optimizer orboundary-condition defect.
The original broader assessment stays failed andthe full world-model goal is
active/unproven. [Protocol](../experiments/2026-11-10-breakout-world-wide-motion-error-cases.md).
Root `/x/Temp/kindle-world-wide-motion-error-cases-20261110`.

## Fixed support and case findings

All18 serve/four wall plans,both histories,five endpoints,eight origins/eight
model draws remain:44 plan-history groups/220 endpoint cells/1,760 origin rows.
Tables below are history16,all model–oracle pairs. These are related reused
development cases,not independent trials oruntouched confirmation.

|Plan/horizon|Within1px /supported pairs|Detected-only RMSE,pixels|Mean ending probability|
|---|---:|---:|---:|
|Wall NOOP1|128/128|.275|.00216|
|Wall NOOP8|248/448|2.338|.03912|
|Wall NOOP16|0/512|7.616|.09683|
|Wall LEFT16|88/512|9.859|.09587|
|Wall RIGHT8|408/448|.572|.01268|
|Wall RIGHT16|152/448|2.713|.09153|
|Wall reversal8|448/448|.446|.03022|
|Wall reversal16|64/448|1.589|.07521|
|Serve FIRE1|252/512|6.750|.00077|
|Serve FIRE8|0/512|unscored|.10313|
|Serve FIRE32|0/512|28.527|.64038|

Serve FIRE8 haszero detected pairs despite512 truth-supported pairs;the fixed
instrument conflates missing anddiffuse renderings. At32 it detects248/512 but
none within1px. Serve RIGHTFIRE(action11) haszero detections at1/8/32 on512
supported pairs each. Serve NOOP appropriately hasno scored ball feature,with
zero detections atthose horizons;that does not establish useful launch dynamics.

Wall NOOP/LEFT/reversal32 havezero supported true ball-coordinate pairs but
448/448/384 predicted detections ontruth-absent feature pairs. This is absence
under the fixed region/detector,not proof the physical ball is absent. Gray
walls,brick-region motion andunsupported balls limit interpretation. These
false-feature counts differ from coordinate precision andall original scores
stay unchanged. RIGHT32 has16 detected/64 supported pairs,zero within1px.

All selected oracle paths havezero game-over terminals,while many model ending
probabilities arelarge. A visible rendered ball alone doesnot prove a generated
path isalive;the saved image/event observations do not isolate why heads fail.
Detected-only RMSE excludes missing balls andcannot replace supported accuracy.

## Images,scope and integrity

Five fixed lane0/model-draw0/oracle-draw0 grids areviewed:the four wall commands
andserve FIRE,history16,1/4/8/16/32 endpoints. They show good early agreement in
some wall paths,then position drift,streaks,wrong later trajectories andball-like
features. The serve row loses coherent ball motion. A starting `wall` label
does not localize later error to the first wall collision;later brick/paddle/
region transitions also matter. No favorable row ornew gate isselected.

One4.040675s CPU analysis verifies209 pinned inputs,finite arrays and1,760 ending-
probability joins toexisting reports. All raw per-origin features,probabilities,
supported/missing/false pairs anddetected errors remain in `cases-v1.json` and
`group-summary-v1.json`. Five PNGs andall closed outputs seal after log exit.
Zero model/GPU/emulator query ortraining update. Constructors trapped,Vulkan off;
inputs verify before/after. No failure orretry in the declared analysis.

Before declaration,a read-only header inspection used system Python3.14 with
the project's NumPy2.5.2/cp313 extension andfailed import. The original error is
retained in `preparation-import-failure-v1.json`;the separate project-Python3.13
header inspection succeeds. Packages andnative runtimes are unchanged. This is
a preparation error,not a repaired model/assessment orhidden analysis rerun.
Scientific sourcee292290c;helper/runtime bytes arepinned.

## Next hypothesis

Source inspection shows reward/continuation heads read h/z,whereas the complete
local world state also includes24,576 image-history values. That contract does
not prove h/z lacks sufficient information. Select a separately scoped
[decoded-RGB event-head prototype](../experiments/2026-11-10-breakout-world-rgb-event-heads.md)
to test direct visual readout ofthe model's own forecast alongside h/z. This is
a hypothesis for one measured failure,not a claimed fix for wall/serve physics.
Earlier generated-image feedback andall failed learner recipes remain closed.

No new learner orGPU allocation isselected here. Full physics,natural event
calibration,brick memory,late/tunnel/second-wall/above-top coverage,untouched
families andfresh confirmation remain required. Actor/critic,CDP andfive-game
work stay parked;the full goal remains unproven.
