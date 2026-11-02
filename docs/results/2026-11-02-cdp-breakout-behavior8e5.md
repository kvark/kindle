# Breakout actor/critic-rate comparison

The completed fresh learner scores **59.000 sampled / 117.292 greedy**, versus
**56.292 / 104.667** for the retained behavior4e-5 control: **+4.81% / +12.06%**.
Training takes **21m44.517s**, **1.31%** longer than21m27.701s, for the same
200,000 actions /12,477 updates. All four candidate arms have **0/24 two-wall
successes**. The original task gate still fails.

| Fixed 200k recipe | Sampled mean | Greedy mean | Training time | Two-wall successes |
| --- | ---: | ---: | ---: | ---: |
| Behavior4e-5 control |56.292|104.667|21m27.701s|0/24 in each mode|
| Behavior8e-5 candidate |59.000|117.292|21m44.517s|0/24 in each mode|
| Actual initialization, both recipes |1.417|2.208|—|0/24 in each mode|

Initial-subtracted return per training minute improves **3.58% sampled / 10.87%
greedy**. Retain8e-5 as an incremental development option and the higher sampled
point for this recipe. The separate dynamics1e-4/behavior4e-5 result remains
stronger in greedy score at138.750, with lower sampled50.625. This comparison
uses the same development seed1009 and evaluation cohort; it does not establish
seed reliability, a general rate preference or an isolated actor/critic cause.
No extension, rate sweep, confirmation or automatic successor is selected.

Only the shared behavior rate changes. Native447fe50/SHA35ad0db, runner6c1b69fb,
teacher,12M64/Auto/N8/B16/T16/H15/R16/expert100, base/encoder4e-5 and dynamics4e-4
remain fixed. No experience is reused. Actual training overrides fall from
99,343 (49.6715%) to81,918 (40.959%). Late50k replay expert CE changes.558097→.551145,
while aided online last50 mean falls103.8→66.0. Neither online returns nor replay
fit substitute for the frozen scores.

## Completed evidence

All282 pins,100k/6,227 retained recovery,357 bounded/control-matching initial
tensors,800 sampled/795 greedy initial semantic events,357 invariant tensors per
frozen arm,12,477 finite supervised reports, complete training pixel/action/label
replay, four frozen replays/videos, host guards, curves, exposure and comparison
pass. Training records799,577 executed frames. The four frozen arms total84,288
actions /336,977 frames /77.365s, withzero updates or truncated episodes. Each
score uses the firstthree natural episodes per eight lanes. Ordinary frozen
logs lack live pixel hashes; replay does not compare unavailable original images.

Original mainPID873316/invocation`56e692c47ef348ff8acd60089b7b46df` succeeds
**01:00:07.889 UTC November2**, postprocessing **01:00:31.461**, and collection
**01:01:01.857**. All five original services are terminal success. The read-only
terminal reader completes **01:01:40.653** in1.979s (2.000s wrapper), verifies
original invocation journals and the collector payload, and recounts the fixed
scores from existing replays. It does not repeat an experiment stage.

The earlier CPU prelaunch assertion failure remains preserved separately; it
created no service or GPU allocation. No training, frozen, replay, postprocessing
or collection stage fails or repeats after the first actual GPU launch. CPU
co-host costs remain recorded without time correction. The package is closed;
do not rerun the learner, collectors or completed result reader. No Kindle GPU
allocation remains active at closure; check live state before subsequent work.
The five-game goal remains unfinished.

| Completed artifact | SHA256 |
| --- | --- |
| Declaration | `d8d4f5820b24749d08edc9be44a22f39e7fb000dd6eae209874e54be5ef996f8` |
| Learning review | `955286fcb832da766f80ef57475b25b8b3f0d3f0b2edc53191f3c930df7faefe` |
| Original completion | `7e0389c224a309a2ab9f87581aca0fedfcaf6847038da6c6a8be82441e546150` |
| Paired comparison | `0f2163113c0cfd1222d480be06d116c0cb3b9e6783b02233a98695c9540154c7` |
| Terminal summary | `76dfd6e3a3a83df5e91aa42ee3e31619f6792c8270c77cc2fb78dd018e2b3cbf` |

Evidence root: `/x/Temp/kindle-breakout-behavior8e5-20261102`; the full trial is
`full/breakout-12m-64-b16-r16-behavior8e5-expert100-200k-seed1009`.
The following sections preserve preparation and milestone history.

## Bounded execution check

The bounded8e-5 behavior-rate check passes **8,192 actions /32,724 frames /
489 updates in47.149s**. Frozen1,024 actions /4,096 frames take.867s, withzero
updates andno natural completed episodes. Its conservative200k projection is
**1,266.197s (21m06.197s)**, below the30-minute limit. This establishes execution
andearly cost, not gameplay benefit.

Only the shared actor/critic learning rate changes fromunset (effective4e-5)
to8e-5. Keep base/encoder4e-5,dynamics4e-4 andcenteredCDP/12M64/Auto/N8/B16/T16/
H15/R16/expert100, sampled collection, unit label weights, world gradients,
all-state corrective labels and20% guided episodes/cap1920. Native447fe50/
SHA35ad0db, original teacher andrunner6c1b69fb areunchanged byte-for-byte.
There isno new loss, build or per-update computation. Actor andcritic effects
are not separated; the reused1009 seed/cohort doesnot establish reliability.

A read-only review of the exact native source447fe50 clarifies the scope: the
behavior rate feeds the shared LaProp optimizer. The actor objective includes
imagined policy loss and replay imitation; the critic includes imagined and
replay value losses. The updated heads are synchronized into the world graph,
where their weights are frozen while gradients can flow to the representation.
Thus fixed world learning rates do not isolate representation changes. This is
source inspection, not a measured gradient effect or causal diagnosis. See
[behavior optimizer](https://github.com/kvark/kindle/blob/447fe5013b6442c477fc7bfef7eab4122082bc54/kindle/src/dreamer/agent.rs#L2197),
[behavior objective](https://github.com/kvark/kindle/blob/447fe5013b6442c477fc7bfef7eab4122082bc54/kindle/src/dreamer/behavior.rs#L151)
and [frozen world heads](https://github.com/kvark/kindle/blob/447fe5013b6442c477fc7bfef7eab4122082bc54/kindle/src/dreamer/world.rs#L574).
The package's `behavior-rate-scope-review.json` records source hashes and the
unchanged live process at00:40:38 UTC;0.012s CPU, no GPU work or time correction.

All252 pinned inputs,357 matching control/own initial tensors,47 identical
pre-update transitions,489 finite supervised reports,changed CNN/reward/actor/
value tensors,357 invariant frozen tensors,complete training pixel/action/label
replay andhost guards pass. All117,048 labeled replay rows haveunit weight;
world/actor expert losses agree within8e-7. Actual overrides are3,733/8,192
(45.5688%). Online guided returns are not frozen competence.

CPU preparation passes00:24:51.686 UTC November2/source55b56d2 in.657s (.690s
wrapper): two actual commands,15 config rejection cases,357 schema tensors,
227 control pins andexact cadence. Seven direct-global checks andthree observer
fixtures pass. No native module,agent or emulator isconstructed. No qualification
failure orrepeated successful stage occurs. The full-script preparation performs
a1.126s file operation while the bounded service isactive; no training-time
correction isapplied andno isolated timing effect isclaimed.

MainPID872732/invocation`6303ac2d8ea54b5186f0bdbc78fde936` launches00:25:28.961 UTC
November2/source72c1afc andsucceeds **00:26:32.410**. Original collectorPID872735/
invocation`a781d44fbea445c6b16cdaf239e5bcdc` writes completion **00:26:39.698** and
is terminal success. Both original identities,complete collector journal payload,
all252 inputs andcompletion-file hashes areverified. No original stage fails or
repeats. Root `/x/Temp/kindle-breakout-behavior8e5-20261102/native-check` isclosed;
do not repeat it orreuse its experience.

| Bounded artifact | SHA256 |
| --- | --- |
| Declaration | `fe1b43af7ac20a53951965ad16b2e4acc9dfd562eeacde7f79f7cb799a05a00c` |
| Learning review | `94ed75a89d775a8b45771a2a1858eda2ee7642db8bc6d7f8a133a52f8e11d76c` |
| Original completion | `15c75e4cda07d0a2770d4f39e1340412deb327576ce0f7dd59caa4a6f0cf740c` |
| Guided replay | `9256df40b1775aa6b17d59d959d3c7dbc0138420e498796017da9a750f31c306` |

## Full preparation passes

The [separate200k protocol](../experiments/2026-11-02-cdp-breakout-behavior8e5-200k.md)
retains200k actions /12,477 updates /30min,100k/6,227 recovery andfour actual-
initial/final sampled/greedy24-episode assessments. Compare sampled56.292 primary
andgreedy104.667, allzero two-wall, at21m28s. The original task gate staysfixed.

Full CPU qualification passes **00:28:25.327 UTC November2** in1.399s (1.498s
wrapper): five actual commands,28 config cases,12 header cases,four observer
argument cases,eight recovery-body/four metadata cases,six retention/two actual
collector fixtures,44 paths and18 scripts. Seven global screens pass, andthe
collector/watcher unit lists agree. All252 bounded/227 control pins stayintact.
No native or emulator construction, failure orrepeated successful stage occurs.
Preflight SHA256`3268fe898cf6cb569377f99115c2c5fa973900794bfc0cafd8c36a0759b1fc7b`.

The full learner is separately selected after this qualification andanother
idle-host check. Its first actual GPU launch follows the CPU repair below.
No earlier weights, replay, beliefs, games, RNG orbounded experience are reused. No retry, rate sweep, extension,
confirmation orautomatic successor. The five-game goal remains unfinished.


## Prelaunch CPU assertion repaired before any GPU allocation

The first full launch invocation fails at 00:32:12.591 UTC November 2/source
7dfa470, before service creation. A copied preflight assertion requires the old
`--dynamics-learning-rate 1e-4` flag, absent from the unchanged control runner.
The original CPU preparation parsed actual commands but did not execute this
separate preflight check. Its passing receipt remains historical, with this
coverage gap recorded.

The correction checks the intended `--behavior-learning-rate 8e-5` flag instead;
dynamics remains the bounded-verified 4e-4 default. Separate actual-preflight
qualification passes at 00:33:24.500 in 2.911s (3.012s wrapper), including rejection
of wrong and missing behavior rates. All other 270 original pins stay unchanged.
No native or emulator is constructed, no service starts and no GPU stage repeats.
The full directory snapshot, original declaration, controller and failed preflight
are retained under `prelaunch-failure/`; failure receipt SHAe4cb42a2. Corrected
controller SHA3441f72b. Repin this correction and evidence before the first GPU
launch. The training recipe and all caps remain unchanged.


## First GPU launch and startup

The first GPU launch is **00:34:24.392 UTC November 2/source3099ff7**. Main
PID873316/invocation`56e692c47ef348ff8acd60089b7b46df` was active at startup. Original
postprocessing/progress/recovery/collection PIDs are873319/873322/873451/873454,
with identities retained in `observers.json`. All training/evaluation caps and
settings remain fixed. Declaration SHAd8d4f582 pins282 inputs, including the
preserved original CPU failure and separate correction.

The read-only startup check passes **00:34:37.150** in1.019s wrapper. All282 pins,
357 finite control/bounded-matching initial tensors,47 matching pre-update
transitions, exact configuration/native/runner/teacher, original process identity
and fresh counters pass. First six updates occur at376/392/408/424/440/456;
77 finite supervised reports cover the complete prefix through1,608 actions.
This verified startup only; the completed frozen result is recorded above.

The original service, four observers,100k/6,227 recovery and final/actual-initial
sampled/greedy assessments completed without repeats.
No extension, rate sweep, confirmation or automatic successor is selected.


## Terminal reader prepared while the original learner runs

A separate read-only result reader passes preparation at **00:37:52.447 UTC
November 2** in0.087s (0.113s wrapper). It reproduces all four retained control
scores from the first three natural episodes per lane and rejects four altered
score/gate/environment cases. It checks existing replay/log/video hashes without
constructing a native agent or emulator. The original learner identity remains
unchanged before and after the check; retain this CPU co-host cost without time
correction.

After the original services became terminal, the package's
`review_full_terminal.py terminal` ran once through `run_cpu.py`. It verified all282 inputs,
completion-file hashes, original invocation journals, the collector payload,
retained100k checkpoint, paired initial traces, four frozen scores and the
preserved CPU prelaunch failure. The separate completed readout is recorded above. Do not rerun it or any
original experiment stage. ReaderSHA
`ba571025beca4480f21224292cb0f1e08e9faa67c94c181dd281174060c76f28`;
identity-helperSHA131bee46. Preparation receipt: `terminal-reader-preflight.json`.


## 100k recovery milestone

The original recovery observer completes at **00:45:30.437 UTC November 2**,
with100,000 actions/6,227 updates,357 finite tensors and206,067,163 retained
bytes. Its complete log prefix contains103,064 actions/6,419 finite reports;
latest progress is102,400/6,377 at656.434 training seconds. The copy takes0.307s
and preserves the source. Weights and metadata are retained; replay, live belief,
games and RNG are not an uninterrupted restart state.

Separate read-only confirmation passes **00:46:33.271** in0.870s (0.890s wrapper):
all282 declaration pins, four retained files, exact saved metadata/configuration,
original recovery invocation and complete journal payload pass. MainPID873316/
invocation56e692c47ef348ff8acd60089b7b46df is unchanged before and after the check.
At this milestone the recovery observer was terminal success; learner, progress,
postprocessing and collection continued. No observer, GPU stage or checkpoint
creation is repeated. CPU co-host costs remain uncorrected.

Recovery review SHA118856b1, retention SHAf9cf5bd5, metadata SHA65a62fc9.
Confirmation is `recovery-confirmation.json` in the trial. Frozen gameplay was
unmeasured at this milestone; the original fixed200k learner and four assessments
subsequently completed as recorded above.
