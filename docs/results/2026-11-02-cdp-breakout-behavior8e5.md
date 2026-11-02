# Breakout actor/critic-rate comparison

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


## Fresh 200k learner active; gameplay result pending

The first GPU launch is **00:34:24.392 UTC November 2/source3099ff7**. Main
PID873316/invocation`56e692c47ef348ff8acd60089b7b46df` remains active. Original
postprocessing/progress/recovery/collection PIDs are873319/873322/873451/873454,
with identities retained in `observers.json`. All training/evaluation caps and
settings remain fixed. Declaration SHAd8d4f582 pins282 inputs, including the
preserved original CPU failure and separate correction.

The read-only startup check passes **00:34:37.150** in1.019s wrapper. All282 pins,
357 finite control/bounded-matching initial tensors,47 matching pre-update
transitions, exact configuration/native/runner/teacher, original process identity
and fresh counters pass. First six updates occur at376/392/408/424/440/456;
77 finite supervised reports cover the complete prefix through1,608 actions.
This verifies startup only. No full frozen score or gameplay gain is available.

Retain the original service and four observers,100k/6,227 recovery and final/
actual-initial sampled/greedy assessments. Do not restart or overlap GPU work.
No extension, rate sweep, confirmation or automatic successor is selected.


## Terminal reader prepared while the original learner runs

A separate read-only result reader passes preparation at **00:37:52.447 UTC
November 2** in0.087s (0.113s wrapper). It reproduces all four retained control
scores from the first three natural episodes per lane and rejects four altered
score/gate/environment cases. It checks existing replay/log/video hashes without
constructing a native agent or emulator. The original learner identity remains
unchanged before and after the check; retain this CPU co-host cost without time
correction.

After the original main, postprocessing, progress, recovery and collection
services become terminal, run the package's `review_full_terminal.py terminal`
once through `run_cpu.py` with a new receipt name. It verifies all282 inputs,
completion-file hashes, original invocation journals, the collector payload,
retained100k checkpoint, paired initial traces, four frozen scores and the
preserved CPU prelaunch failure. Preparation is not a terminal result, and the
reader must not replace or rerun any original experiment stage. ReaderSHA
`ba571025beca4480f21224292cb0f1e08e9faa67c94c181dd281174060c76f28`;
identity-helperSHA131bee46. Preparation receipt: `terminal-reader-preflight.json`.
