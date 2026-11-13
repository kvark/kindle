# Breakout H5/R24 lowers frozen scores despite better replay imitation fit

The fixed200k-action comparison scores **42.333 sampled /76.083 greedy**,
below the retained H15/R16 control's **56.292 /104.667** by24.80%/27.31%.
Every final arm has **0/24 two-wall successes**. Training takes **22m50s**,
6.42% longer than the control's21m28s. Retain H15/R16; no extension, cap
increase, confirmation learner or automatic Breakout successor is selected.

| Frozen mode | H15/R16 control | H5/R24 candidate | Own actual initial, both | Two-wall successes, both |
|---|---:|---:|---:|---:|
| Sampled, primary |56.292|42.333|1.417|0/24|
| Greedy |104.667|76.083|2.208|0/24|

Both actual-initial arms also have zero successes. Initial-subtracted return per
training minute falls2.557→1.792 sampled and4.774→3.235 greedy. Four frozen arms
total76,224 actions/304,749 frames/55.127s,
with zero updates and truncations. Select the first three natural episodes in
each of eight lanes; excess episodes and partial tails remain in the evidence.

## Learning and cost

Training completes200,000 actions/799,605 emulator frames/18,715 updates
in1370.360s, under the30-minute cap. Change H15/R16→H5/R24
jointly: approximately50% more world/expert updates and50% fewer scheduled
imagined transitions, with changed cadence, sampling and action-clock warmup.
Keep12M64/Auto/N8/B16/T16/encoder4e-5/base4e-5/dynamics4e-4, sampled collection
and posteriors, actor-unimix.01, expert100/unit action weights/world gradients,
all-state corrective labels,20% episode guidance/cap1920,full18/sticky.25/repeat4
and one GPU resize64. Native447fe50,runner6c1b69f and teacher remain unchanged.

Final50k replay expert CE improves0.558097→0.380887
while frozen control worsens. Actual teacher overrides fall99,343
(49.6715%)→86,942 (43.471%).
Aided online last50 return is76.820, versus103.800.
Replay fit and online guided returns are not held-out accuracy or unaided
competence. Neither the loss change nor this joint recipe comparison identifies
the causal bottleneck. Learner1009/evaluation4000001009 are reused development
inputs. No isolated horizon effect or seed reliability is claimed. The earlier
negative H5/R32 Breakout result remains; Qbert's H5/R24 gain does not transfer
to this Breakout comparison.

## Verification and closure

All301 declared inputs,100k/9340 recovery and preserved metadata,357 finite and
matched initial tensors,47 matching pre-update events,357 invariant tensors per
frozen arm, finite supervised reports and changed CNN/reward/actor/value pass.
The actual-initial sampled/greedy semantic traces match800/795 events. Complete
training pixel/action/teacher-label replay, frozen outcome/frame/reset replays
and videos, host guards, curves, exposure and comparison pass. The training
replay takes98.198s. Ordinary frozen logs lack live image hashes; their replay
checks recorded outcomes rather than direct live-image equality.

Main completes **10:08:37 UTC November1**, postprocessing10:08:46,
collection10:09:18, and independent2.171s terminal confirmation10:09:38.
All six services succeed, with zero failures or repeated original/GPU stages.
Original sourcec4fa576,PID777532/invocationacf63b2de9614a9dab625bcc6f6cbe52.
The comparison plot is visually inspected. No GPU allocation remains active at
this closure; inspect live state before new work.

Startup and supplemental checks, the1.133s Qbert seed preparation,3.729s full
CPU preflight (5.333s wrapper),1.285s terminal-verifier preflight (2.847s wrapper)
and light source review co-host training. Retain these timings without correction.
The Qbert preparation remains a separate decision after closure.

[Full protocol](../experiments/2026-11-01-cdp-breakout-h5-r24-200k.md).
Evidence root`/x/Temp/kindle-breakout-h5-r24-20261101/full`,trial
`breakout-12m-64-b16-h5-r24-encoder4e5-expert100-200k-seed1009`.
Do not rerun the learner, collectors, confirmations or bounded stages. The
five-game goal remains unfinished.

| Receipt | SHA256 |
|---|---|
| Declaration |`8c7daae2a9bc85b6414f582e43f13f17aca0df050f4a881798e17f742cac8d41`|
| Learning review |`faffe8967b9e0d1b3f0705f9b42ea4b439c433f2951e15f3efcb21d9e4876ec1`|
| Comparison |`eb8988ef3f34a73add2b75dd674751c381fc5dfb7e664ae6f6da3e7c24bd4320`|
| Completion |`c3d72ef97f76c624e74030f8110a418bca995f704707ef1fc86911fa71784fbb`|
| Independent summary |`5d29d112cc3da00ae0a63bbdcb6dd95c12d9a8ede96d9e0f08195c1fb1a1c468`|
| Terminal confirmation |`aa84babe7ea32d9bbc86a60a878d861c7a40c028c01c9995708b5bc03e3dbcba`|

## Bounded execution and cost checks

The fresh encoder4e-5 check completes **8,192 actions /32,723 emulator frames /
733 updates in50.971s**. The following1,024 frozen actions /4,096 frames take
0.884s,withzero updates and no completed natural episodes. This establishes
execution and early cost,not task competence.

The linear200k projection is20m44s;the fixed10% margin gives **22m49s**,below
the30-minute gate. A full learner's cost and frozen benefit remain unmeasured
at this stage. The separately selected
[fresh200k comparison](../experiments/2026-11-01-cdp-breakout-h5-r24-200k.md)
uses no check experience.

Change H15/R16 → H5/R24 jointly while preserving12M64/Auto/N8/B16/T16,
encoder/base4e-5,dynamics4e-4,sampled collection/posteriors,actor-unimix0.01,
expert100/unit action weights/world gradients/all-state corrective labels,
20% episode guidance/cap1920,full18/sticky0.25/repeat4 and one GPU resize64.
Native447fe50,exact runner6c1b69f and teacher remain unchanged. More world/expert
updates,shorter imagination,cadence,sampling and warmup duration change together.
The older negative Breakout H5/R32 result is retained;Qbert's H5/R24 benefit is
not assumed to transfer. Reused1009/cohort does not establish reliability.

All267 pins,357 finite/matching initial tensors,47 matching pre-update semantic
events,357 invariant frozen tensors,733 finite supervised reports,changed
CNN/reward/actor/value weights,full training pixel/action/teacher-label replay
and host guards pass. Actual overrides are3,800/8,192=46.387%,distinct from20%
episode selection. The original service succeeds **09:36:47 UTC November1**,
source1b4671b,PID777123/invocation0d189ef7da054ce0866a40c9b9a0cc4b,after1m08s.
Terminal collection succeeds **09:37:06**,verifying the original invocation
and all267 inputs. No failure,retry,GPU repeat or cap change occurs.

The0.636s bounded CPU preflight checks two actual CLI paths,15 config cases,
357 schema tensors and227 control inputs. The subsequent1.433s full preflight
checks five CLI paths,24 config cases,10 header cases,four observer-parser cases,
seven recovery-body cases,44 paths and15 scripts,plus267 bounded/227 control
pins. Both forbid native/emulator construction. Light source reading co-hosts
the bounded service;retain its timing without correction. No build occurs.

[Bounded protocol](../experiments/2026-11-01-cdp-breakout-h5-r24.md).
Root `/x/Temp/kindle-breakout-h5-r24-20261101/native-check`,trial
`breakout-12m-64-b16-h5-r24-encoder4e5-expert100-check`.
The check and collector are complete;do not rerun them. A full learner is a
separate selection,with its original two-wall gate and fixed actual-initial
comparisons. No allocation is active at bounded closure;check live state.

| Receipt | SHA256 |
|---|---|
| Declaration | `dd871c3848cadf49aa20f60b55162d4c06464933ff9b8b19d43ca840290f4cb6` |
| Review | `0c5f1040829d37a03ed7b111103b2b9192774cb0e064237810404883773d033b` |
| Guided replay | `e09808444c95282a340e2e264fc0f2edd22cb2fd9bb3f73bbe7a8340206ac053` |
| Completion | `9a7e92a47d0eddfd3902de1479386270acf0bf2bd9a4394ea92bdbb65a390ba5` |
