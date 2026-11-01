# Breakout H5/R24 passes bounded execution and cost checks

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
