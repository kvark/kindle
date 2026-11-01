# Qbert dynamics-group1e-4: frozen gain awaits repaired CPU audit

The [Qbert dynamics-group1e-4 comparison](../experiments/2026-11-01-cdp-qbert-dynamics1e4-200k.md)
finishes allfive GPU stages:200k actions/18,715 updates in25m40.027s,thenfour
frozen arms. Preliminary selected24-episode means aregreedy15,471.875 and
sampled14,862.5 versus12,678.125/12,093.75;own initials168.750/167.708 match.
Pyramid/replay/final closure checks remain pending; do not claim an audited gate.

Original mainPID857991/invocation1e44edc5d824407b8a21cd62657f6ee2 fails23:00:53
UTC November1 in its CPU audit:undefined`CONTROL`,before any replay. Original
collection fails;postprocess exits with`postprocess-incomplete.json`. Preserve
these failures. No GPU stage failed or repeats; original100k/9,340 recovery passed.

Separate CPU repair`kindle-qbert-dynamics1e4-cpu-repair-20261101` launches23:06:15
UTC/source240bb89,PID862557/invocatione48562cf43cd4ba8927950760ef61922.
Its3.959s qualification passes606 original/111 repair pins and verifies that
CONTROL is the sole missing audit global. It injects the exact retained-control
path without editing the original source; native/emulator construction is blocked
in the parent. Missing frozen/training replays,plots andcomparison run once.
Two failed preparation checks are preserved:import-versus-construction and raw
journal JSON equality; corrected checks retain original journal bytes and compare
decoded records. Original services stay failed; no original completion is created.
Root`/x/Temp/kindle-qbert-dynamics1e4-20261101/full/cpu-repair`;declaration77ce1326.
No new learner,confirmation,rate sweep,extension or GPU successor is selected.
The five-game goal remains unfinished. Do not restart the original services or
repeat completed training/evaluation/replay stages; check live repair state.


The [fixed bounded check](../experiments/2026-11-01-cdp-qbert-dynamics1e4.md)
completes8,192 actions/32,756 frames/733 updates in **57.461406s**. Frozen1,024
greedy actions take.853438s withzero updates andno natural completed episodes. All616 pins,357 matching initial
and invariant frozen tensors,47 pre-update semantic events, finite reports,
full guided replay and host guards pass. Short frozen execution does not
establish task competence.

Conservative200k projection is **1,543.153s (25m43.153s)** with10% margin,
below30min. This is an early estimate, not a measured full-run speedup or
gameplay improvement. A fresh full comparison requires separate selection;
bounded weights and experience will not be reused.

Only dynamics-group4e-4→1e-4 changes from retained Qbert12011. Keep centeredCDP/
12M64/Auto/N8/B16/T16/H5/R24, encoder/base/actor4e-5,greedy collection,sampled
posteriors,expert100/unit weights/world gradients and the same pixel guidance.
The group contains recurrent dynamics,posterior,future-prediction andexploration
parameters. This does not isolate the recurrent core or prove a mechanism.
Learner12011/evaluation4000012011 are reused development seeds.

Native447fe50/SHA35ad0db and fast teacherfa261c53 stay unchanged. Private runner
e161bf38 adds exactly the12 previously qualified dynamics-override lines to
Qbert runneraba4300d, with no other edits. The production runner and its471-test
qualification remain unchanged. CPU preflight1.542s passes two actual commands,
22 config cases, both357-tensor shape references,562 control/64 preceding-ranking
pins,16 runtime files and733/9340/18715 cadence without native or emulator
construction. No build occurs.

Original main succeeds **22:27:02.719 UTC November1**, source1a1eceb, PID857526/
invocation655eec1976bd483e9f5b780a773a69bc. Original collectorPID857531/invocation
a0cc6a65e7e54a1d9b582e6791eac825 completes **22:27:09.257**. Both services
are terminal success; no original experiment stage fails or repeats. No GPU
allocation remains active at closure. Root
`/x/Temp/kindle-qbert-dynamics1e4-20261101/native-check` is complete; do not rerun.
No retry, cap increase, extension, rate sweep or automatic successor is selected.
The five-game goal remains unfinished.


All733 reports are supervised and finite;175,835 labeled replay rows equal
their weight sum. Actor/world expert loss differs by at most8e-7. CNN,reward,
actor andvalue weights change. Full replay verifies3,068 actual overrides
(37.4512%); these aided actions and online scores are not frozen competence.

| Bounded artifact | SHA256 |
| --- | --- |
| Declaration | `85331eb6c7906f65d4023f297adcf13535f7ba8881d3b119a012c6ce88f7c41e` |
| Review | `a1c431c5583f84949f14d77031ebe9fd358ede97e9c9971b7d3a9679511b223e` |
| Original completion | `76b18e1361ae88ff627696e298164a5085cf5deb511da30fd38f6e2226b4e56e` |
| Guided replay | `7a3e3643f2ac4519bb5bb1a6dbbf5190f1bebaa373bdd01691c1089fd47b9683` |

## Separate full comparison is CPU-qualified

The [fresh200k comparison](../experiments/2026-11-01-cdp-qbert-dynamics1e4-200k.md)
passes its repaired CPU preparation22:30:48.848 UTC November1/sourcec268129 in
4.011s (4.105s wrapper). Five actual commands,32 config/five teacher/eight mode/
17 header/four observer/12 recovery cases, six retention/two actual-collector
fixtures,49 paths/17 scripts andall616 bounded/562 control pins pass. No native
agent or emulator is constructed; no full learner has launched at qualification.

The first1.565s CPU attempt fails before construction because a service-prefix
replacement also rewrites the retained control path. Original sources and failed
receipt remain under `cpu-preparation-failure-1` and`full-preflight-execution.json`.
Four unselected controller references are corrected; generator restores the
protected control after all service-prefix substitutions. The repair receipt
and original failure are retained and pinned. No bounded/GPU stage repeats.


## Fresh full comparison is active

The separate200k learner launches22:32:25.422 UTC November1/source0d8ec52 with
606 pins, originalPID857991/invocation1e44edc5d824407b8a21cd62657f6ee2. All four
original observers start. Startup22:33:29.495 passes357 bounded/control initial
tensors,47 matching pre-update transitions, correct cadence and723 finite reports
through8088 actions. Its1.711s CPU wrapper is retained as co-host timing, without
correction. No final frozen result is available. The original bounded stages
remain complete and are not repeated or used as training experience.

Fullroot `/x/Temp/kindle-qbert-dynamics1e4-20261101/full`; service
`kindle-qbert-dynamics1e4-200k-20261101`. Preserve original main/observers and
fixed200k/30min,100k recovery and four24-episode assessments. Do not restart or
overlap another GPU allocation. The five-game goal remains unfinished.


The separate [current training-cost summary](2026-11-01-cdp-current-training-cost.md)
reads four completed learners while this original run continues. Its first
1.894s CPU wrapper fails on an overstrict timer-sum assertion;the separate
4.289s analysis (4.308s wrapper) accounts explicitly for inter-stage time and
passes22:42:05 UTC. All21 analysis/606 active inputs and originalPID857991/
invocation1e44edc5d824407b8a21cd62657f6ee2 remain unchanged. No native/emulator
import,GPU work,update or completed stage repeat occurs. Retain both co-host
costs without correcting this learner's wall time.


## Original100k recovery passes

The original recovery observer succeeds22:45:31 UTC November1. The declared
100,000-action/9,340-update checkpoint has357 finite tensors. Allfour retained
files total206,067,159 bytes and preserve the source; retention takes.308s.
MetadataSHA256`dc674aca3829e78eee532d2df278c997669d7b54a5d594cac31cd4f1b860e4e0`,
retention receiptSHA256`768d710980f8109c863c75ae829d612597f8b808dbb3ee05f4dee0a33f4da749`.
The observer also checks9,663 sequential finite reports through103,448 actions.
A separate read-only check verifies allfour retained hashes and the original
invocation32e0b7e1cf72435388c4a7225db0e41c journal's single audit/completion.
No copy,audit,collector or GPU stage repeats.

Recovery service is terminal success; original mainPID857991/invocation
1e44edc5d824407b8a21cd62657f6ee2 and remaining observers continue. These are
weights/metadata only; replay,live beliefs,games and RNG are not retained.
No intermediate evaluation or checkpoint selection occurs. Final200k/18,715
updates/30min and allfour frozen assessments remain unchanged and pending.
Keep CPU co-host costs without wall-time correction; no restart or overlap.
