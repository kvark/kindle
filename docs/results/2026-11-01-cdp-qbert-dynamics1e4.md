# Lower dynamics-group rate improves frozen Qbert at nearly unchanged cost

**Completed with a separate CPU audit repair:** lowering dynamics-group rate
4e-4→1e-4 raises greedy return **12,678.125→15,471.875 (+22.04%)** and sampled
return **12,093.750→14,862.500 (+22.89%)**. Both final modes clear the first
pyramid in24/24 episodes. Greedy passes the unchanged15k/90% gate; sampled
still misses the mean threshold. Training takes **25m40.027s**, only.83% longer.
Retain this useful development point; learner12011/evaluation4000012011 were
reused,so this is not independent confirmation or broad reliability.

| Dynamics-group rate | Greedy primary | Sampled | Training time |
| --- | ---: | ---: | ---: |
| Retained4e-4 |12,678.125;24/24|12,093.750;24/24|25m27.366s|
| Candidate1e-4 |**15,471.875;24/24;pass**|14,862.500;24/24|25m40.027s|
| Actual initial,either rate |168.750;0/24|167.708;0/24|—|

The original main and collector remain failed because the CPU audit referenced
an undefinedCONTROL path after allfive GPU stages completed. The separate
297.912s CPU closure supplies all missing audits and reporting without repeating
training or frozen evaluations. Its status is`complete_with_cpu_audit_repair`;
there is no original passing completion. See the final section for exact failure
and repair provenance. No successor learner is selected here.

The bounded qualification below is historical and remains complete.

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


## Completed full comparison and repaired closure

Fresh training completes200,000 actions/799,807 frames/18,715 updates in
1,540.027384s versus1,527.365963s. The four frozen arms total138,760 actions /
554,893 frames /144.341s,zero updates/truncations. Every score uses the first
three natural episodes per eight lanes,guidance off. Actual-initial means and
all1,176 sampled/1,223 greedy semantic events match the retained control.
All357 initial tensors match both bounded and retained controls; all357 tensors
remain invariant in every frozen arm. The100k/9,340 four-file recovery remains
retained with verified hashes.

Initial-subtracted return per training minute rises491.410→596.215 greedy
(+21.33%) and468.494→572.514 sampled (+22.20%). The final50k-action replay
imitation CE instead worsens.044040→.069685. Better frozen gameplay alongside
worse imitation fit reinforces that aggregate training loss is not a competence
measure; it does not identify the causal optimizer mechanism. The rate jointly
changes recurrent dynamics,posterior,future-prediction andexploration groups.
Native447fe50/SHA35ad0db,teacherfa261c53,model geometry,horizon,ratio andother
settings stay fixed; private runnere161bf38 has only the qualified12 CLI lines.

All18,715 reports are finite andsupervised,with4,488,424 labeled rows equal to
target weight sum. World/actor expert losses agree within8e-7. CNN,reward,actor
andvalue parameters change. Every training pixel,action,corrective label,reward,
reset and outcome reproduces. Actual overrides47,700 (23.85%) exceed control
44,539 (22.2695%) on their changing trajectories; online last50 means15,505.0/
15,340.5 include this guidance and are not unaided scores. All frozen replays,
videos,host guards,curves,coverage,exposure and fixed comparison pass. Ordinary
frozen source logs lack live-image hashes; do not claim unavailable live-pixel
parity. The comparison plot is inspected and labels online versus frozen results.

Original main fails **23:00:53.476 UTC November1** on undefinedCONTROL while
constructing the CPU audit result,before any replay. The original postprocess
exits with its incomplete-review receipt; original collection fails explicitly.
Those statuses,journals andsources remain unchanged. The qualification missed
this free-global error; the repair now checks referenced audit globals before
and after supplying the retained-control path. Original source bytes are not
edited. CPU repair preparation preserves two further failed attempts: imports
were incorrectly forbidden despite no construction,and raw JSON journal bytes
were compared despite unstable field order. Constructor guards and decoded-row
comparison fix these preparation checks without changing experiment data.

The separately declared CPU service launches23:06:15/source240bb89 and succeeds
**23:11:13.343 UTC**,PID862557/invocatione48562cf43cd4ba8927950760ef61922.
Its **297.912s** controller completes missing audit288.804s,plot1.168s,coverage
.019s,comparison4.311s andexposure.778s. All606 original/111 repair inputs remain
unchanged,including completed logs/checkpoints/guards and original failure
receipts. No completed GPU,replay,postprocess orcollector stage repeats.
`full/cpu-repair/completion-review.json` is the separate closure; the original
trial has no`completion-review.json` and original main/collection remain failed.

A subsequent read-only terminal summary first fails because journalctl's default
JSON output represents the long completion MESSAGE asnull. The failure andsource
are retained. A separate reader uses`--all`,verifies the exact logged completion,
all five repair stages,original identities/states,unchanged pins/files and the
24-episode scores. It passes in2.050s at23:12:15 UTC; no original stage repeats.
The earlier original-success summary remains unexecuted. CPU document/source
review and the recorded timing diagnostic co-hosted training; preserve all
measured wall times without correction. Repair and terminal-summary time occur
after GPU work and do not alter the25m40.027s training cost.

All original services andthe CPU repair are terminal;no Kindle GPU allocation
is active atclosure. Retain1e-4 as a promising development recipe,but do not
combine its reused12011 result with the distinct4e-4 seed10007 as two learners
of one fixed recipe. Fresh-seed confirmation remains open. No extension,rate
sweep orautomatic successor is selected. The five-game goal remains unfinished.
Root`/x/Temp/kindle-qbert-dynamics1e4-20261101/full` iscomplete;do not rerun.

| Artifact | SHA256 |
| --- | --- |
| Repaired learning review |`afd992829f94e45a417db803716c556f79e1aadcd25afc310ad3b62cbb2c315e`|
| Full guidance replay |`3ee11705beb9bd650f400e9f4777f8fa380a8f4e6addf5c53b62c55a9c9f874d`|
| Fixed comparison |`f733b7cba1c3d5794e9170e8d7a6a86d039b3d49f564caeb69d47376511d9815`|
| Separate CPU closure |`8f9d7c25efd4a9f3ef2056792d7377995372ac84274f9d7648538f7fe1382397`|
| Read-only terminal summary |`a82186f3a35373c8abb5f1b34e4efb8af1be9e5eaee20d126e5cf38d4399520e`|
