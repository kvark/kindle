# More Qbert experience at nearly the same update count

Select one fresh8192-action bounded check at replay ratio16. Retained expert100/
100k/R32 scores8365.625/24 first pyramids greedily and7339.583/22 sampled in
18m59s. Both miss mean15000. The last coefficient increase gives a modest gain;
this candidate instead allocates a similar update count across more experience.
The existing R16 results used other supervision settings and do not establish
the outcome here.

Keep4M48/Auto/N8/B8/T16/H15/coefficient1/actor-unimix0/full18/original rates,
20%episode continuation/cap1920/expert100/all18 action weights1/world gradients/
all-state corrective labels,4013/eval4000004013. Native447fe50, packaged runtime,
runner and teacher stay identical. Native max-pooled frames get one GPU resize.
Only ratio32→16 changes in this bounded check. A later fresh200k/R16 comparison
would change experience and ratio together against100k/R32:24,970 versus24,939
updates, not an isolated replay-ratio or experience effect.

Check8192 aggregate actions/994 updates,first update248/debt0,then1024 greedy
frozen actions/zero updates. Require finite reports,357 control-matched initial
and invariant frozen tensors,changed CNN/reward/actor/value,equal labeled-row/
weight sums,aligned raw world/actor expert losses,passive labels and actual/
teacher disagreements,complete label/input-pixel/action replay,source/config/
counters and host guards. Raw losses exclude the expert coefficient. This is
execution and early-cost evidence, not natural-episode competence.

Caps:training120s,frozen60s,CPU audit180s,owning service420s. No retry,cap increase,
automatic promotion or reuse of check experience. Root
`/x/Temp/kindle-qbert-expert100-r16-20261028`,service
`kindle-qbert-expert100-r16-check-20261028`. Pin inputs before launch. Previous
expert100 services are terminal success; host idle and kernel log clear at
selection. Recheck live state before GPU work; stop on real faults/nonfinite
reports, with no driver recovery.

A fresh200k comparison may be selected separately after this check and measured
cost review. It must fit a30-minute training cap,retain fixed final and own actual
initial sampled/greedy assessments and original mean15000/90%-pyramid criteria.
No completed life is extended. Reused4013/cohort remain development evidence;
the five-game goal remains unfinished.

## Completed bounded check

The service starts20:59:41 UTC/source`b2caaad`, PID203058/invocation
`4e6d8dd358bc4f6d994a57afb09e7702`. Declaration
`2437330c15af390815ded64fa9702e56fbc47bfffb0d7a89e6be9b89e63f5c34` pins36
inputs; CPU preparation parses2 commands without native/environment construction
and verifies all111 control pins.

8192 actions/32,758 frames/994 updates take60.947s.1024 greedy frozen actions/
4096 frames take.505s,zero updates/no natural episodes. All36 pins,357 matched
initial/invariant frozen tensors,finite reports,changed CNN/reward/actor/value,
equal119,176 row/weight sums,raw world/actor loss alignment (max8e−7),full teacher-
label/input-pixel/action replay and guards pass. Labels include1778 hops,3682
passive states and2906 actual/teacher disagreements;4510 overrides. This is
execution and early-cost evidence only. No failed attempt,GPU repeat or runtime
edit. Main succeeds21:01:17 UTC on October28,whole1m36.076s.

## Separately selected fresh200k comparison

The early rate projects24m48s/200k. Select one fresh200k-action/24,970-update life
under a30-minute training cap, keeping the bounded recipe. Compare with retained
expert100/100k/R32 at24,939 updates/18m59s. Budget and ratio change together;
compute is nearly equal in update count, not wall time. Report any task gain
alongside the extra wall cost. No weights,replay,belief,games or RNG are reused.
Save/audit100k recovery at12,470 updates and fixed final; no checkpoint selection.

Evaluate fixed final and own actual initial with sampled and greedy native
actions, guidance/labels off. Greedy is primary against8365.625/24; sampled
7339.583/22 remains explicit. Each of4 arms selects first3 natural episodes per
8 lanes=24,seed4000004013,original mean15000/90%-pyramid criteria,zero updates/
truncations,357 invariant tensors,full outcome replay/videos and guards. Require
357 matching initial tensors and paired actual-initial semantic traces in both
modes. Ordinary frozen logs do not contain live pixel hashes.

Require complete training label/input-pixel/actual-action replay,finite reports,
equal row/weight sums,source/config/counters,curves,coverage,exposure and paired
comparison. Guided online returns are not unaided competence. Reused seed/cohort
and all preceding development costs remain explicit.

Caps:training1800s,each frozen arm300s/600k actions,training replay900s (outer
timeout930s),CPU audit1500s,owning service4800s. No retry,cap increase,extension or
automatic successor. Full root`/x/Temp/kindle-qbert-expert100-r16-20261028/full`,
service`kindle-qbert-expert100-r16-200k-20261028`. Pin exact inputs and check live
state before launch. At selection the bounded check and all predecessors are
terminal success, with no active GPU allocation.

## Full launch

CPU preparation parses5 commands,compiles11 controllers and verifies retained
control/357 initial tensors without native/environment construction. Declaration
pins138 inputs. Main starts21:02:45 UTC/source`0d73300`,PID203447/invocation
`e042da0cefc4457f826202e4d6a37661`; declaration
`dbd098bed2c7278622c467b9e305e688dae67249869c7c528e16a38a403eecd0`.
100k recovery,progress and terminal postprocessing are automated. No frozen
outcome yet; no extension or automatic successor selected.

Startup passes at4256 actions/501 finite supervised reports:all138 pins,357
initial tensors matching control and check,exact config and first6 updates.

Two affined CPU-only Breakout staging checks overlap this life:1.734s guarded
tests and30.982s recorded-reference parity (parity service21:09:03–21:09:34 UTC).
All138 active inputs and the same learner PID/invocation remain unchanged. Tests
construct no native agent; parity imports no native module. These are co-host
timing caveats, with no correction to measured training time. No production or
active-runtime inputs change. See the separate Breakout corrective-label protocol.

The100k recovery audit succeeds21:16:21 UTC at12,470 updates,357 finite tensors.
Its complete log prefix has103,728 actions/12,936 finite reports. The recovery
watcher exits successfully without changing the learner.
