# Qbert expert scale100 at the cheap fixed budget

Select one fresh4096-action bounded check. Expert10/100k greedy policies score
7700/24 first pyramids on4013 and3832.292/23 on1009 in18m55s/19m11s. Both miss
mean15000. The paired4013 scale1→10 comparison improved frozen learning at the
same budget; this motivates one further coefficient test, not an assumption
that stronger supervision must help.

Change expert scale10→100, affecting actor and world supervision together.
Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor-unimix0/full18/original rates,
20%episode continuation/cap1920/all18 action weights1/world gradients/all-state
corrective labels,4013/eval4000004013. Native447fe50, packaged runtime and exact
runner/teacher remain unchanged. Native max-pooled frames get one GPU resize.
Do not extend either completed life or reuse check experience.

The bounded check requires4096 actions/963 updates,first248/debt0,then1024 greedy
frozen actions/zero updates. Audit finite reports,357 control-matched initial and
invariant frozen tensors,changed CNN/reward/actor/value,equal labeled-row/weight
sums,aligned raw world/actor expert losses,passive labels,actual disagreements,
full teacher-label/input-pixel/actual-action replay,source/config/counters and
host guards. Raw losses exclude the coefficient; config must explicitly show100.
This checks execution and early cost, not natural-episode competence.

Caps:training120s,frozen60s,CPU audit120s,owning service360s. No retry,cap increase,
automatic full promotion or checkpoint selection. Root
`/x/Temp/kindle-qbert-expert-scale100-20261028`, service
`kindle-qbert-expert100-check-20261028`. Pin inputs before launch. Previous
expert10-seed1009 services are terminal success; host idle and kernel log clear
at selection. Check live state again before GPU work; stop on real faults or
nonfinite reports, with no driver recovery.

A fresh100k comparison can be selected separately after this check and its
measured cost pass review. It must retain sampled and greedy final/actual-initial
natural-episode assessments and both original mean15000/90% gates. Reused4013
and its cohort remain development evidence. The five-game goal is unfinished.

## Bounded launch

CPU preparation initially stops on a copied `gpu_stages_repeated` field absent
from the full-run control's completion schema. The original script and failure
receipt remain external. The corrected check verifies the actual terminal
service states and completion-file hashes; no GPU stage had started. Both
commands then parse without native/environment construction,84 control pins
remain unchanged,and36 check inputs are pinned.

The service starts20:26:52 UTC on October28/source`6a144ee`, PID199497/invocation
`d9ed7d25684341e1857a79106c5fdf82`; declaration
`9b3e5f57a21d9e4fabdb41aa55bf240a7f87a7624701ca19aff7311918670bc0`.

## Completed bounded check and separately selected100k comparison

4096 actions/16,381 frames/963 updates take43.170s;1024 greedy frozen actions/
4096 frames take.497s,zero updates/no natural episodes. All36 pins,357 matched
initial/invariant frozen tensors,finite reports,changed CNN/reward/actor/value,
equal115,504 row/weight sums,raw expert-loss alignment (max8e−7),full label/pixel/
action replay and host guards pass. There are823 hop labels,2048 passive labels,
1579 actual/teacher disagreements and2048 overrides. Service succeeds20:27:57
UTC,whole1m05.506s. The CPU preparation failure above remains; no GPU repeats.

Early cost projects17m34s/100k,close to the preceding scale10 check. Select one
fresh100k-action/24,939-update life,under a30-minute training cap,with the exact
bounded recipe. Only expert scale10→100 changes against retained4013/100k.
No weights,replay,belief,games or RNG are reused. Save/audit50k recovery at12,439
updates and fixed final; no checkpoint selection or extension.

Evaluate fixed final and own actual initial using both sampled and greedy native
actions,without guidance/labels. Primary greedy comparison is against7700/24;
sampled5572.917/24 remains explicit. Each of4 arms selects first3 natural
episodes per8 lanes=24,seed4000004013,original mean15000/90%-pyramid gate,zero
updates/truncations,357 invariant tensors,full outcome replay/videos and guards.
Require357 matching initial tensors and paired initial semantic traces for both
action modes. Ordinary frozen logs do not contain live pixel hashes.

Require complete training label/input-pixel/actual-action replay,finite reports,
equal row/weight sums,source/config/counters,curves,coverage,exposure and paired
comparison. Guided online scores are not unaided competence. Reused seed/cohort
and earlier development costs remain explicit.

Caps:training1800s,each frozen arm300s/600k actions,training replay600s,CPU
audit1200s,owning service4500s. No retry,cap increase,extension or automatic
successor. Full root`/x/Temp/kindle-qbert-expert-scale100-20261028/full`,service
`kindle-qbert-expert100-100k-20261028`. Pin exact inputs and check current host
before launch; all preceding services are terminal at selection.

## Full launch

CPU preparation parses5 commands,compiles11 controllers and verifies the saved
control and357 initial tensors without native/environment construction. The
declaration pins111 inputs. Main service starts20:29:26 UTC/source`a6ee6da`,
PID199894/invocation`1ae99e7bbe5f43a4868074fc15338a80`; declaration
`c358c45d2517a252343eedfa3ce5e1c9a25ad4b67719c5d2e1971e2ff281b891`.
Startup passes at944 actions/173 finite supervised reports,all111 pins,357
initial tensors matching both control and check,exact config and first6 updates.
50k recovery,progress and terminal postprocessing are automated. No frozen task
outcome yet; no extension or automatic successor selected.
