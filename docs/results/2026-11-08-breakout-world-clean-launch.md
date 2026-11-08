# Breakout: all18 commands collected from clean waiting states

The [focused fitting collection](../experiments/2026-11-08-breakout-world-clean-launch.md)
completes288 branches from eight fresh fitting seeds. Allnine FIRE commands
produce a finite ball reference at the first decision in every retained repeat;
allnine non-FIRE commands remain without one through32. The fixed native preview
also shows launches and movement. No world-model training or forecast occurs;
this establishes data availability, not learned competence.

Root `/mnt/data/kindle-breakout-world-clean-launch-20261108`.

## Data and cost

Use seeds951001+1009/lane and32-arrival prefixes:reset,2/3/4/5 LEFT commands
for lanes0–3 or RIGHT for4–7,then NOOP to31 actions. All prefixes retain constant
positive lives, absent ball references, zero reward/endings and16 final NOOP
commands. No exact native16/32-history match exists with any of56 retained
development prefixes. This is not an untouched test or independent learner.

Paddle references end at60,24,8,8,144,144,144,144:onlyfour distinct positions,
not eight. Keep all declared seeds/parents without replacement. Different source
transcripts and sticky RNG do not imply distinct physical states; exact native
histories numberfour at16 arrivals andfive at32,as retained in
`closure-review.json`.

Every parent branches all18 constant commands for32 arrivals, twice with sticky
probability.25 and repeat4. All288 branches reach32 with no game over;some lose
lives or relaunch. Outcomes are retained, not filtered for successful rebounds.

|Work|Executed actions|Emulator frames|Explicit resets|
|---|---:|---:|---:|
|New fitting branches|9,216|36,864|0|
|Source prefixes|248|992|8|
|Private sticky-action reconstruction|8,928|35,712|288|
|Total|18,392|73,568|296|

Collection takes35.002455s, including native construction, resize and serialization.
Eight environments share the retained small64 final16,384 camera,exact native/
source/runner7f797a2b,SHA38680c73,Auto. RGB is native max-pooled160×210 with
one GPU resize to64;no CPU training resize. Native weights,moments and all
metadata/ages stay identical across source/before/after240 tensors. No model
prediction, derivative, optimizer, build or backend qualification occurs.

## Verification and limits

The new collection passes397 predeclared pins. Independent CPU review in4.701s
checks all288 combinations,each32-arrival prefix,source transcript,seeds,RGB
hashes,incoming commands/outcomes,terminal tails,frame differences,source/replay/
new counts,240 tensors and metadata/ages. The service terminates successfully;
its recorded invocation and existing host guard/9 sealed files pass. Closure
retains764 input/data/runtime/guard/review/preview pins in.045111s.

`preview.png` shows fixed lane0/draw0 native frames at0/1/4/8/16/24/32 for
commands0/1/10/11/12/17. No favorable repeat is chosen. RAM annotations remain
metadata, never model inputs or training targets. Finite references can be stale
or offscreen; the restricted detector and selected early states cannot establish
full-game physics,collision adequacy,event calibration or later brick memory.

The completed collector stays closed. A corrective learner requires a separately
declared fixed budget/sampler and must retain the original failed small-world
comparison. No automatic extension,seed campaign or goal completion follows.
