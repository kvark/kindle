# Lower dynamics-group rate passes execution and early-cost checks

The fresh bounded Breakout check completes8,192 actions/32,733 frames/489
updates in **47.412638s**. A1,024-action/4,096-frame frozen greedy restore takes
.856243s withzero updates andno natural completed episodes. Both original
stages, full training replay, audit and terminal collector pass. Gameplay
benefit remains unmeasured; no short-run score is interpreted as competence.

The conservative200k projection, including10% margin, is **1,273.288629s
(21m13.289s)**, below the30-minute limit. This is an early estimate, not a
measured full-run speedup. It permits considering a separately declared fresh
200k comparison, but no full learner is selected here.

## Configuration and native scope

Only dynamics learning rate4e-4→1e-4 changes from the retained encoder4e-5
control. Native447fe50 applies it to recurrent dynamics, posterior, future
prediction and exploration prefixes together. It does not isolate the recurrent
core or prove any optimizer mechanism. Encoder/base rates remain4e-5, behavior
is unset and follows the base rate, and warmup stays1000 updates.

Keep centeredCDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
sampled collection/posteriors,actor-unimix.01,action-effects1,replay100k,
expert100/unit weights/world gradients/all-state corrective labels,20% episode
guidance/cap1920,full18/sticky.25/repeat4,native pooled RGB andone GPU resize.
Learner1009/evaluation4000001009 are reused development seeds. This check starts
fresh; no earlier weights, replay, beliefs, environments or RNG are reused.

The production runner exposes the existing native config through fresh-CDP-only
`--dynamics-learning-rate`, with F32 rate/multiplier checks and restore rejection.
Its12 added lines exactly match the isolated retained runner additions. No other
private runner lines change. All471 production vector tests pass in1.82s.
CPU preflight6.533s passes two actual CLI paths,15 config cases,357 schema tensors,
227 retained-control inputs and489/6227/12477 cadence without native or emulator
construction. The [fixed protocol](../experiments/2026-11-01-cdp-breakout-dynamics1e4.md)
preserves the full limits and rationale.

## Validation and closure

All250 declared inputs remain unchanged. All357 initial tensors and47
pre-update semantic events match the retained control. The actual native config
differs only in dynamics rate;16,334,353 parameters andfirst update376/residual.5
match. All489 reports are finite andsupervised. Actor/world imitation losses
agree within8e-7;117,043 labeled replay rows equal their target-weight sum.
CNN,reward,actor andvalue weights change. All357 frozen tensors remain invariant.

The full guided replay verifies every input pixel, action and corrective label,
including3,828 actual overrides (46.7285%). Both host guards pass. Guided online
returns are not frozen competence; the frozen check completes no natural episode.
No GPU stage, replay, original audit or collector repeats; no native build occurs.

Original service`kindle-breakout-dynamics1e4-check-20261101` launches21:07:40.819
UTC November1/sourcee333fa0,PID847195/invocation03896c77109f470ea5f1097dc8b92477.
It succeeds21:08:45.000 after64.184s wall time. Original terminal collection
completes21:08:56.004. The main service andcollector both succeed on their first
attempt; there is no cap change, retry, extension or automatic full allocation.

A separate post-closure journal assertion incorrectly required a message from
the silent parentPID itself. That failed supplemental assertion is retained.
The subsequent read-only review verifies exactly one start/end with the original
invocation andall logged child processes in that invocation; the original launch
andlive checks retain the parentPID. Original completion bytes are unchanged.
No original collector, audit or GPU stage is rerun. Supplementary evidence is
under`native-check/supplemental-journal*`; it does not replace the passing
original experiment receipts.

Root`/x/Temp/kindle-breakout-dynamics1e4-20261101/native-check`, trial
`breakout-12m-64-b16-r16-dynamics1e4-expert100-check` is complete; do not rerun.
No Kindle service orGPU allocation remains active atclosure. The five-game goal
remains unfinished. Check live state before future work.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `9df05e237d53a3520804812418521a39d7ed07ce05f9e262f3275a386b5b1c9b` |
| Original review | `561d6295ce894c34c8c2e4dbca4baf2e19a83cde9b3154598953122d35f1530e` |
| Guided replay | `d183aaa1dfdd72eb33cc3295505bf74d478fde98a2207d46885358dbb539d7cf` |
| Original completion | `13a866441ccc836dbd9de52e0c2a2927a2a499daae50843533944b4657bf41e9` |
| Private runner | `fe5cf3470e7fdaca69deecec0337909e66dd448d1d9ad30e57972378cab1dd07` |
| Unchanged native | `35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2` |
