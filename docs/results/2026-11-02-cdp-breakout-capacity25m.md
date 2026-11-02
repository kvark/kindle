# Breakout 25M passes bounded learning and restore, at higher cost

A separately selected [fresh200k comparison](../experiments/2026-11-02-cdp-breakout-capacity25m-200k.md)
starts10:47:21.939 UTC November2/source0e69b83,main933689/invocation
`d3457431b99a41aebd2ec054df3f4d7d`,with a45-minute training cap and420 pins.
Startup matches357 bounded initial tensors/47 pre-update transitions and records
163 finite reports through2,984 actions. Original progress,recovery,postprocessing
and collection observers are active. Full frozen results are not available.
The bounded result below is complete; do not repeat it or overlap the new learner.

The existing 25M preset completes the declared **8,192 actions / 32,722 frames /
489 updates in84.864s**. Its **1,024-action /4,096-frame frozen restore takes
1.343s**, with zero updates and no completed natural episodes. Execution and
the early cost screen pass. Gameplay benefit remains unmeasured; no full
learner is selected at this check's closure.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-capacity25m.md)
changes only the model preset from the retained replay-value1 recipe. It expands
the CNN, recurrent state, categorical classes and behavior heads together.
Trainable parameters grow **16,334,353→36,608,401**. RGB64, Auto precision,
N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100,
sampled collection, guidance and the full18 Atari protocol remain fixed.
Native447fe50, runnercf4ea238 and the teacher are unchanged; no build occurs.

| Bounded learning cost | Retained 12M | Candidate 25M |
| --- | ---: | ---: |
| Actions / updates | 8,192 /489 | 8,192 /489 |
| Training seconds | 47.690 | 84.864 |
| Conservative 200k projection, including 10% margin | 21m20.729s | 37m59.071s |

The candidate costs **77.95% more** on this short check. Its projection clears
the separately declared45-minute ceiling for a possible larger-capacity life.
The projection is not a full-run measurement or a speed improvement. A future
comparison must report absolute frozen scores and initial-subtracted return per
training minute against the retained12M scores70.750 sampled /143.667 greedy.
Useful quality per GPU-hour, and the original90%-two-wall gate, remain the aim.

The CPU schema derivation exactly reproduces the actual12M checkpoint before
deriving25M. All **357 actual candidate tensors** match, including342 changed
shapes; initial/final values are finite and all357 frozen tensors are invariant.
The different architectures have different initial values and trajectories:
no equality with12M is claimed. A separately selected full25M life must match
this bounded initial checkpoint and keep its own initial evaluations.

First update376, residual credit.5, exact configuration and all489 finite reports
pass. All489 reports include supervision, covering117,037 labeled replay rows
with equal weight sums; maximum raw world/behavior imitation-loss difference is
9e-7. CNN/reward/actor/value parameters change. Full guided replay verifies
every recorded input pixel, teacher label, action and outcome;3684 actions are
overridden,4508 have passive labels and3979 differ from their teacher label.
These guided training observations do not measure unaided competence.

Both GPU host guards, all **380 pins** and both original services pass. Main
PID933088/invocation`0bf2c9a4f0564d3695115d95b027b2aa` starts10:37:27 UTC and
finishes **10:39:11.653 UTC November2**, source`a8144ec`; the collector's review
completes10:39:18.624. Final read-only confirmation passes10:40:52.368, with both
services terminal and no active Kindle allocation. There are no preparation,
GPU or audit failures, repeated stages, cap changes or reused learner experience.
No automatic promotion, capacity sweep, retry, extension or full learner follows
from this check alone. The five-game goal remains unfinished.

Root `/x/Temp/kindle-breakout-capacity25m-20261102`; trial
`native-check/breakout-25m-64-b16-r16-replayvalue1-expert100-check`. The check and
collector are complete; do not repeat them.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `16622f5edd1c5de2180a707ca03b7791b56f16d4bbb8981306ffa5dc924136de` |
| Review | `0bcca24c778376b7a1904d546beecaa154acfe997f0681d97191ea3c2941ee69` |
| Guided replay | `946b62433928bd19b221e4d6adc017cf877d4b0b98800c245c0a6d9c81f3cf09` |
| Completion | `8ee60866c7400d9805428c0caf959d1c82e28957454a62fea369eba4d02ee24d` |
| Terminal summary | `df437e69f40f17d75881788f1320878688d170826f969c6fc86230dcbd4f06bd` |
