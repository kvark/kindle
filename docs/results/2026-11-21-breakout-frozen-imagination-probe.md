# The frozen-encoder learner still imagines a forgiving paddle

Result of [the probe declaration](../experiments/2026-11-21-breakout-frozen-imagination-probe.md).
Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-28.json` (4
jobs), unit `kindle-breakout-imagined-frozen-20261121`, 15:11 to 15:39
UTC November 21, all jobs complete, no GPU fault. Evaluation only: the
pinned `breakout_imagined_contact.py` with its declared settings (24,000
actions of own play, seven fixed action sequences, eight steps, two
emulator clones) on three frozen-encoder checkpoints. Readings in
`analysis/imagined-*`.

**Result.** The first declared reading applies: the missed-loss share is
.15 or more on all three checkpoints. Freezing the encoder halved the
world model's latent loss and KL and left its imagination at contact
where it was.

## Measures

Declared rows (ball one to three actions from the paddle row).

| Checkpoint | Pairs | Missed-loss share | Loss rate of pairs rated below .02 | False alarms | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| `E22` 3.6M (seed 22003, frozen from 800k) | 13,363 | .205 | .23 | .029 | .881 | .139 |
| `F22` 2.8M (seed 22003, frozen from 1.2M) | 12,810 | .210 | .24 | .018 | .878 | .146 |
| `F50` 2.4M (seed 5011, frozen from 1.2M) | 13,524 | .192 | .21 | .035 | .886 | .130 |
| `L22` 1.6M (unfrozen) | 13,776 | .203 | .25 | .031 | .885 | .150 |
| `O22` 1.6M (unfrozen, open-loop H8) | | .102 | .06 | .044 | .919 | .115 |

Missed-loss share: of the pairs the emulator loses in both clones, the
share imagination rates below .1. Earlier unfrozen learners: `B22` at
1.2M .244, held-out `H50` at 1.2M .245.

On the narrow paddle the shares are .156, .186 and .118 (`L22` .138),
and the pairs rated safest lose .28, .32 and .25 of the time (`L22`
.32). Imagination still orders losing against saving sequences from the
same state correctly in .897, .893 and .907 of pairs (`L22` .909). The
probes' own play: 51, 51 and 55 lives lost in 24,000 actions.

## Reading

Three learners with a still encoder, two seeds, up to 2.0M more actions
than the unfrozen comparison: a fifth of the sequences that lose the
ball are still imagined safe, and a sequence the model rates safest
still loses the ball in the emulator one time in four to five. The
policy learns only in imagination, so this is a candidate cause of the
late, moving paddle that survives every change so far. Open-loop
supervision is the one change on record that moved this measure (.203
to .102); it was tested on the unfrozen recipe, where it trailed the
control in training returns (232.0 against 263.8 per game over 800k to
1.6M) and did not improve play.

Predictions: .14 to .22 on seed 22003 (.205, .210) and .16 to .25 on
the held-out seed (.192) held; the first reading (two in three) held.

## Limits

- One probe per checkpoint; 51 to 55 lost lives each.
- The unfrozen comparison checkpoints are earlier in training.
- Fixed action sequences from the agent's own states, not the sequences
  the actor's imagination samples.

## Next

[The frozen recipe with open-loop supervision](../experiments/2026-11-21-breakout-open-loop-freeze.md),
read on training returns against `E22` and `F22`.
