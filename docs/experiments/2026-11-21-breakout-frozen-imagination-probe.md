# Does the frozen-encoder learner still imagine a forgiving paddle?

Prepared 15:11 UTC November 21; **launched by hand at 15:11 UTC**.
Follows [the 800k-freeze result](../results/2026-11-21-breakout-encoder-freeze-800k.md):
the frozen-encoder learner is stable and levels off at about 345 per
training game with the same lost ball as every learner before it (a
paddle that arrives late and moving). Evaluation only; no training.

## Why this probe

On the unfrozen recipe [imagination was forgiving at the paddle's edge](../results/2026-11-18-breakout-imagined-contact.md):
of the action sequences the emulator loses within eight actions, the
world model rated .203 as safe (`L22` at 1.6M; `B22` .244, held-out
`H50` .245), and a quarter of the sequences it rated safest lost the
ball. [Open-loop supervision](../results/2026-11-18-breakout-open-loop-contact.md)
halved that (.102) and play did not follow, on a recipe whose encoder
was drifting and whose training returns were 30 points behind the
control's. With the encoder still, the model's latent loss and KL are
about half the unfrozen recipe's. Whether its imagination at contact is
now calibrated has not been measured, and it decides what the next
training run should be.

## Run

The pinned probe `breakout_imagined_contact.py` (root
`/x/Temp/kindle-breakout-imagined-20261118`, `sources.sha256` verified)
with its declared settings (24,000 actions of own play, seven fixed
sequences, eight steps, two emulator clones) on three checkpoints:

- `E22` at 3.6M (seed 22003, frozen from 800k),
- `F22` at 2.8M (seed 22003, frozen from 1.2M),
- `F50` at 2.4M (held-out seed 5011, frozen from 1.2M).

Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-28.json` (4
jobs: a 300-action smoke on the first checkpoint, then the three
probes), guards `guards-28`, unit
`kindle-breakout-imagined-frozen-20261121`, reviewed runner with this
boot's review file. About 9 minutes per probe, 5 MB each. Readers:
`imagined_contact_read.py`, `imagined_missed.py`,
`imagined_reliability.py` from the probe's root, unchanged.

## Measures

On the declared rows (ball one to three actions from the paddle row):

1. **Missed-loss share:** of the pairs the emulator loses in both
   clones, the share imagination rates below .1 (`L22` .203, `O22`
   .102).
2. **Loss rate of the pairs rated safest** (imagined loss below .02;
   `L22` .25, `O22` .06).

Reported beside them: false-alarm share, AUC, Brier score, the narrow
paddle separately.

## Readings

- **Missed-loss share .15 or more on all three:** the forgiving
  imagination survives the freeze. The next training run is the frozen
  recipe with open-loop supervision, read on training returns against
  `E22` and `F22`; declared separately.
- **.12 or less on all three:** the still target already calibrated
  imagination and the plateau is not this. Open-loop supervision is
  dropped as the next candidate; the actor and critic side is next.
- **Between, or mixed across checkpoints:** reported as mixed; the
  open-loop run is then a weaker candidate and is weighed against a
  long held-out run of the frozen recipe.

Predictions: missed-loss share .14 to .22 on the seed-22003 checkpoints
and .16 to .25 on the held-out one; the first reading with two chances
in three.

## Limits known in advance

- One probe per checkpoint, 24,000 actions (about 45 to 60 lost lives);
  single checkpoints.
- The comparison checkpoints (`L22` at 1.6M, `B22` and `H50` at 1.2M)
  are earlier in training than the frozen ones, so a difference mixes
  the freeze with 1.2M to 2.0M more actions.
- All three earlier GPU faults hit frozen-inference jobs, one of them
  this probe. A fault stops the queue and needs the user's review.
- The design is the assistant's; the user has not been asked.
