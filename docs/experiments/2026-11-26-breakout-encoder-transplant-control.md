# The transplant's control: seed 5011 on its own frozen encoder from the first update

Declared 17:53 UTC November 26, before launch. The assistant's design;
the user was not asked. This is the control named in
[the transplant declaration](2026-11-26-breakout-encoder-transplant.md).

## Why

`X50`, the recipe on seed 5011 with `D60`'s frozen encoder from the
first update, finished at 17:48 UTC (21 of 21 jobs, no fault; its
report follows this declaration). Its level test passes in the top
band: 425.1 per training game over 2.8M to 3.6M (292 games, standard
error 6.9) against `D50`'s 364.2 and seed 6007's 411.2. It also learns
far sooner: 308.3 per game over 0.4M to 0.8M against `D50`'s 70.6 and
`D60`'s 109.3. Play pools to 426 / 427 / 423 with 50 / 51 / 38 of 144
games past the wall, no game cut or at zero (`D50` 388.5 / 404.5 / 398
with 11 / 17 / 22).

Three things differ between `X50` and `D50`: whose encoder it is, that
it is mature and frozen from the first update, and the random streams.
This run changes only the first of them relative to `X50`.

## Run

`S50`: `X50`'s command with the donor replaced by `D50`'s 3.6M
checkpoint (seed 5011's own encoder as frozen at 1.2M) and nothing else
changed: seed 5011, `--encoder-freeze-update 1`, `--replay-compact`,
the same restore path and so the same random streams. Then the same 20
evaluation, dump and probe jobs: 21 jobs. Same root
(`/x/Temp/kindle-breakout-transplant-20261126`, same pinned sources,
`sources-45.sha256`), `make_transplant_queue.py`, `queue-45.json`,
guards `guards-45`, unit `kindle-breakout-transplant5011s-20261126`,
run directory
`S50-cdp-life-ratio16-96-encoderD50-freeze1-lrdrop150k-3600k-s5011`.
The review file is the one of 20:44 UTC November 24 (same boot, no
fault since). About 7 hours of training.

## Checks

21 of 21 jobs; the kernel log; the header records the transplant; the
encoder's tensors equal `D50`'s frozen ones in every checkpoint; every
other initial tensor equals `X50`'s; no training game cut.

## Tests

1. **Identity:** `X50`'s training return per game over 2.8M to 3.6M
   exceeds this run's by 25 or more (this run is at 400.1 or less).
2. **Maturity:** this run's training return per game over 0.4M to 1.2M
   is 220 or more (`D50` 110.9, `X50` 323.9; the midpoint is 217).

Observations as for `X50`: the return by stretch against `X50` and
`D50`; latent loss and KL from the start; the 3.6M dump's readouts;
play by checkpoint; return rates; serve delay.

## Readings

- **Test 1 passes:** whose encoder it is carries the level. Seeds
  differ in the encoder they grow, and the next question is how to get
  a good one on every seed without labels (choose among a few short
  donor runs by a label-free measure, or find what in the first 1.2M
  actions makes one).
- **Test 1 fails with this run at 412 or more** (within 13 of `X50`):
  the identity does not matter; a mature encoder that is still from the
  first update does. The recipe then becomes two-stage on every seed:
  grow an encoder for 1.2M actions, restart on it.
- **Between 400 and 412:** both contribute; not separated by one run.
- **Test 2** is read on its own: passing says the early speed comes
  from the encoder being mature and still, whoever grew it.

## Predictions

Test 1: one in two. Test 2: three in four.

## Limits

- One run each; the difference of two levels has a standard error of
  about 10, so 25 is two and a half of them.
- Seed 5011's run-to-run spread under this path is not measured.
- One donor per seed.
- The same three cohorts as every run since November 17.
