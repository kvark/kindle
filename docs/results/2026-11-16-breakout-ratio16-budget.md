# Interim: the 1.2M-action budget run reaches 300 per game online on seed 22003; a GPU fault stopped the queue after its first evaluation

Partial result of [the ratio-16 budget declaration](../experiments/2026-11-16-breakout-ratio16-budget.md).
Root `/x/Temp/kindle-breakout-budget-20261116`, `queue-12.json`, guards
`guards-12`, unit `kindle-breakout-budget-20261116`, started 09:58 UTC
November 16. **2 of 32 jobs completed**: the seed-22003 training run
(`B22`, 74,977 updates in 148.8 minutes) and one matched evaluation at
1.2M. The second evaluation faulted the GPU as it started and the host
guard stopped the queue at 12:28 UTC. The declared reading needs all
four cells at 400k and 1.2M on both seeds and is **not yet evaluable**;
seed 1009 has not run.

## What is in hand

**Online returns of `B22`** (training games, belief restarted at each lost
life), mean per game over 50k-action bins:

| Actions | 350-400k | 400-450k | 600-650k | 800-850k | 1,000-1,050k | 1,100-1,150k | 1,150-1,200k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Return per game | 69.8 | 86.2 | 153.8 | 198.5 | 253.2 | 302.4 | 293.0 |
| Lives lost per 50k actions | 152 | 148 | 130 | 127 | 111 | 109 | 108 |
| Policy entropy (nats) | 1.49 | 1.64 | 2.04 | 2.13 | 2.15 | 2.22 | 2.15 |
| Return scale | 11.8 | 17.8 | 65.3 | 149 | 151 | 161 | 156 |

The rise continues through the whole run. The return scale grows to
about 150 by 800k and stays there, and entropy settles near 2.2 nats;
the named risk (a peak followed by a fall as the scale rises) did not
occur on this seed.

**The first 400k actions repeat `K` exactly.** `B22`'s online curve
matches `K`'s to the printed digit in every bin, and its 400k
`world.safetensors` and `behavior.safetensors` have the same SHA-256 as
`K`'s. Training is reproducible for a seed, so the 1.6 to 2.5 times
spread between seeds is the seed, not the run, and `K`'s matched 400k
cells (82.9 / 101.7 / 80.7 / 92.8, median 73) are `B22`'s.

**Matched frozen play at 1.2M, one cell of four** (cohort 4000777001,
sampled, 24 games, `--life-loss-reset`): **mean 278.8, median 277, best
421**; 21 of 24 games at 100 or more, 10 above 300, none above 432. The
same cell at 400k: mean 82.9, median 74.5, best 148.

**Replay of that cell by life** (`breakout_life_replay.py`, CPU), with `K`
at 400k pooled over its four cells:

| | Returned, lives 1 to 5 | Paddle hits per life | Under 6 px | 6 to 9 | 9 to 13 | 13 or more | First life from 12 hits |
|---|---|---|---:|---:|---:|---:|---:|
| `B22` 1.2M, one cell | .99 / .97 / .93 / .95 / .84 | 20.3 / 13.7 / 8.0 / 8.1 / 5.3 | .98 (332) | .97 (431) | .97 (299) | .88 (331) | .97 (240) |
| `K` 400k | .96 / .95 / .95 / .93 / .81 | 11.6 / 7.5 / 6.7 / 5.7 / 4.4 | .98 (1427) | .94 (1515) | .91 (322) | .76 (439) | .92 (270) |

More of the same training does buy fast-ball precision on this seed:
descents at 13 px per action or more are returned 88% against 76%, and
the first life no longer degrades after twelve hits.

## Against the declaration, so far

- Budget pays on seed 22003 (median at least 1.5 times its 400k median,
  every cell above its 400k cell): the one cell in hand has 3.7 times the
  median and 3.4 times the mean of its 400k cell. Three cells open.
- Learned-well marker (pooled median at least 150 on both seeds): the
  one cell's median is 277. Seed 1009 open.
- Predictions: cell means 120 to 250 (the cell is above the range at
  278.8), fast balls returned .75 to .85 (above at .88), at least one
  game above 432 (not in this cell; best 421).

## GPU fault

At 12:27:50 UTC the greedy evaluation of the same checkpoint (process
3235535, same runner and command shape as every earlier evaluation) hung
the graphics ring within seconds of starting, before it wrote any log
line:

    amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=211881260, emitted seq=211881262
    [drm:gfx_v11_0_bad_op_irq [amdgpu]] *ERROR* Illegal opcode in command stream
    amdgpu 0000:03:00.0: MES failed to respond to msg=RESET
    amdgpu 0000:03:00.0: GPU mode1 reset
    amdgpu 0000:03:00.0: VRAM is lost due to GPU reset!
    amdgpu 0000:03:00.0: [drm] device wedged, but recovered through reset

The ring reset failed and the kernel fell back to a full mode-1 reset,
which succeeded. The host guard recorded "GPU kernel fault; this boot
requires explicit recovery review" and the queue runner exited. RX 7900
XT, RADV Mesa 26.0.8, kernel 7.0.0-38. The previous job (the sampled
evaluation) had ended cleanly seconds earlier, after 2.5 hours of
training on the device. Kernel log and guard verdict are kept in
`gpu-fault-20261116/` under the root; the device coredump is readable by
root only and was not copied.

No GPU job has been started since, and no reset, reload or reboot was
done. The frozen cumulative-survival learner was not touched; its unit
is still loaded, but the reset reported VRAM lost, so its device state
is presumably gone.

## To finish

`queue-12b.json` in the root holds the 30 remaining jobs (`B22`'s eleven
evaluations, the seed-1009 run and its twelve evaluations, six probes).
It is not launched; the guard will keep refusing this boot until the
fault is reviewed, and that review is the user's.
