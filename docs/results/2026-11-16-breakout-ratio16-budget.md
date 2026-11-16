# Interim: K's recipe at 1.2M actions plays about 280 per game on seed 22003; a GPU fault leaves seed 1009 open

Partial result of [the ratio-16 budget declaration](../experiments/2026-11-16-breakout-ratio16-budget.md).
Root `/x/Temp/kindle-breakout-budget-20261116`, `queue-12.json`, guards
`guards-12`, unit `kindle-breakout-budget-20261116`, started 09:58 UTC
November 16. **2 of 32 jobs completed**: the seed-22003 training run
(`B22`, 74,977 updates in 148.8 minutes) and one matched evaluation at
1.2M. The second evaluation faulted the GPU as it started and the host
guard stopped the queue at 12:28 UTC. The three open 1.2M cells of seed
22003 were then evaluated on a software Vulkan device under the
[declaration's amendment](../experiments/2026-11-16-breakout-ratio16-budget.md),
so **the seed-22003 tests are read below and pass**. The both-seed tests
need the seed-1009 run, which needs the GPU and **has not run**.

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

**Matched frozen play at 1.2M** (24 games per cell, `--life-loss-reset`;
mean, median in brackets, best after the semicolon). The first cell ran
on the GPU, the other three on the software device (see below):

| Point | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 100 or more / above 300 / above 432 of 96 |
|---|---:|---:|---:|---:|---:|---|
| 400k (`K`, GPU) | 82.9 (74.5; 148) | 101.7 (72; 324) | 80.7 (76.5; 143) | 92.8 (74.5; 359) | 73 | 27 / 4 / 0 |
| 1.2M (`B22`) | **278.8** (277; 421) | **302.5** (350; 495) | **233.8** (189; 412) | **304.1** (362; 428) | **324.5** | 82 / 54 / 1 |

Two of the 96 games at 1.2M score 50 or less. One game (495, greedy,
cohort 777001) passes 432 and so clears the first wall, the first on
record for any Kindle learner. Eighteen more end between 400 and 431:
a fifth of the games stop just short of the wall's 432 points, with a
few bricks left.

**Replay of those cells by life** (`breakout_life_replay.py`, CPU), four
cells pooled on each row:

| | Returned, lives 1 to 5 | Paddle hits per life | Under 6 px | 6 to 9 | 9 to 13 | 13 or more | First life from 12 hits |
|---|---|---|---:|---:|---:|---:|---:|
| `B22` 1.2M | .98 / .97 / .95 / .96 / .86 | 19.0 / 13.8 / 8.3 / 8.7 / 6.2 | .98 (1322) | .97 (1796) | .97 (1146) | .89 (1380) | .97 (884) |
| `K` 400k | .96 / .95 / .95 / .93 / .81 | 11.6 / 7.5 / 6.7 / 5.7 / 4.4 | .98 (1427) | .94 (1515) | .91 (322) | .76 (439) | .92 (270) |

More of the same training does buy fast-ball precision on this seed:
descents at 13 px per action or more are returned 89% against 76% (.88,
.89, .88 and .91 in the four cells), and the first life no longer
degrades after twelve hits. The agent now meets three times as many fast
descents per game as at 400k, and the fifth life is still the weakest
(.86).

## Software-device cells

The amendment's cells ran on Mesa's llvmpipe (CPU only; the faulted card
was not opened), one at a time as memory-capped units, 73 to 100 minutes
each for 5,900 to 7,300 decisions. It is the same checkpoint and runner
on different floating-point code, so each cell is another draw of the
same policy rather than the game the GPU would have played.

The declared check, the GPU's cell repeated on llvmpipe (777001
sampled):

| Device | Mean | Median | Best | 100 or more / above 300 of 24 | Balls returned | Fast balls returned |
|---|---:|---:|---:|---|---:|---:|
| GPU | 278.8 | 277 | 421 | 21 / 10 | .954 | .88 (331) |
| llvmpipe | 249.1 | 292.5 | 428 | 19 / 11 | .946 | .86 (315) |

The means differ by 29.7 with a standard error of 35.3 (24 games each,
game standard deviations 108 and 135), the medians go the other way, and
the replay rates agree, so the two devices are not distinguishable at
this sample size. A cell mean carries about 25 points of sampling error,
and the largest gap between two of the four cells above (70) is twice
the standard error of such a difference.
The table of declared cells uses the GPU log for this cell; with the
llvmpipe repeat instead, the pooled median is 326 and the pooled mean
272.4 against 279.8.

One protocol failure is retained: the first attempt at the 022003 greedy
cell ran beside another cell under a 6 GB cap and was killed by its own
cgroup at 2,949 decisions (llvmpipe's shared memory grows by about 1 MB
per decision, to roughly 10 GB for a cell). Its partial log is kept in
`cpu-oom-20261116/` and not used; the cell was rerun alone from the
start under a 13 GB cap.

## Against the declaration

- **Budget pays on seed 22003: passes.** The pooled median at 1.2M is
  324.5 against 73 at 400k (4.4 times; the test asks for 1.5), and every
  cell mean is above its 400k cell (2.9 to 3.4 times).
- **Learned-well marker** (pooled median at least 150 on both seeds):
  seed 22003 reaches 324.5. **Seed 1009 is open**, so the marker is not
  yet read. At 400k that seed played 52.8 to 60.9 (median 54) and
  returned 54% of fast balls, so it starts from further back.
- Predictions: cell means 120 to 250 (one cell inside at 233.8, three
  above), medians 100 to 180 (all four above, 189 to 362), fast balls
  returned .75 to .85 (above, .88 to .91), at least one game above 432
  (one, 495). The run did better than predicted on every count.
- The intermediate points (400k as `B22`'s own cells, 800k) and the
  reset probes were not run; they are GPU jobs in `queue-12b.json`.

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
fault is reviewed, and that review is the user's. Its three `B22` 1.2M
cells supersede the software-device cells when they run. The seed-1009
training run (about 150 minutes on the GPU) cannot be moved to the
software device.
