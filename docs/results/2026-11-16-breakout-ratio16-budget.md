# K's recipe at 1.2M actions plays about 300 per game on both seeds; the declared learned-well marker is met

Result of [the ratio-16 budget declaration](../experiments/2026-11-16-breakout-ratio16-budget.md).
Root `/x/Temp/kindle-breakout-budget-20261116`. The clean centered-CDP
recipe with the life-loss boundary (`K`: 12M, 96x96, train ratio 16,
sticky .25, 18 actions, no guidance or labels) was trained for 1.2M
actions on seeds 22003 (`B22`, 74,977 updates in 148.8 minutes) and 1009
(`B10`, 74,977 updates in 150.7 minutes) and evaluated matched at 400k,
800k and 1.2M.

| Seed | 400k cells (pooled median) | 800k | 1.2M | 1.2M games at 100 or more / above 300 of 96 | Fast balls returned at 1.2M |
|---|---|---|---|---|---:|
| 22003 | 80.7 to 101.7 (73) | 193.5 to 309.5 (246) | **276.3 to 322.3 (334)** | 86 / 55 | .90 |
| 1009 | 52.8 to 60.9 (54) | 170.3 to 244.2 (154) | **264.3 to 353.3 (350)** | 88 / 62 | .91 |

**Both declared tests pass on both seeds**: the budget pays (4.6 and 6.5
times the 400k median) and the learned-well marker (pooled median at
least 150 on both seeds) is met at 334 and 350. The 1.6 times gap
between the seeds at 400k is gone at 1.2M. Three of the 192 games at
1.2M clear the first wall; 38 more stop between 400 and 431.

The run was interrupted: the first queue (`queue-12.json`, guards
`guards-12`, started 09:58 UTC November 16) completed 2 of 32 jobs
before a GPU kernel fault stopped it at 12:28 UTC. Three cells were then
evaluated on a software Vulkan device under the declaration's first
amendment. After the user reviewed the fault, the remaining 30 jobs
(`queue-12b.json`, guards `guards-12b`, unit
`kindle-breakout-budget-b-20261116`) ran from 23:30 UTC on the same boot
under the second amendment and all completed at 02:16 UTC November 17
(2 hours 46 minutes). Every cell reported as a declared reading is a
GPU cell.

## Seed 22003

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

**Matched frozen play** (GPU, 24 games per cell, `--life-loss-reset`;
mean, median in brackets, best after the semicolon):

| Point | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 50 or less / 100 or more / above 300 / above 432 of 96 |
|---|---:|---:|---:|---:|---:|---|
| 400k | 82.9 (74.5; 148) | 101.7 (72; 324) | 80.7 (76.5; 143) | 92.8 (74.5; 359) | 73 | 14 / 27 / 4 / 0 |
| 800k | 193.5 (148.5; 425) | 309.5 (343; 435) | 213.8 (151; 424) | 246.5 (258.5; 412) | 246 | 2 / 78 / 41 / 1 |
| 1.2M | **278.8** (277; 421) | **322.3** (362; 441) | **276.3** (312.5; 434) | **293.5** (331.5; 433) | **334** | 1 / 86 / 55 / 3 |

`B22`'s own 400k cells repeat `K`'s to the digit, as its weights do. The
pooled mean rises 89.5, 240.8, 292.7. Four games pass 432 and so clear
the first wall (435 at 800k; 433, 434 and 441 at 1.2M), the first on
record for any Kindle learner. Fifteen more of the 1.2M games end
between 400 and 431, just short of the wall, with a few bricks left.

**Replay by life** (`breakout_life_replay.py`, CPU), four cells pooled on
each row:

| | Returned, lives 1 to 5 | Paddle hits per life | Under 6 px | 6 to 9 | 9 to 13 | 13 or more | First life from 12 hits |
|---|---|---|---:|---:|---:|---:|---:|
| 400k (`K`) | .96 / .95 / .95 / .93 / .81 | 11.6 / 7.5 / 6.7 / 5.7 / 4.4 | .98 (1427) | .94 (1515) | .91 (322) | .76 (439) | .92 (270) |
| 800k | .97 / .94 / .93 / .89 / .83 | 17.1 / 10.7 / 8.9 / 6.3 / 4.7 | .99 (1369) | .96 (1633) | .93 (871) | .81 (1057) | .96 (729) |
| 1.2M | .98 / .97 / .95 / .95 / .86 | 19.8 / 13.6 / 9.6 / 8.0 / 6.1 | .98 (1348) | .97 (1784) | .97 (1181) | .90 (1431) | .97 (1006) |

More of the same training does buy fast-ball precision on this seed:
descents at 13 px per action or more are returned 76%, 81% and 90% at
the three points, while the agent meets three times as many of them per
game as at 400k. The first life no longer degrades after twelve hits,
and the fifth life is still the weakest (.86).

**Post hoc, where the games end** (the four 1.2M cells replayed, by the
game's score at the time; not a declared reading):

| Score so far | Ball descents | Returned | Points per descent | Lives lost per 100 points |
|---|---:|---:|---:|---:|
| 0 to 100 | 3,930 | .95 | 2.4 | 2.2 |
| 100 to 200 | 825 | .91 | 9.7 | 1.0 |
| 200 to 300 | 325 | .86 | 19.4 | .8 |
| 300 to 350 | 278 | .84 | 8.9 | 1.8 |
| 350 to 400 | 379 | .84 | 3.8 | 4.1 |
| 400 to 432 | 215 | .81 | 1.6 | 11.5 |

Games reaching 300 / 350 / 400 / 420 / 432: 55 / 41 / 18 / 12 / 3 of 96.
Once the ball is behind the wall a return is worth 10 to 19 points; with
few bricks left it is worth 1.6, while one return in five is still lost,
so the last 32 points cost about 3.7 lives on average. Eleven of the 18
games that reach 400 arrive with one or two lives left. Clearing the
wall reliably needs either the late-game return rate near the early-game
one or returns aimed at the remaining bricks.

## Seed 1009

**Online returns of `B10`**, mean per game over 50k-action bins:

| Actions | 350-400k | 400-450k | 600-650k | 800-850k | 1,000-1,050k | 1,100-1,150k | 1,150-1,200k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Return per game | 50.4 | 64.6 | 141.7 | 207.6 | 204.7 | 249.8 | 260.5 |
| Lives lost per 50k actions | 175 | 155 | 134 | 116 | 120 | 110 | 118 |
| Policy entropy (nats) | 1.34 | 1.43 | 2.14 | 2.25 | 2.24 | 2.22 | 2.22 |
| Return scale | 7.2 | 9.1 | 54.6 | 125 | 136 | 139 | 150 |

The same shape as seed 22003, a little lower online: a steady rise,
return scale settling near 150, entropy near 2.2 nats, no late fall.

**Matched frozen play** (GPU, same layout as above):

| Point | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 50 or less / 100 or more / above 300 / above 432 of 96 |
|---|---:|---:|---:|---:|---:|---|
| 400k | 57.6 (52; 189) | 60.9 (58; 95) | 54.6 (52.5; 93) | 52.8 (52; 89) | 54 | 40 / 3 / 0 / 0 |
| 800k | 182.5 (138; 401) | 202.5 (199.5; 372) | 170.3 (125.5; 415) | 244.2 (258.5; 419) | 154 | 4 / 70 / 29 / 0 |
| 1.2M | **264.3** (304; 421) | **353.3** (384; 428) | **321.1** (363.5; 424) | **272.3** (312; 426) | **350** | 2 / 88 / 62 / 0 |

`B10`'s 400k cells repeat `K10`'s to the digit. The pooled mean rises
56.5, 199.9, 302.8. No game passes 432 (best 428); 23 of the 1.2M games
end between 400 and 431.

**Replay by life**, four cells pooled on each row:

| | Returned, lives 1 to 5 | Paddle hits per life | Under 6 px | 6 to 9 | 9 to 13 | 13 or more | First life from 12 hits |
|---|---|---|---:|---:|---:|---:|---:|
| 400k (`K10`) | .96 / .95 / .94 / .93 / .79 | 8.9 / 6.9 / 6.1 / 5.0 / 3.8 | .97 (1469) | .93 (1450) | .77 (111) | .54 (156) | .91 (145) |
| 800k | .96 / .94 / .94 / .94 / .84 | 15.5 / 11.2 / 8.7 / 7.2 / 5.1 | .97 (1425) | .95 (1671) | .94 (853) | .85 (957) | .93 (594) |
| 1.2M | .98 / .97 / .95 / .95 / .86 | 20.7 / 14.6 / 9.4 / 7.2 / 6.2 | .98 (1327) | .97 (1745) | .97 (1265) | .91 (1485) | .98 (1089) |

The seed that returned 54% of fast balls at 400k returns 91% at 1.2M and
is indistinguishable from seed 22003 by life and by speed. Its score
bands (post hoc) repeat the other seed's: returns fall from .95 below
100 points to .78 between 400 and 432, where a return is worth 1.5
points and 14 lives are lost per 100 points, so the last 32 points cost
about 4.6 lives. Games reaching 300 / 350 / 400 / 420 / 432: 62 / 49 /
23 / 11 / 0 of 96; 16 of the 23 that reach 400 arrive with one or two
lives left.

## Reset probes

Linear ball and paddle readout of the model state on held-out teacher
play with the belief restarted at each lost life
(`breakout_agent_probe_life.py --life-loss-reset`), source pixels:

| Checkpoint | Ball x / y error | Ball within 8 px | Paddle error | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| `B22` 400k | 7.7 / 8.3 | .51 | 5.4 | .73 / .78 |
| `B22` 800k | 7.1 / 8.4 | .53 | 5.0 | .76 / .82 |
| `B22` 1.2M | 6.8 / 6.0 | .62 | 5.6 | .80 / .81 |
| `B10` 400k | 8.3 / 8.2 | .50 | 5.8 | .60 / .68 |
| `B10` 800k | 7.3 / 7.8 | .51 | 5.8 | .66 / .69 |
| `B10` 1.2M | 7.8 / 5.6 | .57 | 4.5 | .75 / .80 |

The ball readout improves slowly (within 8 px .50 to .57 and .62) while
play rises four to six times, so most of the gain is in the policy and value on top
of a state that already held the ball at 400k; the vertical position and
the velocity readouts sharpen most in the last third.

## Software-device cells

Before the fault was reviewed, the first amendment's cells ran on Mesa's
llvmpipe (CPU only; the faulted card was not opened), one at a time as
memory-capped units, 73 to 100 minutes each for 5,900 to 7,300
decisions. It is the same checkpoint and runner on different
floating-point code, so each cell is another draw of the same policy.
The GPU cells above supersede them in the declared reading; all four
now have a GPU twin:

| 1.2M cell | GPU mean (median; best) | llvmpipe mean (median; best) |
|---|---:|---:|
| 777001 sampled | 278.8 (277; 421) | 249.1 (292.5; 428) |
| 777001 greedy | 322.3 (362; 441) | 302.5 (350; 495) |
| 022003 sampled | 276.3 (312.5; 434) | 233.8 (189; 412) |
| 022003 greedy | 293.5 (331.5; 433) | 304.1 (362; 428) |
| Pooled, 96 games | 292.7 (334) | 272.4 (326) |

The pooled means differ by 20 with a standard error of 18, and the
replay rates agree (balls returned .955 against .953 with the first
cell from the GPU, fast balls .90 against .89), so the two devices are
not distinguishable at this sample size and llvmpipe is usable for
frozen evaluation when the card is not. Its 495 is the highest game on
record. A cell mean carries about 25 points of sampling error.

One protocol failure is retained: the first attempt at the 022003 greedy
cell ran beside another cell under a 6 GB cap and was killed by its own
cgroup at 2,949 decisions (llvmpipe's shared memory grows by about 1 MB
per decision, to roughly 10 GB for a cell). Its partial log is kept in
`cpu-oom-20261116/` and not used; the cell was rerun alone from the
start under a 13 GB cap.

## Against the declaration

- **Budget pays: passes on both seeds.** Seed 22003: pooled median 334
  against 73 at 400k (4.6 times; the test asks for 1.5), every cell 3.2
  to 3.4 times its 400k cell. Seed 1009: 350 against 54 (6.5 times),
  every cell 4.6 to 5.9 times its 400k cell.
- **Learned-well marker: met.** The pooled median at 1.2M is at least
  150 on both seeds (334 and 350); it is also met at 800k (246 and 154).
- Predictions: cell means 120 to 250 (all eight above, 264 to 353),
  medians 100 to 180 (all eight above, 277 to 384), fast balls returned
  .75 to .85 (above, .90 and .91), at least one game above 432 (three on
  seed 22003, none on seed 1009). The run did better than predicted on
  every count but the last, which one seed misses.
- The fallback clause (fast-ball return under .7 on both seeds, then
  change control or input resolution) does not apply.

## Limits

- Two development seeds on two reused evaluation cohorts, 24 games per
  cell; a cell mean carries about 25 points of sampling error. No
  held-out seed has been run at this budget.
- The first wall is cleared in 3 of 192 games at 1.2M, and the best of
  them ends nine points into the second wall (441). Both seeds lose
  about four lives over the last 32 points. The original 864-point /
  90% gate is far from met; what stands is a declared median marker.
- Each seed costs 150 minutes of training on one RX 7900 XT plus about
  30 seconds per evaluation cell.
- The curves were still rising at 1.2M on both seeds; nothing here says
  where they stop.

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

No GPU job was started for the next eleven hours, and no reset, reload
or reboot was done. The frozen cumulative-survival learner was not
touched; its unit is still loaded, but the reset reported VRAM lost, so
its device state is presumably gone.

The user reviewed the fault late on November 16 and chose to continue on
the same boot without a reboot. The existing guard cannot record that,
so it is wrapped, unchanged, by `gpu_host_guard_reviewed.py`, which
accepts exactly the nine journal records of this fault
(`gpu-fault-20261116/reviewed-kernel-faults.json`) and still stops on
any other. The first job after the review was the evaluation that had
faulted; it ran cleanly in 31 seconds, and all 30 jobs, including the
150-minute training run, completed with no further kernel GPU message.
