# Vectorized LeVJEPA collection and GPU throughput

Started 2026-09-06 after an explicit request to prioritize vectorized environments
and batched inference, even at the cost of a day of setup. This is one shared
policy/learner, not swarm learning or asynchronous actor/learner services.

## Baseline and decision

The original Atari runner used one environment and one-frame live inference.
Only replay training was batched (B16×T64). Its last inspected 30-minute GPU
trace averaged 58.64% activity, 8,217 MiB and 117.90 W on the RTX 5080. Activity
is a sampled busy fraction, not achieved FLOPs or occupancy; power below the
360 W limit alone does not prove inefficiency. Wall-time stages do establish a
large learner bottleneck: about 0.54 s/update and 5.5 executed actions/s.

The serial mastery queue was stopped with SIGINT at 21:01 UTC. Its last logged
learner record is action 64,390 / update 15,827; its verified 60k checkpoint and
logs are preserved. Unlogged in-flight work is not an audited endpoint. The
single-stream LeVJEPA mastery gate remains unachieved; details are in the
[original experiment](2026-09-06-levjepa-pong.md).

## Implemented collection contract

- One frozen LeVJEPA weight set. Dense layers process all stream rows together;
  attention currently dispatches independent stream blocks in the same session.
  Each stream has its own 16-arrival KV history and position. Missing arrivals
  do not advance it. A real reset affects only that stream; chunk boundaries
  still do not reset the RSSM.
- Batched live RSSM posterior and policy sessions. Each stream keeps its own
  belief, pending action and policy/posterior RNG. Stream seeds are model seed
  plus stream index (wrapping u64); environment seeds are separately recorded.
- One learner, optimizer, return normalizer and shared-capacity replay. Replay
  retains per-stream order, evicts globally by arrival age, drains fresh starts
  before uniformly sampling eligible starts across streams, and refreshes the
  correct stream's cached posterior context.
- Every executed action advances the global counter and, once warm, adds
  R/(B×T) credit. Initial/episode-reset frames do neither. Warmup counts eligible
  sequence starts across streams. The first eligible vector round gives one
  update; later rounds drain all due updates, without silently discarding debt.
- Python steps the independent ALE instances synchronously, submits their
  observations together, consumes terminal frames before resetting affected
  streams, and logs all actions/rewards/boundaries. CPU environment time is
  below 1% of measured wall time; subprocess environments are not needed yet.
- Checkpoints additionally record `collection_streams`. Model state can be
  restored into a fresh vector collector or a single-stream frozen evaluator;
  replay, exact RNGs, environments, live beliefs and visual caches are not saved.

Maintained entry points: `VectorDreamerAgent`, `kindle.VectorAgent`,
`python/examples/atari_vector.py`, `python/examples/profile_atari_vector.py`,
and the CPU-only `kindle._vector_audit` ledger checker. The original `Agent` and
Atari runner remain the serial/DINO control. The new vector frontend is LeVJEPA.

## Correctness before throughput

Initial native checks on the selected RTX 5080:

- Batched LeVJEPA matches serial dense and pooled features **bit-for-bit** over
  36 vector rounds with reversed input order, unequal reset times, missing
  arrivals and chunk boundaries. The existing pinned PyTorch Pong reference
  also passes (worst dense absolute error 0.00026131, relative L2 <1e-4).
- Three batched live beliefs and sampled policies match independent serial
  streams under different terminal/timeout/reset schedules.
- The N=1 vector learner matches serial scheduled losses and final world
  parameters exactly, including reset handling and checkpoint restore. The
  initial full-size N=1 benchmark's learner metrics also match the interrupted
  serial seed-0 prefix (world/behavior metrics, not wall-time fields).
- Replay tests cover uneven streams, global eviction and context refresh;
  scheduler tests count actions, not vector ticks or reset records. Python
  audits reject missing updates, wrong resets/rewards, and null/nonfinite losses.

## First matrix: same learner, batched collection

`runs/vector-20260906-first/`: N=1,2,4 executed serially, seed 0, 3,072 aggregate
actions each. Published Atari wrapper, full 12M recipe, B16×T64, full BPTT64,
row microbatch16, R256, learning rate4e-5, warmup1000, AGC0.3, reconstruction0,
future prediction0.25, extrinsic-only own sampled actions. All start fresh.
Timing uses the final 1,024 actions, after warmup: exactly 256 updates in each
window. Construction and prefill are excluded from that window and reported
separately. Brief CPU builds/checks also ran during parts of the exploratory
matrix; small differences need a quiet repeat, not an overprecise speed claim.

| Environments | Aggregate actions/s | Mean update, s | GPU activity | Mean W | Window peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.395 | 0.54395 | 58.38% | 117.43 | 8,249 |
| 2 | 5.853 | 0.54632 | 53.45% | 120.87 | 9,145 |
| 4 | 6.029 | 0.53499 | 55.81% | 123.30 | 10,708 |

Cold construction is 91.02 / 92.38 / 91.87 s. Full runs have 497 / 481 / 450
updates because each additional stream needs its own sequence context during
warmup. All three full accounting audits pass. The N=4 prefix completes no
games; this matrix is not a gameplay result or an amended mastery endpoint.

Native SHA-256:
`e99944abd4953bcf3ab6e41cd95c9e6bc95a12fbc49eef2e6bd9488f1b34c4d9`.
Runner SHA-256:
`340ed16d8a2d03653cf1afe47d5ef0d1abd7c0c0238834431df01a610bd61a1b`.
Meganeura remains `35a410ce1262c396e137db5bfab1d58e35cee50a`, Blade
`b208f3b1f97196c2971436b5726e61e71b149c37`. The native executable is archived
beside the matrix. Each run records the pinned encoder identity and wrapper hash.

Vectorization alone yields about 12% more aggregate actions/s. At N=2 the
1,024-action window spends 141.13 s in learning, 33.15 s in observation handling,
0.47 s stepping environments and 0.13 s selecting actions. The learner remains
about 81% of wall time. Aggregate speed must not be called per-stream speed:
at N=4, 6.03 total actions/s is roughly 1.51 actions/s per environment.

## Same-math learner cleanup

`runs/vector-20260906-cpu/` repeats N=2, B16 and the same 3,072-action protocol
after caching softmax exponentials, reusing categorical/decoder scratch and
submitting independent imagination heads before one combined readback/wait.
All **481 world and behavior reports match the original N=2 run bit-for-bit**;
the complete interaction/credit audit passes. Timing fields are excluded from
the numerical comparison.

The final 1,024 actions / 256 updates take 165.16 s: **6.200 actions/s**, mean
update 0.51189 s, GPU activity 58.58%, mean power 125.34 W and peak 9,145 MiB.
That is about 6% faster than the initial N=2 path, not a solution to the remaining
GPU bubbles. Cold construction takes 91.38 s. Native SHA-256 is
`8184d4db5a47c2ddd7ad9956a17ddf2463311834af70b8023d4a46d53fc7f761`;
the runner and dependency pins are unchanged. This executable is also archived.

The numerical/model GPU tests pass. In one combined-process run, the last two
readback tests failed while creating their GPU contexts (`NoSupportedDeviceFound`);
both passed in a fresh process. The cause is not diagnosed, so this is not an
unqualified clean full-suite result. Long training uses one shared context.

### Bounded CPU row parallelism

`runs/vector-20260906-parallel/` adds one local pool of at most eight CPU workers
for large independent categorical and scalar-decoder rows. Uniform draws are
generated in the original serial RNG order; each row retains its arithmetic
order. Small/live batches stay serial. This is not actor/learner separation.

All 481 world/behavior reports and every transition, episode and reset agree
exactly with the preceding N=2 run. The accounting audit passes. The final
1,024-action / 256-update window takes **157.54 s, 6.500 actions/s**, with a
0.47898 s mean update, 61.18% GPU activity, 129.87 W and 9,145 MiB peak VRAM.
Cold construction takes 91.60 s. This is about 20% faster than the initial N=1
path; at N=2 it is 3.25 actions/s per environment, still below real-time learning.

Native SHA-256:
`9d6290a3c4b39f826771e1020976d2be81a5b9e6e3a1a80577ee5a285efac38b`.
Runner SHA-256 (now records CPU worker count):
`ee636253c5c90eb8c7902752660c14ee7b077e025e27a17c982b71e97be9c373`.
The executable and runner are archived; dependency pins and scientific recipe
are unchanged.

## Larger replay batch: separate optimizer-cadence experiment

`runs/vector-20260906-b32/` keeps N=2, T64/full BPTT, R256 and the same learning
rate recipe, but doubles B and its row microbatch to 32. It runs 4,096 aggregate
actions so the measured final 1,024 actions are all after the larger warmup.
The window contains exactly **128 updates**, not 256: equal total replay samples
do not imply equal optimizer trajectories, warmup in interactions or learning
quality. This is not a numerics-preserving systems comparison.

The window takes 126.10 s: **8.121 actions/s**, 0.69326 s/update, GPU activity
53.82%, 135.17 W and 10,416 MiB peak. Cold construction is 93.93 s. The full
accounting audit passes. Native/runner hashes match the CPU-parallel B16 run.
The lower busy fraction despite higher useful throughput is another reason not
to optimize the utilization percentage alone. Keep B16 as the learning control
until a predeclared quality comparison validates a larger batch.

## Next measured changes

Current CPU checks: 75 Kindle and five gym tests pass, with 15 hardware tests
explicitly skipped by the ordinary command; 135 Python tests pass. Rust/Python
Clippy and formatting checks pass. The vector auditor also verifies reported
aggregate/per-stream game clocks and rejects invalid throughput accounting.
Later API-only cleanup exposes learner metadata directly instead of returning
an unused single-stream diagnostic core; the measured binaries remain archived.

1. Measure larger learner row batches if memory permits. Preserve total replay
   ratio and full BPTT; declare changed optimizer cadence as an ablation.
2. A bounded-operand compensated tensor-core frontend is an experimental
   candidate, not enabled in the running matrix. It must pass the existing
   strict causal/reference gates and demonstrate a latency win. Never re-enable
   reduced-exponent-range kernels for sensitive learner gradients.
3. Only after throughput and accounting checks, predeclare fresh vectorized
   Pong training seeds, aggregate interaction budgets and final frozen gates.
   Preserve failed/interrupted runs and do not select winning checkpoints.
