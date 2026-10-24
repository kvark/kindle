# Host-generated randomness is a measurable learner bottleneck

October 14, 2026. [Declaration](../experiments/2026-10-14-device-rng.md) ·
[Numerical/timing data](2026-10-14-device-rng.json) ·
[Performance plan](../backend_training_performance_plan.md).

**Moving identical random draws onto the GPU saves 6.80% of complete RGB
updates and 8.44% of larger-batch CDP updates. Every measured metric and saved
tensor remains identical.** RGB still takes **2.34×** the retained complete
JAX caller boundary; this identifies another contributor, not the whole answer.

## One controlled change

The old learner generates and uploads 1,083,136 F32 uniforms per RGB update,
4,332,544 for CDP. The candidate writes these inputs directly with a Blade
ChaCha12 shader. Explicit rand_chacha0.9 state preserves the pinned rand0.9.4
StdRng seed, stream, word position and draw order. Replay/live-policy streams
remain separate. No random algorithm, model, batch, precision, target, optimizer
or backend graph changes.

This uses the previously qualified wider-fusion backend, **not** the unpromoted
[multi-output scalar trial](2026-10-14-pointwise-regions.md). Complete backend
source SHA matches its immutable control snapshot exactly. New RNG submissions
precede posterior/imagination on the same queue; consumers need no added host
wait. Physical extents, non-overlap, dispatch bounds and resource completion
are checked. There are46 new small RNG dispatches in two submissions per update;
the existing compiled session plans are unchanged.

## Complete-update timing

One ordinary pair per workload,16 warmup+256 measured updates, no profiler or
concurrent compilation/analysis. Order: RGB control/candidate, CDP candidate/
control. Same checkpoint/configuration per pair; these are synthetic updates,
not gameplay or time-to-score.

| Workload | CPU uniforms | GPU uniforms | Less wall time |
| --- | ---: | ---: | ---: |
| RGB12M, B8/T16/H15, BF16 | 32.2465ms | 30.0528ms | 6.80% |
| CDP12M, B32/T16/H15, mixed | 104.0477ms | 95.2688ms | 8.44% |

RGB imagination/targets falls9.9864→7.9496ms and posterior4.5034→4.4159ms.
CDP falls39.8699→31.5743ms and11.9542→11.5699ms respectively.
World training is essentially unchanged:15.1764→15.1617ms RGB and
46.2333→46.1603ms CDP. The complete savings are2.1937/8.7789ms, almost exactly
proportional to the fourfold draw count. This intervention measures the combined
change in CPU generation/allocation/upload and replacement GPU work, not an
independently timed cost for each component.

The earlier JAX control remains12.4205ms core /12.8365ms including caller
seed/metric handling. It was **not rerun** here. Its wrapper returns the previous
metric record; first empty/final pending limitations remain. CDP's larger batch
and exploration are not compared directly with that RGB/JAX number. Kernel
algorithms, dependency execution and remaining host targets/EMA still matter;
do not add isolated savings or claim SM utilization.

## Exactness and operational evidence

- Four CPU tests pass, including73,746 mixed StdRng/ChaCha12 draw comparisons,
  WGSL validation and stream isolation; release Clippy passes after one retained
  style failure.
- **2,107,118 exact GPU values:**1,053,117 raw words and1,054,001 F32 values.
  Twenty length/offset/stream cases include zero, irregular and million-word
  outputs, low-counter carry and full-period wrap. Canaries remain intact;
  split/consecutive fills preserve the sequence. Three subsequent-session
  consumers pass without a producer host wait.
- Both24-update smokes and all272 reports per full-update arm pass. First
  categoricals/targets and all292 RGB /346 CDP tensors at initial, first and
  final snapshots are bit-for-bit exact. Existing precision/dispatch plans
  match exactly. This scopes equivalence to these fixtures, not all workloads.
- Seven guards pass; three allocation warnings retained, zero GPU job failures,
  recovery or separate NVML polling. Sampled Vulkan headroom passes≥2GiB.
  Total1,136 full synthetic updates, zero game actions. No worker remains.

Artifacts: `runs/device-rng-20261014.EZ1UNxiK`; native57a1a1a3 versus
controlaca8359b. The initial backend diff omitted newly created untracked files;
the incomplete snapshot remains. Reverse-apply verification and a complete
intent-to-add snapshot reproduce the original qualified source SHA exactly,
without a binary/source change or GPU retry. Logs, immutable source, controls
and all warning records remain.

**Keep GPU RNG in the isolated candidate.** Production remains on its previous
qualified ~53ms RGB runtime; no installed pins/defaults, learning allocations
or neighboring user work change. Continue device-side slow-critic EMA/targets,
then row-level normalization/backward fusion and safe layout/view propagation.
The broader root-cause/performance goal remains active.
