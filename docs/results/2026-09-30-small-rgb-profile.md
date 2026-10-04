# Why the small native RGB pilot is slow

The [interrupted pilot](2026-09-30-small-replication-learning.md) spends97.8%
of run wall time inside learner calls. A representative update takes
73.93ms, including54.69ms world training and12.24ms imagination. Acting and
emulation are not the bottleneck; eight environments and time-independent world
operations are already batched.

A guarded synthetic profile restores a **copy** of its saved model, runs one
synthetic B8/T16/H15 update, then profiles resident session inputs. It does not
resume replay or interact with a game. Before/after profile captures retain
identical parameters and optimizer moments. Guard/seal and native test pass.
This is not another gameplay attempt; two of ten remain used.

The important finding is **low-parallelism convolution weight gradients**:

| Shape: batch, input/output channels, spatial side, kernel | Workgroups | Profiled interval |
| --- | ---: | ---: |
| 128, 3→8,64,5×5 | 5 | 18.55ms |
| 128, 8→3,64,5×5 | 13 | 13.17ms |
| 128, 8→12,32,5×5 | 13 | 5.00ms |
| 128, 12→8,32,5×5 | 19 | 4.46ms |

The first two reduce over128×64×64 =524,288 positions using very few workgroups.
Meganeura **already implements split-K convolution weight gradients**, with
independent F64 checks and bounded measurement. Ordinary construction does not
select them; measured-search candidates exist. Test that existing mechanism
before adding kernels or changing learning. The September30 upstream recheck
still finds`7c29497` (caller-owned submission API), not a newer fix for this path.

Timing limits matter. Ordinary world forward/backward wall median is54.68ms;
the associated whole-submission GPU timestamp median is26.87ms. This is not SM
utilization. Per-dispatch profiling changes submission/pass boundaries and costs
4.51× wall time (246.67ms, four windows); the table is a bottleneck locator,
**not additive ordinary execution time**. The capture excludes optimizer/AGC
passes. The sizeable ordinary wall/GPU gap also deserves attention after these
long reductions. Re-test whole-update time and numerical parity for any change.

No backend optimization or learning improvement is established by this profile.
Driver580.178.04, RTX5080, F32, Meganeura22c31b94/Blade7cca6377; no NVML polling.

The first split-K diagnostic exposed a CPU pipeline-cache lookup panic: the
measurement helper prepared a 64-wide kernel but looked up the original 16-wide
kernel. Meganeura `101443fc` preserves the declared geometry. Its GPU regression
passes for 16/32/64-wide tiles, including independent F64 checks, optimizer
updates, unchanged live state/plan, aliasing and scratch limits. The failed
invocation is retained; its broad kernel snapshot hit the capture size cap,
while incremental checks and a fresh bounded journal read show no kernel fault.

The corrected two-shape diagnostic finishes without a crash and leaves all
saved tensors unchanged. However, **both unsplit controls fail the existing
strict F64 tolerance**, before candidate timing. This is not a qualified
speedup. The subsequent independent check confirms ordinary long-reduction
rounding: **every unsplit output exactly matches a separate/FMA sequential
float32 CPU sum**. Its relative L2 error against F64 is 1.25–1.47e−5.

Two declared partition counts, 64 and 1,024, are checked against independent
F64 whole/partial sums at ordinary and 1e−12 gradient scales. The 64-way split
fails the strict gate; the **1,024-way split passes every final/partial check**,
with relative L2 error 4.32–4.60e−7. No tolerance was relaxed. Isolated whole
sequence medians (including the final reduction) fall 18.68→0.544ms and
18.76→1.197ms for the two shapes. Eight paired samples alternate execution order.
These are not whole-learner or gameplay speedups. The existing kernels suffice;
the candidate uses 512 reduction positions per partition, not a new shader.

The subsequent **whole-update comparison is 2.37× faster: 72.62→30.58ms**
median, with unchanged Size1M/B8/T16/H15/F32 learning settings. It includes
posterior inference, imagination/targets, world/behavior updates, AGC/optimizer
and synchronization; resident synthetic batches exclude replay sampling and
gameplay. Two warmup updates precede eight paired samples with alternating
order, two CPU-equivalents, no compilation and timestamps disabled in both arms.
The earlier timestamp-asymmetric measurement is retained but superseded.

First-step reports are exact. All 68 raw-gradient tensors and 292 saved
parameter/moment tensors pass the declared max-absolute≤1e−6 **or** relative-L2
≤2e−4 check. Maximum raw-gradient absolute/L2 differences are .02261/5.27e−5;
the large absolute gradient difference is not an absolute-tolerance pass.
Saved-state maxima are 2.06e−5/1.63e−5 (a decoder first moment). Both ten-update
paths finish at optimizer step22,515 with finite checkpoints. Later trajectories
are not claimed bitwise identical. Guards pass; ≥2GiB sampled Vulkan estimated
headroom is retained.

Meganeura `75d08173` exposes opt-in, budgeted lowering using existing kernels.
Kindle enables it for low-parallelism training convolutions: eight gradients
split in this fixture, at most512 reduction positions per partition and64MiB
logical partial storage. This changes summation order, not pixels, model
capacity, losses or replay ratio. The gameplay pilot starts **fresh**;
the interrupted checkpoint is not resumed.

That [fresh gameplay pilot](2026-09-30-small-replication-learning.md) now passes:
200,000 actions / 49,939 updates in 26m10s, 127.42 actions/s. At the common 90,112-action
prefix, the old/new runs each complete 22,467 updates in 1,756.32/705.40s: **2.49×
actual throughput** at the same recipe and seed. Floating summation changes
later trajectories; this is not a claim of identical learning. The new run's
first20/last50 online means are 83.0/328.8, with all 236 episodes/tails retained,
finite checkpoints, zero debt and passing guard/seal/counter audits.

[Compact data](2026-09-30-small-rgb-profile.json) ·
[Raw profiles](../../runs/small-dreamer-replication-20260930.vRE7ag/world-profile-8lcscp1o/profile/profiles).
