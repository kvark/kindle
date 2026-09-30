# Why the small native RGB pilot is slow

The [interrupted pilot](2026-09-30-small-replication-learning.md) spends97.8%
of measured stage time inside learner calls. A representative update takes
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
[Compact data](2026-09-30-small-rgb-profile.json) ·
[Raw profiles](../../runs/small-dreamer-replication-20260930.vRE7ag/world-profile-8lcscp1o/profile/profiles).
