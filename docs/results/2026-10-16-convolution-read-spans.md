# Read-span composition is correct but substantially slower

October16,2026. [Declaration](../experiments/2026-10-16-convolution-read-spans.md)
· [Compact evidence](2026-10-16-convolution-read-spans.json).

The shared address/mask/read-span implementation passes the complete numerical
family, but increases complete convolution GPU time **18.809→28.187ms,49.86%**.
Every changed contract regresses. It executes **18.01% more warp instructions**
despite10.86% less L2 traffic. **Reject this implementation as a whole:** no
integration, favorable selector, retiming or full-update trial.

The qualified isolated runtime remains58aff452: **RGB46.199ms versus inclusive
JAX27.223ms,1.70×**. CDP43.939ms is a different workload. No production or Atari
change, backend rebuild or Naga feature is part of this study.

## What changed

One shared F32/F16/BF16 generator composes four valid contiguous operand reads.
Aligned spans use vector loads; unaligned spans use scalar loads under the
common predicate. Tails and discontinuities retain scalar behavior. The dW
gradient operand retains one cursor instead of four and derives crossings.
Aligned bindings, divisible logical extents and allocation capacity are checked;
otherwise the original source is retained. Padding is never read.

All20 contracts /23 occurrences retain tile geometry, reduction order, splits,
compensation, barriers, rounding and output handling. Fourteen packed
occurrences remain unchanged. The observer replaces only the heavy dispatch,
keeping its original buffers, transforms and reductions. Actual replacement
source/bindings/segments are audited separately from original Session exports.

## Complete hardware-counter result

| Main-kernel counter | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Warp instructions |10.296B|12.150B|+18.01%|
| ALU-pipe instructions |6.819B|7.647B|+12.14%|
| Control-flow-pipe instructions |25.593M|350.212M|+1268.37%|
| FMA-pipe instructions |1.141B|1.590B|+39.27%|
| LSU-pipe instructions |774.339M|892.568M|+15.27%|
| Tensor-pipe instructions |180.224M|180.224M|unchanged|
| L2 bytes |15.964GB|14.230GB|−10.86%|
| DRAM bytes |1.629GB|1.636GB|+0.43%|

Counts are weighted by all23 actual occurrences. Every control instruction
count matches the retained native capture; packed candidates also match.
Tensor and launch counts stay exact. All heavy/helper/empty ranges reevaluate
offline with the retained41-metric SDK configuration. These are not latency
shares, floating-point operation counts or source-line attribution. This
configuration does not measure candidate local-memory traffic.

The new common-span and alignment paths are associated with much heavier
control execution, not the hoped-for reduction. Mapping that increase to exact
driver instructions or subgroup divergence remains unproven. No implicit
local-ID/subgroup mapping is assumed, and no assembly-query retry follows.

## Ordinary timing rejects the candidate

| All23 complete operation occurrences | Native | Identical-source control | Candidate |
| --- | ---: | ---: | ---: |
| GPU time |19.025ms|18.809ms|28.187ms|
| Queued wall, excluding recording |19.073ms|18.876ms|28.240ms|
| Command recording |.0147ms|.1022ms|.0989ms|

The observer/native calibration passes the declared5% envelope:−1.13% GPU and
−1.04% queued wall. Candidate time worsens49.86%/49.61%, failing both5% benefit
gates. Original transforms, scratch and reductions are included. Counter replay
timings are not used. Custom dispatch recording is separately reported, not
hidden in the GPU comparison or claimed faster than native caching.

All nine changed contracts regress42.02–102.72%. Typed forward totals rise
4.514→7.540ms; typed derivatives9.225→15.575ms. The fourteen unchanged packed
occurrences total5.070→5.072ms. These standalone timings are not a complete-
update prediction or comparable directly with JAX's full update.

All7,680 ordinary samples remain. Weighted first/last32 means are native
19.036/18.889ms, control18.755/18.854ms and candidate28.199/28.189ms. One native
contract's first/last32 differs33.14%; the raw series is retained without
trimming, explaining it away or retiming. Aggregate calibration still passes.

## Correctness and scope

CPU checks validate304 modules from152 original/candidate variants:123 change
and29 remain exact. All196,946,944 actual typed address positions,27 small
odd/strided/boundary cases and binding/fallback negatives pass. Matrix arithmetic,
compensation, barriers and outputs remain unchanged. F32/F16 variants are CPU
qualification, not new GPU numerical evidence.

All40 numerical jobs pass **2,690,432 independent F64 comparisons** and
**258,287,296 exact paired output values**, including zero interiors, changed
inputs and restoration. Two active images exercise the full512-image grids and
original dW splits; ordinary/counter fixtures populate every image. Earlier
cancellation-fixture failures are not replaced or relaxed by these fixtures.

Retained CPU failures: the first build rejects unused inherited generator fields;
asserting their preservation fixes it. The first address audit incorrectly
requires each small image stride to be aligned; explicit per-image checks fix
the audit without changing the shader. The first observer compile passes usize
binding indices where u32 is required; three explicit indices fix it. An ad-hoc
historical receipt-membership assertion fails0/137: five relevant libraries
match other historical seals, all137 are sealed now, and missing old seals are
not invented. Original sources/logs remain. No GPU retry is used for these fixes.

All140 GPU guards pass, with133 recorded allocation warnings and no recovery
or separate NVML polling.1,360 counter passes/4,756 offline metric values,
10,960 standalone operation calls and1,033,149,184 bit-exact paired values are
retained across all three cohorts.1,331 source seals pass. Minimum sampled
Vulkan budget headroom is15,827,992,576bytes, not physical free or peak memory.
Last GPU-guard event is**18:23:15UTC**; bounded aggregate review finishes
**18:23:50UTC**. Zero learner updates/actions; no study worker remains.

## Decision and remaining question

This result rejects the proposed source composition, not the measured
convolution feeding deficit. Fewer logical reads and lower L2 traffic are
insufficient: the compiled workload adds substantial branch, address/control
and other execution work. The individual native instructions responsible are
not identified by aggregate pipe counters.

Next inspect the emitted common-predicate/alignment/fallback control on CPU
against these complete counters and the retained library feeding contract.
Distinguish a demonstrated source/IR change from a claim about driver lowering
or subgroup divergence. No revised width/tile/layout sweep, new GPU allocation,
compiler feature or automatic learning promotion follows. Keep58aff452/1.70×;
host/graph execution, recurrent boundaries and the full parity goal remain open.

Artifacts: `runs/convolution-read-spans-20261016.ZN5Ij9VI`.
