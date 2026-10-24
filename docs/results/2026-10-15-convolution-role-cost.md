# Most of the convolution intervention is RGB reconstruction

October15,2026. [Declaration](../experiments/2026-10-15-convolution-role-cost.md)
· [Data](2026-10-15-convolution-role-cost.json)
· [Whole-family predecessor](2026-10-15-convolution-value-bypass.md).

**The encoder accounts for6.90ms and the RGB decoder13.04ms of the19.94ms
valid-value intervention.** These are symmetric conditional differences in
one fixed world graph, not pure kernel latencies. About65% belongs to a decoder
that CDP does not have. This materially narrows which convolution improvements
could help the main learning path; it does not measure actual CDP throughput.

| Products replaced by exact recorded-value copies | Fixed-world GPU | Queued wall |
| --- | ---: | ---: |
| None |34.216ms|34.225ms|
|11 encoder products |27.285ms|27.294ms|
|12 decoder products |21.142ms|21.155ms|
| Both,23 products |14.280ms|14.285ms|

The encoder difference is6.931ms with the decoder intact,6.862ms with its
products substituted. The decoder difference is13.073/13.005ms respectively.
Averaging each pair allocates6.896/13.039ms; the interaction is.069ms. This
near-additivity is observed here, not an assumption about future optimizations.
The remaining14.280ms still includes matrix/scalar work and device execution;
other learner stages are outside this diagnostic. Neither role alone explains
the complete native/JAX gap.

Fresh control/both means differ from the preceding experiment by+0.071%/−0.013%,
inside the declared3% comparison gate. All512 measured samples remain: eight
counterbalanced blocks0,1,2,3,3,2,1,0,16 warmups+64 measured replays each.
Control blocks34.136/34.295ms are retained. These are repeated fixed inputs,
not independent learner replicates or a learning-quality confidence interval.

## What is controlled

Same RGB12M/B32/T16 fixture, one initializer update, then no optimizer, clipping
or gradient accumulation. All23 outputs stay resident in855,506,944device-local
bytes in every condition. The original4,023 descriptors/1,258 runtime groups,
convolution/non-convolution dependency boundaries, allocations and original
pipelines remain. The qualified integer-word copy shader/bindings are unchanged;
only a checked set of dispatch indices selects substitution. Unknown/duplicate
indices reject. No new arithmetic, precision policy or kernel implementation.

Every condition matches all23 captured outputs at17 live boundaries and every
finite full-graph output before timing. Full outputs remain exact after each
block and final restoration. Initial/first292 tensors, first targets/reports
match the retained control; before/after292 tensors are unchanged. The CPU audit
independently checks all three tensor pairs and streams complete plan, constants,
pipeline/group and configuration semantics. All1,157 source seals pass.

Replacement copies change computation and cache traffic. The differences are
not exclusive operation sums, a zero-cost lower bound, or independently
removable training savings. We do not subtract the historical copies-only
measurement or compare these intervals directly with JAX kernel sums. No
profiler, concurrent build or CPU analysis runs during GPU timing.

## Consequence for the investigation

The prior library evidence remains fundamental: pass-specific reuse and
multistage feeding accomplish the same useful contractions with much less
supporting work. This experiment does not isolate asynchronous copies, shared
memory capacity, padding or instruction counts as the sole cause. The rejected
fixed workgroup family and earlier numerical failures remain rejected.

Do not optimize the decoder and call it a CDP improvement. Conversely, do not
assume that removing reconstruction removes all convolution cost. The encoder
still matters, but CDP changes gradients and adds its action-effects ensemble.
Source review confirms that its exploration already reuses state projections
and centers before the final projection; the old18-way first-projection cost
is not current. Its retained Mixed policy has distinct inference/training
precision and cannot automatically reuse the BF16 staged posterior. That policy
was chosen before BF16 derivatives and staging existed. The separately
[existing-policy comparison](2026-10-15-cdp-precision-recheck.md) subsequently
finds CDP78.206→50.122ms,35.91%, without a new kernel or learning allocation.
That cost result still needs independent numerical qualification; it does not
turn this RGB intervention into a CDP performance measurement.

The next algorithm decision must address supporting work or execution structure,
not another small tile/iterator variation. Current RGB results do not justify
claiming that a compiler feature alone will reach JAX, nor that Vulkan parity is
impossible. The root-cause goal remains open. Latest ordinary complete RGB
remains54.224ms versus JAX26.632ms,2.04x; no production promotion or Atari restart.

## Completion

One guard passes, zero new allocation warnings, recovery, separate NVML queries
or game actions. Native`f282ad74`;647 full-world replays including validation/
warmup and one initializer update. GPU service completes in30.37s; independent
CPU review in67.01s,77.2MiB peak under1CPU/2GiB/no swap. Minimum sampled Vulkan
estimated headroom12,165,971,968bytes, not physical/peak-free memory. Four CPU
selector/shader tests and both strict Clippy checks pass. No new failed process
or relaxed gate, and no worker remains. Historical failures stay in their
original immutable artifacts.

Artifacts: `runs/convolution-role-cost-20261015.8xUKRn0w`.
