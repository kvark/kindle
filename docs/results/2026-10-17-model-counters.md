# Real-model counters confirm convolution's instruction overhead

October 17, 2026. [Evidence](2026-10-17-model-counters.json) ·
[Original CPU contract](2026-10-16-model-counter-contract.md) ·
[Backend plan](../backend_training_performance_plan.md).

The user-supplied Nsight Perf SDK works on the RTX5080/580.178.04. A complete
**current native world-model capture** now confirms the earlier standalone
convolution counts: all92 CTA/warp/tensor/ALU anchors across23 products match
exactly. Synthetic operands were not the explanation for the instruction gap.

Convolutions account for **85.83% of world-program warp instructions** and
execute **4.255× JAX's convolution instructions**, including7.287× ALU work.
This is not a time fraction or a4.255× speedup opportunity. The isolated complete
RGB12M/B32/T16 result remains **43.197/27.223ms,1.59×** inclusive JAX;
native`e45b9c43` stays qualified. No new ordinary timing, production promotion,
learning result or Atari restart.

## Actual execution, not a standalone proxy

| Counter | Complete native world | Its23 convolutions | Actual JAX23 convolutions |
| --- | ---: | ---: | ---: |
| Warp instructions |11.995B|10.296B|2.420B|
| ALU-pipe instructions |7.500B|6.819B|.936B|
| Tensor-pipe instructions |186.941M|180.224M|227.524M|
| LSU-pipe instructions |975.226M|774.339M|236.699M|
| CTAs |1,694,477|283,600|108,540|

The whole/detailed CTA, warp-instruction and tensor-instruction sums match
exactly. Every convolution also matches its retained standalone anchors, so
the [complete-family comparison](2026-10-16-jax-convolution-counters.md) now
has actual native model-context support. All3,751 native dispatches have zero
measured local read/write traffic in both whole and detailed observations.
This rules out local-memory traffic as their measured bottleneck, not all
compiler-managed storage or register/resource constraints.

The remaining native world work still executes1.700B warp instructions across
3,728 dispatches. Fewer instructions do not imply proportionally less latency;
dependencies and memory waits remain separate questions. This capture excludes
optimizer, imagination, behavior and target programs. It is not a full-update
instruction profile. JAX's different layouts, helpers, padding and gradient
compensation remain disclosed; this is not numerical or learning parity.

## Observer and qualification

Markers live inside the original cached compute pass. They preserve all
3,751 dispatches,1,112 groups and1,131 barriers, with no new compute boundaries,
submissions, bindings, kernels or shader/compiler changes. Marker removal
recovers the qualified recording source exactly. The original24-update
instrumentation-disabled control matches all reports, first targets, full
programs and initial/first/final292-tensor states.

The repaired partition contains23 individual convolution ranges and18 adjacent
non-convolution spans, plus one empty control. Non-convolution markers can
span original group barriers; those barriers remain in place. CPU partition
tests and four negative mutations pass. One cached secondary records the41
ranges once, then replays49 times; the primary records50 empty controls.
All182 push/pop events reconcile without nesting or dropped ranges.

For the new capture, all initial/first/frozen292-tensor pairs, every first target
and report,813 protected-buffer hashes, full plans/pipelines/groups/boundaries,
and1,310,729 complete output values match exactly. Ordinary restoration also
matches. All2,552 metrics across both captures reevaluate offline. Independent
publication verifies787 original partial-review and2,095 repaired-capture
source/artifact receipts. Test checkpoint files contain tensors only; no new
checkpoint-metadata equality or full-model independent oracle is claimed.

## Failures retained, not erased by the repair

The first allocation completes its whole capture, then aborts before detailed
dispatch because1,139 counter ranges require599,932,980 bytes, above the
observer's128MiB limit. This is **host profiler storage**, not a failed GPU
allocation or a wedge. CPU-only SDK sizing proves42 ranges need22,129,916 bytes.
The separately declared repair keeps the original limit and complete coverage;
it does not repeat the completed whole50-pass cohort.

That original observer also omits the seven resident target arrays from its
diagnostic export. Its target gate stays incomplete. The new observer invokes
the existing diagnostic materialization before export; every array matches
the retained control. This does not retroactively qualify the first allocation.

CPU failures remain with original sources/logs: the first observer's undeclared
`bytemuck` dependency, the partial review's missing-target assertion, the new
observer's single-element-loop Clippy error, preflight launched before a
successful build receipt, and the final review's nonexistent metadata-file
assumption. Reviewed byte hashing, explicit incomplete status, a lexical block,
correct preflight ordering and actual three-file checkpoint membership resolve
them. None triggers a GPU repeat or relaxed numerical gate. The controller's
CPU-review failure remains recorded; the separately run terminal audit passes.

Total: **three guards, two passing/one failed; zero allocation warnings**,
26 synthetic updates,100 counter replays,35 ordinary fixed-world replays,
zero game actions. Last GPU guard finishes**00:19:53 UTC**; independent review
00:21:05, publication00:22:19. No worker remains. Raw artifacts:

- `runs/model-counters-20261016.wXXkmWyB` — original control, whole capture,
  failed detailed allocation, CPU storage proof and incomplete target diagnosis.
- `runs/model-counter-coalesced-20261017.vFEgVtiE` — isolated observer`47dfc6f7`,
  complete detailed capture, restoration and independent audits.

## Decision

The convolution instruction deficit survives real model values and unchanged
execution boundaries. Do not restart generic precision, spilling, tile,
split-K or local-address sweeps. Previously rejected global tensor-load and
software-pipeline families remain rejected; an API name alone is not a fix.

Next use the retained41-range capture to map convolution dependency counters
and the18 non-convolution spans to their actual producers/consumers on CPU.
Distinguish feeding/control instruction work from execution dependencies before
selecting a genuinely different lowering. No new counter cohort is needed to
begin that analysis. Counter replay durations, instruction fractions and logical
bytes must not become additive latency savings. The remaining15.97ms complete
gap and a demonstrated path to parity are still open.
