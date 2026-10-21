# A separate cooperative-load question survives the predicate fix

October 21, 2026. [Compact evidence](2026-10-21-shared-read-residual.json)
· [Predicate hardware capture](2026-10-21-predicate-counters.md).

**The current typed convolutions have an exact additional shared-read counter
pattern that the predicate fix does not remove.** Across every comparable typed
region, the historical-to-current increase in shared-read wavefronts minus
reported read-bank conflicts equals **twice the tensor-instruction count**.
All corresponding packed regions have zero increase in that derived quantity.
This narrows the next diagnostic to cooperative operand-load representation;
it is not yet a source-instruction attribution or a speedup.

Separately, [Meganeura PR276](https://github.com/kvark/meganeura/pull/276),
commit `e545a9f`, passes all six CI1050 jobs, including Metal and lavapipe.
It remains unmerged. Production and Atari are unchanged; the retained complete
RGB/JAX gap is still 1.642×, not a new performance result.

## What the saved captures establish

| Nine typed occurrences | Historical native | Current control | Predicate candidate |
| --- | ---: | ---: | ---: |
| LSU instructions | 652.545M | 1,015.119M | 1,017.806M |
| Shared-read wavefronts | 425.749M | 635.268M | 635.327M |
| Read-bank conflicts | .482M | .286M | .345M |
| Wavefronts minus conflicts | 425.267M | 634.982M | 634.982M |
| Tensor instructions | 104.858M | 104.858M | 104.858M |
| Active threads per issued warp instruction | 31.99945 | 31.31420 | 31.99952 |

The exact derived increase is **209,715,200**, twice the unchanged 104,857,600
tensor instructions. It holds independently in all seven current typed ranges
covering nine occurrences; two ranges combine neighboring dX/dW dispatches.
The ten packed ranges cover fourteen occurrences and retain their derived
shared-read counts. No per-kernel result is invented inside a merged range.

The predicate rewrite restores near-full active-thread coverage and removes
most control instructions, but leaves this shared-read relation untouched.
Thus the current residual is not just the short-circuit branch problem again.
The reported increase also is not explained by more measured read-bank
conflicts. The subtraction is an explicitly defined counter diagnostic, **not
bytes, a pure kernel cost, or independently removable latency**.

Independent readers reconcile raw range-membership files, all 3,751 dispatches,
CTA/tensor counts and every selected additive counter. Four malformed membership
cases reject. The existing compiler-input audit supplies all 23 exact historical/
current-control reconstructed SPIR-V pairs; candidate memory/matrix operations
were separately qualified with the predicate rewrite. These are not archived
driver binaries or native SASS. Historical driver580/runtime/profiler conditions
versus current615 are not a driver-only causal experiment. No fresh JAX counter
ratio is formed. This relation was found post hoc, not confirmed on held-out data.

## A focused CPU-only diagnostic now validates

Keep each four-BF16 shared packet's logical bits, addresses and cooperative-load
stride operands, but expose its workgroup storage as two unsigned 32-bit words.
Each existing staging store gets a bitcast; the cooperative matrix result stays
BF16. This changes neither global gathers nor packet width/orientation, tiling,
arithmetic, branches, output handling or synchronization. It is distinct from
the rejected conditional read-span and packet-axis implementations.

All nine actual typed occurrences pass an isolated reversible SPIR-V transform:

- All 18 original/transformed modules validate offline.
- Removing only the declared view/type changes reconstructs every original
  instruction exactly. Four stores and four cooperative loads per loop retain
  their original indexing/stride operands; no ordinary stage read is admitted.
- All 65,536 BF16 payloads in each of four packet positions round-trip exactly:
  **262,144 bit-pattern checks**, including zeros, infinities and NaN payloads.
- Packet payload/alignment and logical array extent remain equal. The original
  arrays have **implicit workgroup layout**, however: physical placement,
  compiler-managed storage and generated machine instructions are unproven.

Pinned Naga `a489554` still lacks the requested component-type field in
`CooperativeLoad`. This diagnostic bypasses no production type check: it is an
offline binary transformation only. No compiler edit, runtime integration,
GPU numerical check, counter capture or timing has occurred for the view.

## Failures, accounting and next step

The first counter reader names the prior compiler receipt `review.json` instead
of `result.json`; the filename-only repair preserves its failure. The first view
reader incorrectly expects an explicit ArrayStride decoration. Its revised
contract preserves implicit layout and makes no physical-layout identity claim.
The next module fails validation for redeclaring an existing unsigned vector
type; reusing that type fixes it. All original sources/logs/modules remain.
No GPU cohort is repeated or numerical gate relaxed.

Counter review finishes 11:07:13 UTC; view validation 11:11:57 UTC. This turn uses
**zero GPU calls, builds, learner updates or game actions**. All local services
are terminal. Raw evidence: `runs/predicate-residual-20261021.bHKQt4Xo`.

Next prepare a separately declared Vulkan numerical/counter diagnostic of this
one raw-word shared view, covering all nine typed occurrences. First reconcile
actual submitted bindings/modules and unchanged recording; then compare exact
outputs and independent references before interpreting counters. A useful
instruction result would still need complete-operation/update cost qualification.
Do not change Naga merely because the modules validate, revive rejected feeding
families, retime old candidates or restart Atari automatically. Backend parity
and the full five-game goal remain open.
