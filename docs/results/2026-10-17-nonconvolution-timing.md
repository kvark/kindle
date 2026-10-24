# Recurrence: similar launch counts, unequal execution costs

October 17, 2026. [Evidence](2026-10-17-nonconvolution-timing.json) ·
[Qualified timing](2026-10-17-boundary-timing.md) ·
[Backend plan](../backend_training_performance_plan.md).

**The remaining imagined-recurrence gap is not a large matrix-count or launch-
count mismatch.** Current native recurrence plus per-state features has 180
matrix products within 631 dispatches. JAX has the same 180 products within
630 kernels, plus 45 device copies. Both matrix execution and surrounding
processing remain substantial native costs. This does not establish identical
program interfaces or isolate one causal slowdown.

Keep isolated `e45b9c43`: **43.197 ms versus inclusive JAX 27.223 ms, 1.59×**.
This study uses existing captures only: no GPU call, native build, learner
update, game action, production change or Atari restart.

## Current native imagination

Versioned output dependencies distinguish recurrence from the three rollout
assembly outputs and already-batched heads. The current copy-composed graph
has 681 dispatches: 20 input-independent parameter conversions, 631 recurrence/
feature dispatches, 13 rollout assembly dispatches and 17 head dispatches.
Some intervals cross these boundaries; none is split or assigned arbitrarily.

| Existing completion intervals | Mean |
| --- | ---: |
| Wholly within recurrence/features | 5.758 ms |
| Recurrence mixed with rollout assembly | 0.365 ms |
| Rollout assembly alone | 0.296 ms |
| Batched heads alone | 0.821 ms |
| Initial parameter preparation mixed with recurrence | 0.073 ms |
| Complete imagination | 7.312 ms |

Within the 5.758 ms recurrence-only intervals:

- Matrix-only intervals: **2.472 ms**, containing 148 products, not all 180.
- Matrix/non-matrix mixed intervals: **0.827 ms**.
- Intervals without a matrix: **2.458 ms**. Their composed-operation labels
  distinguish 0.880 ms copy/conversion, 0.559 ms normalization, 0.420 ms
  categorical processing, 0.446 ms other processing and 0.153 ms mixed
  copy/other work. These describe complete shaders, not individual primitives.

The remaining matrix products also occur in mixed intervals. These numbers
are marked completion-frontier intervals, not pure kernel latency; their
measurement overhead cannot be subtracted uniformly. The complete observer's
3.66% overhead and all original samples remain in the preceding result.

## Corresponding JAX recurrence

All eight retained full-JAX updates give 180 matrix kernels totaling **2.106 ms**,
450 other kernels totaling **0.747 ms**, and 45 indexed copies totaling
**0.055 ms**. Kernel/copy activity union is 2.907 ms; first-to-last span is
3.147 ms. The previously audited action-channel padding remains explicit.

Do not divide the whole native 7.312 ms by that span: native also contains
heads and different output assembly. Even the recurrence boundary differs:
native exports F32 features/one-hot actions and receives prepared uniforms;
JAX exports BF16 states/integer actions and generates random samples in its
loop. Neither column is SM utilization or an additive savings budget.

This rules out simply blaming hundreds of extra launches in this particular
region. It also prevents the opposite shortcut: the preceding **7.42% fewer
native recurrent-matrix warp instructions** do not imply equal matrix latency.
The separate, still-unqualified training-prior batching experiment remains
distinct; imagined recurrence must stay causal.

## Complete membership, including ambiguity

The CPU review covers all four native programs: **4,692 dispatches, 8,948
versioned reads, 716 matrix products and 1,699 intervals**, totaling the retained
39.909 ms cached execution. World posterior/other-forward/adjoint dependencies
and every mixed boundary remain in the raw inventory; no non-convolution tail
is omitted. All 28,064 JAX kernels retain their per-update counts and durations.

Generated symbols can be shared by several HLO computations. Matching only an
SSA name was insufficient. Exact physical shapes, operand order, constants and
computation bodies now reconcile 669 grouped fusion-symbol rows, with 1,120
independently checked symbol/candidate pairs. Multiple equivalent candidates
stay grouped. Another 82 rows map directly through their launch thunks, six
rows to the sort launch family, and 54 library-main/helper groups retain their
candidate sets rather than invented individual-operation ownership.

One **0.01120025 ms** symbol remains unresolved at exact-computation level
inside its known posterior context. It is neither omitted nor attributed to
imagination. Thus complete accounting does not mean every library launch has
a unique HLO owner. Twenty malformed-membership/name checks and six valid
sort-name cases pass. Independent review finishes **03:27:19 UTC**.
The final source-closure check covers 1,712 seals, including all 1,677 original
GPU-decision seals, and finishes **03:30:48 UTC** without GPU work.

The first sort reader incorrectly requires a suffix after the actual `sort.5.0`
name; the second accepts only its exact normalized symbol and misses five
suffixed launches. Both failed sources/logs remain. Full six-name enumeration
and a separate third reader pass; no GPU repeat or numerical gate changes.

## Upstream changes the next implementation question

Five new Meganeura commits through **`5d6d9a36`** add a typed rectangular
workgroup-matrix implementation and measured-search integration. Matching
operand storage loads directly through workgroup cooperative operations; this
is a new implementation relative to our current runtime, not a demonstrated
Kindle speedup. Upstream pins Blade `4845d652` and Naga `13f582dd`.

The diagnostic override defaults to `None`; use requires explicit selection
or measured search, plus device/shape/type checks. Our retained runtime does
not contain this family. No silent refresh, compiler edit, search or new
performance claim occurs here. Upstream's other-model measurements are not
our full-update acceptance test.

Next CPU-audit this existing upstream implementation against **all current
dense interfaces**, including input/output types, transposes, split reductions,
binding extents, device support and compiler compatibility. That must precede
any separately declared complete comparison. It is not permission for another
geometry sweep or a favorable subset. Non-matrix processing and convolution
feeding remain separate problems; this new family alone is not a parity plan.

Artifacts: `runs/nonconvolution-timing-20261017.bGwyvF7S`. All raw memberships,
explicit ambiguities, source receipts, upstream diff and failed readers remain.
CPU services use one CPU, 2 GiB and no swap. The root-cause/parity goal stays open.
