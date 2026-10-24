# The imagined matrix work matches; individual launch ownership is unresolved

October16,2026. [Declaration](../experiments/2026-10-16-jax-matrix-contract.md)
· [Compact evidence](2026-10-16-jax-matrix-contract.json).

The common imagined recurrence contains **12 product roles/180 products per
update in both stacks**. This CPU audit reconciles current58aff452 native plans
with the retained actual JAX capture. No new GPU counters or complete-update
measurement: **46.199/27.223ms,1.70×** stays. Production/Atari remain unchanged.

## Interfaces and coverage

All15 native occurrences of each role match JAX dimensions/precision, except
JAX pads action width18→24 in actor output and action-input projection. Native
nominal arithmetic is44.865946G MACs, JAX44.889539G. Extra native matrix capacity
does not explain the gap.

Eight ordinary non-action products have matching dense row-major BF16 layouts.
The two grouped products swap logical operand order in JAX, but physical layouts
match: input `(row,group,k)` strides `(groups*K,K,1)`, weights `(group,k,column)`
strides `(K*N,N,1)`, and JAX output `(group,column,row)` strides `(N,1,groups*N)`
equal native `(row,group,column)` storage. All20,933,632 physical-address checks
pass, including explicit mapping of unpadded action coordinates into padded
arrays—not equal physical action-row strides or model values.

**Grouped boundaries differ:** JAX uses `beta=1` and a BF16 addend in the library
call; native's matrix has no addend and `BlockLinear::forward` adds bias outside.
All12 JAX epilogues are `DEFAULT`. A kernel name containing `relu` does not prove
a model ReLU. Retain surrounding additions/conversions/rounding when comparing
complete operations.

Existing native counters cover **150/180 recurrent occurrences**, through seven
tiled contracts. The30 action-related products are not covered at their actual
512-row shapes; older checked controls use other shapes. Do not substitute those
controls or silently omit these products from a full recurrence ratio.

## What the trace establishes

Two enclosing command buffers contain five/seven matrix launch nodes. Their12
stable graph-node identities each occur15 times in eight updates: **1,440
launches**,2.105519ms summed matrix-kernel time/update. Independent SQL counts,
durations and geometry agree. This is not a native pass or whole-update boundary.

Most names identify32×32 WMMA-family kernels with9,728/18,944/37,376B shared
storage. One identifies a64×256 tensor-operation family with81,920B and242
registers/thread. These are trace metadata, not native instruction counters or
an efficiency explanation. The80KiB case exceeds the retained48KiB Vulkan
explicit shared limit, but establishes neither its latency advantage nor parity
impossibility. Convolution feeding cannot be assigned to every matrix kernel.

The trace proves enclosing command-buffer membership, **not each individual
HLO operation's launch identity**. Several operations share names/geometry.
First-launch order or a suggestive name is insufficient; no invented per-layer
timing or HLO-specific selector is published. Aggregate collection over all12
nodes avoids that ambiguity; individual claims need stronger ownership evidence.

## Upstream and next action

Meganeura advances through **12df95c**, six commits covering operand rejection,
generalized F16/BF16 cooperative products, bounded K-partition tuning, shared
fast dense staging and native-matrix accuracy. Blade remains4845d652. Our isolated
checked path already stages vectors but is not the same new implementation.
Upstream `coop_tiled.rs` retains blobcdb65862 across these commits and lacks the
isolated typed/grouped branches used by the150 recurrent tiled occurrences.
This is source review, not runtime equivalence or qualification of a refresh.

The primitive accuracy policy explicitly does not waive differentiated-graph
failures. It does not retroactively qualify prior batching or change retained
gates. No repin, backend build or old-result relabeling occurs.

Next prepare one bounded **actual full-JAX12-node counter collection**, retaining
full-update numerical/compiled-work controls, plus the two missing native action
contracts. Report the recurrence aggregate first, with padding/addend limits.
This is unchanged-kernel observation, not a new variant/geometry/layout/precision
sweep or learning allocation. No GPU work is launched by this CPU report.

## Review

Seven negatives reject wrong output orientation/group stride, missing addend,
omitted action padding, invented ReLU, lost occurrence and ambiguous ownership.
19 source/artifact seals pass; CPU review finishes14:51:56UTC. One CPU,2GiB,no
swap; all services terminal. Zero GPU calls/builds/updates/actions.

The first reviewer incorrectly asserted upstream had no `coop_tiled.rs`; source/
failed log remain. A separate source-membership repair proves the unchanged blob
and isolated typed branches, without changing numerical gates or repeating GPU
work. Exploratory JSON/path reader errors were not measurements. Artifacts:
`runs/jax-matrix-contract-20261016.y2ImIOLk`.
