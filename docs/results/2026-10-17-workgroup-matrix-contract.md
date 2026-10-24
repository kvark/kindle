# Upstream workgroup matrices fit the current dense interfaces

October 17, 2026. [Evidence](2026-10-17-workgroup-matrix-contract.json) ·
[Backend plan](../backend_training_performance_plan.md).

**CPU compatibility passes; GPU correctness and speed remain untested.** The
existing upstream implementation can cover much of our dense work without a
new Naga feature. This does not change the qualified complete result:
**RGB 43.197 ms versus inclusive JAX 27.223 ms, 1.59×**, native `e45b9c43`.
Production is unchanged; Atari remains stopped.

## One policy, complete membership

Audit all current RGB/CDP matrix occurrences against Meganeura `5d6d9a36`.
Use one upstream policy: **32 rows × 64 columns, K32, 64 invocations**. It also
fits the actual 32-row observation products. No geometry search or performance
selector occurs.

| Current workload | All products | Compatible dense products | Grouped fallback | Unaligned fallback |
| --- | ---: | ---: | ---: | ---: |
| RGB | 716 | 504 | 129 | 83 |
| CDP | 769 | 552 | 126 | 91 |

The compatible products comprise **61 distinct contracts and six shaders**:
BF16 operands, F32 accumulation, NN/AT/BT layouts, F32 or BF16 output, and the
actual AT/BT addend variants. They span world, imagination, behavior and slow
value programs. Grouped RSSM and unaligned products keep their existing paths.
No claim that all recurrent work or all heavy arithmetic changes.

All **106 selected split products** retain identical K partitions, F32 partial
buffers and their existing `SumRows` consumers. The upstream balanced partition
formula differs in general, but not for these actual contracts. Multiple splits
with reduced output or an addend are forbidden. Tensor accumulation itself may
still round differently; unchanged partitions do not establish numerical parity.

## Compiler, device and address checks

The generator function, matrix types and BF16 rounding helper are copied
unchanged from upstream. A small offline harness uses upstream Naga `13f582dd`;
it is not a native runtime refresh or compiler modification. All six actual
variants emit SPIR-V 1.6 and validate for Vulkan 1.3 with the installed SPIR-V
tool. No-transform validation preserves every binary word.

Independent review checks workgroup scope, matrix roles and pointer roots,
descriptor numbers, the nested 16-byte uniform ABI, output ownership and
**4,936,704 tiled address comparisons**. All 1,485 occurrence memberships and
current semantic plan identities reconcile. Fourteen selection/kernel negatives,
seven device-capability negatives and six compiler-capability negatives pass.

The retained 5080 query supports the fixed BF16 dimensions. Explicit output
scratch is zero for F32 output and 8 KiB for BF16 output; adding the reported
8 KiB compiler reservation fits the 48 KiB explicit limit. These are feasibility
bounds, not measured allocation, occupancy, feeding overlap or latency.
Blade `4845d652` enables the required workgroup/flexible-dimension features;
the retained runtime does not. Actual logical-device initialization and dispatch
must therefore be qualified in the next observer, not assumed from properties.

## Failures and next decision

Preserve four failed CPU readers: raw-versus-semantic plan hashes; the current
typed checked kernel's `Default` tag; stripped SPIR-V debug names; and Naga's
outer uniform-block wrapper. Separate readers repair those assumptions without
changing the shaders, numerical gates or resource limits. Both original
inventory/review sources and logs remain. Compiler build has one expected
dead-code warning for unused upstream type helpers. There was no GPU retry.

Independent review finishes **03:47:15 UTC**, covering 71 receipts. All services
are stopped. CPU services used one CPU, 2 GiB, zero swap and 720-second limits.
This allocation made **zero GPU calls, learner updates or game actions**.

Next prepare a bounded, fixed-policy GPU comparison using the existing observer:
qualify feature enablement, native binding/control calibration, independent
numerics and all compatible contracts; account for unchanged fallbacks and
split reductions in complete operation costs. Review those results before any
complete-model trial. No tile search, favorable subset, compiler feature
project, silent dependency refresh or learning promotion. Non-matrix work and
convolution feeding remain separate causes of the remaining gap.

Artifacts: `runs/workgroup-matrix-contract-20261017.IWJLx2Fi`.
