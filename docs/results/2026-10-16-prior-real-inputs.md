# Real-input prior divergence starts at the first matrix

October16,2026. **Replacing only the batched first prior product with the serial
product makes all later activations and logits exact**, for both saved RGB/CDP
inputs. This attributes the frozen forward difference, not all failed world
gradients. Prior batching remains unqualified; no timing follows.
[Declaration](../experiments/2026-10-16-prior-real-inputs.md),
[data](2026-10-16-prior-real-inputs.json),
[failed world qualification](2026-10-16-prior-world.md).

## Intervention

Reuse the unchanged standalone B32/T16 plans and qualified834241b6 libraries,
with all eight actual prior parameters and16 deterministic states from each
world checkpoint. CPU version analysis proves that102 matrix occurrences and
their adjacent kernels match the actual-world contracts, including time order.
Only the closed prior-logit dependency chain executes:226 serial/29 batched
dispatches. Original allocation, pipeline and group boundaries remain; unrelated
KL/backward operations and zero posterior placeholders are not consumed.

Each workload has three conditions: serial, batched, and batched with an exact
GPU copy of the serial first-product BF16 bytes before its consumers. The
ordinary batched product is captured before replacement and remains exact with
the separate batched control. All40 input/parameter arrays remain immutable.

| Frozen workload | Different first-product BF16 values | Different final logits | Different logits after replacement |
| --- | ---: | ---: | ---: |
| RGB | 135 /131,072 | 8,028 /262,144 | 0 |
| CDP | 163 /131,072 | 3,549 /262,144 | 0 |

All3,145,728 matrix/bias/normalization/activation values across the two
substituted paths match serial exactly, including the copied products. The
524,288 final logits are included in that count, not additional independent
values. An independent CPU audit rechecks7,340,032 paired values, all source
parameter/deterministic-state bits, closure membership and524,288 copied bytes.
Four negative tests reject altered values/shapes, missing work and a posterior
dependency. This is a controlled forward intervention, not a speed measurement.

## What changed numerically

The serial first layer uses eight K partitions, F32 partials and sequential
F32 reduction before BF16 conversion. The larger batched product uses one tiled
accumulation with BF16 output. Captured serial partials reproduce `SumRows` and
its final BF16 boundary exactly;2,097,152 partial/F64 comparisons pass the
unchanged2e-5+.02|reference| and relativeL2<=.01 gates. Normalization and SiLU
each cover1,572,864 values compatible with the previously checked precision
contracts. The later layers introduce no difference when given identical first
products, for these frozen inputs.

The CPU dot16/F32 hypothesis predicted only15/17 changed BF16 values, versus
the GPU's135/163. It is **not a bit-exact GPU simulator**. Relative to F64 rounded
to BF16, serial changes20/29 values; batched changes145/168. The precise internal
cooperative accumulation rule is not established. This experiment isolates
the first product's changed implementation, not partitioning versus tiling as
independent causes. It establishes neither a missing BF16 conversion nor a
general error in recurrent dependencies.

## Limits and next step

The original113 world-gradient failures remain failed. No raw backward was
executed here; in particular, batched gradient reductions may introduce
additional differences even with an identical forward tape. Next use this
exact first-product intervention in a separately declared full-world raw-
gradient diagnostic, with an unchanged ordinary control and both retained
target conditions. It must reproduce the old values before attribution. Do not
relax the original gates, replace the failed cohort or time the candidate.
No K/tile/precision sweep, backend feature project or training restart follows.

One guard/no new warnings, six forward closures/568 graph dispatches plus two
replacement copies; zero raw backwards, optimizer updates or game actions.
GPU service completes09:46:51UTC; independent CPU review passes, no worker
remains. Minimum sampled estimated headroom16,446,586,880bytes.2,855 source
seals pass. No backend rebuild; only the small reader is linked to sealed
libraries. Preserve the CPU stdlib-shadowing, constant-field bound, external-
input/schema-reader and preflight status-label failures; their source/logs and
repairs remain. Independent full-JSON checks cover all six repaired constant
readers/four external-input readers without raising limits or repeating GPU work.

Keep834241b6: complete RGB48.068ms versus inclusive JAX27.223ms,1.77×.
Production/Atari and the wider performance goal are unchanged. Raw artifacts:
`runs/prior-real-inputs-20261016.At0oVqjy`.
