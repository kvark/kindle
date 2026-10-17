# Convolution load contract: CPU diagnosis and feasibility

Keep58aff452/46.199ms and production/Atari unchanged. The original CPU-only
declaration is retained in
`runs/convolution-load-contract-20261016.fWW2fJ4W/declaration.md`.

Reconcile all23 actual products/20 contracts with retained generated WGSL,
runtime-option SPIR-V and counter captures. Count reachable operand loads,
predicates, address operations and cooperative operations, excluding dead
helpers. Check actual storage types and bounds policy, not presumed BF16
conversions or injected checks. Preserve existing cursors, splits, compensation,
barriers, rounding, tile geometry and numerical gates.

For all nine typed occurrences, prove four-value operand addresses/masks and
full aligned contiguous spans in the current32×32×32 staging schedule.
Keep partial, unaligned and noncontiguous groups; no read into allocator padding
or beyond the logical input. Account for unchanged fourteen packed products.
Counts describe source operations, not hardware instructions/traffic/speed.

Independent review may re-emit CPU shaders using the retained observer, without
rebuilding the native backend. Check original source receipts, all actual
modules and a separate address oracle. Review ordinary/external/donor binding
construction; the raw alias helper does not itself establish alignment.

One CPU,2GiB,no swap,720s per CPU stage. No GPU calls, native build, learner
updates or game actions. No shape/geometry/precision sweep, rejected pipeline
or packing-limit retime, compiler feature or promotion. Implementation and any
GPU qualification require a separate declaration after this review.

Upstream12df95c/4845d652 remains unchanged. [Result](../results/2026-10-16-convolution-load-contract.md).
