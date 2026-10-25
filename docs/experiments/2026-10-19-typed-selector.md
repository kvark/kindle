# Typed reuse in the common selector

Start from the qualified October19 isolated stack `d3ff4948`, not the slower
first refresh. RGB42.982911ms versus inclusive JAX27.222569ms remains1.58x.
No learning, production promotion or old queue restart.

First audit the actual RGB/CDP plans and the selector's storage, dimensions,
partition/reduction, addend, aliasing and device-resource contracts on CPU.
The retained typed dense entries are currently absent from `TuneClass`.
Its typed domain also excludes the larger shared-reuse implementation.
Ordinary construction does not run measured search.

Expose the existing shared typed implementation, not another BF16 shader or a
new geometry catalog. Preserve the ordinary incumbent, all conversions,
partition reductions, checked tails and optional device capabilities. Reject
unsupported grouped layouts explicitly rather than treating groups as K splits.
CPU fixtures cover F32/F16/BF16 storage, transposes, addends, split reductions,
resource limits and unchanged actual-incumbent generated sources. Review the
actual candidate inventory before any separately declared GPU qualification or
whole-program search. No GPU allocation is authorized by this CPU declaration.

Use an isolated source/cache under `runs/typed-selector-20261019.MhXVhJTs`;
preserve the qualified source and unrelated repository work. Preparation,
builds and audits use1CPU/2GiB/no swap/720s per service. Keep failures and source
receipts. No convolution sweep, precision change, Naga feature, relaxed numerical
gate or timing claim follows from CPU selector coverage.
