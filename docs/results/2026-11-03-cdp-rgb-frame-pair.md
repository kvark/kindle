# Native RGB frame pairs: implementation qualification

The optional six-channel input is implemented, with CPU and GPU geometry checks
passing. A numerical derivative fixture needs its perturbation corrected before
remaining native checks or Atari training. Gameplay benefit and bounded cost are
unmeasured. This work is part of the existing five-game goal; Breakout retains
70.750 sampled/143.667 greedy and its original864/90%-two-wall gate.
See the [fixed protocol](../experiments/2026-11-03-cdp-rgb-frame-pair.md).

## Implementation and package

Production source `24e701bce7e65004a84f54fa29073fc4ef1f3355` adds default-off
`rgb_frame_pair`. Each arriving native pooled image is resized once on GPU,
then paired with the same stream's previous resized image, oldest first. The
first image after reset is duplicated. Missing arrivals preserve that stream's
private history. Replay stores both images together; restore retains geometry
and starts fresh live history. RGB reconstruction and color-centroid supervision
reject this option. Only the first convolution widens3→6, adding2,400 weights
at12M. Later widths stay fixed; RGB replay storage doubles.

The isolated package root is `/x/Temp/kindle-rgb-frame-pair-20261103`.
`build-receipt.json` records220 source pins, compiler/dependency identity and
separate build/test logs. It completes at23:04:52 UTC November3 with171 Kindle
and5 gym Rust CPU tests,993 packaged Python tests (10 deselected), strict
workspace/binding Clippy and a built wheel. No native agent is constructed by
the packaged Python CPU suite. Runtime is `runtime/`, native SHA256
`b451bc981a76a7ccc455ca22ffe637151d99353ea520fb9642cae217e8cc73e9`.
Original validation executable SHA256 is
`d3af6742847fa5b3a31a1e10f9a0832487ffb7fa92218edf5ae2440c03488600`.
Build cache remains pending final packaging closure.

## Geometry passes; original derivative failure is preserved

The declared serial native queue passes the independent32/48/64 GPU resize/pair
fixture, including multiple streams, missing arrivals, resets, and CPU/resident
frame sources. The test takes0.75s; its host guard passes. It will not repeat.

The second fixture passes scalar F64 CNN outputs/loss but fails a derivative
assertion: GPU0.057106383 versus finite difference0.056595063837576225.
The difference0.000511319 exceeds the original tolerance0.000469785. The native
process exits101; there is no recorded GPU fault or host recovery. The original
queue stops, preserving its logs, declaration and `native-command.json` exit1.
The third replay/restore fixture remains unstarted at this point.

## Targeted derivative diagnosis

Test-only source `3d24bd20` adds an exclusive capture of parameters, gradients,
pixels and scalar/GPU outputs from the same seed103 six-channel fixture. It
changes no production source or packaged native binary. The original test source
is saved byte-exact as `derivative-diagnostic/rgb-tests-original.rs`, resolving
the sole changed path among the220 original source pins.

The diagnostic test binary builds successfully. Initial CPU preparation assumes
only one Cargo test executable, but package/workspace builds have different
hashes; the assertion stops before queue creation. A subsequent command fails
on the missing queue file. Both errors are retained in `preparation-failure.json`
and original queue logs. Corrected preparation selects the exact executable from
the successful build log; no build or GPU geometry stage repeats.

One new guarded derivative capture passes in0.80s, with no optimizer updates or
Atari actions. CPU analysis takes0.119s. Its independently implemented NumPy F64
CNN agrees with the Rust scalar outputs to2.89e-15 and GPU outputs to1.61e-6.
Five fixed perturbations,1e-3 through1e-7, cover the same five selected derivative
probes, including both input-image halves. The failing probe is first-layer
weight75, in the current-image half. At1e-4, the positive/negative perturbations
change two/one max-pool winners in the second layer. At1e-5 and1e-6 no winners
change, and the derivative converges to0.057106416561, within3.35e-8 of the GPU.
All five probes converge with no pool changes at those smaller steps.

The fixture correction uses1e-5 for paired input only, preserves1e-4 for the
legacy three-channel test, and leaves the acceptance tolerance unchanged.
This diagnoses the fixed fixture's discrepancy; it is not broad numerical or
gameplay evidence. Captured data, analysis, all failures and both test sources
remain under `derivative-diagnostic/`.

## Remaining work

Corrected guarded derivatives and the unstarted replay/restore fixture remain
pending. Legacy single-image compatibility must then pass before the separately
prepared8192-action/489-update and1024-action frozen check. Cost must project
under30min/200k with10% padding, and projected full100k replay must retain2GiB
GPU headroom. No full learner, extension or automatic successor is selected.
