# A useful convolution improvement from hardware-row alignment

October12,2026. [Meganeura PR252](https://github.com/kvark/meganeura/pull/252)
corrects one conservative selection condition; no shader or precision policy
changes. Batched channels need to align with independently guarded16-row
hardware stores, not the full32-row workgroup. This admits Kindle's48-channel
forward/input-gradient layers. [Declaration and failure review](../experiments/2026-10-12-convolution-subtiles.md)
· [Data](2026-10-12-convolution-subtiles.json).

| Ordinary complete update | Control | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| CDP12M/B32/F16 |161.512 ms|158.118 ms|2.10%|
| RGB12M/B8/F16 |71.576 ms|70.056 ms|2.12%|

CDP posterior19.380→18.262 ms; world training/sync56.536→54.492 ms.
RGB world training/sync39.743→38.618 ms. Each arm has16 warmup+256 measured
updates, identical checkpoints/config/RNG, no dispatch instrumentation,
compilation or other GPU work during timing. World-gradient forward cooperative
counts rise3→4 for CDP and5→7 for RGB. This is synthetic whole-update throughput,
not game throughput, learning quality, time-to-score or SM utilization.

## Numerical scope and retained failure

The first broad fixture stops after3.12s in the **unchanged unsplit scalar weight
gradient**, not the newly selected forward/input kernels: index11344 is
−.0020698346 versus F64−.002082056555, outside the unchanged absolute-plus-relative
bound. A separately declared unchanged control with all cooperative paths
disabled reproduces the exact failed output bits and F64 error. The failed
guard/source remain; this is not a successful F64 weight-gradient check.

The reviewed qualification therefore tests the two changed primitives directly,
with the same32-image shapes16→48 and48→16,5×5 windows/random data and unchanged
tolerances. Every forward/input output passes independent F64 checks for
F16-rounded operands plus unrounded scalar/tiny-gradient controls. Both actual
cooperative kernels must be selected. It passes on both the integration
foundation and standalone maind7a7293/Blade56f0565. No smaller fixture, tolerance
relaxation, weight-gradient change or claim that the broader failure disappeared.

All272 full-update reports/first targets and initial/first/final checkpoints are
finite; exact initial346/292 tensors match. Controls exactly reproduce the
retained pinned F16 fixture's reports/targets/tensors. Candidate masters change.
Six selection unit tests and strict all-target/all-feature Clippy pass on the
standalone patch and integration. Seven guards pass across the scoped timings,
unchanged-control diagnosis and standalone qualification; one failed original
guard and one allocation warning remain. Zero game actions or GPU recovery.
Preparation's private-method error and missing-spec controller failure started
no GPU child; source/logs remain.
Standalone [CI961](https://github.com/kvark/meganeura/actions/runs/38192173821)
passes all jobs. The final seven-guard CPU audit passes in4.08s.

## Provenance and decision

Standalone9e02edd is based on maind7a7293; integration6698592 contains the same
two-file patch on BF16/tracingb0ba23c. Full timings compare7ba39d8 with that
integration; b0ba23c changes only the equivalent deadline-reservation predicate/
test. Shared Bladeb52965b/Naga forkc9fd6a85a. No rejected weight-gradientPR248
or row-layoutPR249 is stacked. Embedded679ea87 is not the effective revision.

Keep this small measured improvement for integration, pending upstream review;
only the user merges. PR248 remains unadopted. Broader useful mixed precision
and the final combined stack remain unfinished; do not restart games.
Artifacts: `runs/conv-subtile-20261012.h7EJMiRL`, including source archives,
all binaries/guards, full reports/checkpoints and independent CPU audit.
