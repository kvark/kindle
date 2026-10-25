# Convolution gain does not survive complete execution

October12,2026. The [bounded repeat](../experiments/2026-10-12-convolution-attribution.md)
keeps PR248 draft and unadopted. Its isolated kernel win is not a useful learner
win. [Data](2026-10-12-convolution-attribution.json).

| Ordinary full update | Control | Unchanged convolution candidate |
| --- | ---: | ---: |
| CDP12M/B32/F16 |161.311 ms|161.483 ms|
| RGB12M/B8/F16 |71.111 ms|72.822 ms|

CDP is effectively unchanged; RGB regresses2.41%. Corrected CDP captures include
all four world chunks in every256 measured updates. World GPU pass intervals
are52.930/52.810 ms; the final, convolution-backward-containing chunk is
29.540/29.416 ms. World-stage ordinary wall times56.447/56.246 ms agree with
this lack of a substantial gain. The earlier instrumented12.29→4.44 ms kernel
improvement does **not** appear as an8 ms saving hidden by CPU overhead.

This localizes the discrepancy to execution/instrumentation context, not to a
missing kernel selection: three CDP and seven RGB cooperative weight-gradient
dispatches are selected. It does not yet explain the kernel-level cause.
Per-dispatch instrumentation substantially perturbs execution and cannot stand
in for ordinary critical-path measurements. Complete-pass tracing also adds
2.87%/2.77% wall overhead; its intervals are not SM utilization.

The capability export contains F16 and BF16 cooperative shapes, no native F32
shapes. This separately motivates checking why the BF16 dense option disables
Auto acceleration for unconverted operations; it does not explain the unchanged
F16 convolution comparison or authorize another kernel sweep.

Seven guards pass, zero allocation warnings, zero game actions. F64/F16-rounded
derivatives and strict-range fallback pass. All272 reports/first targets and
checkpoints are finite; actual initial346/292 tensors match. Each trace exactly
matches its own ordinary arm's reports, targets and initial/first/final tensors.
The refreshed control also exactly matches the retained pinned control.

Effective Meganeura control7ba39d8, candidate d2de660 (unchanged PR248/7553463
on that base); shared Bladeb52965b/Naga forkc9fd6a85a. BF16 selection is off.
Kindle0010772 plus diagnostic-only shader/capability exports is archived with
both lockfiles. Embedded679ea87 is not the effective backend revision.
Artifacts: `runs/conv-attribution-20261012.j7K9T35C`. CPU preparation's missing
Cargo PATH and corrected invocation remain. No worker or automatic retry remains.
