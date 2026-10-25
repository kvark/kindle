# Resident targets implemented; numerical qualification remains open

October16,2026. [Declaration](../experiments/2026-10-16-resident-targets.md)
· [Evidence](2026-10-16-resident-targets.json).

**The complete selected target path is implemented in isolation, but it does
not pass the declared end-to-end F64 envelope.** The same envelope also rejects
the original CPU arithmetic on another coefficient. No full-model timing,
promotion or training restart follows. Keep qualified native58aff452:
RGB46.199/JAX27.223ms,1.70×. The broader convolution/library gap remains open.

Artifacts: `runs/resident-targets-20261016.6aBO2GAy`; observer27b8befc.
Meganeura/Blade artifacts are unchanged. Latest upstream12df95cd/4845d652 was
checked before implementation; no backend rebuild or repin.

## Implemented path

F32 imagined/replay returns and weights, resident linear5/95 percentile/EMA,
weighted action/dense two-hot targets, GPU replay-feature handoff and staged
world late-target writes. Existing losses, RNG, masks, arrival-aligned CDP bonuses,
normalizer checkpoint/restore and production replay-context refresh remain.
The CPU normalizer is not used by the selected GPU path.

Seven behavior inputs are written directly by compute; the replay feature comes
from the existing GPU imagination output. Existing checked copies deliver the
world targets. The normalizer stays resident; its CPU checkpoint mirror and small
report are refreshed after the world-training wait. Both consumers must finish
before target inputs are reused. The existing posterior context readback remains.

For actual RGB/CDP B32/T16/H15, the new module owns598,892bytes and submits10
compute dispatches.81 additional device-copy regions cover6.486MB logically,
including replay features and80 world inputs. Host replay scalar inputs are
6,144bytes, report/status104bytes. These are extents and source counts, not
measured traffic or savings. The new sort supports at most8192 returns and
requires staged full-BPTT/full-microbatch training; unsupported recipes reject.

## Offline and GPU evidence

- Both actual12M plans pass16 behavior/160 world-target interface checks.
  Neither posterior prefix reads the late targets; no target is output scratch.
  Eight malformed-interface mutations reject.
- Ten Vulkan shader modules/11 entry points, the bitonic ordering/interpolation
  CPU check, seven existing distribution tests and Clippy pass. Maximum explicit
  workgroup memory is32KiB.
- Four fixed component shapes each exercise initial inputs, changed inputs with
  carried normalizer state, and restored initial inputs/state:12 generations.
  Identity consumers, mean-buffer canaries/values, all759 output arrays and301
  restored raw files pass their structural/exact checks. These component fixtures
  use48-wide features; they are not actual12M world-training qualification.
- One GPU guard passes with zero new allocation warnings, no learner updates
  or game actions. Last guard observation21:48:36UTC; no worker remains.

## Numerical result and calibration

The independent oracle performs19,652,262 value comparisons, including copied
world targets and exact-zero checks. Two occurrences fail: one replay two-hot
coefficient and its unchanged world copy, in B32 changed-inputs case3/round1.
All other declared comparisons pass.

| Quantity | Value |
|---|---:|
| Independent exact-rational reference |0.3009710599457349|
| GPU coefficient |0.3009541928768158|
| Absolute error |0.0000168670689191|
| Original allowed error |0.0000160194211989|

The failed reference is independently verified with rational arithmetic.
Given the same GPU scalar returns, encoding agrees with the CPU encoder within
one F32 ULP. The larger discrepancy is upstream in return arithmetic: this
replay return is0.05133003369 on GPU versus0.05133056641 under separately
rounded source-F32 operations, even with the captured GPU bootstrap supplied.
No native instruction-level contraction/reassociation cause is claimed.

The original Rust `lambda_returns` and `TwoHotBins::encode` are separately
compiled and run on all12 retained cases. Their7,839,540 values exactly match
the source-F32 transcription. That original CPU arithmetic also fails the same
F64 envelope at four occurrences of another coefficient, although it passes
the GPU's failed coefficient. Thus forcing CPU equality alone would not satisfy
this end-to-end envelope. These are calibration results, not erased failures.

One explicit-FMA affine interpolation was evaluated **on CPU only**, with the
same inputs and gates. It fails78 occurrences and is rejected; it is not added
to the shader or tried on GPU. Preserve its initial missing-helper-constant
failure, the two earlier mutable-test-caller errors and the reserved WGSL name
failure. No numerical tolerance was widened and no GPU cohort repeated.

## Next

Resolve the scalar-return versus encoding precision contract on this same
retained cohort before selecting a repair. Preserve the failed end-to-end gate
and the failing CPU control; do not chase only the GPU's single outlier, repeat
the rejected affine rewrite, or start a compiler feature project from it.
Any repaired GPU qualification and complete RGB/CDP comparison must be separately
declared. Production/Atari remain stopped; this is neither a speedup nor a
learning-quality result. The full JAX-parity goal remains open.
