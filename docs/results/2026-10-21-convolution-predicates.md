# Pure convolution predicates pass the complete-update cost gate

October 21, 2026. [Compact evidence](2026-10-21-convolution-predicates.json) ·
[Backend plan](../backend_training_performance_plan.md).

**An isolated bounds-predicate rewrite reduces complete RGB update time by
11.65% and CDP by 3.25%, with exact synthetic training trajectories.** This
passes the predeclared RGB ≥3% saving / CDP ≤2% regression gates. Keep the
candidate for further qualification; production and Atari remain unchanged.

| Complete update, ms | Fresh control | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| RGB12M/B32/T16 | 48.452 | 42.808 | 11.65% |
| CDP12M/B32/T16 | 41.927 | 40.563 | 3.25% |

Both controls calibrate against the retained current-driver results: +0.020%
RGB / +0.081% CDP, within the original 5% gate. Every arm retains 16 warmup
and 256 measured updates, all samples and the caller-inclusive boundary.
There was no profiling, concurrent build, retime or favorable subset selection.
The candidate's RGB time is 1.642× retained current-driver inclusive JAX
26.066ms, down from the retained native 1.858×. JAX was not rerun here.

## Mechanism and what remains unproven

Only two copied Meganeura files change: `codegen/cooperative/convolution_words.rs`
and `shaders/conv_tile_address.wgsl`. Pure compound bounds checks use
`all(vecN<bool>(...))` instead of short-circuit conjunctions. Storage reads
remain conditional. Input-gradient nonnegative-coordinate guards still precede
signed remainder evaluation. No tile, split, precision, rounding, K order,
compensation, address, layout, matrix operation or barrier changes.

The CPU audit covers all 23 actual convolution occurrences. Nine typed
occurrences change; fourteen packed products retain byte-identical WGSL and
SPIR-V. Four distinct runtime shader bodies change per workload. Call-expanded
static conditional branch sites fall from 69→21 for typed forward, 85→37 for
input gradients and 75→27 for weight gradients. These are compiler-IR counts,
**not measured hardware instructions**. Both old and new memory-site sequences,
resource contracts, matrix operations and barriers reconcile.

The ordinary stage measurements place most improvement in world training:
RGB 30.738→25.183ms and CDP 14.215→12.991ms. This does not isolate hardware
branch cost, prove why the driver emits different code, or reproduce cuDNN's
larger tiles and asynchronous feeding. The historical native/JAX feeding deficit
and the current-driver instruction increase remain distinct questions.

Next: independent GPU primitive/reference qualification of the changed family,
then a bounded actual-model counter comparison using the already-qualified
barrier-aligned observer. Preserve exact source membership and ordinary outputs;
do not insert synchronization merely to obtain finer attribution. No complete
timing repeat, geometry sweep, new compiler feature or automatic promotion.
The successful cost result alone is not arbitrary-input correctness, learning
parity, or five-game competence.

## Qualification

- CPU: 92 boolean cases, 81,920 coordinate boundary cases, 23 actual plus 42
  general shader configurations; all 130 old/new SPIR-V modules validate.
  F32/F16/BF16 variants, boundary guards and independent finite source rewrites
  are covered. Naga and Blade build artifacts are unchanged; Clippy passes.
- GPU synthetic model checks: exact 24-update prefixes and all 272 reports per
  timing arm, first targets, initial/first/final 292 RGB or 346 CDP tensors.
  Fresh controls also reproduce retained trajectories; candidate timing matches
  its own prefix. Complete plans, pipeline exports, groups and command-cache
  records reconcile. All 205/225 shader bodies match the declared transformation.
- Six passing GPU guards, zero new allocation warnings, 1,136 synthetic updates
  and zero game actions. No NVML polling or recovery. Independent primitive GPU
  oracles, staged raw-gradient checks and candidate hardware counters are unrun.

## Retained failures and repairs

The first preparation passed a WGSL file to rustfmt; a separate build script
selects the Rust source. The first compiled probe then fails Naga parsing of
comparison expressions in vector constructors. Parenthesized expressions fix
that ambiguity in a separate source copy and build cache; both originals stay.

The first prefix queue fails schema validation before spawning any GPU child:
its historical host object contains an unsupported, empty
`reviewed_process_faults` field. Removing only that empty field passes the
current guard preflight; no fault is waived.

The timing controller incorrectly expects CDP's final learner step to be 272,
although its restored checkpoint starts at 124,939. It stops after three
successful children. Their completed trajectories are audited before a separate
continuation runs **only the unstarted CDP candidate**. Both CDP arms finish at
125,211; the corrected review checks initial step +272. The original controller
failure remains. No completed GPU cohort is repeated for these repairs.

The final independent timing reader initially uses Python 3.14's compensated
`sum`, unlike Rust's sequential F64 fold, causing a 0–3 ULP mean mismatch.
A separate explicit-fold reader restores exact equality with no tolerance or
sample change. Final publication checks all 883 source receipts at 10:14:56 UTC;
all services are terminal. The original reader failure remains.

## Artifacts and cleanup

Original failed source: `runs/convolution-predicates-20261021.wsBteZ9d`.
Qualified candidate, independent build, CPU proof, prefixes, complete samples,
guard logs and publication audit:
`runs/convolution-predicates-fixed-20261021.oF3Xs5YJ`.
Candidate binary SHA begins `79bc7cec`; retained control is `72005c8b`.
Meganeura upstream remains `a75c38e`; neighboring repositories are untouched.

The separate `runs/readout-dedup-20261021.dhR8EhJK` cleanup reclaims **18.26GiB**
from three completed immutable convolution studies. It replaces 247 exact
duplicate readout files with hardlinks, verifies all 273 original paths and
bytes, and retains original metadata in its manifest. No unique data, source,
binary, checkpoint or path is removed. These archives now share inodes: copy
to an independent file before mutation. This is not deletion of failed evidence.
