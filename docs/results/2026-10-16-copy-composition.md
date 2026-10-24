# Private copy/cast composition qualifies

October16,2026. **Complete RGB updates are4.02% faster, CDP3.79% faster, with
exact trajectories.** Keep isolated native`58aff452` in
`runs/copy-composition-20261016.ABr4pCUH`. RGB is now **1.697× inclusive JAX**,
not parity. Production and Atari remain unchanged; no worker remains.

| Complete12M/B32/T16 update | Control | Copy composition | Less wall time |
| --- | ---: | ---: | ---: |
| RGB |48.133ms|46.199ms|4.02%|
| Centered CDP |45.669ms|43.939ms|3.79%|

The unchanged inclusive JAX control is27.222569ms; CDP is a different workload.
Fresh controls differ from retained834241b6 by+.1363%/+.3604%, within5%.
Both performance gates pass: RGB improvement≥3%, CDP regression≤2%. All256
measured samples per arm and16 warmups remain. RGB first/last32 means are
48.065/48.163ms control and46.150/46.326ms candidate; CDP45.733/45.651ms and
43.823/43.783ms. No retiming or sample trimming.

## What changed

A general compiler pass composes private concat/repeat copies, optionally ending
at an existing BF16/F16 conversion. Seven input bindings plus one output retain
the existing bound; larger trees compose in pieces. No arithmetic, reduction
order, precision policy, matrix geometry, model, targets or replay ratio changes.
Keep required tapes, external values, metadata reads, source versions and cast
boundaries. Reject in-place aliases and overwritten inputs. BF16 rounding and
F16 rounding/odd-tail packing reuse the original implementations.

Across complete RGB programs,93 rewritten operations remove362 dispatches and
1,190,194,944 logical write bytes. CDP112 rewrites remove458 dispatches and
1,206,650,112 logical bytes. Slow-value and behavior programs are unchanged.
Posterior aliases the staged world program; it is not counted twice. These are
logical write extents, **not measured DRAM traffic or VRAM savings**.

Most measured benefit appears in imagination/targets: RGB11.923→10.208ms and
CDP21.403→19.663ms. These caller stages include host work/waits, not pure kernel
costs. RGB posterior+world changes31.762→31.440ms; CDP19.713→19.734ms. Do not
add the earlier saved-output intervention differences to these measurements.

## Qualification and provenance

- Independent materialized arrays verify319,752,192 source addresses across
  all eight programs. Eleven Rust fixtures cover ownership, metadata, versions,
  aliases, casts, odd tails and the seven-binding limit.615 typed shader variants
  validate offline; backend and observer Clippy pass.
- Actual RGB/CDP raw gradients match bit-for-bit for original and changed reward
  targets:47,081,862 values per complete comparison,68/86 parameter tensors per
  condition. Whole/staged execution agrees with late targets NaN-poisoned during
  preparation and restored before completion. All18 raw checkpoint pairs pass;
  moments/parameters stay unchanged during diagnostics and restore exactly.
- Both24-update smokes and272-update timings match every metric report, initial/
  first/final292 RGB or346 CDP tensors, first targets and retained controls.
  Each timing prefix also matches its smoke. No tolerance was relaxed.
- Complete plans/constants, source versions, protected outputs and unchanged
  pipeline keys reconcile against the independent policy. Captured world-group
  membership/dependencies pass. **Non-world runtime groups were not exported**;
  do not claim their independent membership audit.
- Every executed copy shader matches a fresh CPU generation from its actual
  emitted kernel:24 RGB/32 CDP modules across93/112 occurrences. Six occurrences
  merely exchange independent helper nodes2/3; exact index-expression and node-
  permutation proofs explain their different source hash. Three additional
  modules per workload are outside the exported learner programs. Construction
  also creates live-actor sessions; they are not exercised by these updates.
  No live-actor or F16 GPU qualification is claimed.
- All12 GPU guards pass, ten allocation warnings are retained, no recovery or
  separate NVML polling. Minimum sampled budget headroom is12,315,787,264bytes,
  not physical free/peak VRAM. Total1,188 synthetic updates,16 extra raw backwards,
  zero game actions. GPU finishes11:51:32UTC; final CPU audit11:59:49UTC.
  All3,044 timing source seals pass. Upstream remainsf648a453/4845d652; no repin.

Keep flags `MEGANEURA_COPY_COMPOSITION=1`, `MEGANEURA_CONVOLUTION_WORDS=1`,
`MEGANEURA_IMAGE_VIEWS=1`, `KINDLE_DEVICE_MEANS=1` and cached recording. The
same binary with composition disabled reproduces834241b6's control. Diagnostic
test additions do not run inside timed updates. No production adoption follows.

## Retained failures and decision

The first arity fixture was disconnected; the corrected eight-leaf fixture
enforces the bound. Keep explicit-reference Clippy failures, the unknown`Sqrt`
fixture and the observer's unavailable`bytemuck` dependency; corrected sources
pass. CPU publication initially looked for groups in the emptied plan field,
then compared a construction-wide shader dump against learner-only modules.
Its first scope repair missed helper reordering, and the next observer counted
two occurrences instead of six. All original sources/logs remain. Corrected
schema, expression/permutation and execution-scope reviews pass; **no GPU
retry was made**, limits raised or numerical gate changed.

This confirms a real general materialization/dispatch cost, but removes only
1.934ms of the previous complete RGB gap. The remaining difference is18.977ms.
Next compare the remaining matrix/convolution instruction and specialization
contracts against matched JAX work before another implementation. No copy/tile/
precision sweep. Prior batching stays numerically unqualified and untimed;
convolution feeding, whole-stack parity and learning quality remain open.

[Data](2026-10-16-copy-composition.json) ·
[CPU declaration](../experiments/2026-10-16-copy-composition.md) ·
[Implementation](../experiments/2026-10-16-copy-composition-implementation.md) ·
[Numerical allocation](../experiments/2026-10-16-copy-composition-qualification.md) ·
[Reader repair](../experiments/2026-10-16-copy-composition-review-repair.md) ·
[Timing](../experiments/2026-10-16-copy-composition-timing.md) ·
[Shader scope review](../experiments/2026-10-16-copy-composition-shader-review.md)
