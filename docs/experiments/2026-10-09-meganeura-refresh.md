# Backend refresh after the complete five-game capacity comparison

The fifteen 12M learners and their frozen controls are complete. Larger capacity
improves Breakout/Pong/Qbert rewards but not every game or task milestone, at
3.56x the 1M training wall time. All five quality targets remain open. Before
another learning allocation, qualify current upstream on unchanged CDP/RGB
graphs and measure one old/new full-update pair. No learning extension follows
automatically.

## Exact candidate and scope

Rechecked October 9 after the allocation: Meganeura
`3102683311f3376026928364928a4aa966102311`, Blade
`56f05651cca73659130e924e497dda633f839d40` (also Meganeura's pinned dependency).
The prior qualified control is f104f354/e349cddf, native83be73bf. Preserve its
binary/native and all completed learning identities. Neighboring user worktrees
are untouched; update only Kindle's dependency pins, reported revision constants
and generated lockfiles.

Blade reuses Vulkan descriptor sets across command-buffer recordings. Meganeura
aligns fused normalization rounding, fixes logical-input padding, changes tuning
and adds native-f32 alternatives, largely for AMD. The final additional commit
preserves native16 f32 split-reduction boundaries across staging widths. These
are relevant correctness/overhead changes, not measured NVIDIA gains or an
identified explanation for an earlier learning failure. No graph or precision
policy changes; the rejected projection-reuse candidate stays rejected.

## Bounded qualification and timing

Reuse the October 7 refresh's exact 18 checks and tolerances: grouped RSSM
outputs/gradients and independent F64 forward, four exploration checks,
raw/centered cosine values/derivatives, 1,300 CDP and 1,524 RGB upstream
comparisons including raw gradients and optimizer/EMA. Three production/frozen
smokes cover 1M raw CDP Pong, 12M centered CDP Breakout and 1M RGB Seaquest:
1,024 training actions/195 updates then 1,024 frozen actions/zero updates each.
Audit actual initial weights, counters, finite checkpoints, exact saved tensors,
native identity and >=2 GiB sampled Vulkan estimated budget headroom.
These checks do not erase the separate stronger full-head F64 failure.

After successful qualification review, one timing per backend, using preserved
f104 control and freshly built candidate test binaries. Restore the exact same
346 tensors/config from Breakout1009's completed 12M checkpoint, fixture RNG701,
16 warmup plus256 measured ordinary full updates. No compilation, heavy CPU
analysis or per-dispatch timestamp instrumentation during timing. Every first
output requires max absolute difference<=1e-5 OR relative L2<=2e-4, unchanged
from the prior declaration. Retain every update and finite final tensor; later
stochastic trajectories need not be bitwise identical. Report measured change
without claiming whole-game throughput, SM utilization or a universal speedup.
Review >5% slowdown before selecting a new learning allocation. Do not sweep
configurations or relax numerical gates.

GPU tests/timings120s each; smokes300s; qualification service1h, timing service10m.
Serial host guards, Restart=no, KillMode=control-group, RTX5080/580.178.04,
boot3e89d55c-a9e5-472f-a18a-06508c5bafa7. Standalone allocation warnings are
recorded/nonblocking; other API/numerical/hard faults and deadlines stop the
affected queue for review. No blind retries, NVML polling or recovery.
CPU build/preparation/audit: one CPU,2 GiB,zero swap; build deadline1h. GPU workers
retain ordinary host allocation. No GPU work is launched by the build service.

Artifacts: `runs/meganeura-refresh-20261009.EZAbGTI2`. Preserve source hashes,
upstream review, binaries, declarations, logs, every failure and prior controls.
This is backend qualification, not evidence of five-game competence or RGB
compute savings. Choose the next bounded learning study only after review.
