# Updated Meganeura qualified; Freeway diagnosis unchanged

Meganeura main `592a2f5a` replaces `13b19d33`; Blade remains `e349cddf`.
This follows the user's standing direction to pick up upstream fixes before
rediscovering them. The five-game matrix stays stopped. No intrinsic arm,
larger training budget or historical result relabeling accompanies this update.

The new main includes contraction-dimension corrections, stricter requested-GPU
selection, cooperative-attention precision/shared-memory fixes and substantial
shader/compiler refactoring. Its attention value-gradient alias fix was already
in our previous qualified pin. CDP/RGB do not use attention, and frozen Tiny
disables cooperative kernels; the new attention fixes do not explain the
[zero-reward Freeway traces](2026-10-04-freeway-zero-reward-diagnosis.md).

## Checks

- **CPU:** 111 Rust tests pass, 55 GPU/fixture tests ignored by the CPU suite;
  1,129 Python tests pass. Workspace/binding formatting and strict Clippy pass.
- **CDP loss:** four cosine values and 1,024 raw derivatives match independent
  F64 references; target gradients remain detached.
- **Upstream components:** four synthetic updates per arm pass 1,300 CDP and
  1,524 RGB comparisons, including raw gradients, optimizer states and EMA.
  Maximum parameter errors are 2.38e-7 and 1.19e-7 respectively. Raw gradients
  use identical pre-step weights; the independent optimizer receives common
  gradients. These are not bitwise stochastic-trajectory comparisons.
- **Integration:** CDP/RGB pixel replay, encoder updates, stream isolation and
  checkpoint restores pass. Frozen Tiny matches independently generated dense
  causal references, including chunk wrap and resets; two-stream batched/serial
  maximum absolute difference is zero. No tolerances were relaxed.
- **Production:** each arm completes a fresh N8/Size1M/B8/T16/H15/R32 smoke of
  1,024 actions/195 updates, with zero learner debt, then 1,024 frozen actions
  with zero updates. All saved tensors are finite and unchanged by evaluation:
  CDP 250, RGB 292, Tiny 241. The 3,072 training and 3,072 evaluation actions are
  excluded diagnostics, not benchmark runs or evidence of competence.
- **Host:** all 14 guarded processes and evidence audits pass; no new allocation
  warning, validation warning, host recovery or NVML polling. Minimum sampled
  Vulkan estimated budget headroom across production smokes is 15,269,101,568
  bytes. This is not physical free/peak VRAM or measured GPU utilization.

Only the dependency pin, both locks and reported provenance constant change in
production code. The initial preparation service exited 127 before compilation;
the replacement supplies an explicit Cargo PATH. A subsequent consistency test
caught the old reported revision; it was updated, not the test weakened. Both
failures are retained. New CPU fixtures were generated because the existing
fixture's source hash predates the current reference script; no pretraining ran.

This qualifies the exercised backend paths, not learning reliability, a speedup,
joint-Tiny training or cooperative attention. The reward-discovery diagnosis
stands; neither the unchanged Freeway matrix nor the diagnostic UP pulse was
repeated. The research-order choice remains open.

[Compact results/configuration](2026-10-04-meganeura-main-qualification.json) ·
[Declaration, logs and CPU audit](../../runs/meganeura-main-20261004.DVmPtD3u) ·
[Per-process GPU evidence](../../runs/meganeura-main-20261004.DVmPtD3u/queue)
