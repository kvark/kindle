# Phase 2 external interruption and fresh continuation

[Compact evidence](2026-09-28-learning-interruption.json).

The user identifies the September 28 interruption as an **unplanned external
accident, not caused by this work**. The host rebooted; the transient controller
and its worker are gone. This is not classified as a GPU fault. No reset,
driver change, reboot, NVML query or host recovery was performed by the agent.

- **Eight completed runs are preserved.** Post-reboot trajectory, checkpoint,
  hash and guard audits reproduce their published results unchanged.
- Large Seaquest seed1009 has a last durable transition at **136,722 actions /
  33,829 updates**, 10:35:40 UTC. Its logs have zero-filled tails, no terminal
  guard result and no final checkpoint. Preserve them; do not count this as a
  completed seed or splice it into another run.
- The last memory sample has **2.56 GiB estimated Vulkan headroom**. Retained
  logs contain no new kernel fault, but cannot establish the cause, precise
  stop time or GPU health. The initial all-history host capture exceeded its
  output limit; the separate bounded relevant-window capture passes its audit.
- The fresh declaration uses boot `3e89d55c-a9e5-472f-a18a-06508c5bafa7`, the
  same driver, native package, upstream inputs, configurations and budgets.
  Restart the interrupted seed from scratch, then the original 36 unstarted
  entries. New output directories; serialized workers; stop on failure.

The accepted matrix still contains **45 × 200,004 actions**. Actual cost is
higher: this discarded attempt adds at least **136,722 actions** and its GPU
time. No learned state transfers from it into the replacement run.

Meganeura upstream was checked again: `7c29497` adds caller-owned submission
APIs; it does not identify a new correctness fix for the existing step path.
Keep this comparison's package unchanged. Evaluate the new API separately
after the matrix. Blade upstream remains `fbb4f28c`.

Local evidence: [original attempt](../../runs/representation-learning-20260927.bTE8iK/learning-remainder/large-seaquest-seed1009),
[host captures](../../runs/phase2-interruption-20260928.KE04qR),
[fresh declaration](../../runs/representation-learning-20260928.kjidlR/learning.json).
Execution continues under `kindle-phase2-learning-after-reboot-20260928.service`;
[PR31](https://github.com/kvark/kindle/pull/31) remains the live dashboard.
