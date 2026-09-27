# Phase 1a: share direct learner parameters

**Adopted:** compatible direct weights now share Meganeura GPU allocations
between training and inference sessions. The slow critic stays independent.
Inputs to fused/derived parameter images remain private and use the existing
refresh path, batched across target sessions. This uses existing upstream APIs;
no backend fork, precision, sampling, objective or optimizer change.

## Fixed-recipe timing

One control/candidate pair on RTX 5080 / driver 580.178.04, Meganeura 367e53d4 /
Blade 7cca6377. F32 / 12M / B16 / T64 / full BPTT / M16 / H15, prediction-only
objective, seed 0, identical synthetic replay. Run 36 updates; exclude the first
four from timing only. No tracing or concurrent compilation.

| Mean stage time | Control | Shared parameters |
| --- | ---: | ---: |
| Complete update | 227.03 ms | 212.78 ms |
| World synchronization | 13.84 ms | 0.069 ms |
| Behavior synchronization, including host EMA | 2.81 ms | 2.98 ms |
| Posterior recurrence | 33.34 ms | 33.33 ms |
| Imagination/targets | 84.63 ms | 84.21 ms |

Complete update time falls **6.28%** (1.067x throughput). Medians are 227.42 and
212.84 ms. All 36 non-timing update reports, scalar checkpoint metadata and all
**241 tensors / 146 optimizer moments match exactly**, with finite values.
[Self-contained configuration, timings and per-update curves](2026-09-27-shared-parameters.json).

This is one short synthetic comparison, not sustained Atari actions/s, a
multi-seed learning experiment or achievement of the 3x target. Behavior sync
did not improve; host EMA remains. Posterior and imagination are now the main
host-round-trip targets. Do not add this gain to earlier unrelated percentages.

## Integration

- The new native sharing test matches an independent unshared session over
  three parameter initializations; it also checks that derived images and
  their inputs are not shared. It runs in the existing Linux GPU CI group.
- Six-stream native-input Pong: **1,536 actions / 34 updates per arm**, trained
  Tiny encoder, seed 7301, F32/12M/B16/T64/R256, no assistance. All actions,
  rewards, boundaries, non-timing learner reports, checkpoint scalars and
  **241 tensors / 146 moments match exactly**. Input shape is 210x160x3.
  Minimum sampled Vulkan estimated headroom is 13,386,317,824 bytes (control)
  and 13,419,937,792 bytes (candidate). This is plumbing, not Pong competence
  or a whole-agent throughput measurement.
- Native vector/serial learning, GPU replay eviction/context refresh and
  checkpoint restore parity pass.
- 98 Rust CPU tests, 846 Python tests, formatting and release Clippy pass.

Both Pong arms use Tiny `7fe9b252`, pretrained on 250,000 random-play RGB64
observations from Boxing, Pong, Freeway, Breakout and Qbert (45k training + 5k
validation per game). This is additional same-title experience, not online-only
learning; it is unchanged between arms. The synthetic timing has no encoder.

Each GPU invocation passes the host guard and reaps its direct child. No NVML
polling, new kernel fault or recovery. Raw declarations/logs/checkpoints and
read-only comparison scripts: [local run](../../runs/strategy-sync-20260927.jfYyDI/).
The initial reader incorrectly required equal serialized safetensors hashes:
parsed headers and every tensor byte are equal despite different container
bytes. The corrected reader verifies each file's own hash before comparing
metadata and tensors; no numerical tolerance was loosened.

This completes the first synchronization step, not Phase 1. Next: resident GPU
posterior/imagination recurrence, then the <=1-hour three-seed small-model
screening workflow before further representation/exploration comparisons.
