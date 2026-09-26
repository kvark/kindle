# GPU-resident acting and native capture

The single and vector pixel agents now use the same resident path:
GPU capture/raw-byte upload -> preprocessing -> causal Tiny encoder -> pooling ->
RSSM posterior sampling -> live belief -> policy sampling -> selected actions.
Collection copies observation/state into paged GPU replay, without feature or
belief readback. Explicit world-model probes and recording remain opt-in
diagnostics. The learner still reads sampled replay batches and performs host
target/imagination work; this is **not a GPU-only learner** or a measured
end-to-end training speedup.

## Implementation

- Removed DINO encoder, weight loader, bindings and its obsolete GridWorld probe.
  Tiny is the default for both pixel agents; Large requires explicit selection.
  Single-environment acting delegates to the vector implementation. CPU
  feature-facing `DreamerCore` remains a synthetic/reference learner, not a pixel
  fallback. Host hash-visitation collection is refused before GPU construction;
  adapters can still supply intrinsic rewards explicitly.
- GPU categorical sampling preserves independent per-stream RNG draws, masks,
  greedy/sample choices, overrides, resets and causal replay action alignment.
  Invalid logits propagate to an invalid action, not a silent valid command.
- `CaptureStream` imports Dullahan's OPAQUE_FD allocation using its exact size,
  memory type and device/driver UUIDs. Producer release-to-EXTERNAL completes
  before FRAME; consumer acquire/read/release completes before ACK. Generation
  and slot checks prevent stale reuse. A dropped lease sends STOP after release.
  Legacy SHM-ready flags are not GPU synchronization. The handshake is deliberately
  serialized; external semaphore overlap is not implemented.
- Backend revisions remain recorded provenance but no longer block compatible
  checkpoint restore. Current tensor/architecture/encoding checks remain; there
  is no old-encoding migration layer and historical scores are not relabeled.

The small backend additions are pushed separately:
[Blade 7cca6377](https://github.com/kvark/blade/commit/7cca637791a57d9cacf4e99c10a87211f3c11b6a),
[Meganeura 367e53d4](https://github.com/kvark/meganeura/commit/367e53d4de73aea6432afd40aa6fd69b5fcd4e8e)
(dependency-only carry over latest upstream ee3aea42),
[Dullahan bd52f76](https://github.com/kvark/dullahan/commit/bd52f76e5c5775d08b9b042378092dac7767899a),
and [mind-games f5d4405](https://github.com/kvark/mind-games/commit/f5d4405624d256d6317bdafaafb5ce01a2c309c8).
Kindle resolves one shared Blade type from Git, without local path overrides.
Unrelated user changes in Blade/Meganeura were preserved.

## Checks

Final release CPU checks: **98 Rust tests, 846 Python tests**, formatting and
Clippy. Blade's allocation-geometry CPU test and both Dullahan handshake tests
pass. CI now includes GPU vector/mask/replay tests in addition to pixel and
causal encoder references.

[Six-test native suite](../../runs/gpu-resident-suite-20260926.EDQIU3/result.json)
passes on RTX 5080 / driver 580.178.04:

1. Uploaded/resident RGB/RGBA/BGRA against independent pixel references, including
   offsets, strides and aspect ratios.
2. Six independent Tiny streams, resets/gaps/chunk wraps: batched versus serial
   maximum absolute error **0**.
3. Live beliefs, sampled/greedy masked actions and independent RNG streams versus
   serial reference.
4. Overridden actions retained in subsequent belief and replay.
5. GPU replay eviction/context refresh, scheduled learner losses, world parameters
   and checkpoint restore versus serial reference.
6. Two native Vulkan contexts sharing three external slots across twelve
   generations: exact bytes, partial socket reads, reuse and STOP cleanup.

A separate [external-ring run with Khronos synchronization validation enabled](../../runs/gpu-external-validation-20260926.nhIj2y/guard/)
also passes, without reported validation errors or synchronization hazards.
Its only layer messages are two warnings about the deprecated enable-setting
name; future checks can use `VK_LAYER_VALIDATE_SYNC=1`.

The separate [dense Tiny reference](../../runs/gpu-resident-encoder-20260926.Mo5rn1/guard/child.stderr)
passes all 37 comparisons with the existing error bounds, including GPU pooling.
The integration uses the actual independently trained Tiny export `7fe9b252`,
not the synthetic weights used by numerical fixtures.

Early development failures are retained: the first actor test rejected a reserved
WGSL name (`gpu-acting-20260926.n14mns`); the first native-game attempt rejected a
pooling reshape before gameplay (`gpu-vkquake-20260926.W0kIL7`). Both were CPU
validation errors, not driver faults. Shader validation and one/six-stream pool
geometry now have CPU regression tests. Corrected fresh invocations pass.

## Real game boundary

[First native vkQuake run](../../runs/gpu-vkquake-v2-20260926.Q9Tsq1/rollout/result.json):
128 selected actions, full Dreamer 12M, native 640x480, zero updates, zero acting
pixel/feature readbacks. A single explicit diagnostic readback produced this
[visual audit](../../runs/gpu-vkquake-v2-20260926.Q9Tsq1/rollout/snapshot.png): a real
e1m1 game frame with intact scene/HUD. No privileged state entered the policy.
The game, private Xvfb display, config and asset link belong to the run; existing
game configs/saves and the user's desktop were not modified.

This first runner included xdotool's default key-event delays (17.81 s / 128 actions),
not a fair GPU latency measurement. The final runner changes held keys only when
needed, explicitly disables injected delays, and uses the prior-feature objective.
Its final checks both pass:

- [Full 12M frozen](../../runs/gpu-native-final-20260926.R2s0lG/frozen/result.json):
  **256 actions in 2.758 s (~92.8 actions/s)**, zero updates or pixel/feature
  readbacks. Mean observation-to-selected-action 7.87 ms, median 8.04 ms,
  p95 12.52 ms; minimum sampled Vulkan estimated headroom 14,330,822,656 bytes.
- [Small-world learning plumbing](../../runs/gpu-native-final-20260926.R2s0lG/learn/result.json):
  **128 actions, 105 updates in 2.971 s**, with the trained Tiny encoder and prior
  feature-prediction objective. All 105 world/behavior loss pairs are finite.
  An independent safetensors check finds all 241 saved entries finite and changed
  (world 164, behavior 66, slow critic 11), including optimizer state. This uses the
  explicitly small learner preset, **not 12M training throughput**. One
  [diagnostic snapshot](../../runs/gpu-native-final-20260926.R2s0lG/learn/snapshot.png)
  confirms intact pixels; it is outside the acting path.

These short loop times exclude model/game startup; the learning case includes
its final checkpoint save. They are native integration measurements, not
matched before/after benchmarks, sustained rates or competence. Both guards
pass and the owned game/display processes are reaped. No GPU job remains.

There is no Quake reward/terminal adapter or competence claim. The native example
uses mind-games' capture layer/assets; the old structured-state KindleActor and
multi-frame GameSession/Chronolock controller are **not** converted. Port the
per-frame lease contract before using that controller or vector native games.

All native work uses bounded host-only guards, serialized GPU execution, explicit
native-device and sampled Vulkan-budget checks. No separate NVML polling or host
recovery. Successful execution does not explain the historical driver faults.
