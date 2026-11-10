# Kindle builds against upstream Meganeura main

On November 10 the working branch moved from the preserved Meganeura fork
`423d3736` (handover/cdp-qualified-20261025, Blade `bbbc93b7`) to Meganeura
main `a0eff600` with Blade `d9112e26` and wgpu/naga `99647333`. Upstream had
108 commits past the fork's base; the fork carried about 8,400 lines of
CDP-specific backend work (command cache, image views, convolution reuse,
pooled tapes, qualified cooperative matmuls, diagnostic groups) that main never
absorbed, while main gained `Session::blend_parameters_from`,
`Graph::argmax_inner`/`one_hot`, prepared steps and its own kernel tuning.

## What changed in Kindle

- Only test and diagnostic code needed porting: three fork-only
  `upstream_reference` fixtures (captured convolution groups, inline gaps,
  convolution cuts) are deleted with their plan/command-cache dumps; hard
  categorical samples are recognized by `Op::ArgMaxInner`; block-matmul gradient
  precision is checked through `Node::full_precision`.
- Main has no accessor from a parameter name to its device buffer, so the
  optional actor-dynamics/expected-value parameter snapshots cross the host.
  Both options are retained off. A one-method upstream addition,
  `Session::slot_buffer`, is prepared as a patch and bundle under
  `/x/Temp/kindle-meganeura-slot-buffer-20261110/` (commit `da446a26` on a
  scratch branch, clippy and fmt clean); after it lands, `device_copy.rs` can
  return to a same-queue copy.
- The fork's `MEGANEURA_COMMAND_CACHE`, `MEGANEURA_IMAGE_VIEWS`,
  `MEGANEURA_CONVOLUTION_WORDS`, `MEGANEURA_COPY_COMPOSITION` and
  `MEGANEURA_CONVOLUTION_REUSE` flags are inert. Device selection
  (`MEGANEURA_DEVICE_ID=0x744c`, `VK_DRIVER_FILES`) and
  `KINDLE_EXPECT_DEVICE_NAME` are unchanged.

## Checks

| Check | Result |
| --- | --- |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean after a separate format-only commit |
| `cargo test --workspace --all-targets` (CPU) | 428 lib tests pass, 171 GPU tests ignored; kindle-gym passes |
| Python crate `cargo check`, `maturin develop --release` | pass |
| GPU smoke on the RX 7900 XT (RADV, Mesa 26.0.8) | tiny act/learn cycle, device copies, shared parameters and fused rollouts pass |
| Standalone 1M world learner, B8/T32/H16 `motion_event_rollout`, synthetic batch | 128.1 ms per update after warm-up; 2.52 GB device usage; first update 0.136 s |

The fork's closed early-feature learner spent 1,841 s of native calls over
16,384 updates (112 ms each, including first-update checks and saving) on the
same recipe, so main is comparable to slightly slower on this workload before
any tuning. Checkpoints written by the fork still load; new learners are not
byte-comparable to the fork's qualifiers. Retained runtimes under
`/x/Temp/*/runtime` keep their fork binaries.

The frozen cumulative-survival learner (`kindle-world-survival-loss-learning-20261110.service`,
paused at update 9,964) was neither thawed nor stopped.

## Operational note

Running the CPU test suite bare in the session shell peaked at 17.6 GB and the
kernel OOM killer took the session down. Heavy jobs now run as
`systemd-run --user` units with `MemoryMax`; the suite itself needs about 13 GB
with four test threads because the 12M world-learner graph tests are large.
