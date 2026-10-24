# Word staging reduces the complete RGB/JAX gap to 1.77×

October 16, 2026. [Family declaration](../experiments/2026-10-16-convolution-word-staging.md)
· [Full-update declaration](../experiments/2026-10-16-word-staging-update.md)
· [Compact data](2026-10-16-word-staging.json).

**A counter-directed, type-aware staging change improves complete updates with
exact numerical trajectories.** It changes neither model capacity nor arithmetic
precision. Keep this as an isolated candidate; production and Atari stay unchanged.

| Complete B32/T16 update | Control | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| RGB12M | 51.763 ms | 48.068 ms | 7.14% |
| Centered CDP12M | 46.929 ms | 45.505 ms | 3.03% |

RGB is **1.766× inclusive JAX27.223ms**, down from the retained1.908×. CDP has
an additional exploration workload and is not this JAX comparison. B8 was not
remeasured. All256 samples per arm remain: RGB candidate first/last32 means
48.067/48.080ms; CDP45.482/45.523ms. These are synthetic complete updates, not
learning-quality or game-throughput results.

## What changed and what the counters establish

One shared implementation replaces eight separately addressed scalar slots per
lane with two typed four-element packets. Each packet shares reduction-address
work. Shared row pitches40 for BF16/F16 and36 for F32 rotate row starts while
preserving16-byte row alignment. Current32x32x32 tiles,128 threads, barriers,
K order, dW splits/compensation, rounding and storage interfaces stay unchanged.
All nine scalar-staged occurrences change; the14 packed occurrences do not.
This is neither asynchronous copying nor a global-vector-load claim.

All23 complete convolution operations total **22.493→19.112ms,15.03% less**;
queued wall improves15.00%. Packed operations remain5.032→5.030ms. One input
gradient (`1-32-3-64`) regresses0.666→0.747ms; it remains included, without a
favorable selector. Standalone savings are not added to full-update savings.

| CNN1 derivative | Ordinary complete operation | Warp instructions | Shared-bank conflicts |
| --- | ---: | ---: | ---: |
| dX | 2.155→1.119 ms | 1,554→841 million | 122.08→21.54 million |
| dW | 2.025→1.841 ms | 1,208→1,092 million | 157.98→26.87 million |

Tensor instruction counts remain exactly unchanged. dX tensor-pipe activity
rises18.51→35.37% of active-cycle peak while occupancy falls97.70→57.50%:
useful work, not high occupancy alone, is the objective. These are individual
kernel counters, not whole-learner utilization. All20 fresh control instruction
counts match the [retained native capture](2026-10-16-native-counters.md).

The dW result is especially informative: shared conflicts fall83%, but latency
falls only9.1%. Its uniform-pipe instructions fall315.42→0.46M while ordinary
ALU instructions rise507.57→765.88M. The source moves reduction bases from
`lane/32` to `lane/8`; under contiguous32-lane grouping, one shared base becomes
four different bases. This is **consistent with losing warp-uniform addressing**,
not a SASS attribution or a guarantee of Vulkan lane mapping. Addressing,
packet stores and shared pitch changed together; their individual latency
effects are not isolated. Current dX/dW still execute8.63×/8.17× the retained
full-JAX warp instructions. Those ratios are not speedup promises.

Next prove a packet mapping that preserves shared address calculations, including
subgroup mapping and operand layout, before proposing another bounded test.
Do not infer a better tile from occupancy or revive rejected reuse/pipeline
policies. No layout/geometry sweep, compiler feature project or automatic GPU
allocation follows. Host targets and non-convolution costs remain separate;
the full root-cause/parity goal stays open.

## Qualification and retained evidence

- Complete family:2,690,432 independent F64 comparisons,774,861,888 exact paired
  output values across numerical/timing/capture phases, original gates unchanged.
  Actual generated shaders, full plans, outputs and changed-input/restoration pass.
- Full updates: exact272 reports, first targets and initial/first/final292 RGB
  or346 CDP tensors, including retained controls and24-report prefixes. Full
  semantic plans/constants/pipelines/groups match. Four shared shader modules
  change in each workload;174 RGB/186 CDP modules remain identical. The world
  plan has nine/four affected RGB/CDP occurrences; posterior labels alias its
  staged session and must not be double counted.
- CPU:152 typed variants per arm validate with Naga and SPIR-V tools, both with
  default and Blade-aligned emission.123 variants change,29 remain identical.
  5.428M sampled/exhaustive source addresses,5.248M complete cursor transitions,
  5.244M gradient transitions and10,240 ownership/fragment checks pass. Actual
  large-case addresses are sampled as declared; small ragged/strided cases are
  exhaustive. Clippy passes; CPU type coverage is not GPU qualification of every type.
- 124 passing guards, five allocation warnings, no failed GPU jobs;9,320
  standalone calls/1,360 counter replays and1,088 synthetic learner updates,
  zero game actions.116 counter ranges/4,756 metrics reevaluate offline.
  1,266 family and1,775 full-update source/input seals pass. Minimum sampled
  Vulkan estimated headroom12,380,798,976 bytes is not physical/peak free VRAM.
- GPU work finishes05:53:42UTC; full-update audit finishes05:54:22 in9.93s/1.6GiB.
  No GPU worker remains. Retain the typed-zero constructor failure, the offline
  default-initialization observer mismatch and the two-Blade-path build collision.
  Their narrow CPU repairs change no GPU results; no GPU retry or relaxed gate.
  The first publication-link check wrongly rejected an existing directory link;
  its source/log remain, and the corrected existence check accepts files/directories.

Artifacts: `runs/convolution-word-staging-20261016.ssN5oC6N`, including isolated
sources, complete samples, raw counter images, source seals, journals and audits.
Native SHA256 starts`834241b6`; enable`MEGANEURA_CONVOLUTION_WORDS=1`,
`MEGANEURA_IMAGE_VIEWS=1`, and`KINDLE_DEVICE_MEANS=1`, with cached recording.
The same binary with word staging disabled is the fresh control;`67559ce5`
retains the preceding image-view candidate. No production dependency repin.
