# Native playfield crop passes geometry, compatibility and bounded training

The optional Breakout crop completes **8,192 actions / 32,725 emulator frames /
489 updates in 47.995 seconds**. Frozen restore completes **1,024 actions /
4,096 frames in 0.882 seconds**, with zero updates and no natural completed
episodes. The conservative 200k-action projection is **21m28.933s**, below the
30-minute screen. This establishes execution and early cost, not gameplay
benefit, a speed improvement or a selected full learner.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-playfield-crop.md)
follows the negative [25M capacity comparison](2026-11-02-cdp-breakout-capacity25m.md).
The goal remains useful frozen learning per GPU-hour on the five games;
Breakout's two-wall gate is still unmet.

## Change and limits

`--image-crop 0 32 160 168` selects native rows32–199 and the full160-pixel
width before the existing single antialiased GPU resize. Output stays RGB64;
the retained12M preset still has16,334,353 parameters. There is no extra resize,
CPU pixel transform, teacher change or added network head. Python supplies the
original native max-pooled frame and the teacher continues to label that frame.

Vertical input scale increases from64/210 to64/168; horizontal scale stays
64/160. The inspected retained video frame contains roof edge, bricks, ball and
paddle within the crop. The removed HUD contains score/life information, and one
frame does not establish that all useful information is preserved across play.
The experiment changes input allocation, not a demonstrated causal bottleneck.

Keep Auto/N8/B16/T16, full BPTT/microbatch16, H15/R16, encoder4e-5,
behavior8e-5, dynamics4e-4, replay-value1 with imagined bootstrap, expert100,
unit label weights, world gradients and all-state corrective labels. Guidance
selects20% of episodes with cap1920; actor unimix.01, action-effects1,
replay100k, warmup1000, full18/sticky.25/repeat4/noops0 and seed1009 stay fixed.
No prior weights, replay, beliefs, games or RNG state transfer into the check.
The native implementation and runner differ from the retained447fe50 control;
default compatibility is established below, without an isolated causal claim.

Crop geometry is optional checkpoint configuration, absent by default. Cropped
checkpoints use architecture tag `dreamerv3-visual-features-rgb-crop-v1`, so an
older native loader cannot silently ignore the crop. Restore uses the saved
rectangle and rejects fresh overrides. Empty, overflowing and out-of-frame
rectangles, non-RGB input and fixed native color-centroid supervision combinations
are rejected. Original strides, pixel formats and frame uploads are preserved.

## Qualification and preserved failures

Native source **acb175f** passes **153 Rust CPU tests**, **642 packaged Python
CPU tests**, workspace strict Clippy and binding strict Clippy. The isolated
wheel is under `/x/Temp/kindle-playfield-crop-20261102/repaired/wheels`; runtime
is `repaired/runtime`. Native SHA256:

```text
a97bca1b417f192d78268f62a41f9aadaa81bf56fea436b57a8a1c654dfb68a1
```

The original c275c1e compile fails before tests because an existing gradient
diagnostic fixture still uses the old three-argument RGB constructor. Its
24.924s receipt, compiler log and exact source remain retained. The correction
explicitly passes `None`, preserving that fixture's full-frame behavior.

The separate build completes all five compile/test stages but its final binary
lookup still points inside `repaired/build` instead of the shared parent cache.
The original successful stage receipts and missing final receipt remain intact;
the process exit status was unavailable after handover. A separate packaging
repair copies the already tested executable from its verified actual path.
No completed compile/test stage or GPU work repeats for either correction.
The completed packaging receipt is `repaired/packaging-repair/build-receipt.json`.

The guarded GPU geometry test passes all32/48/64 outputs, four crop rectangles
(selected Atari, odd rectangle, full frame and single edge pixel), CPU uploads
and resident RGB/RGBA/BGRA paths. It checks an independent scalar reference,
odd offsets, padded strides and inactive-stream clearing. The controller takes
2.805s and performs no game actions or learner updates; all seven pins and host
guard pass. Review SHA256 prefix `7e27a811`.

The new native default full-frame restore matches **128 semantic events** from
the retained capture and preserves **357 checkpoint tensors**. It executes
1,024 actions /4,096 frames in0.894s with zero updates. All29 pins and the host
guard pass. Only the new native arm runs; the old runtime and trajectory are
reused as recorded evidence. Review SHA256 prefix `831114e8`.
The new configuration explicitly serializes the existing `imagined` replay-value
bootstrap default; `observation_crop` remains absent in this compatibility arm.

## Bounded training evidence

The fresh check has a120-second training cap and60-second frozen cap. Its actual
commands pass CPU parsing/configuration checks before launch, including ten
positive/rejection config cases,357 expected shapes and exact teacher identities.
All **374 declaration pins** remain unchanged. The complete audit passes:

- Correct cadence and489 finite learner reports; CNN, reward, actor and value
  weights change. The parameter count and complete crop configuration match.
- All357 fresh initial tensors match the retained same-preset initialization;
  all357 frozen tensors stay invariant. Crop inputs intentionally differ, so
  initial semantic equality to full-frame control is neither required nor claimed.
- Full native-input/action/label outcome replay and teacher identity checks pass.
  Actual overrides are3,618/8,192 (**44.165%**); guided play is not unaided skill.
- Original training/frozen host guards pass; neither GPU stage repeats.

Main PID949416/invocation `64cb109da5ca4d25bd114cf97948f54a` succeeds
**12:13:29.808 UTC November2**. Original collection PID949419/invocation
`7675b14a99ad4cb2af405ebde5e913a1` succeeds **12:13:34.786**. Terminal
confirmation completes **12:14:29.788** in1.532s. Reader preparation takes.002s
while the learner's service is active; no time correction is applied.
All services are terminal; no Kindle GPU allocation remains active at closure.

Evidence root: `/x/Temp/kindle-playfield-crop-20261102`.
Trial: `native-check/breakout-playfield-crop-check`.

| Artifact | SHA256 |
|---|---|
| Bounded completion | `73213f266526a4ba7e3dcbdc2f03016fdba258571a3c0cb2ad8023f76d7ec006` |
| Bounded audit | `efbf828b469e81ff08afff63af4bb927e05ccf0fa03cd9296bab82c7a4bfb7fa` |
| Terminal summary | `3b02bd7eda7e9131328ddef16b2714fc978b3d86415520f857d691a725cdc932` |
| Packaged native receipt | `65a54cef388d048aa76340d3c245023aa88fc6ec8b87c8d18ba790f8f901a787` |

Released 2,962,861,213 bytes of disposable build products after qualification;
retained wheels, isolated runtime, test executables and evidence. One 248,640,312-byte
original test executable remains at its pinned cache path. The separate cache
release receipt records this; the historical packaging receipt is unchanged.

## Decision

The optional crop is qualified for execution and early cost. Keep it default-off
and retain the current full-frame learner. No full comparison, crop sweep,
extension, cap increase or automatic successor is selected. Do not repeat the
completed geometry, compatibility, bounded training, replay or collectors.
A separately selected full comparison would need own actual-initial controls
and24 natural episodes per sampled/greedy arm against the retained scores,
with the original two-wall criterion. The five-game goal remains unfinished.
