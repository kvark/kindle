# Playfield crop lowers frozen Breakout scores at similar cost

Retain the full-frame input. The separately selected
[200k-action comparison](../experiments/2026-11-02-cdp-breakout-playfield-crop-200k.md)
finishes successfully but lowers sampled return **33.51%** and greedy return
**20.07%**, for **1.65% more training time**. All arms still have zero of24
two-wall successes. The crop remains optional and default-off; no extension,
confirmation, crop sweep or automatic successor is selected.

| Input | Sampled final / own initial | Greedy final / own initial | Two-wall successes | Training time |
|---|---:|---:|---:|---:|
| Retained full frame |70.750 /1.417|143.667 /2.208|0/24 in each arm|21m27.910s|
| Crop0/32/160/168 |47.042 /1.375|114.833 /1.917|0/24 in each arm|21m49.159s|

Initial-subtracted return per training minute falls **3.230→2.093 sampled**
(−35.20%) and **6.590→5.175 greedy** (−21.47%). Both learners use200,000
aggregate actions /12,477 updates; the crop executes799,600 emulator frames.
No long extension or checkpoint selection follows this result.

## Full comparison and interpretation

The crop changes input pixels at the same RGB64/12M model dimensions. It enlarges
vertical playfield features and removes HUD score/life information. The teacher
continues to receive the original native max-pooled frame. All learning settings
remain fixed: Auto/N8/B16/T16/H15/R16, encoder4e-5, behavior8e-5, dynamics4e-4,
replay-value1 with imagined bootstrap, expert100 and20% episode guidance/cap1920.
The source/native and runner differ from retained447fe50/cf4ea238. Bounded default
compatibility does not establish full default learning equivalence or isolate
which input difference matters. Reused seed1009/cohort4000001009 is development
evidence, not independent confirmation or a general conclusion about cropping.

All357 initial tensors match both the same-preset control and bounded crop check.
The47 pre-update transitions match the bounded crop run. Actual-initial frozen
traces differ from full-frame control as permitted: sampled798 versus800 events,
greedy744 versus795. Own initial scores are therefore retained separately.

Final50k replay expert cross entropy improves slightly **.560514→.543409**, while
both frozen scores decline. Actual guidance is83,817/200,000 (**41.9085%**),
versus86,117/200,000 for the control. The guided online last50-episode mean80.18
versus67.78 is not unaided competence. Neither lower replay loss nor higher guided
return justifies choosing the crop.

The saved-log analysis reuses the completed full-frame control without recounting
its trajectories. Candidate counts reconcile across283,824 actions /1,134,745
frames, including training and four frozen captures. For pre-action scores216–431,
all training actions fall5,947→5,560 and native actions fall **1,262→1,057**
(.631%→.5285% of all training). Neither learner has training exposure at432+.
The crop's selected sampled episodes reach at most94 points; greedy reaches364,
with2/24 episodes reaching216 and none432. These score fractions are not verified
wall boundaries, equivalent states or causal coverage evidence.

## Full execution and retained artifacts

All **413 full pins**, exact crop configuration/architecture, finite reports and
checkpoints, changed CNN/reward/actor/value,100k/6,227 retained recovery, full
training pixel/action/label replay, four frozen outcome replays/videos, host
guards, curves, exposure and comparison pass. Each frozen arm preserves357
tensors with zero updates and truncations. The four captures total83,824 actions /
335,145 frames /82.821s. Ordinary frozen logs lack live pixel hashes; their replay
cannot prove unavailable live-image equality.

All five original services succeed. Main PID950033/invocation
`7bc8fe9931bd4f3fbee1f43739efb056`, launched12:26:48.481 UTC/source53a75d7,
finishes **12:52:52.604 UTC November2**. Postprocessing and collection finish
**12:53:23.251 /12:53:27.613**. The100k recovery and its separate confirmation
are already complete; no observer, learner, GPU stage or successful replay repeats.
No full-run preparation or execution failure occurs. The earlier bounded build
and packaging failures remain preserved in their separate qualification evidence.

The prepared terminal reader completes once **12:54:38.573** in2.282s
(2.305s wrapper). The prepared coverage reader completes once **12:55:05.372**
in3.053s (3.087s wrapper), with54 diagnostic/413 full pins unchanged. CPU co-host
times remain recorded without training-time correction. All services are terminal;
no Kindle GPU allocation remains active at closure.

Full root: `/x/Temp/kindle-playfield-crop-20261102/full`.
Trial: `breakout-12m-64-b16-r16-playfieldcrop-expert100-200k-seed1009`.
Package-root `full-terminal-summary.json` and `score-coverage-review.json` are
complete. Do not rerun the recovery, terminal or coverage readers, collectors,
training, assessments or replays. The five-game goal remains unfinished.

| Full artifact | SHA256 |
|---|---|
| Learning review |`21a086def8e81b4fcaa5132c123ee910a6f77598c7cc59fe9be17ef701702d22`|
| Completion |`701b92466aa9fbf4b3c5d39c22b41994e206e6a1c460197e6a6e015e2aaedfa7`|
| Comparison |`b147e227145827ea1aaf76cb10b1531294338d0eddac6fa6c8dd7c7bce248f56`|
| Terminal summary |`26d91aa162fb2359e029e1bb808ba3c40b782effd5f633d949e446bebbeb0bbd`|
| Coverage review |`92d30ffb3d845661a8164fdfbedfd57460dad34ef813e9d36afad56fb408efb1`|

## Preceding bounded qualification

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

## Retained decision

The earlier bounded check established execution and early cost. The subsequent
full comparison above lowers both frozen scores at slightly higher training cost.
Keep the full-frame learner and crop default-off. No retry, extension, cap increase,
crop sweep, confirmation or automatic successor is selected. Check live state
before new work; Breakout's original two-wall gate and the five-game goal remain
unfinished.
