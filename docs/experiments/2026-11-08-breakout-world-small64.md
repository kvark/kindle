# Breakout: smaller RGB64 world under a fixed learning budget

The closed [action/transport trial](../results/2026-11-08-breakout-world-action-transport.md)
does not solve rollout drift or events. Earlier local-action H16 improves short
forecasts but still fails retention. Select a distinct capacity/budget recipe:
the existing **1M world at RGB64, with 16,384 fresh updates**. Keeping native
RGB64 separates this from the earlier weak 1M32 whole-agent Breakout recipe.
Smaller models may allow more useful fitting per GPU-hour; this is a hypothesis,
not evidence that excess capacity caused the errors or that 1M is sufficient.

Root `/mnt/data/kindle-breakout-world-small64-20261108`. Reuse the qualified
v18 local-action native/source/runner `7f797a2b`, SHA256
`38680c739929c82c8e1e5a4f15bfa1277ba32170e9c7d14034c2d9c3f75ba323`.
No Rust change, build, backend primitive, collector, object/RAM training labels,
pretrained encoder, actor/critic or default change. The existing three local
motion layers stay width16 with direct command planes. CNN depth16→4, recurrent
state2048→512, categorical classes16→4 and latent feature2560→640 change together.
Keep RGB64/native pooling/one GPU resize, B8/T32/H16/full BPTT32, Auto, seed41001,
all rates4e-4, mixing.01, RGB100, KL1/.1/free-nat1 and event scales1.

Use the exact original 8,192-row plan twice in one fresh learner, without resetting
weights, optimizer age or RNG between cycles. Keep the original16 fitting/eight
development streams. Multiply sealed exposure counts by two except unique source
arrivals; do not recount or recollect. There are1,951,882 valid suffix targets,
19,140 terminal and32,016 positive suffix targets. A thin pinned Python driver
passes this plan/counts to the unchanged native corpus training loop. Python
does no learning or inference. Compare the fixed final with cached H0/local-action
and last action/transport controls; capacity and optimizer budget both change,
so this is not an isolated capacity or matched-compute comparison.

Before allocation, verify data/config/runtime/source identity and CPU driver
controls. Then separately declare one bounded600s host-guarded smaller-shape
preflight: save zero-update initial, record all exact parameter shapes/scalars,
13 finite first-update reports/mask/total, changed encoder/dynamics/decoder/head
families, all own finite/frozen tensors, and18 exact initial/trained restore
arrays. Reuse unchanged primitive/causal/mask/derivative receipts; do not rerun
the old fixtures or old checkpoints. No full learner is allocated at selection.

After reviewed preflight only, separately declare fresh16,384 (3600s) and exact
final restore (600s). Retain initial/2048/4096/final, periodic2048 evaluations,
384 dense actual-initial/final origins, all13 reports and full plan. Final alone
is selected, without best-checkpoint/draw selection. After reviewed learning,
separately declare final-only84 groups (1800s), histories16 primary/32 secondary,
eight draws and1/4/8/16/32 endpoints against the unchanged cached oracle. Use
cached control reports/point summaries rather than rereading/recomputing their
RGB arrays; preserve their closed provenance and all per-action regressions.

Original gates stay: observed ball>=90%; one>=89/109; eight>=28/107; balanced
survival-weighted16/32 ball energy<=.12712666431732942; serve-NOOP false balls
<=10%; ending32 Brier<=.10; ending16<=.01; RIGHT16 risk<=.10; every nonending
category mean risk<=5%; freeflight8 presence>=90% for every command. Retain
every individual alarm and ball/paddle/reward/ending/brick regression. Pixel
accuracy cannot replace event or calibration requirements.

AMD RX7900XT/RADV Auto, live boot/driver/process checks, existing host guard,
stop on real faults/nonfinite state, no reset/reload/recovery/NVML. Report new
environment actions/emulator frames as zero, native/loop/preflight/assessment/
preparation/failure costs separately. Other host GPU work may exist. An observer
timeout is not a stopped job. No extension, sweep or automatic successor.
Precise physics, event calibration, persistent brick memory, late/tunnel/second-
wall coverage, untouched families and fresh confirmation remain required.
Current oracle has no above-top-brick endpoint. The full goal remains unproven.
