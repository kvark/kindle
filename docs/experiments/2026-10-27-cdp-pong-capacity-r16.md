# Pong12M/RGB48/B16/R16 development check

Selected after the completed Qbert expert500k/R16 comparison and terminal
services. The strongest compact Pong4M48/B8/R32 recipe scores14.958 and14.000
on two learners; both win24/24 but miss mean15. Historical12M64/B8/R32 learners
passed that gate on three seeds on the original host. This supports testing
capacity, without predicting AMD learning or speed from those older results.

Use centered CDP12M/RGB48/Auto/N8/B16/T16/H15/R16, microbatch16, replay100k,
intrinsic coefficient.01, actor mixture0, original rates, disabled actor/critic
gradients and no external aid or expert imitation. Native max-pooled frames
receive one GPU resize. Keep full18/sticky.25/repeat4/noops0 and100000-frame
episode limit. Fresh1009/eval4000001009 are reused development seeds.
Capacity4M→12M, batch8→16 and ratio32→16 change together versus compact Pong;
initialization, optimizer cadence and centering population also differ.
Native/CLI79c9a7f is already qualified; no native rebuild or backend work.

The selected check is **16,384 training actions /1,001 expected updates**,
first376 and residual credit.5, under180s including construction. Ordinary
frozen restore runs1024 actions under90s. Audit under120s; service cap450s.
Require exact config/source/counters, every finite report,14,123,473 parameters,
finite checkpoints, changed CNN/reward weights and all346 frozen tensors
unchanged. Compare the saved actual initial with the existing same-seed12M48
initial, and compare cost with the compact Pong control's16,384-action prefix.
Different native/kernel versions and recipes limit attribution. The short
restore has no natural-match competence gate. No retry, cap increase, full-run
promotion or reuse of smoke experience is automatic.

Artifacts:`/x/Temp/kindle-pong-capacity-20261027/pong-12m-48-b16-r16-check`.
Controller:`/x/Temp/kindle-pong-capacity-20261027/run_check.py`.
Owner:`kindle-pong-12m48-b16-r16-check-20261027`.
The declaration pins24 input files. CPU preparation corrected an unsupported
bonus flag before execution; its parser-only preflight initially omitted the
`Path` symbol, then both commands parsed. No environments or agents were
constructed during preparation. No full learning allocation is selected yet.
