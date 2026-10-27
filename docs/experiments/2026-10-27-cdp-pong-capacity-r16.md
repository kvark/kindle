# Pong12M/RGB48/B16/R16 capacity comparison

The bounded check completes successfully; the separately reviewed fresh500k
allocation below is active from01:08:02 UTC on October27.

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
constructed during preparation. The bounded allocation is terminal; its result and subsequent decision follow.


## Bounded result

16,384 aggregate actions /65,523 emulator frames /1,001 updates complete in
**78.872s**, versus94.054s for the compact4M/B8 prefix,16.14% less training time.
Construction2.171s is separate. First376, residual credit.5,14,123,473 parameters,
all finite reports/config/counters, changed CNN/reward weights and both guards
pass. All346 actual initial tensors match the existing12M48 initial despite the
new native binary; no equality across model sizes is claimed. Ordinary1024-action
frozen restore takes.708s with zero updates and346 unchanged tensors.
Service starts01:02:54 and exits successfully01:04:22 UTC on October27,
1m28.029s total. No GPU retry or audit repair. Review SHA256:
`b56204eb1e0d6233546e1d2832aba8b8ca37a526f3a60bffcfacbde2f2845b5f`.

## Selected fresh500k comparison

The bounded result fits the intended cost range, so select one fresh500k-action
life on1009, with the same recipe and **45-minute training cap**, including
construction. A linear projection from the short prefix is40m07s; this is an
estimate, not measured full-run throughput or learning quality. Training expects
31,227 updates, first376, residual credit.5 and initial six updates at376,392,
408,424,440,456. Recovery saves every100k:6227/12477/18727/24977 updates before
final31227. No smoke checkpoint, replay, belief, environment or RNG is reused.

Fixed final and own actual initial each assess the first three natural matches
per eight streams, with eval4000001009, sampled actions and no aid. Keep the
original mean15/90%-wins gate; zero updates/truncations and346 unchanged tensors
per frozen arm. Each arm has600k-action/300-second caps. Complete replay checks
pixels, actions, rewards, frames and episode boundaries, with videos. Training
must pass all finite reports, exact configuration/source/counters and guards;
startup compares all346 initial tensors and the prelearning trajectory against
the bounded check. All four recoveries must be audited before replacement.

The complete comparison retains4M/B8/R32 and4M/B16/R32 fixed500k controls,
which score14.958 and11.875 in49m55s and38m45s. Capacity, batch and ratio change
together versus B8; native/runner and control kernel differences remain explicit.
No initial-tensor equality across capacities or isolated causal effect is claimed.
No full-run score is known yet. A promising result requires separate seed
confirmation before reliability claims. No extension, cap increase, checkpoint
selection or automatic successor follows this allocation.

Artifacts:`/x/Temp/kindle-pong-capacity-20261027/pong-12m-48-b16-r16-intrinsic001-500k`.
Controller`run_full.py`, startup`check_startup.py`, recovery`watch_recovery.py`,
comparison`compare_capacity.py`, completion`finish_full.py` in the same parent.
Owner`kindle-pong-12m48-b16-r16-500k-20261027`; service cap4620s, including
training, evaluations and1200s CPU audit cap. The declaration pins42 files.
Both existing controls pass the comparison reader and all three commands parse
before selection; preflight passes without constructing an agent.


## Active startup

The fresh service starts01:08:02 UTC on October27 from source
`002e403b553d791808c1406f6aabc8d2931d66d1`, PID66140, invocation
`ee2a9c4847594b40baf4aa99d3e034a3`. Startup passes at3480 complete actions and
194 finite reports: exact config/source, zero starting counters/no restore,
14,123,473 parameters, all346 initial tensors matching the bounded check,
all47 prelearning transitions identical, and first six updates at the declared
376/392/408/424/440/456 actions. All42 declared input pins remain unchanged.
Progress,100k-recovery and terminal CPU postprocessing observers are attached.
This is startup evidence; no full-run quality or completion claim.
Startup receipt SHA256:`077ba1494d6fb928464c8cbc4cecce7b0338086e34f4178f51e819c79d514e2e`.
Launch declaration SHA256:`11f2eef80e359cf510e07ef49cd47cdbc07301b2c4e4c013953da5cca6fbf911`.
