# Test a native policy exploration floor for Breakout

The4M/RGB64 control gains to6.2083 versus initial1.375 in22m30s, but remains weak.
A [saved-action replay](../results/2026-10-26-cdp-cheap-breakout.md) verifies one
selected episode waiting12061 actions after a lost life before choosing FIRE.
Test whether a small uniform mixture in the native actor prevents these long
command gaps and improves useful learning at the same budget.

Change only **actor_unimix:0→0.01**. Native acting, imagined action sampling and
the actor objective already use this field. The CLI exposure is prepared in an
isolated copy while Boxing runs; apply it only after that service and its audits
are terminal, so the active runner's pinned bytes remain stable. No native rebuild
or backend optimization. Default mixture remains0, and restore uses its saved
policy configuration rather than accepting CLI overrides.

For18 unmasked actions,1% uniform mixture assigns the nine FIRE variants at least
0.5% combined probability per sampled decision. This reduces the probability of
long command gaps; it does not impose a maximum gap or guarantee reward improvement or executed FIRE on
every chosen command under sticky actions. The mixture remains part of the policy
during frozen sampling. No external override, forced FIRE, life-count/RAM input
or new reward enters training.

Keep fresh seed1009, Size4M/RGB64/Auto,4,131,729 parameters, centered CDP/
action-effects1, N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k.
Full18/sticky.25/repeat4, native max-pooled frames and one GPU resize. Compare
all346 actual initial tensors against the0-mixture control; this is a fresh life,
not a checkpoint continuation. Audit the full config difference and runner hash.

Fixed200k aggregate actions, expected49939 updates, first update248, zero debt.
Training cap1800 seconds including construction. Save actual initial and100k
recovery weights, replacing the latter with fixed200k final. No selected checkpoint,
cap increase or automatic extension. Frozen seed4000001009, first three natural
sampled episodes per lane,24 selected per final/initial arm, zero updates;
600k-action/300-second caps per arm. Retain all excess episodes, tails and cutoffs.
The existing90%-two-wall gate remains; fewer stalls alone is not task success.

Check every numeric report, complete config/source/counters, finite checkpoints,
all346 unchanged frozen tensors and complete CPU replays/videos. Audit cap1200
seconds; service cap3660 seconds. Report frozen scores, episode lengths and
chosen-FIRE gaps separately. CPU life-loss diagnostics may inspect saved actions
afterward; their state never reaches the policy. Seed1009 is reused development
data, and a promising result still needs seed confirmation.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-unimix001-200k`.
Controller: `run_breakout_unimix_200k.py`; intended service owner:
`kindle-cheap-breakout-unimix001-200k-20261026`. Exact commands, caps and hashes
are recorded before launch. This test is prepared, not automatically queued;
start only after reviewing the active Boxing4013 confirmation.
