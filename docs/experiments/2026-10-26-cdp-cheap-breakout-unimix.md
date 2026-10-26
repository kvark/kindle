# Test a native policy exploration floor for Breakout

The4M/RGB64 control gains to6.2083 versus initial1.375 in22m30s, but remains weak.
A [saved-action replay](../results/2026-10-26-cdp-cheap-breakout.md) verifies one
selected episode waiting12061 actions after a lost life before choosing FIRE.
Test whether a small uniform mixture in the native actor prevents these long
command gaps and improves useful learning at the same budget.

Change only **actor_unimix:0→0.01**. Native acting, imagined action sampling and
the actor objective already use this field. CLI exposure was checked in an
isolated copy, then applied after Boxing and its audits finished. Commit`bcef410`
passes all255 vector-runner CPU tests. No native rebuild or backend optimization.
Default mixture remains0, and restore uses its saved
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
are sealed before launch. Boxing4013 passed79.1667 versus initial−0.625 with24/24
wins in28m13s; its complete audit was reviewed before starting this test.

Started October26 at00:58:24 UTC. Startup checks pass: all346 initial tensors
equal the0-mixture control, the complete configuration differs only in
actor_unimix, native SHA is unchanged and first updates are248:1,256:2,256:3.
`startup-review.json` retains those checks. Source at launch`bcef410`; native
source`bfb8c31`, SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
The result remains pending; inspect the service before further GPU work.

## Separate frozen sampling control

Prepare one posthoc frozen control to distinguish the effect of the action
mixture during evaluation from training with that mixture. Copy the completed
0-mixture200k final checkpoint and change only saved actor_unimix0→.01. All
346 source tensors and the original checkpoint remain identical; no learning
or warm-start extension occurs. Compare this policy with the original frozen
score and the new mixed-training policy, reporting all three separately.

Use the same frozen seed4000001009, first three natural sampled episodes per
lane,24 selected,600k-action/300-second cap, zero updates and no external aid.
Reuse the primary mixture experiment's actual-initial evaluation only after
verifying identical346 initial tensors and the same saved configuration,
native/runner, environment protocol and evaluation seeds. Audit the new metadata
copy, all unchanged frozen tensors, complete CPU replay/video and host guard;
audit cap600 seconds and service cap960 seconds. Preserve excess/tails/cutoffs.
This control is selected after observing the baseline stall and is not an
independent trained learner or held-out test.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-unimix-acting-only-probe`.
Controller: `run_breakout_acting_probe.py`; intended service:
`kindle-cheap-breakout-unimix-acting-probe-20261026`. Launch only after the active
200k run and its complete frozen audit finish. Review this short control before
starting the separately prepared Freeway test. The primary training comparison
retains its declared budget and fixed final/initial evaluations.
