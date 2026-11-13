# Breakout: bounded recurrent-state energy learner

The [state substitutions](../results/2026-11-07-breakout-world-state-swap.md)
repair newer-model eight-step ball counts8→90/108 with teacher h,versus12/108
with teacher z. The earlier categorical-overshooting learner fails external
tracking gates. Select direct multi-step recurrent-state supervision,without
privileged inference inputs or a capacity increase. The substitution is not a
causal error fraction or proof of a successful training intervention.

Root `/mnt/data/kindle-breakout-world-recurrent-energy-20261107`. Actor/critic,
collectors,backend optimization and CDP efficiency comparison stay parked.

## Objective fixed before results

Add optional `motion_recurrent_energy`,default off,H16. Keep observed motion RGB
cross-entropy,one-step KL1/.1,reward/continuation1,all68 parameters and inference.
Replace the failed extra categorical-suffix KL with two causal prior trajectories
from the same detached posterior h/z at index15. Independently sample their
intermediate prior z using existing .01 mixing/straight-through derivatives.
No current reward/continuation sample or score-function term; no extra decoder.

At each suffix step,let y be the detached recurrent state on the real observed
trajectory. Define d(a,b)=sqrt(mean_components((a−b)²)+1e−6),then
E=.5(d(h1,y)+d(h2,y))−.5d(h1,h2). Use RMS distance so units do not grow with state
width. Average over the declared B×16,including zero-weight rows,and add at the
existing reconstruction scale100. This is a fixed development setting,not a
scalar selected from gradient-noise results. The base losses are unchanged.

The two worlds share their detached prefix state,then have independent future
sampling streams. This scores conditional futures from that internal state,not
an exact integration over all possible prefix beliefs. Target h changes with the
jointly learned model; stopped gradients do not make it a fixed external target.
Intermediate ST gradients remain biased. The objective may collapse or distort
hidden states; image/physical forecasts and state spread must reveal that,not
latent energy alone. No globally unbiased/proper-observable-calibration claim.

Never feed future true images,teacher h or teacher z into a prior transition.
Mask from the first suffix reset onward,including reset arrival; keep genuine
terminal arrivals. No loss crosses a reset. Target/mask/seed gradients are stopped;
added gradients reach only dynamics through recurrent transitions/past samples.
Last-step z need not be sampled because the scored state is h. Report base nine
plus recurrent energy,target distance,pair distance and suffix-valid fraction.
Total adds100×energy;validity uses the H16 suffix,not all32 observed arrivals.

## Qualification and allocation

CPU F64 values/derivatives/finite differences cover smoothed RMS energy,masking
and stopped targets. Test causal ancestry,gradient scope,unchanged parameter
shapes and explicit checkpoint objective. Test Python report/total/mask contracts;
run affected checks,strict Clippy and package exact source/native/runner bytes.
Preserve failures; do not rerun completed old workers/primitive fixtures.

After source/data/executable pins and live-process/host review,declare one new
native value/derivative fixture(180s) and full-shape preflight(900s). Require204
matching initial tensors,18 initial arrays,finite13-report update with energy/
total/validity identities,204 invariant frozen tensors,18 restore arrays and
legacy coverage/latent restore. Source parameters/rates/geometry cannot change.
No assertion of learning from this qualification alone.

After qualification passes,separately pin one fresh8,192-update learner(3600s)
and final restore(600s):8,852,675 parameters,RGB64/12M/B8/T32/Auto,seed41001,
rates4e-4,exact coverage corpora and4/2/1/1 locations. Preserve the H16 counts:
975,941 valid suffix rows,9,570 valid suffix terminal arrivals;other counts match
coverage. This is reused development seed/data,not confirmation. Retain0/2048/
4096/8192 checkpoints,curves,256 fitting/128 development final exports and all
1/4/8/16/32 horizons. Record posterior/prior recurrent-state spread from retained
features,without using its magnitude as a physical competence gate.

After learner/restore audits,separately declare only the new checkpoint's frozen
oracle assessment(1800s):history16 primary/32 secondary,eight fixed draws,84 groups.
Reuse the original oracle/controls; no recollection or old-model assessment rerun.

Development gates:observed ball>=90%,one-step>=89/109,eight-step>=28/107,
primary16/32 ball-feature energy<=.12712666431732942(25% below coverage).
Serving NOOP8 false balls<=10%,ending32 all-four-plan Brier<=.10 and non-ending
group mean ending32<=5%. Add explicit timing checks justified by the last failure:
ending16 all-four-plan Brier<=.01 and RIGHT16 mean ending probability<=.10
(oracle0% at16). Preserve every horizon/category/action/reward/brick regression,
even if an aggregate gate passes. Do not reinstate the invalid FIRE90% pair gate.

These are intermediate retention checks,not full-goal completion. Precise all-
horizon trajectories,event/reward calibration,brick memory,late-game/tunnel/
second-wall coverage,untouched families and fresh confirmation remain required.
No unchanged extension,scalar sweep,automatic seed campaign or successor.
Use existing AMD host guards/Auto;stop on numerical/real GPU faults,no NVML,
reset/reload/reboot or automatic retry. No new environment actions/frames.
