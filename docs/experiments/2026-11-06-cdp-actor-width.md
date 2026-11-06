# Optional actor-only hidden-width development

The retained actor's update is dominated by imitation and optimizer history,
and the additional dynamics.5 term barely changes two late-checkpoint updates.
Earlier larger-world and shorter-imagination learners improve replay imitation
fit while lowering frozen quality. These findings do not establish a capacity
bottleneck. Test the narrower hypothesis that actor-only capacity improves the
mapping from the existing recurrent representation to corrective actions at an
acceptable cost. The25M preset changed the encoder, RSSM and critic as well.

Add optional `actor_hidden_units`, absent by default and omitted from legacy
serialization. Every actor owner uses this width: behavior training, live
inference, imagined rollout and frozen world-supervision head. Keep critic,
CNN, RSSM, latent dimensions, losses and optimizer ownership unchanged. Restore
preserves the width and the CLI rejects fresh-width overrides on restore or
evaluation. Old native packages must reject enabled or restored width settings.
Use existing MLP operations; no backend work or new gradient estimator.

The selected development width is512 versus256 in the12M control, adding
1,054,720 trainable actor parameters. This is a single capacity choice, not a
sweep or an isolated claim about all actor capacities. Numerical correctness
and better replay fit alone are insufficient to retain it.

CPU tests must prove all actor owners agree, non-actor shapes remain unchanged,
explicit preset width preserves default graphs, legacy config round-trips,
invalid width rejection and CLI compatibility before any construction. Package
the native wheel and dedicated test executable with exact source hashes. Run
one new guarded Tiny qualification with a doubled actor width: two synthetic
updates, all synchronized actor weights equal, finite policy loss, changed
actor/critic parameters and exact weights/moments after restore. Zero Atari
actions. Existing arithmetic qualification remains reused, not repeated.

After qualification, separately declare an8,192-action/489-update Breakout
cost check and1,024 frozen actions, preserving12M64/Auto/N8/B16/T16/H15/R16,
encoder/base4e-5,dynamics4e-4,behavior8e-5,critic1/1,expert100/world gradients,
sampled collection/posteriors,latent/actor mixing.01,disagreement1,all-state
labels,20% guided episodes/cap1920,full18/sticky.25/repeat4/noops0 and one native
RGB GPU resize. Coordinates and actor dynamics stay off. Verify exact initial
non-actor tensors; preserve own initial actor values without claiming identical
actions or initial scores across widths. Require full replay/frozen invariance,
finite supervised reports,2GiB projected full-replay headroom and conservative
200k training cost at most30min. A full learner requires a separate declaration;
no automatic learner, sweep, unchanged extension or old-stage repeat.

Root `/x/Temp/kindle-actor-width-20261106`. Use existing host guards on an idle
healthy AMD device; stop on real faults or numerical failures, no reset/reboot.
Native Tiny job cap120s/controller180s. Preserve failures and exact artifacts.
The retained143.667/70.750 control and original864/90%-two-wall gate remain fixed;
the five-game goal is unfinished.
