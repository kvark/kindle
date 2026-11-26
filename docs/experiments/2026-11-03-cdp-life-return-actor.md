# Optional life-bounded completed-return actor: native and cost qualification

The [fixed return/life audit](../results/2026-11-03-cdp-breakout-return-life-audit.md)
finds that later lives supply70.74% of positive actor weight on480 retained
states. Select a default-off auxiliary return-boundary prototype, then native
qualification and one bounded execution check. This tests whether such credit
can be represented correctly and cheaply; no full learner is selected.

Add `replay_actor_auxiliary_boundaries=false`, omitted from default checkpoints.
Require positive completed-return actor scale and explicit observed lower-bound
mode with the retained positive disagreement bonus. The generic vector observe
API accepts an optional boolean per arrival, valid only with this configuration.
Validate it before native state mutation. Ordinary observe remains compatible.

An auxiliary boundary closes the discounted observed return for preceding
outgoing actions, inclusively using the reward on its arrival. It does not
change rewards, game flags, recurrent state, episode/reset counters, scheduling,
sampling, critic targets or model continuation. The boundary arrival itself
remains pending until its next segment completes. Later completion must fill
that state's outgoing target but must not overwrite preceding segment targets.
True terminal and timeout semantics remain authoritative; a timeout must not
finalize an unfinished segment. Preserve interleaved streams, retained suffixes,
ring eviction and context-prefix slicing. Require nonnegative observed rewards
for this lower-bound mode and reject incompatible calls before mutation.

The optional fresh-training CLI `--breakout-return-life-boundaries` observes
life-count decreases at each raw emulator frame, before four-frame pooling.
Gains alone do not cut returns; simultaneous loss/gain remains visible. A small
adapter carries the boolean and loss/gain counts to observe and the transition
log. No life count enters the actor/world observation. Preserve raw reward and
actual game termination. Defaults and frozen evaluation omit the adapter; frozen
restores retain configuration but performzero learning. Check a specific native
capability on fresh selection and restore, and audit declared log fields.

The actor objective remains `clip((G_segment−V_combined)/scale,0,1)` weighted CE
on executed outgoing actions, with detached inputs/targets and coefficient100.
There are no new parameters or graph arithmetic. This is conservative positive
reinforcement, not a per-life advantage estimate or unbiased policy gradient.
The game-level critic remains deliberately unchanged. Removing later-life
credit could discard useful signals; gameplay benefit is unmeasured.

Before GPU work, preserve prior source/runtime evidence and pass Rust CPU replay,
configuration and API validation checks, Python adapter/CLI/auditor checks and
strict Clippy. Package an isolated runtime with exact source/wheel/native hashes.
Run separately guarded F64 actor-loss/derivative/mask cases with the new config.
Check legacy frozen compatibility against the retained semantic prefix and all
357 tensors without repeating any prior stage.

Then select one fresh8192-action/489-update check with the retained
12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1,
encoder4e-5/dynamics4e-4/behavior8e-5,policy1/critic1/1,expert100 with world
gradient,disagreement1,actor100/lower-bound and auxiliary boundaries enabled.
Keep seed1009,full18/sticky.25/repeat4/noops0,native pooled RGB/one GPU64 resize,
sampled collection/posteriors,teacher20%/cap1920 and the remaining retained
settings. No prior weights, replay, games or experience transfer.

Cap training at180s; then1024 frozen actions at120s with zero updates and357
invariant tensors. Check357 initial tensors against the prior candidate,
47 pre-update semantic events before learning, first update376, finite reports,
positive segmented targets and actual raw-life boundary accounting. Preserve
all pixels/actions/teacher labels/life changes through an independent CPU replay.
Keep host guards and report actual elapsed time, including CPU observers.
Compare cost with the completed whole-game bounded candidate50.754s, without
rerunning it. Require a padded200k projection below30min before considering a
separately declared full learner. Bounded frozen actions are not competence.

Root `/x/Temp/kindle-life-return-actor-20261103`. No automatic full learner,
extension, coefficient sweep, confirmation or successful-stage repeat. Preserve
failures and original identities; check live state before each GPU allocation.
The retained70.750/143.667 result and original864/90%-two-wall gate remain.
