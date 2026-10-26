# Compact Qbert with six action choices

Selected after Pong4013 completes:22/24 frozen wins and mean7.333 in50m14s, with
all audits passing and the service exited. Both compact Pong learners improve
substantially but miss mean15. Retain those results and allocate this bounded
Qbert test next. The controller requires the completed Pong review and the
validated CPU replay change described below before launching.

The compact4M/RGB48/R32 control completes200k actions in19m45s but scores809.375
with0/24 first pyramids. Increasing the native exploration coefficient to100
leaves the score unchanged and reduces mean coverage16.625→13.708 of21 cubes.
Test the existing `published-minimal` action protocol at coefficient1 instead.
This is a hypothesis about learning with fewer action choices, not evidence that
the excluded choices are ineffective or that vocabulary caused either failure.

ALE exposes six Qbert actions: NOOP, FIRE, UP, RIGHT, LEFT and DOWN. The protocol
changes only full-action-space selection; no-op starts remain0, action repeat4,
sticky actions.25 and the episode limit100000 emulator frames. A bounded CPU
check executes4096 aggregate actions in each vocabulary across eight lanes:
observations, rewards, termination, RAM, lives and frame counts match exactly
for the shared six choices. Each arm executes16369 emulator frames and eight
natural episodes. The paired check takes4.741 seconds and constructs no learner.
It does not prove equivalence for every state or demonstrate a training gain.

The selected24 frozen episodes of the coefficient1 and100 policies use choices
outside the minimal vocabulary for56.37% and63.21% of actions, respectively.
These are action-choice frequencies, not ineffective-action rates. Changing
18→6 also changes action-dependent network dimensions, initialization, policy
probabilities and exploration predictions. Same seed does not imply identical
initial weights. Keep this protocol variant separate from full18 reference
results and retain its own actual-initial frozen control.

Keep fresh seed1009/evaluation4000001009, centered CDP, Size4M/RGB48/Auto,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k, actor_unimix0 and native
action-effects coefficient1. Native max-pooled frames receive one GPU resize.
The full native configuration must match the coefficient1 control except
action_count18→6. Record the resulting parameter count at startup. No external
action aid, RAM input, reward shaping, native rebuild or backend optimization.

Fixed200k aggregate actions, approximately800k emulator frames,49,939 expected
updates and first update248. Training cap1800 seconds including construction.
Save100k recovery weights then fixed200k final, replacing the recovery path.
No best-checkpoint selection, automatic resume, cap increase or extension.

Evaluate fixed final and actual initial for the first three natural sampled
episodes per lane,24 per arm, zero updates and truncations. Each arm has
600k-action/300-second caps; retain excess episodes and partial tails. The
existing gate remains at least20 episodes, mean≥15000 and≥90% first-pyramid
completion. Also report score, coverage and task completions as development
outcomes. Reused seed1009 is not a held-out test or a reliability cohort.

Audit every numeric report, source/config/counters, finite checkpoints, all346
unchanged frozen tensors, full CPU replays/videos and host guards. Audit cap1200
seconds; service cap3660 seconds. Stop on real faults or numerical failure.

The training adapter already supports this vocabulary. CPU replay previously
allowed minimal actions only for Breakout. A change adds an explicit
Qbert six-action declaration while preserving exact vocabulary, ROM and protocol
checks.138 CPU tests pass both staged and after application, including real
full/minimal Atari trajectories and rejection of missing, reordered or mismatched
declarations. Applied only after Pong's complete audit and terminal service;
native and training CLI remain unchanged.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-minimal-200k`.
Controller: `run_qbert_minimal_200k.py`; intended service:
`kindle-cheap-qbert-minimal-200k-20261026`. Native source`bfb8c31` and CLI`1ed07af`
remain fixed. The declaration seals exact commands, caps and control identities.
CPU action review: `qbert-action-vocabulary-review.json`, SHA256
`276f2d6483c4b8247975951dfd0bc7e3c240f3220f7fb5ab7a49e59258fe3d54`.
Staged replay patch and validation: `qbert-minimal-replay/` in the same artifact
root. Selection allows this one bounded learner; no automatic next allocation
or extension follows. Record startup checks and the actual parameter count below.
