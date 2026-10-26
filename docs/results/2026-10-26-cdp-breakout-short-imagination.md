# Breakout H5 warm start regresses after50k new actions

The bounded H5 warm start finishes in **8m02s**, but frozen score falls from
**67.5833 to27.2083**, with0/24 two-wall successes in both policies. All audits
pass. Retain the source and stop this continuation; fast execution does not
demonstrate better learning, and there is no matched H15 training arm to isolate
the effect of the shorter horizon from new experience and replay restart.

The [declared test](../experiments/2026-10-26-cdp-breakout-short-imagination.md)
changes only imagined rollout length15→5 relative to the verified Auto source.
It keeps12M/RGB64/N8/B8/T16/R32,16,334,353 parameters, centered CDP/action-effects1,
actor_unimix0 and all learned weights, optimizer, slow critic and normalizer
state. Replay, games, live belief and RNG restart; this is a warm start rather
than uninterrupted learning or an independent learner seed.

Before training, H5 exactly reproduces the current-host H15 frozen source:
21,976 actions/87,859 frames, the same action/reward/boundary/frame trace and
replay outcomes,67.5833 selected mean, all346 tensors unchanged. Only wall
timestamps are excluded from episode-record identity. The source check and
full CPU replay/video pass before any new learning begins.

Training completes **50,000 new actions /199,878 emulator frames /12,439 new
updates in481.992s**,103.74 actions/s, within the20-minute cap. First update248,
zero debt,89 natural training episodes, final online last50 mean24.34. The25k
recovery checkpoint passes its finite/config/counter check, then fixed50k
replaces it. Starting counters499939/2027152 advance to512378/2077152; the
source export's earlier evaluation history is preserved, not relabelled.

| Frozen policy | Selected episodes | Mean score | Two-wall successes | Full evaluation actions / frames / wall |
|---|---:|---:|---:|---|
| Source |24|67.5833|0/24|21,976 /87,859 /14.234s|
| Fixed50k final |24|27.2083|0/24|22,896 /91,524 /14.751s|

Both evaluations use the first three natural sampled episodes per lane, seed
4000001009. Source collects30 episodes, final37; six/thirteen excess episodes
and partial tails are retained. Zero truncations and zero learner updates;
all346 tensors remain exact per arm. The final all37 log mean26.8108 differs
from the selected24 score27.2083. The existing864-point two-wall gate fails.

All12,439 strict numeric reports, full config/source/counters, finite checkpoints,
full CPU action replays/videos and all three guards pass. Service exits
successfully. Source at launch`bc7c47a`, native`bfb8c31`, runner`1ed07af`;
native SHA256`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-retained-h5-50k`.
Review SHA256`6ec2a86faf37545e89b9755a3334ec44deb964864054e8db4eb24cbfa6bc1b82`.

Independent curves reconcile every episode and action/frame/time totals. The
reader now explicitly supports restored counters while reporting new updates;
its default and fresh-study exclusions remain. Applied after the completed
learning audit, it passes70 CPU checks and reports12,439 curve updates against
the preserved499939 source counter. No training or native code changed.

Do not extend this failed continuation or select a favorable earlier checkpoint.
The original source remains intact. The next focused test checks the compact
Pong recipe on one new seed; Qbert and Breakout still need improved learning.
