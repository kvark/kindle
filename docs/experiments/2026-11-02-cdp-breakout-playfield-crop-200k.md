# Breakout playfield crop: fresh 200k-action comparison

Separately select one full development learner after the
[bounded crop check](../results/2026-11-02-cdp-breakout-playfield-crop.md)
passes geometry, default compatibility, 8,192-action learning/restore and all
374 pins, tensors, guided replay and host guards. Its conservative 200k-action
projection is 21m28.933s, within the existing 30-minute ceiling. No qualification
stage or bounded experience is repeated or transferred.

The larger 25M model was slower and scored worse than the retained 12M control.
This comparison instead tests a different input allocation at the same tensor
and model size: native crop **x=0, y=32, width=160, height=168**, followed by the
existing single RGB64 GPU resize. Vertical playfield scale increases; the HUD's
score/life information is removed. The pixel teacher still receives the original
native pooled image. No additional head, resize or teacher change is introduced.
Better control is a hypothesis, not a consequence established by the cost check.

Keep 12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16, encoder/base4e-5,
behavior8e-5, dynamics4e-4, replay-value1 with imagined bootstrap, expert100,
unit label weights, world gradients, all-state corrective labels, 20% episode
guidance/cap1920, actor-unimix.01, action-effects1, replay100k and warmup1000.
Preserve full18/sticky.25/repeat4/noops0. Seed1009 and evaluation4000001009 with
the eight lane offsets are reused development inputs, not fresh confirmation.
Start without transferred weights, replay, beliefs, games, RNG or experience.

Use the qualified native **acb175f**, SHA256
`a97bca1b417f192d78268f62a41f9aadaa81bf56fea436b57a8a1c654dfb68a1`,
and runner SHA256
`046aa5bc6d2f15250192345c4e77a71f85cadbc9d05ffc46113a74ca2c0ea807`.
Runtime: `/x/Temp/kindle-playfield-crop-20261102/repaired/runtime`.
Native/runner differ from retained447fe50/cf4ea238; the completed default frozen
compatibility check does not establish full default learning equivalence.
Keep that limitation and do not claim an isolated crop effect.

Train **200,000 aggregate actions /12,477 updates** under **1,800 seconds**.
First update376, final credit.5. Retain the100k/6,227 recovery checkpoint before
replacement and the final200k checkpoint. Require all357 initial tensors to match
the bounded crop initialization and its47 pre-update transitions. Initial tensors
also match the same-preset full-frame control, but input trajectories and actual
initial frozen scores can differ. Do not require semantic equality to full-frame
control. No intermediate assessment or checkpoint selection.

Assess fixed final and actual-initial sampled/greedy policies, sampled primary,
first three natural episodes in each of eight lanes: **24 per arm**. Each arm has
600k-action/300-second limits, zero updates, guidance, labels and truncations,
and357 invariant tensors. Preserve the original **90%-two-wall** task gate.
Compare with retained full-frame **70.750 sampled /143.667 greedy**, own initial
1.417/2.208, in21m27.910s. Report absolute scores, task successes and
initial-subtracted return per training minute; keep sampling-mode tradeoffs
explicit. A lower training loss or larger image objects alone cannot select
the candidate. A promising reused-seed outcome would still need confirmation.

Require exact crop metadata/architecture, configuration, source/native identity,
counters, finite reports/checkpoints, changed CNN/reward/actor/value, full native
training pixel/action/label replay, four frozen outcome replays/videos, host
guards, curves, actual guidance exposure, comparison and terminal collection.
Ordinary frozen logs lack live pixel hashes; replay cannot prove unavailable
live-image equality. CPU preflight must exercise the actual commands, crop
configuration/argument rejection, startup headers, observers, checkpoint schema,
recovery retention and all input paths before launch.

Keep600s inner/630s outer guided replay,500s per frozen replay and1200s total
CPU audit. Main service cap4500s; postprocessing/progress5100s with4800s waits;
recovery3000s with2700s internal wait; collection5400s with5100s internal wait.
Observers cannot restart or extend training. Record CPU co-host time without
correcting training duration. Declare exact files, schemas and limits before
launch. Pin the protocol snapshot rather than a mutable status document.

Check live state before launch; no overlapping GPU work. Stop on real faults or
nonfinite values without host recovery. Preserve failures and inspect completed
artifacts before a separate CPU repair. No retry, cap increase, extension, crop
sweep, seed confirmation or automatic successor is selected by this protocol.

Root `/x/Temp/kindle-playfield-crop-20261102/full`, trial
`breakout-12m-64-b16-r16-playfieldcrop-expert100-200k-seed1009`.
Main `kindle-breakout-playfield-crop-200k-20261102`, with matching postprocess,
progress, recovery and collection observers. The five-game goal is unfinished.

CPU preparation passes in2.345s (2.451s wrapper): five actual CLI commands,
38 audit/comparison configuration cases, seven prelaunch command cases,
17 startup-header cases, four observer argument cases and13 recovery-body cases.
Recovery metadata, retention and collector checks pass; all374 bounded and338
control pins verify,357 initial tensors match and18 scripts compile. No native
agent or environment is constructed. The selector's actual input list is
resolved separately before launch; no earlier qualification stage repeats.
