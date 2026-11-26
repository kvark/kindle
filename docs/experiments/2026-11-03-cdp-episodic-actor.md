# Optional actor learning from completed real episodes

The [replay-target audit](../results/2026-11-03-cdp-breakout-replay-actor-targets.md)
finds a missing direct return-weighted actor path, but existing critic targets
retain imagined bootstraps. Select an optional native completed-return actor
objective. This implements a different target; no gameplay benefit is assumed.

Add `replay_actor_scale`, absent by default. When enabled, native sequence replay
finalizes discounted observed returns only at a true terminal. Each state stores
the sum of subsequent observed scaled rewards through that terminal, with the
usual continuation discount and no lambda mixture or value bootstrap. Retained
suffixes remain valid if earlier frames were evicted. Stop at episode boundaries;
never finalize unfinished or time-limited episodes or cross streams/resets.
Do not preserve replay or these targets across checkpoint restore.

The optional loss is mean categorical action cross-entropy weighted by
`clamp((G_complete - V_current) / return_scale, 0, 1)`, using the outgoing executed
action at each eligible replay state. Targets and features are detached from
this term; it updates only the existing actor parameters. Keep the existing
imagined actor/critic, replay critic and teacher losses. Normalize over the
ordinary B*(T-1) rows, including masked rows. Report completed/positive row counts,
weight sum and raw loss separately. No new parameter or checkpoint tensor is added.

Completed returns include the stored extrinsic/intrinsic channels with their
configured scales. Model disagreement bonuses are computed later from current
parameters and are not stored observed rewards. Reject this optional loss when
that model-derived bonus is active, rather than compare unlike return and value
definitions. A first Breakout candidate therefore also disables disagreement;
it differs from the retained action-effects1 recipe. Keep that difference
explicit and qualify a bonus-off control before attributing a gain to the actor
loss. Any stored visitation bonus remains an observed reward. Reject resident
device-target mode for this experimental path; AMD Auto already uses CPU target
construction with native GPU graph learning.

Require CPU tests for complete/partial/timeout/reset episodes, retained suffixes,
stream separation, reward scales, outgoing actions, prefix slicing, positive
advantage/clipping/masks, finite values and default compatibility. Require
unchanged parameter layouts and conditional serialization/report fields.
Run ordinary Rust/Python CPU checks and strict Clippy, and package an isolated
runtime. Qualify native actor-only loss/gradients against F64, including zero
targets and uniform action mixing, before any learner. Restore a retained
checkpoint with the feature off for bounded legacy compatibility.

After those checks, separately declare a fresh short Breakout bonus-off control
and scale100 candidate using12M64/Auto/N8/B16/T16/H15/R16 and retained rates,
constant expert100/critic1/1, full18/sticky.25/repeat4, native RGB and one GPU resize.
The check must establish real terminal targets actually occur, correct cadence,
finite gradients/reports, changed actor parameters, full replay accounting,
frozen zero-update behavior and acceptable early cost. It must not reuse an
earlier learner's experience. A full learner, cap, seed comparison and gate
decision require separate selection after bounded evidence. No sweep or
unchanged extension is authorized by this protocol.

This is a clipped, normalized positive-advantage imitation variant, motivated
by [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf),
not an exact reproduction of that algorithm. Completed off-policy outcomes can
include teacher overrides and stochastic actions. They do not identify causal
action value or guarantee improvement. Initial target availability depends on
natural episode completion, and the critic may suppress positives by
overestimating returns. Preserve every selected check, including failures.

Use `/x/Temp/kindle-episodic-actor-20261103` for isolated builds and evidence.
Keep old runtimes/closed experiments unchanged. Check live state before GPU
work; no overlap or automatic full learner. The five-game goal is unfinished.
