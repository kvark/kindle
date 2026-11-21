# Initial-episode Breakout guidance: bounded execution check

Select a fresh collection schedule: each of eight independent lanes receives
pixel guidance in its first two natural episodes, then returns to the retained
20% Bernoulli episode rule. Every selected episode retains the 1,920-action cap.
Lanes change phase only on their own natural resets, never at a global action
boundary. Draw once per episode even when selection is forced, preserving the
later per-lane random sequence. Corrective labels and actual-action feedback
continue on every state, including after release and in unselected episodes.

This tests whether more useful early experience improves subsequent learning.
It changes timing, quantity and distribution of guidance together. The completed
constant-50% and 32-action block experiments failed; neither tested this initial
two-episode schedule. Do not attribute any outcome to timing alone. Keep the
existing teacher and its limitations, including no demonstrated two-wall ability.

The fresh-only option is `--breakout-guidance-initial-episodes 2`, default absent.
Require corrective labels, positive episode probability and an explicit cap;
reject block-guidance combinations and training overrides on restore. Log the
distinct schedule kind and initial episode count. Existing audits and full pixel
replay must reconstruct the schedule from the exact pinned module.

Keep retained 12M/RGB64/Auto/N8/B16/T16/H15/R16, seed1009, replay100k,
centered CDP, 32×16, actor/latent mixing .01, base/encoder4e-5, dynamics4e-4,
behavior8e-5, imagined/replay critic1/1 and expert100/world gradients. Full18,
sticky.25, repeat4, noops0, native pooled RGB and one GPU resize remain. No new
loss, model input, architecture or native code. All optional actor paths,
coordinates, frame pairs and cropping stay off.

Use the qualified native7704d25c package with current matching Python wrappers
and the modified guidance module/runner in an isolated runtime. This is a newer
default-off runtime than retained447fe50; disclose the source difference and
require all357 actual-initial tensors to match retained. Early action traces
must differ because the initial selection changes. Explicit serialization of
the existing imagined replay-value-bootstrap default is allowed. Verify exact
teacher class ASTs against retained, and pin native, runner, package, helpers,
ROM, emulator, protocol and qualification receipts. No native rebuild.

First qualify CPU boundaries, independent random draws/counters, asynchronous
resets, cap release, continuing labels/feedback, CLI configuration/restore
rejection and replay accounting. Exercise both actual queued commands with
native/emulator construction forbidden. Preserve any failed fixture separately.

Then run exactly 8,192 fresh training actions and 1,024 frozen greedy actions
from its final checkpoint, seed4000001009, with no aid or learning. Require
first update376, final489 updates/debt.5, finite and correctly weighted losses,
unchanged architecture/count16,334,353 and changed CNN/reward/continuation/
actor/critic parameters. Verify all checkpoint schemas, initial equality and
all357 frozen tensors invariant. Replay all training actions, pixels, labels,
rewards, frames and boundaries; independently replay frozen actions/outcomes.

The short run may remain entirely within initial guided episodes. Report actual
episode coverage, cap releases, phase exits and override fraction; do not demand
or claim a live phase exit. CPU fixtures must cover the two-to-three boundary.
Any later full comparison must audit every lane's phase exit independently.

Use idle healthy AMD and existing host guards. Caps:180s train,60s frozen,
300s queue,180s audit,120s training replay and30s frozen replay. Stop on real
faults/nonfinite values; no driver recovery, automatic repeat or cap increase.
Retain original failures. A separate CPU repair may complete missing evidence
without repeating successful GPU stages. Verify original terminal identities.

Compare against the retained bounded47.689684s. Require the padded projection
`training_seconds*200000/8192*1.1 <= 1800` and at least2GiB headroom after
subtracting the entire5,939,200,000-byte full replay footprint from the minimum
observed bounded headroom. These are feasibility gates, not learning results.
No full learner, sweep or automatic successor is selected. A later comparison
needs its own fixed budget and frozen actual-initial controls, judged by useful
return gained per training minute. Do not reuse bounded experience.

Root `/x/Temp/kindle-breakout-initial-guidance-20261106`. Retain RGB64 and
143.667 greedy/70.750 sampled. Original864/90%-two-wall and the five-game goal
remain unfinished.
