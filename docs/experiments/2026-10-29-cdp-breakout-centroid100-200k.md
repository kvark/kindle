# Fresh Breakout 200k with current-pixel position supervision

Select one fresh **200,000-action / 12,477-update** learner after the completed
[native numerical and bounded checks](../results/2026-10-29-cdp-color-centroid.md).
Use a **1,800s training cap**. The bounded 48.101s extrapolates to 19m34s, or
21m32s with 10% allowance; neither is a full-run time guarantee. The retained
control takes 22m19s and scores 13.208 sampled / 17.875 greedy, with zero of 24
two-wall successes. The new loss has no measured gameplay benefit at selection.

Keep 12M/RGB64/Auto/N8/B16/T16/H15/R16, full18 actions, actor-unimix .01,
original rates, expert100 with world gradients and corrective labels, all18
action weights1, 20% episode overrides and cap1920. Add only the configured
centroid loss at fixed scale100: red excess `max(R-1.5*max(G,B),0)` inside
x[8,152),y[90,185), with detached normalized coordinate targets and a
training-only one-hidden-layer256-unit head. Native max-pooled frames receive
one GPU64 resize. No RAM targets, position input to inference or new teacher.

This adds an explicit color/region prior and 656,386 trainable parameters,
bringing the total to16,990,739. Checkpoints have372 tensors including the ten
Adam moments omitted by the original bounded CPU schema. Preserve that original
failure and its separately passing CPU repair. Use audited native2974160,
SHA256`e7ba0d4b208fa15be5790615034fb36ab727e4bd02b041114775b42b8e888e90`,
runtime`/x/Temp/kindle-color-centroid-20261029/runtime`. Native and runner differ
from retained447fe50; this is not an isolated causal comparison, even if shared
initial tensors and traces match. No rebuild or backend optimization is selected.

Seed1009 and evaluation4000001009 are reused development inputs. Start fresh;
reuse no weights, replay, belief, games, RNG or bounded-check experience. Require
first update at376 actions, one update per16 actions thereafter, and residual
credit .5. Audit100k/6,227 recovery before final200k/12,477. All372 initial
tensors must match the bounded learner; all357 shared initial tensors must
match the retained control. Do not select a checkpoint from intermediate scores.

After training, assess fixed final and actual initial policies under sampled
and greedy actions, sampled primary. Each arm selects the first three natural
episodes per eight lanes,24 total; retain excess episodes and partial tails.
Each has a600k-action/300s cap, zero guidance/labels/updates/truncations and372
invariant tensors. Both actual-initial semantic traces must reproduce the
retained control, excluding declared identity and finite wall-time fields only.
Keep the original90% two-wall gate (864 points) and report every failure.

Require finite numeric/config/source/counter checks, changed auxiliary/CNN/reward/
actor/value weights, full training pixel/action/label replay, full frozen
outcome/action/frame/reset replay and videos, host guards, learning curves,
exposure and paired comparison. Ordinary frozen logs contain no live pixel
hashes. Report expert CE and raw centroid loss/valid fraction over fixed50k-action
windows; these training-replay metrics are not held-out accuracy or competence.

Caps: training1800s, each evaluation300s, training replay600s inner/630s outer,
CPU audit1200s, main service4200s, postprocessing/progress/collection6000s with
5700s internal waits, recovery3600s with3300s internal wait. Read-only observers
cannot restart the learner. Terminal collection runs once after all required
services succeed; failures remain recorded. Stop on real GPU faults or numerical
failures; no automatic driver reset/reload/reboot.

CPU preparation exercises five actual CLI paths,15 scripts,42 resolved paths,
18 actual config cases,six header-readiness cases,four observer-parser cases
and five recovery-body cases. All41 bounded and132 control pins pass; no native
agent or environment is constructed. The actual plotting interpreter also
imports the comparator, checks shared357/full372 initial tensor comparisons and
runs the489-report bounded exposure analysis with native/ALE/Gym imports
forbidden and zero attempted imports. Its audit validates the new position-loss
window summaries. These checks precede declaration and GPU work.

No retry, cap increase, extension, second learner, sweep, automatic successor or
reuse of this life follows this selection. The earlier H5/R32 and modal-posterior
assessments remain terminal. A better training loss alone will not promote this
recipe; task performance and compute cost determine its value.

Root:`/x/Temp/kindle-color-centroid-20261029/full`.
Trial:`breakout-12m-64-b16-r16-centroid100-expert100-200k-seed1009`.
Main service:`kindle-breakout-centroid100-200k-20261029`.
Check live state before launching; never restart merely because observation
times out or a completed transient service unloads.

## Launch and startup

The selected service starts **20:53:05 UTC October29**, source`b58d3a2`, with187
unchanged declaration pins. PID346205/invocation3522002a6e344decb27f4ddd8b8f6988
passes startup at20:53:26:372 finite initial tensors matching the bounded check,
correct first-six update cadence and171 finite supervised reports through3112
complete actions. Progress,recovery,postprocessing and terminal-collection
observers are active. The final gameplay outcome remains pending.

Declaration SHA256:
`21c5a539eb81847da6cd10dbe16c1bf9a3889fae456af22ee899b51d4eb32df7`.

## Midpoint recovery

The declared100k/6,227-update recovery passes **21:04:40 UTC October29** with
all372 finite tensors. The observer checks6,516 finite reports through104,632
complete actions and exits successfully. Original PID346205/invocation
3522002a6e344decb27f4ddd8b8f6988 and all187 pins remain unchanged. The learner
continues; no frozen task outcome is established at this milestone.

## Terminal outcome: original comparison fails

Training completes200k actions/799,500 frames/12,477 updates in21m45s. Final
sampled14.375 and selected-natural greedy20.125 compare with13.208/17.875;
actual initials1.417/2.208 match. All0/24 two-wall successes. Final greedy has
one25,000-action timeout, so the declared zero-truncation requirement fails.
These scores remain descriptive; the four-arm comparison is not accepted.

All five GPU stages finish with passing guards. Main fails **21:18:18 UTC**
on the protocol assertion; automatic postprocessing/collection preserve that
failure. Separate CPU closure completes missing saved replays, full training
pixel/action/label replay, curves, exposure and comparison at **21:25:40**,
without repeating GPU stages. It verifies187 original/234 closure pins,
372 bounded-matching initial/invariant frozen tensors,357 shared control initial
tensors and both actual-initial semantic traces. Collection completes21:31:01;
status remains`closed_protocol_failed`, with no original passing review created.

Retain the control; no retry,extension,confirmation or automatic successor.
All services are terminal and no GPU allocation remains active at collection.
See the [full result and limitations](../results/2026-10-29-cdp-color-centroid.md#full-200k-comparison-fails-the-evaluation-protocol).
