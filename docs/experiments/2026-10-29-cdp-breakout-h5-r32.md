# Breakout shorter imagination with more fitting updates

Select one fresh bounded 12M64/Auto/N8/B16/T16/H5/R32 check. Breakout's retained
H15/R16/expert100 learner scores 13.208 sampled and 17.875 greedy, with zero
two-wall successes. Its corrected replay expert cross-entropy still falls over
the four 50k-action windows, ending at .696. Reconstruction and alternative
action readouts have not improved control; the posterior-mode assessment fails
its initial-control truncation requirement. None identifies a causal bottleneck.

Test allocating more work to fitting the existing state and policy. Change
H15→5 and R16→32 together, preserving 12M64/B16/T16 and 256 replay positions
per update. At 200k actions this doubles world/expert updates 12,477→24,954
while reducing total scheduled imagined transitions by one third. Sampling,
optimizer cadence, and the action-clock duration of the unchanged 1,000-update
warmup change too. This is a practical recipe comparison, not an isolated
optimization, horizon, or representation test. Actual gradient magnitudes and
interference remain unmeasured. Native optimization uses LaProp with adaptive
gradient clipping; scalar loss magnitudes do not establish parameter influence.

Qbert's H5/R32 recipe passes its gate on one of two learners. Earlier unaided
4M Breakout H5 and retained-weight H5 warm starts were weak; they remain
negative evidence, not matching controls for this supervised fresh recipe.
Do not assume that Qbert's benefit transfers.

Keep centered CDP, action-effects coefficient1, actor-unimix .01, full18,
native max-pooled frames and one GPU64 resize, sticky .25/action repeat4,
base/actor rate4e-5, encoder6e-6, dynamics4e-4, replay100k and original rates.
Keep the pixel edge-reset teacher, 20% episode overrides capped at1920,
expert100 with all18 weights1, world gradients and all-state corrective labels.
Seed1009/evaluation4000001009 are reused development inputs. Use unchanged
native447fe50/SHA35ad0db from
`/x/Temp/kindle-breakout-corrective-20261028/runtime` and the exact retained
runner. No build, weights, experience, replay, belief, games or RNG are reused.

Run 8,192 aggregate actions /978 updates, then 1,024 greedy frozen actions
with zero updates and no guidance. Expected first update376, then every8
actions, residual credit0. Require exact config/source/device/seeds/counters,
357 finite initial tensors matching the retained 12M64 control, 357 invariant
frozen tensors, changed CNN/reward/actor/value weights, finite aligned world
and actor expert reports, actual override exposure, complete training pixel/
action/corrective-label replay, and host guards. CPU preflight exercises both
actual CLI validation paths before native or environment construction, checks
the actual config assertion and the existing 12M64 checkpoint schema.

Caps: training180s, frozen60s, replay120s inner/150s outer, CPU audit180s,
whole service480s. The conservative 200k cost projection is
`training_seconds * 200000 / 8192 * 1.10`; require ≤2700s (45min) before
considering a separately selected full learner. This check measures execution
and early cost, not natural-episode competence or a speedup over the control.
No retry, cap increase, extension, sweep, automatic promotion or reuse of check
experience follows. Stop on real faults or numerical failures; no driver recovery.

Retained control:
`/x/Temp/kindle-breakout-expert100-capacity-20261029/full-repaired/breakout-12m-64-b16-r16-expert100-200k-seed1009`.
Evidence root: `/x/Temp/kindle-breakout-h5-r32-20261029/native-check`, trial
`breakout-12m-64-b16-h5-r32-expert100-check`.
Service: `kindle-breakout-h5-r32-check-20261029`.
The preceding modal assessment is terminal and explicitly protocol-failed;
its closure and all saved host guards pass. No learner is active at selection.
Recheck actual live state before launching; preserve that earlier failure.
