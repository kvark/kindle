# Fresh guided four-action Breakout at200k actions

Separately select one full comparison after the [bounded four-action check](../results/2026-11-03-cdp-breakout-minimal-guided.md).
It uses13.75% less logged training time,12.02% less including construction,
with a padded200k projection18m24.645s. It does not establish gameplay quality.
The original preflight initialization and CPU collector failures remain
preserved; repaired qualification and terminal closure pass. Do not repeat the
closed short learner, its frozen stage, replay or collectors.

Keep the qualified exact12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1
configuration, encoder/base4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/
replay-value1, disagreement1, expert100/unit weights/world gradients, sampled
collection/posteriors and actor-unimix.01. Use the existing `published-minimal`
NOOP/FIRE/RIGHT/LEFT adapter, sticky.25/repeat4/noops0/native pooledRGB and one
GPU64 resize. Preserve20% episode guidance/cap1920 and all-state corrective
labels. The mapping translates around the unchanged full18 teacher internally.
Use native447fe50 and the exact bounded runtime/runner under
`/x/Temp/kindle-breakout-minimal-guided-20261103`; no rebuild or native change.

Start fresh with learner1009/evaluation4000001009 and the retained eight lane
offsets. Transfer no weights, replay, live states or experience. Stop at200k
actions/12,477 updates or30min, whichever comes first. Save the actual initial
and read-only100k/6,227 recovery with moments; these omit replay, games, beliefs
and full RNG state, so do not support exact continuation. Require first update
376, correct cadence,357 initial tensors matching the bounded four-action
candidate and matching pre-update semantic events. Full18 initial tensors and
traces are not expected to match: dimensions and initialization change.

Assess final and own actual initial with both sampled and greedy actions,
without guidance or labels. Each arm selects the first three natural episodes
per eight lanes,24 total, with zero updates/truncations and357 invariant tensors.
Preserve600k-action/300-second bounds and100k-frame episode limits. Keep excess
episodes and partial tails; never substitute timeouts for natural episodes.
Require full training pixel/action/label replay (900s cap), full frozen recorded-
action replays/videos (500s each), finite reports, exact sources/native/ROM/host
guards, curves, guidance exposure and score-band coverage. Aided online returns
and replay fitting are not unaided competence.

Reuse the closed full18 retained200k arms: greedy143.666667/own initial2.208333,
sampled70.750/own initial1.416667, training1287.910495s. Primary evaluation is
greedy; sampled remains a reported secondary outcome. Report absolute returns,
actual training time, separately recorded construction, and initial-subtracted
return per training minute. Use10,000 paired lane-bootstrap resamples, seed0,
with95% intervals. Keep the actual own initial for each vocabulary.

Favor replacement of the primary development recipe only if greedy return and
gain/min both rise and the greedy gain/min interval excludes zero. Separately
identify a cheaper budget point if training time falls and that gain/min
interval excludes zero, even if absolute return is lower; it does not replace
the strongest absolute score. Check the original mean864/90%-two-wall task gate
directly. A positive development comparison would need separately selected
fresh-seed confirmation before a reliability claim.

Changing action count also changes action-effect centering, uniform mixture,
policy/dynamics dimensions and initialization. This is a game-specific recipe
comparison, not an isolated output-layer speedup. The reused seed/cohort and
historical control timing are development evidence. Preserve CPU co-host times
without correcting training duration. The original five-game goal is unchanged.

Before launch, exercise the changed CPU accounting/comparison helpers, all five
actual queued commands and collector identity checks. Resolve the bounded
protocol and retained replay source from their exact preserved bytes; never
rewrite original declarations. Resolve native metadata before replacing both
package and native agent exports in CPU command fixtures, preventing the prior
lazy-loading failure. Keep native learning unchanged and check live state.

Root`/x/Temp/kindle-breakout-minimal-guided-20261103/full`; service
`kindle-breakout-minimal-guided-200k-20261103`. Main/terminal caps7200/7500s;
separate CPU recovery and terminal observers never launch GPU work. No overlap,
successful-stage repeat, cap increase, coefficient sweep, extension, second
learner or automatic successor. Stop on real faults or numerical failures.
