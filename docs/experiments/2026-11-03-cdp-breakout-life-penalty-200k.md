# Fresh 200k Breakout comparison with one-point life-loss feedback

Separately select one fresh learner after the [bounded native check](../results/2026-11-03-cdp-breakout-life-penalty.md)
passes 8,192 actions / 489 updates in 47.317s. All 408 pins, 357 matching initial
and invariant frozen tensors, raw-score/life/reward accounting, full guided
replay and host guards pass. All 118 observed lost lives receive negative
training feedback. The conservative 200k projection is 21m10.713s, below 30min.
Both original services succeed; preserve the CPU fixture, path and terminal
reader failures and their corrections. Do not repeat the completed check.

The hypothesis is that explicit negative feedback at lost lives may help the
learner distinguish paddle failures sooner. Life loss already affects future
scoring opportunities; the saved zero immediate rewards do not establish a
missing-loss bug. This comparison measures one additional training aid, not
an isolated architectural mechanism or independent seed reliability.

Use **200,000 fresh aggregate actions / expected 12,477 updates**, with a fixed
**1,800-second training cap**. Keep centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/
microbatch16/H15/R16/lambda.95, encoder/base4e-5, behavior8e-5, dynamics4e-4,
replay-value1, expert100/unit weights/world gradients/all-state corrective
labels, sampled collection/posteriors, actor-unimix.01, action-effects1,
replay100k, warmup1000, 20% episode guidance/cap1920, learnerseed1009 and
evaluationseed4000001009. Full18/sticky.25/repeat4/noops0, native pooled RGB
and one GPU64 resize remain. No weights, replay, belief, games, RNG or bounded
experience transfer. Native447fe50/SHA35ad0db and the teacher remain exact.

Use the qualified runner SHA9ff2552b and isolated runtime under
`/x/Temp/kindle-breakout-life-penalty-20261103/runtime`. Only training enables
`--breakout-life-penalty`: F32 training reward is raw reward minus net lost lives.
Preserve the versioned reward contract, separate raw/training fields, real
termination/truncation flags and reset baselines. ALE life counts are reward
feedback, never an actor input. Gains have no penalty; game-over losses count.
All online and task scores remain raw. Full guided replay must independently
verify actual ALE life counts and penalty arithmetic alongside pixels, actions
and teacher labels. The original raw-score audit remains enforced separately.

Require 357 initial tensors matching bounded candidate and retained control,
correct first update at376, and paired pre-update semantics. Against control,
normalize only the declared extrinsic training difference, retaining raw reward,
action, boundary, teacher and intrinsic fields; the qualified full auditor checks
the declared difference. Candidate/bounded comparison retains all shaping fields.
Retain 100k/6,227 recovery weights with exact metadata and the final checkpoint.

Run fixed final and actual-initial sampled/greedy assessments: first three
natural episodes per eight lanes, **24 per arm**, zero updates/truncations,
guidance/labels/shaping off. Each arm has a 300-second cap and unchanged
25,000-action episode limit. Require 357 invariant tensors per arm, matching
actual-initial raw semantic traces, full recorded-action/reward/frame/reset
replays and videos, finite training, changed CNN/reward/actor/value and guards.

Compare sampled and greedy separately against retained **70.750 / 143.667**
and own initials **1.417 / 2.208**, all zero of24 two-wall. Report actual time,
actions/frames/updates, raw and initial-subtracted return per training minute,
natural episode outcomes, both seeds and guidance exposure. Retain the original
90% two-wall gate. A development score gain is not that gate or seed reliability.
No checkpoint/cohort selection after outcomes, penalty sweep, retry, extension,
cap increase, confirmation or automatic successor.

Automate progress, 100k recovery, postprocessing and original terminal collection.
CPU preflight must exercise actual training/frozen commands, exact configuration
and shaping contracts, startup gates, recovery retention, observer arguments,
source/runtime provenance and terminal journal identity. Use the existing
4,500s main service and 1,200s CPU audit envelopes. Check live state before launch;
allow no overlapping Kindle GPU jobs. Retain CPU co-host timing caveats without
correcting training time afterward.

Prospective root `/x/Temp/kindle-breakout-life-penalty-20261103/full`, trial
`breakout-12m-64-b16-r16-life-penalty-expert100-200k-seed1009`, service
`kindle-breakout-life-penalty-200k-20261103`. Full-run preparation and launch are
unfinished at selection. Gameplay benefit is unmeasured; the five-game goal
remains unfinished.

## CPU preparation

Full preflight passes **06:14:37.261 UTC November 3**, source `2eafbf2`, in
2.237s (2.340s wrapper). Five actual commands, 34 exact configuration cases,
24 startup-header cases, seven pre-update comparisons, 12 recovery-body cases,
four observer-parser cases, eight prelaunch command cases, four metadata cases,
six retention cases and two collector-retention cases pass. All 19 scripts
compile and 834 declaration path entries resolve. The 408 bounded and 338
control pins remain unchanged. Native/teacher/qualified runner remain exact;
no native or emulator is constructed. No preparation failure or repeated
successful stage occurs. Full selection and launch remain pending at this milestone.
