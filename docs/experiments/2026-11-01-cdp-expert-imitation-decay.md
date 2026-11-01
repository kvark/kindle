# Optional native imitation decay before a new learner

The Breakout action interventions show that disagreement with the pixel teacher
is not necessarily a worse local action. Earlier gradient attribution also
found large imitation contributions. Select an optional native schedule that
can preserve early imitation and then release its direct loss pressure,without
changing teacher labels or online action guidance. This is a hypothesis to test,
not evidence that imitation causes the remaining gameplay gap.

Add`expert_imitation.decay` with`start_update` and`end_update`,counted by the
one-based native LearnReport counter. Keep the original scale through the start,
linearly reduce it to zero at the end,and hold zero thereafter. Apply the same
scale to actor and optional world imitation losses. Keep raw weighted cross
entropy,labeled-row counts and target-weight sums unchanged in reports;report
the scheduled scale separately. Existing actor-update gating remains separate.
Existing optimizer moments are preserved,so zero current imitation gradient
does not erase its earlier influence.

Default omission must preserve legacy config serialization,parameter names,
graph construction and constant weighting. Restore keeps the stored schedule
and learner counter. The CLI accepts only fresh-run schedule overrides and
must reject old native runtimes that would silently ignore the new field.
Training and inference remain native on Meganeura+Blade. No Python learning,
pretrained encoder,teacher rewrite or backend optimization is selected.

Qualification requires CPU tests for boundaries,large counters,serialization,
invalid intervals,graph inputs/parameter invariance,CLI propagation and report
auditing. A separately guarded native numerical fixture compares scale100,50
and0 against F64 loss/derivatives,including an independent auxiliary gradient;
raw imitation metrics must remain unchanged. Complete Rust/Python CPU checks,
strict Clippy and isolated runtime packaging before bounded Atari work.

Then select a bounded fresh Breakout execution/cost check,8,192 actions under
120s,using12M64/Auto/N8/B16/T16/H15/R16/expert100/encoder2e-5 and a compressed
decay interval128→256 updates. This checks both boundaries and a zero-weight
tail within489 updates;it does not measure gameplay benefit or share experience
with a later learner. Require exact config/counters,finite reports,raw actor/world
expert-loss alignment,applied scales,initial tensor matching,full training
pixel/action/label replay,host guards and1,024 frozen actions with zero updates
and357 invariant tensors. Preserve checkpoints,evidence and failures;never
repeat completed GPU stages to fix CPU reports.

A potential full comparison would use the retained22m27s Breakout recipe with
scale100 held through update6227 (100k actions),decaying through9352 (150k),
then zero for the final3125 updates to200k. Keep online guidance and corrective
labels fixed. This would test loss scheduling,not complete removal of human
aid. Select a full learner only after qualified runtime and bounded cost pass,
with its own protocol and frozen final/actual-initial sampled/greedy assessments.
No full learner,coefficient sweep,unchanged extension or automatic successor is
authorized by this qualification declaration. The five-game goal is unfinished.

Evidence root:`/x/Temp/kindle-expert-decay-20261101`. Check live state before
GPU work and retain the accepted runtimes and completed learners unchanged.
