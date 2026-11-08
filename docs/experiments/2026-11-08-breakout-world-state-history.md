# Breakout: privileged state and image-history readouts

The [closed H32 recipe](../results/2026-11-08-breakout-world-h32.md) worsens
fitting and development forecasts despite explicit32-action supervision. Earlier
[state substitutions](../results/2026-11-07-breakout-world-state-swap.md) repaired
many readouts with image-informed recurrent h,but used models with no spatial
image-history route. Repeating that four-mode test cannot separate the current
local-action decoder's recurrent/categorical features from its two-image memory.
Select a new eight-mode frozen intervention before another training recipe.
Root `/mnt/data/kindle-breakout-world-state-history-20261108`.

Use exactly local-action H16 final8192 and H32 final4096,one at a time. Both are
v18,80 logical/8,893,203 scalars/240 tensors,RGB64/B8,categorical RSSM,history2,
direct action planes,width16. Their saved configs differ only in T/full BPTT,
world/dynamics rates and optimizer age/H16 versus H32. Keep every weight/moment
and age unchanged. These related development checkpoints are not independent
confirmation. No actor/critic,optimizer step,emulator,collection,new objective,
architecture or production-forward change.

Use one new native test executable,without a wheel or runtime overlay. The
existing state-substitution implementation/receipts stay unchanged. Reuse exact
production observe/transition/decoder/head graphs and causal history helpers.
Only new test-only orchestration is selected. CPU checks validate factor choices,
lane order,reset-aware older/newer histories and no hybrid feedback. Before GPU,
archive exact source/executable/dependencies and validate data. Reuse unchanged
native math receipts;do not repeat primitive qualification.

Take the first32 of the same128 dense development origins,in original B8 order,
and all supported original-development terminal windows centered at future17
after history16. Record every exclusion and padding lane;never score padding or
unknown post-terminal/reset arrivals. Use seeds83001/83003/83009/83023. Reuse RGB64
from the original single GPU resize. Targets/phase/lives/RAM/rewards/terminal
labels remain CPU metadata;no label enters or trains a network.

For each checkpoint/batch/draw,compute one canonical prior and a separate
image-informed48-frame teacher. Prefix features must match exactly. Retain all32
prior/teacher features and prior/teacher logits. Replay cached prior transitions
only to expose logits;states/samples must match exactly with the captured RNG.
The first draw's eight shared natural-case arrays must reproduce each model's
cached dense export exactly. Other frames/interventions are new missing outputs,
not a repeat of the closed assessment.

At each future step decode all2^3 choices of recurrent h,categorical z and image
history. Each block comes from the fixed prior or fixed teacher. Teacher h uses
true observations through the previous arrival;teacher z also sees the current
true image. Teacher history contains only the two earlier true images,with the
same reset masks as production. The submitted incoming action stays identical
in all eight modes. Every teacher substitution after the prefix is privileged
future information. No hybrid output or substituted state feeds either path.

Canonical history starts with the production prefix history and shifts in only
canonical prior images. Readout-only history substitution must not change any
head output. Require bitwise prior RGB/reward/positive/continuation parity at all
32 steps,bitwise head equality across history pairs,and bitwise prefix/history
controls. At step1,the two history choices are identical;h-substitution controls
agree within1e-5 absolute plus1e-5 relative across observe/transition graphs.
Verify full-teacher final features/posterior against the separate canonical
48-observation predictor. Save first reports before optional derived assertions.
All outputs are finite;initial/final checkpoint tensors and ages are exact.

After reviewed CPU/data/build checks,separately declare one900s AMD Auto native
diagnostic under the existing host guard. No full learner is selected. Record
native,worker,preparation and CPU analysis costs;other host GPU work may exist.
Check live process/device/boot/driver identity,stop on real fault/nonfinite state,
and perform no reset/reload/recovery/NVML. An observer timeout is not a stopped
job;never restart the same worker because observation expired.

Retain five endpoint images and all32 head traces for every mode/draw;full32
images for draw0 of the first natural and terminal batch of each checkpoint.
Keep all actual masks/locations/actions and before/after checkpoints. CPU analysis
reports observed/full-teacher ceilings,all1/4/8/16/32 ball detection/within1px/
error,paddle,bricks and per-origin paired rescues/harms. Report every mode and
both groups,not a best mode. Cumulative event scores require a complete valid
prefix;terminal arrival17 is supported and subsequent arrivals are unscored.
Keep natural and outcome-selected terminal slices distinct.

History pairs' identical heads establish a structural invariance,not a learned
repair. Mixed states can be off-manifold;privileged readout improvements are not
causal percentages,an achievable forecast,calibration or a training result.
Do not replace an original retention gate with a teacher-assisted score. All
original failed learners,assessments,collectors and diagnostics remain closed.
No sweep,extension,automatic successor or completed-stage repeat. Full physics,
events,brick memory,late/tunnel/second-wall coverage,untouched families and fresh
confirmation remain required;the world-model and five-game goals stay unfinished.


CPU preparation:the first test-executable build failed eight counter-type
comparisons (expected usize against learner u64/optimizer u32). Its original
compiler log/failure receipt remain retained. Separate explicit u64 comparison
repair passes three new CPU checks,with the native diagnostic ignored,and
strict release core/test Clippy passes. Cached data preparation passes first:
32 natural/10 terminal origins,six batches/six padding lanes,eight exact source
reference joins and48 planned queries. No GPU/model/emulator construction.
Production Rust changes are only a cfg(test) module registration plus the new
orchestrator;all production graphs and earlier diagnostics are unchanged.


The original GPU diagnostic is protocol-failed:its first trace/common-cache/
prefix/history/head controls complete,but the final canonical teacher call
rejects an empty future list (API requires1..64 full batches). No per-query export
or result is completed;the unsaved trace and9.72s native-test cost remain spent.
The guard stops on process exit101,no device fault/recovery,and seals27 pins.
A separate repair is selected only for the still-missing diagnostic outputs.
Preserve original source/executable/guard/empty query log unchanged. Supply one
unused complete NOOP future batch to the canonical48-observation predictor;
its posterior is computed before that extra prediction. The dummy future never
feeds a hybrid or a score. No production graph/API validation is changed,and no
completed assessment or collector repeats. Rebuild only the test executable,
review CPU/source/data pins,and separately declare a new900s repair unit/root.
The first packaging CPU assertion also falsely matched the dependency name
`thiserror`;its failure/repair remain and completed Clippy was reused.
