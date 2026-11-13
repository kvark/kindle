# Linear future references do not make pixel velocity controls a general fix

The [CPU-only diagnosis](../experiments/2026-11-07-breakout-world-linear-reference.md)
is closed. The models still make forecast errors on exact linear reference
futures, but neither fixed pixel point control solves the broad slice. The past
can include bounces or other motion changes even when the selected future is
linear. On the rare fully linear observed-and-future cases, the history-line
control works well, with very limited support. **This does not identify an
isolated architectural cause, qualify long forecasts or change any failed gate.**

The finding narrows the interpretation of the failed
[paired-data learner](2026-11-07-breakout-world-paired-actions-learning.md).
Freeflight-origin labels did not guarantee collision-free futures. Removing
nonlinear reference futures still exposes error, but a simple past-image
constant-velocity baseline is not a demonstrated replacement. No model/emulator
query, GPU allocation or learning update occurs in this analysis.

## Fixed support and controls

Use the original dense256 fitting/128 development origins for coverage, motion
rollout and paired-data final worlds, first fixed model draw. The separate448
paired fitting origins use that new world's existing export. Recover sixteen
past RGB64 images from cached source data; current/previous images, targets,
positions, lives and masks match every export exactly. No CPU resize enters model
data. Source/action construction is checked, not native argument buffers that
were never saved.

Require real available future, no terminal/life change, finite moving native ball
references with a stable last transition, current/target RGB support and exact
constant-velocity extrapolation of every reference future step. No tolerance is
searched. The stricter slice also requires the entire observed reference history
to follow that line without a reset/ending/life change. These are future-outcome-
conditioned diagnostics, not deployable filters or calibrated population tests.
Finite RAM alone does not establish ball presence; the conservative RGB detector
also excludes upper-board objects.

The last-two control extrapolates the last two observed RGB ball centers. The
history-line control fits ordinary least squares at actual indices−15..0 over
all visible observed centers (at least four including the current one), without
reset/ending/life changes. Neither sees future data or references when fitting.
Out-of-field predictions are not clamped. Missing support is a failure in the
full-eligible denominator; common-control-supported tables keep identical cases.
Mean position errors are conditional on a returned model point, so detection and
tracking counts stay explicit. These are point estimates, not new distribution
scores or acceptance gates.

## Common-control-supported future-linear slices

Each cell is within-one-resized-pixel / identical supported origins. Counts are
related states with reused seeds, not independent episodes.

|Cohort / horizon|Common origins|Coverage|Motion rollout|Paired data|Last two|History line|
|---|---:|---:|---:|---:|---:|---:|
|fitting / 1|111|104|88|57|84|23|
|fitting / 4|88|40|50|22|15|17|
|fitting / 8|42|9|20|3|3|5|
|fitting / 16|9|0|2|0|0|0|
|fitting / 32|0|untested|untested|untested|untested|untested|
|development / 1|75|67|59|32|55|5|
|development / 4|59|16|24|2|12|2|
|development / 8|32|3|9|1|2|0|
|development / 16|6|0|0|0|0|0|
|development / 32|0|untested|untested|untested|untested|untested|
|paired fitting / 1|224|—|—|194|200|72|
|paired fitting / 4|128|—|—|60|32|48|
|paired fitting / 8|40|—|—|10|8|8|
|paired fitting / 16|0|untested|untested|untested|untested|untested|
|paired fitting / 32|0|untested|untested|untested|untested|untested|

Full eligible counts at1/4/8/16/32 are148/114/62/10/0 fitting,
98/74/40/6/0 development and328/224/48/0/0 paired fitting. Full-eligible,
control-specific support, exclusion cascades and common-support tables remain
in `result.json`; all4,160 origins/horizons, including excluded cases, remain in
`cases.jsonl`. Do not silently replace an unsupported cell with a different set.

For development at8, common support has32 origins: coverage detects30 and tracks3;
motion detects27/tracks9; paired detects8/tracks1. Last-two point error averages
2.435 pixels; history-line error32.658. There are no fully linear observed16+
future8 development cases. Thus the large history-line error is not evidence
that the model was given a clean sixteen-frame straight path. The finite-step
last-two image estimate is also inaccurate when extrapolated. The current study
does not isolate quantization, pooling, occlusion, past collisions or an optimal
history estimator as its cause.

Paired fitting at8 has40 common-control origins but only **five distinct cached
RGB/action histories**, repeated under related plans/repeats. Paired tracks10/40,
last-two8/40 and history-line8/40. The observations reconstruct40/40, yet only20
forecast balls are detected. This supports further forecast investigation without
an independence, architecture-defect or proof-of-unobservability claim.

## Fully linear observed and future references

At4 the strict fitting slice has six origins: history-line6/6, last-two0/6,
coverage3/6, motion4/6, paired1/6. Strict development has two origins:
history-line2/2, last-two0/2, coverage1/2, motion1/2, paired0/2. Mean history-line
errors are .255/.152 pixels; last-two errors2.212/1.499. These few cases show the
line fit works when its assumptions hold, while the learned futures can still
miss. They do not establish reliability, a global explanation or a broad remedy.

At8 only one strict fitting case exists: history-line1/1, motion1/1, coverage0/1,
paired0/1. There are no strict development or paired fitting cases at8, and no
eligible32-step cases in any cohort. Long-horizon competence remains untested by
this restricted diagnosis. The all-world failures from prior assessments remain.

## Execution and provenance

Root `/mnt/data/kindle-breakout-world-linear-reference-20261107`, source base
`f883fe08816171fc3713a419773d5521f7333e70`. All500 primary input pins,21 source
field joins,56 cross-export field matches and42 unfiltered published instrument
controls pass. Algebra/known-linear/missing-support/reference-disruption/image
fixtures pass. Main analysis4.418s; no native/emulator package is even imported.
The output pins current authoritative cached bytes and checks their consistency;
it does not claim all historical closures sealed every raw export.

A separately pinned505-input CPU review independently recounts260 forecast/
control cells from the ledger and joins all448 paired parents back to their
retained fitting sources/seeds. It takes .255s, preserves the primary ledger and
creates `paired-parent-metadata.json`; the original paired ledger's absent seed
fields are not rewritten. Primary analysis and recount pass first try. The original closure pins its
active stdout while still empty, then writes its final line, leaving that one
hash stale. Preserve the original receipt/log and failed independent validation.
A separate CPU-only seal repair excludes the active writer, retains historical
document bytes and verifies all unchanged analysis inputs/results without
rerunning analysis or changing gates. `comparison.png`
shows tracking and support separately and was visually inspected.

The controls are an analysis instrument only. No new loss, decoder, policy or
critic is trained, and none of the original acceptance gates changes. The
[separately selected frozen-observation-model phase](../experiments/2026-11-07-breakout-world-frozen-visual.md)
will test a practical staged training recipe: inherit the good observed H0 world,
hold its learned visual weights fixed, and train H16 causal transitions. That
native phase is not implemented or allocated yet, and its outcome is unknown.
It combines warm initialization, H16 training and a frozen subset, so it would
not isolate freezing as a cause. The prior RGB-coordinate whole-agent experiment
remains negative/closed; its native qualification is not repeated. Full causal
physics, events, brick memory, late/tunnel/second-wall coverage, untouched families
and fresh confirmation remain necessary for the actual world-model goal.
