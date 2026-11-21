# RGB event-head saved score: residual images and survival weights

The [CPU-only diagnostic](../experiments/2026-11-10-breakout-world-rgb-event-heads-survival.md)
closes with every original score reconstructed. Holding rendered features fixed,
oracle-survival weighting changes primary ball energy.142943→.073155. This is a
future-label score substitution, **not a passing score, learned repair or causal
error fraction**. The [ordinary assessment](2026-11-10-breakout-world-rgb-event-heads-branches.md)
still fails its original joint-energy and four event gates; the full goal remains
active and unproven.

| Primary history16 score | Original | Same model mean | Oracle-alive mean | All alive |
|---|---:|---:|---:|---:|
| Ball, all categories |.142943452|.142919549|.073155000|.111101429|
| Ball, nonending categories |.090231898|.090203981|.083956683|.083956683|
| Paddle, all categories |.087931692|.087934534|.003173169|.061793019|
| Paddle, nonending categories |.020526411|.020529495|.003512099|.003512099|

Same-mean weights remove draw association without materially changing these
aggregates. The oracle-mean variant sets each origin's model-survival weights to
its realized oracle-alive fraction; all-alive ignores ending predictions. Neither
variant changes model images or learns a predictor, and no original gate is
applied to their values.

| Category, mean ball energy at16/32 | Original | Oracle mean |
|---|---:|---:|
| Serve |.327604896|.336614274|
| Freeflight |.030512828|.023652536|
| Contact |.037318308|.035908926|
| Brick |.045210182|.036338322|
| Wall |.075514658|.058692045|
| Historical life_loss |.025230516|.012533996|
| Ending |.459212774|.008344901|

The wider model's cached oracle-weighted ball residual was.127843450 overall,
.496434598 for wall and.276915584 for serve; the new values are.073155000,
.058692045 and.336614274. Wall image-feature residual improves substantially,
while serving remains poor. A detector feature can represent a diffuse centroid
rather than a coherent ball; these scores do not establish complete physics.
The selected ending score is highly sensitive to survival predictions, alongside
its ordinary premature ending risks. This does not identify which part of the
model or training caused them. Historical life_loss includes surviving edge
rebounds, and restricted detector support is not physical absence.

## Closed integrity

Source`39477a361678b68dd8966e322ed3465367982cd0`; root
`/x/Temp/kindle-world-rgb-event-heads-survival-20261110`. The new declarationv2
pins317 inputs.6,720 cases,840 groups and420 ending-probability/Brier checks pass;
all original per-origin/group/primary scores reconstruct within1e-12, maximum
error4.44e-16. All6,720×4×5 values are finite. CPU analysis takes5.062s.
Declaration and16 closed files verify after execution. Full case/index/group
arrays and all four additive terms remain retained for both histories.

No model/GPU/emulator query, update or collection occurs. Constructors are
trapped and Vulkan disabled. Unchanged F64 math and sealed fixture receipts are
reused without rerunning old analyses/fixtures; only old scalar decomposition
JSON is read for comparison. Ordinary branch source files and checkpoints remain
unchanged.

## Original failure and separate compatibility check

Preparationv1 fails an inherited assertion that the entire runner file is
byte-identical to the older cached runner. A subsequent attempted analysis stops
at the missing declaration before scoring cases. Both scripts and logs remain.
They are not relabeled as passed.

The diff adds optional RSSM/denoising dispatch, a CLI option and descriptive
metadata. A separately pinned check verifies exact AST equality of every module
node except the two dispatch entrypoints, including all scorer/feature functions
and constants; both full source hashes are retained. The closed native job
explicitly used RSSM. Separately namedv2 declaration/analysis then execute once
and reconstruct the ordinary scores above. No GPU job, old analysis or fixture
is repeated. This compatibility finding does not alter the ordinary failed gates.

## Next question

Select the [saved serving/ending timing review](../experiments/2026-11-10-breakout-world-rgb-event-heads-timing.md)
for CPU implementation and separate declaration. Inspect risk accumulation and
fixed images/prefix metadata before choosing another learning change. No new
learner,native job,extension or automatic successor is selected. Full physics,
natural event calibration,persistent brick memory,late/above-top coverage,
untouched families and fresh confirmation remain; actor/critic,CDP and five-game
work stay parked.
