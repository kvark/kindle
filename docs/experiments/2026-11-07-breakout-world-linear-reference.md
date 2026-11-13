# Breakout: retained forecast errors on linear reference segments

The [paired-data learner](../results/2026-11-07-breakout-world-paired-actions-learning.md)
fails broad reconstruction and forecasting. It also tracks only 4/64 supported
balls at eight decisions on its new fitting freeflight-origin slice. Those
origins can encounter bricks later; a freeflight starting label does not prove
an entire future is collision-free. Before selecting another learner, isolate
errors on reference trajectories that are exactly linear and visibly supported.

Select CPU-only analysis at
`/mnt/data/kindle-breakout-world-linear-reference-20261107`. No new model query,
learning update, emulator, collector, GPU allocation or native rebuild. Do not
repeat any completed learner or assessment. This is an error diagnosis, not a
replacement world-model success criterion or an automatic successor allocation.

Use coverage/motion-rollout/paired final dense fitting and development exports,
with their identical original source locations and first fixed model draw. Keep
the paired-only 448-origin fitting export separate. Examine all 1/4/8/16/32
horizons. Report unsupported/empty cells without replacing their cohorts.

For each origin/horizon, require all real future arrivals available, no ending
or life-count change, finite native ball references, a nonzero last observed
reference displacement, and supported RGB ball centers at the current/target
arrival. Define an exact future-reference-linear slice: every reference future
position equals the last observed position plus its action index times the last
observed displacement. Separately identify origins whose entire observed history
also follows that line. Exactness is deliberate; do not search tolerances to
obtain a favorable cohort. RAM can be stale, so finite coordinates alone do not
establish presence. Preserve all eligibility counts and exclusion reasons.

This slice conditions on actual future outcomes/references. It is not a deployable
filter, independent test, causal error fraction or calibration assessment. Phase
and RAM references remain analysis-only and never enter a model or training loss.

Measure decoded observed/forecast detection, within-one-resized-pixel counts and
position error on identical supported images. Add point controls using only past
cached RGB64 ball centers: last-two-center constant velocity and a least-squares
line over the sixteen observed centers (at least four detections and the current
one). Report missing baseline support; no future image/action outcome enters a
control fit. Verify recovered source RGB histories against exported current and
previous contexts, with exact geometry/action/source identities. No CPU resize
enters model data; no model re-encoding occurs.

Keep model and control support denominators explicit; report both full eligible
and common-control-supported comparisons. Report distributions/categories where
available, preserving related/duplicate prefixes and seed-family limits. Baselines
are diagnostic point estimates, not new proper distribution scores. Do not change
any original perception, point, energy, event, false-alarm or presence gate.

Pin protocol/scripts, exact qualified instruments, manifests, source histories and
closed export bytes before execution; rehash afterward. Check algebra/known linear
fixtures and source/control joins. Preserve any original failure and separate
repair. Retain compact findings in docs/results and update PR31. Full-world
forecasting, events, brick memory, late/tunnel/second-wall coverage, untouched
families and fresh confirmation remain required. Select no new GPU experiment
from this protocol alone.

## Fixed implementation details

Last-two control support requires both observed RGB centers, no reset/terminal
between them, and unchanged lives. The history-line control additionally requires
at least four observed centers (including the current center), no reset/terminal
or life change in the sixteen-image observed history. Fit ordinary least squares
at the actual observation indices -15..0; extrapolate its intercept and slope.
Do not clamp an out-of-field extrapolation into the ROI. Missing support stays
missing and counts as a tracking failure in the full-eligible denominator.

The future-linear eligibility check requires a stable last observed transition,
but does not require an entirely linear past. The separate past-and-future slice
also requires finite exact linear references and stability over all sixteen
observations. Preserve overlapping flag counts and a fixed exclusion cascade:
availability, no terminal, last-transition stability, same future lives, finite
references, nonzero velocity, current RGB support, target RGB support, exact line.
Report full-eligible and common-two-control-supported tables. Reproduce all seven
exports' unfiltered published observed/five-horizon counts before interpreting
subsets. Recover all source positions/lives/masks and cached RGB contexts/targets
exactly. No native package, emulator or learner is imported by the analysis.
