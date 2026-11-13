# Exact-input ambiguity in the recorded paddle response

The [four-coefficient refit](../results/2026-11-01-cdp-breakout-response-refit.md)
fails its fixed held-out screen. Inspect the saved observations before selecting
another response model. This is descriptive CPU analysis, with no regression,
controller trajectory, native model, emulator or learner update.

Group transitions by exact logged `(paddle_x, observed_displacement,
current_command, previous_command)`, retaining the stationary-edge reset already
represented by those features. Do not round or bin continuous inputs. For each
group, retain the histogram of observed next paddle positions. Report repeated
groups/rows, groups and rows with more than one next position, ranges of at least
four pixels, median/p95 range across repeated groups, and RMS difference across
all unordered within-group pairs. Pair counts weight frequent input states;
this is empirical ambiguity, not a predictive score or a noise-floor estimate.

The primary subsets are the existing episode2 held-out rows and their
near-descending subset. Also report held-out current-position interiors
(`20 < x < 143`), held-out stationary edges and all selected episodes0–2.
Filtering interiors uses current position only; no target-based selection.
Retain every group and source row count. No parameter search, alternative fit,
loss, gameplay trial or automatic successor follows this diagnostic.

Different next positions at identical logged inputs can arise from hidden
emulator state, sticky actions, pooling or pixel measurement. They do not prove
irreducible uncertainty, identify which missing feature matters, or imply that
any other controller would score better. Episodes share teacher and selection
history; observations are not independent samples.

Use one low-priority CPU core, a30-second cap and only the saved37,448 rows.
Record hashes of inputs, code and protocol; verify the original active Qbert
identity and442 input pins before/after. Keep the fit and its rejection unchanged.
Check pair-count/RMS arithmetic on fixed small examples before analysis, then
independently recount saved group histograms. Retain CPU co-host timing.

Root `/x/Temp/kindle-breakout-response-ambiguity-20261101`. No outcome exists at
selection. Do not repeat completed fits, training, replay or qualification.
