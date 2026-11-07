# Breakout: bounded CPU test of sparse-wall aiming

Select one CPU reference-controller comparison. The retained teacher predicts
where to catch the ball but does not inspect remaining brick geometry. The
completed fixed nearer-edge three-pixel aiming controller regressed, and the
outer-entry outcome screen failed. Neither selects a general edge bias. This
new candidate uses the observed remaining wall only when it is sparse; its
benefit is unestablished. Retain those failures and do not repeat their stages.

Keep the original RGB ball/paddle detector, landing estimate, cached descent,
five-step mean paddle response, stationary-edge reset, clipping, action effort,
FIRE fallback and response coefficients. Add a fixed RGB brick-cell heuristic
to choose a small landing-target offset. This is an explicit hand-written
teacher proposal, not a learned world model or a new Kindle score.

## Fixed pixel rule

Use the native RGB rectangle x8:152,y57:93, divided into6 rows by18 columns
of6-by8-pixel cells. A pixel is colored if its largest channel is at least100
and largest-minus-smallest is at least40, calculated without U8 wraparound.
A cell is occupied if at least24 of its48 pixels are colored. Use equal weight
per occupied cell and its horizontal center8+8*column+3.5. This is a pixel
heuristic, not an emulator-state measurement. Small ball fragments should not
satisfy the half-cell threshold; native max-pooling and detection error remain.

The two saved PNGs support the grid geometry (108 and80 detected cells), but
their colors are compressed-video observations, not exact native-palette
oracles. The exploratory exact edge-pixel color match found no occupied cells;
do not use that rejected palette method. Use synthetic fixtures and require
108 occupied cells on every initial native frame before candidate actions.

After the original unbiased landing cache is resolved, steer only when the
ball and paddle are seen, descent is projected,1–36 cells are occupied, landing
is within24–136 inclusive, and the occupied-cell centroid differs from landing
by at least16 pixels. Request ball contact3 pixels toward the centroid by
subtracting that signed offset from the desired paddle center. Otherwise use
the original target exactly. Do not store the biased target in the landing
cache. No response-model refit, uncertainty penalty or angle-response claim.

Record occupied count/centroid, reason, unbiased/biased target, requested offset,
selected plan and the original-cost first command at the same candidate state.
Changed-command counts are conditional comparisons within candidate trajectories,
not paired trajectory effects or causal reward attribution.

## Qualification and fixed allocation

CPU fixtures cover full/empty/partial grids, threshold boundaries, small sprites,
gray pixels, dtype/shape rejection, direction/sign/deadband/interior/phase gates,
unbiased cache handling, invalid/missing objects and disabled-policy equivalence
to the retained mean planner. Tests and command preflight construct zero emulators
and import no Kindle native module. Initial live-frame checks belong to the
single declared collection, before its first action.

One fresh candidate uses eight independent environments, full18/sticky.25/
repeat4/noops0, native max-pooled RGB and no RAM policy inputs. Reuse development
seed4000001009 and its original lane seeds. Select the first three natural
episodes per lane,24 total, retaining excess episodes and partial tails.
Caps:600k actions,300s collection,300s replay,660s controller; one low-priority
CPU core. Retain every action/debug/input-output pixel hash and a lane0 video.
Replay every decision, reward, frame, reset and task outcome. No control rerun.

Compare with the retained225.083 mean/0-of24 two-wall teacher. Require at least
10% higher mean, at least five improved lane means, no fewer two-wall successes,
24 natural selected episodes and zero truncations. Report lane returns, action/
frame totals, collection/replay cost, eligibility/fallback and changed-command
counts, including remaining-cell bands1–12,13–24 and25–36. Independent recount
must reproduce scores, declared gates and decision counts from the saved ledger.

Pin exact scripts, ALE/ROM/adapter, original teacher, response coefficients,
retained control, prior failures and image provenance. Preserve all new failures;
no overwrite, stage repeat, threshold search, extension or automatic successor.
Passing permits only a separately declared fresh-cohort comparison, not native
integration or training. Original864/90% two-wall and five-game gates remain open.

Root `/x/Temp/kindle-breakout-brick-aim-20261106`. Keep retained CDP prediction
scale500 and Breakout143.667 greedy/70.750 sampled. No GPU allocation is selected.
