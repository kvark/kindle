# Fixed Breakout readout with one preceding pixel observation

The current RGB encoder consumes one pooled image; the RSSM carries history.
The completed current-position readout did not outperform the retained actor,
but supplied no motion information. Select one CPU diagnostic of whether
explicit, causal pixel motion improves that fixed readout before considering
a native temporal-input prototype. Keep the five-game goal and Breakout's
original two-wall gate unchanged.

Reuse the original 6e-6 control corpus, completed encoder4e-5 feature capture,
actor probabilities and all saved baseline readout predictions. Keep 2,064 rows
and the original lane split: 954 train on lanes0–3, 509 validation on4–5, 601
test on6–7, including136 near-descending test rows. These are reused development
states, not the latest policy's own trajectories or independent confirmation.

The existing pixel archive contains only stride8 selected observations. One
new CPU replay captures each selected observation's immediate predecessor,
which that archive lacks. Use the unchanged RGB64 color/crop detector and verify
every recorded input hash, action outcome, frame count and reset. Reproduce all
2,064 saved current-position predictions exactly. This is a new motion capture;
do not rerun the original pixel analysis, feature capture, fits or collectors.
Keep missing and extra detections; do not select rows by availability or score.

The new block has exactly ten columns: current ball presence,x/159,y/209,
paddle presence,x/159; ball motion presence,dx/159,dy/209; paddle motion
presence,dx/159. Motion is current minus immediately preceding observed position
in the same natural episode. At episode start or missing either detection,
its presence and differences are zero. Never bridge resets or search backward
for another visible object. Do not include teacher coordinates, commands,
landing targets, phase, future observations or labels as features. Motion is
observed displacement per decision, not physical velocity or fitted dynamics.

Use the previously qualified additive RBF/ridge procedure unchanged. The primary
readout averages equally weighted kernels over the 3,584 learned features and
the ten pixel/motion columns. Training rows alone set normalization, bandwidth
and kernel centering. Validation MSE selects among the same seven penalties
1e-4 through100 per one-hot action target, with no validation refit. Report a
pixel/motion-only readout separately. Reuse saved actor, learned-kernel,
current-pixel-only and learned-plus-current-pixel predictions without refitting.

Require primary agreement at least five percentage points above both the actor
and the saved learned-plus-current-pixel readout, overall and near descending;
also exceed both in each test lane. Report all results, confusion counts and
selected penalties regardless of outcome. No feature, penalty, split, bandwidth
or threshold retry follows a negative result. A positive screen only supports
considering a native prototype; it cannot establish gameplay benefit, optimal
teacher decisions or a causal representation defect.

Before capture, qualify position bounds/missingness, displacement arithmetic,
immediate-predecessor joins and reset boundaries. Pin consumed data, source,
runtime dependencies and this protocol. Use one low-priority CPU core and BLAS
thread, capture cap180s, fit cap60s, controller cap300s. Block the native learning
extension throughout and emulator imports during fitting. Check service state
before/after; no learner or GPU job is selected. Preserve failures and exclusive
output files. Root `/x/Temp/kindle-breakout-pixel-motion-20261103`.
