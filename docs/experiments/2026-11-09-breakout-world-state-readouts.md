# Frozen local-world state readouts

The [posterior-history RGB100 learner](../results/2026-11-09-breakout-world-posterior-natural100.md)
improves observed reconstruction and paddle prediction but worsens future ball
positions, including fitting data. The earlier trained centroid head and its
contract concern different12M/auxiliary-loss models. Select one new CPU-only
diagnostic on already exported1M states to distinguish a recoverable readout
gap from failures visible in these restricted state readouts. Root
`/x/Temp/kindle-world-state-readouts-20261109`.

## Fixed inputs and scope

Use dense fitting256/development128 exports from exactly three frozen cases:
shared actual initial, retained small64 prior-suffix final16,384 and posterior-
history RGB100 final16,384. Reuse saved first-draw posterior640 and prior640
features at1/4/8/16/32. The current encoder256 output supplies an observed-only
reference. No new model, GPU, emulator, collection or optimizer call. Native
weights, prior states, image decoder and all completed scores remain unchanged.

Verify all physical fields align across the three cases, and all expected
feature shapes/finite values. Targets are the unchanged conservative RGB ball
detector applied to saved current/true future images, in RGB64 pixel coordinates.
Keep its exact visible support and existing reset-valid suffix masks. Do not
use future native position/RAM labels or supply any label to a world model.
No target is manufactured for an absent/ambiguous ball. Presence is not predicted.

## Readout families and selection

For each case, use two predeclared families: standardized affine ridge and RBF
kernel ridge. Include an unregularized intercept by centering fitting targets.
Feature means/scales come only from each fit's training partition; near-constant
feature scales are replaced by1. Linear kernel is XZ^T/feature_width. RBF uses
exp(-mean_squared_distance/(2m)), where m is the median strictly positive
training-pair mean-squared distance; use1 if every distance iszero. This is a
fixed unsupervised bandwidth rule, not a development-tuned parameter.

Choose ridge alpha from[1e-4,1e-3,.01,.1,1] by pooled squared Euclidean error in
four fitting-only folds. Assign whole source-stream lanes to folds using sorted
lane IDs modulo4; the same origin stream cannot enter training and validation.
Recompute normalization, bandwidth and target mean inside every fold. Break
ties by grid order, then fit all supported fitting rows once. Development
targets cannot affect normalization, bandwidth or alpha. Related rows/draws
are not independent episodes; all cases reuse development seeds/origins.

Fit one observed-posterior readout and transfer it unchanged to every prior
horizon. Separately fit horizon-specific readouts using that horizon's supported
fitting prior states/true positions. This distinguishes posterior-to-prior
readout transfer from failure of the selected family on each prior distribution.
Fit the encoder reference only at observed time. No broad family/seed sweep,
neural probe, feature selection or development-based model selection.

Report every family/case/horizon on fitting and development: support, within1px
and2px, Euclidean RMSE/median/90th percentile, out-of-image coordinates, chosen
alpha and fitting CV error. Keep per-origin predictions/errors and folds.
Report the original image detector on the identical support, counting missing
predicted balls as failures. No coordinate clipping or filtering bad outcomes.
Affine/kernel coordinates conditional on a visible true ball do not establish
presence, image coherence, action response, event calibration or usable control.
They cannot replace any original RGB or full-game gate.

## Integrity and interpretation

Keep CPU scripts immutable after declaration, with exact input/helper/source
hashes, configuration and original completed learner receipts. Use meaningful
synthetic CPU checks for primal/dual affine agreement and group-separated CV.
Disable Vulkan and native model imports; cap the single CPU analysis120s.
Stop on structural/nonfinite failures without automatic retry. Declare only
closed inputs; exclude open logs and outputs from their own manifests. Preserve
any original failure before separately declaring a repair. Retain plots and
raw predictions outside Git; publish compact results and update PR31.

A strong readout would demonstrate information accessible to that readout on
these reused recorded-control states, not an isolated decoder cause. Weak
readouts do not prove absence of information or that every possible decoder
fails. Horizon-specific fitting labels and conditional presence make these
diagnostic probes, not replacement agents or learned-world improvements.
No new native qualification, learner, assessment, collector, extension or
automatic successor is selected. The complete world-model goal remains active
and unproven; physics/events, brick memory, full-game coverage, untouched families
and fresh confirmation remain required.
