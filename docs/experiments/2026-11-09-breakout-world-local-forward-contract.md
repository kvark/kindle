# Frozen local worlds: training and streaming forward comparison

The [completed late-command learner](../results/2026-11-09-breakout-world-late-action-learning.md)
fails its observed-reconstruction gate and retains long-rollout drift. The
earlier [plain12M forward contract](../results/2026-11-09-breakout-world-forward-contract.md)
explicitly excludes spatial decoders. Small decoder fixtures and matching
default-off forecasts do not verify the complete local training/streaming path.
Select this bounded correctness measurement before another learning change.
Root `/x/Temp/kindle-world-local-forward-contract-20261109`.

## Fixed scope

Query only the retained early physical-feature finalv25 and the new late-command
finalv26, both at16,384 updates.1M/RGB64/B8/T32/H16/Auto,80 weights/603,915 scalars/
240 saved arrays. These two local paths have never received this measurement;
do not repeat the old plain-world cases or any completed physical assessment.

Reuse the two sealed mixed/contact T32/B8 batches already prepared at
`/x/Temp/kindle-world-centroid-contract-20261109`, without reconstructing or
collecting data. Keep fixed diagnostic noise pairs95001/96001 and95003/96003:
two models ×two batches ×two pairs =eight cases. Reused data/noise are controls,
not untouched generalization evidence. Feed the persisted physical action table
through the production action_inputs helper, including independent reset masks.

Build each model's actual MotionEventRollout training graph. Test instrumentation
may expose existing h, categorical z, RGB logits, existing generated mean pixels
and loss nodes without adding an operation/parameter or changing original outputs.
Run the same full graph in Training mode with its optimizer cleared and in
Inference mode, with identical weights/inputs/uniforms. Stream the production
observe/transition/decoder graphs with forced training categorical states, causal
real history for observed arrivals and self-generated two-image history for the
suffix. Direct decoder commands must be the same masked incoming features.

Compare full-graph modes, streaming h, categorical Gumbel controls and RGB at
every arrival. Include actual existing suffix mean-pixel nodes separately from
F64 sigmoid-of-logit references, so conversion rounding remains visible. The
streamed suffix starts after arrival15, sees only submitted commands and its
own generated history; no actual future image or outcome may correct it.
Actual observations/terminal/reset labels remain stopped targets or loss masks.

Keep the prior strict continuous tolerances atol2e-4+rtol2e-4 and exact categorical
equality; copy the sealed F64 loss tolerances unchanged. Independently recount
observed/suffix motion BCE and the configured total including future events.
Separate supported suffix targets from reset-crossing/unscored states. Retain
all failures, extrema, raw values and both models; no favorable-case omission.

Common-sample equivalence is not unconstrained prediction, a probability
calibration test or ordinary inverse-CDF inference RNG parity. The Gumbel control
only compares matched-uniform categorical choices. A discrepancy alone does not
establish the cause of inaccurate trajectories; passing would qualify a narrow
implementation property, not full physics or the world-model goal.

## Execution and integrity

Add test-only instrumentation and focused CPU checks for graph/output/parameter
identity, local history/action binding and both decoder configurations. Keep all
production Rust byte-identical to qualified81a5e4e3 except cfg(test) module
registration; do not rebuild/replace the installed runtime or change arithmetic.
Run focused CPU checks and strict core/test Clippy. Package and pin a separate
diagnostic executable/source, settings, inputs, checkpoints and closed receipts
before native execution. Keep executable debug-section stripping explicit.

One host-only guarded native job capped1,800s. Zero optimizer steps, collection,
emulator actions/frames, actor/critic or model modifications. Verify all80 session
parameters after every case, original checkpoint hashes and zero Adam age. Use
the healthy AMD/Auto declaration and stop on real faults/nonfinite values; no
reset, retry, extension, old queue or automatic successor. A numerical contract
failure is retained as a failed contract, not repaired by changing tolerance.

After terminal native success, declare CPU analysis against sealed outputs and
verify the guard. Use immutable input lists; exclude the output being written
and open logs from their own manifests. Preserve every original failure and
separate any CPU repair. Release disposable compiler cache after packaging.

No new learner, physical-branch assessment or collector is selected here.
Full physics/action responses, natural event calibration, changed-brick memory,
late/above-top coverage, untouched families and fresh confirmation remain
required. The full goal stays active and unproven.

## Selected terminal CPU consequence check

The single native job has completed all eight cases in268.267s. Its strict
analysis fails on observed RGB and four categorical blocks, while every forced-
sample recurrent state and actual generated suffix pixel agrees exactly.
Before interpreting these small image differences, select one CPU-only derived
check of the saved paired images. No model or emulator query is allowed.

Reuse the unchanged conservative ball detector from the prior plain-world
consequence check. Include both local models, both batches, both draws and every
observed/suffix frame; do not rerun the prior plain-world cases. Compare observed
F64 sigmoid-of-training-logits with the saved streaming image, and actual native
training-generated suffix means with the saved streaming suffix. Report detection
availability, coordinate differences and within1px decisions, with truth-supported
and reset-valid denominators separate. Retain missing/diffuse cases. This is a
descriptive instrument consequence, not a new physics assessment or a replacement
for any original tolerance/gate. Keep the original strict contract failed.

Also recount exact full-graph values from the saved arrays so within-tolerance
equality is not mislabeled as bitwise equality. Declare immutable inputs before
calculation, preserve the original analysis/closure and exclude the new outputs
and open logs from their own manifests. No automatic native successor is selected.
