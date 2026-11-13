# Frozen world position-head contract and gradient diagnosis

The closed RGB1/position1000 learner fails all three broad RGB gates. Its CPU
position readout is coarse even on observed fitting states. Before another loss
or learner, measure whether saved-feature affine readback matches the native
operator, whether observed supervision matches the same-time training states,
and how the actual weighted objectives interact in shared parameters.

Root `/x/Temp/kindle-world-centroid-contract-20261109`. This is a new frozen
measurement, not a rerun of the closed spatial-world gradient experiment.
Use the final centroid checkpoint (8,192 updates), exact RGB64/12M/B8/T32/H16,
Auto, original parameters and scales. No optimizer or checkpoint writes, emulator
actions, collection, actor or critic. Do not change the qualified learner runtime.
The new native executable contains test-only instrumentation on the current
pinned dependency stack; retain its source archive, executable and hashes.

Two recorded batches: the exact last mixed broad training minibatch, and the
existing eight contact clips (fitting data of the old contact learner, not this
learner). Two new declared posterior/suffix noise seed pairs, 93001/94001 and
93003/94003. Reusing inputs is not fresh generalization or the actual old RNG
state. Their full tensors and sources are pinned before execution.

Run one native affine forward on each of 128 saved development observed, prior1
and prior8 feature rows. Compare the stored weight/bias CPU F64 calculation with
native values, maximum absolute tolerance 3e-5 in normalized coordinates.
This tests head storage/operator layout, not the entire recurrent inference path.

Then run five weighted frozen backward components on both batches/two draws:
observed+future RGB, observed position, future position, KL+observed events, and
the original joint loss. Twenty backward calls, zero optimizer steps. Feed
identical sampled inputs to every component. Export all 70 logical gradients,
raw loss reports, and the joint pass's time-major h/z and duplicate affine xy
readout. Preserve all weights exactly after every component and the source file
hash. Duplicate readout uses the existing shared parameter nodes; it does not
replace a loss, target, state or original output.

CPU checks must verify component values/derivatives with unequal synthetic
scales and source compile/strict Clippy. An initial short-name exact filter
selected zero tests; preserve that log and require the full-name single-test
pass separately. Do not count the empty invocation as validation.

After successful terminal native execution, independently declare CPU analysis.
Require native/CPU affine parity; F64 same-time masked centroid loss must match
the actual observed report (rtol2e-4, atol1e-7). Require component losses to sum
to the joint value (rtol2e-4, atol1e-5), and the summed parameter gradient relative
L2 error against joint <=.002. All feeds/gradients/readouts must be finite and
the categorical state blocks must remain one-hot. Report weighted gradient
norms/cosines by encoder, recurrent core, prior, posterior, decoder, event and
position head. Loss magnitudes alone do not establish gradient dominance;
gradient norm/cosine alone does not establish the cause of learning failure or
predict an Adam/AGC update. Keep fitting/development errors and prior RGB gates
unchanged. A contract pass does not prove world-model quality.

Use the existing host-only guard, stop on real GPU/numerical faults, cap this
single stage at 1,800 seconds. No automatic native retry, repair, learner,
extension or successor. Full physics, event calibration, brick memory and
late/tunnel/second-wall/above-top coverage and fresh confirmation remain open.
