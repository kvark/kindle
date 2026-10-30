# Frozen state and actor readouts after the encoder-rate improvement

The completed [encoder2e-5 learner](../results/2026-10-30-cdp-breakout-encoder-moderate.md)
improves compact Breakout sampled13.208→25.292 and greedy17.875→42.625 at nearly
unchanged training cost. Select a short frozen diagnostic to measure what
changed in accessible state information and teacher-action agreement before
selecting another learning intervention. No new learner is selected.

Reuse the exact [control representation corpus](2026-10-29-cdp-breakout-representation.md),
its saved control features, pixel reference and fixed linear/RBF readouts.
These are the control's unaided final-greedy trajectories with teacher labels
added offline, not teacher-guided trajectories. Replay20,960 recorded actions/
83,799 frames and verify every input pixel hash, reward, boundary, reset and
frame count. Preserve every-eighth-row sampling and the original first-three-
episodes selection:954/509/601 train/validation/test rows on lanes0–3/4–5/6–7.
No row, feature, target or penalty is selected using the new test results.

Use the fixed encoder2e-5 final checkpoint and qualified native447fe50 runtime,
with no native build or code change. Add an explicit frozen comparison profile
that permits only source encoder6e-6 versus checkpoint2e-5; every other config
field must match. Reject combined profiles. Restore must exactly match the
checkpoint configuration. No teacher target enters the agent or replay.

Run three distinct guarded stages:128-action feature compatibility check under
120s;128-action native actor-probability check under120s using those saved
features; and one complete feature capture under300s. Each must preserve all357
checkpoint tensors and zero learner updates. Require finite readbacks and
unchanged state/probabilities/counters at existing checkpoints. The actor check
requires128 matching greedy choices, maximum CPU/native probability error1e-4,
and exact equality to the16 prior selected feature rows. Reuse the already
qualified control actor check after verifying its checkpoint, native binary,
inputs and unchanged F64 probability/metric functions by exact source. Do not
repeat any prior capture or completed GPU stage.

CPU state analysis uses the existing target list, training-only normalization,
training-derived kernel bandwidth and validation-only ridge selection. Retain
all targets, feature variants, missing-value masks, per-lane and near-descending
results. Compare position on exactly shared pixel-supported rows. CPU actor
analysis applies each saved learned actor to its own recurrent states, reports
teacher agreement/probability/error patterns on all601 test rows and136
near-descending rows, and fits the existing linear/kernel action readouts.
Reproduce retained control metrics exactly. Keep state and actor predictions.

Each CPU analysis has a120s cap; whole service900s. CPU preflight must verify
the actual configs, all command parsers, source/checkpoint/corpus pins, lane
alignment, missing targets and absence of native/emulator imports. Historical
source pins resolve to exact Git bytes with the original declarations intact.
The scoped suite passes69 tests plus two isolated pure regression fixtures,
with zero native/emulator import attempts. No broader native qualification is
claimed; the numerical actor reference and learner runtime are unchanged.

Stop at a failed stage and preserve its evidence; no GPU retry, cap increase,
training extension, fresh policy evaluation or automatic successor. Poor
decoding does not prove information absent; better decoding does not establish
causal policy use. The pixel teacher's crop-limited labels are estimates, not
physical ground truth. Scalar posterior draws differ from original vector
acting, and this reused control-state distribution is not candidate-policy
coverage or independent learner confirmation. Any gameplay gain is supplied
by the completed full comparison, not inferred from these readouts.

Root `/x/Temp/kindle-encoder-representation-20261030`;
service `kindle-encoder-representation-20261030`.

## CPU preparation passes

The71 scoped checks pass with zero native/emulator import attempts. Corrected
preflight takes3.032s: all three actual command parsers, exact config,954/509/601
rows,55 original representation pins,289 actor pins and173 completed-learner
pins verify. Historical source resolution preserves4 representation/4 actor
substitutions from their exact Git commits. The three F64 actor functions and
two numerical constants match the previously qualified source exactly.

The first CPU preflight incorrectly requires`complete` for the pixel-reference
receipt, whose original successful status is`passed`. Its exact source and raw
traceback remain under`preflight-failure`; the corrected reader requires the
original exact status for each artifact. Existing resolved-source files are
reverified without overwrite. No native/emulator import or GPU stage occurs.
