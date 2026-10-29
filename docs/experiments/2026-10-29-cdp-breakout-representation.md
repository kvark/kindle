# Breakout representation diagnostic preparation

The retained 12M greedy policy agrees with the pixel teacher on 54.18% of
near-paddle decisions. The [completed phase analysis](../results/2026-10-26-cdp-breakout-pixel-reference.md#larger-frozen-policy-still-disagrees-on-urgent-paddle-commands)
does not distinguish missing visual or motion information from weak policy
learning. Prepare a frozen readout comparison within the five-game goal.
This document selects the measurement design; native capture requires its
own pinned declaration, CPU checks and a bounded check after the active Qbert
comparison and audits have finished. No new learner is selected.

## Fixed states and targets

Use the retained Breakout12M64/B16/R16/expert100 learner's greedy log under
`/x/Temp/kindle-breakout-expert100-capacity-20261029/full-repaired/`
`breakout-12m-64-b16-r16-expert100-200k-seed1009`.
Replay all 20,960 recorded actions and 83,799 emulator frames for both its fixed
final checkpoint and its actual initial checkpoint. Preserve every real reset,
reward, action and boundary. Reuse the matching native447fe50 runtime at
`/x/Temp/kindle-breakout-corrective-20261028/runtime`; no native rebuild.

The existing CPU phase rows supply input pixel hashes and teacher detections.
Check every input hash and recorded transition, including unselected tails.
Select every eighth current-input row in the original first three natural
episodes per lane. Use lanes0–3 for readout training,4–5 for validation and6–7
for testing. This gives954/509/601 examples and220/122/136 near-descending
examples. Neither task score nor target availability selects the rows.

Targets are the teacher detector's ball x/y, paddle x, backward ball dx/dy and
response-target x, in source-frame coordinates. Missing values remain masked;
do not interpolate across unseen balls or reset boundaries. These are existing
teacher estimates, not independent ground truth. The detector's ball crop is
y90..185, so the result does not cover the entire playfield.

## Frozen capture and CPU readouts

Capture the current CNN encoding and posterior recurrent feature before each
selected recorded action. Native inference consumes every intervening frame.
The scalar diagnostic re-observation does not reproduce the original vector
actor's categorical draws; report that distinction. Verify readbacks leave
state, action probabilities and counters unchanged at bounded check points.
No teacher target enters the agent or its replay, and no learner updates occur.
Require all357 checkpoint tensors to remain byte-identical after each capture.

Fit separate linear ridge readouts for each representation/model using the
existing CPU implementation in `kindle._representation_probe`. Choose each
target's penalty from its existing fixed grid using validation error only;
normalization uses training data only, with no validation refit. Report all
targets, selected penalties, support counts, test MAE/RMSE/R² and the training-
mean baseline. Also report near-descending test rows and results by test lane.
Do not choose features, targets or penalties using test outcomes.

CPU preflight must check lane separation, exact feature/target row alignment,
missing targets, boundary handling and the absence of native construction in
readout-only work. A128-action bounded capture must establish finite readouts,
pixel/action replay and tensor invariance before the two full captures. Fix a
120-second bounded cap,300 seconds per full capture and600 seconds for CPU
readouts; no automatic retries, cap increases or new training.

Good decoding would establish accessible information under this readout;
poor linear decoding would not prove that the representation contains none.
Initial and trained features use identical recorded states, but these are
reused development trajectories from a selected learner. Held-out readout lanes
are not independent learner confirmation. A task intervention requires a
separate decision after the complete diagnostic.

Evidence belongs in `/x/Temp/kindle-breakout-representation-20261029`.
`support-review.json` verifies the proposed split against the existing phase
rows while all242 active Qbert pins and its process identity remain unchanged.
It constructs no agent and allocates no GPU work.
The CPU-only `prepare_corpus.py` subsequently aligns every recorded action,
episode and step with those phase rows in.797s. `corpus.json` preserves all
20,960 rows and their targets/input hashes, with SHA256
`eee7ed02bff000a180d18a6295865e09f32da3adf5b252101a61d372e4d0fad0`.
No native module is imported or GPU device opened.
