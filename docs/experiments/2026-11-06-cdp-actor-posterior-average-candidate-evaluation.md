# Fixed trained-policy assessment of eight-sample actor averaging

The [separate candidate compatibility/cost check](2026-11-06-cdp-actor-posterior-average-candidate-bounded.md)
passes on its first execution:4,096 actions,357 invariant tensors, full replay
and1.018472× observed per-action cost versus the saved ordinary control. Select
one new frozen trained-checkpoint greedy evaluation with eight actor posterior
samples. The original two-arm pipeline and its repair remain failed; no prior
GPU stage, completed audit or replay is repeated.

Use the same retained200k final checkpoint, packagefddf48ba/native960667d1,
matching runner and environment configuration as the bounded candidate: eight
independent environments, seed4000001009, full18/sticky.25/repeat4/noops0,
native max-pooled RGB, one GPU64 resize, sampled live state and greedy actions.
No learning, aids, labels, modal posterior or temperature. The actor averages
eight conditional probability distributions; it does not average values or
maintain eight recurrent histories.

Select the first three natural episodes from each lane,24 total. Require zero
truncations and zero learner updates. Preserve additional episodes/partial tails
but exclude them from the fixed cohort. Cap600,000 aggregate actions and300s
per GPU child, with the existing host guard and2GiB minimum budget headroom.
Do not increase caps, extend, retry or launch a count/seed sweep.

Reuse the retained ordinary greedy evaluation's exact24 episode IDs and full
replay, mean143.666667, all0/24 two-wall. Reuse the completed new-runtime ordinary
prefix to establish compatibility; do not repeat ordinary evaluation. This is a
serving comparison at fixed learned weights, not a new learner, initial-subtracted
learning-efficiency result or training-seed reliability test. No initial-model
arm is selected. The cohort and training seed have been used in development.

Fix the decision rule before execution: favor averaging only if its mean is at
least1.10× ordinary, the95% paired eight-lane bootstrap difference interval has
lower bound>0 (10,000 resamples, NumPy default_rng seed0), its two-wall successes
do not decrease, and elapsed seconds per action are≤2× the saved ordinary
4,096-action control. Report construction separately. The short-control versus
full-candidate timing is descriptive and includes different trajectories; it
does not establish a general inference-speed ratio. Original task gate remains
mean≥864 and≥90% two-wall successes, evaluated separately.

Require exact inference/configuration/native/runner metadata,357 finite invariant
tensors including moments, full action/reward/frame/boundary replay and video,
unchanged source and prior evidence, terminal original process identity and no
remaining render holder. Logs lack live image hashes; do not claim unrecorded
pixel equality. Exercise the actual CLI, metadata/cohort/decision rejections and
corrected guard identity/service checks on CPU before allocation.

Root `/x/Temp/kindle-actor-posterior-average-20261106/candidate-evaluation` retains
exclusive declaration, queue, logs, checkpoints, replay/video and result receipt.
Stop on fault, nonfinite value, invalid evidence or surviving child. No reset,
host recovery, repeated completed stage, learner or automatic successor. A failed
retention criterion leaves ordinary inference and143.667/70.750 retained. The
five-game goal remains unfinished regardless of this experiment's completion.
