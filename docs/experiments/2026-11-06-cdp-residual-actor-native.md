# Guarded native qualification of the residual actor

Select the two new native fixtures after the [CPU qualification and package](../results/2026-11-06-cdp-residual-actor.md)
pass:196 Kindle/5 adapter tests, strict workspace/binding Clippy, and1,125 installed
Python checks/10 deselected with zero native construction attempts. This does
not select an Atari learner or cost check. Earlier CPU fixture failures remain.

Use source7e0bb827 and the exact package under
`/x/Temp/kindle-residual-actor-20261106`. Native SHA256
`3003151fd8134e636f156d82221c4455dcdbc2176d7d138e892bd5dee5bb45e9`;
test executable SHA256
`a0b707a9292a75ca5fe1c38e12b7660804f4b24f5f820eda1ee9d422e6c9c19f`.
The build receipt pins243 source files. No overlay or rebuild is selected.

Run serially on idle RX7900XT/RADV/Auto with the existing host guard:

1. `residual_native_logits_derivatives_and_gradient_routing`: Tiny RGB32/three
   actions, two synthetic feature rows. Check exactly matching zero-correction
   logits, independent F64 affine output and mixed-softmax score/entropy loss
   derivatives for every residual weight and bias. Use one isolated policy and
   one isolated expert backward execution; optimizer and clipping disabled.
   Require zero policy gradient in base/critic, zero expert gradient in residual/
   critic, nonzero permitted derivatives and unchanged parameters/counters.
2. `residual_native_learning_sharing_and_restore`: Tiny RGB32/B2/T4/H3 with two
   synthetic streams,16 transitions each and passive expert labels. Exactly two
   ordinary learner updates must change base/residual/critic parameters with
   finite losses. Verify actor synchronization in scalar/vector acting and
   imagination, base-only world supervision, and exact checkpoint weights,
   moments and counters after restore. Restored replay is empty. No ALE instance.

Each native job has a120s cap; controller300s. Record the original binary/source
identity, queue, declaration, stdout, command receipt and both guard audits.
Require all pins unchanged and both original jobs successful. Stop on real GPU
faults, numerical failures or any assertion; preserve the original failure.
No automatic repetition, reset/reboot, learner, old diagnostic or collector.

Passing establishes these finite fixtures only. A separate8,192-action bounded
declaration must measure cost and actual replay/restore behavior before a full
learning comparison. Retain143.667 greedy/70.750 sampled and the original
864/90%-two-wall gate; the five-game goal remains unfinished.
