# Breakout fixed-budget return-lambda1 check

The retained replay-value1 critic remains weak on complete sampled-policy
returns. Changing replay bootstrap to posterior values did not improve its
accepted sampled result. Select one distinct existing native setting: return
lambda .95→1, keeping imagined bootstrap and the retained recipe. At1, continuing
traces remove intermediate value mixtures and use all available rewards followed
by the rollout-end bootstrap. Real last/terminal boundaries still stop or cut
the trace according to the existing implementation. This is not complete-episode
Monte Carlo training, removal of all bootstrap bias or an identified bug fix.

The same config field governs imagined actor/value and real-replay targets;
both change. It may increase variance and sensitivity to imagined reward error.
No isolated critic, replay-only or causal mechanism claim follows. The objective
is useful frozen learning at fixed cost, with the original Breakout two-wall
gate and five-game goal unchanged.

Expose `--return-lambda` only for fresh training, accepting finite values in[0,1].
Omission preserves the native default. Reject restore/evaluation overrides before
environment construction. Keep native447fe50/SHA35ad0db and its qualified runtime;
no native build, shader, optimizer, dependency, precision or checkpoint-format
change. The private runner derives from the exact retained control runner with
only this CLI forwarding change; pin the diff. Check boundary forwarding,
default preservation, invalid values, restore rejection and existing target
recurrence semantics before GPU work.

The first allocation is a fresh8,192-action/489-update bounded check, followed
by1,024 frozen actions with zero updates. Keep12M64/Auto/N8/B16/T16/fullBPTT/
microbatch16/H15/R16, encoder/base4e-5,behavior8e-5,dynamics4e-4,replay-value1,
expert100/unit weights/world gradients/all-state corrective labels,sampled
collection/posteriors,actor-unimix.01,action-effects1,replay100k,warmup1000,
20% episode guidance/cap1920,seed1009. Preserve full18/sticky.25/repeat4/noops0,
native max-pooled frames and one GPU64 resize. No prior weights or experience
are reused. Require exact config exceptlambda,357 control-matching initial
tensors,matching pre-update semantics,finite reports,changed CNN/reward/actor/
value tensors,complete guided pixel/action/label replay and host guards.

Use180s training,60s frozen and180s audit caps. Conservative projection is
training seconds×200,000/8,192×1.10 and must fit30minutes. The check establishes
execution and early cost, not competence. Preserve failures, cutoffs and original
service identities; no repeated successful stage, cap increase, broad sweep or
automatic successor. Check current host/service state before launch.

A full learner requires a separate explicit selection after this check closes.
It must retain200k fresh actions,100k recovery and final/actual-initial sampled
andgreedy24-episode assessments against70.750/143.667 and the original gate.
Reused1009/cohort is development evidence, not independent confirmation.
Root `/x/Temp/kindle-breakout-lambda1-20261103`.
