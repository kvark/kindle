# Pong B16 actor/critic gradients

**Select only a bounded4096-action/466-update learning and1024-action frozen
restore check.** No full learning allocation is selected yet. Native and CLI
are unchanged. Keep the existing host guard and300/60-second stage caps; no
retry, cap increase, extension or automatic next allocation.

The completed B16 Pong recipe takes38m45s instead of B8's49m55s, but frozen
score falls14.958→11.875, both24/24 wins and below mean15. Doubled global rates,
shorter imagination and smaller action vocabulary did not improve completed
results. Frozen greedy evaluation also lowers the retained B8 score. In
Breakout, enabling native actor/critic gradients gave a modest unconfirmed
9.542→11.208 gain at nearly unchanged cost. That motivates testing the existing
option on faster B16 Pong; it does not predict a Pong gain or establish
raw-gradient correctness. Reward/continuation/replay-value gradients already
supervise the world; this adds initial imagined policy/value losses through
frozen behavior heads.

Fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B16/T16/H15/R32,
full microbatch16, replay100k, coefficient.01/actor0, original4e-5/6e-6/4e-4
rates, full18/sticky.25/repeat4/native max-pooled frames/one GPU48 resize, no aid.
Only `actor_critic_gradient=false→true` changes versus the completed B16 control.
Expected3,578,225 trainable parameters, first updates376:1/384:2/392:3,
466 updates at4096 actions,119,296 replay positions and zero debt. Keep the
actual initial checkpoint; the1024-action frozen restore has zero updates.

Check every finite report, nonzero initial policy/value losses, selected
encoder/reward weight changes, exact configuration/counters and both guards.
The expected357-tensor schema is derived from the completed Pong control plus
the11 frozen actor copies observed in Breakout. Verify actual keys/shapes/dtypes,
copy consistency, all346 shared initial tensors and all357 frozen tensors.
These checks establish execution only; natural-match competence remains untested.
Same native/GPU/Mesa, but the control used the earlier kernel/module.

CPU preparation passes schema/control-checkpoint, source cadence, command/cap,
compilation and unselected-refusal checks. Source4096 prefix takes16.216s; it
is context, not a full-run speed prediction. Artifacts:
`/x/Temp/kindle-cheap-20261025/pong-b16-actor-critic-smoke`.
Controller SHA256`a194e2e66ef43a7df9162e2882da0e0b30edbb9a296ddfe9bb2a93f25b1964c6`;
queue`43d381d9dbb42a16cc3d3bb14aa97b06e9c9ecb14776241a281c9cad14cf9fd7`;
expected schema`69286fe580d6ccd81ab6f6ad081042010a4eeeb7c540a2d770584c4ad70a9027`;
CPU preparation`e997c86efad4d06c4115a5c5ceed91a3b911ceb65537880acb2ed05733e51f4c`.
Native`bfb8c31`, runner`fbdaee3`, replay`5043b7a`. Owner
`kindle-cheap-pong-b16-acgrads-smoke-20261026`; launch pending.
