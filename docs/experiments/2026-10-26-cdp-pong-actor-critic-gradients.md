# Pong B16 actor/critic gradients

**Complete: retain disabled actor/critic gradients.** The fresh500k comparison
finishes in37m52s but scores−0.542/10of24 wins versus11.875/24of24 for the B16
control. All audits and comparison pass; service exits19:36:47 UTC. No extension
or confirmation. See the [completed result](../results/2026-10-25-cdp-cheap-pong.md#b16-actorcritic-gradients-lower-frozen-quality).
The declaration and progress evidence below describe the completed experiment.

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
`kindle-cheap-pong-b16-acgrads-smoke-20261026`; completed as recorded below.


## Bounded learning and restore pass

The check starts18:50:37 UTC/source`19d75a4` and completes4096 actions/16,384
emulator frames/466 updates in16.126s. Construction2.087s is separate. The
1024-action/4096-frame frozen restore takes0.436s with zero updates. Both guards,
all466 finite reports and exact357-tensor schema pass. All466 initial policy
and value loss reports are nonzero; selected CNN/reward weights change. All346
shared initial tensors match the B16 control, and all357 final/frozen tensors
match exactly. All11 frozen actor copies match behavior tensors. Service exits
successfully18:51:00 UTC,23.053s total. This is execution evidence, not competence
or raw-gradient correctness. The older-kernel control's4096 prefix takes16.216s;
do not infer a full-budget speed improvement from this short comparison.

Review SHA256`45e319c37919ef2acca0c7c503126f2ee1c94d371d61feed1d9a2156510f93e7`;
terminal review`95c30132ba2e929afa47c198df2a1e3cf4abb5d13edf37d3157b6b940a43e10c`;
selected declaration`741a7a78c49746b91f4215878e1cdb3e6fc549da61ce61c4a8d92b4516645f2d`.

## Selected fresh fixed-budget learning comparison

After reviewing the check, select fresh1009/eval4000001009 with the same exact
native config. Keep500k actions/62,454 updates,15,988,224 sampled positions,
239,823,360 imagined actor rows, first376 and zero debt. Save100k/200k/300k/400k
recoveries at12,454/24,954/37,454/49,954 updates and fixed500k final at62,454.
Training cap2700s, frozen caps600k actions/300s per arm, audit1200s and total
service4560s. Evaluations sample native actions, with no aid; first3 natural
matches per lane give24 per arm, with excess/tails retained and zero updates/
truncations. Preserve mean15/90%-wins and full357-tensor/config/source/counter/
replay/video/guard checks. No checkpoint selection or extension.

The paired B16 control scores11.875/24wins in38m45s. Stronger B8's14.958/24wins
in49m55s is separate cost/quality context. Both miss mean15. This trial tests
whether the existing gradient option helps the faster recipe, not equivalence
to B8 or five-game reliability. Reused development seed, older-kernel control.

CPU preparation passes full-queue normalization against the B16 control,
actual100k schedule/caps, source compilation, unselected refusal, the observed
357-tensor helper and recovery checker. The helper rejects346-tensor inputs.
The plot-venv reader accepts the completed disabled control, rejects it when
labelled enabled, and compares the bounded357-tensor exports without importing
ALE. Full frozen comparisons therefore use the correct schema and Pong score
keys; existing historical scripts remain unchanged.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-acgrads-intrinsic001-500k`.
CPU preparation SHA256`1d397f9ccdd9a5d731d970e7fb63f9d5c1fad35abbef9e2fa86200e650945f09`;
controller`e127dd965b01cbbeaa5f3ead54e25d2953a957d421c8fef65360b51767723f4f`;
queue`8920aaf32480a1ceabba07c7cef44bb8eb78906898e82d67bea93d7c6bb83f76`;
comparison`7cc9047dfb25af4d3d7896ab1d775f513eff981a93251beb0d10c02502e6f452`;
watcher`fa40a38be18f10c3f02b07aa5f6c6a39cb2afe97d933d88c1fc5ccc87dacba14`.
Owner`kindle-cheap-pong-b16-acgrads-500k-20261026`; active as recorded below.


The fresh run starts18:57:18 UTC/source`7ebd7f4`, invocation
`fc31a824524e42468bc29cef010c86a5`. Startup passes exact config versus the smoke
and only the flag changed versus control,3,578,225 trainable parameters, all357
initial tensors matching the smoke and346 shared tensors matching control.
First updates376:1/384:2/392:3/400:4 are finite with nonzero initial policy/value
losses. Fresh counters and no restored experience are verified. Four recovery
checks, fixed evaluations, complete replays/videos and curves/comparison
monitoring are attached. Frozen quality remains pending.
Selected declaration SHA256`b40d8dc902253a5cf60379410cdb9ba3f3bc7d79a507ef9519ff5731b901be3b`;
selection review`e5eb3c90bdd16b4b0e2395f3b33cdc70ccfb7f6fc77f20fbdcef20689cd492fb`;
startup`a30a913b2cee766c8559867e02ef0cc4bdb8a3e2c387f013bf69ad3b625d9dc0`.

The100k/200k recoveries pass12,454/24,954 updates, all357 finite tensors and
actor-copy consistency. At200k, the complete201,528-action log prefix contains
25,144 finite reports;200,704 progress is907.621s with25,042 updates and zero
debt. Recovery reviews SHA256`3dc619ede216081e0459fc12f57f629a96a4498caf4b685c8409e906597d3904`
and`b5fef7efdd00f144c724782c9f9e76bdaab505f77523d5c9e3cbce756ea7c482`.
Frozen quality is still pending.

The separate progress observer exited143 for an unmeasured reason. Checks at
19:03:55 and19:16:09 UTC confirm the same active training PID24946, invocation
and boot; training did not restart. A replacement observer attached19:04:15,
while the recovery and final-analysis monitors remained live. Original and
replacement receipts are retained as `progress-observer-interruption.json`
and `progress-observer-reattachment.json`. This is not a recorded GPU fault or
training-service failure.

The300k/400k recoveries also pass37,454/49,954 updates, all357 finite tensors
and copy checks. The watcher exits successfully after all four milestones.
At405,504 actions, progress is1,841.062s with50,642 updates and zero debt.
The fixed500k evaluation remains pending; this is recovery evidence only.
