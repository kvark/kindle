# Qbert fits guided movements better than later autonomous movements

The frozen final Qbert actor agrees with 96.92% of teacher movement labels on
forced later-play training states, versus85.56% in wholly unguided episodes
and75.10% after guidance releases. These are the last50k recorded training
actions, evaluated with fixed final weights and their original action histories.
The earlier selected failure-window movement agreement was52.34%; those states
are a different, failure-selected distribution, not a matched generalization test.

| Last50k input-state group | Movement labels | Final greedy agreement | Mean teacher probability | Teacher CE |
|---|---:|---:|---:|---:|
| Before first pyramid, passive |1,435|95.96%|.9370|.1417|
| After first pyramid, forced |3,901|96.92%|.9467|.0982|
| After first pyramid, all passive |7,312|83.79%|.7892|.4826|
| After first pyramid, unguided episode |6,079|85.56%|.8116|.4390|
| After first pyramid, released guided episode |1,233|75.10%|.6787|.6975|

Later passive wait agreement is97.78% on18,308 labels. Overall later passive
agreement is93.79%, again hiding weaker movement decisions. The original
changing training policy matches59.31% of those same7,312 movement labels;
the frozen final actor's83.79% is higher. This comparison changes both model
time and action selection, and uses reconstructed full-history beliefs. It
does not isolate sampling or reproduce original training hidden states.

The preceding.909s CPU recount finds all200k actions already have verified
teacher labels. There are58,487 later-play actions in wholly unguided episodes,
64,600 forced later actions and12,782 native actions after guidance releases.
Only159 actions in wholly unguided episodes reach episode age1920+, versus all
12,782 released actions. In the last50k, movement agreement of the actual
training choices is63.84% in unguided episodes and36.98% after release.
These are experience counts, not sampled replay counts or a causal coverage test.
All25 first-disagreement input hashes lack an exact full-frame match in training;
exact equality says nothing decisive about nearby semantic or belief states.

The result supports testing experience collection with greedy training actions:
greedy frozen play already scores14,028.125 versus11,522.917 sampled and lasts
longer, while the final policy fits its guided paths better than its autonomous
later states. Such a test would change exploration and visited states together;
it could also reduce useful diversity. It is not a demonstrated improvement or
proof that coverage causes the gap. No new loss or automatic learner follows
from this diagnostic; any learner needs its own bounded protocol.

## Execution and validation

The [fixed protocol](../experiments/2026-11-01-cdp-qbert-training-readout.md)
uses the existing447fe50+readout runtimef864fe4, without native changes or a build.
The256-action smoke takes4.708s. The full200,000-action /799,739-frame replay
takes128.729s. All recorded input hashes, commands, rewards, boundaries and
cumulative frames match. Resets per lane are21/22/26/17/22/20/19/24. No new
game score is produced: every action is forced to its original training command.

Each capture preserves357 checkpoint tensors withzero updates. The full capture
passes98 repeated readbacks. At128 opening decisions, F64 actor probabilities
agree with native-logit probabilities within1.101e-8 absolute error;all greedy
choices agree. Smoke/full logits and qualification arrays match exactly.
Counts and original agreement match the independently retained phase recount.
Four CPU tests,1.765s preflight,562 pinned inputs and both host guards pass.

Service`kindle-qbert-training-readout-20261101`,originalPID753330/invocation
`43cdd07c80c947f58b16991ec5d329d4`,succeeds05:56:14 UTC November1 after139.692s,
source5d53dac. The original.702s terminal collector passes. No failure,retry,
repeated GPU stage or learner update occurs. No GPU allocation remains active
at closure;check live state before work. The five-game goal remains unfinished.

Evidence roots: `/x/Temp/kindle-qbert-training-coverage-20261101` and
`/x/Temp/kindle-qbert-training-readout-20261101`.

| Artifact | SHA256 |
|---|---|
| CPU coverage review |`d47ac80d9c0d8cd72a1f89c8e3979b6b307158ccb156de35350ae219e0afe1b3`|
| Readout declaration |`573e69ccc9a8e69295c270105886caca1ba17fea250aede2388f6860c3ac48f6`|
| Readout review |`56b296720defc680da57fead6018845c2511670f39defca5bf70fc4d766b6e47`|
| Analysis |`ee31bed68e62c8d3dbfd42dc0203864afa173465885b5d6c54aac74a78c2e840`|
| Capture |`b4714271dfbfaa068f4f8cedfe049866f1c658ac427a322e11d0a1d18322dee3`|
| Completion |`075001c9986ac9ab1964d7f541d14c0719a47a9bc42f8686faf281152b8e86f4`|
