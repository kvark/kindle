# Greedy live collection lowers frozen Breakout scores

The [fresh200k comparison](../experiments/2026-11-01-cdp-breakout-greedy-training-200k.md)
finishes200,000 actions /799,639 frames /12,477 updates in1305.866s (21m46s).
Sampled score falls30.64%,56.292→39.042,and greedy falls5.33%,104.667→99.083.
Both remain0/24 two-wall successes. Training takes1.41% longer. All original
audits and independent terminal verification pass;retain sampled collection.

| Fixed assessment | Sampled collection control | Greedy collection candidate |
|---|---:|---:|
| Final sampled mean, primary | 56.292 | 39.042 |
| Final greedy mean | 104.667 | 99.083 |
| Two-wall successes, either mode | 0/24 | 0/24 |
| Actual-initial sampled / greedy | 1.417 /2.208 | 1.417 /2.208 |
| Training wall time | 21m28s | 21m46s |
| Initial-subtracted sampled gain per training minute | 2.557 | 1.729 |
| Initial-subtracted greedy gain per training minute | 4.774 | 4.451 |

Only live action collection changes sampled to greedy. Keep centered CDP12M64/
Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100,actor unimix.01,all18 imitation
weights1,world imitation gradients,all-state corrective labels,20% episode
guidance/cap1920,native447fe50 and exact qualified runneraba4300d. Posterior and
imagined actor sampling remain. The additive runner differs from the sampled
control only by the qualified explicit collection-mode support. Reused seed1009/
evaluation4000001009 is development evidence,not broad reliability. Changing
exploration and visited states together does not isolate a coverage mechanism.

Actual overrides fall99,343→81,725 (49.6715%→40.8625%). Late50k replay expert
CE rises.558097→.623240,while aided online last50 mean rises103.8→125.64 despite
weaker frozen scores. These training measurements do not substitute for unaided
competence or explain the causal mechanism. No extension,cap increase,confirmation
learner or automatic successor follows this negative comparison.

All272 pins,100k/6227 recovery,357 initial tensors matching control/check,357
finite invariant tensors per frozen arm,47 matching bounded pre-update training
transitions and47 expected sampled-control differences pass. Actual-initial
sampled/greedy semantic traces match800/795 events. Full training pixel/action/
teacher-label replay verifies226 completed episodes and81,725 overrides. Four
frozen outcome replays/videos,guards,curves,exposure and comparison pass. Frozen
arms total80,728 actions /322,745 frames /77.631s,zero updates or truncations.
Each assessment selects the first three natural episodes in each of eight lanes;
excess episodes and partial tails remain retained.

All five original services succeed:main08:20:54 UTC November1,postprocessing
08:21:10,collection08:21:42,sourcebd51be9. OriginalPID766156/invocation
`d07a6c26ba914fe9a6032d8ec765b0dd`. Independent recovery confirmation passes
08:06:22 in.843s and terminal confirmation08:22:04 in2.048s;their separate
observer also succeeds. No original/GPU stage or confirmation fails or repeats.
The earlier CPU preparation control-path failure remains preserved. Keep all
CPU co-host timing without correcting training time.

Root `/x/Temp/kindle-breakout-greedy-training-20261101/full`,trial
`breakout-12m-64-b16-r16-encoder4e5-greedy-expert100-200k-seed1009`. All services
are terminal and the host is idle at closure. Do not rerun this learner,collectors
or confirmations. The five-game goal remains unfinished.

| Full-comparison artifact | SHA256 |
|---|---|
| Declaration | `75e284da17a23cc93b346d42d180562080babc03815f909e61b2ff7757cc4b5c` |
| Learning review | `89180da070f5a7d64b7592f5337ab50706d5b3a2b3f88b20c7a1a6b2873b6e98` |
| Comparison | `a5750a29d9c09850b95b3147790da84523dab66b25d8a176e0149a4cfc137441` |
| Original completion | `0cec16ab5cb8754ea9a90a408838acd34fc27c9f5b3deffa8ce48243f3628308` |
| Terminal confirmation | `6cf2e8e5b77dec20259d637eb28e104a1fd2bf5b13dd078c61f8c8b46f5edbac` |
| Confirmation observer | `711ab576a3588108967c2383f631124fa1aa0fe6671bf7cfeef3c5726c343da7` |

## Bounded execution and early cost pass

The [bounded check](../experiments/2026-11-01-cdp-breakout-greedy-training.md)
completes8192 actions /32,740 frames /489 updates in46.633s. Frozen1024 actions /
4076 frames take.874s withzero updates;its eight natural episodes all scorezero.
This establishes execution and early cost,not competence. The conservative
200k projection is1252.339s (20m52s),below the30-minute gate.

Only live action collection changes sampled to greedy in the retained12M64/
Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 recipe. Actor unimix.01,all-state
corrective labels,world imitation gradients,20% episode guidance/cap1920,teacher,
losses,posterior and imagined sampling remain unchanged. Native447fe50 and exact
qualified runneraba4300d are reused without a build. Seed1009/evaluation4000001009
are development inputs,not independent confirmation.

All252 pins,357 matched initial and invariant frozen tensors,489 finite reports,
correct first update376/residual.5,changed CNN/reward/actor/value,full training
pixel/action/teacher-label replay and host guards pass. All47 pre-update semantic
events deliberately differ from sampled control. Training executes3079 overrides
(37.5854%);guided returns are not unaided competence. The original service succeeds
07:51:36 UTC November1 and terminal collection07:51:45,sourceccb4387.
OriginalPID765731/invocation`706343652a1e4e1aac60099e83b198e8`. No original stage
fails or repeats. The prior.648s CPU preflight is retained without repetition.

Root `/x/Temp/kindle-breakout-greedy-training-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder4e5-greedy-expert100-check`. All services are
terminal at closure. Do not rerun this check or collector. No check experience
may be reused. Full gameplay benefit remains unmeasured;any full comparison
requires a separate selection and declaration.

| Artifact | SHA256 |
|---|---|
| Declaration | `5469d5903091f73d2c9276042364db7a265cd33f84c534f06496b81a6373fc3b` |
| Review | `191f7fa6163afe9d667350ded1408280de04af22765da5526f657eb4fe951d9b` |
| Guided replay | `ffe00013c78a9d05d9f26b63b809d58a06fc886b40b911fc01bd776a0b66bce4` |
| Completion | `7c353908b4526a69dddadd812782b57197f8add64863ed3e17192b69b421967c` |
