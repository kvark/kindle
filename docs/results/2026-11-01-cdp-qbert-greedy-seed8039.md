# The cheaper Qbert recipe misses its gate on the fresh seed

The [fresh seed8039 check](../experiments/2026-11-01-cdp-qbert-greedy-seed8039.md)
completes200,000 actions /799,796 frames /12,477 updates in1654.009s (27m34s).
Frozen greedy11,376.042 with23/24 first pyramids and sampled8,292.708 with24/24
both miss the unchanged15,000 mean gate. All original audits and terminal
verification pass. This accepts a negative fresh-seed result; the cheaper
greedy-collection recipe passes primary greedy evaluation on one of two learners.

| Fixed assessment | Development4013 | Fresh8039 |
|---|---:|---:|
| Final greedy mean | 15,515.625 | 11,376.042 |
| Greedy first pyramids | 24/24 | 23/24 |
| Final sampled mean | 11,989.583 | 8,292.708 |
| Sampled first pyramids | 24/24 | 24/24 |
| Own-initial greedy / sampled | 80.208 /177.083 | 251.042 /128.125 |
| Own-initial first pyramids | 0 /0 | 0 /0 |
| Training wall time | 27m33s | 27m34s |
| Initial-subtracted greedy gain per training minute | 560.413 | 403.565 |
| Initial-subtracted sampled gain per training minute | 428.876 | 296.174 |

Keep both results. The successful4013 development point remains useful evidence,
but reliable success is unresolved. There is no sampled-training control for8039,
so this is not an isolated replication of the greedy-collection gain. Only learner
and evaluation seeds change:8039/4000008039 versus4013/4000004013. No experience,
weights, belief, game state or RNG is reused. Keep centered CDP12M48/Auto/N8/
B16/T16/H15/R16/encoder4e-5/expert100,greedy live collection,native447fe50,exact
runner,teacher,losses,preprocessing and posterior/imagined sampling.

Relative to4013, greedy return is26.68% lower and sampled30.83% lower at.09%
more observed training time. These are cross-seed differences, not causal effects
of collection mode or speed. Late50k replay expert CE is slightly lower
(.081752 versus.082934),and the aided online last50 mean is higher
(14,123.5 versus13,663),despite lower frozen scores. Actual overrides are
64,021 (32.0105%) versus60,226 (30.113%). Guided curves and replay fit therefore
remain unsuitable substitutes for frozen competence.

All359 input hashes,100k/6227 recovery,357 finite initial tensors,48 recorded
seed-dependent initial differences,357 finite invariant tensors per frozen arm,
full training pixel/action/teacher-label replay, frozen outcome replays/videos,
host guards,curves,coverage,exposure and comparison pass. Initial semantic traces
are retained without cross-seed equality: sampled1139/1363 events and
greedy1101/1468. Each assessment selects the first three natural episodes in
each of eight lanes; excess episodes and partial tails remain retained. Four
frozen arms total121,768 actions /486,909 frames /73.164s,withzero updates and
truncations. The full training replay verifies130 episodes and64,021 overrides.

All five original services succeed. Main finishes07:44:45 UTC November1,
postprocessing07:45:00,collection07:45:02,sourcecbee206. Original main
PID760870 / invocation`a0b219d7702b4c82a65e9b33498c3544`. The independent.878s
terminal confirmation passes07:45:29,verifying original identities,all359 pins,
completion hashes and eight control/candidate videos. No original stage or
verification fails or repeats. Preserve earlier CPU co-host timing without
correcting training wall time.

Root `/x/Temp/kindle-qbert-greedy-seed8039-20261101/full`,trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-200k-seed8039`.
All services are terminal;no allocation remains active at closure. Do not rerun
this learner,collectors,recovery or terminal confirmation. No extension,cap
increase,third Qbert learner or automatic successor is selected. The five-game
goal remains unfinished,including Qbert reliability and Breakout.

| Artifact | SHA256 |
|---|---|
| Declaration | `9130cf1713fd7d26fb96ae257fdc868720e3242c3f3bf61f30bc19858e5d00ab` |
| Learning review | `ba258c63699a049052b3dc1f6bbda54a58a33726f83d909aa1d2ea39dea0f169` |
| Comparison | `b62d32961e25c1048e43fa55097707e18d584930acef9647ca236aac68851ba1` |
| Original completion | `9b02964a14e57bec5d237f88a59dbcfccb54848e3c30ae37895bd2e12dae7cef` |
| Final summary | `95a86337317cf61b555bcb11781ebae23414c73d7d83f9e21aea2401ed04b82a` |
| Terminal confirmation | `c7ce25eaef7ff25c77b189717219a2241f35cdc31f8b57c4b685f83dfc94e82e` |

## Similar total later exposure, fewer long autonomous continuations

The [fixed CPU recount](../experiments/2026-11-01-cdp-qbert-seed8039-coverage.md)
finds that8039 has only1.84% fewer later actions in wholly unguided episodes
overall,but46.12% fewer unguided actions at episode age1920+. The difference
is larger late in training:14.94% fewer later unguided actions and52.32% fewer
unguided age1920+ actions. Longer continuation after teacher guidance increases.

| Training exposure | Development4013 | Fresh8039 |
|---|---:|---:|
| After first pyramid, wholly unguided episodes | 81,530 | 80,031 |
| Unguided episode actions at age1920+ | 5,525 | 2,977 |
| After guidance releases | 13,428 | 14,358 |
| Last50k: after first pyramid, wholly unguided episodes | 31,625 | 26,900 |
| Last50k: unguided episode actions at age1920+ | 4,541 | 2,165 |
| Last50k: after guidance releases | 3,954 | 4,809 |
| Last50k later unguided movement agreement | 73.63% | 71.78% |
| Last50k later released-guidance movement agreement | 47.06% | 43.32% |

These results localize an exposure difference to long autonomous continuations;
they do not show that it caused the frozen score gap. Weaker control can itself
shorten training episodes. Aggregate later exposure is quite similar,and the
learners and frozen evaluation cohorts differ. Episode age is not a verified
game-level identifier. The movement agreement rows describe changing training
policies on their own visited states,not frozen-policy accuracy or teacher
correctness. All200k actions already carry labels;these are experience counts,
not sampled replay counts. The original8039 gate miss remains unchanged.

Only8039 is newly recounted. Exact retained phase-counter,row-reconstruction
and grouping functions are reused;4013's completed counts are reused without
repeating their analysis. All200k actions/799,796 frames,boundaries,partial tails,
teacher-label totals and64,021 overrides reconcile with the verified replay and
independent phase aggregation. The same recipe except seed is checked.

Analysis completes08:00:47 UTC November1,sourceafa6ae6,in.847s;the full command
takes5.579s including verification. All381 diagnostic and272 active Breakout
pins remain unchanged,as does originalPID766156/invocation
`d07a6c26ba914fe9a6032d8ec765b0dd`. No native model,emulator,GPU work,learner update,
failure or retry occurs. Retain CPU co-host timing without correction. Root
`/x/Temp/kindle-qbert-seed8039-coverage-20261101`;complete,do not rerun. No third
Qbert learner,loss change or automatic successor follows this descriptive result.

| Diagnostic artifact | SHA256 |
|---|---|
| Declaration | `4bd5d19c5540065134749177c50de88cb1e2b46df4f4a56defd08fa4a2019352` |
| Review | `e907067dd8ee882bd164b0b393b48138497168b70f94fb3a815838951a2561ad` |
| Completion | `33006ff380154ac816a625de4aff674869d6b10a0890555eaa49c5f671c2cba0` |
| Execution | `def6505fb67611ccec9881dd006ccc9aaed3affd9c43da7cf7267ad5babcd7d0` |
