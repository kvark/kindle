# Late imitation decay lowers frozen Breakout scores

Retain constant expert100 for the compact Breakout recipe. Holding imitation
through100k actions and fading it to zero by150k lowers both final frozen means
at200k. The measured training-time reduction does not offset the gameplay loss.
The five-game goal remains unfinished; no extension,coefficient sweep,seed
confirmation or automatic successor is selected by this result.

| Recipe | Sampled mean | Greedy mean | Two-wall successes | Training time |
|---|---:|---:|---:|---:|
| Retained constant imitation | 25.292 | 42.625 | 0/24 in each mode | 22m27s |
| Hold100k,fade tozero by150k | 17.167 | 25.167 | 0/24 in each mode | 21m30s |
| Own actual initial,both recipes | 1.417 | 2.208 | 0/24 in each mode | — |

Sampled return falls32.13% and greedy40.96%,for4.23% less measured training time.
Own-initial-subtracted return per training minute also falls:sampled1.063→.732,
greedy1.800→1.068. These are descriptive results on reused seed1009 and evaluation
cohort4000001009,not independent confirmation or broad reliability.

## What changed and what it means

The [declared comparison](../experiments/2026-11-01-cdp-breakout-expert-decay-200k.md)
uses12M64/Auto/N8/B16/T16/H15/R16/expert100/encoder2e-5,base/actor4e-5,dynamics4e-4,
warmup1000,actor-unimix.01 and full18 actions. The teacher,20% episode guidance/
cap1920 and all-state corrective labels stay fixed. No prior weights,replay,
games,RNG,belief or bounded-check experience is reused.

The optional native schedule holds100 through update6227,falls tozero at9352,
and remains zero through12477:3,126 zero-weight reports. It applies to actor and
optional world imitation together. Optimizer history and online action guidance
remain,so this does not remove all teacher influence. The implementation and
compressed check were [qualified separately](2026-11-01-cdp-expert-imitation-decay.md).
Native65ed45f/runner differ from the retained447fe50 control;no isolated causal
or hardware-speedup claim is made.

The first100k actions match the control before the fade:12,500 transition
records including actions,rewards,pixel hashes,labels and overrides;6,227 learner
reports,185 episodes and184 resets after excluding timing and requiring candidate
weight100. All357 initial tensors also match. This supports recorded pre-fade
comparability;hidden states,gradients and intermediate parameter tensors were
not captured and compared.

Raw replay cross entropy in the final50k window rises to.824 versus control.591.
The replay/state distributions change,so this is not held-out accuracy or proof
of a forgetting mechanism. The last online50-episode mean is higher77.78 versus
73.32 while frozen gameplay is worse. Online guided scores therefore do not
justify selecting this recipe. Actual overrides are106,630/200,000 (53.315%).
The teacher-disagreement diagnostic and large imitation gradients did not imply
that removing late imitation pressure would improve control.

## Execution and evidence

Fresh training completes200,000 actions/799,518 emulator frames/12,477 updates in
1290.345s. The100k/6227 recovery has357 finite tensors. All12,477 finite reports
have exact scheduled weights; boundary values include100 at6227,99.967995 at6228,
.031999998 at9351 andzero at9352/12477. The separate zero-boundary check confirms
raw actor/world losses remain positive on labeled rows and agree within8e-7
through its captured prefix. CNN,reward,actor and value parameters change.

All four frozen arms use the fixed first3 natural episodes per8 lanes=24 selected
per arm,with guidance/labels off,zero updates and zero truncations. The complete
arms total51,168 actions/204,500 frames/33.755s. Each preserves357 tensors; own
initial sampled/greedy semantic traces reproduce800/795 events. Some vector arms
finish extra episodes before all lanes reach quota; the table uses the declared
24-episode selection,not the unfiltered run-end mean.

All236 pins,full training pixel/action/label replay,frozen outcome replays/videos,
host guards,curves,exposure,comparison and original terminal collector pass.
The original main succeeds02:52:26 UTC November1/source90b1c7e,postprocessing
02:52:59 and collection02:53:02. All five services are terminal success;no GPU
stage repeats or full-run failure occurs. The earlier CPU preparation import/
path failures remain retained. An auxiliary summary initially counted trace
dictionary keys instead of its event field; both summaries and the correction
receipt remain,with authoritative comparison and all GPU stages unchanged.
CPU co-host timing caveats are preserved without wall-time correction.

Evidence root:`/x/Temp/kindle-expert-decay-20261101/full`;
trial`breakout-12m-64-encoder2e5-expert-decay-200k-seed1009`.

- Learning review SHA`17a32bfe4010a12197d4b80047d3b7e85815652dd65c53fbe1eae44945d11275`.
- Comparison SHA`edc2883ac600d79645f33934d56e58fb74a9e2697d77e7eea738cfebf485dc62`.
- Completion SHA`ad2d80f9e7a4e99668fb785aa203a11beff825415deaf25951c8431fbcded8b8`.

The shared checkpoint,initial/frozen tensors,replays,videos and comparison PNG/SVG
remain under that trial. No GPU allocation is active at closure. Keep the retained
constant-imitation learner and the optional schedule disabled by default. This
negative comparison narrows the next learning choice; it does not finish the
five-game goal or identify a single remaining architectural cause.
