# Fresh 200k Breakout comparison with late imitation decay

Select one fresh200k-action/12,477-update learner after the
[numerical and bounded qualification](../results/2026-11-01-cdp-expert-imitation-decay.md)
passes. The intended learner-setting change from the retained22m27s control is
constant expert100 to a hold-and-fade schedule. Keep100 through update6227
(100k actions), linearly reach zero at9352 (150k), then stay zero through12477.
The endpoint itself has zero weight:3,126 reports from9352 through12477.
Apply the same weight to actor and optional world imitation. Online teacher
guidance, corrective labels and optimizer history remain; this does not remove
all teacher assistance or establish that imitation caused the remaining gap.

Keep12M64/Auto/N8/B16/T16/H15/R16/expert100/encoder2e-5,base/actor4e-5,
dynamics4e-4,warmup1000,actor-unimix.01,full18 actions,20% episode guidance/cap1920,
all-state corrective labels and expert-world gradients. No centroid or open-loop
loss is enabled. Preserve the native max-pool/one-GPU-resize path. Use qualified
native65ed45f/SHA`cac21ef6949f55c7ef3ff8d5a8a2d34eeb8c13f5106c217f2c289f2fae3b9028`
and its packaged runtime. Native and runner differ from retained447fe50; even
matching initial tensors/traces would not establish an isolated causal effect.

The bounded check completes8,192 actions/489 updates in46.991s,with exact
compressed128→256 scheduling and234 zero-weight reports. Its conservative200k
projection21m02s passes the unchanged30-minute cap. Frozen1,024/.902s haszero
updates and357 invariant tensors;all357 initial tensors,47 pre-update events,
full training replay,416 pins and host guards pass. This establishes execution
and early cost,not gameplay benefit.

No weights,replay,games,RNG,belief or bounded-check experience are reused.
Seed1009/evaluation4000001009 and the evaluation cohort are reused development
inputs,not independent confirmation. Training cap1800s; recovery at100k/6227;
final200k/12,477; first update376 and residual credit.5. Keep1200s CPU-audit,
4500s main-service and5100s postprocessing caps. Never repeat completed GPU
stages to repair CPU evidence.

Assess fixed final and own actual initial in sampled and greedy modes,sampled
primary,first3 natural episodes per8 lanes=24 per arm. Each frozen arm has300s/
600k-action caps,zero updates,guidance/labels off,357 invariant tensors,zero
truncations and the unchanged90% two-wall gate. Retained control sampled25.292/
greedy42.625 and own initials1.417/2.208 all havezero of24 two-wall successes.
Report scores and training time by mode,including own-initial-subtracted frozen
return per training minute. Online guided returns and replay CE are not frozen
competence. No checkpoint selection based on outcomes.

Require357 initial tensors matching bounded check and control,matching pre-update
training events and both actual-initial frozen semantic traces,exact source/
config/counters,finite reports and applied schedule,raw world/actor expert-loss
alignment,changed CNN/reward/actor/value,recovery validation,full training
pixel/action/label replay,frozen outcome replays/videos,guards,curves,exposure
and completed comparison. Ordinary frozen logs lack live pixel hashes; replay
does not compare unavailable live images.

Before launch,CPU preflight exercises all five actual CLI paths,actual audit/
comparison config assertions and negative cases,observer parsers,header readiness,
checkpoint schema,update accounting and schedule endpoints. Pin exact scripts,
runtime,control artifacts and protocol. Stop on real faults or nonfinite reports;
no driver recovery,GPU retry,cap increase,extension,coefficient sweep,seed
confirmation or automatic successor. This is a deliberate new comparison within
the existing unfinished five-game goal.

Root`/x/Temp/kindle-expert-decay-20261101/full`;
service`kindle-breakout-expert-decay-200k-20261101`.

## CPU preparation passes

Five actual CLI paths,30 actual audit/comparison config cases,44 resolved paths,
16 compiled controllers,6 header-readiness cases,4 observer-parser cases,5 recovery
body cases and7 schedule cases pass in.984s. All416 bounded and173 control pins
remain unchanged;357 checkpoint-schema tensors and100k/6227,200k/12,477 accounting
pass. The declared schedule has3,126 zero-weight reports. No native agent or
environment is constructed. Two CPU preparation failures are retained: a converter
was imported as a module export although it is local to the audit function,and
a service-name substitution also changed the historical control path. Both are
corrected before declaration/launch,with zero repeated GPU stages.

## Launched and startup verified

The original service starts02:27:56 UTC November1,source90b1c7e,
PID728595/invocation`2c767c73ae394a30b2ea792b28eb3aff`. Startup at02:28:22 UTC
verifies236 pins,357 initial tensors matching bounded check and retained control,
47 pre-update transitions,exact config/cadence and213 finite reports through3784
actions. Applied imitation weight is100. Main,progress,recovery,postprocessing and
terminal-collection observers are active. The read-only startup audit takes.965s
co-hosted with training; retain that caveat without wall-time correction.

Declaration SHA`e4195829cfee830a49d3c033f7fc39a4dd3bc20f3d3af05d2aa17b9f9cc314c8`;
startup SHA`9ba74c5828bcf72d8ae48d29ea5047f6bc58ff839b48f16fe5d022187dadb08a`.
No frozen gameplay result is available. Do not restart the learner,repeat completed
stages or schedule another GPU job while this original service is active.

## First8192 actions match before the fade

A separate CPU comparison fixes the first8192 full-learner actions,well before
the declared100k fade boundary. All1024 transition records match,including8192
actions,rewards,input pixel hashes,teacher labels and overrides. All489 learner
reports,21 episode records and20 reset records match after excluding only wall
clock fields,learner timing and the candidate's applied-weight field (required100).
The original timing-inclusive comparison remains preserved; its observed
differences are timing fields. Both passes use saved log prefixes,no native model
or environment work. This establishes early recorded equivalence,not equality of
uncaptured hidden state,gradients or parameters and not isolated causality.

The original pass takes0.122s and semantic follow-up0.079s,
co-hosted with the original learner;retain that cost without time correction.
Evidence`/x/Temp/kindle-expert-decay-20261101/prefade-prefix-8192`;
review SHA`b889bc5fcf3917c462b3219c8efab262470587dfc2763b06039dfacc215df13e`;
semantic review SHA`3706f6962ea0d087d7537c51c27d73a39425a42a438549d7f60fae0ce53d5f5f`.
The full learner remains active under its original PID/invocation;no GPU job,
setting or declared input changes. Recovery and frozen gameplay remain pending.

## 100k recovery and the start of fading verified

The original recovery observer succeeds at02:39:01 UTC November1:100,000 actions/
6,227 updates and357 finite tensors. Its complete prefix has103,288 actions/
6,432 finite reports. Read-only confirmation at02:40:14 UTC preserves all236 pins
and the original mainPID728595/invocation`2c767c73ae394a30b2ea792b28eb3aff`,still active.
It checks7,132 scheduled reports through114,472 actions: weight100 at6227/action
99,992;99.967995 at6228/action100,008;99.936005 at6229;99.904 at6230;71.04 at7132.
The recovery observer exits successfully. The confirmation takes.659s CPU;
retain co-host timing without correction. No GPU stage repeats or setting changes.

Recovery review SHA`5f7af233a2d2d4cc2fff23931e5c7050d5ec64d4ab74b77a01ecc51568230e50`;
checkpoint identity`76157e1a0548d9957188aa40a64ac773caf988b6f33898acf6f51908eba72f78`;
confirmation SHA`c499776dfa6513246d6306e24ebce276a4f53acfaaa019f4b1703ed662d83f40`.

The separately selected fixed100k prefix comparison also completes:all12,500
transitions (100,000 actions,rewards,pixel hashes,labels and overrides),6,227
learner reports,185 episodes and184 resets match the retained control after the
same timing exclusions and requiring candidate weight100. Selection precedes
the100k boundary;original timing-inclusive evidence is retained. This extends
recorded equivalence through the final held-weight update. It does not compare
uncaptured hidden state,gradients or intermediate parameter tensors.

CPU comparison1.202s plus semantic review0.998s;
zero native/environment work,co-host caveat retained. Evidence
`/x/Temp/kindle-expert-decay-20261101/prefade-prefix-100k`,with selection
`/x/Temp/kindle-expert-decay-20261101/prefade-100k-selection.json`.
Semantic review SHA`53b6224396c67ff6a265fa901b0816a90ad1f730dc7319d66df14c4c79fdfaf0`.
Training continues toward200k with the original process;zero-weight tail and
all four frozen assessments remain pending. No competence claim or successor.
