# Retained Breakout improves substantially between 100k and 200k actions

The retained replay-value1 learner's frozen scores rise **22.125→70.750 sampled**
and **33.250→143.667 greedy** between its100k and200k checkpoints. All eight
lane means improve in each mode. Every arm still has **zero of24 two-wall
successes**; the90%/864-point task gate remains failed.

| Fixed evaluation | Actual initial | 100k /6,227 updates | 200k /12,477 updates |
| --- | ---: | ---: | ---: |
| Sampled, primary |1.417|22.125|70.750|
| Greedy |2.208|33.250|143.667|
| Training time |—|10m29.382s–10m55.956s|21m27.910s|
| Sampled initial-subtracted score/min |—|1.894183–1.974159|3.230038|
| Greedy initial-subtracted score/min |—|2.839369–2.959252|6.590132|

The halfway time is bounded by the original98,304- and102,400-action progress
records surrounding the100k checkpoint. It is not an interpolated or exact
checkpoint time. Original training and co-host costs remain unchanged; the new
52.534s of frozen execution does not add training experience.

The second half improves both absolute score and initial-subtracted score per
training minute, even at either endpoint of the halfway time interval. This is
direct evidence of useful progress across those two checkpoints. It supports
testing more learning work on this recipe before another architecture change.
It does not establish a learning slope at200k, an optimal budget, eventual task
success, or that a larger fresh run will improve. Earlier negative400k/R8,
H5/R32 and old warm-start experiments remain separate evidence: they changed
different settings or used weaker recipes.

## Lane comparison and limits

Each entry is the mean of the first three natural episodes in that lane.
No selected or extra episode is truncated. The saved final and actual-initial
evaluations are reused without another execution or replay.

| Lane | Sampled100k | Sampled200k | Greedy100k | Greedy200k |
| --- | ---: | ---: | ---: | ---: |
|0|12.667|89.667|49.667|150.333|
|1|24.667|51.000|43.333|129.333|
|2|14.667|81.333|21.667|59.667|
|3|24.000|49.667|26.667|105.667|
|4|22.333|78.000|27.000|238.667|
|5|23.000|57.000|34.667|172.667|
|6|19.667|81.000|25.667|113.000|
|7|36.000|78.333|37.333|180.000|

Paired200k-minus100k mean differences are **48.625 sampled** and **110.417 greedy**.
The predeclared2,000-resample lane bootstrap gives descriptive95% intervals
**[36.333,61.042]** and **[80.292,145.875]**. These describe this reused learner
and evaluation cohort; they are not independent-seed intervals or evidence of
broad reliability. The assessment is retrospective and does not change the
original completed200k experiment's evaluation or selection rule. It selects
neither a checkpoint replacement nor a loss change.

Both new arms restore exactly the retained100k/6,227 checkpoint with original
native447fe50/SHA35ad0db and runnercf4ea238. Config, runtime, teacher-free
evaluation, sampled posteriors and full18/sticky.25/repeat4/noops0 protocol are
unchanged. No guidance, labels, learner updates, native build or old-stage repeat
occurs. The source checkpoint retains weights/metadata, not replay, games, live
belief or RNG state; it is used for frozen evaluation only.

## Execution and verification

[Fixed protocol](../experiments/2026-11-03-cdp-breakout-halfway.md).
The17 fixture cases cover wrong checkpoint/mode/update/cap/header rejection,
cohort handling and paired arithmetic. Both actual CLI paths stop before native
or emulator construction. All338 original pins, recovery hashes,357 finite
checkpoint tensors and413 selected pins pass. Preflight finishes
**14:01:49.764 UTC November3**, in3.210s body/3.369s wrapper.

The single service launches **14:02:21.424**, source
`ca6bc06ec7bc47bf96c3052ab3385c016276ed30`, PID1032697/invocation
`c47bd45edffe4af5871bef32e1714dac`. Sampled23,224 actions take31.535s and
greedy27,824 take20.999s: **51,048 actions/204,112 frames/52.534s total**, zero
updates. Each arm preserves all357 tensors exactly and has a finite frozen
export with correct counters. Full recorded-action, reward, frame, boundary and
task-outcome replays, videos and host guards pass. Ordinary frozen logs lack live
pixel hashes; no unavailable live-image equality is claimed.

The original service succeeds **14:03:53.486**. The fixed CPU terminal collector
passes once **14:04:28.611**, in.988s body/1.015s wrapper, verifying the original
journal identity even though the transient unit is unloaded. All413 pins and
all completion-file hashes pass. No preparation, GPU, replay, audit or collector
failure or repeat occurs. Root `/x/Temp/kindle-breakout-halfway-20261103` is
closed; do not repeat its stages. No Kindle allocation remains active at closure.
Breakout and the five-game goal remain unfinished.

The separately selected [fresh400k budget comparison](../experiments/2026-11-03-cdp-breakout-budget400k.md)
uses this evidence; CPU qualification is required before launch. It is not a
resume of either completed life, and the halfway assessment itself launches no
successor.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`1871547566ab040209845801c4c9dbb01cb46ee450398794d42d62df0454f8b6`|
| Comparison |`a8e2033433e37f4ab8dd87266edab1ce06f9a13963f8b86f0957f8903eb017a9`|
| Completion |`61cf85c7bc762932e04d18de1c47e26b575f329bd449d5c3cfc1ae2341a9d4fd`|
