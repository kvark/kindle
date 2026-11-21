# Frozen world position-head contract and gradient diagnosis

The measured position-head contract passes. RGB gradients are substantially
larger than the position gradients in the shared encoder of the recorded mixed
batch. This supports testing objective balance before adding another head; it
is not an isolated explanation of failed learning or proof that reweighting helps.
The Breakout world-model goal remains active and unproven.

[Protocol](../experiments/2026-11-09-breakout-world-centroid-contract.md).
Root `/x/Temp/kindle-world-centroid-contract-20261109`.

## What the contract checks establish

- Native affine readback on128 saved development rows per observed/prior1/prior8
  state matches CPU F64: maximum errors0.00004522/0.00004137/0.00003681 pixels.
  The stored head matrix layout and saved-feature calculation are consistent.
- Two full T32/B8 recorded batches, two diagnostic noise pairs: F64 same-time
  centroid losses match native reports. Mixed valid191/256,contact243/256.
  Native observed conditional centroid RMSE2.487/2.819 pixels onmixed and
  3.160/3.288 oncontact. The earlier coarse CPU readout is not a storage-order
  artifact. This is a target-centroid metric,not the ordinary RGB detector gate.
- Joint h/z→xy readback matches CPU affine within9.35e-7 normalized coordinates.
  All sampled categorical blocks are one-hot. Time-major state/observation
  alignment,normalization andmask accounting pass onthese measured batches.
- Four weighted gradient components sum tothe actual joint gradient: relative
  L2 error1.28e-7–2.49e-7 globally,at most1.09e-6 across measured parameter blocks.
  Weighted scalar components also reproduce the joint loss. The extra diagnostic
  affine output does not replace orfeed any training state,loss ortarget.

Source inspection agrees onh-then-flattened-z feature order andincoming-action/
arrival convention. Training uses Gumbel-max andinference inverse-CDF sampling
withthe same mixed probabilities;they use different random draws andtraining
uniform draws exclude1e-6 endpoint tails. This is not a bitwise recurrent-path
parity test orproof ofidentical finite-precision distributions. The experiment
checks native affine execution andthe observed supervision contract,not every
inference/training state orfuture-target alignment.

## Weighted gradients at the retained checkpoint

L2 norm ofobserved+future position gradient divided byRGB gradient. Cosines are
between those position andRGB gradients. All are raw gradients before Adam/AGC;
parameter-block norms do not predict the final optimizer update.

|Batch|Block|Position/RGB,draw0 /draw1|RGB–position cosine,draw0 /draw1|
|---|---|---:|---:|
|mixed|encoder|0.0461 /0.0679|-0.1464 /-0.2467|
|mixed|posterior|0.0587 /0.0541|0.0686 /-0.0458|
|mixed|core|0.1269 /0.1205|0.1454 /0.0680|
|mixed|prior|0.0571 /0.0460|0.0284 /0.0629|
|contact|encoder|0.1850 /0.1902|-0.2942 /-0.3791|
|contact|posterior|0.1868 /0.1936|-0.0210 /0.0092|
|contact|core|0.4474 /0.4578|0.3004 /0.2967|
|contact|prior|0.3217 /0.2941|0.4503 /0.4368|

Onmixed data,the encoder position gradient is4.61–6.79% ofRGB andpartly opposed;
the prior ratio is4.60–5.71%. Position supervision isconnected andnonzero,butis
comparatively weak inthese shared blocks atthis checkpoint. Oncontact clips,
position gradients are larger relative toRGB (encoder18.50–19.02%,core44.74–45.78%).
A fixed large multiplier could therefore change different states unequally.
Observed/future position-head gradients have positive cosines. The decoder has
only RGB gradients;the position head hasonly position gradients as expected.
Ratios withzero RGB norm are undefined andreported asnull,not giant numbers.

Position losses contribute only0.0378–0.0420% ofjoint scalar loss onmixed and
1.010–1.057% oncontact. These scalar fractions are not gradient fractions.
No absence-of-information,capacity-limit oroptimization-cause claim follows.
Two batches/two draws atone reused development checkpoint do not establish a
representative training average,an isolated cause orreliability.

## Execution,failures and retained evidence

One successful terminal guarded stage:20 backwards,three small affine forwards,
zero optimizer steps,new emulator actions orframes. All70 weights remain exact
at the end ofeach offive components;source checkpoint hash unchanged. The
recorded mixed batch isexactly the final learner minibatch withnew diagnostic
noise;contact clips belong tothe older contact learner's fitting set,not this
learner's fitting set. It does not replay the original learner RNG orlearning.

Test-only sourceedbe1d0c,executable SHA256
`4043b8afb54452dfbb5f91703e51db876ce8247e30ca413b7ed33531ecfa945e`.
Qualified learner runtime remains unchanged. CPU synthetic component values/
derivatives andstrict core/test Clippy pass. The initial short-name exact test
filter selectedzero tests;its log remains andis not counted. The separate full
name invocation passesone test. CPU analysisv1 used a tiny denominator for
zero-RGB norm,creating meaningless large head ratios;v2 preservesv1 andreports
null. Allnative stages succeed without repair/retry;none repeats forCPU reporting.

Native measurement384.029s;test harness392.70s includes more wrapper work.
The executable is an unoptimized diagnostic build;these times are not training
orbackend speed comparisons. CPUv2 verifies71 pins. One host-only guard passes
withnine sealed files,no unsealed files,no NVML/reset/reload. Invocation
`9297baf572c34c33b527b7b89f8a6b25`. Own2,234,490,858-byte disposable buildcache
released;native executable,source,raw gradients,states,readouts andlogs remain.

Allprevious RGB gates andnegative learner results remain unchanged. No new
learner,extension,collector orautomatic successor isselected. Full precise
physics,event calibration,removed-brick memory,late/tunnel/second-wall/above-top,
untouched families andfresh confirmation remain open. Actor/critic,CDP comparison
andfive-game work remain parked.

Closure verifies94 pins,one terminal successful service andnine sealed guard files.
