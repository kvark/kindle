# Bounded Breakout check with less categorical uniform mixing

Select one fresh8,192-action check of native `unimix` .01→.001. This parameter
mixes the categorical posterior and prior with a uniform distribution. Keep
sampled states, actor-unimix.01 and all retained learning settings. This is not
a repeat of the completed frozen modal-posterior assessment.

The timing audit finds no support for shifting labels. The observed gap between
precise pixel readouts and learned control leaves state noise as one plausible
limitation. Lower mixing is a learning hypothesis, not a diagnosed fix. For16
classes, the mixture's probability floor falls .000625→.0000625. In a mixture
interpretation with32 independent categories, the chance that at least one
uniform branch is selected falls1−.99^32→1−.999^32. These are distributional
identities, not the frequency of changed sampled states or wrong actions.

Native `unimix` also changes prior imagination, KL distributions, straight-through
posterior derivatives and subsequent experience. Preserve a positive floor;
no deterministic-state training, actor-noise change, coefficient sweep or
isolated perceptual claim. No old learner, evaluation or collector is repeated.

## Fixed recipe and execution

Use centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy/value/replay-value1,
discount333/lambda.95,continuation1,dynamicsKL1/representationKL.1/free-nats1,
CDPfuture500,disagreement1,expert100/unit weights/world gradients and all-state
corrective labels. Keep20% episode guidance/cap1920, sampled collection and
posteriors, actor-unimix.01,replay100k/warmup1000,full18/sticky.25/repeat4/noops0.
Preserve native pooled RGB and one GPU64 resize. Fresh seed1009, frozen4000001009;
no transferred weights, replay, live beliefs, games or RNG.

Reuse qualified native447fe50/SHA35ad0db from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Add only the exposed
`--latent-unimix` option to a copy of retained runnercf4ea238 and to the repository
runner. Omission keeps the native default. Before construction, reject nonfinite,
negative, F32-underflowing positive or F32-rounded-to-one values, plus explicit
restore/evaluation overrides. Qualify actual commands and default equivalence
without constructing an agent or emulator. No native rebuild.

Train8,192 actions/489 updates, first update376 and residual credit.5, within180s.
Assess1,024 frozen greedy actions within60s, guidance/labels off and zero updates.
Require357 finite initial/final tensors, exact initial parameter equality with
retained control, changed CNN/reward/continuation/actor/value, correct cadence,
finite weighted reports, full training pixel/action/label replay,357 invariant
frozen tensors and host guards. The pre-update prefix must be fully accounted
for, but equal initial weights do not require identical draws/actions after
mixing changes. Record every differing paired event; do not demand baseline
trajectory equality or relabel differing initial behavior as a failure.

The fixed cost screen is training_seconds×200000/8192×1.1≤1800. This checks
early cost and execution, not gameplay. Replay cap120s,audit180s,main600s,
collector650s. Stop actual GPU faults/nonfinite values; never reset/reload the
device, repeat completed GPU stages or restart an observed live service.

Only after successful execution/cost checks may a separately declared fresh200k
comparison follow:30-minute training cap,100k/6227 recovery,12477 final updates,
final/actual-initial sampled and greedy24-natural-episode assessments. Its own
initial behavior may differ. Greedy is primary; favor a candidate only when
raw greedy return and own-initial gain/min improve with a positive paired
lane-bootstrap gain/min interval. Keep the original864/90%-two-wall gate and
retained143.667/70.750. One reused seed/cohort cannot establish broad reliability.

Root `/x/Temp/kindle-breakout-latent-unimix001-20261104`; service
`kindle-breakout-latent-unimix001-check-20261104`. No full learner or automatic
promotion is selected here. Check live state before launch.
