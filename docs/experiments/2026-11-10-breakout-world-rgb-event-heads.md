# Decoded RGB features for standalone world-event heads

The [closed wider-motion assessment](../results/2026-11-10-breakout-world-wide-motion-branches.md)
fails ending/false-alarm andjoint-energy gates. Its [saved cases](../results/2026-11-10-breakout-world-wide-motion-error-cases.md)
also show substantial wall/serve image errors. Event heads currently read h/z;
the full local world state additionally carries two past images. This doesnot
prove information absent from h/z oridentify a training cause. Select a bounded
implementation/native-qualification stage for a direct visual readout hypothesis.
Root `/x/Temp/kindle-world-rgb-event-heads-20261110`.

## Intended architecture

Add default-off standalone `rgb_event_heads`. Reward andcontinuation heads read
`concat(h,z,CNN(decoded_current_RGB))`. Reuse the existing jointly learned RGB
encoder weights;add no encoder parameters,object labels,emulator state,pretrained
weight orbackend primitive. Encoder input is the already decoded centered RGB64
image,withno extra resize. Keep input replay RGB3/native max-pooling/one GPU resize.

Observed event losses use the current decoded posterior image;imagined event
losses andinference use that step's own decoded prior image. No actual future
image,target orreset label corrects imagined state. CNN readout doesnot advance
GRU,resample/correct z,orfeed image features into the next transition. Keep the
two-image history,local width16,encoder/RSSM geometry,action placement andlosses.
Event-loss gradients must reach the generated-image decoder andshared encoder.

At1M/RGB64,h/z640 plusencoder256 gives896 head inputs;units64. Expect onlytwo
first-layer head matrices tochange640→896,+32,768 scalars:80 logical weights,
636,683 scalars/240 arrays versus603,915. Derive/verify exact shapes andunique
shared parameter names before native declaration. Activation/buffer cost can rise
despite modest weights;no faster learning orinference claim follows from counts.

Scope categorical local-action/fusion standalone MotionEventRollout,H16,all
world weights,RGB64/1M andqualified physical table. Default-off/legacy paths stay.
Reject whole-agent use andunqualified wide/late/transport/residual/feedback,
Gaussian/mean,previous-frame,auxiliary/teacher/phased/prior-head compositions
before GPU. New explicitv28 binds flag,head geometry andv25/v18 base metadata;
reject legacy disguises/missing flag/wrong bases. No changed-shape restore or
warm start. Existing runtimes,checkpoints andpinned scripts remain immutable.

The [earlier image-feedback failure](../results/2026-11-08-breakout-world-image-feedback.md)
re-encoded generated images through posterior sampling,changed recurrence and
added future event losses together. This prototype changes head inputs only,
keeps raw prior transitions andretains the already-present imagined event losses.
It is not a claim that image access isuntested orthat the prior failure isfixed.
Position-head,color weighting,encoder coordinates,wider decoder andother failed
recipes stay closed. This head experiment cannot byitself prove precise physics.

## Qualification and limits

Use focused CPU checks fordefaults/serialization,unique parameter counts/shapes,
new flag/header/scope rejection,observed/imagined image ancestry,causal/reset
masks andgradient connectivity. Reuse qualified CNN/MLP/normalization primitives;
small F64 reference values/derivatives andlane isolation must exercise the new
image-to-head path. No broad backend parity/optimization project. Strict core/
binding checks andisolated packaged Python API checks;package a separate matching
optimized release native/runner/test executable/source. Preserve any failures.

Pin exact source/native/runner/config/batch before one guarded native stage,
cap1,200s,healthy AMD/Auto,no emulator. Require small-fixture native values/
derivatives,lane isolation,default-off restore matching cached controls,full-shape
finite first update withrequired changed families,exact own initial/frozen arrays
andnew-format trained restore/forecasts. Check head image sensitivity/connectivity
without changing latent transition chronology. Reuse equivalent physical-command
checks withthe persisted table. Keep all counters andactual-initial evidence.

Qualify implementation only. Stop on genuine faults/nonfinite values,no reset,
retry,extension orautomatic successor. Seal closed receipts andrelease disposable
build cache after packaging/terminal checks;retain package/executable/source.
No main learner isallocated. After review,any fresh learning comparison needs
its own fixed recipe/budget/data/initial/forecast/restore declaration andthe
unchanged dense/broader gates. Compare cost andall regressions;one reused seed
isnot reliability. Actor/critic,CDP andfive-game work stay parked.

Full action-conditioned physics,serve/collision behavior,natural event calibration,
persistent brick memory,late/tunnel/second-wall/above-top coverage,untouched
families andfresh confirmation remain required. The world-model goal stays
active andunproven;this hypothesis doesnot narrow orreplace it.
