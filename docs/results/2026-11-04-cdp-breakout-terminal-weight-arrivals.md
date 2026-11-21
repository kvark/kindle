# Terminal-only weighting improves ending predictions with more false ending indications

On the fixed retained-policy cohort, terminal-only weight10 substantially lowers
observed game-ending continuation error: posterior MSE **.644558→.054219**,
paired difference **−.590339**, lane95% interval **[−.718137,−.451811]**. All24
individual terminal errors and all eight lane means improve. Continuation outputs
below.5 rise **5/24→22/24**.

The tradeoff is worse predictions at96 intermediate life losses, where play
continues. Posterior MSE rises **.003178→.099410**, with below-.5 outputs
**0/96→11/96**. One-step prior MSE rises **.001530→.153795**, with below-.5
outputs **0/96→23/96**. This demonstrates changed predictions on the selected
states; it does not establish population calibration or causally explain gameplay.

The completed learner still scores **120.542 greedy/66.375 sampled**, below
retained143.667/70.750, all0/24 two-wall. Keep the existing sampler and
continuation1 with terminal weighting off. No new loss, weight sweep, correction,
learner or automatic successor is selected. The original864/90%-two-wall and
five-game goal remain unfinished.

## Fixed comparison and results

The [protocol](../experiments/2026-11-04-cdp-breakout-terminal-weight-arrivals.md)
reuses the control's [arrival-head capture](2026-11-04-cdp-breakout-arrival-head.md).
The candidate re-observes the exact43,424-action/173,660-frame source trajectory
once with its completed200k/12,477 checkpoint. The120 endpoint identities and
all recorded pixels/actions/rewards/frames/boundaries/resets match. The model
does not choose these actions. Control outputs and the fixed24 game-ending/
96 intermediate-life endpoints are reused, without repeating control execution
or selecting examples from candidate predictions.

Each model uses its own five head tensors and its own prior/posterior features.
Observed terminal targets are0; intermediate-life targets are
F32(1−1/333)=.9969969988. Posterior is the fixed primary; prior results remain
separate. Every interval uses5,000 paired whole-lane resamples, seed20261104.

| Endpoint /state | Control mean continuation | Candidate mean continuation | Control MSE | Candidate MSE | Below.5, control /candidate |
|---|---:|---:|---:|---:|---:|
|Game ending, posterior (24)|.759636|.153422|.644558|.054219|5/24 /22/24|
|Game ending, prior (24)|.952850|.551892|.911006|.423453|0/24 /12/24|
|Intermediate life loss, posterior (96)|.979466|.837008|.003178|.099410|0/96 /11/96|
|Intermediate life loss, prior (96)|.979477|.739154|.001530|.153795|0/96 /23/96|

| Endpoint /state | Candidate−control MSE | Paired lane95% interval | Individual errors lower /higher /tied |
|---|---:|---|---:|
|Game ending, posterior|−.590339|[−.718137,−.451811]|24 /0 /0|
|Game ending, prior|−.487554|[−.639575,−.330826]|24 /0 /0|
|Intermediate life loss, posterior|+.096232|[.055674,.133283]|14 /75 /7|
|Intermediate life loss, prior|+.152265|[.126948,.173387]|8 /84 /4|

The1e-6 error-difference tie tolerance is unchanged. All eight lanes improve for
both terminal states and worsen for both intermediate-life states. These are
reused evaluation-lane intervals, not training-seed reliability estimates. No
pooled metric combines the deliberately different target groups.

Within the candidate, observing the terminal frame lowers MSE.423453→.054219;
22 of24 individual errors improve andtwo worsen. The paired difference is
−.369233,[−.565308,−.191345]. Observing an intermediate-life frame also lowers
candidate error.153795→.099410, but both remain worse than the corresponding
control. Prior/posterior categorical draws differ; this is not a controlled
intervention on the observation alone.

## What this changes in the diagnosis

The targeted loss does change predictions in the intended direction at true
endings. Its weak gameplay result therefore cannot be described simply as an
absence of terminal fitting on this cohort. At the same time, more false ending
indications appear when a life is lost but the game continues. Remaining-life
information, class discrimination, probability levels and the use of continuation
inside imagined behavior remain distinct unresolved questions.

The cohort is selected at life-loss endpoints and contains20% true game endings;
it excludes ordinary continuing frames and the candidate's own failure-state
distribution. Threshold counts do not establish an acting rule, and these errors
do not measure population calibration. Native versions, learned representations,
state draws and experience differ. The diagnostic cannot isolate the head,
transition model or shared gradients as the causal source of weaker control.
The [full gameplay result](2026-11-04-cdp-breakout-terminal-weight.md) remains the
recipe-selection evidence.

## Validation and closure

The one candidate capture takes **50.596s**, with zero updates and all357 tensors
finite and unchanged. All120 source joins and repeated prior queries pass;
queries leave live features, posterior action probabilities and counters unchanged.
Features have2048 deterministic coordinates and32 one-hot categorical groups of16.
Prior/posterior deterministic coordinates match exactly. No checkpoint, live
trajectory, GPU stage or completed control capture is repeated.

The saved F64 head reference agrees with120 native prior outputs to maximum
absolute error **1.605995e−7**, below the fixed1e-4 tolerance. Posterior head values
are offline F64 estimates on exported observed-arrival features, not direct
native posterior readbacks. The candidate runtime is unchanged native3672a21a/
SHAa8b483a3; it differs from retained447fe50. Its only declared configuration
additions are terminal continuation weight10 and the serialized existing
`imagined` replay-value bootstrap default. Terminal replay mixture is absent.

All16 CPU checks pass. Preparation takes2.798s with zero native-agent/emulator
construction; all1,085 pins preserve completed control/candidate evidence and
exact native graph sources. CPU analysis takes.272s on one core at nice10.
Source **ba1ae856df3d0860b173878fd91ffcdb96ccd22f**, original PID1144468/invocation
`2af21b50903a4334ac1255ae42636242`, launches **07:12:04 UTC November4** and
succeeds **07:13:01**. The host guard passes without recovery or unfinished children.

Independent **4.290s CPU closure** completes **07:14:59**, verifying44 artifacts,
all pins,120 endpoint identities, head arithmetic, feature layout, checkpoint
invariance, separate paired errors/intervals/threshold counts and the original
terminal process. No failure, repair, tolerance change or repeated stage occurs.

Root `/x/Temp/kindle-terminal-weight-arrivals-20261104` is closed. No learner or
GPU allocation is active at closure. Check live state before further work; do not
restart the capture or analysis. The source checkpoints and original evidence
remain unchanged.

| Artifact under root | SHA256 |
|---|---|
|`declaration.json`|`c1170decd837c40092b28371b8db31576f8aad193605ab4f341322695df726c6`|
|`capture/result.json`|`4e4b1d2a66e915ae965c0edc46919c25247512499aae6132fbe5ea35622f4b45`|
|`analysis.json`|`8f6f8acf3723f4cb55d106f9216518a4c791ccc993e7c3a247ea372d1741ea3a`|
|`completion.json`|`7abba82c386e13d3be6b24832e7e5542d53ae0799c6bd9d36f89421bfc1e6b3a`|
|`terminal-summary.json`|`74af091399529a8e6e96d797e96809d7da9235ca066061db9ded70fb5570afdb`|
