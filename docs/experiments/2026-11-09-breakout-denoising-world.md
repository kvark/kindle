# Breakout conditional pixel denoising world model

Select a distinct native pixel generator after the closed
[posterior-history trial](../results/2026-11-09-breakout-world-posterior-history.md).
The [existing DIAMOND review](../results/2026-11-08-breakout-world-pixel-generators-review.md)
and temporary image-history refresh recovery motivate a conditional denoiser,
not another RSSM recurrence modification. Root
`/mnt/data/kindle-breakout-denoising-world-20261109`. The full world-model goal
remains active; actor/critic, CDP comparison and the game campaign stay parked.

Add an explicit `DenoisingWorldLearner` with independent checkpoint format
`kindle-denoising-world-v1`; existing world learners, formats and defaults stay
unchanged. Four RGB frames and four transition commands condition a small
U-Net (16/32/64/64 channels, skip concatenation, existing 5x5 convolution,
per-pixel RMSNorm/SiLU and nearest upsampling). No pretrained weights, RAM,
object labels, teacher features, new backend primitive or learned latent state.
External centered RGB64 and the qualified single native GPU resize stay intact;
internal images use [-1,1]. Actions are spatial one-hot planes.

Use EDM preconditioning with sigma_data .5 and training noise
exp(-1.2+1.2*N), clipped [.002,5]. Uniform pixel MSE trains the preconditioned
residual; do not apply truth-dependent motion weighting to the density objective.
Sampling uses three Euler evaluations, rho7 schedule [5, midpoint,.002,0],
starting at 5*N. Quantize denoised estimates to nearest RGB8 at every evaluation.
The observed-image diagnostic denoises an observed noisy frame at sigma .5,
with an independent RNG from the future sampler. This is **not RSSM posterior
reconstruction** and is labeled separately. No copied-input reconstruction or
fabricated latent features count as a learned pass. Future generation uses
only past generated images and submitted commands after its real prefix.

A separate CNN (16/32/64) with independent 128-unit MLP reward/continuation
heads reads the four-frame/action condition and next image. Train reward two-hot
255-bin CE and ending BCE on real next images; inference reads generated next
images. There are no shared image/event weights. Their summed losses share one
optimizer session. This tests a generator/event architecture together; no
isolated-cause or exact DIAMOND reproduction claim. Official DIAMOND differs
in network size, reward/ending recurrence, offset noise, quantization, sampling
initialization and optimizer/update accumulation.

Use fresh seed41001, trainB32/inferB8, RGB64, LR1e-4, LaProp beta2 .999,
epsilon1e-8, AGC .3 and Auto on the RX7900XT. Each training call receives eight
arrivals: four real initial images and four transitions. After each transition,
perform one optimizer update and shift in its detached denoised estimate from
its noisy realized target. Parameters change between transitions. True reset
images clear that lane; reset transitions have zero loss and no target event.
An all-reset transition skips the update. Checkpoints save weights, moments,
update age and fixed configuration, but omit RNG/history and replay.

Select 2,048 calls, exactly8,189 transition updates/260,807 valid targets, not
an equal-update or equal-data comparison with old RSSM learners. Concatenate four
B8 rows of the retained4,096-row corrective plan into B32; process two cycles.
The first call uses five arrivals for a single checked update before any
remaining update. Every later call has eight arrivals. Each lane uses a deterministic eight-arrival block at offset0/4/8/...24 within
its original32-arrival window, changing offsets in cycle two. Count these new
actual exposures once before allocation:1,416 ending/2,675 positive targets,1,241 masked reset
targets and51,784 unique valid arrivals. Reuse the496 retained natural, launch,
contact, life-loss and ending streams; no gameplay collection or untouched-test
claim. Preserve original input digests. No extension, sweep, best-checkpoint
selection or automatic successor is selected.

Focused CPU checks cover configuration/header rejection before GPU construction,
action/reset alignment, EDM formulas, shapes and separate loss ancestry, and
masked MSE values/derivatives against independent F64. Preserve preparation
failures and corrections. Package/pin matching source/native/runner before GPU.
A bounded host-guarded native scalar-loss value/derivative check precedes a
separately guarded first real update (included in budget). Require finite loss
components, their correct total, all finite saved weights/moments, and changed
denoiser/event families before remaining updates. Stop on real GPU faults or
nonfinite values; no reset/reload. Inspect live service identity before work.

Save actual initial/first/call512/call1024/final checkpoints with exact ages and fixed natural
curves. Run actual-initial and final dense fitting256/development128 origins;
one draw for dense checks. Keep images, rewards, endings and proper distribution
scores, finite losses, exposure/compute counts and timings. Verify all frozen
weights/moments and exact seeded final restore arrays. No old model is queried;
reuse closed control reports only. Native inference remains batch eight.

After CPU review of learning, separately declare final chosen-action assessment
on the retained84 groups: histories16/32, eight draws, horizons1/4/8/16/32.
Retain every physical/event/brick regression and individual false-ending alarm;
mark observed denoising as a different task. All original short-forecast gates
stand, including primary ball energy <=.12712666431732942, eight-step ball
precision and free-flight presence. Sharp images/lower loss cannot discharge
physics, event calibration, persistent removed-brick memory or the full goal.
Late/tunnel/second-wall/above-top coverage, untouched families and fresh learner
confirmation remain required. The current oracle has no above-top endpoint.

Preparation: six CPU checks and347 installed-package checks pass; strict core
Clippy passes. Original struct-pattern compile, flat-NCHW shape and fixture-name/
nonbinary precision failures remain with separate corrections. The original
package extraction hits the full host cache; its subsequent collection fails
because installation did not complete. Corrected extraction uses its own
/mnt/data cache, and the separately logged packaged checks pass. No GPU or
emulator ran during those failures.

The [completed result](../results/2026-11-09-breakout-denoising-world.md) fails
this early check:zero ball detections,primary energy.648302,seven of nine
comparable forecast gates fail. Original launch ordering fails and stays failed;
separate CPU repairs complete technical evidence. The trial is closed;no
extension or successor is selected. Full goal remains unproven.
