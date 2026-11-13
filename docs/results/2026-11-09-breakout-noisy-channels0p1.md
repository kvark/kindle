# Frozen noisy-residual attenuation regresses prediction

Scaling the three noisy-current residual input channels by .1 in the frozen
reward32 model reduces ordinary ball matches to **15/13/12/17 of110**, versus
35/38/34/33. Actual-history one-step matches fall to **90/91/90/87 of198**, versus
140/149/140/140. Both models pass zero of24 fixed gates. Retain the original gain.

The stage trace is informative: first-stage matches remain115/115/115/115,
versus114/114/107/111, but the next stage falls to90/91/90/87 and the last stage
adds none. This intervention also weakens conditioning on partially generated
residuals at later stages. It does not isolate initial noise or show that a model
trained with lower gain would fail. No training or architecture change is adopted.

Only1,200 coefficients in pixel.denoiser.enc0.weight change;134 other tensors,
all57 event tensors, untouched entries and checkpoint metadata remain exact.
The source checkpoint stays unchanged. Both native stages preserve135 candidate
frozen tensors. All108 traced final batches/864 images and20 normal origin model
arrays agree exactly with candidate outputs; the final stage equals served RGB.
Reward outputs remain effectively unchanged: one causal positive is approximately
.990714 for1. This reused slice has no terminals and cannot establish calibration.

Assessment takes6.538s and trace5.617s, including restore/readback/serialization
on the shared host. There is no build, learner, emulator action or new frame.
All1,376 physical endpoints and2,592 intermediate endpoints are retained, with
16 and27 fixed lossless panels respectively; no threshold or best-case selection.
Root: /mnt/data/kindle-breakout-noisy-channels0p1-20261109.
Native60b9b672/source8dbf9ab5 and trace executable786ab275 are reused unchanged.

The initial observer fails to import breakout_world_model before observing or
auditing because its PYTHONPATH was missing. The unchanged observer is then run
with the correct runtime/runner path against the stopped successful invocation;
the two guards are audited once each. Neither native job is repeated. A separate
CPU pixel-count script initially references train/ rather than query/; its source
and traceback remain, and separately declared pixels_v2.py completes the missing
analysis without new queries. These observer/CPU failures do not qualify any
failed native job: both original native jobs themselves completed successfully.

Full physics/events, natural calibration, changed-brick memory, late/tunnel/
second-wall/above-top coverage, untouched families and fresh confirmation remain
unproven. This frozen sensitivity result does not complete the world-model goal.
Previous negative trials and the fullres protocol failure remain closed.

Closure verifies 218 pins, one terminal successful service and18 sealed guard files.
