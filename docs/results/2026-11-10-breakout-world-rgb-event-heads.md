# Decoded-RGB event heads: implementation and CPU checks

The [selected prototype](../experiments/2026-11-10-breakout-world-rgb-event-heads.md)
is implemented and passes focused CPU checks. Native qualification and learning
are **not yet established**. The Breakout world-model goal remains unproven.
Artifacts: `/x/Temp/kindle-world-rgb-event-heads-20261110`.

Reward and continuation read `concat(h,z,CNN(decoded_current_RGB))`. The existing
jointly learned CNN is reused. Observed event losses reuse the current decoder
logits to produce their image; suffix losses use their own decoded prior image.
Inference supplies that same prior image to its head session. No extra decoder
pass, resize, recurrent transition, posterior correction or latent sample is
added. The local decoder and its two-image history remain unchanged.

At 1M/RGB64, the two first head matrices expand from 640 to 896 inputs, with 64
hidden units. CPU inspection confirms **80 unique logical parameters and 636,683
scalars**, versus 603,915 previously: exactly 32,768 added scalars, only
`world.reward.layer0.weight` and `world.continuation.layer0.weight` change shape.
There are no extra encoder parameters. Native saved-array counts, buffer costs,
numerical equivalence and speed remain for qualification.

The default is off. Explicit checkpoint v28 binds the flag, latent/visual/input/
hidden geometry 640/256/896/64 and base v25 (physical action features) or v18 (local
actions). Missing flags/geometry, legacy disguises, wrong bases and phased
training scopes are rejected. Public construction is restricted to 1M/RGB64,
H16 MotionEventRollout, all world weights and the selected plain local decoder.
Whole-agent and unqualified compositions are rejected before GPU construction.

The final CPU checks pass:

| Check | Passing tests |
|---|---:|
| New geometry, headers, scope, ancestry and F64/lane checks | 6 |
| Existing configuration checks | 29 |
| Training scope | 5 |
| Wide/late/action-feature headers and geometry | 3/3/4 |
| Image-history reset/causality | 3 |
| Existing default-off suffix event contract | 1 |

Strict core/all-target and Python-binding Clippy checks pass. Vulkan is disabled
for the final CPU test set. The new ignored native test has not run. Separate
packaged Python API checks and an optimized release package are next; no main
learner is allocated.

Graph ancestry checks show observed events reach decoded posterior images and
the shared encoder. Suffix events reach their generated images; raw prior states
never read those images, future labels, reset weights or later actions. Compiled
event-loss derivatives reach encoder, decoder, dynamics and both event heads;
there is no suffix posterior-correction path. These are wiring checks, not proof
of accurate physics or calibrated events.

## Preserved preparation failures

The initial core build fails on an incorrect conversion to `std::io::Error` in
the new restore guard. An explicit conversion fixes it. The first formatter call
also runs before the newly declared test module exists; the module is then
written. Original build output and a preparation receipt remain.

The initial finite-difference fixture fails for the RGB-logit direction
(.112841 versus .106416), and a second fixture with random image inputs and a
smaller perturbation still fails (-.008021 versus -.007808). Both source fixtures
and logs remain. On the second fixture, the larger perturbations switch two
pooling winners for image inputs and 12 for first-convolution weights. The final
check explicitly requires unchanged pooling winners and a nonzero perturbation,
then compares finite differences with the actual f32-rounded direction. Stable
steps are 1e-4 for image inputs and 1e-6 for first-convolution weights; both head
matrices use 1e-3. All four checks pass at unchanged tolerances, alongside exact
first-lane isolation and second-lane image sensitivity. No backend or production
architecture change is used to repair these fixture checks. This does not
relabel either original failed check as passed.

The next allocation requires a separate declaration pinning the final source,
release package, runner, batch and controls. Its cap remains 1,200 seconds with
Auto precision, a host guard and no emulator. Full-shape finite learning, exact
own initial/frozen/restore tensors, cached default-off controls, physical-command
equivalence and native readout values/derivatives remain unqualified. No main
learning run, extension or automatic successor is selected. All original physics,
event calibration, brick memory, coverage and fresh-confirmation requirements
remain.
