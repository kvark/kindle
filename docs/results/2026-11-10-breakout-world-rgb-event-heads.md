# Decoded-RGB event heads: implementation and CPU checks

The [selected prototype](../experiments/2026-11-10-breakout-world-rgb-event-heads.md)
is implemented and passes focused CPU checks. Native qualification also passes; main learning
is **not yet established**. The Breakout world-model goal remains unproven.
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
There are no extra encoder parameters. Native qualification confirms240 saved arrays; main learning cost and useful
forecast improvement remain unmeasured.

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
for the final CPU test set. All77 isolated packaged Python API checks pass, and the optimized release wheel
and stripped validation executable are retained. The native test also passes
under its separate guarded declaration; no main learner is allocated yet.

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

## Native qualification closed

Source45633605d587df73ec4c824f75b2552934d7ac32; release-native SHA256
`f71e9354a7ba420a263ab790e297385df608a76d34cc59174a7be07f4c2a7b8d`.
The separately declared1,200-second stage completes in **45.880 seconds** on
AMD/Auto, with no emulator. Small-fixture native values, all parameter derivatives
and lane/image-sensitivity checks pass against F64. Cached default-off control
forecasts match32 arrays exactly and preserve all240 source tensors.

Two fresh physical-command alias/canonical instances each perform one update.
Their13 reports and240 trained arrays match exactly; required families change,
with77/80 logical weights changed. Three reward-hidden weights retain their
initial values on the first update, consistent with the initially zero reward
output kernel. Both instances share the same240 actual-initial arrays, and all
initial/frozen/trained/restore tensor and forecast checks pass. Savedv28 headers
bind geometry640/256/896/64 andv25/v18 correctly. No main-learner update is used.

All150 execution pins and85 closure pins verify,192 distinct files in their
union. One host guard/nine sealed files pass. Main service invocation
`98236027c76e4394bb5b1e5155cfcc1b`/PID2258259 and observer
`b33f95e94e4443bb8c40dddbebc3ef73`/PID2258315 are inactive/successful;
both completion journals match their recorded invocations and processes.
A later observation incorrectly asserts the observer is still active after it
has completed. That assertion is retained separately; terminal-journal review
verifies the original successful job, without restart. Native qualification and
the observer themselves complete successfully. Original preparation failures
above remain separate.

After terminal verification, release3,133,976,576 bytes of disposable build
cache. Package/native/executable/source full hashes pass before and after;
those artifacts and all receipts remain retained. No runtime overlay is used.

Select the separately scoped [fixed learning comparison](../experiments/2026-11-10-breakout-world-rgb-event-heads-learning.md)
for preparation/declaration. It must start fresh, retain the original dense and
broader gates and report cost/regressions. Qualification does not demonstrate
accurate physics, calibrated events, faster learning or the full world-model goal.
