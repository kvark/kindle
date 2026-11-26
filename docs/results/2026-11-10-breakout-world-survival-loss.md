# Cumulative-survival loss: native qualification closed

The optional loss passes bounded native qualification in **42.759904s**.
This proves the added scalar objective, derivative path and checkpoint plumbing
on the declared fixtures. It does not demonstrate improved learned physics,
event calibration or the full Breakout world-model goal.

[Protocol](../experiments/2026-11-10-breakout-world-survival-loss.md).
Root `/x/Temp/kindle-world-survival-loss-20261110`.

`world_survival_scale` defaults absent. With a positive coefficient, the isolated
RGB-event world adds masked mean squared error between its cumulative survival
product and actual survival. Both products remove the existing gamma discount.
Existing per-step continuation/reward/RGB/KL losses remain. The coefficient is10
in this qualifier. Targets and reset masks only enter the loss, never latent
states, generated images or inference. The existing terminal-arrival mask is
retained; reset and later rows are excluded. The report appends one unweighted
survival Brier component, giving14 values. No parameter or inference change.

Scope is1M/RGB64/H16 motion-event/RGB-event heads, all-world weights. v29 wraps
v28, storing the exact positive coefficient and base. Whole-agent uses, invalid
coefficients, horizons, objectives, headers and scopes reject before allocation.
The64-bit product/analytic-logit derivative fixture includes mixed live/terminal/
reset lanes, all-masked cases and both sides of clipping. CPU bounds are1e-7;
native values2e-5 and derivatives2e-6. Labels/masks have zero derivatives. Graph
checks retain causal prior states/images and the existing RGB-gradient route.

Three new Rust CPU tests pass alongside six affected RGB-event checks,29
configuration/five scope checks and strict core/test Clippy. The survival filter
also includes one existing action-probe CPU check. Fourteen corpus tests pass
both from the checkout and against the isolated package. Eight packaged invalid-
constructor/restore fixtures pass with Vulkan disabled, with no GPU/emulator
construction. Existing CNN/RSSM backend math is reused; no new parity project.

The native qualifier verifies every small-fixture logit derivative, then checks
default-off v28 restore against32 cached forecasts and240 source tensors.
Enabled actual-initial240 tensors and32 forecast arrays also match the original
RGB-event qualifier exactly. Four fresh fixture updates yield finite14-component
reports, correct weighted totals and unchanged valid fraction. First unweighted
components agree with the original fixture within the declared comparison bound.
All80 parameters change by update4; the first update is retained separately.
240 frozen and240 restored tensors are exact, as are32 restored forecast arrays.
No new emulator action/frame, main learner or old model requalification occurs.

Source **64324882235fdae3ce66239efe41189ee1ff0455**, native SHA256
**1966cae4ce2080a35407dbf240aa4a275117a39446c25b154dd3a411d21ddc14**.
Wheel SHA256 `9643e027af417c45b8b78d1eb774830bd1d00c5ffd6bef40c670a1836b3fde44`;
stripped validation executable
`9cce693b34e144ae13280300d49c8bc1c169d63da5a0395ed68aa30fa279bbeb`.
The matching isolated runtime/runner is retained under the root.

All143 execution pins/77 closure pins/186 distinct combined pins pass, including
the separately pinned launch. One host guard/nine sealed files pass on AMD/Auto.
Both services are terminal; invocation-specific completion journals verify
the original workers: native7b2f21b1c513441dabb2ebfc8311bc37/PID2286703 and
observer5389b27f33ed4ce796af083f7ca54517/PID2286706. No native/observer failure,
retry, GPU reset/reload or NVML polling occurs. After package/executable and
input hashes verify, release **3,064,246,272 disposable build-cache bytes** and
recheck all186 pins. Runtime, source, wheels, validation executable and failures
remain. The older-log archival choice is separate and still pending.

Preparation failures remain inspectable: the first compile omits the new field
in a legacy posterior-history initializer; the repaired initializer explicitly
sets it absent. Initial checkout Python collection fails because its package is
not on the search path; the separate explicit-path invocation passes. The first
formatter-scope assertion is too strict about reordered modules/trailing commas;
all unrelated formatting is restored from the initially clean source, with the
snapshot retained. A preliminary built wheel remains; review adds an independent
whole-agent guard, followed by the final separate build. That preliminary wheel
never runs on a GPU. These failures are not relabeled successful comparisons.

Select a separately declared [fixed learning comparison](../experiments/2026-11-10-breakout-world-survival-loss-learning.md)
next. This qualifier does not select an extension, coefficient sweep, collector
or automatic assessment. The per-draw Brier objective is not calibration proof
for the latent-marginal distribution; H16 does not certify H32. Diffuse serving
images may persist, and full stochastic physics/events, brick memory, late/
above-top coverage, untouched families and fresh confirmation remain required.
