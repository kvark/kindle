# Reviewed repeated-row qualification follow-up

The [original declaration](2026-10-15-repeated-row-composition.md) stopped in
the original compiler, before candidate execution or timing. Its13 retained
fixtures pass1,567,698 pointwise comparisons and every forward/raw-gradient
array's relative gate. Only the training/inference scalar loss of
129×256×18 fixture0 fails: reference−1.3139604e−6, absolute error1.0132e−10,
relative7.7109e−5 versus2e−5. Independent saved-value reconstruction gives
cancellation condition123,122.6. The strict signed-scalar qualification stays
failed; neither the candidate nor the GPU caused that observed failure.

First execute the previously unstarted original candidate numerical process,
bounded120s, keeping its original assertions. It is expected to stop at the
same scalar gate. Keep its failed guard. A CPU review must verify exactly the
same13 retained cases, all1,567,698 actual values bit-exact against the original,
identical independent checks, no additional error/warning, and no unfinished
child. A different failure stops this follow-up. This is a new reviewed paired
diagnostic, not a passing GPU guard or a retry of the failed control.

Then use separate source copies with only the linear probe shifted by+1.
States, action/shared biases, norm weights, shapes, precision, F64 formulas and
all tolerances stay unchanged. This avoids the near-zero scalar denominator;
it does not repair or erase the signed-probe failure. Sixteen fixtures per arm,
forward-only and raw-backward each. Require all independent checks and exact
original/candidate arrays, and distinct binaries/CPU plan checks. This supports
a scoped view-equivalence claim, not universal baseline F64 compliance.

Two new120s numerical processes. Only after both reviews pass, execute the
previously unstarted two24-update smokes and four ordinary timing jobs from the
original declaration, unchanged native076f3135 versus99f8aab0. Keep its>=3% CDP
benefit/no>3% RGB regression and exact metric/tensor gates. No extra timing pair,
precision/geometry sweep, tolerance relaxation, production promotion or game
actions. Original signed failure, build-cache provenance error and unrelated
regression-test Clippy failures remain. CPU1CPU/2GiB/no swap, serialized guarded
GPU, expected5080/580.178.04/current boot,>=2GiB sampled estimated headroom,
warnings retained, no recovery/NVML polling. Service bounded20min; failures
other than the explicitly diagnosed candidate scalar assertion stop for review.
