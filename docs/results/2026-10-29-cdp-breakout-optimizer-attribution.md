# Breakout optimizer estimates: encoder clipping and inherited momentum matter

The [native follow-up](2026-10-30-cdp-breakout-native-optimizer.md) measures similar
group movement, but its full comparison remains protocol-failed on six scalar
F64 rounding allowances. The source corpus is unaided greedy evaluation with
offline teacher labels; earlier prose calling it teacher-guided was incorrect.

A CPU calculation through the saved AGC/LaProp state changes the picture from
[raw gradients](2026-10-29-cdp-breakout-gradient-attribution.md). All 12 encoder
parameter tensors are clipped on both fixed batches. Their aggregate gradient
norm falls from 4,431/17,331 to 6.118 in each case. With the configured learning
rates and saved moments, the estimated encoder update is 0.00212–0.00234% of its
weight norm, versus 0.1449–0.1490% for recurrent dynamics. This makes the encoder's
small parameter movement a concrete question to investigate; it does not prove
that the chosen learning rate or clipping is responsible for weak play.

Saved momentum contributes substantially. Total estimated update alignment with
the current imitation contribution is only 0.299–0.784 across encoder/posterior/
dynamics and both batches, versus 0.925–0.997 for the earlier raw-gradient
comparison. Those are different spaces: clipping, preconditioning, learning-rate
multipliers and inherited momentum all intervene. Momentum may usefully smooth
batch noise; this result is not an optimizer defect or a reason by itself to
remove it.

The calculation completes in 4.150s on CPU, source `46534b7`, and terminal review
finishes 23:53:36 UTC October 29. All 191 pins, both complete 86-parameter batches,
raw-matrix reproduction and transformed-component checks pass. Twelve CPU tests
pass; there are zero native import attempts, GPU work or learner updates. No new
learner or optimizer setting is selected. Native parameter movement remains
unmeasured; the five-game goal is unfinished.

## What was calculated

The [fixed protocol](../experiments/2026-10-29-cdp-breakout-optimizer-attribution.md)
uses the same two B16/T16 batches as the qualified gradient capture, the accepted
200k-action control checkpoint and its saved optimizer step 12,477. Every batch
starts independently from the same weights and moments. All 86 trainable world
parameters and 14,688,256 elements are included, with complete first/second moments.
The previous 139 pins and all 37 native capture artifacts are checked; all 357
checkpoint tensors remain protected by their original hashes.

The pinned Meganeura `423d373` shader/runtime applies per-parameter AGC before
LaProp. Kindle uses beta1=0.9, beta2=0.999, epsilon=1e-20, AGC=0.3 and minimum
parameter norm=0.001. Warmup has finished. Effective learning rates are 6e-6 for
the encoder, 4e-4 for dynamics/posterior/prediction/disagreement, and 4e-5 for
other heads. Weight decay is zero and clipping runs every update. The CPU
calculation uses F64 arrays with the corresponding F32 configuration coefficients.

For each tensor, let `g` be the complete raw gradient and `g_k` one already
weighted loss component. The estimate follows:

```
s = min(1, AGC * max(pmin, norm(weight)) / norm(g))
g_clipped = s * g
v_next = beta2 * v_saved + (1 - beta2) * g_clipped²
denom = sqrt(v_next / (1 - beta2^next_step)) + epsilon
history = -lr * beta1 * m_saved / (1 - beta1^next_step)
current = -lr * (1 - beta1) * g_clipped / denom / (1 - beta1^next_step)
estimated_update = history + current
```

Zero-gradient and zero-bound cases follow the pinned shader's explicit guards.
Each component uses the *same total-gradient* `s` and `denom` when replacing `g`
with `g_k` in the current term. These conditional contributions reconstruct the
current estimate; they are not separate optimizer runs. Changing a loss would
also change `s`, `denom`, future moments and trajectories. Norms are not additive
percentages of the total update.

## Clipping is strongest in the encoder

|Batch|Group|Clipped tensors|Median AGC factor|Raw total norm|Clipped total norm|
|---|---|---:|---:|---:|---:|
|0|encoder|12/12|0.0012685|4431.24|6.1175|
|0|posterior|2/5|1|75.7716|17.0269|
|0|dynamics|13/22|0.726728|130.174|35.784|
|0|other|4/47|1|141.81|27.489|
|1|encoder|12/12|0.000325356|17331.4|6.1175|
|1|posterior|4/5|0.721031|181.556|17.5838|
|1|dynamics|12/22|0.665589|151.525|35.757|
|1|other|4/47|1|129.068|27.5437|

Encoder factors range from 6.881e-6 to 0.1561 on batch 0, and 1.690e-6 to 0.04884
on batch 1. All encoder tensors are at their norm bound. Uniformly scaling an
entire already-clipped tensor's gradient upward would leave its clipped vector
unchanged at these fixed weights. Increasing only the imitation coefficient can
also change direction; this calculation does not predict that counterfactual.

The encoder's learning rate is about 66.7 times smaller than the recurrent
model's. Relative update norms below are consistent with much slower encoder
movement on these states, but different parameter groups need not benefit from
equal relative movement. A [previous higher encoder rate](2026-10-26-cdp-breakout-pixel-reference.md#higher-encoder-rate-lowers-frozen-breakout-quality)
on the smaller, weaker-imitation recipe did not improve Breakout; this is not evidence for
repeating that intervention without a new, explicit comparison.

## Inherited momentum changes the total direction

|Batch|Group|Full update norm|History norm|Current norm|Full update / weight norm|Full–history cosine|Full–current-expert cosine|
|---|---|---:|---:|---:|---:|---:|---:|
|0|encoder|0.000431601|0.000325374|0.000274045|0.002117%|0.7728|0.5184|
|0|posterior|0.0852171|0.0741483|0.0452259|0.144745%|0.8478|0.4423|
|0|dynamics|0.232056|0.204997|0.102766|0.149005%|0.8967|0.3092|
|0|other|0.277389|0.221719|0.165474|0.179396%|0.8026|undefined|
|1|encoder|0.000477005|0.000325374|0.000251139|0.002339%|0.8709|0.7843|
|1|posterior|0.0877869|0.0741483|0.0415359|0.149110%|0.8818|0.5425|
|1|dynamics|0.225653|0.204997|0.0904276|0.144894%|0.9162|0.2992|
|1|other|0.327064|0.221719|0.234103|0.211522%|0.6986|undefined|

History is identical between the two estimates because neither estimate is
applied. The full update's cosine with history is 0.773–0.916 in the three
representation groups. These values cannot establish whether momentum helps or
hurts the next action or later learning. `Other` includes several distinct heads;
its imitation contribution is zero because the world loss uses frozen actor
weights.

For completeness, the conditional current-contribution norms are:

|Batch|Group|CDP|Dynamics KL|Representation KL|Reward|Continuation|Replay value|Disagreement|Expert|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|encoder|0.000112971|1.83927e-05|6.47202e-06|3.06475e-07|4.40625e-08|1.01812e-06|0|0.000240019|
|0|posterior|0.0158399|0.00281678|0.00130308|6.15325e-05|1.46708e-05|0.000203222|0|0.0412116|
|0|dynamics|0.0506071|0.0513415|0.000584843|7.58173e-05|1.32874e-05|0.000729593|0|0.0715443|
|0|other|0.076311|0|0|0.00196026|0.00222725|0|0.146798|0|
|1|encoder|3.37824e-05|6.33469e-06|1.1017e-06|1.08961e-07|1.82017e-09|1.44897e-06|0|0.000234262|
|1|posterior|0.0078872|0.00128813|0.000543741|3.78546e-05|3.84879e-07|0.000355144|0|0.0381588|
|1|dynamics|0.034523|0.0455585|0.000409488|7.4236e-05|5.46854e-07|0.00131026|0|0.0690689|
|1|other|0.0812996|0|0|0.00249905|0.000231746|0|0.219518|0|

Imitation remains larger than CDP in each representation group. Preconditioning
also makes dynamics KL comparable to CDP in the recurrent group, despite its
smaller raw norm. Disagreement contributes only to its own trainable heads here;
a large pooled head-update norm does not show a direct encoder gradient.

## Validation and limits

All per-parameter and group raw dot-product matrices reproduce the native
capture's F64 accumulation within rtol 1e-9 / atol 1e-8. The transformed component
sum matches the current estimate with maximum group relative L2 error 1.153e-6
and maximum absolute error 8.275e-10, within the declared 1e-3 bounds.
The history/current sum reconstructs the full F64 estimate exactly. All arrays
are finite; none of the saved variance entries is zero or F32-subnormal.

Twelve tests cover first-step bias correction, nonzero saved momentum, cancellation
before clipping, per-tensor bounds, zero gradients, parameter-prefix rates,
warmup, invalid state, input invariance and undefined zero-vector cosines. Both
the initial and final scoped runs pass; no analysis failure or retry occurs.
The CPU import guard records zero native/JAX/PyTorch import attempts. All 191
input pins are rechecked after analysis and during terminal collection.

This is a mathematical estimate, not measured native movement. GPU reduction
order, F32 rounding, shader contraction and final parameter-write rounding are
not emulated. Both reused batches come from the control's unaided final-greedy evaluation,
with corrective teacher labels added offline. Earlier training, broader autonomous
state coverage, other learners and gameplay effects remain unmeasured. A separate native check
on disposable copies can qualify these estimates before an optimizer intervention;
no training extension or coefficient sweep follows from this report.

Evidence: `/x/Temp/kindle-optimizer-attribution-20261029`.

- `analysis/batch-0.json` and `analysis/batch-1.json` retain every parameter,
  clipping factor, update-to-weight ratio and complete raw/clipped/update matrix.
- `analysis/result.json` SHA256
  `7eebad23939ab734f74692ae6d08feaf443f471a8ed28e507b5593c3c95005b4`.
- `declaration.json` pins the prior evidence, backend source and analysis inputs.
- `completion.json` records 191 unchanged pins, 86 parameters per batch,
  CPU-only execution and the explicit absence of native optimizer validation.

No GPU allocation is active at terminal review. Check live state before later
GPU work. The accepted learner and packaged native runtime remain unchanged.
