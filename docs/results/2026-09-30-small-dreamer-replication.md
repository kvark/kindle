# Small Dreamer: numerical replication before learning

The faithful RGB control passes four-update comparisons with pinned upstream
DreamerV3. This closes the numerical prerequisite, **not gameplay replication
or Phase 2**. New gameplay attempts used: **0/10** at this report's publication.
The cancelled 12M study retains its 24 completed runs separately.

## Implementation

The old patch CNN/dense decoder is removed. Its replacement matches upstream's
four convolution/max-pool/channel-RMS/SiLU stages and spatial convolutional
decoder. Size1M has 14,304 encoder and 80,595 decoder parameters. The RGB control
uses a single GPU RGB64 resize with Pillow-equivalent antialiasing and per-axis
byte rounding; CPU work prepares cached geometry coefficients, not pixels.
JEPA's native-detail path is unchanged. Replay stores pixels, re-encoded with
current weights. No old-checkpoint compatibility path is added.

Source inspection also corrected the default categorical actor's uniform mix
to zero: the pinned upstream categorical head does not use its config's nominal
`.01` policy mix. The RSSM retains `.01`. This is a protocol correction, not an
explanation established for the historical score gap.

Meganeura `22c31b94` extends the existing transpose to batched last-two-axis
transpose, with independent CPU/finite-difference and GPU tests. Blade remains
`7cca6377`. September 30 upstream checks identified no newer training fix.

## Evidence

Reference: [DreamerV3 e3f02248](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe).
Native and JAX run serially on RTX 5080, driver 580.178.04, F32. The full fixture
uses Size1M, B2/T4/H3, 18 actions, learner seed103, fixture seed701 and four updates,
with nonzero initial belief, resets, termination and nonterminal truncation.
Common initial weights and explicit random draws isolate numerical differences.
The reference invokes upstream `Agent.loss`, optimizer and slow-target update;
only scan unrolling and random-number supply differ from its model code.

| Check | Result |
| --- | --- |
| Batched transpose values/gradients, ragged tiles | Pass |
| CPU Pillow byte fixtures, seven source shapes; resident/host GPU inputs | Pass; JEPA checks also pass |
| Independent scalar RGB encoder and finite-difference gradients | Pass |
| Four upstream RGB forward/raw-gradient updates | Pass |
| Upstream optimizer on identical RGB raw gradients | Pass; maximum parameter error 1.19e−7 |
| Full learner states, eight losses, raw gradients, optimizer and EMA | **1,524 comparisons pass** |
| Diagnostic raw-gradient capture leaves production execution unchanged | All **1,280 saved tensors**, batches and reports identical |
| Native CPU tests / targeted Python tests / release Clippy / formatting | 93 / 56 pass; Clippy and formatting pass |

Full-learner maximum relative L2 error is 1.03e−6 across losses and 3.24e−4
across raw-gradient tensors. Maximum absolute parameter error is 1.19e−7 and
slow-target error 1.49e−8. [Compact data and tolerances](2026-09-30-small-dreamer-replication.json)
retain separate categories instead of hiding scale behind one maximum.
All eight successful GPU invocations have passing host guards, sealed logs and
reaped workers. No application NVML polling; utilization remains unmeasured.

## Retained diagnostic failures and limits

Initial attempts caught a reserved WGSL identifier, a missing reference-only
Python dependency and raw-gradient instrumentation reading an AGC-clipped
buffer. The corrected diagnostic capture is checked against unchanged complete
production state. The failures remain in the linked run directory.

A stricter independently evolved RGB optimizer test exposed first-step LaProp
sensitivity to a near-zero gradient: one coordinate was +1.89e−6 natively and
−2.33e−6 upstream. Its normalized momentum changes sign, despite matching raw
gradients within tolerance. After four updates, one of 94,899 weights differed
by 5.98e−5. This is not evidence of an optimizer bug: the actual upstream
optimizer matches native parameters and moments when supplied **identical raw
gradients**. Both RGB and full-learner checks therefore compare raw gradients
first and qualify optimizer math on common gradients separately. They do not
claim bitwise-equivalent independently evolving CUDA/Vulkan training.

Synthetic checks do not cover long-horizon learning, replay sampling, environment
collection or competence. No rewards, pretraining or gameplay aids are involved;
there are no learning curves yet. The next step is one bounded paired small-model
learning pilot, then unchanged qualifying pilots plus additional seeds within
the **10-attempt total cap**. Reduced replay ratio/BPTT/capacity defines a new
screening recipe, not an unchanged-learning optimization gain.

Local artifacts: [numerical run directory](../../runs/small-dreamer-replication-20260930.vRE7ag),
[full result](../../runs/small-dreamer-replication-20260930.vRE7ag/step-reference-4907gudr/reference/upstream-result.json),
[full native fixture](../../runs/small-dreamer-replication-20260930.vRE7ag/step-reference-v2),
[isolated optimizer](../../runs/small-dreamer-replication-20260930.vRE7ag/rgb-optimizer-reference/result.json).
