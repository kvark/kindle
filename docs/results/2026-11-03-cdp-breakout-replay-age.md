# Breakout replay starts commonly use contexts hundreds of updates old

The CPU reconstruction of the retained70.750 sampled/143.667 greedy Breakout
learner finds substantial age in the cached recurrent context. Over the final
50k actions, contexts not masked by an immediate reset have **median246**,
**mean365.099**, **p90 862** and **p99 1,776** learner updates since their last
computation. **6.75%** are at least1,024 updates old. This measures cache age;
it does not measure latent error or establish the cause of weak gameplay.

All12,477 logged updates match reconstructed scheduler cadence,replay length,
labeled-row count and weight sum. All199,632 sampled sequence starts are
included;310 have their initial context masked by a first-frame reset.
The analysis takes **.387s CPU**, with no learner,emulator or GPU work.
Retain the current control. No buffer change,burn-in,loss change or learner is
selected by this diagnostic.

| Context population, entire run | Count | Mean age | Median | p90 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All sampled starts | 199,632 | 269.209 | 154 | 681 | 1,523 | 6,006 |
| Initial context not reset-masked | 199,322 | 269.233 | 154 | 681 | 1,523 | 6,006 |

Age is measured in completed learner updates, not emulator frames or seconds.
Quantiles use nearest rank. The fixed action windows give:

| Actions | All / unmasked contexts | Mean age, all / unmasked | Median, all / unmasked | p90, all / unmasked | p99, all / unmasked | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–50k | 49,632 / 49,512 | 89.265 / 89.271 | 47 / 47 | 230 / 230 | 532 / 531 | 3,052 |
| 50k–100k | 50,000 / 49,922 | 259.257 / 259.205 | 176 / 176 | 613 / 613 | 1,207 / 1,206 | 6,006 |
| 100k–150k | 50,000 / 49,936 | 361.732 / 361.796 | 244 / 244 | 861 / 862 | 1,726 / 1,726 | 3,952 |
| 150k–200k | 50,000 / 49,952 | 365.257 / 365.099 | 246 / 246 | 863 / 862 | 1,778 / 1,776 | 3,745 |

| Age bin | All contexts | Unmasked contexts |
| --- | ---: | ---: |
| 0 | 16 | 16 |
| 1–15 | 28,293 | 28,236 |
| 16–63 | 32,827 | 32,763 |
| 64–255 | 67,093 | 66,998 |
| 256–1,023 | 63,630 | 63,548 |
| 1,024+ | 7,773 | 7,761 |

The last computation is a replay refresh for187,439 contexts (**93.89%**) and
collection for12,193. The corresponding unmasked counts are187,149 and12,173.
Of2,989,405 labeled actor replay rows,2,949,392 (**98.66%**) occur before a reset
cuts their sequence's dependence on its initial context. This is potential
structural dependence, not a measured effect size. Per-window provenance,
bins and affected-row counts are retained in the
[complete numeric result](2026-11-03-cdp-breakout-replay-age.json).

## What was reconstructed

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-replay-age.md)
uses the original control's log and exact447fe50 replay module. A private CPU
harness executes its original selection,eviction,fresh-first queue,flags,
expert-target alignment and context-refresh code with small numeric markers
instead of pixels and neural state. All device paths fail if called. The
original scheduler and replay RNG constant are extracted from the same source;
the locked rand0.9.4/ChaCha12 dependencies and compiled executable are pinned.
Fixtures check sampling invariance across different marker widths.

Each collection arrival is stamped with the current completed-update count.
Each sampled context refresh uses the pre-update count, matching native
`sample_posterior_batch` followed by `update_context`. The pinned cf4ea238
runner orders transitions,learning and resets. The reconstruction reconciles
38,051 events,200,000 actions,200,271 arrivals (eight initial and263 resets),
12,477 updates,progress debt and every logged supervised batch count.

The native learner actually uses device replay; this diagnostic follows the
same sampler and refresh bookkeeping through its host-storage branch. It
does not reconstruct unlogged GPU tensors or claim full execution equivalence.
The marker values never pass through a neural network. Age records only the
last state computation: inherited history can be older, live acting states
also span model updates, and a recently computed state can still be inaccurate.
Parameter movement and the sensitivity of decisions to stale state remain
unmeasured.

The [smaller-buffer comparison](2026-11-03-cdp-breakout-replay16k.md) regressed
while also changing recency,diversity,reuse and coverage. The earlier
[B8/T32 comparison](2026-11-01-cdp-breakout-context32.md) on an older recipe
improved greedy130.417 versus104.667 but lowered sampled48.583 versus56.292.
Neither isolates context freshness. Any later intervention needs a separate
declaration and a frozen gameplay/cost comparison; these age statistics alone
do not select a new training recipe.

## Closure

All **27 diagnostic /338 original control pins** remain unchanged. Compilation
passes strict warnings in.979s, and all15 CPU fixture cases pass in.001s.
Fixtures cover collection/refresh stamps,fresh-first order,eviction,stream
separation,reset masking,label alignment,geometry invariance and inconsistent
log rejection. The one full analysis completes **12:35:18.845 UTC November3**
in.387s; post-analysis pin verification completes **12:35:19.800**, source
`b729ab2396e4b3f9229a5c0c4868b7b86a716c6f`.

No preparation or analysis failure, training update, emulator/GPU operation,
original replay or prior reader repeat occurs. The control and H3 experiments
remain closed. Root `/x/Temp/kindle-breakout-replay-age-20261103` contains the
historical sources,original-module-plus-marker harness,executable,receipts,
declaration,review and all199,632 rows in `contexts.csv`. This diagnostic is
complete; do not repeat it. No Kindle allocation is active at closure. The
five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `7ed8d16c73d9128bed766f7b3efa942e228c95b8ae088d246f55c155f978109d` |
| Exact historical replay module | `c252d85d787726fb1a5038b76fc84a0ce821da78dc6af69735cea3d0cef80355` |
| Review and numeric result | `afc96c3feb83b3aab25dafcc0425b48a183b4fff35804a6a0de6cdb1ff39b1ac` |
