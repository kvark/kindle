# Latest ordinary backend qualifies; measured construction does not advance

**Adopt Meganeura2c42b448 / Blade56f0565, native0f26ee63 for ordinary CDP/RGB.**
The fixed-checkpoint control measures107.931→108.302ms per full update,0.34%
longer: no speedup. First outputs, all272 numerical metric reports and all346
final tensors agree exactly. The compiler-search attempt stops at its numerical
gate and is not integrated. Its test-only hook is removed; source and binaries
remain with the artifacts. No production graph or learning recipe changes.

[Original declaration](../experiments/2026-10-11-cdp-measured-build.md) ·
[Reviewed preflight repair](../experiments/2026-10-11-cdp-measured-bound-repair.md) ·
[Compact evidence](2026-10-11-cdp-measured-build.json) ·
[Independent CPU review](../../runs/cdp-measured-bound-repair-20261011.E12THDy8/completion-review.json).

## Ordinary execution passes

- All19 qualification guards pass: independent RSSM/cosine/action-effects
  checks,1,300 CDP and1,524 RGB upstream comparisons, three production
  train/frozen pairs and the read-only live-value test. Numerical tolerances
  are unchanged. Smokes add3,072 training actions/585 updates and3,072 frozen
  actions/zero updates; exact346/346/292 frozen tensors remain.
- The two ordinary timing guards pass. Both binaries restore the same346
  tensors/config from completed12M Breakout1009, with fixture RNG701,
  sixteen warmup and256 measured full updates. World/behavior training,
  synchronization and target construction are timed; fixture generation,
  export and initialization are not. No build, heavy analysis or dispatch
  timestamp instrumentation overlaps timing.
- Mean/median/p90 are107.931/107.864/108.556ms on31026833 versus
  108.302/108.256/108.911ms on2c42b448. Both retain1,210 imagination dispatches.
  Exact metric/tensor agreement is scoped to this fixture, not universal parity.
- 116 CPU Rust tests,1,217 Python tests, formatting and both strict Clippy
  checks pass. The first CPU build used the wrong `inherits_parameter` API;
  its source/log remain. Correcting the donor argument changes no GPU kernel
  or numerical tolerance. That failed build launched no GPU work.

Qualification finishes18:31:10 UTC; ordinary timing finishes18:34:41.
The ordinary result is not measured whole-game throughput or SM utilization.
No identified upstream fix explains earlier learning weakness. Historical
learning keeps its original backend/native identities.

## Two failed search attempts, neither hidden

The first attempt ends after9.94s with "no qualified program within the search
bounds", before any numerical callback. My4GiB limit counted **logical** buffers,
not allocated VRAM. The retained ordinary plan already uses9,788,951,108 logical
bytes versus2,400,187,972 allocated bytes. All346 calibration tensors remain
exact; zero optimizer updates/game actions and no new warning. The failed API
does not return its per-program report, so no exact new-plan sizes were saved.
This is a bad preflight, not evidence against compiler tuning.

A separately declared correction changes only that logical pair-plan cap to
20GiB and records the current ordinary size, which matches9,788,951,108 bytes.
It retains the2GiB sampled Vulkan-headroom floor, graph, precision, four graph
forms/eight programs/180s search bound, original inputs and numerical gates.
Only the test binary changes; the qualified native extension stays untouched.

The corrected attempt stops after14.78s on its **third qualification callback**.
The first two callbacks match every output exactly. On the third, actions and
all categorical latent samples stay exact and all outputs remain finite, but:

| Output | Maximum absolute error | Relative L2 | Declared gate |
| --- | ---: | ---: | --- |
| Value logits |0.005533 |2.06345e−5 | Fails |
| Intrinsic bonus |2.65911e−5 |8.18783e−5 | Fails |

The fixed gate is maximum absolute error≤1e−6 **or** relative L2≤2e−5. Other
outputs pass. No tolerance is relaxed and there is no further retry. These
small differences do not establish a GPU defect or explain historical policy
failures. The panic loses the search report: the precise program/kernel choice,
private probe decisions and any internal timing are unavailable. This was the
third callback, not necessarily the third candidate. No selected program or
ordinary whole-update timing exists for measured construction.

Both attempts preserve the source checkpoint exactly and perform zero optimizer
updates/game actions. The corrected run's minimum sampled estimated headroom is
8,116,043,776 bytes, well above2GiB; this is not physical free or peak VRAM.
There are21 passing guards and two retained failed-job guards, with
ten new allocation warnings total: nine during qualification, one during the
corrected search. No GPU reset/recovery, separate NVML polling or remaining
worker. SM utilization remains unmeasured.

The final attempt ends18:41:17 UTC. Independent CPU review takes0.85s, checks
both original source manifests, declarations, sealed failures, calibration
tensors, callbacks and the ordinary control. Failed/corrected builds and all
search preparation are additional costs, not excluded successful attempts.

## Next: learning stability, not another optimization gate

Keep the qualified ordinary runtime and12M/T16 centered CDP/action-effects
recipe. Qbert's completed2M budget, forecasts and live values demonstrate useful
learning without identifying a shared model/critic collapse. The cost remains
about108ms/update; compiler search has not improved it.

Next separately declare a **three-seed Pong experience-budget study**, targeting
the current69/72 wins and inconsistent mastery. Keep all three seeds, actual
initial controls, natural frozen cohorts, full videos and the original20.45
long-run target. Do not extend all five games, resume incomplete lifetimes,
change entropy/representation or retry rejected optimizations. The other four
games and the matched RGB budget question remain part of the open full goal.
