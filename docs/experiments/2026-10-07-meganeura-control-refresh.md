# Latest backend on the unchanged CDP control graph

The separate [projection-reuse trial](2026-10-07-cdp-exploration-throughput.md)
stopped at its fixed numerical gates. Tiny/1M B3 heads passed F64;12M B3 head0
exceeded its pointwise limit despite candidate/control outputs being identical.
The separately reviewed, previously unstarted1M B128 check then found a
candidate/control relative L2 of4.91e-4, above2e-5. Neither failure is a measured
speedup or evidence of a wedged GPU. No tolerance is relaxed, no timing/training
ran, and the proposed graph/test source is preserved under the trial root.

Restore the original expanded-action graph. Do not stall learning on an
optimization sweep. Meganeura's Auto policy can use reduced-input cooperative
products on f16-only devices; changed matrix shapes can change selection. That
is a possible explanation, not established causality or a new precision policy.
Latest main remains `f104f354f50339f610a04d81fa75808c0cc535b0` after rechecking.

## Bounded refresh

- Meganeuraf104f35, Bladee349cddf, unchanged Kindle production graphs and Auto
  precision policy. Reuse the preserved same-source control Rust binary; build
  its Python extension. Old qualified native6d38eea2/c637 remains preserved.
- Existing four independent exploration/detachment/learning checks, original
  and centered cosine F64 derivatives, grouped RSSM output/gradient checks,
  and all2,824 upstream CDP/RGB comparisons including optimizer/EMA.
- Three production/frozen smokes:1M raw CDP Pong,12M centered CDP Breakout,
  1M RGB Seaquest. Each1,024 training actions/195 updates then1,024 frozen
  actions/zero updates and exact saved tensors. Actual initial weights,
  counters, finite metrics/checkpoints, native identity and sampled headroom
  are audited. Total3,072 training plus3,072 frozen actions; no competence claim.
- These are the established scoped qualification checks, not a claim that
  the control passes the failed new full-head F64 gate. Preserve that failure
  and the distinction between approximate forward arithmetic and raw-gradient
  comparisons. No CPU learner, global scalar forcing or numerical relaxation.

After qualification review, one ordinary full-update timing per backend:
preserved c637 grouped-control binary versus f104 original-action binary,
both restored from12M Breakout1009. Same exact initial346 tensors/config,
fixture RNG701,16 warmup plus256 measured updates. No timestamp instrumentation,
compilation or other heavy work during timing. Each first saved output requires
maximum absolute difference<=1e-5 or relative L2<=2e-4; report all differences
and every update, without asserting later stochastic trajectory equality.
This measures a backend refresh, not the rejected graph rewrite. GPU utilization
remains unmeasured. Review substantial regression before another allocation.

GPU tests/timing120s per process, smokes300s, qualification service1h. Serial
host guards, Restart=no, KillMode=control-group, RTX5080/driver580.178.04,
>=2GiB sampled Vulkan estimated headroom. Allocation warnings are recorded;
API/numerical/hard-fault/deadline failures stop the affected queue for review.
No separate NVML polling or recovery. CPU preparation/audit:one CPU,2GiB,
zero swap; GPU workers retain ordinary CPU allocation. Reused binaries and
new jobs are hashed before launch. No automatic learning extension.

Artifacts:`runs/meganeura-control-refresh-20261007.FZHHoQCK`.
All five DreamerV3 quality targets and the budget comparison remain open.
