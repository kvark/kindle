# Reuse state projections across exploration actions

The [12M capacity study](../results/2026-10-07-cdp-capacity.md) improves every
Breakout seed, but costs 3.58× the small-model training time. Whole updates
average 110.03ms, 55.4% in imagination. All five quality targets remain open.
Qualify one performance change before declaring a larger learning allocation.

## Fixed change

Each exploration head currently repeats `[state, onehot(action)]` across all
18 actions before its first matrix product. Compute `state @ W_state` once,
then broadcast and add each action's weight row and the original bias before
the unchanged RMSNorm/SiLU. Later layers, all-action centering, disagreement
and selected-action weighting stay unchanged. The observed-action training
loss, parameter leaves/initialization, optimizer and stop-gradients do not
change. No new shader or algorithm setting.

The [CPU algebra check](../results/2026-10-07-action-affine-check.json) agrees on
saved weights/states. About 63% fewer ensemble dense MACs is a hypothesis for
speed, not a measured whole-update gain. F32 reassociation need not preserve
bitwise stochastic learning trajectories.

Adopt latest Meganeura main `f104f354f50339f610a04d81fa75808c0cc535b0`, checked
October7 before this trial. It includes F32 acceleration, fused transposed
matmul-add promotion, kernel-family selection and bounded dispatch geometry.
Both control and candidate use it. Blade stays at Meganeura's pinned
`e349cddf`; newer presentation-damage changes are unrelated to this learner.
Neighboring user checkouts remain untouched. Preserve the old qualified native
and a freshly built same-backend control; do not repin completed learning.

## Qualification and timing

- CPU checks: unchanged parameter schema, reduced first-layer matrix rows,
  detached state/action inputs, formatting, strict Clippy and existing tests.
- Independent F64 complete-head checks at small row counts with Tiny/1M/12M
  widths; original versus factored GPU outputs at posterior/imagination batch
  sizes128/1920 for1M/12M. Head-output gates: maximum absolute error
  `2e-6 * (1 + abs(reference))` and relative L2 `2e-5`; bonus absolute gate
  `2e-6`, as in existing independent bonus checks. Report relative bonus error
  too, without using relative error at the zero floor as a gate. No relaxation
  after seeing results.
- Keep existing independent action-effects/loss/raw-gradient/detachment/learning
  checks, original/centered cosine derivatives and all2,824 upstream CDP/RGB
  comparisons including optimizer/EMA. The changed attention path is not
  qualified for Tiny video merely by CDP/RGB checks.
- Production/frozen smokes:1M raw CDP,12M centered CDP and1M RGB. Each fresh
  learner executes1,024 actions/195 finite updates, then1,024 frozen actions
  with zero updates and exact saved tensors. Retain actual initial weights,
  counters, source identities and sampled headroom. These are sanity checks,
  not new competence evidence:3,072 training plus3,072 frozen actions.
- One matched ordinary full-network update timing per binary, both restored
  from completed12M Breakout1009:16 warmup plus256 measured updates, fixture
  RNG701, no GPU timestamp instrumentation. Same config and exact initial
  tensors. Preserve every report, first outputs and initial/first/final tensors.
  Compare numerical outputs before speed. Fixture generation, replay storage,
  exports and game stepping are outside the timer; all production network
  update stages, targets, synchronization and slow EMA are inside.
  First-output comparisons require maximum absolute error<=1e-5 or relative
  L2<=2e-4 for each saved output; later stochastic trajectories need not match.

Build both binaries before timing; no overlapping build or GPU work. GPU tests
and timing processes have120-second limits; production smokes300 seconds.
Persistent serial host-guarded services use Restart=no, KillMode=control-group,
bounded service deadlines, expected RTX5080/580.178.04 and>=2GiB sampled Vulkan
estimated headroom. Record allocation warnings; actual API/numerical/hard-fault
or deadline failures stop for review. No separate NVML polling or recovery.
CPU preparation uses one CPU,2GiB and zero swap; GPU workers retain normal CPU
allocation. Machine-readable jobs and binary/source hashes are sealed before
launch. Artifact root:`runs/cdp-action-throughput-20261007.dAJvnek`.

Retain the graph rewrite only if qualification passes and mean ordinary full
updates are at least5% faster. Otherwise keep the qualified control graph;
do not turn this into an optimization sweep or indefinite learning gate. An
upstream refresh is not attributed to the factorization. After review, declare
the next finite allocation toward all five games, preserving three learner
seeds, actual-initial/frozen controls, curves and videos. Faster updates alone
do not complete the quality or budget goal.

## Outcome

[Rejected at the fixed numerical gates](../results/2026-10-07-cdp-exploration-throughput.md).
No timing/training runs or tolerance relaxation. Both failing jobs and the
removed candidate source/binaries are retained. The separately declared
[backend-only control refresh](2026-10-07-meganeura-control-refresh.md) uses
the original production graph; its scoped checks do not erase these failures.
