# Integrated backend reduces full-update time by about16%

October12,2026. [Declaration](../experiments/2026-10-12-backend-integration.md)
· [Data](2026-10-12-backend-integration.json)
· [Backend plan](../backend_training_performance_plan.md).

The combined stack is useful, but does not catch JAX. Keep native BF16 storage,
repeated-row addressing and hardware-row convolution selection. Reject PR248's
weight-gradient kernel, which did not speed complete updates. Auto remains the
default; game training and frozen rollouts stay stopped.

| Complete synthetic update | Retained F16 control | Integrated candidate | Less time |
| --- | ---: | ---: | ---: |
| CDP12M/B32, `mixed` | 161.316ms | 135.168ms | 16.21% |
| RGB12M/B8, `bf16` | 72.685ms | 61.146ms | 15.88% |

Same5080/driver580.178.04, checkpoint, workload and fixture RNG within each native
pair;16 warmup+256 measured updates, no tracing or simultaneous compilation.
Candidate medians135.094/61.096ms. Four native processes plus two fresh JAX
controls finish02:27:23 UTC in3m47s. No game actions in these GPU experiments.
These are measured combined gains, not sums of the isolated improvements.

Fresh JAX RGB12M/B8: **FP32/highest24.287ms; BF1612.467ms**. Native RGB remains
**2.52x/4.90x slower**. Shapes and update boundaries are matched, but graphs,
weights, RNG and synthetic fixtures differ. CDP has a larger batch and an
exploration ensemble; its135ms is not a direct CDP-versus-RGB/JAX comparison.
No learning-quality, time-to-score or SM-utilization claim follows.

## Where the gain comes from

CDP stage times (control→candidate): posterior19.35→16.63ms, imagination/targets
70.20→54.18ms, world training/sync56.49→54.42ms, behavior/sync15.28→9.93ms.
World training and imagination now contribute about40% each. The world model
alone is not the entire remaining bottleneck.

RGB world training/sync39.68→36.68ms and imagination20.14→13.72ms improve.
Connected BF16 convolution storage helps RGB world training, but the
[all-BF16 CDP world path was slower](2026-10-12-bf16-convolution.md).
The [fixed mixed-stage policy](2026-10-12-mixed-world-precision.md) retains
F16-input/F32-storage world gradients, with BF16 posterior/imagination/behavior.
No hidden per-title switch. Masters, optimizer state and sensitive reductions
remain F32; convolution derivatives and grouped products are not all BF16.

Row addressing removes eight imagination dispatches versus the preceding mixed
candidate (1,346→1,338). BF16 boundaries still increase dispatches versus the
old F16 graph (1,229→1,338); this is not a whole-graph fusion victory. The
logical imagination-buffer sum is unchanged14.733GB versus that preceding mixed
candidate, while sampled headroom rises10.290→11.290GB. Logical buffers and
Vulkan budget-minus-usage are not measured traffic or peak physical VRAM.
RGB sampled headroom falls14.880→14.716GB versus its F16 control; no universal
memory-saving claim. Every measured arm remains above the2GiB floor.

## Qualification and provenance

- Four native numerical guards pass: nine BF16 checks (1,035 RNE conversions,
  18 matrix/F32-gradient and18 reduced-gradient cases,32 finite Adam updates),
  16 convolution/master-gradient cases, scoped hardware-row forward/input
  cases, and independent row-addressing values/gradients.
- All six complete-update guards pass. Three allocation warnings and known
  VUID10684 remain recorded; no hard fault, recovery or separate NVML polling.
- Exact initial346 CDP/292 RGB tensors,272 finite reports per native arm,
  finite first imagined targets and F32 first/final checkpoints; masters update.
  Both retained controls exactly reproduce their earlier reports/targets/tensors.
- New RGB exactly reproduces the preceding BF16-convolution candidate's272
  metric reports and all tensors. New CDP retains exact first imagined targets
  from the preceding mixed candidate, but hardware-row selection changes its
  arithmetic:228 first-step and346 final tensors differ. No trajectory-parity
  claim or weakened independent kernel gate.
- Thirteen backend CPU checks,14 Kindle config/policy/pin checks, strict
  workspace/Python Clippy and193 refreshed-binding runner tests pass. Both
  Cargo locks pin the same graph/graphics/compiler stack. Independent CPU
  review passes in4.767s with500.8MiB peak memory; no GPU retry.

Kindle runtime commit300363b uses Meganeura **d68f1a5** (BF16 PR250/fd9608f,
row addressingPR249, hardware-row selectionPR252), Blade **663afd2** and the
user-approved Naga fork **c9fd6a85a**. No Cargo path overrides. Current upstream
baseseb8c90f/7909fc1 were included before final qualification, without merge
commits; historical controls retain679ea87/b52965b and their original compiler.
Unrelated new attention/cache/Android changes are not explanations for old
learning failures. Only the user merges; no upstream wgpu PR was opened.

Meganeura BF16 [CI965](https://github.com/kvark/meganeura/actions/runs/38194560994),
row [CI966](https://github.com/kvark/meganeura/actions/runs/38194604386), standalone
alignment [CI961](https://github.com/kvark/meganeura/actions/runs/38192173821), and
Blade [CI1172](https://github.com/kvark/blade/actions/runs/38194491591) pass.
Kindle runtime [CI355](https://github.com/kvark/kindle/actions/runs/38195552476)
passes all Linux/lavapipe, Metal and Python jobs. The publication check verifies
255 local links across six documents and the result's raw-audit correspondence.
Full wgpu xtask remains unrun without cargo-nextest; scoped Naga checks passed.

Artifacts/source/locks/binaries/guards/checkpoints/audit:
`runs/backend-integrated-20261012.xWfA9jYb`. Full hashes are in the linked data.
Native binary83a9227f; refreshed Python extensionf2f7b999. The dense-only
`bf16_dense` CLI mode was removed; its evidence and source remain in history.

Retain three CPU preparation/review failures: a formerly invalid `bf16` fixture
reset eight ALE environments before native rejection (setup frames unrecorded,
no actor or learner update); the first extension service lacked rustc in PATH;
the first audit requested a nonexistent Blade result field. Corrected refusal
tests prohibit environment construction, the extension builds with explicit PATH,
and provenance is checked in both frozen locks. Original scripts/logs remain.
The [unchanged scalar weight-gradient oracle failure](2026-10-12-convolution-subtiles.md)
also remains; these scoped checks do not erase or qualify it.

## Decision

Keep the three useful changes and explicit `mixed`/`bf16` candidates. This
bounded implementation/measurement round is complete; the JAX speed target and
broader CDP world/backward reduced storage are not.
Next performance work should target the remaining world/backward and recurrent
costs with complete-submission profiles, not more precision-policy switches.
Before promoting a learning default, run a separately declared fresh three-seed
learning/time check. Do not resume either stopped lifetime or start another long
five-game campaign from these synthetic results.
