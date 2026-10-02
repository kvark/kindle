# Joint Tiny qualification: implementation, not learning evidence

The user authorized joint Tiny versus frozen pretrained Tiny. The
[protocol](../experiments/2026-10-02-joint-tiny.md) specifies at most three paired
Seaquest seeds. **No learning run has started.** The new path is not GPU-qualified.
Phase 2's frozen-frontend evidence and the learned-RGB screening default remain.

## Implementation and CPU evidence

- Full differentiable, frame-causal Tiny clips use the pretrained encoder's
  weights, RoPE and fixed JL64/pool2. All 148 used parameter tensors have
  backward graph paths from reward, continuation, replay value and prior
  prediction separately; frozen mode stops them. This is structural evidence,
  not evidence of numerical nonzero gradients or successful updates.
- Replay retains native-detail GPU-preprocessed pixels and re-encodes complete
  16-arrival chunks with current weights. Metadata tests exercise asynchronous
  streams, resets and eviction. Host feature arrivals cannot fabricate missing
  pixels. The first lifetime chunk is excluded because it lacks preceding RSSM
  context; subsequent real resets use their keep masks. Both arms share this rule.
- Joint updates rebuild current acting KV prefixes from GPU pixel history,
  retaining per-stream arrival counters and RSSM state. Scheduled and manual
  single-agent learning use the same refresh. Copies finish before destroying
  their source buffers. Numerical cache-refresh/restore integration is unrun.
- The inference-only scale operation was replaced with differentiable scaling;
  splits/concats now obey Meganeura's flat-tensor contract. CPU autodiff caught
  both problems before any Tiny GPU execution. No backend autodiff fork.
- The declared variance/covariance regularizer exposes its loss and latent
  spread. Its statistics use `microbatch × 16` frame means, not the accumulated
  full batch; microbatch size therefore changes this objective and must be fixed
  across arms. It is not original LeVJEPA pretraining or a collapse guarantee.
- An independent CPU/PyTorch F64 oracle covers Tiny outputs, all 148 parameter
  gradients, and regularizer values/gradients. Native comparison tests are
  implemented but **ignored/unrun** pending native qualification.

Validation: **105 workspace/all-target CPU tests and 992 Python tests pass**, including
113 host-guard tests. Workspace and Python-binding strict Clippy and formatting
pass. Python tests use a freshly rebuilt binding. GPU tests remain ignored;
these counts do not imply GPU numerical or learning parity. The CPU oracle ran
with CUDA devices hidden, one thread and no GPU execution.

CI256's 13 existing Linux/lavapipe Dreamer canaries pass, as do Python and
macOS. That run still fails: its broad `tiny_` filter accidentally selects the
new native-fixture cache test, which refuses on a missing environment variable
before device initialization. Rename that test out of the broad filter, retaining
its explicit native qualification requirement; no existing CI gate is removed.
This remote software-Vulkan evidence is not local NVIDIA qualification.
CI257 also selects it through the separate `vector::tests::` filter; all four
existing vector tests pass before that fixture refusal is reported. The test
now lives in `vector::joint_qualification`, outside both generic selectors.
Both failures are retained; the existing 13/4-test groups are unchanged.

The implementation uses sampled pixel-batch readback/upload at the existing
learner boundary, not a fully device-resident learner. Pixels/encoder weights
are much larger than the old frozen-feature replay. 8,192 F32 patch frames alone
cost 4.59 GiB, before features/state, live KV history and backward activations.
Explicit replay capacity is required by the experimental CLI. Training graphs
and inference can share compatible weights; derived layouts may still require
the existing synchronization copy. Measure these costs before assigning a budget.

## Native blocker, 17:31 UTC

Merged Meganeura `6268ea54416810a9c97d7f4511cf6f6b1cd1700e` and Blade
`e349cddf7a9bf181a362507e640b3d576b1aac72` were adopted, including host optimizer
bias correction. Adjacent backend worktrees were not changed. A single guarded
ordinary **Size1M RGB** fixed-batch export was attempted before Tiny qualification.
It was stopped/reaped after about five seconds; there is no completed-update
evidence. This was not an external-memory import retry or a Tiny learning run.

The guard now rejects NVIDIA allocation warnings. Only exact, separately reviewed
historical journal cursor/message pairs can be admitted at the baseline; new
occurrences and hard faults cannot be admitted during a run. Three prior records
were declared. A fourth `NV_ERR_NO_MEMORY` record caused the stop.

| Event | Monotonic seconds |
| --- | ---: |
| New warning's kernel source timestamp | 355753.044395 |
| Guard's child-spawn timestamp | 355757.493786 |
| New warning's journal receipt timestamp | 355757.944116 |

The source/receipt timestamps differ by 4.90s. These records do not establish
which process triggered the warning. It reports allocation failure at
`_memdescAllocInternal`; the [driver's memory descriptor implementation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.178.04/src/nvidia/src/kernel/gpu/mem_mgr/mem_desc.c)
handles several address spaces. The message does not distinguish physical RAM,
VRAM or allocation constraints. Available host RAM was about 25.8 GiB despite
few large buddy blocks: fragmentation is a possibility, not a diagnosis.
The service peaked at 432 MiB RSS. There is no new Xid, recorded hang or OOM kill.
No NVML polling, reset, cache dropping, driver change or other recovery was done.
No unfinished child remains; utilization and usable GPU headroom are unmeasured.

### Separate shader validation error

The canary's stdout contains repeated `VUID-StandaloneSpirv-None-10684` errors:
workgroup arrays have `ArrayStride` decorations. This is not evidence that the
error caused the allocation warning. A CPU-only Naga reproducer confirms that
the pinned writer emits that combination for a minimal workgroup-array shader:

```text
OpDecorate %5 ArrayStride 4
%5  = OpTypeArray %float %256
%12 = OpTypePointer Workgroup %5
%11 = OpVariable %12 Workgroup
```

[Vulkan's shader interface rules](https://docs.vulkan.org/spec/latest/chapters/interfaces.html)
restrict these decorations; the old-version exception covers Private/Function,
not Workgroup. Naga `323acfb` decorates every ordinary array in its SPIR-V writer.
The [latest upstream writer inspected, `dd033bf`](https://github.com/gfx-rs/wgpu/blob/dd033bfb7fb3690eb2e80a971877a1ef8ecabe59/naga/src/back/spv/writer.rs)
still contains that unconditional emission. Updating blindly is not a demonstrated
fix. The local standalone `spirv-val` command is unavailable; the original Vulkan
validation diagnostics are retained, not suppressed. The host guard now also
rejects emitted Vulkan validation errors, including a zero-exit native process.

## Evidence and next boundary

Local [run directory](../../runs/joint-tiny-20261002.6BaPBD/):
[canary declaration](../../runs/joint-tiny-20261002.6BaPBD/backend-declaration.json),
[sealed stop](../../runs/joint-tiny-20261002.6BaPBD/backend-guard/result.json),
[native stdout](../../runs/joint-tiny-20261002.6BaPBD/backend-guard/child.stdout),
[host-only snapshot](../../runs/joint-tiny-20261002.6BaPBD/incident-host/),
[CPU shader reproducer](../../runs/joint-tiny-20261002.6BaPBD/spirv-repro/),
[F64 oracle](../../runs/joint-tiny-20261002.6BaPBD/oracle-reg/).
The host snapshot's 21 files verify. The first encoder-only oracle is also retained.

Resolve/review the allocation warning and fix the shader validity problem before
another native launch. Do not simply whitelist the fourth warning and retry.
Then qualify encoder/regularizer values and gradients, optimizer motion, cache
refresh/restore, stopped targets and full-agent time/memory. Only afterward set
the matched common gameplay budget. No inference of learning quality is possible
from this interrupted qualification.
