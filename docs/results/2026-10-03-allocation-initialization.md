# Allocation warning does not prevent this native execution

The user authorized a short instrumented initialization diagnostic and classified
`VUID-StandaloneSpirv-None-10684` as non-blocking. The earlier failed canary and
its logs remain unchanged. This is not a learning experiment or a GPU health
certificate. [Compact result](2026-10-03-allocation-initialization.json).

## Results

- **Native Meganeura/Blade probe passes:** RTX5080, driver 580.178.04, existing
  boot; context creation, Device/Shared/Upload allocations of 4096 bytes each,
  one 256-element GPU square operation with exact expected outputs, and teardown.
  Elapsed 2.20691s; minimum sampled Vulkan estimated budget-minus-usage 15.423GiB.
  This budget is not physical free or peak VRAM; allocator backing blocks can
  exceed the requested buffer sizes. CPU cgroup peak 81.7MiB.
- **One allocation warning remains:** journal receipt at 06:07:16.861177 UTC,
  105.416ms after the pre-context marker, before context completion and all
  explicit probe buffers. No allocation API failure or device loss was observed.
  The known shader-layout diagnostic is retained, and computation succeeds.
- **Independent CUDA initialization passes:** `cuInit`, device count and name
  return success for the expected GPU, with no new allocation warning. This
  process creates no explicit CUDA context, device allocation or kernel.
- **Retained helper failure:** the first CUDA probe reaches successful `cuInit`
  and device-count calls, then fails on a duplicate Python logging argument.
  It is a failed run, not a driver failure. A corrected, CPU-mocked helper runs
  once under a fresh declaration and passes; neither earlier script nor evidence
  is overwritten. Both CUDA attempts have no recorded allocation warning.
- No new Xid, hang, OOM kill, NVML polling, privileged tracing, driver change,
  reset or reboot. All three children exit and all evidence seals audit cleanly.
  A failed run's valid seal does not make its execution successful.

## Interpretation and scope

The startup warning alone was too conservative a reason to infer that ordinary
GPU execution was blocked. It can coexist with successful initialization,
allocation, dispatch and readback on this machine. It does not establish that
every allocation warning is harmless, or identify the failed internal request.
The kernel record carries no caller, size, address-space or stack information.
Receipt-time overlap localizes observation, not definitive process attribution;
kernel source and host monotonic clocks must not be naively subtracted.

Yesterday's two successful CPU/PyTorch oracle jobs coincide with similar warnings.
CPU tensor execution did not guarantee no driver calls: the installed PyTorch
[autograd engine](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/autograd/engine.cpp#L1587)
enumerates accelerator devices. Today's warning-free `cuInit` does **not** prove
CUDA initialization caused those earlier records. Large-page fallback remains
a hypothesis, not a diagnosis; no matching fallback message was observed.

Upstream rechecked: Meganeura `6268ea5` and Blade `e349cddf` remain current main heads
and match the probe. No backend fork, dependency change or CPU learner workaround.
Next: bounded backend/Tiny numerical qualification, then cost/learning checks.
No training run, frozen competence or Tiny gradient correctness is claimed here.

## Implementation and evidence

The host guard defaults remain strict. An explicit `allocation_diagnostic`
declaration can retain at most two occurrences of one exact warning for at most
120 seconds, never a blanket training waiver. Other warning sites, excess
occurrences, faults, timeouts and native failures still stop. Baseline warnings
require their exact reviewed cursor/message pairs. The sole reviewed validation
VUID is recorded rather than suppressed; other VUIDs remain fatal.

The small [native probe](../../kindle/examples/gpu_init_probe.rs) logs stage times,
backend resource sizes and estimated memory budgets; it exercises no model,
replay or optimizer. **1,004 Python tests, including 125 guard tests**, strict
workspace/all-target Clippy and formatting pass. No unchanged native library
rebuild was needed for Python tests.

[Native declaration](../../runs/allocation-init-20261003.YNlShm/native-declaration.json),
[native evidence](../../runs/allocation-init-20261003.YNlShm/native-guard/),
[retained CUDA helper failure](../../runs/allocation-init-20261003.YNlShm/cuda-guard/),
[corrected CUDA evidence](../../runs/allocation-init-20261003.YNlShm/cuda-v2-guard/).
