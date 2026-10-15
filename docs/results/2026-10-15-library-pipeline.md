# Fixed smaller-pipeline library controls are unsupported

October15,2026. [Declaration](../experiments/2026-10-15-library-pipeline.md)
· [Data](2026-10-15-library-pipeline.json)
· [Prior resource/transfer result](2026-10-15-convolution-transfer.md).

**Both exact-engine, reduced-stage configurations fail before execution-plan
construction.** No smaller-pipeline kernel was executed or timed. This closes
the fixed library-side control, not the shared-memory question or performance
goal. Complete isolated RGB12M/B32 remains54.224ms versus retained JAX26.632ms;
production and Atari stay unchanged.

The two original cnn1 derivative plans reconstruct with their saved shapes,
NHWC/KRSC strides, BF16 tensors/F32 accumulation,32-byte alignment, engines,
knobs and workspaces. Original dX engine40 uses0 workspace bytes; dW engine33
uses23,961,728. Their captured kernels use four stages and76/80KiB shared memory.
The serialized local GPU profile confirms **99KiB CUDA opt-in shared memory**
versus the separately queried **48KiB explicit Vulkan workgroup limit**.
This is a real interface/resource difference, not its measured latency cost.

For each engine, changing only stage choice3→1 fails with
`CUDNN_STATUS_NOT_SUPPORTED` at **engine-configuration finalization**, before
an execution plan exists. Both engines advertise the individual stage range0–7;
individual ranges do not guarantee a supported combination with the saved tile/
split knobs. Choice1 remains a hypothesized two-stage control, not an observed
kernel. No stage0/2, other engine/tile, heuristic search or library upgrade follows.

## Four stopped processes, not four passing qualifications

| Process | Retained outcome |
| --- | --- |
| Original plan/query | Original dX reconstructs; INT64 shared-memory query fails2010, bad attribute type |
| Reviewed INT32 query | Same original dX JSON; query fails3000, not supported |
| Remaining fixed plans | dX choice1 configuration fails3000; dW not attempted |
| Previously unstarted dW pair | Original dW reconstructs; choice1 configuration fails3000 |

Three original plan constructions cover two distinct plans. All four host
guards remain failed with child exit1; every failure was reviewed before the
separately declared follow-up. Zero new allocation warnings, unfinished children,
recovery, separate NVML queries, convolution executions, optimizer updates or
game actions. Library initialization/plan construction can allocate internally;
the harness allocates no data tensors or workspaces. Sampled Vulkan estimated
budget headroom is at least16,594,173,952bytes, not physical or peak-free VRAM.

Each failure's broad kernel-log snapshot hits its output limit. Those four
incomplete capture receipts remain alongside the completed baseline/incremental
monitoring, API stderr and serialized plans. Do not call the failure captures
complete or the failed processes hardware qualifications. The pre-launch dW
wording correction, its original declaration/queue, and all executed sources
remain. CPU review verifies source seals, descriptors, knobs, workspaces,
GPU profiles, failure stages and zero-execution source paths without GPU retries.

Installed cuDNN9.24 exposes the stages knob and memory-used attribute but lacks
the newer documented heuristic memory-limit attribute. No undocumented enum
was injected. Reconstructing engines/knobs follows the pinned
[XLA convolution backend](https://raw.githubusercontent.com/openxla/xla/3d5ece64321630dade7ff733ae1353fc3c83d9cc/xla/stream_executor/cuda/cuda_dnn.cc);
the serialized `is_cudnn_frontend:false` field does not establish legacy
execution. Current online APIs are not proof of installed-library support.

Artifacts: `runs/library-pipeline-20261015.I0dyGaYL`; independent review and
all four failed queue receipts are retained. Latest upstream99e3de27/e7c3f7b
is unchanged. No compiler/backend mutation or speedup follows from this check.

## Next question

Do not turn the remaining investigation into a sequence of small iterator/tile
changes. Earlier zero-work masks left invalid intermediate data and produced
large, sometimes negative interactions. Check the feasibility of substituting
**saved exact convolution outputs at their original dispatch slots**, preserving
their consumers, allocations and dependencies. A fixed-input, optimizer-free
comparison could bound the importance of the entire convolution family without
pretending invalid masked work is ordinary kernel latency. It requires its own
declaration, exact-value checks and copy-cost accounting; it is not a training
optimization, pure-kernel measurement or automatic new GPU allocation.
