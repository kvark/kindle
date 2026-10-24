# RGB-boundary algorithm diagnosis — October 14

The best isolated complete RGB update is 25.530 ms versus retained JAX
12.836 ms. Command reuse saves 10.16%; it does not explain the remaining
device execution gap. Five checked convolutions account for 3.785 ms of the
previous interacting cost allocation, despite only 11.1% of convolution MACs.

Inspect exactly those five operations: encoder/decoder forward, decoder input
gradient, and both weight gradients. All use B128, RGB64, 5x5, stride 1, pad 2;
channels are 3->32 or 32->3. Retain the qualified checked 32x32x32 kernels and
128-way compensated weight-gradient split. No geometry/precision-policy sweep,
learner update, game action, production change or training restart.

The actual JAX executable returns BF16 gradients; native returns F32 gradients.
Both multiply BF16 operands. Treat that distinction as part of the comparison,
not as numerical parity or proof of a performance cause. Compile the five JAX
primitive shapes, retaining selected algorithms/HLO, and three additional F32
gradient-output controls. Check whether these controls really preserve BF16
operands and whether primitive algorithms match the full executable. Do not
silently equate a widened-input control with a mixed-output cuDNN kernel.

Use deterministic exactly representable BF16 inputs in native NCHW/OIHW and
JAX NHWC/OHWI layouts, arranged before timing. Save full outputs and independent
128-point F64 controls per operation. Check finite outputs, changed-input
execution and all matched output elements. F32 probe gate is
`abs(error) <= 2e-4 + 2e-5*abs(reference)`; BF16 adds one output rounding ULP.
These are diagnostic checks, not qualification of a new learner precision policy.

Native: five cases, two changed-input executions plus 16 warmups and 128
pre-recorded executions each. Keep wall/recording and GPU pass times separate.
JAX: eight cases, two changed-input executions, 16 warmups, 128 ordinary calls
and a 16-call Nsight Systems CUDA capture per case. Save the capture and
compiled HLO; count actual kernels/copies. At most one separately declared
Nsight Compute follow-up after numerical review, if needed to explain the
dominant algorithm. No repeated autotuning/tile search.

Retain all failures. Ordinary native/JAX jobs each have 300-second deadlines;
capture 300 seconds, enclosing service 15 minutes. Use the host guard, expected
5080/driver/boot, >=2 GiB sampled Vulkan budget headroom and recorded allocation
warnings. No NVML polling or recovery actions. CPU preparation/analysis uses
one CPU, 2 GiB, zero swap. Neighbouring worktrees stay untouched.

Artifacts: `runs/boundary-algorithms-20261014.boE7ZosI`. Backend retains the
command-cache experiment's qualified source (cache off in this primitive
fixture), not the rejected narrow-SIMT candidate. Latest upstream remains
Meganeura `7574ad1`, Blade `e7c3f7b`; no new relevant fix since the prior review.

## Reviewed counter follow-up, 09:58 UTC

The five native descriptors/grids match the learner exactly. All 1,664 F64
probes pass; all eight full JAX outputs equal native at the declared output
precision. Decoder dW is dominant: native 1.742 ms versus cuDNN 0.338 ms
including transforms/reduction, with exactly the full learner's algorithm.
The encoder forward and decoder dX are competitive; do not optimize all five
indiscriminately. Decoder forward and encoder dW primitive autotuning selected
different algorithms/knobs from the full executable; retain that limitation.
The three F32-output controls widen operands and cannot isolate storage policy.

Profile only decoder dW next. Translate its unchanged checked 32x32 BF16 loop,
128 partitions and 256-term compensated F32 accumulation into CUDA for an
Nsight Compute bridge. Require all 2,400 outputs exactly equal the qualified
native fixture before counters; preserve native code and its F64 controls.
One 16-warmup/128-call CUDA batch, then one CUDA-twin and one matching cuDNN
kernel counter capture (one selected invocation each; profiler replays disclosed).
cuDNN's typed contract/algorithm must match the completed primitive and full
learner before its capture. Each process <=300 seconds. No new algorithm,
precision relaxation, replacement runtime, geometry sweep or whole-model claim.
Counters describe the CUDA translation, not native Vulkan machine instructions;
compare its ordinary time with native before extrapolating.

The initial native fixture's unsupported BF16 graph-input assertion happened
before kernel execution. Its binary/source/guard remain. The repaired fixture
uses existing internal conversion buffers, not a new backend input API.
