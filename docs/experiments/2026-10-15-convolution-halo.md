# One fixed-geometry convolution halo-cache trial

Declared October15,2026 before implementation/build/GPU work. Follow the
[CPU address/resource proof](../results/2026-10-15-convolution-matrix-feasibility.md):
all15 forward/dX occurrences fit a shared spatial cache alongside the unchanged
staging/output buffers, at most47,104 bytes. This tests source-data reuse,
not a tile/precision/layout search or the rejected matrix2 implementation.

Use an isolated copy of the current repeated-row backend. Add one generic typed
cache for rectangular stride1 convolution tiles: preload their surrounding
image region, replace subsequent image reads with corresponding shared reads,
and preserve original arithmetic, reduction order,32/64 geometry, A/B staging,
output scratch/scatter and barriers. Add only the required initial cache barrier.
Do not change weights, all eight dW products/compensation, graph/session groups,
packing operations or learner behavior. A shape/resource admission check keeps
unsupported geometry on the original path. No game-specific shape selector.
An explicit diagnostic switch chooses original/cache in the same binary; its
pipeline key includes the admitted dimensions. No production defaults change.

CPU requirements before GPU: all actual source contracts admitted; dW/unsupported
geometry/oversized shared-memory cases rejected; generated F32/F16/BF16 modules
validate with type-appropriate memory accounting. Independently confirm that
the source rewrite only replaces image reads and adds cache initialization;
all matrix math/output code remains. Use a distinct Cargo target directory and
old/new pipeline assertions; binary hashes alone do not prove implementation.

GPU order, serialized and fail-stop:

1. Numerical original/cache on all13 distinct actual forward/dX shapes, retaining
   their15 occurrences. Batch1 independent dyadic inputs; all outputs compared
   to direct F64 convolution at existing gates: derivative pointwise
   `1e-4 + 1e-5*abs(reference)`, relative L2`2e-5`; BF16 forward pointwise
   `1e-4 + .008*abs(reference)`, relative L2`.004`, independently rounded.
   Require bit-exact original/cache outputs, changed-input and restored results,
   actual pipeline selection and unchanged dispatch/layout contracts. Retain
   all failures; no tolerance relaxation. This is scoped numerical evidence.
2. Only on numerical success: one B512 family timing pair, original then cache,
   each16 warmup+128 measured calls/shape; include existing input transforms,
   output handling, allocations and every occurrence. Keep all samples and
   disclose regressions. Require at least10% less weighted GPU time and exact
   final outputs. No fastest-subset promotion or timing/geometry sweep.
3. Only if useful: one full RGB and centered-CDP B32/T16 smoke/timing pair on
   the retained fixed checkpoints/configurations,16 warmup+256 measured updates.
   Exact initial/first/final292 RGB/346 CDP tensors, all272 reports and first
   targets required. Accept only at least3% complete RGB improvement without
   over2% CDP regression. No independent whole-model or learning-parity claim.

The same-binary control must preserve the current076f3135 behavior. If the family
fails, no full-update trial or automatic optimization follows. These gates do not
complete the broader JAX parity/root-cause goal. Shared-memory residency, cache
hits and added initialization can offset fewer logical global loads.

Use persistent systemd user service(s), Restart=no,KillMode=control-group,
30min GPU allocation; each numerical/timing worker<=300s, full-update worker<=
180s. Existing host guard; require5080/580.178.04/boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`,>=2GiB sampled Vulkan estimated headroom.
Record standalone allocation warnings. API/numerical/hard/deadline failures
stop affected work for review, not blind retry. No recovery or NVML polling.
CPU preparation/build/review:1CPU,2GiB,no swap; never concurrent with timings.
Seal sources/binaries before GPU. Upstream99e3de27/e7c3f7b rechecked unchanged
at08:08UTC. No Naga/Blade change, CUDA learner, game actions or Atari restart.

### Reviewed numerical-contract repair,08:37UTC

Both numerical GPU workers pass, but CPU review rejects the harness: compiling
at B1 changes the channel-packing admission, so some checked kernels replace
the real B512 packed kernels. Only the first131,072 F64 values were reviewed;
these runs do not qualify the actual family. Keep both guards/results and the
failed review. No backend/shader repair or tolerance change is indicated.

In a separate `contract-repair` allocation, use the retained actual B512
per-operation plans for B1 numerical checks. Change only image batch/input
extent and matching pack/grid counts; lock the existing selected schedule.
Keep allocated buffer capacities (well within budget), weight tensors, types,
packing, heavy geometry and output path. The same complete-family contract
audit must pass before any B512 timing. Original B512 timing construction stays
unchanged. This repairs coverage, not a numerical failure or a new candidate.
Original CPU build receipt-path failures also remain; no GPU retry of them.
