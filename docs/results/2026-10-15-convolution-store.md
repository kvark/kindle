# Contiguous output does not rescue the native workgroup convolution

October15,2026. [Declaration](../experiments/2026-10-15-convolution-store.md)
· [Data](2026-10-15-convolution-store.json).

Fresh direct/contiguous-output operations take**7.42984/7.53448ms**,1.41% slower
with the explicit GPU transpose included. Queued wall time agrees,1.45% slower.
Input packs, input/math shader source, BF16/F32 arithmetic,256x32 output tile,
64-wide per-tap reduction and128 threads are unchanged. The extra64MiB output
scratch and transpose do not produce a material improvement. Close this output
intervention without a tile/store sweep or backend promotion.

This rejects an inexpensive output-layout remedy, not an isolated measurement
of the store's latency or proof about generated instructions. The input-address/
matrix implementation remains unresolved. The retained ordinary native control
is2.147ms; complete RGB12M/B32 stays60.041ms versus JAX26.632ms. No new whole-update
timing, production change, game actions or learner updates.

All three guards pass with zero warnings: B1 validation smoke and fresh B512
direct/candidate pair,16 warmup+128 timed calls per arm.33,587,200 exact output
checks and393 F64 checks across264 distinct flat indices (265 shape/index cases) pass, including changed
input/restore.295 complete operation calls/1,034 shader dispatches; minimum
sampled Vulkan headroom15,934,619,648 bytes, not physical free/peak. CPU input/
math equality, unchanged packing shaders and32,768 transpose addresses pass.
Both typed shaders compile; F16 is not GPU-qualified by this study.

Raw guards,52 sealed source hashes and machine-readable smoke/ordinary reviews:
`runs/convolution-store-20261015.7IoSJ9SV`. All workers finish. No GPU recovery,
separate NVML polling or failure retry. Upstreams rechecked unchanged99e3de27/
e7c3f7b. Next use identical dyadic values to isolate BF16 versus F16 workgroup
lowering, not change the learning recipe or restart a precision sweep.
