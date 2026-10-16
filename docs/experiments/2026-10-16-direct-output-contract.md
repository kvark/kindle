# Direct F32 convolution output: CPU contract

Audit the existing checked and packed convolution epilogues before allocating
GPU work. Keep native `e45b9c43`, its backend, all 23 products, original tiles,
K order, compensation, rounding, external layouts and consumers unchanged.

One semantic rule: an F32, affine row-major output can use the existing
`coopStoreT` directly when the entire workgroup tile is in bounds and the
pointer/row stride meet Vulkan's 16-byte alignment requirement. Keep the
original scratch/scatter path for tails, reduced outputs and packed dW's
channel/tap permutation. Eligibility must be workgroup-uniform; subgroup
ownership must not assume a particular mapping of local invocation IDs.

This is not the rejected larger-tile/fragment-scratch policy or a new tensor
store/compiler API. Retain scratch declarations for fallback tiles. Prove
complete and unique output ownership, every actual batch/split, alignment,
fallback coverage and unchanged K-loop/bindings. Include odd extents, all
power-of-two subgroup widths through 128, and negative mutations. Validate the
actual Naga/SPIR-V output and independently reconcile current world membership.
Count eligible output work, not an inferred latency benefit.

Also finish the isolated upstream `12df95cd..41a15d31` audit: distinguish new
search/calibration from the ordinary execution path. Do not repin or run search.

CPU only, persistent `Restart=no` services, one CPU, 2 GiB, no swap and a
720-second deadline. Keep failures and sources. No GPU calls, backend rebuild,
timing, production changes or learner restart. A passing proof permits review
of a separately declared complete-family test, not automatic execution.

Artifacts: `runs/convolution-direct-output-20261016.93l7AixM`.
