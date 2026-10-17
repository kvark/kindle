# Upstream workgroup support is not a new convolution result

October16,2026. [Declaration](../experiments/2026-10-16-workgroup-contract.md) ·
[Compact evidence](2026-10-16-workgroup-contract.json).

The properties-only query passes on the5080/580.178.04. All seven optional
Cooperative Matrix2 features are advertised, including tensor addressing and
block loads. Blade4845d652 enables workgroup scope/flexible dimensions only;
its pinned Naga13f582dd retains pointer/stride/row-major matrix load/store IR,
not tensor layouts/views. Meganeura f648a453 is unchanged. No runtime repin.

These are distinct optional interfaces, not a measured feeding improvement.
The extension provides compiler-managed staging for workgroup matrices; the
driver reports8KiB reserved storage and48KiB explicit workgroup storage.
[Vulkan extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_matrix2.html),
[property definition](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeMatrix2PropertiesNV.html).
All23 current convolution occurrences fit the advertised BF16/F16 dimensions
at128 invocations. Their existing explicit arrays plus the reservation fit
17,408/34,816/33,792B for9/10/4 occurrences. This is a resource bound, not
compilation, actual hidden allocation, instruction lowering or speed evidence.
Six invalid type/dimension/workgroup cases reject;25 flexible and11 subgroup
property records are retained without selecting favorable shapes.

## Keep the rejected route closed

The [October15 family](2026-10-15-convolution-family.md) already used
`coopMatLoadTensorNV`, tensor layouts/views and workgroup matrices. It lost
41.29% across the complete forward/dX family, including aligned-channel cases.
New Blade exposure does not overturn that result or justify implementing its
missing tensor interface. No scope-only retry, compiler project, kernel sweep,
new candidate or integration follows this audit. The library's feeding advantage
remains unresolved; supported features alone do not reproduce its schedule.

One guard passes, zero warnings/NVML queries/recovery; no logical device, shader,
dispatch, optimizer update or action. Sampled estimated Vulkan budget headroom
is16,594,173,952B, not physical free/peak VRAM. The first CPU reader failed by
using historical dispatch indices against the current image-view plan. Preserve
that source/log; the corrected full shader/parameter/grid membership audit
reconciles all23 unique occurrences and72 seals without another GPU query.
All workers exit. Artifacts: `runs/workgroup-contract-20261016.8xpKH4aw`.

Keep qualified834241b6 and RGB48.068/JAX27.223ms,1.77×. Prior batching remains
unqualified; next resolve its finite-precision reference discrepancy using
retained data. Production/Atari stay unchanged and the broad parity goal is open.
