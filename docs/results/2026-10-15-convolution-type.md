# The slow workgroup convolution is not specific to BF16

October15,2026. [Declaration](../experiments/2026-10-15-convolution-type.md)
· [Data](2026-10-15-convolution-type.json).

Fresh BF16/F16 operations take**7.43375/7.43317ms**, a0.0078% difference;
queued wall differs0.0416%. All input values are exactly representable in both
formats. The shared typed shader, packs,256x32 workgroup tile,64-wide per-tap
reduction, buffers/barriers and F32 accumulation/output remain unchanged.
Both complete outputs match the native reference exactly. Close this typed
lowering hypothesis; there is no evidence of a BF16-specific slowdown here.

This does not qualify F16 for real training gradients or identify identical
machine code. The [contiguous-output test](2026-10-15-convolution-store.md) also
fails to improve the operation. Next distinguish general tensor boundary/
addressing work from the actual matrix calculation using explicit zero halo
and fixed valid strides, with the GPU transform cost included. No dtype/geometry
sweep, production change or automatic compiler feature work.

Three guards pass,zero warnings;295 operations/885 shader dispatches,
33,587,200 exact output checks/393 F64 checks across264 distinct flat indices (265 shape/index cases),
including changed input/restore and F16 upload round-trips. B1 validation and
the device's explicit workgroup-F16 support pass.49 sealed source hashes;
minimum sampled Vulkan headroom15,999,565,824 bytes,not physical free/peak.
Raw evidence: `runs/convolution-type-20261015.XU8DtcWM`. No workers remain.
Zero game actions/learner updates/NVML polling/recovery. Full update remains
60.041/26.632ms,not remeasured; root-cause and quality goals remain open.
