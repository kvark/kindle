# Allocation observability after the third GPU fault

Status: **CPU-compiled diagnostic component only; no GPU work or adoption**.
The [pixel incident](2026-09-14-pixel-initialization-incident.md) leaves boot
372a5604 faulted. A host-only boot-ID read still matches it; no NVML query or
recovery action is performed. The five-game objective, throughput priority and
held Pong roots remain unchanged.

The isolated Blade branch `exp/vulkan-allocation-observe-20260914` at
**4d8c8bca21f5faf15d97e5781d53064952abc60f** starts at f6f2729e. Fresh upstream
reads in the [CPU preparation](../../runs/vulkan-allocation-observability-20260914.Hx1FRJ/declaration.md)
still find Meganeura 428fc2d / Blade 68a23e49. All f6-to-68 changes remain in
the unused renderer; no native upstream fix is omitted. The user's original
Blade worktree and every existing Kindle/Meganeura package remain unchanged.

## What the instrumentation adds

The existing Meganeura trace identifies logical slots but cannot identify their
underlying Vulkan memory objects or shared suballocations. The isolated patch
adds immediately flushed records around:

- Vulkan buffer creation and binding;
- the `gpu-alloc` request and host-access selection;
- returned memory object, offset, block size, actual memory type/properties,
  heap index, heap size and heap flags.

Process/request IDs and buffer/allocation links distinguish repeated `buf_0`
names. Names are hex-encoded so embedded newlines cannot inject trace records.
Records retain host realtime and relative elapsed time, not calibrated GPU
execution timestamps. Failed writes/flushes stop the diagnostic.

Memory types and heaps are copied from the **existing context initialization
query**. The patch adds no GPU query, allocation-policy/flag change, queue
operation, arithmetic change or resource-lifetime change. Existing Vulkan error
handling remains intact. The always-on diagnostic is isolated, not a new public
production API or an overhead-qualified profiler.

`allocate` brackets an allocator request, not each internal `vkAllocateMemory`
call. `host_access` includes no-map device-local paths and already-mapped
suballocations; it is not a count of Vulkan mapping calls. Memory object handles
may be reused after free. No allocation-lifetime, actual-residency, safety or
fault-origin claim follows from these fields alone.

## Completed CPU checks

The one-shot preparation completes seven command lifecycles: both upstream
reads, toolchain, formatting, offline lock resolution, release library tests
and release Clippy with warnings denied. All **13 CPU tests pass**: eight existing
command/descriptor/flag tests and five new trace tests covering escaping,
single-line flushing, write/flush failure and distinct request IDs. Their bodies
were inspected before execution; none creates a GPU context.

An independent read-only audit reverifies **188 source pins, 191 total input
pins and 21 output pins**. The fresh release test artifact names the isolated
`blade-graphics/src/lib.rs`, has no optional features, and its modification time
lies within the recorded build window. Qualification also re-audits after the
source commit, with unchanged bytes. The build uses a private target, 100% CPU
quota, 2 GiB host-memory cap and zero swap; no measured peak is claimed. Preserve
the initial formatting failure and completed writer; only `prepare_cpu.py --audit`
is reusable.

This standalone Blade lock and executable are **not a source-matched Kindle
runtime package**. No device callbacks, placement records, trace reader or GPU
overhead have been validated on hardware. Matched control instrumentation,
Meganeura constant-upload boundaries, an integrated initialization-only fixture
and a separately declared guarded comparison remain necessary after recovery.
There is no GPU launcher or automatic successor. Do not replay the quarantined
candidate, restart an old queue or infer a fix from these CPU tests.
