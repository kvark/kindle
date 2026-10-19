# Repair the selector harness's device initialization

The first [selector qualification](2026-10-19-typed-selector-gpu.md) stops at its
device assertion: `init_gpu_context()` uses default options, not the declared
device environment, and selects the integrated AMD adapter. No matrix dispatch,
search, learner update or action executes. Retain the failed guard, logs, binary,
queue and source in `runs/typed-selector-20261019.MhXVhJTs`.

In a separate `device-fix` source/cache, change only the test initializer to
`init_gpu_context_with(GpuOptions::from_env())`. Keep the device assertion and
all12 unstarted fixture/search/output/cache contracts, gates and process/resource
bounds unchanged. Recheck the CPU fixtures and actual incumbent shaders, then
run the corrected declaration once. This is a reviewed harness repair, not a
matrix retry or evidence about NVIDIA health. No full-model search follows
automatically; review qualification first. Controllers use immutable binary and
source paths in the repaired allocation.
