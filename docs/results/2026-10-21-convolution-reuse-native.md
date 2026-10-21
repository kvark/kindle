# Fixed convolution reuse integrates and passes actual-model prefix checks

October 21, 2026. The [instruction-screen candidate](2026-10-21-convolution-reuse.md)
now runs through isolated native Meganeura + Blade, not just the primitive
observer. **RGB and CDP each reproduce all24 reports, first targets and five
checkpoint pairs exactly.** No ordinary cost comparison or raw-gradient/staging
qualification has run. Keep79bc7cec/RGB42.808ms/CDP40.563ms/1.642× JAX as the
measured performance boundary; no production promotion or Atari restart.

## What changed

The isolated `ConvolutionReuse` kernel/variant carries the fixed shape through
selection, workgroup geometry and pipeline identity. Shared pipeline identity
also includes the explicit subgroup requirement. The generator uses the
qualified incremental gathers and fragment stores, with original uniform
bindings, K order, precision and weight-gradient compensation. All23 actual
native-generated modules are **byte-identical to the GPU-qualified SPIR-V**:
seven changed products and sixteen unchanged products.

The opt-in selector requires the qualified BF16 arithmetic, 5×5/stride1 shape,
supported16×16×16 matrix operations, enough shared memory and explicitly
supported full32 subgroups. Other cases retain their existing GPU kernels.
Cached selections reset to the incumbent on an unsupported device or when
disabled. Default subgroup width alone is never treated as a guarantee.
The underlying generator remains type-aware; only BF16 has this GPU qualification.

Blade enables the required features and attaches the full-subgroup flag/size
chain only for an explicit request. All other compute pipelines retain default
flags. No Naga change. Latest Meganeura main remainsa75c38e. Both flags
`MEGANEURA_CONVOLUTION_WORDS=1` and `MEGANEURA_CONVOLUTION_REUSE=1` are required
for selection, alongside the retained complete-update flags in the sealed queue.

## Actual-model evidence

| Check | RGB | CDP |
| --- | ---: | ---: |
| Changed world products |7|3|
| Exact metric reports |24|24|
| Exact tensors in each checkpoint pair |292|346|
| Checkpoint pairs: initial, first, final, saved, restored |5|5|
| Added / removed unique runtime shader bodies |4 / 3|3 / 3|

Both posterior and world-gradient exports describe the same staged world plan;
do not double-count its products. All five exported plan/pipeline sets per
workload reconcile against retained controls. Only declared kernels, grids and
labels change; constants, bindings, state, groups and all non-world programs
remain exact. Runtime shader additions match qualified generated bodies;
removals are only replaced incumbent bodies. RGB has206 unique sources, CDP225.
First outputs, precision summaries, command-cache metadata and world groups
match exactly. Saved/restored tensor state is not a replay/RNG/live-belief resume.

CPU tests cover resource/capability fallback, cached fallback, key separation,
serialization,131,072 dimension pairs with no extra output padding, and actual
captured-plan integration. Three retained compiler tests also pass after fixing
their stale APIs. Seven CPU tests total; strict Meganeura/Kindle library Clippy
passes. The preceding Blade tests/lint remain qualified separately.

Two host guards pass with **zero allocation warnings**,48 synthetic updates,
zero game actions and zero guard NVML queries. GPU work ends14:08:31UTC; the
independent prefix review finishes14:08:57. Final705-receipt/state review passes
14:11:21UTC. All workers are terminal. Candidate executablec8aeba28 is isolated
in `runs/convolution-reuse-native-20261021.foJSYVFJ`, not the retained incumbent.

## Failures and storage accounting

Preserve Cargo's refusal to run dependency development tests from Kindle's
workspace. The repaired harness uses Meganeura's own manifest and a separate
test lock seeded from Kindle's exact lock; offline additions are development/
optional dependencies, not a production dependency refresh. The native build
still uses Kindle's unchanged lock.

Two inherited tests referenced an obsolete grouped-generator API and a removed
BF16 capability name. The current grouped generator and unified BF16 capability
repair those test calls; original sources/errors remain. A later observer
expected one matching compiler test but the filter selected two: both passed.
Its failed service remains, with count correction reviewed offline, no rerun.

The prefix disk estimate was too small: approximately2.4GiB of outputs reduced
free space to194MiB, without a failed write. Closed-cache deduplication reclaimed
1,911,484,416 bytes across1,659 paths; post-review prefix deduplication reclaimed
961,978,368 bytes across17 paths. All paths/bytes remain, verified before and
after hardlinking; these shared archives are immutable. The active compiler
cache was never hardlinked. Its earlier relocation preserved the prior source
tree and qualified CPU test executable. About1.1GiB remains free.

Before another allocation, account for every checkpoint, full plan and raw
gradient output plus a reserve—not just reports or the expected final artifact.
Do not delete unique evidence or mutate shared cache inodes.

## Next

Qualify actual-model raw gradients and staged/whole execution, including changed
late targets and prefix isolation, for both RGB and CDP. Then declare one
complete-update control/candidate cost comparison with all transforms, partials,
epilogues and host work included. The earlier21.37% composed instruction
reduction is not a newly measured native speedup. No geometry sweep, production
promotion or Atari restart follows automatically. Five-game CDP mastery remains
unfinished.
