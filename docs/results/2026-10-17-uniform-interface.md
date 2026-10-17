# A parameter-binding choice changes native instruction cost

October17,2026. [Evidence](2026-10-17-uniform-interface.json) ·
[Preceding audit](2026-10-17-model-dependencies.md) ·
[Backend plan](../backend_training_performance_plan.md).

For the first encoder convolution, changing only parameter storage from an
**inline uniform block to a uniform buffer removes14.05% of warp instructions
and22.77% of ALU instructions**, with identical shader bytes, inputs and outputs.
A fixed four-condition comparison separates this from explicitly requesting
full32-lane subgroups: that setting changes none of the instruction anchors.

This identifies an interface-conditioned instruction cost on5080/580.178.04,
not ordinary latency, a whole-family effect or a complete explanation of JAX's
advantage. Keep isolated native`e45b9c43`: complete RGB12M/B32/T16 remains
**43.197ms versus inclusive JAX27.223ms,1.59×**. Production and Atari stay stopped.

## The failed control exposed the difference

Fixed SPIR-V normalization passes all20 numerical contracts/23 occurrences:
3,365,632 F64 comparisons,548,578,560 exact zero checks and275,972,096 independently
rechecked paired output values. Original/normalized outputs are bit-exact,
including changed-input/restoration checks. Precision and numerical gates stay.

Its first counter capture completes50 passes and passes the GPU guard, but the
original shader executes299,892,736 warp instructions versus retained native
348,913,664. The declared native anchor fails, so the controller stops. Its19
remaining captures are unstarted. **That allocation remains failed**; the
normalization's aggregate5% instruction gate is not evaluated. No timing or
normalization integration follows from this partial result.

CPU re-emission with both retained library generations reconciles all20 source
functions and actual Blade options. All function instructions and barriers are
exact. Blade's name-based resource mapping exchanges packed inputs' descriptor
numbers; the failed typed-forward case has no permutation and is byte-exact.
Meganeura leaves ray tracing disabled, so Blade emits SPIR-V1.3. Neither source
version nor shader arithmetic explains this control mismatch.

Blade maps supported64-byte `Plain` data to an inline uniform block; the direct
observer used a UBO. Both are legal backing choices for uniform shader data,
but need not have identical implementation cost. [Vulkan resource specification](https://docs.vulkan.org/spec/latest/chapters/descriptors.html).
The observer also requested full32-lane subgroups; Blade used default stage
settings. These differences define a separate fixed control-reconciliation
test, not a subgroup-size, tile, precision or compiler-flag search.

## Controlled reconciliation

All four conditions use the original unnormalized CNN0 forward module, B512,
the same two-active-image fixture, parameters,128-invocation workgroups,
buffers and barriers. Queried subgroupSize/min/max are32; no local-ID/subgroup
mapping is assumed. This is one deliberately selected **failed-control case**,
not a favorable subset used to claim whole-family performance.

| Parameter storage | Subgroup request | Warp instructions | ALU instructions |
| --- | --- | ---: | ---: |
| UBO | Explicit full32 |299,892,736|172,490,752|
| UBO | Default |299,892,736|172,490,752|
| Inline | Explicit full32 |348,913,664|223,346,688|
| Inline | Default |348,913,664|223,346,688|

All conditions launch65,536 CTAs and execute3,145,728 tensor instructions.
The UBO conditions exactly reproduce the direct observer; inline conditions
exactly reproduce native. Eight endpoint anchors pass. Uniform-pipe counts are
6,029,312 UBO versus3,145,728 inline; this does not identify individual machine
instructions or prove uniformity alone explains the cost. No native assembly
query, driver change or ordinary timing is performed.

Initial/changed/restored full outputs, guards and immutable inputs pass.
The two fresh guards check2,097,152 F64 values,534,773,760 zeros and402,653,184
independently paired output values. Their58-metric/50-pass capture reevaluates
offline exactly. Numerical tolerances, shader operations and model are unchanged.
All450 publication source seals pass. GPU work finishes01:37:38UTC at the latest;
independent publication review finishes**01:39:15UTC**. No worker remains.

Across both allocations:23 GPU guards pass,15 allocation warnings are recorded,
450 main-kernel dispatches/100 counter passes, zero learner updates/actions/
ordinary timing samples. The original controller's failed CPU review remains
failed, not a GPU fault or wedge.

## Decision and retained failures

Do not resume the old normalization queue, search optimizer passes or claim a
14% learner speedup. First audit the existing Blade UBO fallback against actual
RGB/CDP programs: scratch allocation, descriptor extents, cached-command
lifetimes, parameter refresh and all conversion/upload work. Then declare a
same-native, complete-workload numerical/ordinary comparison if that audit
passes. Use the existing fallback, not a new push-constant/descriptor backend
or favorable convolution selector. Full latency is the next relevant test;
the wider root-cause/parity goal remains open.

Retain the original CPU missing-debug-name, descriptor-order and cross-case
negative-extent failures. Exact name re-emission and same-case extent tests
repair those before GPU work. The context helper then assumes nonexistent
derivative output names; actual `dst` names repair it. Its first native-options
report incorrectly assumes SPIR-V1.4: corrected1.3 emission passes with original
reports retained. No completed GPU cohort is repeated for those CPU repairs.
The initial output-validator limitation remains in the preceding result. A
commentary percentage briefly reversed the denominator; this evidence uses the
correct14.05% reduction (16.35% reverse increase). A documentation patch had a
malformed hunk; no experiment or receipt changed during its repair.

Artifacts: `runs/spirv-normalization-gpu-20261017.OEnwWyDu` retains the stopped
family, fixtures, exact emission and failures;
`runs/normalization-context-20261017.c1LhUqjX` retains the separate declaration,
four-condition observer, guards, raw counters and independent publication audit.
