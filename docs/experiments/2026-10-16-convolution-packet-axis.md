# Preserve shared address work in typed convolution staging

October 16, 2026. Follow the [word-staging counters](../results/2026-10-16-word-staging.md).
Production, Atari and the qualified `834241b6` control remain unchanged.

The dW result removed most measured shared conflicts but improved time by only
9.1%. Uniform-pipe instructions fell from 315.42M to .46M while ALU instructions
rose from 507.57M to 765.88M. This suggests duplicated per-lane address work;
it does not identify individual machine instructions. Test one packet-axis
change, not a tile, selector or precision search.

Store both 32x32 operands column-major in the existing padded typed `vec4`
shared arrays. Each local invocation owns two packets: column `lane % 32`,
first row `(lane / 32) * 4 + part * 16`, four consecutive rows per packet.
A therefore packs four M coordinates at one K; B packs four K coordinates at
one N. Retain pitch40 for F16/BF16 and36 for F32, 128 invocations, original
32x32x32 tiles, reduction order, compensation, splits, barriers and outputs.
Use existing column-major cooperative loads. No vector global-load claim.
All nine scalar-staged occurrences change; the14 packed occurrences do not.

Correctness uses the local-invocation-index bijection, **not** a relationship
between local IDs and subgroup IDs. Permuting subgroup membership cannot alter
packet ownership or stored values. The expected uniform-instruction benefit
is conditional on driver lowering, and must be tested with counters. Current
Blade uses SPIR-V1.3 and does not require full subgroups. Do not substitute
`SubgroupId * SubgroupSize + SubgroupLocalInvocationId`: its dense coverage
requires the full-subgroup contract, as documented in the
[Vulkan built-in specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).
No new subgroup assumption, compiler feature, Blade feature or driver change.

CPU gate: independently prove packet ownership and column-major fragment reads,
padding, masked tails, A/B source addresses and incremental cursor transitions.
Exercise arbitrary local-ID permutations and a partial-subgroup counterexample;
separate mathematical correctness from conditional uniformity. Cover actual
contracts plus small rectangular/strided/tail cases. Build and validate all152
typed/control variants with the pinned Naga and runtime-aligned SPIR-V options.
Require123 changed/29 unchanged shaders and unchanged plan selections. Review
CPU evidence before declaring any GPU queue.

After that review, one separately sealed complete-family allocation may reuse
the existing independent F64 gates, exact paired outputs and changed-input
restoration. Ordinary16-warmup/128-sample timings are separate from41-metric/
34-pass counter collections. Include every20 contract/23 occurrence, transforms,
reductions, warnings and sample. Require5% aggregate complete-operation benefit
before proposing full updates. A failure does not authorize a favorable subset,
layout/tile sweep, numerical relaxation or retry.

Preparation and analysis: one CPU,2GiB,no swap. GPU workers, if declared later:
serialized guarded persistent services, no CPU quota, Restart=no,
KillMode=control-group,120-second process limits and bounded queue deadlines.
Expected5080/580.178.04 and >=2GiB sampled Vulkan estimated headroom. Retain
allocation warnings; API/numerical failures, hard faults and deadlines stop for
review. No separate NVML polling, clock changes, game actions or learner updates.

Upstream checked: Meganeura `f648a453`, Blade `01be8446`. Blade's new per-entry
point/override resolution fix takes the unchanged fast path for a single-entry,
override-free module. Prove those conditions for these shaders rather than
silently refreshing the comparison. No source edits in the dirty user Blade tree.

Artifacts: `runs/convolution-packet-axis-20261016.JfluRtMn`.

## CPU review and bounded family allocation

06:16UTC: build/Clippy and304 offline modules pass. All152 fresh control WGSL
and SPIR-V modules equal the retained word-staging control. The123 modified
modules keep four cooperative loads, two products, four vector shared stores,
three workgroup barriers and9,216/13,312B explicit shared storage. All entry
points are single-entry/override-free. Address proof covers5,559,244 B addresses,
131,072 A addresses,5,248,059 cursor transitions,5,243,557 gradient transitions,
40,960 ownership/fragment checks and164,767 masked packet values. Four local-ID
permutations preserve ownership; negative tests distinguish row/column layout,
wrong pitch, duplicate ownership and unsafe dense subgroup IDs. These are CPU
proofs, not measured hardware uniformity or numerical GPU qualification.

Proceed with the declared complete20-contract/23-occurrence paired numerical,
ordinary and counter phases, separately started after review of each phase.
Each phase has40 guarded processes and a30-minute controller limit. Reuse the
existing SDK bridge and41-metric configuration; do not rebuild/instrument the
driver or touch production. The new flag is
`MEGANEURA_CONVOLUTION_PACKET_AXIS=1`; both arms retain word staging.
Full updates are not part of this allocation. Numerical failure ends the
affected phase for review; an ordinary performance rejection still permits
the already-declared counter phase to explain the rejected mechanism.
