# Convolution's instruction overhead covers the complete family

October16,2026. [Declaration](../experiments/2026-10-16-jax-convolution-counters.md)
· [Compact evidence](2026-10-16-jax-convolution-counters.json).

Across **all23 actual convolution products**, native executes **10.296B warp
instructions versus JAX2.420B:4.255×**, despite fewer tensor instructions.
This is a complete-family instruction result, **not a4.255× speedup opportunity**.
The excess appears in every pass, not just the two previously captured encoder
derivatives. It contrasts with [recurrence matrices](2026-10-16-jax-matrix-counters.md),
where native executes7.42% fewer warp instructions than JAX.

Keep qualified isolated58aff452: **RGB46.199/27.223ms inclusive JAX,1.70×**.
No new ordinary timing, backend build, production promotion or Atari restart.

## What the complete counters establish

| Main-kernel counter | Native | Actual JAX | Native/JAX |
| --- | ---: | ---: | ---: |
| Warp instructions | 10.296B | 2.420B | 4.255× |
| ALU-pipe instructions | 6.819B | .936B | 7.287× |
| Tensor-pipe instructions | 180.224M | 227.524M | .792× |
| LSU-pipe instructions | 774.339M | 236.699M | 3.271× |
| CTAs | 283,600 | 108,540 | 2.613× |
| Shared-read wavefronts, LSU path | 579.876M | 485.794M | 1.194× |
| Shared-write wavefronts, LSU path | 605.470M | 20.976M | 28.864× |
| Local read/write bytes | 0 /0 | 78.643MB /1.589MB | — |

These are independently measured counters, not an additive latency budget or
source-line attribution. Shared-write ratios describe the named LSU path, not
all shared-memory transfers: JAX also executes60,533,384 `LDGSTS`,90,026,992
`LDSM`,8,577,304 `LDGDEPBAR` and8,361,496 `DEPBAR` instructions. Direct async
copy and register-mediated load/store are different feeding paths; NVIDIA's
[triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-fills-go-through-the-register-file)
describes that distinction. These counts do not revive the rejected pipeline.

Three JAX forward launches have measured local-memory traffic while native
main kernels have none. Eliminating spilling or maximizing occupancy is not
sufficient to reproduce the library's performance. No new utilization claim.

| Complete pass | Products | Native warp instructions | JAX | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Forward | 8 | 3.711B | .941B | 3.942× |
| Input gradient | 7 | 1.755B | .497B | 3.531× |
| Weight gradient | 8 | 4.830B | .981B | 4.922× |

Partitioning all23 by their existing native implementation also reconciles:
three typed forwards contribute3.141B versus.751B; six typed derivatives
5.559B versus1.170B; fourteen packed products1.596B versus.498B. The nine typed
occurrences account for84.50% of native warp instructions and86.07% of the
instruction-count excess, **not that fraction of time**. Packed products still
have3.20× instruction overhead. Ambiguous layer owners always belong to the
same implementation/pass, so this partition does not invent layer ownership.

The eighteen actual JAX helper launches contribute another61.033M warp
instructions, reported separately. Old standalone native transform ranges do
not describe today's composed image helpers and are not used as their counterpart.

## Coverage and controls

CPU checks reconcile all23 current weight/gradient paths,20 retained native
counter contracts and all28,064 kernels across eight retained JAX updates.
Every update has the same41 convolution launches:23 main and18 helper. Five
repeated identity classes retain ambiguous individual ownership. Full semantic
HLO matches the current JAX control. Native fixtures are independently checked
synthetic operands; JAX uses actual model values. Output layout/precision,
padding, dW compensation and workspace differ; this is not learning parity.

Collection has three separately reviewed allocations:

| Allocation | Profiles | Outcome |
| --- | ---: | --- |
| Original complete counter selector | 36 | Failed completeness; all18 helpers, only18 mains |
| Broader launch-metadata observation | 41 | Exact complete launch identities; one pass each |
| Short missing-name counter selector | 7 | Five missing mains plus two overlapping controls |

The original report/API/log all contain36; offline exact-name, anchored-regex,
family-regex and substring import tests match the retained missing kernel.
The fresh metadata observation establishes all41 actual identities. The short
selector then recovers all five missing identities. The internal profiler cause
is **not established**; the first allocation remains failed, not retroactively
successful. Original36 profiles take precedence; both new controls are retained.
Their CTA/warp/tensor counts match exactly. All16 counters match one control;
the other differs by725 shared-read conflicts/wavefronts, retained as observed.

All three full benchmarks match272 reports,288 initial/final tensor fingerprints
and full HLO exactly. Combined coverage passes23+18; independent publication
recomputes every selected CSV counter, all memberships and240 source seals.
Main instruction counters are checked against the report API in base units.

Two guards pass, one fails completeness; **zero allocation warnings**, no
recovery or separate NVML polling. Total816 synthetic updates,43 counter profiles/
774 counter passes plus41 metadata passes, zero native GPU calls/builds/game
actions. Last GPU service ends**16:25:10UTC**,70.555s/3.4GiB peak host memory;
publication audit finishes16:28:01UTC. No GPU worker remains. Counter-capture
times, uncontrolled-cache/clock warnings and host-memory replay are retained,
not ordinary timing results.

CPU failures also remain: a section-name regex matches `DisplayName`; its
anchored repair exposes non-metadata counters in LaunchStats, so the unexecuted
proposal is narrowed to nine explicit launch metrics before GPU use. A review
starts before the guard receipt; the terminal review then finds a singular
`pass` parsing defect. The combined reader misses automatic CSV megabyte units;
offline base-unit exports fix exact API comparisons. The independent aggregate
initially includes50 native counters instead of the declared16; membership-only
repair passes. Original sources/logs/exports remain, with no GPU repeat for any
reader repair. The CLI documents [unit selection and explicit metrics](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#profile).

## Decision

Convolution feeding/control work is a measured structural deficit across the
family; generic matrix instruction count, host overhead alone, precision alone
and zero spilling do not explain parity. The [complete-update trace](2026-10-16-current-update-attribution.md)
still measures39.650ms inside GPU passes, above the whole27.223ms JAX update.
These boundaries cannot be subtracted into independent savings.

Next CPU-account operand loads, predicates, address updates, cooperative stores
and compensation in the **current nine typed and fourteen packed occurrences**.
Incremental cursors, exact division helpers and typed word staging already exist;
do not rediscover or propose them as absent. Relate remaining source structures
to emitted operations and the captured library schedules before choosing one
complete-family mechanism. Counter ratios alone do not identify source lines.
No geometry/precision/shape-constant sweep, rejected pipeline, prior timing,
Naga feature, new GPU allocation or promotion follows this result. The full
root-cause/parity goal remains open.

Artifacts: `runs/jax-convolution-counters-20261016.1F8DAYTk`.

Publication correction: the following [load-contract audit](2026-10-16-convolution-load-contract.md)
finds that commit9c21174 appended follow-up notes to this experiment's sealed
declaration after the source audit. The original bytes are restored; the edited
version and initial failed audit remain retained. Runtime/results are unchanged,
and the separate census/supplement declarations still document both allocations.
