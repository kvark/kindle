# Same-driver counters confirm the convolution feeding deficit

October21,2026. [Data](2026-10-21-jax-current-convolution.json).

**Current native convolution executes4.280× JAX's warp instructions,6.926×
its ALU-pipe instructions and3.860× its LSU-pipe instructions, despite fewer
tensor instructions.** Both captures use driver615.71.09 on the same boot.
This closes the stale-JAX-counter comparison, not the performance or learning
goal. Keep isolated79bc7cec: RGB42.808ms/CDP40.563ms; RGB/JAX remains1.642×
using the separately measured ordinary inclusive JAX26.066ms. No new timing,
production change or Atari restart.

| All23 main convolution products | Native | JAX | Ratio |
| --- | ---: | ---: | ---: |
| Warp instructions |10.302B|2.407B|4.280×|
| ALU-pipe instructions |6.481B|.936B|6.926×|
| LSU-pipe instructions |1.140B|.295B|3.860×|
| Tensor-pipe instructions |180.224M|227.524M|.792×|
| Workgroups |283,600|108,540|2.613×|

Nine typed products account for8.761B versus1.926B JAX warp instructions,4.548×;
fourteen packed products1.541B versus.481B,3.204×. Every product is included.
Ambiguous JAX identities retain their candidate owners, all in the same native
family. Eighteen JAX helpers add62.869M warp instructions separately; native
helpers are not included in this comparison.

JAX main opcodes include60,533,384 `LDGSTS`,90,026,992 `LDSM`,8,577,304
`LDGDEPBAR` and8,361,496 `DEPBAR`, exactly the historical counts. Total JAX
warp count is.5135% lower than historical, while LSU is higher and uniform-pipe
counts lower. Equal launch metadata or selected opcode counts do not prove
machine-code identity or driver-only causality.

## Qualification and scope

Unchanged upstreame3f02248/JAX0.6.2/cuDNN9.24 RGB12M/B32/T16 BF16 matches
the current-driver control's272 reports,288 initial and final tensor fingerprints,
and full semantic compiled workload exactly. Erasing profiler/logging additions
recovers the ordinary mathematical benchmark byte-for-byte. Packages are unchanged.
All41 launch names/grids/blocks/register/shared identities match the historical
complete census. Eight retained trace windows validate the broad selector; three
malformed membership tests reject.

Native data comes from the qualified actual-model grouped capture, not rejected
B-layout variants. Its17 convolution regions cover exactly23 products. All16
compared counters reconcile against raw retained data; all656 JAX CSV metric
values agree with the report API in base units. Independent family/typed/packed
sums pass. No individual cost is invented inside a grouped range.

**Native and JAX do not use the same kernels or tiling.** The retained
[contract](2026-10-15-convolution-contract.md) documents32/64 native reuse versus
larger pass-specific library tiles and three-to-five-stage feeding. Unchanged
JAX identities preserve that comparison. Inputs, intermediate rounding/layouts,
padding, compensation and workspaces differ. There is no native SASS capture,
source-line attribution, additive latency budget or4.280× speedup promise.
Pipe counters are not a disjoint time decomposition.

## Execution and failures

One GPU allocation captures41 profiles/738 counter passes during update16;
272 synthetic updates total, zero native GPU calls/game actions. One host guard
passes, zero allocation warnings/guard NVML calls/recovery. Sampled estimated
headroom7,774,797,824bytes is not physical free/peak VRAM. The1m40.867s controller
ends12:56:27UTC. All workers are terminal. Capture timings are not ordinary speed.

Preserve three failures and their offline repairs: preparation lacked the guard's
sibling import path; the controller failed after successful capture because its
summary omitted the inherited headroom field; the first reviewer expected shared
counters in a compact native summary that omitted them. Preparation passed before
launch. The original benchmark headroom passes the unchanged2GiB gate, and the
corrected reviewer uses sealed raw native counters with exact checks of all shared
summary fields. The controller remains failed. No GPU repeat or tolerance change.

Independent publication checks59 artifact receipts plus149 preflight source
receipts at12:58:27UTC. CPU work uses1CPU/2GiB/no swap; Meganeura maina75c38e
is rechecked unchanged. Artifacts: `runs/jax-current-convolution-20261021.I5nEf2Pi`.

## Next

The structural deficit survives the driver refresh and predicate fix. Resume
the [existing output-storage proof](2026-10-15-output-contract.md): derive one
complete convolution reuse/operand-storage policy with bounded shared storage
and accumulator live state, preserving compensation, precision and all23
contracts. Quantify repeated operand/address work before implementation; use
existing fragment-output operations, not a new compiler prerequisite. This is
a CPU design, not a tile sweep, automatic GPU allocation or revival of the
failed separate-loader pipeline. The unknown three-channel library tile stays
unknown. Larger reuse is a hypothesis, not a proved speedup. CDP imagination's
separate cost and five-game mastery remain unfinished.
