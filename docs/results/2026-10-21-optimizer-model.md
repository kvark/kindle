# Optimizer repair passes complete-model checks; Metal CI needs a follow-up

October21,2026. The shader-only repair in Meganeura draftPR275/4bdcb8d now
passes full RGB/CDP qualification on the5080. It has negligible measured
complete-update overhead. **This is not yet a portable adoption:** Metal CI
exposes an additional arithmetic-reassociation problem described below.

## Actual learner numerics

The isolated retained backend changes only its optimizer shader. Candidate
`6b621e5c` retains the six flags, BF16/B32/T16/H15, configurations, checkpoints,
compiler policies and all other kernels of controld3ff4948. Current upstream
remainsa75c38e. This is not a silent complete-upstream refresh or production repin.

Four24-update runs compare the ordinary candidate with an optional test observer
for RGB and CDP. At updates1/24 the observer exports every optimized gradient
after in-place AGC and snapshots each training Session immediately before/after
its optimizer update. One world microbatch/chunk guarantees these are the actual
gradient buffers consumed, not unused accumulation buffers. Backward is not
rerun; clipping, moments, target generation and production updates are unchanged.

An independent F64 oracle with F32 stored-state rounding checks **160,998,750
parameter/momentum/variance values**. All pass the existing bounds: variance
`1e-5*abs(reference)+2*2^-149`; momentum/parameter`1e-5*(1+abs(reference))`.
Maximum error/bound is.01193/.04710/.00594 respectively. Optimizer clocks,
per-prefix learning rates, warmup, bias corrections and untouched nonoptimized
parameters are checked. Negative controls reject the original zero-variance
and oversized-momentum failures. **These model snapshots contain zero subnormal
variances:** earlier component tests, not this fixture, establish that repair.

Observer transfer is exact: both24-report trajectories, first imagined targets,
initial/first/final292 RGB/346 CDP tensors, all five program/pipeline pairs and
205/225 runtime shader multisets. Command caches record and replay in both.
Four guards pass, zero warnings,96 synthetic learner updates, zero actions.
CPU review completes08:07:14UTC; all544 source receipts verify.

## Complete-update overhead

After numerical review, four ordinary272-update runs use the original executables
without observers/tracing. All256 measured samples per arm are retained.

| Workload | Fresh old control | Repaired candidate | Overhead |
| --- | ---: | ---: | ---: |
| RGB12M/B32/T16/BF16 |48.403ms|48.446ms|.090%|
| CDP12M/B32/T16/BF16 |41.817ms|41.886ms|.164%|

Both stay within the declared5% regression threshold; this is a correctness fix,
not a speedup. All272 reports and initial/first/final states happen to match
the old control exactly on these fixtures. Each candidate's first24 reports,
first targets and first tensors also match its qualified short run. Full plans/
pipelines match. Exactly one runtime source differs per workload: replacing the
entire original adam.wgsl body with the qualified repaired body reproduces the
new emitted source exactly; every other source/multiplicity is unchanged.

Four guards pass, zero warnings,1,088 synthetic updates, zero actions. Final CPU
audit08:12:48UTC verifies549 receipts. Across numerical and timing allocations:
eight passing guards,1,184 synthetic updates, no GPU worker remains. No current
learning result, intrinsic-reward change or Atari restart. The JAX parity and
five-game objective remain unfinished.

Two setup-script assertions failed before compilation (status spelling and an
over-broad source anchor); originals remain. The first timing observer wrongly
demanded equality of the intentionally changed optimizer shader and stays failed.
A separate CPU-only exact-body-replacement audit passes; no GPU repeat or relaxed
numerical/cost gate. Prior checkpoint byte-order failures also remain retained.

Artifacts: `runs/optimizer-model-20261021.lNTSllDW`; declarations, optional Rust
observer, original/corrected audits, raw gradients, full states and source seals.

## Metal CI: a concrete second issue

[Meganeura CI1047](https://github.com/kvark/meganeura/actions/runs/39565123470)
passes quality, Linux/lavapipe, Windows compile, MSRV and dependency audit. Metal
fails two added regressions: normal variance atg=1e-6/beta2=.999/step2, and
Inf handling atbeta2=0. Do not report the PR green or the fix portable.

The original arithmetic `beta*old + (1-beta)*g*g` gives F32 bits`0x27100a91`.
The FMA form `fma(beta, old-g*g, g*g)` gives`0x27100a10`, **exactly the failed
Metal value**. The same rearrangement changes the beta0/old0/gInf result from
Inf to NaN. Pinned Bladeeae0c6f7 explicitly enables Metal Fast math; Apple's
[shader optimization guidance](https://devstreaming-cdn.apple.com/videos/wwdc/2016/606oluchfgwakjbymy8/606/606_advanced_metal_shader_optimization.pdf?dl=1)
describes reassociation/nonfinite tradeoffs. This is an exact arithmetic
counterexample and strong explanation, not captured Metal machine-code proof.

The isolated follow-up supplies F32`1-beta1`/`1-beta2` independently as uniforms,
preventing the compiler from using that algebraic identity. This is invariant
coefficient preparation like existing bias corrections, not a CPU optimizer.
It needs no global fast-math policy change, Blade/Naga edit or tolerance increase.
The uniform becomes56 rather than52bytes; checkpoint/state layout is unchanged.
Original regression tests stay unchanged. Qualify locally, then use actual Metal
CI before claiming success. The complete-model evidence above belongs to4bd;
it does not automatically qualify the follow-up uniform interface.

Follow-up`5d916851` is pushed to the same draftPR275. The two uniform-layout
tests, rustfmt, regression Clippy and all three unchanged GPU tests pass locally;
53 Session updates,192 exact checkpoint values/metadata, one guard/zero warnings,
338 receipts. JSON metadata key order still differs in three files; no whole-file
identity claim. Review finishes08:17:58UTC. [CI1048](https://github.com/kvark/meganeura/actions/runs/39568247325)
is confirmed in progress, **not yet a Metal pass**. Original CI1047 stays failed.

Follow-up artifacts: `runs/optimizer-coefficients-20261021.pcZGh5zI`. No local
worker remains. Clear this portability issue, then qualify the final interface
before resuming performance attribution or any learning campaign. No automatic
production promotion or Atari restart.
