# Diagnose the remaining prior-gradient discrepancy

October16,2026. Continue the [forward-tape diagnosis](../results/2026-10-16-prior-tape.md),
not a new performance allocation. Prior batching remains unqualified; keep834241b6
and RGB48.068/JAX27.223ms,1.77×. Production/Atari remain unchanged.

Two fixed CPU hypotheses use the captured forward tape: F64 derivatives and
source-ordered F32 backward matrices/SiLU/RMSNorm, with a shared F64 KL derivative.
Neither is fitted to gradient outputs; GPU intrinsic approximations are not
assumed bit-exact. Across16,265,216 checked values, the F32 hypothesis clears all
second-variant gradient gates in both arms. First-variant prior1/prior0 weights
still have3/130 pointwise failures in each arm. Original failures/gates remain.

One separately declared diagnosis-only process captures the first retained12M
batched variant's full forward/backward tape. Reuse the retained execution plan,
parameters and sealed backend libraries. Verify full plan/pipeline identity;
capture selected KL, SiLU/RMSNorm and matrix/quantization boundaries at their
existing barrier groups and actual buffer lifetimes. Keep original allocations,
kernel selection, rounding and aliases. Require all outputs/raw gradients and
parameters to match the retained run exactly. One complete raw backward,
zero optimizer updates/actions. No second variant or original cohort retry.

Independently evaluate local analytic derivatives and products on their observed
rounded inputs. Retain original pointwise2e-5+.02*abs(reference) and relativeL2<=.01
gates, exact BF16 rounding checks and explicit activation/free-nat alignment.
Keep local derivative qualification distinct from end-to-end F64 agreement.
Trace which conversion boundaries differ from the old reference and whether
those differences account for the failed final gradients; do not merely accept
small output error. Preserve every unexplained value and negative check.

No threshold relaxation, replacement qualification cohort, complete-world check,
update timing, production promotion or Atari restart follows automatically.
This resolves the failed test's cause before deciding the appropriate next
qualification. Compiler/tile/precision sweeps and new Naga work remain closed.

Preparation/review: one CPU,2GiB/no swap, bounded persistent services. Compile
only the small standalone reader against the existing libraries. GPU: existing
host guard,180-second process/240-second service deadlines, no CPU quota,
expected5080/580.178.04, >=2GiB sampled estimated Vulkan headroom. Record standalone
allocation warnings; API/hard faults, identity/numerical mismatches and deadlines
stop for review. No GPU recovery, clocks or separate NVML polling. Upstream
f648a453/4845d652 is unchanged. Artifacts: `runs/prior-gradient-20261016.4efbUy2T`.
