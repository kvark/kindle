# CDP qualification complete; paired learning started

October 4, 2026. [Protocol](../experiments/2026-10-04-cdp.md) ·
[Machine-readable results](2026-10-04-cdp-qualification.json).

**Qualification is complete and reviewed; the declared six-run queue is active.**
No learning-efficiency result exists yet. The two short production smokes below
are excluded from learning evidence. Ordinary learning guards have no warning
allowance and stop the serial queue on any new warning or failure.
Learned RGB remains the default. The posterior-Tiny queue remains deferred.
The authorized follow-up passes all1,300 CDP and1,524 RGB comparisons across
four updates per arm, including raw gradients, learning-rate groups, optimizer
moments and EMA. Raw gradients use identical pre-step weights; the independent
optimizer's maximum parameter error is2.38e-7 CDP and1.19e-7 RGB. This is
component parity, not bitwise stochastic-trajectory equivalence. Tolerances
are unchanged; the failed sequential-weight comparison below remains retained.

| Check | Result |
| --- | --- |
| Approved initialization-only diagnostic | Pass: three 4KiB allocations, 256 exact outputs, teardown; one allowed warning |
| CPU validation | 111 Rust and 1,061 Python tests pass; formatting and strict workspace/binding Clippy pass |
| CDP cosine | Pass: four values and 1,024 raw derivatives against independent F64; target detached |
| Native full updates | Four synthetic updates per arm complete; not upstream parity or gameplay |
| Shared initialization | 60 world, 22 behavior and 11 slow-critic shared tensors exactly equal |
| Upstream CDP/RGB | Isolated four-step checks pass1,300/1,524 comparisons; earlier failure retained |
| Replay/restore | Both native debug tests pass, including resident frozen diagnostics |
| Production smokes | Both pass: 1,024 actions /195 updates, zero debt, finite checkpoints |
| CI278 | Linux, macOS and Python bindings pass at `bee4e64` |
| Six-run learning comparison | Started; fixed order/budget, ordinary guards |
| Frozen probes | Await reviewed learning checkpoints |

The small CDP model has 804,785 unique parameters versus RGB's 688,004. Its dense
embedding predictor has more parameters than this small RGB decoder; it can
still require less arithmetic, but no whole-agent speed claim is available.

## Corrections and retained numerical failure

The first native cosine test rejected an inference-only clamp during autodiff.
The replacement expresses the norm floor through differentiable ReLU; a CPU
autodiff check and the GPU F64 test now pass. Initial CPU shape and enum-size
lint failures remain in the build journals.

The released CDP code passes the wrong configuration level to its encoder-width
helper. This silently uses depth64, giving a 4,096-wide predictor for a
256-wide Size1M embedding. The comparison adapter supplies the selected nested
config without changing the upstream loss. Its detached visualization decoder
is deliberately omitted in both implementations. The upstream checkout is
unchanged at `a851fa3e`; this is a small-model adaptation, not the published XL
Crafter experiment.

The corrected upstream run passes 673 comparisons before failing
`step2/gradient/dyn/dynin1/bias`: maximum absolute error .03968, tensor-relative
L2 error 6.07e-6. The latter is small, but does not override the per-component
gate. The cause is **unresolved**, not an accepted rounding explanation.
The next adapter revision compares gradients at identical native pre-step
weights while independently advancing reference optimizer weights/moments/EMA;
it changes no numerical tolerance. The first isolation invocation stopped on
the warning below. The newly authorized follow-up completes all four updates.

## Warning stop and evidence

The next check was stopped at 06:31 UTC on a fresh occurrence of the exact
`_memdescAllocInternal / NV_ERR_NO_MEMORY` message. Child542471 was reaped with
SIGTERM; no new Xid/hang, unfinished child or recovery is recorded. At that stop,
no GPU queue remained active. The initialization-only allowance does not cover this
ordinary qualification invocation; no new warning was silently whitelisted.

The kernel source timestamp predates the child-spawn timestamp, while journal
receipt follows it; do not infer causality from receipt timing. Both timestamps
and the exact cursor are in the JSON. The failure snapshot's broad kernel
capture hit its output limit; the streaming warning record is intact. All eight
guard evidence audits pass with no unsealed files, including failed-job records.
Auditing a failed job's evidence does not make its guard result pass.

Raw evidence: [experiment directory](../../runs/cdp-evaluation-20261004.olfQrV),
[approved initialization](../../runs/cdp-evaluation-20261004.olfQrV/init-guard),
[cosine check](../../runs/cdp-evaluation-20261004.olfQrV/cosine-v2-queue/cosine-f64),
[partial upstream comparison](../../runs/cdp-evaluation-20261004.olfQrV/cdp-upstream-reference-v2/upstream-result.json),
[latest stop](../../runs/cdp-evaluation-20261004.olfQrV/cdp-upstream-v3-queue/cdp-upstream-reference).
These are local artifacts, not public downloads.

The user subsequently approved remaining numerical checks capped at120 seconds
and two exact-warning occurrences per process. Other failures remained fatal;
training stayed stopped until qualification was reviewed. This is not another
initialization-only loop or a training warning waiver.

CI276/277 passed macOS/Python but exposed a debug stack overflow in replay/restore.
Separating test phases reduced the local frame from972,360 to649,672 bytes but
did not fix nested construction. The vector actor now heap-owns its learner
(three changed lines), reducing large runtime stack copies without changing
learning math or stack limits. The next check caught a test-only host/device
replay mix; forecast assertions now use the real GPU actor route. All failures
remain retained. Both native debug checks now pass (CDP22.82s, RGB23.57s),
as does CI278 on all platforms. No CPU learner fallback was introduced.

## Reviewed qualification and learning launch

The successful upstream CDP follow-up records one allowed exact allocation
warning; RGB records none. Both replay checks and both production smokes record
none. All16 guard evidence audits pass, including the retained failed results.
No new recorded Xid/hang, driver recovery or separate NVML polling.

The excluded smokes use seed103, N8/B8/T16/H15/R32 and complete1,024 actions,
195 updates and zero debt each. Both have zero completed episodes and eight
unfinished tails, so supply no competence evidence. Every saved tensor is
finite (CDP173 world /66 behavior /11 slow-value; RGB215/66/11). Across776
samples, minimum estimated headroom is15,900,344,320 bytes; maximum estimated
usage647,102,464 bytes. These are Vulkan estimates, not physical/peak VRAM.

Production library SHA256:
`32353ffb5d4516aa9281e94004f7b7ca2f126c9d29bfd62e33c36f78c44f502a`.
The six-job queue `kindle-cdp-learning-20261004.service` starts with RGB1009,
then CDP1009, CDP2017, RGB2017, RGB3019 and CDP3019. Each run has200k actual
actions,49,939 updates and a60-minute deadline. No automatic retry, numerical
warning allowance, compile during timing or extra learning allocation. The
frozen diagnostics remain to be run after learning and guard review.
