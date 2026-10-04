# CDP qualification: isolated upstream comparisons pass

October 4, 2026. [Protocol](../experiments/2026-10-04-cdp.md) ·
[Machine-readable results](2026-10-04-cdp-qualification.json).

**Learning remains stopped; numerical qualification is approved to resume.**
No CDP gameplay or learning-efficiency result exists.
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
| Replay/restore | RGB passes; CDP follow-up in progress |
| Production smokes | Unstarted |
| Six-run learning comparison and frozen probes | Unstarted |

The small CDP model has 804,785 unique parameters versus RGB's 688,004. Its dense
embedding predictor has more parameters than this small RGB decoder; it can
still require less arithmetic, but no whole-agent speed claim is available.

## Corrections and unresolved numerical result

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
SIGTERM; no new Xid/hang, unfinished child or recovery is recorded. No local GPU
queue remains active. The initialization-only allowance does not cover this
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
and two exact-warning occurrences per process. Other failures remain fatal;
training stays stopped until qualification is reviewed. This is not another
initialization-only loop or a training warning waiver.

CI276/277 passed macOS/Python but exposed a debug stack overflow in replay/restore.
Separating test phases reduced the local frame from972,360 to649,672 bytes but
did not fix nested construction. The vector actor now heap-owns its learner
(three changed lines), reducing large runtime stack copies without changing
learning math or stack limits. The next check caught a test-only host/device
replay mix; forecast assertions now use the real GPU actor route. All failures
remain retained. RGB now passes on the native debug build; CDP and CI follow-up
are in progress. No CPU learner fallback was introduced.

Resolve full-update parity and the remaining qualification gates
before the already declared six learning runs. No CPU learner workaround,
driver recovery, old queue, Phase3 or swarm campaign is introduced.
