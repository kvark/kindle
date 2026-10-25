# Complete-update convolution attribution

Declared October 12, 2026 before execution. The existing PR248 convolution
kernel wins in isolation but not in complete updates. Corrected multi-submission
tracing now makes a bounded attribution possible. No game actions or learning
campaign; no new kernel or precision sweep.

Use the same diagnostic Kindle source (integration0010772 plus shader counts
and matrix-capability export). Control is Meganeura7ba39d8; candidate d2de660
adds only the unchanged PR248/7553463 convolution change. Bladeb52965b and
Naga forkc9fd6a85a are shared. CLI path overrides and exact source archives are
retained; embedded679ea87 is not the effective source revision.

First run existing independent F64/F16-rounded convolution derivative and
strict-range fallback checks on the stacked source, bounded to120 seconds.
Then compare the retained CDP12M/B32 and RGB12M/B8 fixed fixtures with F16
permission,16 warmup plus256 measured full updates each. Both arms keep BF16
selection off. Require exact initial346/292 tensors, finite reports/targets/
checkpoints, changed masters, native RTX5080, at least2 GiB sampled budget
headroom, positive cooperative dispatch selection and passing guards.

Capture a second pair of CDP runs with complete-submission GPU timing; require
four world chunks per measured update and exact targets/reports/tensors against
each arm's own ordinary run. Report diagnostic fence overhead separately.
GPU pass intervals are neither SM utilization nor pure compute time. Each
complete-update process is bounded to600 seconds. Ordinary timing has no
compilation, GPU profiling or concurrent heavy work. All jobs use the existing
serialized non-restarting systemd/host-guard mechanism; retain allocation
warnings and stop/review other failures before follow-up.

Acceptance is an ordinary complete-update win, not the earlier isolated kernel
gain. A neutral result stays unadopted. Use the trace and actual selected
capabilities to explain it before changing more code. Artifacts:
`runs/conv-attribution-20261012.j7K9T35C`.
