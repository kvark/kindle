# Complete updates after word-staging qualification

October 16, 2026. The [fixed family comparison](2026-10-16-convolution-word-staging.md)
passes all20 contracts/23 occurrences:2.690432M F64 values, exact paired full
outputs/plans, and ordinary complete operations22.493→19.112ms (15.03% less).
The separate native counter capture is still running at declaration time.
Require its complete independent audit before building/running this follow-up.
No extra kernel variant, selector or timing sweep.

Use the existing image-view B32/T16 RGB12M and centered CDP12M fixed-checkpoint
recipes, BF16, device means and cached recording. One binary and one changed
mechanism: `MEGANEURA_CONVOLUTION_WORDS=0/1`. Image views stay enabled in both.
Keep model, inputs, precision, optimizer, splits, barriers and K order unchanged.
The only Kindle code changes are the isolated dependency paths and an optional
construction-time shader dump. No dumps/profiling in the measured update loop.

One control/candidate pair per workload:16 warmup+256 measured updates each,
272 reports, first targets and initial/first/final292 RGB or346 CDP tensors.
Require exact paired and retained image-view-control values, including the
retained first24-report prefix. Numerical family qualification already passes;
do not allocate another duplicate24-update pair. This is synthetic-update
qualification, not an independent full-model F64 oracle or learning parity.

Export full plans, pipeline keys and actual construction-time WGSL. Independently
check semantic plan/constants/groups and non-convolution modules unchanged;
only the scalar convolution staging modules may differ. Retain all256 timing
samples and first/last32 means. Propose retaining the isolated candidate only
if complete RGB improves>=3% and CDP regresses<=2%; no retry or favorable
workload selection if it fails. Report complete timings against inclusive
JAX27.222569ms, not the sum of standalone kernel savings. Production and Atari
remain stopped/unchanged regardless of the result.

Four serial guarded jobs, <=240 seconds each, persistent1200-second service,
Restart=no/KillMode=control-group, no CPU quota on GPU workers. Current boot and
driver580.178.04, expected5080 and >=2GiB sampled Vulkan estimated headroom.
Retain allocation warnings; API/numerical failures, hard faults and deadlines
stop for review. No NVML poller, clock/driver change or game actions. CPU build
and streaming audit:1CPU/2GiB/no swap, outside matched timings. Preserve all
earlier CPU observer/constructor failures. Same artifact root as the family.

Pre-launch build review: Kindle's direct Blade dependency must use the same
diagnostic-handle source as Meganeura. The first CPU build rejects two same-version
Blade paths before compiling. Retain its manifest/log; the reviewed build only
aligns this direct path. No runtime kernel change or GPU retry.
