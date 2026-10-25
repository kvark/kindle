# Copy composition observer repair

October16,2026. All eight qualification GPU guards pass. The raw-gradient review
passes, but the short-update CPU reviewer stops on its assumption that serialized
`plan.groups` contains live runtime groups. `Session` deliberately takes those
ranges out of the plan. The unchanged test harness exports world groups separately
as `world-groups.json`; non-world group membership was not exported.

Retain the first reader, failure log, queues and GPU evidence. Repair only the
CPU program reviewer: require the serialized groups to be empty, audit the actual
world export for full membership/dependencies, and report non-world membership
as unavailable. Do not invent groups or treat the failed assertion as numerical
evidence. Every originally declared exact target/metric/gradient/state, protected
read, complete plan/constant/kernel/pipeline check remains required. Reuse the
sealed CPU selections only after matching retained complete-plan identities.

No GPU repeat, source rebuild, changed numerical gate or new cohort. Recheck
the original short-update results in one CPU/2GiB/zero swap with single-threaded
BLAS. Full timing remains conditional on this review passing.
