# Attribute the non-matrix execution cost before implementing more fusion

October14,2026. **Complete:** [result](../results/2026-10-14-materialization-cost.md).
Allocation1.432ms materialization/2.329ms row-reduction/1.631ms other; exact
plans/pipelines/outputs/292 tensors, one guard, zero warnings. Follow the
[comparable-work audit](../results/2026-10-14-executable-work.md).
No new learner, optimization, shader algorithm or ordinary timing campaign.

The fixed graph attributes5.545ms to non-matrix/non-convolution work. Determine
whether materialized layouts/casts or row/backward/scalar computation dominates
before implementing another compiler rule. Raw dispatch counts are not a
speedup estimate; wide head concatenation remains an uncosted opportunity.

Use the retained RGB12M/B8/T16/BF16 staged control, one original preparatory
update, then disable optimizers, gradient accumulation and clipping. Keep all
matrix/convolution dispatches active. Classify all other actual dispatches into:

- Materialization: casts, concatenation, split, transpose, channel packing and
  the broadcast-only global-average-pool gradient.
- Row/reduction: generated reductions, explicit sum/mean, RMSNorm derivatives
  and BCE reduction. Retain their fused scalar work in that same category.
- Other: generated pointwise, SiLU backward and image pooling/upsampling.

Unknown shader families fail classification. Save per-dispatch category,
bindings, output sizes and original group membership. This is a deliberately
coarse attribution, not a claim every row operation can share one kernel.

Run all eight active subsets in three rotated orders for each of the same four
sessions: eight warmups/64 full baseline calls, then two warmup/16 measured
calls per subset,64 final full calls. Zero workgroups suppress a selected
dispatch; retain every binding, dependency and command. Restore full work before
each subset and verify exact full output fingerprints after every round.
Exact292 tensors/first original report/config/plan/pipeline inventory must
reproduce the retained control; masked outputs are never actor/optimizer inputs.
Publish per-order effects and interactions, not additive savings or pure kernel
durations. An unmodified full-graph timing brackets the diagnostic.

One600-second guarded GPU process, expected5080,>=2GiB sampled Vulkan estimated
headroom, no CPU quota, NVML polling or concurrent build. Persistent service
has Restart=no/KillMode=control-group and a bounded deadline. Record allocation
warnings; API/numerical/hard faults stop for review. No automatic follow-up,
masked-gradient optimizer writes, game actions or installed-runtime change.
Preparation/analysis: one CPU,2GiB,zero swap. Preserve complete sources and failures.

Upstream rechecked before05:57 UTC: Meganeura2214412 unchanged; Bladee7c3f7b
and Nagacbbac959 add Metal BF16 only (no Vulkan/SPIR-V changes). Keep qualified
96c1dc7/a97559b for this unchanged-graph attribution, not a combined refresh.
Artifacts: `runs/materialization-cost-20261014.yLfoQTsc`.
