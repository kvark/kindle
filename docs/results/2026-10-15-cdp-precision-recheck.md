# A historical precision policy hides a substantial CDP improvement

October15,2026. [Declaration](../experiments/2026-10-15-cdp-precision-recheck.md)
· [Data](2026-10-15-cdp-precision-recheck.json)
· [Backend plan](../backend_training_performance_plan.md).

**On the unchanged current binary, CDP Mixed→BF16 reduces complete synthetic
updates78.206→50.122ms,35.91%.** This is a large execution-path difference,
not another small kernel tweak. The Mixed policy was selected before BF16
derivatives and staged posterior reuse existed. Its historical advantage does
not survive the backend changes. No source rebuild or new shader was needed.

This is a **cost and finite-execution result, not independent numerical
qualification or learned-quality evidence**. The policies deliberately change
precision, compiled dataflow and trajectories. Keep production unchanged until
the new CDP loss/gradient and staged-execution path is independently qualified.
No game training resumes. The separate RGB/JAX54.224/26.632ms comparison remains
unchanged; CDP's exploration workload is not that JAX baseline.

| Complete-update component | Mixed | BF16 |
| --- | ---: | ---: |
| Posterior plus world training/sync |49.788ms|21.715ms|
| Imagination plus targets |23.940ms|23.941ms|
| Behavior training/sync |4.478ms|4.465ms|
| **Complete update** |**78.206ms**|**50.122ms**|

One pair,16 warmup+256 measured updates per arm. All samples remain; these
means are not confidence intervals or time-to-score evidence. No profiler,
per-dispatch timestamps, concurrent build or CPU analysis. Fixture generation,
initialization and checkpoint exports are outside timing; production update
stages, targets, synchronization and slow critic are inside.

## Why this comparison is warranted now

The [October12 comparison](2026-10-12-mixed-world-precision.md) preferred Mixed
when convolution derivatives/grouped products still retained older precision
paths. The current backend has since added BF16 derivative coverage, shared
typed matrix work, command reuse and staged world execution. Source review
finds that the CDP configuration still excludes itself from the last mechanism:

- Mixed keeps BF16 posterior/imagination/behavior, but world training uses
  F16-input/F32-storage products and a separately recomputed posterior.
- BF16 uses the existing staged world prefix/tail, with1,263 prefix dispatches
  and436 BF16 matrix descriptors in the world plan. Mixed has no such prefix
  and zero BF16 world matrix descriptors.
- The action-effects ensemble is unchanged and already uses the October11
  state-projection reuse/hidden-centering implementation. The old18-way first
  projection is not a current cost or a newly rediscovered fix.

The two mechanisms—world precision/lowering and posterior reuse—change together.
Their individual shares are not identified by this pair. We combine posterior
and world timings because their boundaries differ. Imagination/behavior are
essentially unchanged; world work explains the observed reduction. More BF16
casts or descriptors do not by themselves mean slower execution.

Both arms use native`99f8aab0`, device categorical means and command cache,
the same trained12M Breakout1009 source, centered CDP/action-effects, B32/T16/H15,
18 actions/four heads, fixture RNG701 and all original losses/rates/scales.
The source checkpoint's124,939-step history is retained; these disposable
synthetic updates are not an uninterrupted game lifetime or new experience.
No decoder, image resize, model-capacity or exploration change.

## Numerical observations, not equivalence

Two24-update smokes pass before the ordinary pair. Exact346 initial tensors
match both policies and the retained same-checkpoint initial. All first outputs,
24/272 metric reports and initial/first/final saved tensors are finite. Actual
model parameters update in every arm. Each timing run exactly reproduces its
own smoke's first24 world/behavior reports, first outputs and first346 tensors.
The compiled precision summaries show the declared policy difference.

Cross-policy trajectories are different. First deterministic-state relativeL2
is.01229 and stochastic-state.09244; first action-target.3901. These are not
roundoff-equivalence checks. Of130 actual parameter tensors,129 differ after
one update, relativeL2.0001757; all130 differ after272, relativeL2.05677.
Optimizer moments are reported separately: final first/second-moment relativeL2
1.230/.07743. The aggregate346-tensor value.8551 must not be called a model-weight
error. The initial CPU review's `changed_parameters` label included moment names
ending in weight/bias; publication corrects that label with an independent
parameter/moment split and verifies real updates, without rerunning GPU work.

Neither finiteness nor first-prefix reproducibility proves correct CDP gradients,
stochastic-policy equivalence or preserved game competence. Earlier failed F64
envelopes remain failed; no tolerance is relaxed by this cost diagnostic.

## Decision and completion

The35.91% reduction exceeds the predeclared3% usefulness threshold. Next qualify
the **existing BF16 CDP loss/gradients and staged prefix/tail execution**, with
explicit precision/rounding references, before promotion. Do not repeat timing,
start a precision sweep or restart learning. Convolution data feeding and
dependent execution still leave the RGB/JAX root-cause goal open; this does not
claim parity or make that remaining work disappear.

Four guards pass, one recorded allocation warning,592 synthetic updates and
zero game actions/recovery/separate NVML queries. The whole automated service
finishes05:14:52 UTC in92.70s; CPU smoke/timing reviews7.27/8.53s, publication
parameter audit1.52s. CPU review peaks859.9MiB under1CPU/2GiB/no swap; GPU work
has no CPU quota. Sampled Vulkan estimated headroom11,317,280,768bytes Mixed /
12,378,701,824 BF16, not physical or peak-free VRAM. All454 source seals pass;
Meganeura99e3de27/Bladee7c3f7b upstream heads are unchanged. No worker remains.

Artifacts: `runs/cdp-precision-recheck-20261015.SDZLNuhK`. Production pins,
learning defaults, historical checkpoints/evidence and neighboring work stay
unchanged. No independent numerical or learning-equivalence claim.
