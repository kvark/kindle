# Direct policy gradients into causal Tiny

The optional `actor_critic_gradient` route is **numerically qualified**, not yet
a learning result. [Compact evidence](2026-10-03-policy-tiny-qualification.json).
Sourcec278ba5, Meganeura13b19d33 over upstream6268ea5, Bladee349cddf.
Latest Meganeura main remains6268ea5 at17:34 UTC; no newer fix was skipped.

## What passes

- With **only policy loss** as the world graph output, all148 used Tiny tensors
  have finite nonzero raw gradients and change after an optimizer step. Reward,
  value, KL, prediction and regularizer losses cannot supply those gradients.
  Existing actor/critic weights remain frozen in this graph.
- Three complete joint-Tiny Size1M/N8/B8/T16/H15/R32/microbatch1 updates pass,
  including current-weight replay and live-prefix cache refresh: mean2.207s.
  This is synthetic qualification, not sustained throughput or learning.
- The checkpoint restores all148 encoder tensors exactly and acts with zero
  learner updates. Config retains the enabled gradient option.
- Four fixed native RGB/RSSM updates match pinned upstream Dreamer with
  `ac_grads=True`: **1,524 value/raw-gradient/common-gradient optimizer/EMA
  comparisons pass**. Largest gradient relative-L2 error is3.312e-4 on a tiny
  actor normalization gradient (maximum absolute error1.61e-13). Tiny backward
  separately has the independent [F64 reference](2026-10-03-joint-tiny-backward.md).
- Five ordinary guards pass on RTX5080/driver580.178.04. Minimum sampled native
  estimated budget headroom is8.897GB; this is not physical-free or peak VRAM.
  No new kernel warning, NVML polling, host recovery or unfinished child.

CPU tests:103 Rust and1,021 Python tests, strict workspace/Python Clippy and
formatting pass. [CI263](https://github.com/kvark/kindle/actions/runs/37139107069)
passes. The runner exposes `--actor-critic-gradient`; restoration uses its saved
config and rejects overrides. Existing default-off learning is unchanged.

## Scope and limits

Like upstream, actor/imagined-value gradients reach **initial posterior states**;
later imagined states and return/action targets are detached. This is not full
differentiation through a simulated future. Initial-state losses are averaged
over the complete imagination horizon. The behavior optimizer still owns its
heads; the world optimizer owns Tiny/RSSM. No new optimizer or learner service.

Nonzero isolated policy gradients do not prove useful representation learning,
nor that this signal is large enough beside the combined world objective. The
completed [task-only learning screen](2026-10-03-joint-tiny-learning.md) did not
enable this route. A separate three-seed comparison is declared in the
[protocol](../experiments/2026-10-02-joint-tiny.md), reusing those task-only controls.

## Retained setup failures

Two reference launches failed before GPU initialization: the old JAX environment
lacked `safetensors`, then `vulkan`. Both logs remain. Installed safetensors0.7.0
and vulkan1.3.275.1, checked all reference imports without initializing a device,
then used a new declaration/output. No native numerical change or recovery was
needed. Do not confuse these with GPU failures. The earlier CPU build service
also failed before compilation because its PATH lacked Cargo; a login-shell
service then passed. CI262's broad manual-test filter and missing audit fields
were fixed before CI263. No failure is removed or relabelled as a clean attempt.

Local evidence: `runs/policy-tiny-qualification-20261003.jbuEc8/`.
