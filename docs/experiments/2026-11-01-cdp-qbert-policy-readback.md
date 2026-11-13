# Exact batched actor readout on retained Qbert failure windows

The [teacher intervention](../results/2026-11-01-cdp-qbert-recoverability.md)
avoids25/32 selected later-play losses from learner states. Select a frozen
diagnostic of the learned actor on those original failure trajectories. Measure
its teacher-action preferences and confidence before selecting another loss or
learner. A teacher's successful continuation does not establish that every
individual teacher command is causally better.

## Additive native readout

`VectorAgent.policy_inputs_and_logits()` copies existing row-major actor inputs
and raw logits after `act`, before `observe`. It rejects missing pending actions.
It performs no actor forward pass, normalization, sampling or learner update.
Logits precede temperature and categorical unimix. Both the core method and
Python binding are additive; the production action and learning paths are
unchanged. The GPU test covers greedy, sampled and overridden actions, repeated
readbacks, invalid call timing, counters, recurrent features and RNG state.

Build from the retained447fe50 source plus only these identical readout, binding
and test additions. Preserve the current branch implementation while using an
isolated source archive under `/x/Temp/kindle-vector-policy-readback-20261101`.
The original context-only patch check fails because later posterior-mode APIs
are absent in447fe50; a separately recorded patch applies the exact added blocks
at the corresponding older locations. No build or GPU attempt precedes that
repair. Both patches and the failure are retained.

The resulting runtime SHA256 is
`f864fe4b7af23a8a2c7db15edcec5a13febebf58ee3a8bbf3383429454ae1669`.
All130 Rust CPU tests and585 packaged Python CPU tests pass; strict native and
binding Clippy pass. Python tests construct no native agent. Keep the original
learner runtime unchanged. These CPU checks do not yet qualify GPU execution.

## Fixed capture and gates

Use the completed Qbert encoder4e-5 final checkpoint,12M48/Auto/N8/B16/T16/H15/
R16/expert100,seed4013,actor-unimix0. Replay the original batched greedy path
through20,584 actions/82,324 frames, preserving all eight lanes and resets.
That prefix includes all16,613 actions in the first natural episode of every
lane and some later-episode tail actions needed to keep the batch intact.
Do not replace the original sampled categorical posterior with a modal state.

Run three guarded stages in order: the tiny native readout test; a256-action
batched prefix; and the full20,584-action prefix. Stop on any failure,without a
retry, cap increase or implicit promotion to training. The two capture workers
have120/180-second internal caps and150/210-second guard caps; the native test
has120 seconds and the enclosing service600 seconds. Use the current AMD host
identity and retained guard environment. Require no other Kindle service before
launch. No native learner is selected by this protocol.

Every native greedy action must equal its saved original, without forcing it.
Every reconstructed input/output pixel, reward, boundary and cumulative frame
count must match the completed CPU replay. Preserve the original vector reset
order. Require finite actor arrays and raw-logit argmax equal to the selected
action. Repeated readbacks at the first tick and each intervention-origin tick
must be identical and preserve counters. Save complete logits for the16,613
selected first-episode decisions and features/logits for all1,024 decisions in
the32 preceding-loss windows. Extra prefix actions remain validated but do not
enter the failure-window metrics.

At the first128 decisions, compare F64 actor probabilities from the copied
features and checkpoint weights with softmax/unimix of the native logits:
maximum absolute difference≤1e-4 and identical greedy choices. Require the
smoke/full prefix logits and all128 qualification features to match exactly.
Require all357 checkpoint tensors invariant andzero learner updates after each
capture. Recomputed live batched states are observed directly; unavailable
original hidden-state arrays cannot be compared independently.

Report teacher agreement, teacher/recorded-action probabilities, entropy,
teacher rank, teacher cross-entropy, probability below.1 and NOOP counts. Split
the fixed windows by teacher rescue outcome, origin, last action, early/late
half and lane. Teacher labels and rescue outcomes come only from the completed
intervention artifact. This is selected failure-state evidence on eight reused
episodes,not held-out policy accuracy or a new game score. Its CE is not the
same distribution as training replay CE. No readout is fitted in this capture,
and no causal architecture or single-action attribution follows from agreement.

Four pure analysis tests pass. CPU preflight2.243s verifies source/checkpoint/
runtime identity,exact prefix geometry and all32 origins/1,024 labels without
emulator or native construction. Root:
`/x/Temp/kindle-qbert-policy-readback-20261101`. Source trajectories and checkpoints
remain unchanged. No new loss, learner or automatic successor is selected.

## Completion — November 1

Sourcefd1c586,525 inputs. All three guarded stages and the original service
succeed05:31:49 UTC after29.478s. The full capture takes17.631s,matching all20,584
actions/82,324 frames withzero updates and357 invariant tensors;the128 opening
actor comparisons differ by at most4.762e-10. Smoke/full prefix equality passes.
The separate CPU collector's null-message failure is retained;the.661s repair
completes05:34:08 without repeating GPU/capture/analysis stages. See the
[completed result](../results/2026-11-01-cdp-qbert-policy-readback.md) for fixed
metrics and the explicitly post-hoc NOOP/movement decomposition. No new learner
or loss is selected. The build cache is released;all portable artifacts remain.
