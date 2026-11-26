# Exact Breakout policy confidence on retained failure trajectories

The stronger encoder4e-5 policy often prefers a different action from the pixel
teacher by a substantial margin. Across12,763 disagreements, mean probability
is.634 for its recorded action and.228 for the teacher's;2,713 teacher commands
receive probability below.1. The fixed failure windows show similar preferences.
This establishes the original policy's confidence on its own trajectories. It
does not establish that each teacher action is optimal, explain the causal
learning limitation, or select a replacement loss, head, teacher or learner.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-policy-readback.md)
completes with37,016/37,016 original actions reproduced without forcing,
zero updates and357 invariant checkpoint tensors in each capture. Full capture
takes30.672s. All268 pins, exact prefix comparisons, pixel/outcome/life checks,
CPU/native probability checks and host guards pass. Original service and terminal
collection succeed; no stage repeats. No new gameplay score is measured.

## Original-policy measurements

Use the retained12M64/Auto/B16/T16/H15/R16/expert100/encoder4e-5 checkpoint,
trained200k actions/12,477 updates onseed1009. The original greedy evaluation
seed4000001009 uses sampled categorical posteriors and1% actor unimix. Its
retained score remains104.667, withzero of24 two-wall successes.

The31,161 selected decisions are the firstthree natural episodes in each ofeight
lanes. The full37,016-action capture preserves subsequent tails and reset order.
Probabilities are native raw-logit softmax with the checkpoint's1% actor unimix.
Exact teacher-agreement counts reproduce the earlier CPU phase replay.

| Selected decisions | Rows | Exact teacher agreement | Mean teacher probability | Mean chosen probability | Teacher CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 31,161 | 59.04% | .547 | .713 | .899 |
| Disagreements | 12,763 | 0% | .228 | .634 | 1.766 |
| Teacher requests a direction | 18,642 | 59.48% | .528 | .685 | .911 |
| Far descending ball | 8,048 | 52.40% | .487 | .692 | 1.086 |
| Near descending ball | 6,582 | 59.86% | .553 | .728 | .903 |

Among18,642 directional teacher commands, the policy chooses NOOP3,668 times
and the opposite direction3,794 times. Near the paddle, those counts are1,147
and429 among4,006 directional commands. These describe disagreement with this
teacher; they are not verified errors against optimal actions. Mean teacher
rank is2.304 among all disagreements, and policy entropy there is.838 nats.

## Fixed failure windows

The120 earlier intervention origins are clipped to the current life:117 windows
of32 actions andthree of24, totaling3,816 distinct rows. Earlier teacher takeover
avoids108/120 selected loss endpoints. This intervention changes an entire
feedback sequence, so it cannot assign causal credit to each intermediate label.

| Fixed window rows | Rows | Exact teacher agreement | Mean teacher probability | Mean chosen probability | Teacher CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 3,816 | 63.89% | .603 | .750 | .793 |
| Disagreements | 1,378 | 0% | .224 | .634 | 1.783 |
| Teacher requests a direction | 2,059 | 60.22% | .543 | .701 | .898 |
| Near descending ball | 873 | 56.59% | .540 | .723 | .932 |
| Windows rescued by earlier takeover | 3,432 | 64.13% | .604 | .750 | .786 |
| Windows not rescued | 384 | 61.72% | .589 | .754 | .855 |

Window disagreements include294 teacher commands assigned probability below.1.
Among2,059 directional labels,410 policy actions are NOOP and383 are opposite.
The rescued directional subset still has361 NOOP and345 opposite commands among
1,858 rows, with60.66% exact agreement. Rescue remains a sequence-level fact.

The final action in all120 windows is already a FIRE label and agrees with the
policy at mean probability.987. This is the end of the recorded life-loss window,
not the earlier last controllable paddle contact. The late half also contains
648 FIRE labels versus159 in the early half. Its higher aggregate agreement
(67.71% versus60.02%) must not be interpreted as better paddle control near loss.
Phase and directional subsets retain this distinction. Failure-window CE and
replay-training CE use different states and are not a matched generalization test.

## Execution and retained evidence

The readout copies existing batched actor inputs and raw logits after act and
before observe; it performs no forward pass, sampling or update. Runtime
`/x/Temp/kindle-vector-policy-readback-20261101/runtime` reuses the qualified
additive API over native447fe50, SHA256
`f864fe4b7af23a8a2c7db15edcec5a13febebf58ee3a8bbf3383429454ae1669`.
The earlier same-host native/RNG/counter qualification is reused without a
new build or native test. The ordinary original logs contain no hidden arrays,
so equality to unavailable original hidden arrays cannot be asserted.

The256-action prefix takes4.509606s; full capture takes30.672371s for37,016
actions/148,038 frames. All37,016 input pixels, actions, rewards, boundaries,
cumulative frame counts and life counts match. The37,008 available output images
match; eight final output images were not recorded and remain unverified. Life
counts are checks, never model inputs. Features/logits are finite, raw-logit
argmax matches each original action, and115 repeated vector readbacks preserve
values and counters. Multiple origins can share one vector tick.

All128 opening F64/native actor probabilities agree within9.214e-7 (cap1e-4),
with128 identical greedy choices. Smoke/full256-row logits and128 qualification
features/probabilities match exactly. Both captures preserve all357 tensors and
zero updates. All268 inputs and both host guards pass. Fixed analysis takes.074s;
six CPU fixture groups previously passed in6.385s while preserving all562 active
Qbert pins and its original process. Retain that co-host cost without correction.

Selection follows Qbert's original terminal closure and an idle-host check at
20:57:52 UTC November1, source8481d9c. Service
`kindle-breakout-policy-readback-20261101` launches20:57:55.126, originalPID845612,
invocation831e211938b14d1da284cdefb14e46c6. It succeeds20:58:35.048 after39.919s
wall time. Original CPU terminal collection completes20:58:53.597 in.165s.
There is no failure, cap change, retry, repeated analysis/capture/collector or
new learner. No Kindle service or GPU allocation remains active at closure.

| Evidence | SHA256 |
| --- | --- |
| Declaration | `a5eb69a9fee32f0821bdc4b28d3a58af3cc1bfff39c435da0c92ce15aa2c6538` |
| Full capture | `1c87b5170b5b02bdf0584dd5ec252294ea81478a700865630b0ea8db50a84b63` |
| Analysis | `08f345ea9686356c6757446a4bb5e3cd3cb271a8d64f619bdc682b90858f50f2` |
| Original terminal completion | `63c339a048122366e1757e61e3f45fd16ec7994f298535bd25ecbe9269db5703` |

Root `/x/Temp/kindle-breakout-policy-readback-20261101` is complete; do not rerun.
No successor experiment is selected by this diagnostic. The five-game goal
remains unfinished. The result supports examining how the existing learner
acquires sustained control, while leaving representation, objective conflict
and the teacher's limitations unresolved.
