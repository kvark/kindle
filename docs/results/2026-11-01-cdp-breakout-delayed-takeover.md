# Most early Breakout mistakes remain recoverable after further policy actions

The teacher still averts105/120 recorded losses after eight more recorded policy
actions from the first disagreement, and91/120 after sixteen. Immediate takeover
averts108/120. All comparisons end at the same original recorded loss action
for each window; these are local continuations, not new game scores.

| Recorded policy actions before takeover | Survive original loss endpoint | Already lost before takeover |
|---|---:|---:|
| 0, retained teacher-first baseline | 108/120 | 0 |
| 1, retained policy-first baseline | 108/120 | 0 |
| 2 | 110/120 | 0 |
| 4 | 106/120 | 0 |
| 8 | 105/120 | 0 |
| 16 | 91/120 | 2 |

The first eight recorded commands often leave a recoverable state under this
teacher. Continued policy control through sixteen actions reduces the aggregate
rescue count, but many losses remain avoidable even then. This supports examining
later approach/control decisions rather than treating the earliest disagreement
as the decisive error. It does not establish an architecture, forecasting,
representation or training-coverage cause.

Rescue is not monotonic. Between delays8 and16,84 windows survive both,21 only
the earlier takeover,seven only the later takeover and eight neither. The teacher
is imperfect and sees different states/history at each takeover; its failure
does not prove optimal irrecoverability. The two losses occurring before delay16
remain in the full denominator. Total reward through each original endpoint,
including unchanged recorded prefixes, is366/357/360/365/363/359 for delays
0/1/2/4/8/16. Reward and life preservation are separate outcomes.

All120 windows are retained from the same24 development episodes conditioned on
observed losses, with no outcome/confidence/phase filtering. The fixed delays
count actions, not disagreements. The source encoder4e-5 greedy score remains
104.667 withzero two-wall successes. No native policy is evaluated on the
counterfactual branches; the same feedback-aware pixel teacher controls them.
No new learner, loss, policy replacement or successor is selected.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-delayed-takeover.md)
completes478 new teacher branches in55.638s CPU /57.311s controller, closing
07:12:23 UTC November1, sourcec9fbf9f. It reuses240 completed delay0/1 baseline
branches without rerunning them. All37,016 source actions /148,038 frames,
pixels, labels, lives, rewards, boundaries, outcomes, partial tails and source
aggregate metrics reproduce. Every delayed recorded suffix matches its retained
original. Eight reversed-order checks and parent-state/teacher-memory invariants
pass. New branches execute106,714 emulator frames; recorded/reversed validations
execute45,388. An independent recount from saved branch lives confirms all four
new matched-endpoint counts.

Twenty-one CPU tests,3.093s preflight,334 diagnostic pins and359 active Qbert
inputs pass. Original QbertPID760870 / invocation
`a0b219d7702b4c82a65e9b33498c3544` remains unchanged. No native import, GPU context,
learner update, failed command or retry occurs. Tests and analysis co-host on
CPU; retain their timing without correcting Qbert training wall time.

Root `/x/Temp/kindle-breakout-delayed-takeover-20261101`. The diagnostic is
complete; do not rerun it. The original Qbert8039 learner continues under its
existing recovery and frozen-assessment observers. The five-game goal is unfinished.

| Artifact | SHA256 |
|---|---|
| Declaration | `232e24899facf9aefef33122051b43795bad9d6608174635c455354bb042d3f3` |
| Review | `ebff213b67ed586e76ed34bda2d167690d218fb4ddc9cc3425752055b92e0fe4` |
| Full branch records | `086732ec8a30cb1ac1bbb64582a6f5dac89dfb8724b2f27ba772b730d13b21ae` |
| Completion | `054035708fdb76673217feb9b6bbafc3346c20e26a7f88ae9ff4bbad79c97e00` |
