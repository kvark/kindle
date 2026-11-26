# Teacher takeover before the stronger Breakout policy's recorded losses

The [own-policy replay](../results/2026-11-01-cdp-encoder4e5-representation.md#fixed-windows-locate-lost-lives-without-assigning-a-causal-action-error)
finds120 lost lives in24 selected greedy episodes. Every loss has a recent
near-descending pixel detection;86 last such actions already match the teacher.
Select a CPU counterfactual to measure whether the current teacher can avoid
these recorded losses when it takes over earlier or near the paddle. This is
architecture diagnosis within the existing five-game goal. Qbert continues.

Use all120 losses from the fixed encoder4e-5 final-greedy recording, without
selecting by action agreement, confidence, reward or branch outcome. The early
origin is the start of the32-action window ending at the loss, clipped to the
current life's start. The late origin is the recorded last near-descending row;
require it inside that same life/window. The saved inputs yield240 distinct
origins. This is deliberately conditioned on observed failures, not an average
policy evaluation or an independent test cohort.

At each origin, copy the recorded-history pixel teacher, obtain its proposal,
then branch by process copy so ALE sticky-action memory and wrapper/teacher
state survive. One branch follows recorded commands through the original loss;
the other follows the same feedback-aware teacher for64 actions or until the
first lost life/natural terminal. Teacher commands adapt to branch pixels.
ALE life counts are outcome measurements only, not teacher or policy inputs.

The primary endpoint is survival through the original recorded loss action,
using the same number of actions from that origin for both branches. Compare
reward only through that matched endpoint. Separately report survival through
64 actions, first-loss action and reward through64/first loss. The64-action
follow-up starts at each origin and has different absolute endpoints; do not
silently use it as a common endpoint for the early/late comparison. Report both
timings for every loss, all per-lane counts and paired early-only/late-only/both/
neither outcomes. Teacher failure does not establish optimal irrecoverability.

Restore branch origins with one CPU source replay of all37,016 actions/148,038
frames. Every saved pixel hash, action, teacher label, reward, life count,
episode/task outcome, partial tail and frame/reset count must reproduce. The
resulting complete phase rows and aggregate metrics must match the previous
analysis exactly. Validate recorded branches against saved pixels, actions,
labels, rewards, lives, frames and boundaries through the loss. Parent ALE
serialization and teacher state must remain unchanged after each branch.
Reverse branch order at the first origin in each of8 lanes and require identical
results. Preserve complete origin trajectories and indexed outcomes.

The unchanged qualified `probe_breakout_interventions.py` supplies process-copy
branching and continuation; the previous phase replay supplies source validation.
No native model, policy inference, training or GPU work is allowed. Run on one
low-priority CPU core,180s analysis cap/195s controller timeout. Preserve failures
without changing caps or repeating the intervention automatically. Verify all
inputs and active QbertPID736977/invocation48d20e8959f04654b6f5ff8932328d8d/
276 pins before and after; never restart or disturb it. CPU co-host times remain
recorded without correcting Qbert training time.

Preflight2.507s passes291 input pins,265 previous-analysis/25 qualified-branch
pins,240-origin selection and all276 Qbert pins. Eleven pure tests cover lifetime
and episode boundaries, duplicate timing origins, no future rows, matched versus
64-action endpoints, reward horizons and the existing continuation semantics.
Native and emulator imports are forbidden in these fixtures/preflight.

Root`/x/Temp/kindle-breakout-recoverability-20261101`. No new learner, loss,
policy replacement or successor GPU allocation is selected.

## Completed result

The36.589s CPU analysis completes04:14:23 UTC November1/source10846aa. Early
teacher takeover avoids108/120 recorded losses at the original action endpoint;
late takeover avoids26/120. Paired outcomes are23 both,85 early-only,three
late-only and nine neither. Separate64-action survival is97/120 and22/120.
No exact causal action error or optimal irrecoverability is established.

All240 origins,37,016 source actions/148,038 frames,full source/recorded branch
replays,eight reversed-order checks and293 analysis/276 active Qbert pins pass.
Original Qbert process identity remains unchanged; no failure,native policy
inference,GPU context or learner update occurs. Co-host timing caveats remain.
See the [completed result](../results/2026-11-01-cdp-breakout-recoverability.md).
