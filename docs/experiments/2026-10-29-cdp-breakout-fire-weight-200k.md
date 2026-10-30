# Fresh200k Breakout FIRE-weight4 comparison

Select one fresh learner within the five-game goal. The bounded FIRE-weight4
check passes8192 actions/489 updates in48.059s, all372 initial/frozen tensors,
full guided replay and both host guards. Its conservative200k projection21m31s
passes30min. The eight short frozen natural episodes all score zero, so that
check establishes execution and cost only. The separate timeout replay shows
a12,828-action failure to serve; saved training has FIRE labels throughout but
much shorter passive FIRE-label runs. Test gameplay before selecting this weight.

Use12M64/Auto/N8/B16/T16/H15/R16,200k actions/12,477 updates, seed1009 and
evaluation seed4000001009. Retain centered CDP, native2974160, centroid scale100,
expert scale100/world gradients/corrective labels, original rates, actor-unimix.01,
full18 actions, sticky.25,20% episode overrides capped1920, replay capacity100k
and one GPU resize from native RGB. Set only the exact FIRE imitation weight to4;
all other action weights stay1. No prior weights, replay, belief, RNG, games or
bounded-check experience is reused. Do not change the pixel teacher.

Compare against retained accepted control sampled13.208333/greedy17.875, own
initials1.416667/2.208333. Also retain the previous centroid weight1 learner's
descriptive14.375/20.125 and its failed protocol as context; do not relabel that
comparison as accepted. Relative to that centroid learner, only the class weight
changes in configuration, but trajectories and runner bytes differ. Relative to
the accepted control, native version, centroid head/loss and parameter count also
differ. Weight4 changes total imitation magnitude, not just fixed-total class
balance. Reused seed/cohort provide development evidence, not seed reliability
or an isolated mechanism claim.

The main service has a4200s cap: training1800s, each of four frozen arms300s,
CPU audit1200s. Save and verify100k/6227 recovery and200k/12477 final checkpoints.
Require all372 initial tensors to match the bounded check,357 shared control
tensors and paired actual-initial semantic traces. Preserve finite reports,
weighted target accounting, changed CNN/reward/actor/value parameters and all
frozen tensors. Pin the qualified runtime, current runner, corrected schema,
controllers, evidence and protocol before launch. CPU preflight exercises actual
commands, config rejection, observer readiness/arguments and plotting/exposure
paths without native construction; it must not repeat the prior lazy-export
harness mistake.

Assess final and actual-initial checkpoints in both sampled and greedy modes:
first three natural episodes per each of eight fixed lanes,24 selected episodes
per arm. Use the existing600k aggregate action cap,25k-action episode limit,
zero updates, zero truncations, guidance/labels off and unchanged90% two-wall/
864-point task gate. Keep every recorded episode, including any timeout. Every
GPU stage uses the existing host guard; stop on fault or numerical failure with
no automatic retry, cap increase or host recovery. A failed protocol remains
failed even if selected natural-episode means rise.

Replay complete training pixels/actions/labels and every frozen action/reward/
boundary/reset/frame with retained videos. Finish curves, exposure, comparison,
100k recovery and terminal collection before reporting an accepted result. From
the saved frozen actions, report full and selected-cohort action counts, longest
FIRE-free command intervals and reward-free episode tails; command gaps alone
are not verified waits to serve. Four read-only observers handle progress,
recovery, postprocessing and collection. Their timeouts do not stop or restart
the main learner. Postprocessing/progress/collection have6000s caps with5700s
waits; recovery has3600s with3300s wait. Preserve any CPU-only audit defect and
repair separately without repeating completed GPU stages.

No extension, coefficient sweep, checkpoint selection, confirmation or automatic
successor is selected. Online guided scores are not unaided competence. The
five-game goal remains unfinished regardless of this run's finite losses.

Evidence: `/x/Temp/kindle-breakout-fire-weight-20261029/full`.
Service: `kindle-breakout-fire-weight4-200k-20261029`.

## CPU preparation

Five actual commands parse with native capabilities stubbed and stop before
construction. Twenty config cases, six header-readiness cases, four observer
parser cases and five recovery-body cases pass;17 scripts compile and50 literal
paths resolve. All142 bounded and132 control pins remain unchanged; the actual
bounded initial schema has372 tensors, with357 shared control tensors. No native
extension is imported or agent/environment constructed by this preflight.

Separate analysis checks pass in the plotting environment with native/emulator
imports forbidden. The actual bounded weight4 log reproduces20,312 FIRE replay
rows and its exposure totals. The new frozen command summary passes reset,
partial-tail and FIRE-alias fixtures, rejects seven invalid sources, and checks
all214,432 recorded actions of the retained failed greedy evaluation, preserving
its25k-action timeout and20,819-action reward-free tail. These are saved-log
checks, with zero environment actions or learner updates. Both passes before
and after clarifying count labels are retained; no preflight failure occurs.

## Launched and startup verified

The main service starts **22:43:07 UTC October29**, source `ce86d72`, with196
pinned inputs. PID354340/invocation`ad37c04842fa4b7bb51ec0f9a42ebf7e` runs
on the declared RX7900XT, with progress, postprocessing and terminal collection
observers. The100k recovery observer completes successfully below.
The startup audit verifies all372 initial
tensors against the bounded check,47 matching pre-update semantic events and
the first six updates at376,392,408,424,440,456 actions. Its complete prefix has
7208 actions and427 finite reports, with the declared FIRE weight4 and cadence.
The read-only startup command takes.612s while co-hosting with the learner;
retain this timing caveat without wall-time correction.

At startup no full frozen outcome was available. The original service proceeds
through recovery, all four frozen arms, full evidence checks and terminal
collection without restart; its completed outcome is recorded below.

- Declaration: `3ecbc619b49481cb3cba8809d63777e13b68b35ec270d12b36ca4715237035f9`.
- Startup review: `7c11f2c507022135649448280b29abac8884a4f5cea9476e7fd893602ffb45ff`,
  retained in the trial's `startup-review.json` with the original process identity
  and input hashes.

## Halfway recovery verified

At **22:54:42 UTC October29**, the read-only observer verifies the saved
**100,000-action /6,227-update** checkpoint:295 world,66 behavior and11 slow-value
tensors, all372 finite. It also checks every complete learner report in the log
prefix:6,489 sequential finite reports through104,200 actions. The checkpoint
metadata and file sizes/timestamps stay unchanged during the audit. The recovery
observer then succeeds without restarting or stopping the learner. The original
main PID/invocation remains active at that milestone; frozen competence is then
still unmeasured.

Checkpoint digest: `0fff0ef0afcfa6e159dedd642c73d189aea430dc5f917db5eeb534079cfc3d07`.
Full receipt: the trial's `checkpoint-review-100k.json`. No training setting,
wall cap, evaluation cohort or pinned controller changes.

## Completed outcome

The declared run completes200k actions /799,434 frames /12,477 updates in22m07s.
Sampled12.375 /greedy17.792 do not improve retained13.208 /17.875; own initials
1.417 /2.208 match. All0/24 two-wall successes, with zero timeouts in all four
arms. All196 pins,372 matched initial/invariant frozen tensors,357 shared
control tensors, both initial traces, full replays/videos, guards, curves,
exposure, FIRE-command accounting and collection pass. Main/post/collection
succeed23:08:19 /23:08:40 /23:08:43 UTC October29. No full-run failure or GPU
stage repetition occurs. Retain the control; no extension, coefficient sweep,
confirmation or automatic successor. All services are terminal and no GPU
allocation is active. [Detailed results and limitations](../results/2026-10-29-cdp-breakout-fire-weight.md#full-comparison-completes-without-timeouts-but-does-not-improve-score).
