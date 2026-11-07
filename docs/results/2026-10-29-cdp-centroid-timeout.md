# Breakout timeout includes a long failure to serve

The subsequent [supervision audit](2026-10-29-cdp-breakout-fire-weight.md) finds
FIRE labels present and a much shorter maximum45-action passive FIRE-label run
during sampled training. A separate bounded FIRE-weight check is selected; the
saved timeout and its failed evaluation status remain unchanged.

The saved centroid100 greedy timeout contains a concrete policy failure:
after losing a life at zero-based episode step4,198, the policy takes **12,828
actions without FIRE**, despite two lives remaining. The playfield stays
pixel-identical for the first **12,250 actions**. At two preselected points in
that interval, a short FIRE intervention launches a moving ball immediately.
The recorded actions leave the playfield unchanged.

This establishes a serving failure distinct from the earlier ball-position and
near-paddle readout weaknesses. It does not explain every part of the timeout:
the colors later change, the policy eventually issues828 FIRE actions, and a
late eight-FIRE branch has no visible effect. The ROM-level reason for that
later state is unmeasured. The full learner remains `closed_protocol_failed`.

## Exact replay and local interventions

Replay covers the complete saved lane0: **26,804 actions /107,212 raw frames**,
including its four recorded episodes and every reset. Episode1 is the timeout:
25,000 actions, return27, final20,819 actions reward-free. The other lane0
episodes are natural endings with695/400/709 actions and returns19/8/22.
Every reward, boundary, task outcome, frame count and reset matches the retained
full replay. SHA256 of all raw frames matches the retained video receipt exactly.
No new policy evaluation, native inference or training occurs.

At the three origins declared before replay, CPU process copies retain the full
emulator, wrapper, RNG and sticky-action state. Each branch executes eight
actions /32 frames. All24 recorded-branch rows match the parent replay exactly,
including pixels and teacher annotations; the parent retains its original full
raw-frame hash. The alternative branches execute FIRE eight times.

| Episode step | Recorded continuation | FIRE continuation |
| --- | --- | --- |
| 5,000 | LEFT, RIGHT, LEFT, RIGHT, then four NOOPs; unchanged playfield | A moving ball appears after the first action |
| 10,000 | Eight NOOPs; unchanged full frame | A moving ball appears after the first action |
| 20,000 | Six NOOPs, LEFT, RIGHT; unchanged full frame | Full frame also stays unchanged |

All branches score zero during their short32-frame windows and retain two
lives. Their purpose is to test immediate state transitions, not game returns.
Native before/after images, every branch row and a nine-panel figure are retained
under `/x/Temp/kindle-centroid-timeout-20261029/timeout-branches.png` (also SVG).

## Timeline of the stalled episode

| Event | Zero-based episode step |
| --- | ---: |
| Last reward | 4,180 |
| Life decreases3→2 | 4,198 |
| Constant playfield, with no FIRE | 4,199–16,448 inclusive |
| Playfield colors first change | 16,449 |
| First subsequent FIRE | 17,027 |
| Time limit | After step24,999 |

The playfield hash covers native rows32:185 and columns8:152, excluding the
score header and paddle. Paddle motion accounts for many full-frame changes
while the playfield stays fixed. Lives stay at2 after the final loss. The actual
policy selects NOOP14,472 times during the20,819-action reward-free tail;
RIGHT2,693, LEFT2,826 and FIRE828 account for the rest.

The existing pixel teacher requests FIRE on20,806 of those20,819 inputs.
This is a diagnostic label, not ground truth: its ball crop covers only
rows90:185, and its fixed red detector stops recognizing the paddle for6,759
late inputs after the colors change. The serving conclusion at steps5,000 and
10,000 therefore rests on observed FIRE interventions and exact pixel replay,
not just an absent teacher detection. At20,000, both branches preserve the
same full image; no claim about a longer intervention or reset follows.

## Implication for the architecture and next work

Better ball coordinates cannot by themselves solve a policy that fails to
restart play. The earlier actor corpus showed weaker FIRE agreement, but did
not contain this timeout; this replay supplies direct evidence on the actual
failed trajectory. It supports investigating serving-state coverage and the
actor's imitation targets before another undirected capacity or training-time
increase. It does not establish that the centroid loss caused the problem, that
a class weight will fix it, or that timely serving yields strong ball control.

No new learner, loss weight, evaluation override, cap increase or automatic
successor is selected. The original timeout remains in the evidence and does
not become an accepted gameplay comparison. The five-game goal is unfinished.

## Validation and evidence

The CPU replay completes **22:10:19 UTC October29** in **19.309s**; collection
completes **22:12:16**. Source checkout `45f23ea`, with the exact diagnostic
script pinned separately. All22 declared inputs remain unchanged, all original
lane pixels/outcomes and the24 recorded branch rows match, and all six branches
complete. No failed attempt or repeated diagnostic occurs. Native Kindle imports
are blocked with zero attempts; GPU device descriptors are absent. Total extra
branch work is48 CPU actions /192 frames, with zero learner updates.

[Fixed protocol](../experiments/2026-10-29-cdp-centroid-timeout.md).
Evidence root: `/x/Temp/kindle-centroid-timeout-20261029`, containing the complete
lane annotations, source declaration, branch rows/images, plot, exact replay
review, timeline and completion receipt. Large artifacts stay outside Git.

- Declaration: `ba28303fa09524d48c1db418c039587de2837e23a48352e38845b56220f9db65`.
- Replay review: `a8d8e6d35302a04e354a6de507c180ace0196a9d6fce2044b3cb14f1ef623a68`.
- Timeline: `e7181cddfc180cd08ba88321a46bf389ca5a94edc9eee60af5bd5d39761ae511`.
- Completion: `19227a6021e9195821b202fbefe782ea8da50fb35101b6d55a094119942cf697`.
