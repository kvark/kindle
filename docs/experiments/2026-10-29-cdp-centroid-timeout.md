# Diagnose the saved centroid Breakout timeout

Within the existing five-game goal, inspect the centroid100 final greedy lane0
timeout using CPU replay only. The saved episode lasts25,000 actions, scores27,
and ends with20,819 reward-free actions. Determine whether this interval contains
moving-ball play or an unserved ball before selecting another training change.

Replay all26,804 recorded lane0 actions, including every reset and the episodes
before and after the timeout. Require exact rewards, terminal flags, truncations,
frame counts, task outcomes and the retained video's SHA256 of all107,212 raw
frames. This is a lane-specific diagnostic, not another full eight-lane replay
or evaluation. The source full learner remains `closed_protocol_failed`.

Annotate each current input with the unchanged feedback pixel teacher, executed
action feedback, ALE lives, full-frame and playfield hashes. Reset the teacher
only on recorded resets. Teacher ball visibility covers only its declared crop;
an unseen ball alone must not be interpreted as waiting to serve. Lives are
post-hoc analysis only and never become a policy input or training label.

At timeout-episode zero-based steps5000,10000,20000, fork CPU process copies of
the complete environment. For each origin, compare the next eight recorded
actions with eight FIRE actions. Preserve wrapper state, RNG and sticky-action
memory by process copying, not ALE snapshot restoration. Require the recorded
branch to reproduce all eight original transitions and require the parent replay
to retain its exact raw-frame hash. Save all branch rows and before/after native
images. These are local counterfactual observations, not new policy scores or
evidence of long-horizon competence.

Pin source logs, retained replay/closure, exact runtime teacher, ROM, ALE and
wrapper before execution; verify pins afterward. Block native Kindle imports,
reject open GPU device descriptors, and cap the entire CPU analysis at120s.
No GPU allocation, new learner, cap change, policy intervention in evaluation,
or automatic successor is selected. Evidence:
`/x/Temp/kindle-centroid-timeout-20261029`.
