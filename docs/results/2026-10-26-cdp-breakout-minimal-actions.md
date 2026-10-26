# Four-action compact Breakout: modest gain, task gate still fails

The four-action policy scores **9.542 versus its own initial1.167**, with0/24
two-wall successes. The full18 H15 mixture control scores7.083 versus1.375.
Training takes **20m59s versus22m29s**,6.64% less wall time at the same200k-action
and49,939-update budget. This is a modest development gain, not Breakout task
success or seed reliability. Retain the result; no extension or confirmation is
selected. The completed smaller-bonus follow-up below does not improve it.

Fresh1009/eval4000001009, centered CDP/action-effects1,4M/RGB64/Auto,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k and native actor mixture.01.
Only native action_count changes18→4: NOOP, FIRE, RIGHT, LEFT under the existing
`published-minimal` protocol. Actual parameter count changes4,131,729→4,120,963.
Network dimensions, initialization and policy probabilities change together;
same seed does not imply matched initial tensors. Each arm retains its own
actual initial, and the mixture stays active in frozen sampling. No external
aid, RAM input or reward shaping. Sticky.25/repeat4/zero no-op starts and native
max-pooled frames with one GPU64 resize remain fixed.

| Action protocol | Actions / emulator frames / updates | Training wall | Frozen final / own initial | Two-wall successes |
|---|---:|---:|---:|---:|
|Full18|200k /799,097 /49,939|22m29s|7.083 /1.375|0/24|
|Minimal4|200k /799,106 /49,939|20m59s|9.542 /1.167|0/24|

Recorded training time1,259.111704s,158.84 aggregate actions/s,512 natural
training episodes, first update248 and zero debt. The online last50 mean7.22
is separate from frozen9.542. Reviewed curves are similar near the endpoint;
the shorter runtime does not establish a large learning improvement. The
original864-point two-wall/90%-success gate remains unchanged.

Frozen final/initial use52,080/4,984 actions,208,251/19,891 emulator frames and
24.874/2.507s. Select the first three natural episodes per lane,24 per arm.
Final collects49 natural episodes, retaining25 excess episodes separately;
initial collects24. Full traces retain partial tails. Zero learner updates and
truncations; all346 tensors remain unchanged per arm. Every49,939 numeric report,
100k recovery checkpoint, full configuration/source/counters, finite checkpoint,
CPU replay/video and all three host guards pass. The service exits successfully
at09:01:55 UTC; total service wall22m07.701s includes evaluations and audit.
No retries, cutoff, resume or change to the30-minute training cap.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-minimal-unimix001-200k`,
including `learning-review.json`, `curve/`, `action-vocabulary-comparison/`,
`checkpoint-review-100k.json` and `terminal-journal.txt`.
Review SHA256`5e67881fe9e881c9ca8ccd8ee6ec4e1e738656c015fd45255418fbfe4a8f4c34`;
curve`1e33849fedd40cab0465baf44b488c327eaa5c018ad5b53bd70268b2f210ad95`;
comparison`3483a69fe84acc9d6f606df729aad17d8175e29916323c009316e19916e6ce9b`.
Source at launch`9c8f382`, native`bfb8c31`, runner`1ed07af`, replay`e853818`.
The full18 control's older CLI predates optional imagination-length selection;
both configurations retainH15. One reused development seed; this protocol
variant is distinct from the full18 reference.

## Native exploration bonus in the completed controls

A1.48-second CPU analysis reads both complete200k logs and checks them against
their reviewed curves. During updates37,440–49,939, the four-action policy has
mean scaled imagined bonus0.009418 and inferred extrinsic reward0.009768, ratio
0.964. The full18 mixture control has0.003387 and0.015031, ratio0.225. None of
these12,500 per-update inferred extrinsic means is nonpositive. Both observed
training streams have only nonnegative rewards.

The native combined imagined reward includes its already-scaled bonus; the
analysis subtracts that bonus to recover net predicted game reward. Predictions
can be negative even when real rewards are not. These are minibatch means on
different learned trajectories, not individual-state distributions, absolute
reward magnitudes, gradient contributions or a causal explanation of weak play.
All task outcomes above remain unchanged. The ratio motivated the
[coefficient.01 test](../experiments/2026-10-26-cdp-breakout-smaller-bonus.md)
reported below.

Analysis:`/x/Temp/kindle-cheap-20261025/breakout-intrinsic-scale-review.json`,
SHA256`150c8eb7f2ac0b2e16fb97ee9b76e9547d7f295819c73ab40a25e571154837d6`;
script`review_breakout_intrinsic_scale.py`, SHA256
`5f5d17e3abca62472eac01aac9413263d888f2e47f5ef5edc5b47617b58b4364`.
No agent construction, GPU work, new policy decisions or learner updates.

## Smaller bonus lowers the frozen score

Changing only native intrinsic coefficient 1→.01 lowers frozen score
**9.542→5.042**, both versus initial 1.167 and both 0/24 two-wall successes.
Keep coefficient 1 for this recipe; no extension or confirmation is selected.
This reused seed does not establish a general exploration rule.

The fresh 1009 learner keeps 4M/RGB64/Auto/N8/B8/T16/H15/R32, minimal four
actions, native actor mixture .01, original rates and no external aid. It
completes **200,000 actions / 798,710 emulator frames / 49,939 updates** in
1,267.339s (21m07s), versus the control's 20m59s: 0.65% more training time.
First update 248, 6,392,192 sampled replay positions, 635 natural training
episodes and zero debt. Online last-50 mean 6.58 is separate from frozen 5.042.
The reviewed curves show an earlier mid-run gain that does not survive the
fixed-final comparison; no earlier checkpoint is substituted.

Frozen final/initial use 11,376/4,984 actions, 45,453/19,891 emulator frames
and 5.603/2.510s. Final collects 32 natural episodes and initial 24; the first
three per lane give 24 per arm, with eight final excess episodes and all partial
tails retained. Both have zero updates/truncations and 346 unchanged tensors.
All initial tensors and semantic initial traces match the coefficient-1 control.
All 49,939 numeric reports, full config/source/counters, finite checkpoints,
100k recovery, CPU replays/videos and three host guards pass.

The original service exits with failure at 13:43:51 UTC because preparation
omitted `replay-manifest.json`. All GPU stages had exited successfully. The CPU
repair verifies and copies the unchanged paired-control ROM/emulator manifest,
then reruns the original audit without changing its code or repeating GPU work.
The service journal and incomplete watcher receipt remain intact. Separate
repair and completion receipts record the successful audit, plots and comparison;
the original service is not relabelled successful.

Late scaled bonus/net predicted game reward falls from .964 to .00469
(.00005858/.012489). The smaller ratio does not yield better play. These signed
minibatch means remain descriptive, without a gradient or causal interpretation.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-minimal-unimix001-intrinsic001-200k`.
Source at launch `66d1aed`, native `bfb8c31`, runner `22bd61b`, replay `e853818`.
Review SHA256 `14a9d375e9767e5f0f01b1c1e3cef4831a77572631b5fed2f2f5894d0b7e5941`;
curve `b6cb638a385f0e7663a49c7992bef7ffdf04b22d9126815aedddcf6cc2afbbb8`;
comparison `0e49b543597c67c2052f0e73c363c58c4503ca0b96de551d1faa05aa18ac7043`;
CPU repair `6c649f5d571cdf79a45a778691ba055f7892b72056f60d9e585a6afc7e11c780`.


## Actor/critic gradients give a modest unconfirmed gain

Enabling native actor/critic gradients raises frozen score **9.542→11.208**,
versus the same actual-initial1.167. Both remain0/24 two-wall successes. This
is one reused development seed, far below the864-point/90%-success gate. Keep
this as an unconfirmed development result; no extension, seed confirmation or
global default change is selected.

The fresh1009 learner keeps4M/RGB64/Auto/N8/B8/T16/H15/R32, minimal4,
coefficient1/actor.01, original rates and no external aid. Only native
`actor_critic_gradient` changes. It completes **200,000 actions /799,187 emulator
frames /49,939 updates in1,241.483s (20m41s)**, versus20m59s for the control.
The1.40% lower wall cost is not an isolated flag effect: the kernel/module also
changed. First update248,6,392,192 sampled positions,476 natural training
episodes, zero debt and online last50 mean9.26. Construction2.047s is separate.
The reviewed curves show a late online gain; no earlier checkpoint is selected.

Frozen final/initial use36,928/4,984 actions,147,657/19,891 frames and
15.522/2.209s. Final collects41 natural episodes and initial24; select the first
three per lane,24 per arm, retaining17 excess final episodes and all partial
tails. Both have zero updates/truncations and all357 tensors unchanged. Eleven
extra world tensors are frozen actor copies; trainable count stays4,120,963.
All346 shared initial tensors and semantic initial evaluation traces match the
control. Full config/counter checks,100k recovery, every49,939 finite report,
CPU replays/videos and three host guards pass. Every initial policy/value loss
report is nonzero; this does not prove raw-gradient correctness or causality.

The main training/evaluation service exits successfully at18:05:14 UTC,
21m27.619s total. A separate CPU comparison postprocessor fails because its
checkpoint-helper import transitively requires `ale_py`, absent from the plot
venv. Preserve that original failure. A separate v2 comparator copies the exact
pure shared-initial comparison function and removes the unused ALE import;
all original pinned scripts, GPU logs, reviews, replays and plots are unchanged.
The corrected comparison passes and its plot is reviewed. No GPU stage repeats.

Artifacts:`/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-minimal-acgrads-200k`.
Launch source`710fd6d`, native`bfb8c31`, runner`fbdaee3`, replay`5043b7a`.
Review SHA256`97c4b183bac0b18ee5665bfaf9720d37e54d10754a75c008e781c15659026b4a`;
curve`8775a94c804f243e728ef5a4aaa0f9bb5899c71f9fb0eb475de9f2026f1e19be`;
v2 comparison`1481c67401988023d460b30519f11711f278ecc08c38f7c5d7e44f4d50c407fb`;
completion after CPU repair`da46a2559aa00efe0ea6a26728862dcedffecde3d297276b366d19e422a653cc`.
Original `postprocess-command-1.json`, repaired command, input-invariance and
terminal-service receipts remain beside the result. The
[experiment declaration](../experiments/2026-10-26-cdp-breakout-actor-critic-gradients.md)
retains preparation, startup and recovery evidence.
