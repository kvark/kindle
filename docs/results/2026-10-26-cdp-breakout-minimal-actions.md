# Four-action compact Breakout: modest gain, task gate still fails

The four-action policy scores **9.542 versus its own initial1.167**, with0/24
two-wall successes. The full18 H15 mixture control scores7.083 versus1.375.
Training takes **20m59s versus22m29s**,6.64% less wall time at the same200k-action
and49,939-update budget. This is a modest development gain, not Breakout task
success or seed reliability. Retain the result; no extension or confirmation is
selected. The next allocation is the prepared Qbert experience-budget test.

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
All task outcomes above remain unchanged. The ratio motivates a
[prepared coefficient.01 test](../experiments/2026-10-26-cdp-breakout-smaller-bonus.md),
which is unselected while PongH5 is active.

Analysis:`/x/Temp/kindle-cheap-20261025/breakout-intrinsic-scale-review.json`,
SHA256`150c8eb7f2ac0b2e16fb97ee9b76e9547d7f295819c73ab40a25e571154837d6`;
script`review_breakout_intrinsic_scale.py`, SHA256
`5f5d17e3abca62472eac01aac9413263d888f2e47f5ef5edc5b47617b58b4364`.
No agent construction, GPU work, new policy decisions or learner updates.
