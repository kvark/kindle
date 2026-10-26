# Four-action compact Breakout candidate

**Selected after completed Pong confirmation.** Coefficient.01 scores14.9583
and14.0 with24/24 wins each; both still miss mean15. The confirmation and both
comparisons pass all audits and its service exits at08:27:53 UTC. This allocation
starts a fresh bounded Breakout life; it does not extend the failed H5 life.

The full18 compact H15/200k mixture control scores7.083; H5 reaches its45-minute
cap at487,736 actions and scores8.458. Neither completes two walls. Test the
existing `published-minimal` Breakout protocol with NOOP, FIRE, RIGHT and LEFT
at the original H15/200k budget. Fewer choices change the policy and action-effects
learning problem, but no learning or runtime benefit is established. Qbert's
six-action trial worsened, so this is a game-specific hypothesis.

A bounded CPU check runs eight paired full/minimal environments for4,096 actions
and16,353 emulator frames per protocol, with15 natural episodes each. Shared
actions produce identical native observations, rewards, boundaries, RAM, lives
and frame counters. It takes4.979s, constructs no Kindle agent and performs no
GPU work. This is bounded protocol evidence, not equivalence for every state.

The H15 control and H5 cutoff policy choose actions outside the four-action
vocabulary for66.71% and82.07% of selected frozen actions. Those are vocabulary
frequencies, not ineffective-action rates. Excluded combinations can affect
control. Changing18→4 also changes network dimensions, initialization and
policy probabilities, including distribution of the native uniform mixture.
Retain an own actual-initial control; same seed does not imply matched weights.

Fresh1009/eval4000001009, centered CDP/action-effects1,4M/RGB64/Auto,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k and actor_unimix.01.
Relative to the full18 H15 mixture control, only action_count changes in the
native configuration. Record actual parameter count at startup. Keep sticky.25,
repeat4, zero no-op starts and100000-frame episode limit. Native max-pooled
frames receive one GPU resize. No external aid, RAM input or reward shaping.
The native policy mixture remains active in frozen sampling.

Fixed200k aggregate actions,49,939 expected updates, first update248 and1800s
training cap including construction. Audit100k recovery before fixed200k final;
no best-checkpoint selection, automatic resume, cap increase or extension.
Evaluate final and own actual initial for the first three natural sampled
episodes per lane,24 per arm, zero updates/truncations and600k-action/300s caps.
Retain excess episodes and partial tails. Preserve the864-point two-wall gate,
at least20 episodes and90% success; label the minimal-protocol result separately.

Audit every numeric report, full config/source/counters, finite checkpoints,
all346 unchanged frozen tensors, full CPU replays/videos and host guards.
Audit cap1200s, service cap3660s. Native`bfb8c31`, runner`1ed07af` and explicit
four-action replay`e853818` remain unchanged. Stop on real faults/numerical failure.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-minimal-unimix001-200k`.
Controller: `run_breakout_minimal_unimix_200k.py`; intended service:
`kindle-cheap-breakout-minimal-unimix-200k-20261026`. Preparation verifies the
complete queue differs from the control only in protocol and output paths;
the unselected controller refuses before GPU/log artifacts. Selection must pin
the preceding review and verify its service has exited. No native, CLI or replay
code changes are needed.

CPU action review: `/x/Temp/kindle-cheap-20261025/breakout-action-vocabulary-review.json`,
SHA256`305a33014cc8a2ccadb58aec3534b8bb496e7353a2d741899b22d3538dfe9ffa`.
One reused development seed; no held-out or broad reliability claim.
