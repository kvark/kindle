# Current Breakout actor-gradient diagnostic

The policy100 comparison lowers both frozen scores. The older raw-gradient
diagnostic measures the world graph, not the current actor. Select a diagnostic
of imagined score-function, entropy and replay imitation contributions at the
retained policy1/replay-value1 checkpoint before choosing another coefficient.
No learner, gameplay assessment, coefficient sweep or extension is selected.

Checkpoint: `/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/final`.
Keep its exact12M64/Auto/B16/T16/H15/expert100/policy1/critic1/1 configuration,
native447fe50 and original teacher. The accepted gameplay scores remain
70.750 sampled/143.667 greedy, with0/24 two-wall successes in both modes.

First prepare current latent contexts from its completed sampled evaluation,
35,552 recorded actions. For each lane, eligible preceding-state indices are
multiples of8 in the first three natural episodes, with sixteen subsequent
observations in the same episode and no intervening reset or truncation.
Choose ranks20/40/60/80%, rounded down over the eligible list; require distinct,
nonoverlapping windows. First two ranks per lane form one B16/T16 batch; last
two form another. Selection cannot inspect rewards, labels or gradients.

A single frozen scalar pass follows every recorded action and verifies rewards,
boundaries,resets and frame counts. Read current latent state at each selected
preceding index, preserving state/probabilities/counters under repeated readback.
Save512 native RGB frames and their hashes,32 preceding contexts,previous
actions/rewards and original corrective teacher labels added offline. Scalar
posterior draws differ from original vector draws; these are current model
states on retained trajectories, not the original training minibatches. Original
live image hashes are unavailable. Save a frozen checkpoint and require all357
tensors unchanged. No learning entry point or update may occur.

CPU selection and source/checkpoint qualification must precede the capture.
Use the unchanged packaged runtime under the host guard, with a240s capture
cap and at least2GiB sampled memory headroom. The CPU Pillow resize is reference
data only; the later native batch path must resize raw frames once on GPU.
Do not repeat capture, replay or completed analysis if a later stage fails.

The subsequent backward-only stage requires separate CPU graph qualification
and pinned instrumentation before execution. Use the exact native447fe50
production source with test-only instrumentation. Preserve production graph
nodes/outputs, inputs, parameters and shared draws. Compare gated total with
production, and component sums with total, before interpreting raw norms or
cosines. Separate actor and critic parameters; report zeros and both batches.
Disable optimizer and clipping, retain all weights/moments and counters. Raw
derivatives do not establish optimizer movement or causal gameplay effects.
No backward job is selected until its executable and checks are qualified.

Root `/x/Temp/kindle-breakout-actor-gradients-20261103`. Check live state before
GPU work. No overlapping allocation, automatic retry or successor learner.
The five-game goal remains unfinished.

CPU preparation initially rejects the source header because its fixture expects
`evaluate` instead of the actual `evaluate_sample` mode. Preserve the original
source and failed receipt. Source review also corrects the memory helper call
to its existing one-argument/dictionary interface before any native execution.
No native agent or emulator is constructed in that failed preparation.
