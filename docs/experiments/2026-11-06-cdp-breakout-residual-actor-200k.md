# Fresh 200k Breakout residual actor comparison

Select one fresh learner after the [residual actor bounded check](../results/2026-11-06-cdp-residual-actor.md)
passes native correctness, restore, memory and the30-minute cost screen. Two
retained late batches show imitation and saved optimizer history dominating
the actor update. A separate zero-initialized correction gives imagined policy
and entropy gradients their own parameters and moments. This tests an update
routing hypothesis; useful imagined action values remain unestablished.

Use package source7e0bb827, native SHA256
`3003151fd8134e636f156d82221c4455dcdbc2176d7d138e892bd5dee5bb45e9`,
and the exact bounded runner/runtime/teacher. Set `actor_residual=true`, adding
46,098 trainable parameters, total16,380,451. Policy/entropy gradients reach
only the affine correction; imitation trains the base MLP. Acting and imagination
sum both logits. World expert supervision still uses only the frozen base.
Existing LaProp owns both parameter sets with separate moments; shared replay
and representation remain. No direct dynamics gradient is enabled.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16, encoder/base4e-5, dynamics4e-4,
behavior8e-5, critic1/1, expert100/unit weights/world gradients/all-state
corrective labels,20% episode guidance/cap1920, sampled collection/posteriors,
latent/actor mixing.01 and replay100k. Keep preset256 base and coordinates off,
full18 actions, sticky.25, repeat4, zero reset no-ops, native max-pooled RGB
and one GPU64 resize. CNN/RSSM/critic architecture and other loss settings stay
fixed; their learned weights and experience can change. The bootstrap field
explicitly records its existing imagined default. Native/runner/module hashes
differ from retained447fe50; do not claim historical runtime effects are isolated.

Start fresh with seed1009 and evaluation4000001009, transferring neither
weights nor experience. These are reused development choices. Train200,000
aggregate actions/12,477 updates within1,800 seconds. Retain100k/6,227 recovery,
363 finite tensors and read-only recovery files. All363 initial tensors and47
pre-update semantic events must match the bounded candidate. Require all357
legacy initial tensors and47 initial events to match retained, plus exactly
six zero residual tensors (parameters and moments). Audit configuration, finite
reports, weighted losses, base/residual/critic changes, actor sharing and timing.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Disable guidance, labels and
learning; require363 invariant tensors, full action/outcome/frame/boundary
replays and videos. Keep600k-action,100k-frame-episode and300-second wall caps
per arm. Do not increase caps, repeat stages or substitute episodes. Report
own-initial scores and initial trace differences from retained. Frozen logs
lack live image hashes, so replay does not prove unrecorded image equality.

Greedy is primary. Favor the candidate only if raw greedy return and its own
initial-subtracted return per training minute improve over143.667/6.590132,
and the10,000-resample paired eight-lane gain/min95% interval has a positive
lower bound (bootstrap seed0). Report sampled secondary scores and paired raw
differences. The original task gate stays mean864 and90% two-wall success.
Reused evaluation lanes do not establish training-seed reliability.

Require full training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting,100k recovery observation and terminal
collection. CPU observers remain in actual training wall time. Original CPU
fixture failures and native RGB-constructor failure remain failed; separate
test-only repairs qualify missing evidence. Production native is unchanged.
No completed bounded stage, qualification or old collector repeats. A separate
CPU-only repair may complete missing evidence without rerunning GPU stages or
relabeling a protocol-failed comparison as passed.

Root `/x/Temp/kindle-residual-actor-20261106/full`; service
`kindle-breakout-residual-actor-200k-20261106`. One GPU allocation at a time;
check live identity and stop on real faults or nonfinite values. No automatic
reset, reload, extension, second learner, sweep or successor. Observer timeout
alone does not mean the original process stopped. The five-game goal remains
unfinished.
