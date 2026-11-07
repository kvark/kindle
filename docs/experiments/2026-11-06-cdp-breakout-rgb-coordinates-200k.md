# Fresh200k Breakout comparison with native RGB coordinate channels

Select one fresh learner after the [bounded coordinate check](../results/2026-11-06-cdp-rgb-coordinates.md)
passes execution,derivatives,legacy compatibility and the30-minute cost screen.
This measures whether explicit image position improves frozen learning per
minute. Position readout errors motivate the test but do not identify the
policy's causal limitation. The SuperMimic actor-gradient direction remains a
distinct proposal; no actor-gradient implementation or combined recipe is selected.

Use the bounded source2d651cb0/runtimeSHA b9136467 and runnera0ffbe46, with
`rgb_coordinates=true`. Keep centeredCDP12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/dynamics4e-4/behavior8e-5, critic1/1, expert100/unit weights/world
gradients/all-state labels,20% episode guidance/cap1920, sampled collection/
posteriors,latent/actor mixing.01 and replay100k. Native pooled RGB gets one
GPU64 resize; coordinates add1,600 first-layer parameters with no replay change.
The serialized replay-value bootstrap explicitly records its existing imagined
default. Native,runner and teacher-module hashes differ from retained447fe50;
the selected teacher policy definitions are AST-identical, and full replay is
required. Do not claim isolated effects or baseline initial tensor equality.

Start fresh with seed1009 and evaluation4000001009, without transferring weights
or experience. Both are reused development choices. Train200,000 aggregate
actions/12,477 updates within1,800 seconds. Require100k/6,227 recovery,357 finite
tensors and retained read-only files. Compare all357 actual-initial tensors and
47 pre-update semantic events with the same-architecture bounded run. Validate
all header differences,finite weighted reports and changed learning parameters.

Assess final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Disable guidance,labels and
learning; require357 invariant tensors,full action/outcome/frame/boundary replays
and videos. Keep600k action,100k-frame episode and300-second wall caps per arm.
Do not increase caps,retry,substitute episodes or require retained-initial
trajectory equality after changing architecture. Report each actual initial.

Greedy is primary. Favor the candidate only if raw greedy return and its
own-initial-subtracted return per training minute improve over143.667/6.590132,
and the10,000-resample paired eight-lane gain/min95% interval has a positive
lower bound (bootstrap seed0). Report sampled secondary results and paired raw
differences. The original task gate remains mean864 and90% two-wall success.
Reused evaluation lanes do not establish training-seed reliability.

Require complete training pixel/action/label/override replay,guards,curves,
exposure/score-range accounting and terminal collection. Preserve all bounded
failures,CPU repairs and pause receipts. No completed stage repeats. Retain
historical-control timing and CPU co-host caveats without wall-time correction.

Root `/x/Temp/kindle-rgb-coordinates-20261104/full`; service
`kindle-breakout-rgb-coordinates-200k-20261106`. One GPU allocation at a time;
check live identity and stop actual faults/nonfinite values. No automatic GPU
reset,reload,extension,second learner,sweep or successor. A failing observer does
not imply a stopped learner; inspect its original process/invocation first.
