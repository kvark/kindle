# Fresh Breakout comparison with greedy live collection

Select one fresh200k-action learner after the [bounded check](../results/2026-11-01-cdp-breakout-greedy-training.md)
passes execution,full replay,checkpoint invariance and the conservative30-minute
cost gate. Its20m52s projection leaves room under that cap. This is a separate
decision within the existing five-game goal;no bounded experience is reused.

The retained sampled-collection control scores56.292 sampled/104.667 greedy in
21m28s,withzero two-wall successes in either24-episode assessment. Greedy live
collection is an empirical candidate for improving visited training states and
subsequent control. Qbert4013 improved with this change,but8039 misses its gate;
one of two cheaper Qbert learners passes. Neither Qbert nor Breakout's takeover
diagnostics establishes a coverage mechanism or promises a cross-game gain.

Change only live collection sampled to greedy. Keep centered CDP12M64/Auto/N8/
B16/T16/H15/R16/encoder4e-5/base4e-5/dynamics4e-4/expert100,actor unimix.01,
all18 imitation weights1,world imitation gradients,all-state corrective labels,
20% episode continuation guidance/cap1920,full18 actions,sticky.25,repeat4 and
native64 resize. Posterior and imagined actor sampling remain unchanged.
Use native447fe50 and exact qualified runneraba4300d. The runner adds the explicit
fresh-training mode to the control runner;retain both exact hashes and its diff.
No model,loss,teacher,preprocessing,build or backend change.

Train200,000 aggregate actions/12,477 updates under1800s. First update376,
residual credit.5. Preserve100k/6227 recovery before final checkpoint overwrite.
Use fresh seed1009/evaluation4000001009;these are reused development inputs.
No weights,replay,belief,game state,RNG or check experience is reused.

Compare fixed final and actual-initial checkpoints in sampled and greedy modes,
first three natural episodes per eight lanes (24 per arm). Sampled remains the
primary assessment;retain greedy separately. Guidance and labels are off,
updates and truncations must be zero. Keep the original90% two-wall-success
gate. Each arm has300s and600k actions;no cap increase,checkpoint selection or
retry if it fails its protocol. Preserve excess episodes and partial tails.

Require357 initial tensors matching control and check,357 finite invariant
tensors per frozen arm,finite supervised reports,correct raw losses/weights,
changed CNN/reward/actor/value,full guided pixel/action/label replay,frozen
outcome replays/videos,host guards,curves,exposure and comparison. The initial
training prefix must match the greedy check and record differences from sampled
control;actual-initial frozen semantic traces must match control. Ordinary
frozen logs lack live pixel hashes,so their outcome replay is not live-image
equality. Report actual override exposure and learning per training minute;
guided online scores and replay fit do not substitute for frozen competence.

CPU audit1200s,training replay600s,main service4500s. Original progress,recovery,
postprocessing and terminal-collection observers are required. Preserve CPU
co-host timing without correction. No extension,cap increase,seed confirmation,
coefficient sweep or automatic successor follows this comparison.

The1.357s CPU preflight passes five actual CLI paths,24 configuration cases,
eight training-mode cases,11 startup-header cases,eight recovery-body cases,
four observer-parser cases,45 resolved paths,15 compiled scripts,252 bounded
and227 control pins,and357 schema tensors. Native capability metadata is
stubbed;no native model or emulator is constructed. An initial CPU preflight
fails before construction because a broad service-name replacement also changed
the absolute control path. The exact four source snapshots and failure remain
in `preparation-failure`;only that path is corrected before the passing check.
No training or frozen stage repeats.

Root `/x/Temp/kindle-breakout-greedy-training-20261101/full`,trial
`breakout-12m-64-b16-r16-encoder4e5-greedy-expert100-200k-seed1009`,service
`kindle-breakout-greedy-training-200k-20261101`. Launch once after the selection
commit,clean-worktree,idle-host and input-hash checks. Do not restart or overlap
the original learner. Final gameplay benefit remains unmeasured.

## Original launch and startup

Launch07:55:05 UTC November1,sourcebd51be9,272 pinned inputs. Original main
PID766156/invocation`d07a6c26ba914fe9a6032d8ec765b0dd`;all four original observers
launch successfully. Startup07:55:21 passes357 finite initial tensors matching
control and check,47 matching bounded pre-update transitions/47 expected sampled
control differences,correct first six updates and108 finite supervised reports
through2104 actions. The original process remains active.

Independent CPU terminal preflight completes07:56:34 in.799s. It checks all four
retained control scores/videos,five original control service identities,four
journal fixtures,the Breakout exposure/gate schema,all272 active pins and the
original live identity. It constructs no native model or emulator. Keep this
CPU co-host time without correcting training time. Verifier
`/x/Temp/kindle-breakout-greedy-training-20261101/review_terminal.py` SHA256
`0fba3175a3ab746e4a19cc5968342b8c685a8d532a461556e1dbf129c2f2b243`.

One separate CPU confirmation observer launches07:57:41,service
`kindle-breakout-greedy-training-confirmation-20261101`,PID766527/invocation
`f8fce805f4eb413187067ab39d730232`. It waits for the original100k recovery review,
then preserves exact metadata/tensor hashes with the qualified verifier once.
After all five original services succeed,it runs terminal confirmation once.
It cannot start,stop,resume or modify the learner. Its source/verifier/declaration
hashes and launch identity are retained privately. No manual recovery/final
confirmation while this observer is active;inspect retained failures without
repeating original stages. The original pipeline and its deadlines are unchanged.

At24,576 actions/1513 updates,training elapsed151.225s. No full frozen result
is available. Preserve the original learner;do not overlap GPU work.

## Halfway recovery

The original observer passes100k/6227 recovery08:06:11 UTC November1,with357
finite tensors and6403 finite reports in the complete prefix through102,824
actions. The independent confirmation observer runs its qualified recovery mode
once at08:06:22 in.843s. Exact metadata and all tensor hashes,272 input pins and
originalPID766156/invocationd07a6c26ba914fe9a6032d8ec765b0dd remain unchanged.

Recovery review SHA256
`61885097cbec0a0321c761c232636f697b3c737fd7f5d3af07c18c3a5f0de4e2`;preserved
`recovery-metadata-100k.json` SHA256
`b389d62e83b7431d46c4798eebb96fb14430d7a2c5fa76d56f41a125fa16f15c`.
Do not rerun recovery confirmation;the same CPU observer awaits final closure.

The completed8039 exposure recount and unselected cross-cohort CPU preparation
co-host with training;retain their recorded durations and the first preparation
checker failure without time correction. No new GPU allocation is selected.
The original learner continues;final frozen gameplay remains unmeasured.

The unselected Qbert cross-cohort controller qualification passes08:12:20 UTC
in4.387s (4.424s command),after a preserved6.628s CPU-only import-assertion failure.
Its retained-diagonal audit uses no new GPU stages;all272 active inputs and the
original Breakout identity remain unchanged. Keep this co-host timing without
correction. Breakout reaches159,744 actions/9961 updates in1037.920s;no full
frozen outcome is available and no second GPU job is selected.
