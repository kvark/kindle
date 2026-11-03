# Current Breakout actor-gradient inputs are captured

The [fixed capture](../experiments/2026-11-03-cdp-breakout-actor-gradients.md)
completes **35,552 actions /142,169 frames in34.280s**, with zero updates and
all357 checkpoint tensors unchanged. It retains32 preceding current-model states
and512 raw RGB frames for two B16/T16 batches. All32 repeated state readbacks
preserve probabilities/counters, and all recorded rewards,boundaries,resets and
frame counts reproduce. No new gameplay score or gradient conclusion follows.

Inputs come from the retained policy1/replay-value1 model's sampled evaluation.
Each lane supplies four windows chosen by fixed20/40/60/80% eligible-index ranks,
without looking at reward,labels or gradients. Native447fe50 and original teacher
remain unchanged. Corrective labels are added offline with executed-action
feedback; scalar posterior draws differ from the original vector draws. Original
live pixel hashes are unavailable. CPU resize arrays are validation references;
the native backward stage must resize raw frames once on GPU.

CPU preparation passes eight selection cases,32 input pins and338 retained
learner pins in3.338s at10:13:23.393 UTC November3. The original source-mode
assertion failure and pre-execution memory-helper signature correction are
preserved. Neither constructs a native agent. The original capture service
launches10:14:12.632,sourceef390a4c,PID1004049/invocation
`7499ea127a3d4038ace5ba7c041b6042`, and writes its passing review10:14:48.099.
The service is terminal success. Host guards pass; no capture stage repeats.

Independent terminal collection passes **10:17:29.526** in.465s CPU, checking
all36 declaration pins,all512 raw-frame hashes and exact CPU resize references,
32 finite contexts with categorical one-hot state,source action/reward alignment,
original invocation and terminal service state. No emulator or GPU work occurs
in that reader. Capture manifestSHA246b519b/reviewSHAce8d4b66. The capture and
its reader are closed; do not repeat them.

The [backward-only stage](../experiments/2026-11-03-cdp-breakout-actor-gradients-backward.md)
is now CPU-qualified against an isolated exact447fe50 source archive. Production
files differ only by a cfg(test) module registration. The first build fails on
four omitted include fixtures and the newer four-argument resize signature;
the exact historical fixtures and three-argument call fix preparation only.
The corrected build,132 CPU tests and strict Clippy pass. Subsequent source
review catches a diagnostic shape error: behavior uses240 replay rows,while
imagination uses3840 rows. Its labels sum240; world-graph target normalization
does not apply. The corrected dimensions are now covered by the graph fixture
and pass the final build,132 CPU tests,strict Clippy and nine finite/error-bound
fixtures. Exact source verification checks75 files; final preflight pins155
inputs in.229s. An exclusive-output-path failure preserves the earlier queue
under its own name before final selection; no native work occurs in that failure.
All earlier sources,receipts and the initially passing but incomplete preflight
remain preserved. One180s guarded backward job is separately selected after this qualification.
No native backward result is available at this milestone.

Root `/x/Temp/kindle-breakout-actor-gradients-20261103`. No GPU allocation is
active at capture closure; check live state. No learner,coefficient change or
automatic successor learner is selected. Breakout and the five-game goal remain unfinished.
