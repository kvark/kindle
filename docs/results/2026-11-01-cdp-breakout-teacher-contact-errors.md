# Breakout teacher contact fits expose larger paddle than landing errors

In65 usable fits from120 recorded teacher life losses, mean absolute landing
target error is **1.852 pixels**, versus **4.943 pixels** for predicted paddle
position. In38 of those65, the model predicts an offset inside half the detected
paddle width while the reconstructed offset is outside. This supports inspecting
paddle-response prediction and uncertainty before increasing imitation pressure.
It does not identify a causal failure mechanism or select a replacement teacher.

| Error at the fixed y187 crossing |65 fitted losses: mean / median |674 filtered successful contacts: mean / median |
|---|---:|---:|
| Landing target, pixels |1.852 /0.509|0.370 /0.325|
| Predicted paddle position, pixels |4.943 /4.378|2.317 /1.913|
| Predicted collision time, raw frames |0.461 /0.401|0.562 /0.401|
| Reconstructed ball–paddle offset, pixels |17.094 /14.233|3.110 /2.774|
| Model-predicted offset, pixels |10.796 /5.326|1.795 /1.311|

The source remains the unchanged pixel teacher's first24 natural episodes:
225.083 mean and0/24 two-wall successes. No new trajectory, native policy,
training aid, loss or learner is evaluated. Its known imperfections limit the
meaning of stronger imitation fit; this diagnostic does not explain the native
policy's remaining gap to the teacher.

## Coverage and limits

All120 selected life losses are retained. The fixed filters reject55:
14 short vertical tracks,38 unavailable/cropped/unstable paddle interpolations,
and3 projected crossings outside the playfield. The prior successful-contact
subset contains776 contacts in these24 episodes;102 fail paddle interpolation,
leaving674. These differently selected subsets do not yield unbiased failure
probabilities. No thresholds are relaxed and no rejected case is silently removed.

Among the65 fitted losses,40 model offsets are inside half the detected width;
63 reconstructed offsets are outside, with38 in both groups. Two losses remain
inside this geometric test. Ball radius, true collision height, detection errors,
sticky actions and pooled-frame timing prevent treating the test as exact game
collision physics. Errors near the final plan also do not show when a loss first
became unavoidable or which earlier action should change.

Detected contact widths are16 pixels in53 fitted losses and12 in12; successful
fits contain593 widths of16 and81 of12. Narrower detected widths alone do not
explain all failures. Width measurements at the later life decrement differ
(114 widths of16,six of12), illustrating why those auxiliary counts should not
stand in for contact-time geometry. Rare merged components and crop exclusions
remain in the frame-level evidence.

The one-step model compared with the next **pooled** paddle observation has
MAE1.953 pixels over37,448 pairs and1.894 over8492 near-descending pairs; their
95th-percentile absolute errors are7.878/7.545. Contact-time prediction errors
also involve interpolation and timing. Larger error tails are a concrete topic
for a bounded response/robustness check, not proof that refitting a mean model
will improve control. No policy change or automatic successor is selected.

## Evidence and execution

One new annotation replay covers all52,072 recorded actions/208,254 raw frames,
including25 completed episodes and partial tails. Recorded policy/debug,
pooled input/output pixels,rewards,frames,boundaries,resets and task outcomes
reproduce exactly, as does the earlier aggregate raw RGB hash. It records139
life losses across the whole trajectory,120 in the fixed24 episodes, and no
bonus-life events. Emulator life counts are diagnostic only and never enter the
unchanged pixel teacher. The earlier raw tracks and contact fits are reused.

Replay takes44.731s and analysis1.420s; the low-priority, single-core controller
completes in48.646s at **10:25:39 UTC November1**, sourcef658f2a. All34 diagnostic
pins,11 historical inputs and442 active Qbert inputs remain unchanged. Original
QbertPID781585/invocation539c22282c4e42c7a22bc150961ef963 is preserved.
Sixteen CPU tests and2.429s preflight pass. A separate.005s JSON-only recount
verifies all24 episode life balances,ten scalar means and the38-case count.
No failure,retry,original-stage repeat,native model import,agent construction,
GPU work or learner update occurs. Retain all CPU co-host timing without
correcting Qbert's wall time. The original learner continues.

[Fixed protocol](../experiments/2026-11-01-cdp-breakout-teacher-contact-errors.md).
Evidence/code root`/x/Temp/kindle-breakout-teacher-contact-errors-20261101`.
`frame-annotations.jsonl`, `life-events.json` and `contact-events.json` preserve
the new observations, exclusions and fitted comparisons. Do not rerun completed
replay,analysis or qualification. The five-game goal remains unfinished.

| Receipt | SHA256 |
|---|---|
| Declaration |`ca16fafdc467fd86d7cc5e8cb88da601cf47a432f64cccd7eed7ffaa44e94a5a`|
| Annotation replay |`921769a2980cb556f41cfe0499614639b32c9277bb31a20398849bdb906b8864`|
| Analysis |`d9b442742dd790327d6227229c8af94cc286972f95f1c103c3821500aaf2a6f0`|
| Completion |`e64e88b0af786f9e48365a3a20ffd795d77aaff0fdad95ceae3a370de07fa595`|
| Independent recount |`7ac1f4305eeb971ada40d1891f2b321c6733cb7b88126a67fd66e4f64cbfd39c`|
