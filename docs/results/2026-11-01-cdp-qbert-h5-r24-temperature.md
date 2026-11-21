# Qbert H5/R24 temperature0.5 lowers frozen performance

Retain greedy evaluation for the10007 H5/R24 learner. Fixed temperature0.5
scores **12,840.625**,10.45% below retained greedy14,339.583 and6.69% below
default sampled13,761.458. All three trained modes clear the first pyramid in
24/24 episodes,but their means miss the unchanged15,000-point gate. The original
learner's primary greedy result and the one-of-two passing H5/R24 result remain
unchanged. No temperature sweep,retry,extension or automatic successor is selected.

| Frozen mode | Trained mean | Actual-initial mean | Trained first pyramids | Gain over own initial / original training minute |
| --- | ---: | ---: | ---: | ---: |
| Retained greedy,primary control | 14,339.583 | 1.042 | 24/24 | 502.032 |
| Default sampled,reproduced | 13,761.458 | 171.875 | 24/24 | 475.809 |
| Fixed temperature0.5 | 12,840.625 | 203.125 | 24/24 | 442.474 |

Both new actual-initial arms clearzero pyramids. Training remains200k actions/
18,715 updates/**28m33.661s**;the assessment adds no learning. Keep12M48/Auto/N8/
B16/T16/H5/R24/encoder4e-5/actor-unimix0/expert100 and sampled posteriors.
Guidance and labels are off. The four new arms use the first three natural
episodes per eight lanes at seed4000010007,full18/sticky.25/repeat4,native pooled
frames and one GPU resize. This is inference tuning on a reused development
learner/cohort,not independent confirmation or an isolated architectural cause.

All558 inputs,357 finite/invariant tensors per arm,full recorded-action/reward/
frame/reset/task replays,videos and host guards pass. The qualified temperature
runtimee416696/SHA40872a3 and runnerabc2c2b differ from training447fe50/aba4300d;
the default trained/initial arms reproduce all **6,531/1,237 semantic events**
and their selected scores exactly. Each default and replay passes before its
temperature candidate. Ordinary frozen logs lack live pixel hashes,so replay
does not prove direct live-image equality.

The four arms total **125,944 actions/503,626 emulator frames/105.190s**,zero
updates or truncations. Every selected episode ends naturally. All fixed action,
time and service caps hold. The completed CPU qualification is reused without
repetition;there is no build,failure,repeated GPU/audit stage or cap change.

Source1308ffb launches13:55:56 UTC November1 under
`kindle-qbert-h5-r24-temperature-20261101`,originalPID801574/invocation
`6ce4cfed31e44e73aef685bb38ed64b7`. Main succeeds **13:59:10**,original collector
**13:59:17**;both services are terminal success. The main spans3m14.141s including
audits/replays. All preceding R8 and CPU-reference work closed before launch.
No GPU allocation remains at completion;check live state before work. The
five-game goal is unfinished.

[Fixed protocol and reused CPU qualification](../experiments/2026-11-01-cdp-qbert-h5-r24-temperature.md).
Evidence `/x/Temp/kindle-qbert-h5-r24-temperature-20261101`;complete,do not rerun.

- DeclarationSHA256`f646b008336c3796711ec765a0f46a01f403c096923df81c9d3b0c7abc11ae8a`.
- Comparison`906aaab15776b4de8ce83effa2bb1413424a0f48f361b19ae1ed2e6dc18883e7`.
- Completion`15f098c525836b71bbbf47ada36188018774c46fcf1ef488231918f8c1ddea6b`.
