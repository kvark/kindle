# Greedy Qbert training clears the primary gate on one learner

The [fresh 200k comparison](../experiments/2026-11-01-cdp-qbert-greedy-training-200k.md)
completes 200,000 actions / 799,803 emulator frames / 12,477 updates in
1652.576s (27m33s). Frozen greedy evaluation scores **15,515.625**, with all
24 selected episodes clearing the first pyramid. This passes the unchanged
15,000 / 90% gate on this learner. Sampled evaluation improves to 11,989.583
with 24 first pyramids but remains below the score gate.

| Fixed assessment | Sampled training control | Greedy training candidate |
|---|---:|---:|
| Frozen greedy return | 14,028.125 | 15,515.625 |
| Frozen sampled return | 11,522.917 | 11,989.583 |
| First pyramids, greedy / sampled | 24 / 24 | 24 / 24 |
| Actual-initial greedy / sampled return | 80.208 / 177.083 | 80.208 / 177.083 |
| Actual-initial first pyramids | 0 / 0 | 0 / 0 |
| Training time | 27m41s | 27m33s |
| Initial-subtracted greedy gain per training minute | 503.831 | 560.413 |
| Actual training overrides | 76,800 (38.4%) | 60,226 (30.113%) |
| Last-50k replay expert cross-entropy | .089268 | .082934 |

Only live action collection changes sampled to greedy. Keep centered CDP,
12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100, native447fe50, the teacher,
losses, sampled posteriors and imagined actor sampling. Greedy return improves
10.60% and sampled return 4.05%, at .51% less observed training time. CPU co-hosting
remains a timing caveat; the small time difference is not an established speedup.
Lower replay loss and aided online returns are not independent competence tests.

This is a useful passing development point on reused learner seed4013 and
evaluation4000004013. Collection changes exploration and visited states together;
the result does not isolate later-state coverage as its cause or establish seed
reliability. No control or bounded-check experience enters the fresh learner.
Retain this recipe for a separately declared fresh-seed check. The five-game
goal remains unfinished, including Breakout.

All 317 original input hashes, the 100k / 6227 recovery, 357 matching initial
tensors, 357 finite invariant tensors per frozen arm, full training
pixel/action/teacher-label replay, frozen outcome replays and videos, host guards,
curves, coverage, exposure and comparison pass. The training replay verifies
138 episodes and all 60,226 overrides. Initial sampled/greedy traces match the
control in 1139 / 1101 semantic events. The 47 pre-update training events match
the greedy bounded check and differ from the sampled control as expected.
Four frozen arms total 135,864 actions / 543,302 frames / 82.126s, with zero
updates or truncations. Each score uses the first three natural episodes in
each of eight lanes; excess episodes and partial tails remain retained.

The five original services all succeed. Main finishes 06:56:22 UTC November 1,
postprocessing 06:56:26, collection 06:56:28, sourcec659423. Original main
PID755504 / invocation `5fb0d30ffb5d47aca552009e586bb929`. The independent .821s
CPU terminal confirmation passes 06:57:01, verifying original identities,
completion hashes and all eight control/candidate videos. No original pipeline
stage fails or repeats. The earlier ancillary terminal-verifier CPU preflight
failure remains preserved: successful short observers need not emit systemd
accounting rows. Its qualified correction retains invocation and success checks;
it does not repeat training, frozen assessment or collection.

Root `/x/Temp/kindle-qbert-greedy-training-20261101/full`, trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-200k-seed4013`.
All services are terminal and no allocation remains active at this closure.
Do not rerun this learner, original collectors or terminal confirmation.

| Full-run artifact | SHA256 |
|---|---|
| Declaration | `5b41d78540b8084e1ac3ae84f7009ce78bd156ec8101984ca32aa9a5c06da9a6` |
| Learning review | `1b9b4fe0071a8e413f88a98c5f4a6c7319d196e6bef051866e02e3437c5b7683` |
| Comparison | `2b4a950f4b26a9b30e40cbe1b46106ed60f335c2c45df75081c9d5330c8835d7` |
| Original completion | `901da69e64cca301982611329b5d04bd548536ecdce77ac726212e4342825ab4` |
| Final summary | `041d75e3ac9043bb3b4e159c267200bd6c260f91b2cc6da541b62c5a8029d940` |
| Terminal confirmation | `761056e4a97f352de8e50a585a69f2fac49e03fcfe8d5938f8c42694c12d784f` |

## Greedy collection reaches more later autonomous training states

A post-hoc CPU recount of the completed4013 learners confirms that later
autonomous experience increases under greedy collection. It reuses the earlier
sampled-control counts and the same input-state first-pyramid, episode-guidance
and age definitions. Both learners already have teacher labels on all200k
actions; these are experience counts, not sampled replay counts.

| Training exposure | Sampled collection | Greedy collection |
|---|---:|---:|
| After first pyramid, wholly unguided episodes | 58,487 | 81,530 |
| Unguided episode actions at age1920+ | 159 | 5,525 |
| After guidance releases, all actions | 12,782 | 13,428 |
| Last50k: after first pyramid, wholly unguided episodes | 20,829 | 31,625 |
| Last50k: unguided episode actions at age1920+ | 159 | 4,541 |
| Last50k: after guidance releases | 4,791 | 3,954 |

The increase is concentrated in wholly unguided episodes, rather than only
longer continuations after teacher guidance. This is consistent with the
collection/coverage hypothesis that motivated the experiment. It does not prove
that the changed exposure caused the frozen score gain: exploration, action
sampling, visited states and subsequent learning all change together. Episode
age is not a verified game-level identifier, and this is one reused learner seed.

Actual last50k training-choice agreement on later movement labels rises
63.84%→73.63% in unguided episodes and36.98%→47.06% after guidance releases.
These are changing training policies on different states, with deliberately
different action selection, not frozen-policy accuracy. Forced actions remain
separate from those agreement denominators.

The1.017s analysis completes07:21:44 UTC November1, source234ca3a. Candidate
counts reconcile with the retained independent phase counter across all200k
actions,799,803 frames,episode boundaries,partial tails,label totals and override
counts. Original control counts are reused without repeating that recount.
All332 diagnostic and359 active8039 inputs remain unchanged;original
PID760870/invocationa0b219d7702b4c82a65e9b33498c3544 stays active. No native model,
emulator, GPU work, learner update, failure or retry occurs. Preserve analysis
and verification co-hosting without correcting training time.

Root `/x/Temp/kindle-qbert-greedy-coverage-20261101`;complete,do not rerun.
No new learner or objective change follows this descriptive recount.

| Coverage artifact | SHA256 |
|---|---|
| Declaration | `a86e0291e7935fa093d0854099880631678c87ebbc03dfd04e4d41fc34290154` |
| Review | `8bd7ae0e22f01181db7fa087306b46e1bb74a669e9b9bf31fb63a850c1ccb78e` |
| Completion | `c4e60b88ace3f45fcd1e2482f692c3cce18e02d5fa37cda949895fbfb609a442` |

## Earlier execution and cost qualification

Fresh greedy action collection completes8,192 actions /32,761 emulator frames /
489 updates in66.241s. The1024-action frozen restore takes.798s,with4096 frames,
zero updates and no completed natural episodes. The conservative200k projection
is29m39s,below the35-minute threshold. This establishes execution and early cost,
not gameplay benefit or a selected full learner.

The [bounded protocol](../experiments/2026-11-01-cdp-qbert-greedy-training.md)
changes only training action selection from sampled to greedy. Keep the retained
12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 recipe,20% episode guidance/
cap1920,all-state corrective labels,world expert gradients and native447fe50.
Imagined actor training and sampled categorical posteriors remain unchanged.
The collection rule changes visited states and exploration together; useful
later-state coverage and frozen gameplay remain unmeasured at this budget.

The new `--training-action-mode {sample,greedy}` is explicit and fresh-training
only. Omitting it preserves sampled collection and legacy headers. Frozen
evaluation keeps `--greedy`. Current source passes637 targeted CPU tests;the
isolated retained runner with identical additions passes34. Tests cover actual
plain/exploration/corrective dispatch,override enforcement,feedback,resets,
replay accounting and invalid/frozen modes. Two earlier CPU test-environment
failures are preserved: missing import paths,then newer tests using old packaged
audit code. Correct source paths resolve both without a production-source fix.

All305 declared inputs,357 control-matching initial tensors,357 invariant frozen
tensors,exact counters,finite reports and CNN/reward/actor/value changes pass.
The first update is376,residual debt.5. All489 reports are supervised;117,304
labeled replay rows equal the weight sum. Maximum raw world/actor expert-loss
difference is8e-7. All47 pre-update semantic events differ from the sampled
control,as expected from the deliberately changed collection rule;initial
parameters still match. No paired initial trajectory equivalence is claimed.

Actual overrides are4096/8192 (50%),with4096 passive labels and3543 recorded
action/teacher disagreements. Full training input-pixel/action/teacher-label
replay and both host guards pass. These guided online interactions are not
unaided competence. No checkpoint or experience from this check is reusable
by a later fresh comparison.

Service`kindle-qbert-greedy-training-check-20261101`,sourcec806de5,original
PID754805/invocation`8564577b63e84df8a7d413b2f89d4e5e`,succeeds06:05:29 UTC
November1 after103.759s. Original collection succeeds06:06:03. No original
stage fails or repeats. A supplemental identity assertion incorrectly requires
the silent parent controller to emit a journal row;its children emit the result.
That failure is retained. The.638s CPU correction verifies the unique original
invocation,start/end,child reports,terminal success and all305 pins at06:06:50,
without repeating the collector or GPU stages.

Root `/x/Temp/kindle-qbert-greedy-training-20261101/native-check`,trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-check`.
No GPU allocation remains active at closure. No full learner,extension,retry,
cap increase or automatic promotion is selected. A separately selected fresh200k
comparison would assess the retained14,028.125 primary greedy /11,522.917 sampled
scores and unchanged15k/90% gate. The five-game goal remains unfinished.

| Artifact | SHA256 |
|---|---|
| Declaration |`a8b305dff6b08821d39c3099534ef22497fa38d52b50f4a119fe18fe52da813e`|
| Review |`471ca4cbbe969a2a84fc5d064773de5780a500271661b9ec9f5f3bfbd75fd090`|
| Guidance replay |`8d2c231b51330f93e1fc2f81663ec3efae8565adcdedc863e8be503cd0a9cf5a`|
| Original completion |`4425516f531e18594b2bccc59534a1d43613a44765b74b79ab2cea7c0f7fc55a`|
| Supplemental identity |`3e404eec2ae2fdbe64cbbb2dc318f21121df1cc83da81547eadcd66e1477c5f4`|
