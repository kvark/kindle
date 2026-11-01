# Greedy Qbert training passes execution and early cost checks

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
