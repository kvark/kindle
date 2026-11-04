# Stronger continuation supervision lowers frozen Breakout scores

The fresh200k comparison finishes in **21m28.116s**, essentially the retained
control's21m27.910s. Increasing existing continuation BCE scale1→10 lowers the
fixed primary **greedy score143.667→55.833** and sampled70.750→35.250. All final
arms remain zero of24 two-wall successes. **Retain continuation scale1** and the
existing recipe; no extension, coefficient sweep, second learner or automatic
successor is selected. The original864/90%-two-wall and five-game goal remain
unfinished.

## Full200k comparison

The [fixed full protocol](../experiments/2026-11-04-cdp-breakout-continuation10-200k.md)
changes only the existing continuation loss weight. Both terminal and nonterminal
rows and their shared-world gradients receive the higher weight. This is not
class balancing or a tenfold optimizer-step claim. Intermediate life losses
remain nonterminal; H3 supervision stays off. The10-line runner addition passes
the bounded checks below; no native build, new parameters, runtime overlay or
experience transfer occurs.

Keep native447fe50/SHA35ad0db, centered CDP12M64/Auto/N8/B16/T16/H15/R16,
replay-value1/expert100, base/encoder4e-5, dynamics4e-4, behavior8e-5, sampled
collection/posteriors and20% episode guidance/cap1920. The teacher, all-state
labels and actual-command feedback remain unchanged. This is one fresh learner
on reused development seed1009 and its evaluation cohort, not independent seed
confirmation or an isolated mechanism claim.

| Measure | Retained scale1 | Candidate scale10 |
|---|---:|---:|
|Training actions /updates|200,000 /12,477|200,000 /12,477|
|Executed emulator frames|799,595|799,592|
|Training time|21m27.910s|21m28.116s|
|Sampled final /actual initial|70.750 /1.417|35.250 /1.417|
|Greedy final /actual initial|143.667 /2.208|55.833 /2.208|
|Sampled /greedy two-wall successes|0/24 /0/24|0/24 /0/24|
|Sampled initial-subtracted return/min|3.230|1.576|
|Greedy initial-subtracted return/min|6.590|2.498|

Greedy score falls61.14% and greedy gain/min falls62.10%. The paired greedy
return difference is−87.833, with lane-bootstrap95% interval
**[−125.171,−49.498]**; gain/min difference−4.092 has interval
**[−5.832,−2.306]**. Sampled return difference−35.500 has interval
[−48.959,−20.958], and gain/min difference−1.654 has interval[−2.281,−.977].
These fixed10,000-resample/seed0 intervals describe eight reused evaluation
lanes, not training-seed reliability. The fixed candidate preference rule fails.

## Replay fitting and exposure

Overall replay continuation summaries are slightly lower, while frozen gameplay
is substantially worse. These are averages on each changing model's own replay,
dominated by nonterminal rows; they do not establish improved terminal-specific
or held-out calibration. A matched frozen arrival diagnostic on the candidate
has not been performed.

| Training window | Retained continuation BCE | Candidate BCE | Retained continuation MAE | Candidate MAE |
|---|---:|---:|---:|---:|
|First50k actions|.045152|.036879|.017294|.009819|
|Final50k actions|.022457|.022325|.001767|.001741|

Late expert-label cross entropy also falls.560514→.542827 without a gameplay
gain. The world and actor share changing representations and experience;
these fitting summaries do not identify why control regresses. Every12,477
report has finite values and a correctly weighted world total, with maximum
absolute sum error.000113436. All inspected CNN, reward, continuation hidden/
output, actor and value parameter entries change.

Actual teacher overrides are89,650/200,000 actions (**44.825%**), versus43.0585%
retained. The candidate has109,605 native actions before the guidance cap and745
after it. Its52 completed episodes with teacher actions average210.596, while201
without teacher actions average15.597; online guided returns are not frozen
competence. Score216–431 exposure is6,846 actions:6,157 teacher and689 native,
versus4,685/1,262 retained. Neither learner reaches432 in training; the candidate
maximum including tails is406. Score bands are not verified wall events.

## Execution and closure

All761 current and338 retained pins pass with one exact historical replay-source
resolution. Full helpers, corrupted-header/comparison cases and all five queued
commands pass CPU qualification before launch. All357 initial tensors match the
bounded and retained controls;47 pre-update semantic events agree. The100k/
6,227 checkpoint and independent1.972s recovery verification pass with read-only
weights/moments. CPU co-host time remains included without correction.

All four frozen arms select the first three natural episodes per lane,24 each,
with zero updates/truncations and357 invariant tensors per arm. Actual-initial
sampled/greedy semantic traces match the retained799/794 events. The four arms
total72,064 actions in46.429s. Full training pixel/action/label replay, frozen
recorded-action/outcome/frame/boundary replays and videos, host guards, curves,
expert metrics, exposure and paired comparison pass. Ordinary frozen logs do not
record live pixel hashes; no unrecorded live-pixel equality is claimed.

Source`6a05eb02`, mainPID1103691/invocation
`e7fdd5217d244d6e8e05ffdb0f081e97` starts03:48:24 UTC November4. Training ends
04:09:59, main succeeds **04:13:52**, and the original terminal collector succeeds
**04:13:59** after3.970s verification. All three original services are terminal
success. A separate3.910s read-only report review completes04:15:30, verifies
all56 completion artifacts and checkpoint hashes, and extracts the replay
summaries above. No experiment-stage failure, repair, cap change or completed
GPU/analysis stage repeat occurs.

Training construction is2.774s separately. Learner-wrapper time is1,143.709s,
acting63.013s and environment62.704s; these timers do not isolate GPU kernels.
The total training time changes by only+.016%, without an established speed
difference. Root `/x/Temp/kindle-breakout-continuation10-20261104/full` is closed.
No learner or GPU allocation remains active; check live state before future
work. Do not restart training, assessments, replays or collectors.

| Full-run artifact | SHA256 |
|---|---|
|Declaration|`8033b72b49b1ed03f983ab11c34dfece606f832f88006a16c86a4d874f672868`|
|Learning review|`71bf88336af2b706fc4aec5c38a7c5d397f6cb5da8284158e327fdbc136512ac`|
|Training replay|`d4cd50f57e1a1b5e07055072795ddd813f0109e69aece1d9e284e0ef9023e73a`|
|Paired comparison|`733ae0aec617609e49e3f7f8a2e03e2973f2baa250cd491cd5624b4f8ec0c562`|
|Main completion|`e858b17a63ecc16c654f1084de30687f01fdf4863ab4b60bd5cc9b5c8d273356`|
|Original terminal collector|`7690b9d49883b1410dddea13a3843ba140cb3be74c19aa34be477d5b5ea45814`|
|Read-only report review|`6664fc9ba4d2188b2997b93de2b5ef1403b14bb307a432e60fb3493210dab08c`|

## Earlier bounded execution and cost check

The fresh continuation10 check completes **8,192 actions /32,724 frames /
489 updates in46.764s**. A conservative200k projection is **20m55.877s**, below
the30-minute limit. Frozen restore completes1,024 actions in.872s with zero
updates and all357 tensors unchanged. This establishes execution and early cost;
gameplay benefit and terminal calibration remain unmeasured.

The [fixed bounded protocol](../experiments/2026-11-04-cdp-breakout-continuation10.md)
raises only existing world continuation BCE scale1→10. Both terminal and
nonterminal targets, including their shared-world gradients, receive the higher
weight. It is not class balancing, and optimizer adaptation/clipping prevents
inferring a tenfold parameter step. Intermediate life losses remain nonterminal.
The earlier H3 supervision regression is retained; H3 stays off.

Native447fe50/SHA35ad0db, the corrective runtime, CDP12M64/Auto/N8/B16/T16/
H15/R16/replay-value1/expert100, rates, teacher,20% episode guidance/cap1920 and
all-state labels remain fixed. The new `--continuation-scale` option is for fresh
training only; omitted defaults and restored configurations remain unchanged.
The qualified private runner differs from cf4ea238 by exactly ten added lines,
matching the canonical change. No native build or runtime overlay occurs.

Both canonical and private runners pass **34 focused CPU checks**, including
invalid F32 inputs, restore/evaluation rejection, omitted/explicit defaults and
exact requested-config changes. Actual training and restore commands stop at
mocked construction boundaries; schema, replay argv, cadence, loss-sum rejection
cases and the original terminal collector also pass CPU preflight.

All **699 current pins and338 retained pins** pass, with one exact historical
replay-source resolution. All357 initial arrays and47 pre-update semantic events
match the retained control. Every489 report is finite and its weighted world
loss includes continuation10; independent recount has maximum absolute sum error
.000113436. CNN, reward, continuation hidden/output, actor and value parameters
change. Cadence starts at376 actions and ends with residual credit.5.

Full training pixel/action/label replay passes, including22 completed episodes
and3,733 actual teacher overrides. Guided returns are not unaided competence.
Both GPU stages and host guards pass without recovery. Historical bounded
control time is47.689684s; the candidate is1.94% lower, but this single historical
timing comparison does not establish a speedup. Learner wrapper time is40.413s;
these timers do not isolate GPU kernels.

Source`d5777b14`, original mainPID1103277/invocation
`1bf8e5c290a64498bffb1419675d2326` starts03:41:11 UTC November4. Main and original
collector succeed **03:42:20.787 /03:42:24.382**. Independent **5.398s CPU closure**
completes **03:43:00.161**, recounts all489 reports and47 semantic events, rechecks
all initial/frozen tensors, original service identities, replays and guards.
All84 final artifact hashes verify. No experiment preparation, GPU, audit or
collector failure, repair, cap change or completed-stage repeat occurs.

Root `/x/Temp/kindle-breakout-continuation10-20261104` is closed. Do not repeat
its training, assessment, replay or collectors, or transfer its experience.
The separate fresh200k comparison above was selected after execution and
projected cost passed, and is now complete with lower frozen scores. Retain
scale1 and70.750 sampled/143.667 greedy; the original864/90%-two-wall and
five-game goal remain unfinished.

| Artifact | SHA256 |
|---|---|
| Bounded declaration |`f45310b5855fabe60f17857e8edac981f762f0ed8e347b401e31490765f1e1a2`|
| Bounded review |`c2215977ddfafd74ec230bc7b659f1472e88560d537082cd562b028213743d9e`|
| Full training replay |`e8e7596db03d375d0f2bbf89b0a41045550008c7d506e8892a75b8eeab0304b8`|
| Main completion |`82cc134ed53b0fe99282615ada73b65f235edea387ca9e8ba1435831fe3dd765`|
| Independent closure |`32d6c88dfc3a919829dd4b741bd23085037a30a686cf69265c60b7dd9abbc758`|
| Qualified private runner |`74af89412b43a7521cc0f7d9b7a8c5f9e715534545f0fecf60110778f1e251cb`|
