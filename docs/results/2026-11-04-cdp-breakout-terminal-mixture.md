# More terminal replay does not improve frozen Breakout

The fresh 200k-action comparison finishes in **21m19.292s**, close to the retained
21m27.910s. Terminal replay exposure rises **9.43×**, but the primary greedy score
falls **143.667→85.250** and sampled score falls **70.750→65.708**. Both remain
zero of 24 two-wall successes. Retain the existing sampler and continuation scale1;
the optional mixture stays off. No extension, mixture sweep, second learner or
automatic successor is selected. The original 864/90%-two-wall and five-game goal
remain unfinished.

## Full 200k comparison

The [fixed full protocol](../experiments/2026-11-04-cdp-breakout-terminal-mixture-200k.md)
sets `terminal_replay_fraction=.25` after fresh-sequence priority. It keeps
12M64/Auto/N8/B16/T16/H15/R16, replay-value1/expert100, continuation1,
base/encoder4e-5, dynamics4e-4, behavior8e-5, sampled collection/posteriors,
full18/sticky.25/repeat4 and 20% episode guidance/cap1920. Initialization is fresh;
no bounded weights, replay or experience are transferred. Development seed1009
and the evaluation cohort are reused.

| Measure | Retained sampler | Terminal mixture .25 |
|---|---:|---:|
|Training actions /updates|200,000 /12,477|200,000 /12,477|
|Executed emulator frames|799,595|799,590|
|Training time|21m27.910s|21m19.292s|
|Sampled final /actual initial|70.750 /1.417|65.708 /1.417|
|Greedy final /actual initial|143.667 /2.208|85.250 /2.208|
|Sampled /greedy two-wall successes|0/24 /0/24|0/24 /0/24|
|Sampled initial-subtracted return/min|3.230|3.015|
|Greedy initial-subtracted return/min|6.590|3.895|

Greedy return falls 40.66% and greedy gain/min falls 40.90%. The paired greedy
return difference is −58.417, with the fixed lane-bootstrap 95% interval
**[−107.462,−15.623]**; gain/min difference −2.695 has interval
**[−4.987,−.700]**. Sampled return difference −5.042 has interval
[−29.377,22.125], and gain/min difference −.215 has interval [−1.356,1.058].
The candidate preference rule fails. These 10,000-resample/seed0 intervals describe
eight reused evaluation lanes, not training-seed reliability.

## Exposure succeeds; gameplay does not

Both learners process 3,194,112 world replay rows. The candidate receives 50,706
terminal rows (1.5875%) versus 5,376 (.1683%) in the [retained audit](2026-11-04-cdp-breakout-terminal-replay.md).
Only 161 of 12,477 candidate updates contain no terminal, versus 8,105 retained.

| Training actions | Retained terminal rows | Candidate terminal rows | Retained updates without terminals | Candidate updates without terminals |
|---|---:|---:|---:|---:|
|1–50,000|1,938|12,917|1,643|72|
|50,001–100,000|1,505|12,915|1,891|18|
|100,001–150,000|1,106|12,552|2,185|32|
|150,001–200,000|827|12,322|2,386|39|
|Whole run|5,376|50,706|8,105|161|

The candidate draws 11,776 fresh, 141,120 uniform and 46,736 terminal-route
sequences. Terminal draws are 24.9829% of 187,072 eligible mixture decisions.
Every report passes route accounting and independent integer terminal-count
reconciliation with its continuation-target mean. These are repeated exposures,
not unique endings: candidate experience contains 259 natural game endings,
versus 263 retained. The late window has 1.5403% terminal rows and 1.248% updates
without endings, versus .1034% and 76.352% retained.

Late expert-label cross entropy falls .560514→.480124 on each learner's own
replay, without a frozen gameplay gain. The mixture changes the data distribution
for **all** objectives without importance correction; it does not isolate the
continuation head. This comparison does not measure matched frozen terminal
calibration or establish that the actor's ending predictions improved.

Actual overrides are 91,453/200,000 actions (45.7265%), versus 43.0585% retained.
The 55 completed episodes with teacher actions average 210.218; 204 without
teacher actions average 15.985. Guided online returns are not frozen competence.
At pre-action score216–431 there are 5,708 teacher and 1,136 native actions,
versus 4,685/1,262 retained. Neither learner reaches432 during training; candidate
maximum including tails is416. Score bands are not verified wall events.

## Full-run validation and closure

Native `2b5a3eea`/SHA`5ce1b443` and the bounded runner are reused. Native, runner
and guidance-wrapper bytes differ from retained447fe50. The source review and
paired CPU fixtures described below support the active full18 episode-guidance
path; they do not establish byte equality or broad trajectory parity. These
source differences and changing experience limit isolated causal claims.

All 343 current and 338 retained pins pass, with one exact historical replay-source
resolution. Full helper/comparison qualification takes3.490s, including malformed
headers, losses, exposure counts and comparison decisions. All five actual queue
preflights pass in.732s with zero native-agent or emulator construction.
All357 initial tensors match the bounded and retained controls;47 pre-update
semantic events agree. The first update is376 and final residual credit is.5.
All12,477 reports are finite with correctly weighted world/behavior losses;
maximum world-total sum error is.000115580. Inspected CNN, reward, continuation
hidden/output, actor and value parameters change.

The100k/6,227 recovery and independent.570s CPU verification pass with357 finite
tensors and read-only weights/moments, preserving the original live identity.
CPU co-host time remains included without correction. All four frozen arms select
the first three natural episodes per lane,24 each, with zero updates/truncations
and357 invariant tensors per arm. Actual-initial sampled/greedy traces match799/794
retained semantic events. The four arms total85,544 actions/341,999 frames in54.657s.
Full training pixel/action/label replay, frozen recorded-action/outcome/frame/
boundary replays and videos, guards, curves, exposure, coverage and comparison pass.
Ordinary frozen logs contain no live pixel hashes; unrecorded live-pixel equality
is not claimed.

MainPID1114966/invocation`f812d2f076de4e0a9ff50d60a30199ba` launches
05:18:25 UTC November4/source`f5540278`. Training ends05:39:49, main succeeds
**05:43:42**, and the original collector succeeds **05:43:58** after2.638s.
All three original services are terminal success. The collector verifies all52
completion artifacts, current/retained pins and checkpoint hashes, and independently
recounts all training exposure reports. No full-stage failure, repair, cap change
or completed stage repeat occurs. Earlier bounded source-Python and supplementary
CPU-reader failures/repair remain preserved below.

Construction costs2.440s separately. Learner-wrapper time is1,134.356s, acting
63.209s and environment62.851s; these timers do not isolate GPU kernels. Training
time is .669% lower than the historical control, without an established speedup.
Root `/x/Temp/kindle-terminal-replay-20261104/full` is closed. No learner or GPU
allocation remains active; check live state before future work. Do not restart
training, assessments, replays or collectors.

| Full-run artifact | SHA256 |
|---|---|
|Declaration|`a86cbecdc5f27fad7cda5856078a81094590c076df1ac676bc8746108a9f632a`|
|Learning review|`d73ae778daae9334df1b55d59e691bd00f399ec921a0be8408efd26c2b251ff4`|
|Training replay|`fc49b1553b96ed4937067336691603abaa2340eb1e38aa6ac262e0acbbd7072c`|
|Paired comparison|`650a45ba3769b36631798cbc5c1f0ad2e7b6cc39f3812a74c75aa835f35d7a3b`|
|Main completion|`cc31f114089b95281c08beef80570cd7689c025353c178ee3ef5b1a83fecaa91`|
|Original terminal collector|`257f59b6e334999ea0eea5c6f32a03bb2e359e86d92e326a366f08725d0d4c7c`|

## Earlier bounded execution and cost check

The optional .25 terminal-sequence mixture completes **8,192 actions /32,722
executed frames /489 updates in46.900s**. Its conservative200k projection is
**20m59.517s**, below the30-minute cap. Terminal replay rows rise from322 to1,918
at the same early action/update budget, about **5.96×**. This establishes execution,
exposure and early cost; the short frozen capture completes no natural episodes,
so no gameplay benefit is established.

Retain the existing70.750 sampled/143.667 greedy control and continuation scale1.
The original864/90%-two-wall and five-game goal remain unfinished. No full learner,
extension, mixture sweep or automatic successor is selected by this bounded result.

## Fixed change and measured exposure

The [protocol](../experiments/2026-11-04-cdp-breakout-terminal-mixture.md) keeps
fresh-sequence priority. Each remaining sequence has a .25 chance of being drawn
uniformly from valid starts containing a true terminal in its training rows;
otherwise sampling is uniform. With no eligible starts it falls back to uniform.
Merged start ranges prevent duplicate weighting when several endings occur in a
window. Prefix-only terminals and timeouts do not qualify; intermediate life
losses retain their nonterminal meaning.

| First8,192 actions /489 updates | Retained uniform fallback | Terminal mixture.25 |
|---|---:|---:|
|Replay training rows|125,184|125,184|
|Terminal rows|322|1,918|
|Terminal row fraction|.2572%|1.5321%|
|Updates containing a terminal|225|435|
|Updates without terminals|264|54|
|Training seconds|47.690|46.900|

The candidate draws480 fresh,5,695 uniform and1,649 terminal-route sequences.
There are6,560 eligible non-fresh mixture decisions; the observed terminal-route
fraction is25.14%. Terminal counts match unique integer counts independently
derived from every continuation-target mean, within the fixed5e-5 tolerance.
The historical log digest is verified through its original guided-replay receipt,
which is itself pinned by the original terminal manifest.

Keep12M64/Auto/N8/B16/T16/H15/R16, base/encoder4e-5, dynamics4e-4,
behavior8e-5, replay-value1/expert100, sampled collection/posteriors, full18,
sticky.25/repeat4 and20% episode guidance/cap1920. Actual overrides are3,763;
all-state labels and the existing pixel teacher are unchanged. No experience is
transferred. The sampler changes every objective's training distribution without
importance correction; this is not a calibrated continuation-probability claim.
The native and runner differ from447fe50, and later experience changes. The small
historical timing difference is not an established speedup or isolated effect.

The subsequent full-stage source review clarifies that the **pixel-policy code
and selected guidance configuration** are unchanged, while the packaged teacher
and scheduler files have later optional block/minimal-action wrappers. Their
hashes are498f8f7b/951dd367 versus retained5120d021/12f671f5. Three policy-class
ASTs and constants match; the unchanged scheduler methods and active full18 episode
path pass a.545s paired CPU check over4001 steps/eight lanes,41 synthetic pixel
batches andfive partial-reset batches. This is scoped behavior evidence, not
byte equality or a replay of every trajectory. The [separate full protocol](../experiments/2026-11-04-cdp-breakout-terminal-mixture-200k.md)
records these source differences explicitly.

## Compatibility, execution and closure

Native source`2b5a3eea` packages at
`/x/Temp/kindle-terminal-replay-20261104/runtime`, SHA256
`5ce1b443ac270bd130ac1b1ead54d3e6b573913cfebd953fa3604367032a84a2`.
All177 Rust library/5 gym CPU tests, strict workspace/binding Clippy and1,028
packaged Python checks pass. Six new Rust tests cover exhaustive windows,
eviction, contexts, overlapping terminals, fresh priority, sampling distribution
and unchanged disabled/no-candidate samples/RNG. Twenty-two new Python checks
cover configuration, old-runtime rejection and malformed exposure reports.

The initial source Python check passed567 tests and failed two existing bootstrap
tests because it imported the older native default config. Both pass with the
matching package, without changing their assertions. The failed receipt remains.
Actual three-command preflight takes.974s with zero native-agent or emulator
construction,357 schema tensors,47 retained initial events and338 retained pins.

Before learning, one default-off1,024-action frozen compatibility capture matches
128 retained semantic events and preserves all357 tensors. This is scoped frozen
compatibility, not broad native or training parity. Fresh candidate initialization
matches357 retained tensors and47 pre-update events; first update376 and residual
credit.5 are correct. All489 reports are finite, with correctly weighted world
loss and matching exposure counts. Inspected CNN, reward, continuation, actor and
critic parameters change. Full training pixel/action/label replay passes.

The candidate's1,024-action/4,096-frame frozen capture takes.894s, performs zero
updates, preserves357 tensors and completes no natural episodes. All three host
guards and274 current pins pass. MainPID1114321/invocation
`fff881a0614b48a2b41979bc6f9030ab` launches05:03:05 UTC November4/source`27e00df2`
and succeeds **05:04:16**. Original collector1114324/invocation
`b6b1f648ae6442c2a827bc82ddf24cf9` completes **05:04:27** after1.953s; both
services are terminal success. All66 final artifacts are retained.

The supplementary CPU closure/exposure reader first fails on a historical
manifest lookup: that manifest pins the replay receipt, which contains the raw
training-log digest. Preserve the original reader and failure. A separate1.808s
repair follows that exact hash chain, verifies all66 artifacts and independently
recounts both early logs, completing **05:07:21**. Its status is
`complete_with_supplemental_reader_repair`; no GPU, learner, original audit or
collector repeats. Light progress reads remain a timing caveat without correction.

Root `/x/Temp/kindle-terminal-replay-20261104/bounded` is closed. The1.008GB
logical build cache is released after verifying the retained wheel, runtime and
test executable; the original build receipt stays unchanged and a separate
release receipt records cleanup. No GPU allocation remains active at closure.
Do not repeat completed stages; check live state before future work.

| Artifact | SHA256 |
|---|---|
|Bounded declaration|`3755787cb9ee739affe5fbd0c2a81316c721bb2f994ff0da2c98bd9ab1f016e7`|
|Frozen compatibility review|`23769bdd6f5eb32b31c0b644b4651a0fcb9be2d41224df18690596da13ba810f`|
|Learning review|`1f3c36789e6080936a9a540b946b603bbf2a9e50d92d4ee84156dc06af1b0431`|
|Original completion|`dc7a5a3d909f2d4996e7294e47aef1386cd3d351757261e2ddfad6a799553e5a`|
|Original terminal collector|`93bd595acfd5ce200347c23bdd7b19b198ec783a8d2b95653f793c5144bdc2d0`|
|Supplementary closure repair|`21cbd594ae3e15d003d2afbca746ed98d6ec53ca05612f2f4c26a7b8562e8115`|
