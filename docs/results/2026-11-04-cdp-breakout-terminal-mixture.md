# Terminal replay raises ending exposure at acceptable early cost

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
