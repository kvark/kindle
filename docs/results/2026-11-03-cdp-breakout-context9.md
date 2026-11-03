# Eight-step replay-state recomputation worsens frozen Breakout

The fresh `replay_context=9` comparison lowers sampled **70.750→46.583
(−34.16%)** and greedy **143.667→91.167 (−36.54%)**, with **zero of 24
two-wall successes** in every arm. Training takes **23m30.959s**, **9.55% longer**
than the retained context1 control. Initial-subtracted score per training minute
falls **40.54% / 42.60%**. Retain context1 and the existing control. No extension,
prefix sweep, confirmation learner or automatic successor is selected.

| Fixed evaluation | Retained context1 | Context9 | Two-wall successes |
| --- | ---: | ---: | --- |
| Final sampled, primary | 70.750 | 46.583 | 0/24 each |
| Final greedy | 143.667 | 91.167 | 0/24 each |
| Actual initial sampled | 1.417 | 1.417 | 0/24 each |
| Actual initial greedy | 2.208 | 2.208 | 0/24 each |
| Sampled initial-subtracted score/min | 3.230038 | 1.920680 | — |
| Greedy initial-subtracted score/min | 6.590132 | 3.782889 | — |
| Training time | 21m27.910s | 23m30.959s | — |

Both learners complete 200,000 aggregate actions. The candidate has 799,606 raw
frames and 12,473 updates, versus 799,595 and 12,477 in the control. Each frozen
arm selects the first three natural episodes in each of eight lanes. There are
no truncated selected episodes or learner updates; all 357 tensors remain
unchanged in each arm. Four candidate arms total 77,808 actions, 311,056 raw
frames and 62.208s. Both actual-initial semantic traces match the control at
800 sampled and 795 greedy events, and the initial video hashes match.
The unchanged gate requires 90% success at clearing both walls (864 points).
Breakout and the five-game goal remain unfinished.

## Change and interpretation

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-context9-200k.md)
adds eight forward-only observed steps after a cached posterior and before the
16 learning steps. The prefix uses current world/encoder parameters and an
independent RNG, has no loss targets, and detaches its last state. Only the
16 learning rows refresh replay caches. No parameters or inference inputs are
added. The first update moves from 376 to 440 actions; sampling eligibility,
fresh-window stride, warmup and total recurrent work also change.

Keep centered CDP 12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4, policy1, imagined/replay critic1/1,
constant actor/world expert100, unit action weights and all-state corrective
labels. Actor/critic representation gradients stay off. The same pixel teacher
guides 20% of episodes, capped at 1920 actions; collection and posteriors remain
sampled. Replay capacity is 100k. H3, centroid, crop and imitation decay are off.
No weights, replay, belief, RNG, games or experience transfer from another life.

Prefix timing is positive and finite in all 12,473 reports: **101.886s total**,
**8.169ms/update** on average, range 6.081–11.531ms. This is **7.22% of training
wall time**, not an isolated estimate of overhead or GPU utilization. The full
23m30.959s remains below the 30-minute cap, but exceeds the bounded conservative
projection of 22m37.921s. The measured full cost supersedes that early estimate.

Late 50k-action replay imitation CE is nearly unchanged, **.560514→.559628**,
while frozen scores fall. All updates are supervised, with 2,988,649 repeated
labeled replay rows and equal weight sum. Maximum raw world/actor CE difference
is 8e-7. Actual guidance overrides 88,684 actions (**44.342%**), versus 86,117
(43.0585%) in the control; 111,316 actions receive passive labels. Guided online
last50 mean rises 67.78→72.46. Neither guided returns nor replay fitting establish
unaided competence or explain the frozen regression.

The [CPU age audit](2026-11-03-cdp-breakout-replay-age.md) measured time since a
cached state was computed, not latent error. This negative gameplay result does
not show that cache age is harmless or that the prefix produced more accurate
states. Native65dd69b4/SHA7b15cb69 and runner30788c08 differ from the retained
447fe50/cf4ea238 control. The explicit `replay_value_bootstrap=imagined` is the
old implicit default and the only configuration normalization besides context.
Initial values match all357 tensors against both bounded check and control;
47 common pre-update events match. These scoped checks do not establish broad
runtime equivalence or isolate a causal cache-age effect. Seed1009 and its
evaluation cohort are reused development data, not independent confirmation.

## Full comparison closure

All **667 pins**, cadence and finite-report checks, changed CNN/reward/actor/value,
full training pixel/action/label replay, frozen action/outcome/frame/boundary
replays and videos, host guards, curves, exposure and comparison pass. The training
replay verifies 257 completed episodes. Ordinary frozen logs lack live pixel
hashes; their replay does not establish unrecorded live-image equality.

The original recovery observer succeeds **13:32:50.987 UTC November 3**.
Independent recovery reading passes once at13:33:51.402, verifying100,000
actions/6,223 updates,357 finite tensors and four files totaling206,067,205 bytes.
These retain weights and metadata, not replay, RNG, games or live belief.

All five original services succeed. Main/postprocessing/collection finish
**13:48:19.732 /13:48:49.527 /13:48:54.418 UTC November 3**, source
`55f0a46a5af7a5fda099f8a8d47540fd1848352d`. MainPID1029287 has invocation
`da2979f074bf4fe4b740d734ad33f6cd`; original observer identities remain in the
protocol and launch records. The predeclared independent terminal reader passes
once at **13:49:17.338**, in1.706s body/1.766s wrapper, CPU only.

No full preparation, training, evaluation, replay, audit or collector stage
fails or repeats. An ad-hoc live-state poll rejects the unloaded completed
recovery unit's empty invocation field; its original successful terminal journal
and retained recovery evidence remain intact. The fixed terminal reader verifies
the original service identities from journals, including unloaded units.
No stage is restarted because of that observation-only assertion.

The [earlier qualification failures and repairs](2026-11-03-cdp-replay-context.md)
remain preserved: CPU target-shape/unused-helper issues, the RGB fixture, the
bounded queue's489-vs485 assertion and its decay-only audit schema. The original
bounded service remains failed with `complete_with_cpu_controller_repair` closure.
No successful bounded stage repeats. The qualified native runtime and runner are
reused without a build. Keep CPU co-host timings without wall-time correction.

Root `/x/Temp/kindle-replay-context-20261103/full` and all its readers are closed;
do not repeat them. No Kindle allocation remains active at closure. Check live
state before future GPU work. No extension, sweep or automatic successor follows.

| Full artifact | SHA256 |
| --- | --- |
| Declaration | `32cd53f38f4ccab9dbd13338c0d8c020879b96dfa5583ab22a8b7ebd79100a48` |
| Learning review | `c7bf0176151907cef719422370ba3b6062941bf6002fa1e8337f4971932435e5` |
| Comparison | `899a5c58014bbc108014f29344d8ca26ee670181834768f74d9390f19ed1b5f2` |
| Completion | `2599a36ed6b57d3919e02a2a62cc58d71d725f621e40c7294dee1ed15bfa60f7` |
| Independent terminal summary | `624700fe0d397d90f6b5021088364cf4ed9ddd891ba626349ca3fd66a02eaabc` |
| Terminal execution receipt | `3e170582bd7e1af0a906f25847caac3de14285cc6cdada6a5696719c3c59dc1e` |
