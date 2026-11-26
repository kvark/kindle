# Fixed frozen Breakout temperature assessment

After Qbert H5/R24 completes its original and supplemental audits, separately
select one frozen Breakout assessment under the fixed protocol below. This
selection adds no learner or training allocation.

The retained encoder4e-5 learner scores 56.292 with sampled actions and 104.667
with greedy actions after 200k actions / 12,477 updates in 21m28s. Both have zero
two-wall successes in 24 episodes. One fixed action temperature of 0.5 will test
an intermediate inference choice using the retained final and actual initial
checkpoints. It adds no loss and performs no training.

The primary comparison is final temperature0.5 versus retained greedy104.667;
sampled56.292 is secondary. Report absolute means, two-wall rates, own-initial
subtraction and return per original training minute. The original learner's
sampled-primary result stays unchanged. Keep actor unimix0.01 and sampled
posteriors. No temperature sweep or post-result primary switch is allowed.

## Fixed assessment

Run four arms: final default sample, final temperature0.5, actual-initial default
sample and actual-initial temperature0.5. Keep learner1009/evaluation4000001009,
eight lanes and the first three natural episodes per lane, full18, sticky0.25,
repeat4, native-frame input and one GPU resize to64. Preserve the entire
12M64/Auto/B16/T16/H15/R16/encoder4e-5/expert100 checkpoint configuration.
Guidance and labels are off. Require zero updates and truncations, 357 finite
invariant tensors per arm, full recorded-action/outcome/frame/reset replays and
videos, original process identity and host guards. Retain the 90% both-walls/864
task gate, without inventing a mean-return gate. Keep extra episodes and partial
tails in the evidence. Ordinary frozen logs lack live pixel hashes.

Use the qualified temperature runtime e416696 / SHA40872a3 at
`/x/Temp/kindle-action-temperature-20261029/runtime` and exact runnerabc2c2b from
its packaging receipt. No build or old qualification repeats. Both default
sampled arms must exactly reproduce all retained semantic events and scores
despite the runtime change. Stop on a mismatch; no retry or repeated GPU stage.
Historical source pins that differ from current files are resolved to exact Git
bytes and recorded separately.

Each arm has a 600k-action / 300-second cap; each CPU replay has 180 seconds.
The main service has 2400 seconds and its collector 2700 seconds. Separate
selection requires all six Qbert services and the independent terminal
confirmation complete, an idle host and unchanged pins. No training, extension,
cap increase, checkpoint selection or automatic successor follows.

## CPU qualification

The preflight passes at **09:05:09 UTC November 1**, source6300c1d, in 7.996s
(8.123s including its controller). It exercises all four actual CLI paths to the
boundary before construction, checks 32 valid/invalid headers, and re-audits
the four retained frozen arms, their scores, 357 invariant tensors, full replay
receipts, videos and guards. All 337 inputs and 401 active Qbert inputs pass;
Qbert PID771374/invocation82a217b818e24b5cb7c10801a2b3f237 remain unchanged.
Two historical runner references resolve to identical abc2c2b bytes from Git.

Native and emulator construction sentinels record zero attempts; no native
module, environment, new replay, updates, failure or retry occurs. No queue,
declaration or service is created. Retain CPU co-host timing without subtracting
it from Qbert training time. Do not rerun this passing preflight.

Root: `/x/Temp/kindle-breakout-temperature-20261101`.
Proposed service: `kindle-breakout-temperature-20261101`.

| Receipt | SHA256 |
|---|---|
| CPU preflight | `801c7af7334667dea86a6a556effbb66fb76940b6b34fcc272870604ccbef465` |
| Execution | `233a570f22cf9f58fb9a8bef139a998814d6a809e559ade4a9bc55ef56e633fa` |
| Exact runner | `abc2c2be23ef4c8c9b90e308a1d5a71a9e5201babebe29bd256e1f2395fe68fb` |
| Controller | `65869ade83a52d557f68474ff0d9a2e11a30a6ae232dd68387ec6c871b0547b0` |
| Preparation/closure | `5001434ecb73b35347886fc4bb33c2be540e1b6923681465038cdb73a608fcee` |

## Separate selection after Qbert closure

Qbert's six services finish successfully, with independent terminal confirmation
at09:23:42 UTC November1. Its accepted greedy15,430.208/24-pyramid result and all
audit receipts are recorded separately. The host is checked idle before this
selection, and the 337 prepared inputs are revalidated without rerunning the
passing CPU qualification.

Select the four fixed frozen arms to test whether temperature0.5 improves on
the retained greedy104.667 point at unchanged training cost. Keep every setting,
cap and rejection rule above. The source/runtime difference requires exact
default-sampled compatibility in both final and actual-initial arms. No result
is available at selection. Stop at any failed arm or audit; no retry, broader
temperature sweep, new learner or automatic successor is selected.

## Launch

The separately selected assessment launches **09:25:25 UTC November1**,
source`a285376`, mainPID776380/invocation33ea1a8cc3494d618ebffbe63ef1fa92.
The collector hasPID776383/invocationb96831bc0e6f469281e7ca94e6c19e23.
All348 inputs are pinned;declaration SHA256
`c61000f0934e09853feeaada426d920d033e8549270c3ff67f5fd0770e78374b`.
Startup header matches the selected runtime,runner,evaluation seed and restored
200k-action/12,477-update counters. No complete comparison is available yet.
Do not restart the assessment or schedule another GPU job while it is active.

## Completed assessment

Temperature0.5 scores83.292,20.42% below retained primary greedy104.667 and47.96%
above default sampled56.292. All0/24 two-wall;own temperature initial1.333. Both
default sampled arms reproduce scores and4394/799 semantic events exactly.
Retain greedy for this checkpoint;original sampled-primary result stays unchanged.

All348 pins,357 finite/invariant tensors per arm,full replays/videos,guards and
original terminal identity pass. Four arms81,016 actions/323,877 frames/72.156s,
zero updates/truncations. Original21m28s training remains. Main/collector finish
09:27:46/09:28:08 UTC November1,sourcea285376;both services succeed,zero failures
or repeated stages. No new learner,sweep,extension,cap increase or automatic
successor. See the [completed result](../results/2026-11-01-cdp-breakout-temperature.md).

Comparison`5bea06e0f94c27793fb4408293ce926931d2e07daa9a4caab7fc008a1d1f90a6`;
completion`101f8b1ca0fca32aaad8f95a35107298510c916f23edaf7c68d6c2367322e7cd`.
The assessment is terminal;do not rerun arms,preflight or collector. Check live
state before new GPU work;the five-game goal remains unfinished.
