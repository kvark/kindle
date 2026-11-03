# Breakout: local headroom from two-frame command changes

Select one CPU counterfactual diagnostic after the completed life-loss comparison.
Earlier teacher takeovers avoided many more recorded losses than late takeovers;
the retained controller and Kindle both hold commands for four emulator frames.
The teacher's response model is calibrated to that interval, so changing the
wrapper alone would not cleanly test faster feedback. Measure a narrower question
first: can changing the command halfway through one four-frame action rescue
losses that no constant command in the same family rescues?

Reuse all120 last-near-descending origins from the completed encoder4e5
recoverability diagnostic, one per selected lost life in the first three
natural episodes of eight lanes. This is the older greedy104.667 policy with
behavior4e-5/replay-value.3, not the current143.667 control. Preserve the exact
origins, recorded actions, pixels, teacher history, lives and episode identities.
Do not select new origins after seeing branch outcomes.

At each origin, enumerate all nine pairs from NOOP/LEFT/RIGHT. Execute the first
command for two raw frames and the second for two, with normal sticky probability
.25 and the original final-two-frame max pooling. The three identical pairs are
the constant-four-frame family. Then execute the same recorded continuation
until the original loss endpoint or the first new lost life/terminal/cutoff.
Observe life counts at ordinary four-frame wrapper boundaries; natural terminal
and episode time limits stop immediately, as in the retained wrapper.
Do not reset a branch. Include a separate exact recorded-action continuation.
This oracle chooses after observing outcomes; no controller has this information.

Primary outcome is avoiding life loss through the original endpoint. Report
any-success counts for the three constant commands and nine split commands,
their paired difference and per-lane counts. Retain every individual result,
reward, frame count, early termination and pixel hash. The split family contains
the constant family. A development screen requires at least12 additional saved
losses out of120 and gains in at least five of eight lanes before considering
a separately selected shorter-interval implementation. This is not a gameplay
or training acceptance gate. Failure retains the four-frame recipe without a
window/origin/horizon sweep; it does not rule out repeated two-frame feedback.

Use process copies to preserve ALE RNG and sticky-action memory. Verify unchanged
parent emulator, wrapper buffers/counters and teacher state. Require every
constant-pair branch to match an independently executed ordinary wrapper branch,
and reverse all nine branch orders at the first origin of each lane. The recorded
branch must reproduce the saved trajectory and lost-life endpoint exactly.
Replay the complete37,016-action/148,038-frame source once for these new branches,
matching every saved row and original task ledger. Do not rerun older diagnostic
stages or overwrite their evidence.

Qualify raw-frame command scheduling, final-two-frame pooling, early terminal,
episode cutoff, reward/frame counters, origin selection and paired arithmetic
on CPU before execution. Pin actual sources, prior completed evidence, wrapper,
teacher, ALE and ROM. One low-priority CPU core; analysis cap180s, controller210s,
per-branch5s. No native model, GPU descriptors, learner updates or RAM policy
inputs. Life counts are outcome measurements only. Preserve failures and partial
results. Root `/x/Temp/kindle-breakout-subaction-20261103`. No new learner,
teacher integration or automatic successor is selected by this protocol.
