# A two-frame command switch adds no rescues at the fixed late loss states

Both candidate families avoid the recorded loss in **34 of120 cases**: three
constant four-frame commands, and nine pairs that may switch commands after
two frames. The split family adds **zero rescues in every lane**. It fails the
fixed development screen of12 additional rescues and gains in five lanes.
Retain the four-frame recipe; this result does not select a shorter-interval
learner, teacher change, window sweep or automatic successor.

| Lane | Fixed loss states | Any constant command avoids loss | Any split pair avoids loss | Additional rescues |
| --- | ---: | ---: | ---: | ---: |
|0|15|5|5|0|
|1|15|5|5|0|
|2|15|3|3|0|
|3|15|5|5|0|
|4|15|3|3|0|
|5|15|3|3|0|
|6|15|3|3|0|
|7|15|7|7|0|
|Total|120|34|34|0|

## What was measured

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-subaction.md) reuses
all120 last-near-descending origins from the completed recoverability study,
one per lost life in24 natural episodes. These belong to the older encoder4e5,
behavior4e5/replay-value.3 greedy policy scoring104.667, not the latest143.667
control. Loss conditioning, reused trajectories and correlated states limit
generalization. No origin is replaced or selected using the new branch outcomes.

The three commands are NOOP, LEFT and RIGHT. Each of the nine pairs applies its
first command for two raw frames and its second for two. The three identical
pairs form the constant-command family. Sticky probability.25, final-two-frame
max pooling and actual terminal/time-limit handling stay unchanged. After that
one four-frame interval, every branch executes the same recorded continuation
through the original lost-life endpoint, stopping earlier on observed life loss
or a terminal/cutoff. Life counts are checked at the ordinary wrapper boundary.

The result is an oracle best-of-family comparison: no deployable policy is
given the branch outcomes. The larger family contains the smaller one, so it
cannot have fewer best-case rescues. The equality therefore finds no additional
rescue opportunities from this one local switch under the fixed continuation.
It does not rule out repeated two-frame closed-loop feedback, earlier commands,
different future actions or a differently trained learner. Nor does failure of
all nine plans establish that a loss is physically unavoidable.

The earlier teacher-takeover result remains108/120 rescues from earlier origins
versus26/120 from late origins. Its adaptive continuation differs from this
recorded continuation, so those counts are not a controlled comparison with34.
Earlier anticipation and action choice remain unresolved; this diagnostic does
not establish a causal architectural defect or improve any gameplay score.

## Verification and closure

Eight CPU tests pass, including30 constant-command comparisons with the actual
wrapper across natural terminals, episode cutoffs and ordinary completion.
Command scheduling, final-two-frame pooling, reward/frame counters, invalid
inputs, fixed origins and paired screen arithmetic are checked. Preflight
completes **07:09:08.824 UTC November3** in1.873s; all293 prior dependency pins
match. Selection source is **ba52b20**, with302 declared inputs.

The new diagnostic completes **07:10:16.312**, taking **35.736s**; its controller
closes **07:10:17.145** in36.606s. It evaluates1,080 candidate branches,
360 ordinary-wrapper parity branches,120 recorded controls and eight reverse-
order checks. All constant-pair outcomes match the independently executed
ordinary wrapper exactly. Process copies preserve ALE RNG and sticky-action
memory; parent emulator, wrapper buffers/counters and teacher state are unchanged.
Candidate branches execute23,421 frames and validation branches12,172.

The complete37,016-action/148,038-frame source replay matches every saved pixel,
action, label, life count, reward and boundary row, plus the original episode/task
ledger. Its saved-row SHA62fb6761 matches the prior capture exactly. No older
diagnostic stage is rerun; this replay supplies the new counterfactual branches.
All302 input pins remain unchanged. No failure, native model import, GPU context
or learner update occurs. One low-priority CPU core is used.

Independent terminal recount completes once **07:11:01.186** in.637s, verifying
all completion files and302 pins and deriving the same34/34/0 counts directly
from branch trajectories. Source scores, retained training time and the original
two-wall gate remain unchanged. Breakout and the five-game goal are unfinished.

Root `/x/Temp/kindle-breakout-subaction-20261103` is closed. Its private scripts,
tests, declarations, every branch trajectory, source replay and receipts are
retained. Do not repeat completed qualification, execution or terminal recount.
No Kindle learner or GPU allocation is active at closure; check live state
before separately selected work.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`44cce02644c3939df641fa5a8f948ae37aaa4dd97257b02a191eec641436eeba`|
| Review |`8b92a0c2e775df71621f3037414c6574826f687218bdfdca13b773157d669d88`|
| Branch trajectories |`e832017cab719053b221184cbba7d3c2e03f56374275dd5776d72692d3a47e79`|
| Completion |`7e72eafb15bea30f7f119e7572bb6bc02c4e5fb9280de0bfd96cdcc0dde21fcf`|
| Independent recount |`58e8f65eea8d78ff478fb802a5594d555923c87267409ae12ca6485aa1dbeadd`|
