# Validate the fixed catch signal against raw motion and retained-policy replay

The [first catch-signal check](../results/2026-11-04-cdp-breakout-catch-signal.md)
covers all776 fitted reference contacts but leaves700 other signals unresolved.
Select a CPU-only follow-up within the five-game goal. Keep that detector
byte-for-byte unchanged; no detector threshold change,reward coefficient or
learner is selected.

First inspect the already saved raw centroids and paddle components for the
same24 teacher episodes. Do not repeat the teacher replay or original detector
analysis. Second perform one instrumented CPU replay of the retained
replay-value1 `final-greedy.jsonl`:43,424 actions/173,660 frames,seed4000001009,
eight lanes,native max-pooled RGB,repeat4/sticky.25/full18/noops0. Reproduce every
recorded action,reward,frame counter,boundary,reset and pooled pixel hash against
the existing command-takeover annotations. Capture raw ball centroids,paddle
components,life timestamps and causal pooled observations that were not retained
by that earlier audit. No model construction,inference,training or new policy
trajectory. Select the same first three natural episodes per lane,24 total.

For pooled observations use exactly the existing teacher's red threshold and
ball crop90:185/x8:152,maximum32 red pixels,paddle crop185:198/x8:152 and minimum
five paddle pixels. Only ball/paddle visibility and ball y enter the unchanged
causal detector. Signals observed before actionj credit actionj−1. Verify that
the small observation extractor matches the retained teacher on synthetic frames.
The teacher does not choose or change any replay action.

Use a separately fixed raw-motion witness,applied identically to both corpora.
Reuse the historical raw centroid rule: red pixels from y90 up to the detected
paddle row,capped at189; a ball needs1–16 pixels. Pair this with the existing
longest contiguous paddle-component detector. Within an episode, form raw motion
edges only between valid observations at most two frames apart. A positive-y
edge of at least.5 pixels may be followed by a negative-y edge of at least.5,
with at most six frames between incoming endpoint and outgoing start. Require
both endpoints at y≥170 and their maximum y≥180. Emit one temporal reversal,
then clear the incoming edge until a new positive edge. Invalid long gaps and
episode boundaries prevent joins. Save both raw edge pairs,never a fitted
future projection.

Call a temporal reversal spatially supported when either endpoint has a unique,
uncropped paddle component and ball x is within its half-width plus6 pixels.
This fixed margin allows ball extent and time-discretization uncertainty; it
does not establish exact collision physics. Keep temporal-only events separate.
A pooled signal is an eligible neighbor when its observation frame lies from
incoming endpoint−2 through outgoing start+12. Accept only mutually unique
neighbors; preserve ambiguous,unmatched and temporal-only rows. Match both the
full raw set and the spatially supported subset. No outcome-driven widening.

Report signal coverage/support,raw events without a signal,timing and all
unresolved cases. Split the teacher's776 previously matched and700 previously
unmatched signals. For both cohorts, report immediate raw reward,first-eight-
decision signals,and signals within16 raw frames before or after a life loss.
These are descriptive negative-control windows,not automatic false-positive
labels. Shared color heuristics,selected cohorts and missing detections prevent
claims of population precision or exact ground truth. No game benefit follows.

Qualify raw-motion,spatial,matching and reset rules with small synthetic fixtures;
retain the first detector's16 passed tests without repeating them. Pin protocol,
private scripts,source logs,previous receipts,wrapper,ROM,ALE and all consumed
helpers before data work. Check live state. CPU capture cap180s,analysis60s,
controller240s. Save stage receipts and preserve failures; repair only unfinished
work. Independently recount output matches/totals without repeating capture or
detection. Verify declared inputs afterward. No GPU allocation,checkpoint change,
coefficient sweep or automatic successor.

Root `/x/Temp/kindle-breakout-catch-validation-20261104`.
