# Greedy assessment of the retained equal-weight Qbert policy

**Completed: greedy actions improve this retained policy, but both gate criteria fail.**
The fixed final/own-initial comparison uses **zero new training**.
The completed100k/R32/NOOP-weight1 recipe costs18m58s but scores2260.417/12of24
first pyramids under sampled actions. Its last50k passive wait/movement command
agreement is88.121%/33.651%; weighted replay CE.224. These changing-state metrics
do not prove an alignment error, label accuracy or a cause of weak policy transfer.
The inspected native path pairs outgoing recommendations on frame t+1 with state
t and trains the actor on matching posterior features. Existing alignment tests
and full teacher/pixel/action replay pass. No implementation defect is established.

A greedy assessment cheaply tests whether removing action sampling helps this
particular equal-weight policy. Prior Qbert greedy assessments used different
training recipes and lowered scores; they neither predict this result nor justify
a global default. This is a separately selected frozen check, not an extension
or retry of the completed life. Original sampled outcomes and training cost stay.

Source:`/x/Temp/kindle-qbert-wait-full-20261028/qbert-4m-48-r32-wait1-100k-seed4013`.
Use fixed final at100k actions/24,939 updates and its actual initial at0/0.
Same native447fe50, exact runner/runtime
`/x/Temp/kindle-corrective-labels-20261028/runtime`, wrapper, host and restored
config:4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
expert1/all18 action weights1/world gradients. Guidance and labels are off during
evaluation. No checkpoint metadata, parameters or source files are changed.

Evaluate both with native greedy actions, seed4000004013, first3 natural episodes
per8 lanes=24 each, original mean15000/90%-first-pyramid gate. Require zero
updates/truncations,357 finite and invariant tensors per arm, unchanged source
hashes, full recorded-outcome replay/videos and host guards. Ordinary frozen logs
have no live pixel hashes. One retained learner and reused development cohort do
not provide independent confirmation. Keep this additional evaluation cost visible.

Each arm has300s/600k-action caps; CPU audit600s, owning service1260s. No retry,
cap increase, extension, checkpoint selection or automatic successor. Root
`/x/Temp/kindle-qbert-wait-greedy-20261028`, service
`kindle-qbert-wait1-greedy-frozen-20261028`. The predecessor and its postprocessor
are terminal success; no GPU job is active and no kernel errors follow its last
guard. Check current live state immediately before launch; stop on real faults,
with no driver reset/reload/reboot.

CPU preparation parses both commands before native/environment construction,
compiles6 scripts, checks both357-tensor source checkpoints, reproduces sampled
scores from saved outcome replays and verifies refusal while unselected. All47
new declaration pins and55 predecessor inputs pass. Exact declaration and launch
identity will be recorded externally before execution. No production code changes
or broad backend qualification are part of this assessment.


## Launch

The two-arm service starts19:02:41 UTC/source`1222ce9` on October28, PID192628/
invocation`da526475b8194663b55c86b9c8a23abd`. Declaration
`ae93f4ca26aee04f35d4f357f48c08e02520faa2220808881a8ab8f4bf5dc341`
pins47 inputs; all55 predecessor pins remain unchanged at preparation. A separate
CPU observer verifies terminal success and preserves the complete audit. No
training or runtime/default changes occur; no successor is selected.


## Completed assessment

Greedy final scores4965.625/21of24 first pyramids versus sampled2260.417/12.
Own actual initial greedy263.542/0, sampled182.292/0. Original mean15000/90% gate
still fails both criteria. Retain greedy as a useful development option for the
18m58s learner, not a global default or independent confirmation. Original
sampled outcome and training cost remain. No extension, learner confirmation or
automatic successor is selected.

Final31,544 actions/126,141 frames/14.710s and initial10,432/41,697/4.752s
finish with zero updates/truncations. The declared first3 natural episodes per
8 lanes select24 each from27/25 collected; excess and tails remain. All47 pins,
55 source-training inputs,357 finite/invariant tensors per arm,source-file
invariance,full outcome replay/videos and host guards pass. Main succeeds
19:03:28 UTC,whole46.404s; CPU postprocessing19:03:31 on October28. No GPU stage
repeats or cap changes. Detailed comparison and limits are in the
[completed result](../results/2026-10-25-cdp-cheap-qbert.md#greedy-actions-improve-the-retained-equal-weight-policy).

Review`fa142ce8a695278f1e2e8136cff3e59a325650fbc11c2164fbd38d7900e5e74e`;
completion`df9f451190b2d1b041a032cd170685d130ae18e8a5983442948daa2f8221ff8c`.
No GPU job is active at completion; check live state before future work.
