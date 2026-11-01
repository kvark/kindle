# Frozen Qbert comparison across the two retained evaluation cohorts

Select four new frozen arms after the Breakout greedy-collection comparison
completes all five original services and its independent confirmation observer.
That negative comparison is accepted;sampled collection remains selected for
Breakout. The host is idle at this separate selection. This diagnostic belongs
to the existing five-game goal and trains no new learner.

The fixed Qbert greedy-collection agents score15,515.625 on cohort4000004013
and11,376.042 on4000008039. Learner and evaluation seeds differ simultaneously.
Evaluate each retained agent on the other environment cohort to see whether
the observed ranking persists on matched cohorts. This measures frozen-agent/
cohort dependence;it does not isolate learned weights from checkpoint config
or posterior randomness,and does not establish broad seed reliability.

| New greedy arm | Source checkpoint | Environment cohort |
|---|---|---|
| 4013 trained | Fixed200k final | 4000008039 |
| 4013 actual initial | Saved initial | 4000008039 |
| 8039 trained | Fixed200k final | 4000004013 |
| 8039 actual initial | Saved initial | 4000004013 |

Reuse both completed diagonal final/actual-initial greedy assessments without
repeating them. Report the complete2-by-2 trained and own-initial matrices and
initial-subtracted scores. The original8039 gate miss remains unchanged even
if another cell passes. Do not select whichever evaluation cohort passes or
call these reused agents independent learner confirmation.

Keep sampled posteriors and each checkpoint's own config. On restore,the CLI
seed changes environment/lane seeds while the agent config comes from the
checkpoint. Retain native447fe50,exact runneraba4300d,full18/sticky.25/repeat4/
native48 preprocessing and greedy actions. Guidance and labels are off;no
model,loss,teacher,build,backend or learning-setting change. Original training
times27m33s/27m34s remain;this assessment makeszero updates.

Each new arm selects the first three natural episodes per eight lanes,24 per
arm,under600k aggregate actions/300s. Keepzero updates/truncations,357 finite
invariant tensors,source checkpoint hashes,full outcome replays/videos and host
guards. Retain excess episodes and partial tails. Ordinary frozen logs lack
live pixel hashes. Show the unchanged15k/90%-first-pyramid gate in every cell.
Replay cap180s per arm,main service2400s,terminal-collection observer2700s.

No learner,training extension,cap increase,retry,mode sweep,checkpoint selection
or automatic successor. The controller checks completed Breakout closure and an
idle host before selection,and original invocation before terminal collection.
Preserve failures without repeating completed GPU stages.

CPU command qualification completes08:06:08 UTC November1 in5.637s (5.671s
command):four actual CLI paths,32 header cases,357 finite tensors per source
checkpoint,388 preparation/272 then-active Breakout pins and original identity.
The first checker assumes newer optional CLI fields;its source/failure remain
in`preparation-failure`. Missing fields default to absent and both optional flags
are explicitly forbidden. Exact retained runner/native remain unchanged.

Controller qualification completes08:12:20 in4.387s (4.424s command),reproducing
all four retained diagonal scores and357 finite/invariant tensors per arm,
full stored replay/video/guard identities and four journal fixtures. A first
6.628s CPU check rejects the ALE module import needed for replay identity;its
source/execution remain in`controller-failure`. The corrected test blocks both
ALEInterface exports and gymnasium.make and recordszero construction attempts;
production arm audit is unchanged. No completed CLI qualification or diagonal
GPU stage repeats. Keep all CPU co-host timing without correction.

Root `/x/Temp/kindle-qbert-cross-cohort-20261101`;main service
`kindle-qbert-cross-cohort-20261101`,collector
`kindle-qbert-cross-cohort-collection-20261101`. Prepared sources are pinned;
do not rerun successful preflights. Launch once after the selection commit,
clean-worktree,input-hash and idle-host checks. Cohort dependence is unmeasured
at selection;the five-game goal remains unfinished.

## Original launch

Launch08:24:54 UTC November1,source63fce09,413 pinned inputs. MainPID770056/
invocation`d56c32f594864b018418ed008250835f`;collectorPID770059/invocation
`e17b21c8b78648fb8aeccf316d7cc5df`. The first4013-on8039 final run completes
08:25:49 with a passing host guard;its outcome replay finishes08:26:13 with
24 first pyramids. Remaining arms and terminal comparison are pending. No
original stage repeats;keep this original process and collector running.
