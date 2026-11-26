# Fresh 200k Breakout comparison with replay capacity16,384

**Complete:** replay16k scores34.000 sampled/57.625 greedy versus70.750/143.667
forretained replay100k,in21m44.359s (+1.28%),all0/24 two-wall. Retainreplay100k;
no capacity sweep,extension,confirmation orautomatic successor. Alloriginal
services,recovery,replays andfinal readers pass withoutfull-run failure orrepeat.
The following selection,launch andrecovery sections describe historical states;
theroot isclosed. See the [completed result](../results/2026-11-03-cdp-breakout-replay16k.md).

Separately select one full learner after the [bounded replay16k check](../results/2026-11-03-cdp-breakout-replay16k.md)
establishes execution and cost. It completes24,576 actions/1,513 updates in
158.358s, including515 reports after saturation; conservative200k projection
23m37.590s clears30minutes. All378 original/408 repair pins,357 matching
initial/invariant tensors,16k semantic prefix,full guided replay and guards pass
in the separate corrected CPU audit. The original main and collector remain
failed because they compared wall-clock fields. Require that preserved failure
and repaired closure; do not repeat the check, GPU stages, replay or collectors.

The hypothesis is that retaining more recent experience may help learning as
the representation and policy change. Only the shared replay capacity changes
100,000→16,384 stored frames. It is an arrival budget across eight streams,
not a per-stream action count. Recency, diversity, reuse and coverage change
together. Cache age remains unmeasured; this does not establish stale-state
failure or isolate a mechanism.

Use a fresh200,000 aggregate actions, expected12,477 updates, with a fixed
1,800-second training cap. Keep12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16,
lambda.95,encoder/base4e-5,behavior8e-5,dynamics4e-4,replay-value1,expert100/unit
weights/world gradients/all-state corrective labels, sampled collection and
posteriors,actor-unimix.01,action-effects1,warmup1000,20% episode guidance/cap1920,
learnerseed1009/evaluationseed4000001009. Full18/sticky.25/repeat4/noops0,
native pooled frames/one GPU64 resize, native447fe50/SHA35ad0db, exact retained
runnercf4ea238 and teacher stay fixed. No native build or source implementation
change. No weights,replay,live belief,games,RNG or bounded experience transfer.

Require357 initial tensors matching both bounded candidate and retained control,
matching pre-update semantics and the complete16,000-action pre-eviction prefix.
Reuse the exact corrected reader, excluding only unix_time and elapsed_seconds;
its eight retained fixtures cover semantic differences and the boundary.
Retain100k/6,227 recovery weights with exact metadata and the final checkpoint.

Run fixed final and actual-initial sampled/greedy assessments: first three
natural episodes in each of eight lanes,24 per arm, zero updates/truncations,
guidance and labels disabled. Each arm has a300-second cap and the unchanged
25,000-action episode limit. Require357 invariant tensors per arm, exact paired
actual-initial semantic traces, complete action/reward/frame/reset replays and
videos, full guided pixel/action/label replay, finite learning reports, changed
CNN/reward/actor/value weights and host guards.

Compare sampled primary and greedy separately against retained70.750/143.667
and own initials1.417/2.208, allzero of24 two-wall. Report actual training time,
actions/frames/updates, raw and initial-subtracted return per minute, natural
episode outcomes, both seeds and actual guidance exposure. Preserve the original
90% two-wall gate. A development score gain does not establish success on that
gate, seed reliability or a causal explanation. No checkpoint/cohort selection
after outcomes, capacity sweep, retry, extension, cap increase, confirmation or
automatic successor.

Automate progress,100k recovery, postprocessing and original terminal collection.
CPU preflight must exercise actual command parsers, exact configuration checks,
startup gates, observer arguments, recovery retention, source/provenance paths,
the repaired-prefix source identity and both original failed bounded services.
The training service cap is4,500s; total CPU audit cap1,200s. Existing bounded
observer/collector envelopes remain. Check live state before launch and allow
no overlapping Kindle GPU jobs. Other CPU co-hosting remains a timing caveat;
do not correct wall time afterward.

Root `/x/Temp/kindle-breakout-replay16k-20261103/full`, trial
`breakout-12m-64-b16-r16-replay16k-expert100-200k-seed1009`, service
`kindle-breakout-replay16k-200k-20261103`. At selection, full gameplay benefit
remains unmeasured and the five-game goal remains unfinished.

## CPU preparation

Preparation completes05:07:30.979 UTC November3 in3.683s (3.800s wrapper).
Five actual commands parse without native/emulator construction;36 exact config
cases,20 startup cases,13 recovery cases,four observer argument cases,seven
prelaunch command cases,four metadata cases,six retention/two collector cases
pass. All19 scripts compile and1,209 declaration path entries resolve. Original
378/repair408 bounded pins and338 control pins stay unchanged; the prefix
reader is structurally identical to the repaired reader with eight retained
fixtures. No new failure or repeated successful stage occurs before selection.

## Launch and startup

Launch05:08:40.375 UTC November3/source9cfd8cc selects445 pins; declarationSHA
`75d022883e179433719403cc039faf86506ce535e5c35d7ff92c46c71be972fd`.
Main977721/invocation84334254aa564f5ab49a4907f2d9e950,post977724/
253fdc1765274039b00deb8e25f5a609,progress977727/e702b9e4ff894fb3aead7afb68c2eaf8,
recovery977862/20514ed2ac8c44e8af047343bcd812b1 andcollection977865/
eec7db69ce684cf4a3973d1c1223107d are active atstartup.

Startup review05:08:57.566 passes all445 pins,357 initial tensors matching both
bounded candidate andretained control,47 matching pre-update transitions and
119 finite supervised reports through2,280 actions. The firstsix updates are
376/392/408/424/440/456. No100k recovery orfull frozen result exists atthis
review. Keep1.636s CPU startup wrapper co-hosting without correction. Follow
these original identities; do not restart oroverlap GPU work.

## Read-only result preparation

Terminal-reader preflight completes05:11:24.713 UTC November3 in2.806s
(2.828s wrapper), checking all445 full pins,allfive original identities,the
capacity-only configuration andfive score validation cases. Its source and
provenance are pinned separately; no active controller changes. The recovery
reader isprepared but must run only afterthe original100k observer succeeds.

Before frozen outcomes, fixed score coverage qualifies05:11:33.246 in2.859s
(2.884s wrapper),with50 diagnostic/445 full pins,six configuration cases and
five retained boundary fixtures. Afteroriginal completion andterminal review,
count only the new candidate's train andfour frozen logs using unchanged
pre-action score bins216/432/648/864. Reuse the completed control analysis:
1,262 native actions at216–431 andnone432+. Reconcile all actions,frames and
episodes against original replay artifacts; preserve the fixed first-three-
natural-episodes-per-lane cohort separately from extra episodes andtails.

These bins describe exposure, not verified wall boundaries, visual coverage or
a causal replay mechanism. CPU co-host times remain uncorrected. No GPU work,
learning,emulator replay orcompleted stage repeats. Run recovery,terminal and
coverage readers once aftertheir respective prerequisites; completed preparation
andpreflight must not repeat. Keep bounded failure andCPU repair provenance.

## Retained recovery

The original recovery observer succeeds05:19:46.992 UTC November3. Its saved
checkpoint contains exactly100,000 actions/6,227 updates,357 finite tensors
and206,067,161 retained bytes. Independent confirmation passes05:19:54.953 in
1.884s (1.902s wrapper),verifying all445 input pins,matching metadata andfiles,
andthe original observer journal identity. The checked log prefix contains
6,385 finite reports through102,536 actions. Recovery-reviewSHA
`b8421bdaf8a4eea287075d023f01625ee72e927226a3aaf31d920cd97c7fd63e`,
retentionSHA`64322cb0bfb7140917faa7de0c2ce9d5e95950cd426e49b2609f475023232ce9`.

The original main977721/invocation84334254aa564f5ab49a4907f2d9e950 remains active.
Recovery preserves weights andmetadata only; replay,belief,games andRNG remain
omitted. No training,observer orGPU stage repeats. Do not repeat the completed
recovery confirmation. Frozen gameplay remains pending; retain CPU co-host time
without correcting training time.

## Completion

The candidate finishes200,000 actions/799,573 frames/12,477 updates in1,304.359s.
Fixed finalsampled/greedy means34.000/57.625 fall51.94%/59.89% belowcontrol;
initial-subtracted return/min falls53.60%/61.32%. Owninitials1.417/2.208 match,
all0/24 two-wall. Fourfrozen captures total70,920 actions/283,512 frames/61.452s,
zero updates/truncations. Retainreplay100k; no follow-up sweep orextension.

All445 pins,100k recovery,357 matchedinitial/invariant frozen tensors,16k semantic
prefix,800/795 pairedinitial events,replays/videos,guards,curves,exposure and
comparison pass. Originalmain/post/collection succeed05:34:16.487/05:34:44.611/
05:35:19.248 UTC November3. Allfive originalservices succeed. Terminalreview
05:35:33.562 andcoverage05:35:39.907 pass once;50 diagnostic/445 fullpins and
270,920 actions/1,083,085 frames reconcile. Native216–431 trainingactions fall
1,262→878,none432+; descriptivescore bins donot establishcausality. ExpertCE
falls.560514→.381669 onthe changedreplay distribution despiteworsefrozenplay.
Actualoverrides90,825/45.4125% andonline54.42 arenot unaidedcompetence.

No full-run preparationfailure,GPU stage,replay ororiginalcollector repeat.
Earlier boundedfailures andseparate CPUrepair remainpreserved; theiroriginal
services remainfailed. KeepCPU co-host times uncorrected. Allreaders andstages
arecomplete; donot repeat them. No allocationactive atclosure orautomatic
successor selected. Thefive-game goal remainsunfinished; checklivestate.
