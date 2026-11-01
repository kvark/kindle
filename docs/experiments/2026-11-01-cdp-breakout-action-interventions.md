# Breakout teacher disagreements and local action outcomes

The saved probability diagnostic finds18 of51 near-descending candidate
teacher disagreements with confidence at least.75. These are greedy choices,
but the pixel teacher is not ground truth. Select a CPU-only intervention
diagnostic before choosing another model or loss change.

Use all136 near-descending rows from the existing fixed601-row test set in
lanes6/7 of the retained control-policy corpus. Reuse saved encoder6e-6/control
and encoder2e-5/candidate probabilities;do not run either model again. Keep the
exact input identities,pixel hashes,labels,phase definitions and split. Do not
select origins by reward,confidence or future outcome. Confidence≥.75 and≥.9
are predeclared reporting groups,with every origin retained.

Replay both complete2620-action source streams. At each selected origin,
compare the teacher action and both saved greedy actions,sharing a branch when
actions coincide. Each branch executes exactly one chosen action (the ordinary
four-frame repeat with.25 sticky actions),then the same feedback-aware pixel
teacher for up to63 further actions. Stop at the first lost life,natural
terminal or original episode time limit. Do not reset a branch. The continuation
rule is shared;future teacher commands can differ with the resulting pixels.

Use process copies,not ALE snapshot restoration,to preserve sticky-action
memory,emulator RNG,wrapper frame buffers and the teacher's observation/action
history. The pending origin teacher proposal must receive actual-action
feedback. Verify source emulator and teacher state remain unchanged. Every
origin also gets a recorded-action branch whose input pixels,rewards,frame
counts and terminal flags match the retained trajectory through64 actions or
the next natural boundary. Reverse alternative execution order at the first
origin of each lane and require identical outcomes. These declared CPU
validation branches are separate from the action comparisons.

Report life-loss and accumulated extrinsic-reward differences separately,
including ties,for all origins,teacher disagreements,the two confidence groups
and each lane. Retain full branch trajectories and all counts. A short horizon
may leave many actions indistinguishable. Life loss is read from ALE only for
measurement;the teacher still receives pixels and executed actions only.
These are local interventions under teacher continuation,not complete policy
evaluations,optimal-action labels or a five-game success claim. Correlated
origins on two reused control-policy lanes limit inference.

Require the retained source/corpus/probability/teacher/wrapper/ALE/ROM hashes,
normalized full18 probabilities,exact source stream replay,finite outputs,
unchanged pins,no native learner import and no GPU descriptors. CPU analysis
cap180s,service cap210s,per-branch cap5s;one low-priority CPU core. Preserve
partial outputs and failures;do not repeat completed stages to fix reporting.
No training,model-forward pass,temperature change,new loss or automatic learner
is selected. Root`/x/Temp/kindle-breakout-action-interventions-20261101`.

## Completed

The CPU run completes01:59:51 UTC November1/source544c51e in17.081s,with136
origins,234 distinct action branches and both reverse-order checks passing.
All5,240 source actions/20,950 frames and every recorded continuation match.
All25 pins and32 focused CPU tests pass;the initial test-path failure and first
passing preflight are retained. Terminal collection completes02:01:35 with
completion`3f986ce3e6d7850c14a204ece61c3a6c67677b612449acbdbf185db4ca7a58cb`.
No native/model inference,learning or GPU work occurs.

Among51 improved-policy disagreements,the teacher saves a life in5 cases,the
policy in3,and43 tie. Among18 confident disagreements,the policy saves2,the
teacher1,and15 tie;reward favors policy4,teacher1,and13 tie. These local
teacher-continuation outcomes limit treating disagreement as error;they do not
establish long-run policy superiority or adequate representation. See the
[result](../results/2026-10-30-cdp-encoder-representation.md#local-action-interventions-limit-the-teacher-disagreement-diagnosis).
The service is terminal success,no GPU allocation is active,and no learner or
automatic successor is selected by this diagnostic. The five-game goal remains
unfinished.
