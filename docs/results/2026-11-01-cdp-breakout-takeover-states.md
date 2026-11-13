# Breakout takeover losses are not confined to short estimated contact times

The saved delayed-takeover states do not isolate a simple last-moment correction
problem. Of the21 windows rescued at delay8 but not16,onlytwo have a projected
contact within the teacher's five-action planning horizon at delay16. Nine have
a projected contact beyond five actions,seven are visible but unprojected,and
three have no detected ball. Fourteen of21 recorded commands already match the
teacher at that origin. These observations do not establish correct actions or
true contact times;they limit what can be inferred from immediate agreement and
the teacher's own timing estimate.

All120 original loss windows remain. The8→16 comparison still has84 both-rescued,
21 earlier-only,seven later-only andeight neither. No new interventions,gameplay,
model evaluations or learning occur. The teacher can fail locally,and later
takeover can work better;failure is not proof of optimal irrecoverability.

| Teacher-estimated state at takeover | Delay8: rescued / cases | Delay16: rescued / cases |
|---|---:|---:|
| Projected contact ≤2 actions |0/0|4/5|
| Projected contact >2 and ≤5 |14/15|5/7|
| Projected contact >5 |33/34|42/53|
| Visible, no projection |53/62|36/44|
| Ball unseen |5/9|4/9|
| Already lost before takeover |0/0|0/2|
| All |105/120|91/120|

The teacher-defined near-descending subset at delay16 rescues26/30,far-descending
25/35,ascending33/38,velocity-unavailable3/6 andball-unseen4/9. These are selected,
correlated failure windows with different remaining trajectories,not independent
failure probabilities or a comparison of equally difficult states. The projected
time is an imperfect model output;unprojected states must not be read as distant
contacts. The table does not select a near-paddle loss,teacher change or model
architecture. Sustained trajectory control remains unresolved.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-takeover-states.md)
joins all480 delay2/4/8/16 requests to37,016 saved replay rows. All478 available
origins match phase,debug,input-pixel hash,recorded action,teacher label and life
counts exactly;two unavailable origins remain explicit. The prior110/106/105/91
rescue totals and all paired counts reproduce. Row-level projected bins,phase,
exact/horizontal agreement,target gap and remaining recorded duration are saved.
An independent recount derives survival directly from all478 saved teacher
branches and supplies phase-level counts/denominators. No branch is repeated.

Selection1.488s (1.507s wrapper),analysis **2.766s** (2.816s wrapper) and independent
recount2.555s (2.581s wrapper) pass. Analysis completes **15:23:52 UTC November1**,
source`0e77ede`;recount15:24:44. Eight boundary fixtures,18 action-map cases,
nine declared inputs,all524 active Qbert inputs and originalPID809245/invocation
`361f29163c1e4b2896fd352f2b905ca8` pass unchanged. Single-core low-priority CPU,
stdlib only,zero emulator/native imports,GPU work,updates,failures or retries.
Retain CPU co-host time without correcting Qbert's training time. The existing
Qbert behavior-rate learner continues;the five-game goal remains unfinished.

Evidence`/x/Temp/kindle-breakout-takeover-states-20261101`. Complete;do not repeat
the join,recount or original interventions. No new learner or successor selected.

| Artifact | SHA256 |
|---|---|
| Declaration |`76f658fa7a3f401f09439f0f5182f222be6e9da4dc8413e2ff2a2aae59031f15`|
| Review |`73dbed0b0548565a0d0c286be81bc8a477cd1e2445d1e8badb43c96d7958be17`|
| Joined states |`e061851a2610a937168f9a7f03251c503c98ee2bfd81230f5e76879ce07f0c88`|
| Completion |`a6c86abbd0b94c64b714d64a87ef3d7e52960bb84e96cd34b8f92f069bfcf71c`|
| Independent recount |`49df29d5b96d3be098cca93f0331b978b498780558edfdf29c2cbe32b0454abe`|
