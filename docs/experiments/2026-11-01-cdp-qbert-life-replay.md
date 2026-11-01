# CPU replay of Qbert life losses and cube-color histories

The retained encoder4e-5 Qbert policy reaches greedy14,028.125,versus13,785.417
at2e-5;both clear24/24 first pyramids and still miss15,000. The stronger pixel
teacher's retained22,050 score shows that this is not simply its score ceiling,
but those trajectories differ and do not establish a causal teacher advantage
on learner states. Before selecting another loss or training change,measure
where lives are lost along the two policies' saved greedy trajectories.

Select one CPU-only replay of each complete frozen greedy log: control2e-5
44,512 actions and candidate4e-5 60,648 actions. Execute every recorded command,
including excess episodes and partial tails. Score/aggregate only the existing
first3 natural episodes per8 lanes=24 per policy. Preserve seeds,ROM,emulator,
wrapper,sticky actions,action repeat,resets and full18 commands. Require exact
original rewards,frame counts,boundaries,episodes,task outcomes and partial tails.
Neither a native model nor teacher is constructed; no new decisions,training,
counterfactual branches or GPU context occurs.

At every raw frame,observe ALE lives and the existing21 Qbert cube-color bytes.
Record all life decreases and increases independently: a bonus then loss must
not disappear because the net count is unchanged within an action. Preserve
raw color-change/life-event histories and per-action input/output pixel hashes,
actions,rewards,lives,frame ages and first-pyramid milestones. RAM observations
remain post-hoc outputs,never policy inputs or labels. Color changes include
animation; no generalized level detector or new completion criterion is claimed.
Ordinary frozen logs lack live pixel hashes; reconstruction hashes cannot prove
direct equality to unavailable live images.

Report per-episode initial/final lives and loss/gain balance,selected duration,
score,and life events before/after the already qualified first-pyramid milestone.
Also split losses below/at-or-above episode action age1920,only as descriptive
timing. That split does not measure matching training-state exposure or identify
the effect of the training guidance cap. Keep per-lane and complete trajectories
available; different policy states and reused development cohorts prevent a
matched-state or reliability claim. This diagnostic selects no new learner,
teacher change,loss or automatic successor.

Use one low-priority CPU core,300s total analysis cap and315s enclosing timeout.
Five pure tests cover bonus/loss separation,unchanged/color-change frames,
invalid frame order/geometry,milestone/age classification and duplicate episodes.
They pass in.028s. CPU preflight3.036s verifies291 inputs,the complete source
reviews/replay manifests,the fixed selected scores and all265 active Breakout
inputs,without loading an emulator or native agent. Preserve original Breakout
PID742893/invocationeccf5a76eed6458babd75b872c1704e4 before and after; its GPU
learner continues. Retain all CPU co-host timings without wall-time correction.
No retry or budget increase follows a failure.

Root`/x/Temp/kindle-qbert-life-replay-20261101`. Source trials remain under
`/x/Temp/kindle-qbert-encoder-moderate-20261030/full` and
`/x/Temp/kindle-qbert-encoder4e5-20261101/full`. The five-game goal is unfinished.
