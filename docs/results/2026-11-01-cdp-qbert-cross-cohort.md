# Qbert's agent ranking persists across cohorts, while the passing margin does not

The [fixed cross-cohort assessment](../experiments/2026-11-01-cdp-qbert-cross-cohort.md)
keeps4013 ahead of8039 on both retained environment cohorts. The gap is3937.500
points on4013's cohort and2994.792 on8039's. Changing cohort alone does not
explain8039's lower score. However,4013 falls from15,515.625 to14,370.833 on
the alternate cohort and no longer passes15k. The original gate outcomes remain
unchanged;this adds no independent learner confirmation.

| Frozen trained agent | Cohort4000004013 | Cohort4000008039 |
|---|---:|---:|
| Development4013 | 15,515.625;24/24 first pyramids;passes | 14,370.833;24/24;fails score gate |
| Fresh8039 | 11,578.125;24/24;fails score gate | 11,376.042;23/24;fails score gate |

| Actual-initial agent | Cohort4000004013 | Cohort4000008039 |
|---|---:|---:|
| 4013 | 80.208;0/24 first pyramids | 46.875;0/24 |
| 8039 | 296.875;0/24 | 251.042;0/24 |

| Initial-subtracted mean | Cohort4000004013 | Cohort4000008039 |
|---|---:|---:|
| 4013 | 15,435.417 | 14,323.958 |
| 8039 | 11,281.250 | 11,125.000 |

The matched-cohort differences remain4154.167/3198.958 after subtracting each
agent's own initial result. This supports a substantial frozen-agent difference
alongside evaluation variation. It does not isolate learned weights from each
checkpoint's config or posterior randomness,and two reused cohorts do not
establish a population-wide ranking. Only the original4013/4013 cell clears the
unchanged15k/90%-first-pyramid gate. Do not choose whichever cell passes or
reinterpret these dependent cells as a new independent success rate.

Four new greedy arms assess each final and actual-initial checkpoint on the
other cohort;the four completed diagonal arms are reused without GPU repeats.
Native447fe50,runneraba4300d,checkpoint configs,sampled posteriors and all
preprocessing remain unchanged. Guidance and labels are off. Original training
times27m33s/27m34s remain;this assessment trains no learner and makeszero updates.

All413 pins,357 finite invariant tensors per arm,source identities,first-three
natural episodes per lane,full outcome replays/videos and host guards pass.
New arms total131,080 actions /524,174 frames /97.911s,zero updates or
truncations. Excess episodes and partial tails remain retained. Ordinary frozen
logs lack live pixel hashes;the replay checks recorded outcomes and trajectories.

Main succeeds08:28:02 UTC November1 after3m07s;terminal collection succeeds
08:28:16,source63fce09. OriginalPID770056/invocation
`d56c32f594864b018418ed008250835f`;collectorPID770059/invocation
`e17b21c8b78648fb8aeccf316d7cc5df`. The collector verifies the unique original
start/end and child invocation,all input and output hashes,and four new videos.
No assessment stage fails or repeats. The earlier CPU qualification failures
(missing newer CLI fields and forbidding an identity-only ALE import) remain
preserved;the corrected checks and production arm audits pass. Retain their
earlier Breakout co-host timing without correction.

Root `/x/Temp/kindle-qbert-cross-cohort-20261101`. Both services are terminal and
the host is idle at closure. Do not rerun the assessment,diagonal arms,collector
or successful preflights. No new learner,extension,cap increase,mode sweep or
automatic successor is selected. The five-game goal remains unfinished;Qbert
needs stronger and more reliable learning within the compute budget.

| Artifact | SHA256 |
|---|---|
| Declaration | `7b10b2e56c61e43ce36be65e22726299299e412f59f74eac6042c75acc33289b` |
| Comparison | `55f96454ec9b1c428bd6e51158ba80c203c6175c95e5f81b2b4bd68b14a7f3a4` |
| Completion | `2e2826188370a0c70d81dca1e8d38edee20e0a3d2fc694d5210e906793b0cb52` |
| Controller qualification | `cf8e57dd94264e00fa69fb1beeffa7e11520e21ef5c41a14ffd30ff165844a30` |
