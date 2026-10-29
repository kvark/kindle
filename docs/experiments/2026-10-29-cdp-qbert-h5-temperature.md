# Frozen action temperature on the stronger H5/R32 Qbert learner

The completed H5/R32 learner scores14,050 greedy and13,331.250 sampled, each
with23/24 first pyramids, after200k actions/24,954 updates/40m24s. Both miss
mean15,000. The earlier H15/R16 temperature0.5 assessment gives a small gain
over greedy, motivating one fixed inference-only check on these newer weights.
This is separately selected after the H5/R32 terminal audits; its declared
greedy primary and sampled result remain unchanged.

Select exactly four frozen arms: retained final and own actual initial, each
with default sampling or action temperature0.5. Primary comparison is the
temperature0.5 final versus retained greedy14,050/23; retain default sampled
13,331.250/23 separately. Do not scan temperatures, choose checkpoints or switch
the comparison after results. The mean15,000/90%-first-pyramid gate stays fixed.

Use the qualified native`e416696`/SHA40872a3 from
`/x/Temp/kindle-action-temperature-20261029/runtime`. Its probability/RNG/reset,
learning-rejection, default-parity and prior full frozen assessments are already
complete. No native build, production change or repeated qualification is needed.
The first final-default arm must reproduce the entire retained sampled trajectory
and score before final-temperature runs; initial-default must likewise match its
retained trajectory before initial-temperature. Audit each arm before the next.

Preserve checkpoint config12M48/Auto/N8/B16/T16/H5/R32/expert100, full18,
sticky.25/repeat4, sampled posterior, native max-pooled frames and one GPU resize.
Seed4000004013; first3 natural episodes per8 lanes=24 per arm. Guidance and
labels are off. Keep partial/excess episodes; require zero updates/truncations,
exact config/source/counters,357 finite/invariant tensors, complete frozen
outcome replays/videos and host guards. Ordinary source logs lack live pixel
hashes. Restores omit temperature by default; only the two temperature arms
explicitly select0.5. Temperature changes action sampling only.

Caps:600k actions/300s per GPU arm,180s per CPU replay,1800s whole service.
Stop on real faults or numerical failures; no driver recovery,retry,cap increase,
extension,temperature sweep or automatic successor. This reuses learner4013 and
the same evaluation cohort, so it is development inference tuning, not a new
learner or independent seed confirmation. Original training cost remains40m24s.

Source learner:
`/x/Temp/kindle-qbert-h5-r32-20261029/full/qbert-12m-48-b16-h5-r32-expert100-200k-seed4013`.
New evidence root:`/x/Temp/kindle-qbert-h5-temperature-20261029`;service
`kindle-qbert-h5-temperature-20261029`. The host is idle at selection. Recheck
live state before launch and pin a fixed prelaunch protocol copy.

CPU preflight passes in0.737s:four parsed commands,three compiled scripts,
all229 source-learner pins and357 finite tensors in each source checkpoint.
Retained default final/initial semantic counts are6722/1138. No environment,
native agent or learner update is constructed. Packaging, prior temperature
assessments and completed H5/R32 evidence are checked before declaration.
