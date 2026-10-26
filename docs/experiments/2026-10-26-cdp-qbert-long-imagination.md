# Qbert H30 at the fixed 200k budget

**Prepared, not selected or running.** Review the active Pong B16/R40 fixed-final
result, comparison and terminal service before selecting further GPU work.
No automatic queue follows from this CPU preparation.

The completed Qbert H15 control scores 809.375 versus initial 180.208, with
0/24 first pyramids and mean maximum coverage 16.625/21. More experience at R16,
larger capacity with B16, stronger intrinsic reward, minimal actions and a 5%
training aid have not produced a successful cheap recipe. Test a different
learning hypothesis: longer imagined trajectories may improve the policy's
evaluation of delayed rewards. They may also compound world-model error.

The [CPU timing analysis](../results/2026-10-25-cdp-cheap-qbert.md#reward-timing-and-a-longer-imagination-candidate)
finds positive rewards within 15/30 future actions at 30.81%/45.43% of the same
91,787 late training origins. H15 already bootstraps critic values beyond 15;
this is not a hard reward horizon. Keep discount horizon 333 and lambda .95.
Recorded trajectories do not predict the outcome of a newly learned policy.

Only native `imagination_length` changes 15→30. Fresh 1009/eval4000001009,
centered CDP4M/RGB48/Auto/N8/B8/T16/R32, 3,578,225 parameters, full BPTT/
microbatch8, replay100k, intrinsic coefficient1, actor mixture0 and original
rates. Full18/sticky.25/repeat4/noops0, native max-pooled frames/one GPU resize.
No action aid, teacher, RAM policy input or reward shaping. RAM remains only
the post-hoc CPU coverage observer.

Fixed 200k aggregate actions, 49,939 updates, first248, 6,392,192 sampled replay
positions, 100k recovery at24,939 and zero debt. Imagined actor rows double from
95,882,880 to191,765,760, so compute is not matched. A linear projection adding
the H15 run's imagination and behavior-training durations gives 1,648.550s
(27m29s), versus measured H15 19m45s. This is not measured H30 speed. Set a
35-minute training cap including construction, versus the control's30-minute
cap; retain any cutoff without increasing the cap or resuming.

All346 initial tensors must match the H15 control. Evaluate fixed final and own
initial on the first three natural episodes per lane,24 per arm, under600k-action/
300s caps. Keep15k/90%-first-pyramid, at least20 completed episodes, zero updates/
truncations, excess episodes and partial tails. Audit all numeric reports,
config/source/counters, finite checkpoints, frozen tensor invariance, full CPU
replays/videos and three guards. Compare curves, coverage and initial semantic
traces. Audit1200s/service3960s; no earlier-checkpoint substitution, extension,
broad campaign or automatic next allocation. One reused development seed.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-h30-r32-200k`.
Controller `run_qbert_h30_200k.py`, SHA256
`390ed680e7316c7249e6f0dec623b0c2dc217422af3377b06ed138a6bcac7bf1`;
queue `061d476d2083b355dc4cca866ffd4c931a9281d2dd629961336f8a69441ac32c`;
prepared declaration `8b922f4ec3c96cd14c58db97f43e3ed177e3da9a359bcee364166f197856645a`.
Comparison `773b3a2ad1e8dacfe9de5497001d9759d5f9edcb933ee33143bdbcb9f5d66040`;
postprocessor `be5e17d0069e362017c2da4e6a10fb6b2ee2aeda605afa06be9537098a4f885d`.
Compilation, exact queue normalization, verified replay manifest, unselected
controller/postprocessor refusal and completed-control reader pass. No native
or production CLI edit, GPU shape qualification or process start is required
by this preparation. Runtime remains native `bfb8c31`, CLI `22bd61b`, replay
`e853818`; prepared source `f4ad875`.
