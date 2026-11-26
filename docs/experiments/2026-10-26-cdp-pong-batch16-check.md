# Bounded Pong B16 learning and restore check

**Completed:**16,384 actions/2,002 updates in72.224s,23.21% less early wall time
than B8; full configuration/initial/update/ordinary-restore checks pass. Frozen
1,024 actions leave346 tensors unchanged; no natural-match competence is tested.
Service exits09:38:03 UTC. See the [result](../results/2026-10-25-cdp-cheap-pong.md#b16-bounded-check-passes-with-lower-early-wall-cost).
One fresh full-budget B16 learning test is selected after review.

Selected after completed Qbert500k/R16. That trial scores719.792 with0/24
pyramids and14.125/21 coverage, below the200k/R32 control; all audits/comparison
pass and its service exits09:33:03 UTC. No Qbert extension, full Pong run, backend
rebuild or automatic next allocation is selected.

The [source feasibility review](../results/2026-10-25-cdp-cheap-pong.md#larger-replay-batch-source-feasibility-only)
finds that doubling replay batch size roughly halves optimizer steps at fixed
ratio32, while increasing work per update. Early cost and ordinary restore must
be measured before considering a full learning test. This bounded check uses
the existing native backend and changes only batch_size/world_microbatch_size
8→16 in the completed coefficient.01 primary's native configuration.

Fresh1009, centered CDP,4M/RGB48/Auto, intrinsic coefficient.01, actor_unimix0,
N8/B16/T16/H15/R32, full BPTT/microbatch16, replay100k. Keep full18/sticky.25,
repeat4, zero no-op starts,100000-frame episode limit and native max-pooled
frames with one GPU resize. No action aid, RAM input or reward shaping.
The eight acting environments stay fixed; B16 is the replay batch size.

Train16,384 aggregate actions under a240s cap including construction. Expected
first update376,2,002 updates and zero debt. Check the full native config,
3,578,225 parameters, finite checkpoints and all346 initial tensors against the
completed B8 control. Require changes in CNN0 and the first reward projection;
this checks that those weights move, not gradient correctness or useful features.
Audit every numeric report and exact counters.

Then perform an ordinary saved-checkpoint restore for1,024 frozen actions under
90s, with seed4000001009 and zero updates. Verify all346 tensors remain unchanged
and metadata/counters match. This is a bounded restore check, not24 natural
matches or a task-competence evaluation. Both host guards must pass. CPU audit
cap120s; total service cap600s. Stop on actual faults/numerical failure; no retry,
cap increase or automatic full-run promotion.

Compare the same16,384-action early window with the retained B8 log:4,035 versus
2,002 updates and516,480 versus512,512 replay positions. The B8 window takes
94.054s. Record construction separately. Warmup, cadence, centering population
and learned trajectories differ; an early timing gain predicts neither full
budget throughput nor the unchanged mean15/90%-win Pong gate.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-b16-shape-check`;
controller:`run_pong_b16_shape_check.py`; intended service:
`kindle-cheap-pong-b16-check-20261026`. Native`bfb8c31` and runner`1ed07af` remain
fixed. The queue is checked against the existing train/frozen control templates;
compilation passes and unselected execution refuses before GPU/log artifacts.
Prepared declaration SHA256
`30487298e9ce766b4495a23fe79169393f8bc94c7eefc38956b6699857beae08`;
controller`305fb46e2f99888c1803e879aa879c9937006164f782d9b32bb0b7bdfa397098`.
Selection must pin the preceding completed review and verify its terminal service.
