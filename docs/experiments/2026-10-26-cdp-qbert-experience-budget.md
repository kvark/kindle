# More Qbert experience at lower replay ratio

**Completed:**500k actions/62,470 updates in26m53s; frozen719.792 versus own
initial180.208,0/24 pyramids and14.125/21 coverage. The200k/R32 control scores
809.375 with16.625/21 coverage. All audits/comparison pass; service exits09:33:03.
See the [result](../results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result).
No extension; the bounded PongB16 check is selected next.

Selected after the completed four-action Breakout result. That trial finishes
in20m59s with frozen9.542 and0/24 two-wall successes; all audits/comparison pass
and its service exits09:01:55 UTC. Move this allocation to Qbert with the
previously prepared fixed recipe and limits. No automatic next queue.

The full18 coefficient1 compact control scores809.375 with0/24 first pyramids
after200k actions/49,939 updates in19m45s. Its mean coverage reaches16.625/21;
stronger bonus and six actions worsen coverage. Test more experience while
reducing replay work per action, retaining the better full18 coefficient1 recipe.
The [stage-cost estimate](../results/2026-10-25-cdp-cheap-qbert.md#cost-estimate-for-more-experience-at-lower-replay-ratio)
suggests26.38 minutes, excluding construction. That is planning arithmetic, not
a measured runtime or learning prediction.

Fresh1009/eval4000001009, centered CDP,4M/RGB48/Auto,3,578,225 parameters,
N8/B8/T16/H15, full BPTT/microbatch8, replay100k, intrinsic coefficient1 and
actor_unimix0. Full18/sticky.25/repeat4, zero no-op starts and100000-frame episode
limit remain. Native max-pooled frames receive one GPU resize. No external aid,
RAM input, reward shaping, optimizer change, native rebuild or concurrent learner.

Change native train_ratio32→16 and training budget200k→500k together. Expected
first update248 and62,470 total updates, versus49,939 in the control:2.5 times
the experience and1.251 times the updates. This does not isolate replay ratio.
Save the own zero-experience checkpoint and match all346 initial tensors against
the full18 coefficient1 control. Replay, games, belief and RNG start fresh;
this is a reused development seed, not a warm start or held-out test.

Keep the1800s/30-minute training cap, including construction. Save250k recovery
(31,220 expected updates), then fixed500k final. Audit recovery before replacement;
no best-checkpoint selection, cap increase, automatic resume or extension. Retain
any cutoff explicitly. A completed target requires500k actions and62,470 updates.

Evaluate fixed final and own actual initial for the first three natural sampled
episodes per lane,24 per arm,600k-action/300s caps, zero updates/truncations.
Retain excess episodes and partial tails. Keep at least20 episodes, mean≥15,000
and at least90% first-pyramid success. Coverage remains diagnostic, not a lowered
gate. Audit every numeric report, full source/config/counters, finite checkpoints,
all346 unchanged frozen tensors, full CPU replay/video and host guards. Audit
cap1200s, service cap3660s. Stop on real faults or numerical failure.

Native`bfb8c31`, runner`1ed07af`, CPU replay`e853818` remain unchanged. The older
control's runner predates optional actor-mixture/horizon flags; both recipes
retain0/H15. Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r16-500k`;
controller:`run_qbert_500k_r16.py`; intended service:
`kindle-cheap-qbert-r16-500k-20261026`.

CPU preparation normalizes the complete queue back to the original control:
only ratio, budget, save cadence, expected counts and output paths differ.
Unselected execution refuses before GPU/log artifacts. Selection must pin a
preceding review and verify its service is terminal. The R16 recovery checker
passes against the actual completed Pong200k/R16 log/checkpoint:24,970 updates,
200k actions and346 finite tensors, with original files unchanged. Invalid action
counts and a wrong checkpoint counter are rejected. This is monitoring evidence,
not new Qbert execution. Monitoring check SHA256
`705a2897da7d0bccf422b0d44a9e68a0a10ee428bd304ebd74b74764beab963c`.

Completed-run analysis is prepared in `finish_qbert_experience_budget.py` and
`compare_qbert_experience_budget.py`, alongside the existing generic pyramid
coverage reader. It requires the250k recovery audit, completed500k/62,470-update
review, three host guards, unchanged frozen tensors and complete replays before
plotting. The comparison retains both actual initials, the original task gate,
and separate action/time axes; experience and ratio change together. Compilation
passes and unselected postprocessing refuses before creating artifacts. These
scripts perform CPU analysis only after completed training/evaluation audits.
Analysis preparation: `analysis-preparation.json`; comparison SHA256
`faf4cb4880a48a8ae7117f141d7e39a950477569273afe8d06d1b795d9d3a646`;
postprocessor`7d1f69b6cdcfd429c321c7a7983c4c9232119467b1e3450df7e8a72f4b4eedd0`.

Started09:05:27 UTC, source`640f614`. Startup verifies the full native config
differs only in train_ratio32→16,3,578,225 parameters, zero counters,346 finite
initial tensors matching the full18 control and updates248:1,256:2,264:3.
Initial metadata SHA256
`6ddb2a18dc72b4b74c836bfe8121a8a1410e64bb5320c0008b4aae6dba0c3210`;
selected declaration`76a6e972db87ee93bb1739edb7b8bba3944902060f8b2e6970879125a2bcc6d6`.
Selection pins the completed Breakout review/comparison and terminal service;
no concurrent learner was active. The250k recovery monitor and completed-run
analysis finish successfully. Fixed final/initial evaluations, full audit,
plots and coverage analysis complete; the original learning gate fails.
