# Qbert with bounded persistent exploration

**Prepared, not selected.** PongB16's doubled-rate500k test remains active.
Review its fixed-final evaluation and completed comparison before choosing a
next allocation. This controller refuses an unselected launch; preparation
starts no GPU job and creates no automatic queue.

Qbert12M48/B16 costs63.70% more than4M48/B8 at200k yet still completes no pyramids,
with slightly lower cube coverage. Its saved-action replay shows that no selected
episode ever lights all21 cubes. [CPU random-action controls](../results/2026-10-25-cdp-cheap-qbert.md#persistent-random-action-controls)
show greater per-episode coverage for16-step holds than1-step holds, including
occasional visits to cubes the learned policy never lit. Episodes also last longer;
these controls do not prove a benefit from a sparse training aid.

The candidate adds the existing5% persistent-uniform16-action block aid to the
completed4M/RGB48/B8/R32/200k recipe. A block chooses a uniformly random full18
action with probability.05 and holds it for16 agent actions; otherwise the
native policy acts normally. Blocks reset at episode boundaries. Observations
and learning keep their cadence, and the applied-action ledger is audited.
The aid is disabled for both frozen evaluations. No RAM policy input, teacher,
external reward shaping or native/backend change is introduced.

Fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B8/T16/H15/R32,
3,578,225 parameters, full BPTT/microbatch8, replay100k, coefficient1,
actor_unimix0, full18/sticky.25/repeat4/noops0 and100000-frame episode limit.
Native config, rates, model, image geometry and update schedule match the
unaided control. Native max-pooled frames receive one GPU resize. CLI`22bd61b`
adds optional rate controls but this test keeps the original defaults;
native`bfb8c31` and replay`e853818` are unchanged.

Keep200k aggregate actions/49,939 updates, first248 and6,392,192 sampled replay
positions. Save100k recovery at24,939 updates and the fixed200k final. Set a
25-minute training cap including construction; the completed unaided control
took19m45s, which is context rather than a guarantee. This is a fresh life with
own actual initial, which must match all346 control tensors. Reused learner and
evaluation seeds are development data, not independent confirmation.

Evaluate fixed final and actual initial for the first three natural sampled
episodes per lane,24 per arm, under600k-action/300s caps. Keep the original
15000/90%-first-pyramid gate and explicitly require zero updates/truncations.
Retain excess episodes and partial tails. Audit all49,939 numeric reports,
full config/source/counters, finite checkpoints, the complete exploration ledger,
346 unchanged frozen tensors per arm, CPU replays/videos and all three guards.
Coverage and action-override counts are diagnostics, not lowered task targets.
Audit1200s/service3360s. No cap increase, resume, best-checkpoint selection,
extension or automatic next allocation.

Controller compilation, complete queue normalization against the unaided control
and unselected refusal pass. No new shape qualification or native rebuild is
needed. Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-aided-200k`;
controller:`run_qbert_200k_aided.py`, SHA256
`8fc73dfd0fa756c8d14b50cead674bc7ea0103ca668d9b314f5972fd1dedcfe4`;
prepared declaration`880e0b829068946defd481345b9f33b86f166f2a4b939f78e6116acb243f5019`;
queue`d5a946e5ba19ff42deeb2aa7d50a49462bf4cb75f6a212c04bf640b58d741e6a`.
The existing B8/R32 recovery monitor covers100k/24,939; completed-run analysis
must preserve the aid distinction and fixed unaided final/initial controls.
