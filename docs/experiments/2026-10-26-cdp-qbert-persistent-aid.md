# Qbert with bounded persistent exploration

**Selected after the completed Pong rate comparison.** Doubled rates lower
Pong's fixed-final score11.875→9.125 at essentially unchanged38m32s training cost;
all audits and completed analysis pass. No Pong extension or confirmation is
selected. Advance this separate fresh Qbert200k test under its25-minute cap,
with the original rates and one learner. There is no automatic next allocation.

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

Completed-run automation is prepared: an explicitly labelled aided curve,
generic cube coverage and matched-config unaided-control comparison. The reader
validates the completed control's49,939 updates, gates, guard/tensor/coverage
receipts and log hashes. Its actual initial trace reads successfully across all
1,203 transition/episode/reset/final-counter events; equality to the candidate
remains pending. Candidate analysis requires completed training/frozen audits,
100k recovery and a terminal service. No analysis process or GPU work starts.
Comparison SHA256`5378b754064a2a0bfa33dab771e426040592fd423fd27c193e252dcbe52a29e0`;
postprocessor`0281757bb16c9c4494d4f4843c4962004c2a7d6393b09fe1b4bd996778537e8d`.
Pins and CPU checks are retained in`analysis-preparation.json` and
`initial-trace-preparation.json` under the candidate artifacts.

Started **12:03:20 UTC**, source`1c31c4c`, native`bfb8c31`, CLI`22bd61b`.
Startup verifies the full native config against the unaided control,3,578,225
parameters, zero counters and first updates248:1/256:2/256:3. All346 finite
initial tensors match that control. The first512 vector ticks/4,096 aggregate
actions reproduce the declared exploration RNG and every applied override;
184 actions are overridden in that prefix. Initial metadata SHA256
`a51927112c98eaafa73f7462f2f4d30bd0128ea6ef0e95962cbd529a1a6f0b0b`;
selected declaration`a9f48c9a81c75b2d7ae7f8dfa9346a5cf0ff059508ba4e0b20914a96b4eb0476`.
The100k recovery watcher and terminal-service curve/coverage/comparison automation
are running. Unaided frozen learning results remain pending.

The100k recovery audit passes at24,939 updates,346 finite tensors and25,161
strict numeric reports through100,896 actions. Captured98,304-action progress
is595.749s with zero debt. Recovery metadata SHA256
`0fe0ed9cb3bcdd9a16f2e29e2e701c5f1303a6e059ffd14d1b5c411d98ffd8a9`.
The watcher exits successfully; fixed-final unaided evaluation remains pending.
