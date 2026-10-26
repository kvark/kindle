# Frozen AMD check of the retained Breakout policy

The shared handover contains the learned B8/T16 Breakout2M weights. Before
allocating another learning test, measure that policy with the current AMD Auto
runtime. This is one bounded frozen check, with zero training actions or updates.
The earlier receiving-host restore checks used the context64 smoke; they did not
establish the retained2M policy's frozen competence on this machine.

Source: `/x/Temp/kindle-amd-20261025/handover/breakout-b8/2m-frozen`, recovered from
`/mnt/data/kindle-handover-20261025`. All four files match the shared manifest.
Preserve export counters499939 learner updates and2027152 environment actions;
the handover verified all346 tensors against the original2M checkpoint. The
source's actual-initial and500k checkpoints were not transferred.

Use a separate copy, changing only saved `training_precision` from `bf16` to
`auto`. Keep all346 tensors, optimizer/normalizer state, seed1009, Size12M/RGB64,
B8/T16/H15/R32, centered CDP/action-effects1 and actor_unimix0 unchanged. The CPU
copy audit passes: every tensor is finite and bitwise identical, original file
hashes remain unchanged, and the sole metadata difference is precision.
No agent was constructed during that preparation.

Frozen evaluation uses seed4000001009, eight independent lanes, full18 actions,
sticky.25/repeat4, native max-pooled frames and one GPU resize. Select the first
three natural sampled episodes per lane,24 total. Retain excess episodes, tails
and cutoffs. Use600k aggregate actions and300 seconds including construction as
caps; require zero updates and zero truncations for a completed check. There is
no action aid, mixture adjustment or shorter imagination rollout in this probe.

Audit source identity before and after, full config/protocol/counters, finite
checkpoints, all346 copied/frozen tensor equalities, full CPU action replay/video
and the host guard. Audit cap600 seconds; service cap960 seconds. Stop on a real
GPU fault or numerical failure. No retry, cap increase or automatic warm-start
training follows the result.

The historical source score98.2083 used another GPU, precision and runtime.
Report the current score separately; any difference cannot isolate precision.
This measures transferred competence, not new learning, an independent learner
seed or resumed uninterrupted experience. Any later warm start needs its own
declaration and would start with empty replay, games, belief and RNG.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-retained-2m-auto-frozen`.
Controller: `run_retained_breakout_auto_frozen.py`; intended service:
`kindle-retained-breakout-auto-frozen-20261026`.
Native source`bfb8c31`, runner`1ed07af`; source at preparation`259f32f`.
Native SHA256`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Source metadata SHA256`9a37c5d919bc68f58535ec0ce48397e9bd68b22f8c350c40357a8b267ddb3574`;
copied metadata SHA256`b7c2482902d7be7a10fc8277716d18674880d8a90d7ae9dd92f869eb273b9920`.

Prepared, not started. Review the active Freeway confirmation before selecting
this short frozen check. The Qbert coefficient100 learning candidate remains
prepared; no concurrent learner or automatic queue is created.
