# Full-budget Pong learning with replay batch16

**Selected after the completed bounded B16 check.** Its16,384-action window
uses72.224s versus94.054s for B8,23.21% less wall time. Full configuration,
matched initial tensors, first376/2,002 updates, selected parameter changes,
ordinary restore and346-tensor invariance pass; service exits09:38:03 UTC.
This motivates a full learning test, not a claimed full-budget speedup or task pass.

Fresh1009/eval4000001009, centered CDP, intrinsic coefficient.01,4M/RGB48/Auto,
3,578,225 parameters, N8/B16/T16/H15/R32, full BPTT/microbatch16, replay100k,
actor_unimix0, full18/sticky.25/repeat4, zero no-op starts and100000-frame episode
limit. Native max-pooled frames receive one GPU resize. No action aid, RAM input,
reward shaping, optimizer rescaling or native rebuild. Eight acting environments
remain fixed; B16 is the replay batch size.

Relative to the completed coefficient.01 B8 primary, only native batch_size and
world_microbatch_size change8→16. Keep500k aggregate actions. Expected first
update376 and62,454 updates versus248/124,939 in B8. Sampled positions are
15,988,224 versus15,992,192 because of different replay warmup. Optimizer cadence,
centering population and the per-action duration of the fixed1,000-update
learning-rate warmup change. Do not describe this as identical optimization.

Use a2700s/45-minute training cap including construction, reduced from the B8
control's55-minute cap. The early-window gain does not guarantee completion.
Save100k/200k/300k/400k recoveries at12,454/24,954/37,454/49,954 updates, then
fixed500k final. Retain any cutoff explicitly; no cap increase, selected
checkpoint, automatic resume, extension or next queue.

Save own actual initial and match all346 tensors against the B8 coefficient.01
control. Start fresh rather than restoring the short check: replay, games, belief
and RNG are new. Both seeds are reused development data, not held-out tests.
The16,384-action validation cost is separate from this performance trial.

Evaluate fixed final and own actual initial for the first three natural sampled
matches per lane,24 per arm, under600k-action/300s caps. Zero updates/truncations;
retain excess matches and partial tails. Keep at least20 natural matches,
mean≥15 and at least90% wins. Audit every numeric report, full config/source,
exact counters, finite checkpoints, all346 unchanged frozen tensors, complete
CPU replays/videos and host guards. Audit1200s; service4560s.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-intrinsic001-500k`;
controller:`run_pong_b16_intrinsic001_500k.py`; intended service:
`kindle-cheap-pong-b16-intrinsic001-500k-20261026`. Native`bfb8c31`, runner`1ed07af`
and replay`e853818` remain fixed. Queue normalization/compilation and unselected
refusal pass. The new first376 recovery monitor passes against the actual
completed B16 checkpoint/log, rejects invalid and wrong counts, and leaves
source files unchanged. Monitor review SHA256
`1e7bd24cedf161168b467cd14691e803643506044589cb681041680c1ef27142`.
Selection pins the completed check and terminal service; no concurrent learner.
