# Full-budget Pong learning with replay batch16

**Completed.** [Result and comparison](../results/2026-10-25-cdp-cheap-pong.md#b16-at-the-fixed500k-budget):
500k/62,454 updates in38m45s,22.37% less training time than B8; frozen11.875
versus14.958, both24/24 wins and below mean15. All audits/comparison pass;
service exits10:28:51 UTC. No extension or confirmation selected.

Selected after the completed bounded B16 check. Its16,384-action window
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

Started **09:48:13 UTC**, source`8a966a2`, on the same RX7900XT/RADV host.
Startup verifies3,578,225 parameters, zero starting counters, the complete
configuration and first updates376:1,384:2,392:3. All346 initial tensors are
finite and bitwise equal to the completed coefficient.01 B8 control. Initial
metadata SHA256`6583de533730f11537c05f06b6f8d35a5277619013f6d2fcf04a4169d1fcf974`;
selected declaration`6f7addc8ef8f9289da23caacfceea70d78dffbdac91d87361ffc6c1a9663a46b`.
The100k recovery audit passes at12,454 updates with346 finite tensors, complete
configuration/counters and13,111 strict numeric reports through105,264 actions.
Captured102,400-action progress takes471.056s with zero training debt.
Recovery metadata`7ed837fe29c74debd75a871aebae2b48f4c56c39b1399a9b5afc2ce491970b0b`.

Completed-run curves and B8/B16 comparison are automated after terminal success
and full audits. The comparison checks fixed budgets, distinct update counts,
unchanged gate, full config/source identity, matched initial tensors and frozen
initial semantic traces. CPU validation passes against the completed B8 control;
it does not manufacture a B16 result while training remains active. Comparison
script SHA256`07064b67117a2c862c2798ed99394fa4f571d90af65680487ac34b53bac7c269`;
waiter`4b3b22acfec75d1fb4433af7a145c15e3df3e6bdb3c4f8477f1e949a9900363b`.
