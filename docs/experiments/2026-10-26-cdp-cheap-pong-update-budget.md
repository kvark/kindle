# Test the historical Pong update budget at intermediate capacity

The4M/RGB48/R16 screen scores−17.125 versus initial−20.375 after200k actions and
24,970 updates, with0/24 wins. Historical successful Pong learners used500k
actions and124,939 updates. Test that update budget with the intermediate model
and image size already exercised on this AMD host. This is a fresh recipe test
with a different replay ratio, not a continuation of the completed screen.

Keep fresh development seed1009, centered CDP/action-effects1, Size4M/RGB48/Auto,
3,578,225 parameters, N8/B8/T16/H15, full BPTT/microbatch8, replay100k and no aid.
Native actor_unimix remains0. Full18/sticky.25/repeat4, native max-pooled frames
and one GPU resize. Retain48 to preserve ball contrast. Change replay ratio16→32
and budget200k→500k together. Native source`bfb8c31` is unchanged; CLI`bcef410`
retains the same default mixture. Verify all346 actual initial tensors against
the completed screen and audit the full configuration difference.

The completed screen spends553.604s learning and68.095s elsewhere. Scaling those
measured costs to the new update/action counts projects **2,940s (49 minutes)**.
This is allocation planning, not an observed runtime or learning prediction.
Declare a **new3,300-second (55-minute) training cap** before this fresh test,
including construction. The active Freeway cap remains1,800 seconds. Historical
successful Pong runs took about37 minutes on different GPU/precision/model
settings; no speedup claim follows from reducing parameter count alone.

Fixed500k aggregate actions, expected124,939 updates, first update248, zero debt.
Save actual initial and overwrite one recovery checkpoint each100k actions,
ending with fixed500k final. On cutoff retain the last checkpoint and explicit
failure; no cap increase, automatic resume or selected-checkpoint substitution.

Frozen seed4000001009, first three natural sampled matches per lane,24 per
final/actual-initial arm, zero updates and no aid. Each arm has600k-action/
300-second caps; retain excess matches, tails and cutoffs. Existing gate:
minimum20 natural matches, mean≥15,≥90% wins, zero truncations. Audit every
numeric report, complete config/source/counters, finite checkpoints, all346
unchanged frozen tensors, full CPU replays/videos and three host guards.
Audit cap1200 seconds; service cap5160 seconds.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-500k-r32`.
Controller: `run_pong_500k_r32.py`; intended service:
`kindle-cheap-pong-48-r32-500k-20261026`.
Commands, caps, source hashes and cost projection were prepared before allocation.
Seed1009 is reused development data, not held out. A useful result needs
focused seed confirmation; neither parameter count nor a single seed establishes
reliability or an isolated speedup.

Started October26 at02:01:31 UTC after reviewing compact Freeway's completed
31.0417/24-of24 result and all audits. Source at launch`4918f78`; native`bfb8c31`,
runner`bcef410`. Startup checks pass:3,578,225 parameters, expected configuration
with only replay ratio changed, no aid, all346 initial tensors equal the200k
control and first updates248:1,256:2,256:3. The sealed declaration records the
preceding Freeway review hash.

Completed:500k actions/1,999,756 emulator frames/124,939 updates in48m53s.
Fixed frozen final11.0 with23/24 wins, versus initial−20.375 and0/24. The existing
mean≥15 gate fails. All numeric, finite/config/source/counter, frozen-tensor,
replay/video and guard checks pass; the service exits successfully. See the
[result and comparison](../results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions).
No extension. The declared Freeway seed confirmation starts after this review.
