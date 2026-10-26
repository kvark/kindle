# Confirm compact Pong winning behavior on one new seed

The4M/RGB48/R32 recipe [wins23/24 frozen matches](../results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions)
after500k actions in48m53s, but mean11.0 misses the existing15-point gate.
Test one new learner, seed4013, with that recipe and budget fixed. This checks
promising winning behavior without calling the first result a gate pass or
lowering the target. No broad seed campaign or unchanged extension is allocated.

Keep centered CDP/action-effects1,4M/RGB48/Auto,3,578,225 parameters,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k, actor_unimix0 and no action
aid. Full18/sticky.25/repeat4, native max-pooled frames and one GPU resize remain.
Only learner/environment seed1009→4013 and evaluation seed4000001009→4000004013
change in the recipe. CLI`1ed07af` adds optional imagination length relative to
the primary run; this test omits it and retains15. Native`bfb8c31` is unchanged.

Start fresh weights, replay, games, belief and RNG, saving this learner's actual
zero-experience checkpoint. Different-seed initial tensors are not expected to
match the first learner. Audit full configuration against that run, allowing
only seed to differ, and keep this life's own frozen initial control.

Fixed500,000 aggregate actions, roughly2M emulator frames, expected124,939
updates, first update248 and zero debt. Retain the existing3300-second training
cap including construction. Save the latest100k recovery milestone, replaced
by fixed500k final. Preserve cutoffs and the latest checkpoint; no cap increase,
automatic resume, favorable-checkpoint selection or extension. The primary
2933.450-second run is a cost reference, not a completion or learning guarantee.

Evaluate final and actual initial with the first three natural sampled matches
per lane,24 per arm, zero updates and no aid. Each arm has600k-action/300-second
caps; retain excess matches and tails. Keep mean≥15, win fraction≥.9, at least20
natural matches and zero truncations. Report both learners separately; evaluation
matches are not independent learner seeds.

Audit all numeric reports, complete source/config/counters, finite checkpoints,
all346 frozen tensor equalities, full CPU action replays/videos and host guards.
Audit cap1200 seconds, service cap5160 seconds. Stop on real GPU faults or
numerical failures. The compact Boxing/Freeway confirmations used seed4013, so
this selected seed is not a held-out benchmark.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-500k-r32-seed4013`.
Controller: `run_pong_500k_r32_4013.py`; intended service:
`kindle-cheap-pong-48-r32-500k-4013-20261026`.
Exact commands match the primary run except seeds and artifact paths. CPU
preparation passes; native/runner/controller identities and caps are recorded.
Selected after the completed Qbert coefficient and Breakout H5 results.

Started04:10:30 UTC, source`923b47b`. Startup checks pass: full native config
matches the first500k learner except seed,3,578,225 parameters, H15, no action
aid, zero starting counters, finite346-tensor actual initial checkpoint and
first updates248:1,256:2,256:3. The initial world tensor hash differs from seed1009
as expected. Fixed final/actual-initial evaluations and full audits are pending.
