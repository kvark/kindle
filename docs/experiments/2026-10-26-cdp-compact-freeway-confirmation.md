# Confirm compact aided Freeway on one new seed

The selected RGB32/R32 recipe reaches31.0417 frozen crossings,24/24 successes,
in28m17s; [all audits pass](../results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget).
Test one fresh learner, seed4013, with the successful recipe fixed before launch.
This is focused seed confirmation, selected after reviewing the completed Pong
test. The recipe and budgets below were fixed before allocation.

Keep Size1M/RGB32/Auto,834,673 parameters, centered CDP/action-effects1,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k and actor_unimix0.
Full18/sticky.25/repeat4, native max-pooled frames and one GPU resize remain.
Keep5% persistent-uniform16-action training aid, disabled in frozen evaluation.
Only learner/exploration seed1009→4013 and evaluation seed4000001009→4000004013
change. Audit the complete configuration, aid and native identity against the
successful run; the aid's random choices also change with its seed.

Use fresh replay, games, belief and RNG, retaining this life's actual initial
weights. All346 initial tensors must match the retained Boxing4013 initial
checkpoint: its native configuration is exactly the same, despite the different
game and external action aid. That is an initialization check, not a reused
Freeway evaluation or pretrained policy.

Fixed500k aggregate actions, approximately2M emulator frames, expected124,939
updates, first update248 and zero debt. The unchanged training cap is1800 seconds
including construction. Save250k recovery weights, replaced by fixed500k final.
Retain cutoffs and the latest recovery checkpoint; no cap increase, automatic
resume, favorable-checkpoint selection or extension.

Evaluate final and actual initial with the first three natural sampled episodes
per lane,24 per arm, zero updates and no aid. Each arm has600k-action/300-second
caps. Preserve excess episodes, tails and cutoffs. The existing gate remains
mean≥25,≥90% episodes reaching25 crossings, at least20 natural episodes and zero
truncations. Report the two learner seeds separately; episodes within a learner
are not independent training seeds.

Audit every numeric report, complete configuration/source/counters, finite
checkpoints, exploration ledger, all346 frozen tensor equalities per arm and
full CPU action replays/videos. Use the existing host guards and stop on real
faults or numerical failures. Audit cap1200 seconds; service cap3660 seconds.
The primary run's1,697.484-second training time is a reference, not a guarantee.

Artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-32-500k-r32-aided-seed4013`.
Controller: `run_freeway_500k_r32_32_aided_4013.py`; intended service:
`kindle-cheap-freeway-32-r32-500k-aided-4013-20261026`.
Exact commands, caps, native/runner/controller hashes and the successful1009
review hash are sealed. Native source`bfb8c31`, runner`1ed07af`; no native rebuild.
The seed is selected for this confirmation, without a held-out benchmark claim.
Two successful learners would still leave broader reliability and the other
games' unresolved learning gaps open. No broad seed campaign is queued.

Started October26 at02:55:20 UTC, source`1ed07af`, after Pong's completed audit.
The new CLI exposes an optional imagination length; this run omits it and retains
H15. The sealed declaration records the previous/new runner hashes and263 passing
vector-runner checks. Startup checks pass: expected config with only seed changed,
834,673 parameters, exact aid, all346 initial tensors equal Boxing4013 and first
updates248:1,256:2,256:3. Inspect the named service before further GPU work.
Fixed final/actual-initial evaluations and the complete audit are pending.
