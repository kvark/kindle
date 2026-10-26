# Short-imagination Breakout warm start

The retained2M policy now [scores67.5833 on AMD Auto](../results/2026-10-26-cdp-retained-breakout-auto.md),
with0/24 two-wall successes and all346 tensors unchanged. Prepare one50k-action
warm start with imagination length15→5. This reuses learned weights instead of
repeating the expensive source training. It is a changed learning hypothesis,
not an unchanged long extension or an independent learner seed.

Historical [source-state readouts](../results/2026-10-23-cdp-breakout-state.md)
give ball-x R².766 at one step versus.269 at15 steps. They motivate testing
shorter imagined rollouts but do not establish a cause of weak control or an
optimal horizon. H5 may reduce accumulated model error and update cost while
weakening long-term credit. No learning benefit or speedup is assumed.

Copy the verified Auto input into a fresh directory, changing only saved
`config.imagination_length` from15 to5. Preserve all346 tensors, optimizer and
slow critic state, return normalizer, seed1009 and counters499939/2027152.
Keep12M/RGB64,16,334,353 parameters, centered CDP/action-effects1,
N8/B8/T16/R32, full BPTT/microbatch8, replay100k, actor_unimix0 and discount
horizon333. Native max-pooled frames receive one GPU resize; full18/sticky.25/
repeat4 remains. No external action aid, new labels or backend change.

The CPU copy audit passes: all346 tensors are finite and bitwise identical;
only imagination length differs from the verified Auto source. Original shared
BF16 files remain unchanged. Restore uses the labelled copy's saved config;
the CLI continues to reject training overrides on restore. No production code
change, agent construction or GPU work is part of preparation.

Before training, evaluate this H5 source with seed4000001009 and the first three
natural sampled episodes per lane,24 total. Require zero updates/truncations,
all346 frozen tensors unchanged, full CPU replay/video and a passing guard.
Its complete saved action/reward/boundary/frame trace and replay outcomes must
match the already measured H15 source. If they differ, stop before learning and
review the evidence; do not retry or silently accept a changed starting policy.

Then restore the same input afresh for **50,000 aggregate actions**, approximately
200k emulator frames, expected12,439 updates, first update248 and zero debt.
Replay, games, live belief and RNG restart. This is not uninterrupted learning;
the input is a trained source policy, not zero-experience initial weights.
Expected final counters are512378 learner updates and2077152 environment
actions, preserving the export's prior evaluation history. Save25k recovery
(506128 updates), replaced by fixed50k final. Training cap**1200 seconds**
including construction. Preserve cutoffs and the latest checkpoint; no cap
increase, automatic resume, favorable-checkpoint choice or extension.

Evaluate the fixed final under the same24-natural-episode protocol. Each source
and final evaluation has600k-action/300-second caps; retain excess episodes and
partial tails. Keep the existing Breakout gate: at least20 completed episodes
and≥90% two-wall successes (864-point threshold); additionally require zero
truncations for a completed test. Report score changes as development evidence,
without lowering the gate or treating episodes as independent training seeds.

Audit every numeric report, full source/config/protocol/counters, finite
checkpoints, all346 frozen tensor equalities, full CPU action replays/videos
and host guards. Stop on real GPU faults or numerical failures. Source and
final audits each have600-second caps; service cap3060 seconds. World learning
still uses128 replay positions/update; imagined transitions shrink1920→640.
That does not imply a threefold total speedup. Additional experience, replay
restart and H5 change together; there is no matched H15 warm-start learning arm.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-retained-h5-50k`.
Controller: `run_breakout_retained_h5_50k.py`; intended service:
`kindle-breakout-retained-h5-50k-20261026`.
Source at preparation`4844b8b`, native`bfb8c31`, runner`1ed07af`.
Native SHA256`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Copied metadata SHA256`913fec93245330c44d2bb2fe17176746906bf162053e23438743cc78650b332f`.

Prepared, not started. Review the active Qbert coefficient test before selecting
this allocation. No concurrent learner or automatic next queue is created.
