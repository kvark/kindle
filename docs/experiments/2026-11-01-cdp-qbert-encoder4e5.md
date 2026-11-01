# Check encoder4e-5 in the cheaper Qbert recipe

The retained [Qbert encoder2e-5 learner](../results/2026-10-30-cdp-qbert-encoder-moderate.md)
scores greedy13,785.417 and sampled9,987.500 in30m16s, below the original
15,000/90%-first-pyramid gate. Increasing update work with H5/R32 and the same
encoder rate does not improve its primary score enough to justify the added
cost. The completed [Breakout4e-5 comparison](../results/2026-11-01-cdp-breakout-encoder4e5.md)
improves both gameplay and fixed state readouts at nearly unchanged training
cost. Select one short Qbert execution/cost check at that fixed rate before
considering a full gameplay comparison. This remains within the five-game goal.

Change only encoder2e-5→4e-5 from the retained cheaper Qbert recipe. Keep centered
CDP/12M48/Auto/N8/B16/T16/H15/R16/expert100, base/actor4e-5, dynamics4e-4,
warmup1000, action-effects coefficient1, actor-unimix0, full18 actions,
microbatch16 and replay100k. Keep all-state corrective labels, all action
weights1, world imitation gradients and20% episode continuation guidance with
cap1920. Preserve native max-pooled frames and one GPU resize to48.

Native447fe50, the exact retained optional-rate runner and Qbert teacher stay
unchanged. Use `/x/Temp/kindle-breakout-corrective-20261028/runtime`; no native
build or production change. Seed4013/evaluation4000004013 and the cohort are
reused development inputs. No previous weights, replay, games, belief, RNG or
bounded experience enter this fresh learner. The earlier weaker4M Breakout
rate4e-5 result and the unsuccessful Qbert H5/R32 combination remain evidence
against a universal rate or automatic combination rule.

Train8192 aggregate actions/489 updates under180s, first update376/residual
credit.5. Restore for1024 greedy frozen actions under60s, guidance and labels
off, with zero updates. Require357 finite/control-matching initial tensors,
matched pre-update semantic events,357 invariant frozen tensors, exact source/
config/counters, finite reports and changed CNN/reward/actor/value. Verify
raw expert world/actor losses, row/weight accounting, passive labels, overrides,
complete training input-pixel/action/teacher-label replay and both AMD host
guards. The short frozen segment does not establish natural-episode competence.

Keep120s inner/150s outer replay,180s CPU audit and480s whole-service caps.
The conservative cost rule is
`check_training_seconds * 200000 / 8192 * 1.10 <= 2100`.
A future full comparison would require a separate selection after correctness
and cost pass: fresh200k/12,477 updates under35 minutes,100k/6227 recovery and
fixed final/actual-initial greedy and sampled24-episode assessments. Greedy
stays primary against13,785.417; the original15,000/90% gate remains unchanged.
No full learner, extension, cap increase, rate sweep, confirmation, automatic
successor or experience reuse is selected by this bounded protocol.

CPU preflight passes in.483s: both actual CLI paths,13 actual audit-config cases,
357 checkpoint-schema tensors, the8192/100k/200k update schedule and all235
retained control pins. Native construction attempts and environment creation
are zero. The runner is byte-identical to the retained control. Pin exact
sources, protocol, runtime and control evidence before launch; require an idle
host and unchanged boot/driver identities. Preserve failures and repair CPU-only
defects without repeating completed GPU work.

Root `/x/Temp/kindle-qbert-encoder4e5-20261101/native-check`, trial
`qbert-12m-48-b16-r16-encoder4e5-expert100-check`, service
`kindle-qbert-encoder4e5-check-20261101`. No full learner is selected.
