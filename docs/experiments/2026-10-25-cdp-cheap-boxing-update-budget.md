# Boxing RGB32 with the historical update budget

The200k/R16/seed1009 screen fails: frozen−3.875 versus initial0.00. It uses24970
updates, versus124939 in the historical successful1M/RGB64/R32/500k cohort. Test
whether the smaller image/model can learn within30 minutes when given that
larger update budget. This does not assume the short run failed only because of
too few updates. Start only after the active Pong screen and audits are reviewed.

The [retained1M curves](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-09-cdp-boxing-capacity.json)
provide a useful budget check: their three-seed online mean is2.35 at91136
actions,8.13 at116736 and46.87 at193536, all atR32. The new R16 screen stops
near the update count where those historical curves only begin improving.
These online curves are context, not matched-backend controls or frozen scores.

One fresh seed2017, Size1M/RGB32/Auto, centered CDP/action-effects1,
N8/B8/T16/H15, full BPTT/microbatch8, replay100k, **ratio32 and500k aggregate
actions**. All remaining learner settings match the completed small-image
Boxing screen. Keep full18/sticky.25/repeat4 and one GPU resize from native
max-pooled frames. No action aid. This is an independent fresh life rather than
a warm start;2017 is a historically reused development seed, not held out.

Expected124939 updates, first at248, zero debt. Save actual initial weights;
save the current checkpoint at250k and replace it with the fixed500k final.
Only the final and actual initial are evaluated, at seed4000002017: first three
natural sampled matches per lane,24 per arm, zero updates. Keep the existing
Boxing gate (mean≥50, natural wins≥90%, at least20 matches, no truncations).
No best-checkpoint selection or automatic extension.

The1800-second training cap includes construction and stays unchanged. Scaling
the measured6m03s screen's learning time by5 and remaining time by2.5 estimates
1636.73s (27m17s); this is a projection, not a guaranteed completion time.
A timeout is retained as a failed/cut-off attempt, with any250k checkpoint
preserved. It does not authorize increasing the cap or resuming the life.
Frozen caps remain600k actions/300 seconds each; full CPU replay/video and audit
cap1200 seconds; service cap3660 seconds. Verify finite reports/checkpoints,
all counters/config/source identities and all346 frozen tensor equalities.

Artifacts: `/x/Temp/kindle-cheap-20261025/boxing-1m-32-500k-r32-seed2017`.
Controller: `run_boxing_500k_r32_2017.py`. Intended service owner:
`kindle-cheap-boxing-r32-500k-2017-20261025`. Exact commands, caps, source/native
identity and controller hashes are declared before launch. Existing qualified
runtime; no rebuild or backend change. Ratio, budget and seed change together,
so this tests a useful recipe rather than an isolated causal claim.
