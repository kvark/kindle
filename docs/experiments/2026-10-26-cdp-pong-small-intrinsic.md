# Smaller exploration bonus for compact Pong

**Completed: frozen14.9583 and24/24 wins in49m55s; the mean15 gate still fails.**
All audits pass and the service exits. See the
[result and comparison](../results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result).
One focused seed4013 confirmation is selected; no extension of this life.

Originally selected after the Breakout H5 cutoff and frozen assessment. That run stops
at487,736 actions under its45-minute cap and scores8.458 versus initial1.375,
0/24 two-wall successes. Numerical/checkpoint/accounting and frozen replay checks
pass; the original time-budget failure remains. Both services have exited and
the comparison is reviewed. Select this one fixed-budget Pong test next.

The compact Pong recipe wins 23/24 and 22/24 frozen matches on seeds 1009 and
4013, but means 11.0 and 7.333 miss the existing mean-15 gate. Both runs finish
500k actions within 55 minutes. Their
[completed-log analysis](../results/2026-10-25-cdp-cheap-pong.md#exploration-bonus-scale-in-the-completed-learners)
finds late scaled-bonus/net-extrinsic-prediction ratios of 3.763 and 3.461.
Pong rewards are signed, so cancellation limits this comparison; it does not
establish that exploration caused the score gap. Test a smaller game-specific
bonus while retaining the useful compact recipe and its fixed experience budget.

Fresh seed 1009/evaluation 4000001009, centered CDP, Size4M/RGB48/Auto,
3,578,225 expected parameters, N8/B8/T16/H15/R32, full BPTT/microbatch8 and
replay100k. The native configuration changes only intrinsic_reward_scale 1→.01
versus the primary 500k control. The positive coefficient keeps the action-effects
ensemble enabled. Actor_unimix stays 0. Full18/sticky.25/repeat4, no-op starts0,
100000-frame episode limit, native max-pooled frames and one GPU resize remain.
No external action aid, RAM input, new task reward, optimizer change or rebuild.

Save the actual zero-experience checkpoint and verify all 346 tensors against
the primary seed1009 initial checkpoint at startup. Retain this life's own
frozen initial control. Start learner/environment counters, replay, games,
belief and RNG afresh. The seed is reused development data, not held out.

Fixed 500,000 aggregate actions, roughly 2M emulator frames, 124,939 expected
updates and first update at action248. Training cap 3300s (55 minutes), including
construction. Audit the 100k/200k/300k/400k recovery checkpoints before replacement;
evaluate fixed500k final, with no best-checkpoint selection, automatic extension
or cap increase. Preserve any cutoff and the latest checkpoint explicitly.

Frozen final/actual-initial evaluation selects the first three natural sampled
matches per lane, 24 per arm, with zero updates/truncations. Each arm has a
600k-action/300s cap. Retain excess matches and partial tails. Preserve the gate:
at least20 natural matches, mean≥15, at least90% wins and zero truncations.

Audit every numeric report, full source/configuration/counters, finite
checkpoints, all 346 unchanged frozen tensors, full CPU replays/videos and host
guards. Audit cap1200s; service cap5160s. Stop on real faults or numerical failure.
Native `bfb8c31`, runner `1ed07af` and CPU replay `e853818` remain unchanged.
The runner adds an optional imagination-length flag relative to the primary
control's runner; both recipes retain H15.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-intrinsic001-500k`.
Controller: `run_pong_intrinsic001_500k.py`; intended service:
`kindle-cheap-pong-intrinsic001-500k-20261026`.
CPU preparation checks the complete queue against the primary control: only
the coefficient and artifact paths differ. An unselected controller refuses
before creating GPU/log artifacts. Selection must pin the preceding review
and source commit; the controller also requires the preceding service to be
terminal. No automatic next allocation or extension.

Started06:36:15 UTC, source`12a6553`. Startup confirms complete native config
changes only intrinsic_reward_scale1→.01,3,578,225 parameters, zero starting
counters, the enabled ensemble, no external aid, finite346-tensor own initial
checkpoint matching the control exactly, and updates248:1,256:2,256:3.
Initial metadata SHA256
`39e7168a675225259a52dcc55586845c22861a163c201685729228601d5e488f`;
selected declaration SHA256
`d2249ae6865eaa21169806b7e2331ec83eedac87106769842486e9fd00aeb213`.
Recovery checks and completed-run plotting are monitored automatically. The
read-only checkpoint watch lasts60 minutes so it covers a55-minute training cap;
this changes no training limit. Fixed final/own initial evaluations and full
audits now complete; the original budget and success gate remain unchanged.
