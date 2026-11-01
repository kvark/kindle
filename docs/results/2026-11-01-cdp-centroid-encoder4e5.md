# Position supervision at encoder4e-5 passes bounded execution and cost

The fresh Breakout check completes **8,192 actions / 32,722 frames / 489 updates
in 47.291s**. The following **1,024 greedy frozen actions / 4,096 frames** take
0.883s, with zero updates and no completed natural episodes. This establishes
execution and early cost; task competence remains unmeasured.

The fixed conservative projection is **21m10s per 200k actions**, including the
10% margin, below the 30-minute cap. The retained no-centroid control takes
21m28s for its completed 200k learner. Early projection and completed-run time
do not isolate loss overhead or guarantee the candidate's full-run cost.

Use the existing scale100 position objective with encoder rate4e-5. Keep
12M64/Auto/N8/B16/T16/H15/R16/expert100, sampled collection and posteriors,
actor-unimix.01, the current corrective pixel teacher, 20% episode guidance
capped at1920, full18/sticky.25/repeat4 and one GPU resize from native frames.
Native2974160/SHAe7ba0d4 and the exact existing centroid runner are reused;
no native build or new loss implementation occurs. Native and runner versions
differ from the447fe50 control, limiting causal claims.

All **372** initial tensors match the retained centroid initialization. All
**357** shared tensors and **47** pre-update semantic events match the stronger
no-centroid control. Every frozen tensor remains invariant. The16,990,739
trainable parameters include656,386 in the training-only centroid head.
The corrected schema includes its ten optimizer-moment tensors.

All489 supervised reports are finite. CNN, reward, actor, value and both
centroid weight layers change. Raw centroid loss averages0.097274, range
0.037628–0.501080; valid fraction averages0.817884, range0.578125–0.980469.
These are replay-training statistics, not held-out position accuracy.
Actual guidance overrides3,795 actions (46.326%) and passively labels4,397.
Complete pixel/action/teacher-label replay and both host guards pass.

The2.821s CPU preflight checks two commands, three negative CLI cases,13 config
cases, the update schedule,372 schema entries and357 shared initial tensors.
It preserves227 control,41 original-centroid and21 repair inputs, and verifies
all six preceding Qbert services terminal. No native agent or emulator is
constructed during qualification; expected CLI rejections are recorded.

The service starts **11:02:50 UTC November1**, source`77866fc`, PID787070 /
invocation`881d7709c8604708b02fa09881926e39`. It succeeds **11:03:56**; terminal
collection completes **11:04:29**, verifying the original unique invocation
and all302 declared inputs. No failure, retry or repeated stage occurs. Earlier
centroid numerical, schema and full-evaluation failures remain preserved.

No check experience may be reused by a full learner. A separate fresh200k
comparison is now being prepared; it requires its own qualified controllers
and selection. No full learner is selected at bounded closure. The original
90%-two-wall gate remains unchanged, and the five-game goal remains unfinished.
No learner or GPU allocation is active; inspect live state before new work.

[Bounded protocol](../experiments/2026-11-01-cdp-centroid-encoder4e5.md).
Root `/x/Temp/kindle-centroid-encoder4e5-20261101/native-check`, trial
`breakout-12m-64-b16-r16-centroid100-encoder4e5-expert100-check`.

- Preflight: `246dea04f1b770f74378b40766f27c0a7ce2270d27a62b6f4df3f8329fdbeb16`.
- Declaration: `bf7e648d7febfa2f1ba191c9866137f2d08c18a7afb156e9d410a510e45710ef`.
- Review: `43a5482f9fb9fb23592442181ece5e93d20e656bb720e6b02671b3e55dc57059`.
- Completion: `bb953beb006da1c7efd5a5b7e7dfcaa4becbe559bb27e5523bb383d28337396b`.

Complete: do not rerun bounded stages, passed preflight or terminal collection.
