# Give compact Freeway the successful update budget

The aided1M/RGB64/R16 screen freezes at0.25 crossings after200k actions and24,970
updates. Historical1M capacity results support this model size for Freeway, while
both new1M/RGB32 Boxing learners pass after500k actions and124,939 updates in
about28 minutes. Test whether a smaller Freeway input permits enough learning
work within the same30-minute cap. This is a fresh recipe test, not an extension
of either completed200k life.

Use fresh development seed1009, centered CDP/action-effects1, Size1M/RGB32/Auto,
834,673 parameters, N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k.
Full18/sticky.25/repeat4, native max-pooled frames and one GPU resize. Saved CPU
samples retain visible cars and the player marker at32, but do not establish
that this resolution is sufficient for learning. No CPU downscale/upscale.
Native source`bfb8c31` and runner`bcef410` need no new implementation or build.

Keep the existing training aid:5% persistent-uniform16-action blocks, with no
directional hint or reward shaping; disable it in frozen evaluation. Native
actor_unimix remains0. Compared with the aided screen, change RGB64→32,
ratio16→32, budget200k→500k and the already-qualified native runtime. Retain seed
and aid. Disclose these changes together; do not attribute the outcome solely to
resolution or update budget. The initial weights cannot equal the oldRGB64
checkpoint; compare all346 against the same-seed1M/RGB32 smoke and retain this
run's actual-initial control.

Fixed500k aggregate actions, expected124,939 updates, first update248, zero debt.
Training cap1800 seconds including construction. Save actual initial and250k
recovery weights, replacing recovery with the fixed500k final. If the cap stops
training, preserve the cutoff and last checkpoint; no cap increase or automatic
resume. Boxing's1,693–1,702-second measurements are planning evidence on another
game, not a guaranteed Freeway runtime.

Frozen seed4000001009; first three natural sampled episodes per lane,24 per arm,
final and actual initial, no aid or updates. Each arm has600k-action/300-second
caps; retain excess episodes, tails and cutoffs. Existing gate: mean≥25,
≥90% episodes with at least25 crossings, minimum20 completed, zero truncations.
Audit every numeric report, complete config/source/counters, aid ledger, finite
checkpoints, all346 unchanged frozen tensors, full CPU replays/videos and guards.
Audit cap1200 seconds; service cap3660 seconds.

Artifacts: `/x/Temp/kindle-cheap-20261025/freeway-1m-32-500k-r32-aided`.
Controller: `run_freeway_500k_r32_32_aided.py`. Intended service:
`kindle-cheap-freeway-32-r32-500k-aided-20261026`.
Commands, caps, recipe differences and source hashes are sealed before launch.
Both Breakout mixture reviews complete with failed task gates; see the
[result](../results/2026-10-26-cdp-breakout-unimix.md). This is not an automatic
queue. Seed1009 is reused development data. A useful result warrants focused
confirmation, not a broad seed campaign or five-game reliability claim.

Started October26 at01:24:36 UTC after reviewing both completed Breakout audits.
Startup checks pass:834,673 parameters, expected config/aid/source, finite zero-
experience checkpoint, all346 initial tensors equal the same-seed RGB32 smoke,
and first updates248:1,256:2,256:3. Source at launch`541c90b`; native`bfb8c31`,
runner`bcef410`. `startup-review.json` retains the checks.

Completed:500k actions/1,999,960 frames/124,939 updates in28m17s, within the
unchanged cap. Frozen final31.0417 versus actual initial0.00,24/24 versus0/24
crossing successes; evaluation disables the aid. All numeric, checkpoint,
configuration/source/counter, aid, frozen-tensor, replay/video and guard checks
pass; the service exits successfully. See the [result and CPU reference](../results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget).
No extension. The separate Pong update-budget test is selected after this review.
