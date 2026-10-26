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
Commands, caps, recipe differences and source hashes are prepared. Launch only
after reviewing the active Breakout mixture result; this is not an automatic
queue. Seed1009 is reused development data. A useful result warrants focused
confirmation, not a broad seed campaign or five-game reliability claim.
