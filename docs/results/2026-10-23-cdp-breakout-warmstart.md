# Breakout: six million more actions do not improve this warm-start lineage

The bounded continuation completes, but frozen score falls from98.21 to85.17.
Do not extend it again automatically or call this progress in competence.
Published381.81, the two-wall864-point gate and reliability remain unmet.

| Policy | Total lineage exposure | Frozen mean,24 natural rounds | Two-wall successes |
| --- | ---: | ---: | ---: |
| Unchanged source |2M actions |98.208 |0/24 |
| Warm-start prefix |4M actions |80.167 |0/24 |
| Warm-start final |8M actions |85.167 |0/24 |

The source-policy control exactly reproduces the original98.208 mean. The old
actual-initial control remains1.417, but was not this warm life's starting model.
The final change versus source is−13.042 points (−13.3%). One lineage, reused
development evaluation seeds and24 rounds/policy do not establish a population
effect or isolate why performance regressed.

Replay, games and RNG restarted while model/optimizer/slow critic/normalizer
and counters were retained. This is **not** an uninterrupted8M-action baseline.
Thus the result does not prove that longer uninterrupted learning would fail,
nor that representation capacity alone caused the plateau.

## Cost and checks

The new life executes6M actions/1,499,939 updates in26,848.47s (7h27m28s),
ending at8M total environment actions and1,999,878 learner updates, zero debt.
Its online last50 means at each additional1M are131.64/70.36/60.36/103.60/
83.38/90.68. No post-hoc evaluation of a favorable intermediate checkpoint.
The separate disposable smoke cost4096 actions/963 updates and was not used
as the main starting model.

All four main guards, finite checkpoints, three exact346-tensor frozen
comparisons and complete CPU replay/videos pass. All72 selected rounds are
natural, with zero evaluation updates/cutoffs. Native c8aeba28 and unassisted
12M/B8/T16/BF16/CDP/action-effects1 recipe stay unchanged. Training guard ends
October22 10:01:54UTC; all GPU work ends10:03:32 and audit10:04:41. The service
is terminal success; no learner worker remains. Results reviewed October23.

## Next decision

Existing forecasts show useful cumulative rewards, not complete world quality.
The [greedy-versus-sampled diagnosis](2026-10-23-cdp-breakout-action-mode.md)
completes: source53.46/final97.79/actual-initial2.21. Keep sampled results as
the standard; do not replace them with whichever mode wins. The bounded
12M state diagnosis follows. No backend optimization or further long Breakout
extension is started. All five game targets remain open.

- [Full audit](../../runs/cdp-breakout-warmstart-20261022.ec73sFZx/train/learning-review.json)
- [4M-total whole rollout](../../runs/cdp-breakout-warmstart-20261022.ec73sFZx/train/breakout-prefix.mp4)
- [8M-total whole rollout](../../runs/cdp-breakout-warmstart-20261022.ec73sFZx/train/breakout-final.mp4)
- [Unchanged source rollout](../../runs/cdp-breakout-warmstart-20261022.ec73sFZx/train/breakout-source.mp4)
