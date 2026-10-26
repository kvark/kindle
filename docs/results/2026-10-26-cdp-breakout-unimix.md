# Native policy mixture reduces FIRE gaps; Breakout remains weak

At the fixed200k budget, training with actor_unimix.01 scores7.0833 versus the
0-mixture control's6.2083. Adding the mixture only to the frozen control weights
scores6.0833. All three policies have0/24 two-wall successes. The main useful
change is much shorter chosen-FIRE gaps and frozen rollouts; this is not task
success or evidence that more unchanged training will resolve the weakness.

[Declaration and frozen-control protocol](../experiments/2026-10-26-cdp-cheap-breakout-unimix.md).
Both trained learners use fresh seed1009,4M/RGB64/Auto,4,131,729 parameters,
centered CDP/action-effects1, N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k,
full18/sticky.25/repeat4, native max-pooled frames and one GPU resize. All346
initial tensors match; the complete native config differs only in actor_unimix.
Native code is unchanged. CLI-only commit`bcef410` exposes the existing control
and passes255 vector-runner CPU tests. The mixture affects real acting,
imagined sampling and the actor objective; it remains saved in frozen sampling.

| Training / frozen mixture | Frozen mean | Two-wall successes | Mean selected length, actions | Longest selected no-FIRE gap | Selected FIRE fraction |
|---|---:|---:|---:|---:|---:|
|0 /0, completed control|6.2083|0/24|3,267.00|12,115|1.11%|
|0 /.01, frozen-only control|6.0833|0/24|726.96|724|4.95%|
|.01 /.01, fresh training|7.0833|0/24|499.33|447|13.72%|

Both actual-initial sampled policies score1.375. Every arm selects the first
three natural episodes per lane using frozen seed4000001009. A command gap can
include active play and is not itself a verified relaunch stall. The prior
[saved-action replay](2026-10-26-cdp-cheap-breakout.md) verifies one12,061-action
delay after a lost life; the new summaries bound chosen-command gaps in these
recorded evaluations without using RAM/life state as policy input.

Mixed training completes200,000 aggregate actions,799,097 emulator frames and
49,939 updates in **1,348.669s (22m29s)**,148.29 actions/s, versus1,350.478s for
the0-mixture control. First update248, zero debt,734 natural training episodes,
final online last50 mean7.32. The100k/24,939-update recovery checkpoint passes
finite/config/counter checks before replacement by fixed200k. No retry, cap
increase, selected checkpoint or extension.

Its final evaluation uses16,296 actions/65,145 frames/6.911s, collecting30 natural
episodes and retaining six excess episodes plus tails. Initial uses5,944
actions/23,717 frames/2.612s,29 natural episodes, five excess plus tails.
Both perform zero updates, have zero truncations and preserve all346 tensors.
All49,939 numeric reports, finite checkpoints, complete config/source/counters,
full CPU replays/videos and three host guards pass; service exits successfully.

The separate posthoc control copies the original200k trained weights and changes
only saved actor_unimix0→.01. It uses23,128 actions/92,457 frames/9.801s, versus
175,280 actions/701,036 frames/72.906s for the original frozen evaluation. It
collects30 natural episodes, selects24 and retains six excess episodes and tails,
with zero truncations or updates. All346 copied/frozen tensors, original metadata
and tensor hashes, complete CPU replay/video and host guard pass; service exits.
The mixed run's actual-initial evaluation is reused after matching all346 initial
tensors and the complete frozen protocol, native/runner and seeds.

The frozen-only result shows much shorter rollouts without a score improvement
for these fixed weights. Mixed training has a small observed score gain and
chooses FIRE more often. The sampled policy, state visitation and completed
episode counts differ; faster evaluation is not faster optimizer work. This is
one reused development seed and a posthoc control, not seed reliability or a
held-out comparison. No unchanged Breakout extension follows. The next allocation
is the [compact aided Freeway test](../experiments/2026-10-26-cdp-cheap-freeway-update-budget.md).

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-unimix001-200k` and
`/x/Temp/kindle-cheap-20261025/breakout-unimix-acting-only-probe`.
The first retains`curve/` and`mixture-comparison/`; both retain corrected
`fire-gap-summary.json`, checkpoints, complete logs/replays/videos and source hashes.
An auxiliary frame-summary assertion caught cumulative counters being summed;
the corrected per-episode deltas match independent replays and total frames.
Scores/actions/FIRE gaps/selection are verified identical to the archived v1
summaries. `fire-gap-frame-correction.json` records the correction; the original
strict training/frozen replay audits already used correct frame accounting.

Training review SHA256
`41cb278fbdcc504d0d4262b9b4e006aec6f514242b331340de49c58fec1f14f1`.
Frozen-control review SHA256
`2e0b341a19e7e028bfae855dc795158a23a71800f53e98bd4b83b1e967552eb2`.
Native source`bfb8c31`, runner`bcef410`, native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
