# Five-game CDP learning and the DreamerV3 budget question

October5 user goal: **unlock training on Boxing, Pong, Freeway, Breakout and
Qbert with CDP, and test whether DreamerV3 scores are reachable with less budget.**
This takes priority over the proposed Venture follow-up. It does not restart
the stopped45-run CDP/RGB/Tiny matrix or the cancelled historical12M queue.

## Reference and claim scope

Keep two references separate. DreamerV3's full Atari benchmark uses200M emulator
frames; Atari100k uses400k frames (100k decisions with repeat4), a different
nonsticky/minimal-action protocol. The [paper](https://www.nature.com/articles/s41586-025-08744-2)
and pinned [released curves](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe/scores)
are the source, not old Kindle mastery gates. Current Freeway training used
about800k frames: much less than200M, but **more**, not less, than Atari100k.

The primary long-run target is the equal-seed mean of the released Atari57
curve bins above180M through200M frames. Retain final-bin scores too; do not
choose the more convenient statistic after Kindle's result. All released seeds
are included, even though the available count varies by game. These are online
curve summaries, not our held-out frozen scores or an exact reproduction.

| Game | Released seeds | Atari57 last10% mean | Final bin | Atari100k final bin |
| --- | ---: | ---: | ---: | ---: |
|Boxing|3|99.613|99.793|86.6|
|Pong|6|20.445|19.937|−5.0|
|Freeway|5|33.399|33.733|0.0|
|Breakout|5|381.811|393.683|11.2|
|Qbert|6|193,220.767|223,077.776|3,700.0|

[Reference JSON](../results/2026-10-05-dreamerv3-five-game-reference.json)
retains hashes, seed counts, uncertainty and budget distinctions. Native64 RGB,
reset no-ops, artificial cutoff, model size and learning settings differ from
the published recipe; sticky/full-action agreement alone is not protocol parity.
Report online-versus-online curves and frozen-versus-initial-control scores
separately. Reaching a published number under these differences is a qualified
reference result, not proof of algorithmic superiority. A same-hardware,
same-protocol Dreamer RGB control is required before claiming compute savings.
Do not infer GPU cost or utilization from parameter count or elapsed game time.

## Stage1: fixed small-recipe breadth screen

Reuse all three completed [Freeway200k cohorts](../results/2026-10-05-freeway-effects-200k.md),
including weak seeds and controls. Run only the other four games, in order
Boxing/Pong/Breakout/Qbert, seeds1009/2017/3019. **Twelve new training runs**, not
45:200,000 aggregate actions/49,939 updates each,2.4M new training actions and
599,268 updates total. Each run starts fresh. The complete five-game screen
contains15 CDP learners including the three retained Freeway runs.

Keep native `f4b6a5c7` and the same CDP plus action-effects-disagreement package
for every game: Size1M/N8/B8/T16/H15/R32/microbatch8/replay100000; full18 actions,
sticky.25, repeat4, no reset no-ops,100000-frame cutoff, native input with one
GPU RGB64 resize. CDP cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000,
AGC.3, ac_grads=false, intrinsic coefficient1. No Tiny/DINO, action aid, external
reward shaping, video pretraining, game-specific settings or representation sweep.
This tests a whole agent package, not CDP versus RGB as an isolated loss change.

Meganeura main6288f885 differs from qualified592a2f5a only in docs/paper artifacts;
Blade maine349cddf is unchanged (October5 recheck). Native source is unchanged;
CI303/304 and prior independent GPU qualification still apply. No rebuild or
duplicate numerical campaign. The new orchestration receives a CPU rehearsal
against a completed real training/candidate/control trio before launch.

## Frozen evaluation and resource bounds

After each training audit, evaluate its final checkpoint and an actual initial-
weight control. Reuse the saved zero-action/zero-update initial checkpoints:
all five games share the exact RGB64/full18 configuration and those weights
have never observed a game. Validate configuration and zero-experience metadata;
perform new control gameplay, not reuse Freeway control scores for another game.
Same environment seeds per pair: base2,000,000,000 plus learner seed and stream
offset1,000,003. Sampled policy, first3 completed episodes/stream,24/model,
cap200k actions/30min. Retain natural/cutoff/excess episodes and unfinished tails
separately. Incomplete cohorts have no complete-cohort score. No selected best
checkpoint, success-only video or evaluation updates.

Verify complete counters, zero learner debt, finite final checkpoints, exact
frozen tensor equality and full CPU trajectory replay. Save whole stream0 videos
for all24 new candidate/control evaluations. Audit each stage before advancing.
Game-boundary summaries report every seed and its score/action/time curve;
bootstrap learner seeds, not episodes. Report human-normalized scores and
published-reference gaps without redefining success around the easiest game.

Serialize guarded GPU work in a persistent service: Restart=no,
KillMode=control-group,12h total deadline,45min per training process. Expected
roughly6–7h including evaluation/replay audits. Require RTX5080 and>=2GiB
sampled Vulkan estimated headroom. Record standalone allocation warnings;
API/numerical/hard faults or deadlines stop for review. No NVML polling,
recovery or automatic retry. CPU replay/analysis uses one CPU,2GiB and zero swap.

## Decision after Stage1

The goal is not complete merely because these jobs finish. For each game,
establish whether frozen learning beats initial controls and quantify the gap
to both published references. Distinguish interaction efficiency, learner
updates, world/whole-agent time and all additional diagnostic/development compute.
Only a reached score at fewer measured frames supports a qualified interaction-
budget claim; unmatched hardware cannot support a wall-time speedup claim.

Review the evidence before changing budget, capacity, temporal context or the
exploration mechanism. Diagnose observed failures; no automatic longer runs,
per-title action tricks or unchanged mastery queue. The original five remain in
scope; a successful Boxing/Freeway subset does not complete the goal. Venture,
Tiny comparisons, video priors and native-game work remain deferred here.

Artifacts: `runs/cdp-five-game-screen-20261005.4e20N61X`.
