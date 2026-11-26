# Bounded Breakout check with stronger continuation supervision

Select one fresh execution and cost check after the
[paired arrival diagnostic](../results/2026-11-04-cdp-breakout-arrival-head.md).
Mean continuation remains .759636 after observing24 actual game endings, whose
target is zero. The replay audit found sparse terminal exposures and some fitting.
These findings motivate testing stronger existing supervision; they do not prove
that loss weighting causes the control weakness or that increasing it will help.

Change only `loss_scales.continuation` from1 to10 in the retained recipe. This
scales BCE on **both terminal and nonterminal rows**, including its shared-world
gradients. It is not class balancing or terminal resampling. Optimizer clipping
and adaptation mean a tenfold loss weight does not imply tenfold parameter steps.
Keep intermediate life losses nonterminal and retain the original reward and
episode definitions. The completed H3 world-supervision regression remains
relevant; H3 is off here. No new objective, parameter or native build is needed.

Expose `--continuation-scale` in the canonical vector runner and a private copy
of qualified runnercf4ea238. The option is positive finite F32, fresh-training
only, default omitted; restore preserves checkpoint configuration. The private
runner differs by exactly ten added lines and otherwise retains its qualified
behavior. Use native447fe50/SHA35ad0db and the corrective runtime unchanged.
Verify canonical/private invalid-input, restore, default and requested-config
paths on CPU, plus the actual queued commands stopped before native/emulator
construction. No unrelated runtime overlay or source replacement.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1,
disagreement1, expert100/unit weights/world imitation gradients, sampled
collection/posteriors, actor-unimix.01, replay100k, full18/sticky.25/repeat4/noops0,
native pooled RGB with one GPU64 resize, and20% guided episodes/cap1920 with
all-state corrective labels. Reuse development seed1009 with fresh experience.

Train8,192 aggregate actions/489 updates within180s, first update376 and residual
credit.5; save actual initial/final checkpoints. Require all357 initial tensors
and47 pre-update semantic events to match the retained control. Check exact
configuration, finite reports, cadence, label bounds and the full weighted world
loss sum including continuation10. Require changed CNN, reward, continuation,
actor and value parameters. Complete the full training pixel/action/label replay.

Restore the final checkpoint for1,024 frozen greedy actions at seed4000001009,
guidance/labels off, within60s. Require zero updates, all357 invariant finite
tensors, exact source/native/ROM identities and host guards. This short check
does not establish natural-episode competence, terminal calibration or a selected
full learner. No weights, replay, RNG, live belief or games transfer to a later run.

Replay cap120s, CPU audit180s, main600s and collector650s. Exercise audit
negative cases, actual replay argv, checkpoint schema and original terminal
collector on CPU before launch. Preserve all earlier artifacts, including
historical sources resolved to exact bytes. Check live state immediately before
launch; no concurrent GPU job, repeated successful stage or automatic recovery.

Compare training time with retained47.689684s. The conservative200k projection
`seconds*200000/8192*1.1` must fit30min before separately selecting a full learner.
Any full comparison must use fresh experience,100k recovery and fixed final/
actual-initial sampled/greedy24-natural-episode assessments, greedy primary
against143.667 and sampled secondary against70.750. Keep the original864/
90%-two-wall gate. No sweep, cap increase, extension or automatic successor.

Root `/x/Temp/kindle-breakout-continuation10-20261104`, service
`kindle-breakout-continuation10-check-20261104`. The five-game goal remains unfinished.
