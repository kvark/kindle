# Lost lives have no immediate reward signal in the saved Breakout cohort

All120 observed lost lives in the fixed24-episode cohort have zero raw reward;
96 do not terminate the game. The retained runner passes raw environment
rewards directly to native `observe`, and the current auditor requires stored
extrinsic rewards to equal those raw rewards. This supports testing an explicit
sparse failure signal. It does not prove that absent negative reward causes
weak control: losing a life also removes future scoring opportunities, and
game-over already supplies a terminal target.

| Selected life-loss events | Events | Zero raw reward | Game-over |
| --- | ---: | ---: | ---: |
| Intermediate losses |96|96|0|
| Final losses |24|24|24|
| Total |120|120|24|

The [fixed saved-data analysis](../experiments/2026-11-03-cdp-breakout-life-feedback.md)
reuses the completed encoder4e5 own-policy capture. All37,016 action rows/
148,038 frames join to the original action/reward/boundary log, with ordered
lane/episode steps, continuous life counts and exact selection. The31,161
selected rows reproduce the24 natural episodes and original mean104.667.
All saved selected life-event identities match. Across all observed rows,
including extra episodes and tails, there are141 lost lives andzero gains.
The selected cohort has no negative raw-reward rows.

For arithmetic only, `raw_reward - max(0, lives_before - lives_after)` would
subtract120 points from the selected2,512-point total, yielding2,392, or99.667
per episode. This is not an evaluated policy or a new game score. Every selected
episode loses five lives, so this undiscounted total changes by aconstant five
points per episode in this cohort; timing, discounting and bootstrapped targets
could still change learning. No such learning effect is measured here.

The data come from an older greedy encoder4e5/behavior4e5/replay-value.3 policy,
not the latest replay-value1 learner. Per-decision net life counts do not expose
a simultaneous gain and loss within an action repeat. Teacher labels are not
used as reward targets in this analysis. No causal or reliability claim follows.

Eleven loss/gain/terminal/arithmetic/corrupt-input fixtures and14 pinned inputs
pass. Preparation takes.206s (.228s wrapper); analysis completes
**05:46:57.594 UTC November3**, sourcecbdf048, in **.301s** (.384s wrapper).
No native, emulator or array library is imported; no replay, learning, earlier
analysis or completed stage repeats. No failure occurs. Root
`/x/Temp/kindle-breakout-life-feedback-20261103` isclosed; do not repeat its
preparation or analysis. The retained70.750/143.667 scores andzero two-wall
outcomes stay unchanged.

The next selected development is an optional, default-off one-point training
penalty per observed lost life. Its [bounded protocol](../experiments/2026-11-03-cdp-breakout-life-penalty.md)
requires explicit raw-versus-training reward accounting, reset and bonus-life
handling, frozen evaluation with shaping off, and cost validation before any
full learner. Native learning stays unchanged onMeganeura/Blade. This is an
additional training aid, not the original environment reward or a proven fix.
No implementation, bounded GPU run or full learner has passed yet.

DeclarationSHA`21b6808f5a8fae4831c9d3a80af512f67370eaf90e628dbb8a4ebaf9efeea163`.
Review SHA `bf3b7bafdcf187db74fdf81dd07eab3d82200cbfbbbc8f8122cfc8836035451b`.
