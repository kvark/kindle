# Breakout: fixed posterior-only local world learner

Select one fresh16,384-update learner after the [v20 implementation and native
qualification](2026-11-08-breakout-world-local-h0.md). Root
`/mnt/data/kindle-breakout-world-local-h0-20261108`.

The [loss review](../results/2026-11-08-breakout-world-prior-pixel-loss-review.md)
motivates removing added sampled-prior RGB and imagined-event losses. Also select
observed RGB scale1 rather than100, informed by the [earlier gradient balance](../results/2026-11-08-breakout-world-loss-gradients.md).
Those gradients came from another checkpoint/12M recipe; they do not measure this
1M learner or prove a useful coefficient. Earlier RGB1/H16 failure remains relevant.
This combined recipe changes objectives,data/sampler and loss balance; no isolated
causal claim. No other coefficient,checkpoint,seed or automatic successor is selected.

Use native/source/runner5bcaaebd, SHA6930e7b8, v20, size1M/RGB64,
603,915 weights/80 logical/240 tensors, B8/T32/full BPTT32, H0,
`motion_cross_entropy`, Auto, seed41001, rates4e-4, RGB1, KL1/.1/free-nat1,
observed reward/continuation1, mixing.01. Same CNN/RSSM/local-action decoder and
inference; no new primitive or actor/critic/CDP/object/RAM target. Posterior images
condition observed reconstruction; inference uses prior samples and past pixels.

The qualifier used RGB100 and is preserved unchanged:645 pins, finite9-report
first update,77 changed weights/all required families,240 finite/frozen tensors,
18 exact arrays at initial and trained restores,13.878s. Both240 initial tensors
and18 initial forecast arrays match the retained small64 initial exactly. One
successful nine-file guard and independent36-pin CPU review pass. No qualifier,
old model or completed stage is repeated. RGB scale is an existing exposed scalar,
with unchanged shapes/operators; it needs no rebuild. Keep a separate learning
config so the pinned qualifier/config remain intact.

Before allocating,pin every source/runtime/data/config/plan/receipt. Use the
prepared corrective4,096-row plan repeatedfour times in one learner. Reuse sealed
counts scaledfourfold,not a generic old-plan recount:496 streams/62,666 arrivals,
57,215 unique,20,604 observed terminals,44,564 positive-reward rows,18,268 source
resets and148,988 context resets. No imagined targets. All288 launch branches and
retained contact/life-loss/ending/natural/wait/terminal sources remain as declared.
Replayed examples and reused development seed/families are not fresh evidence.

Require the new fresh initial240 tensors to equal the qualified initial. Check18
actual-initial forecast arrays against the qualifier before training,then reuse
the sealed initial dense/curve controls without querying the old model. The first
actual update counts toward16,384. Require nine finite reports,correct RGB1
weighted total,unchanged unweighted forward components versus the qualifier
within1e-6 relative/absolute,all240 finite saved tensors/age1,and changes in all
required families. Stop on any failure before another update. This one state
check is included in the learner-call timer; it is not an isolated GPU benchmark.

Use one guarded queue, fresh initialization and continuous RNG/optimizer age
through16,384. Save2048/4096/final and curves every2048. Preserve all failures;
no unchanged extension,restarts,best-checkpoint/draw selection or seed sweep.
Final256 fitting/128 development origins use the same original16-stream natural
selection,plus frozen tensors and18-array restore checks. Review all finite
reports/ages/plan/counts/source identity and terminal guard/service state first.

Then separately declare the fixed final-only84-group chosen-action assessment:
42 plans,histories16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints.
Reuse cached controls/oracle outcomes,never old forecasts. Keep every regression,
event/brick/paddle/ball score and individual alarm. Inspect complete sampled
images alongside distribution scores; a centroid of multiple ghost fragments
does not establish coherent physics. Native pooling can legitimately show two
adjacent historical positions; do not invent a one-connected-component gate.

All existing physical/event gates stand. Even passing them leaves precise
full-game physics,event timing/calibration,persistent removed-brick memory,
late/tunnel/second-wall/above-top coverage,untouched families and fresh learner
confirmation required. No actor/critic restart or world-goal completion follows
from lower loss,sharper samples or this one development learner.
