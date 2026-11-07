# Breakout: fixed observation filter and forecast-state distillation

The [state substitutions](../results/2026-11-07-breakout-world-state-swap.md)
repair many forecasts with image-informed h. The [recurrent energy learner](../results/2026-11-07-breakout-world-recurrent-energy.md)
and [frozen visual phase](../results/2026-11-07-breakout-world-frozen-visual.md)
fail retention; both retain changing recurrent targets or conditioning. The
[previous-frame encoder](../results/2026-11-07-breakout-world-previous-frame.md)
also fails, despite accurate observed reconstruction. None isolates a cause.

Select a distinct bounded warm phase: freeze the whole retained H0 observation
filter, including its recurrent core, and train a separate forecast transition
in that fixed coordinate system. Keep the visual weights fixed. This combines
warm initialization, a filter/predictor split, fixed model targets and two
existing rollout losses; do not claim an isolated target-drift effect.

Root `/mnt/data/kindle-breakout-world-fixed-teacher-20261107`. Source checkpoint
is the retained coverage H0 final at8192 updates. No completed learner or
collector repeats. No object/RAM labels, pretrained external encoder, actor/
critic or new backend primitive. Teacher inference is not a second learner.

## Fixed functions and causal targets

New explicit `start_fixed_teacher_phase` API, objective `motion_fixed_teacher`,
H16, all-weight source H0 only. Preserve all204 student weights/moments and
optimizer age. Keep the RGB64/12M/B8/T32/full BPTT/Auto configuration, mixing.01,
seed41001,rates/scales and original data. Previous-frame and observation-only
architectures are not selected. Ordinary restore accepts no phase overrides.

The observation session is an independent, immutable copy of the source filter;
never share or synchronize its dynamics with the student. Prediction observes
only the given prefix using this fixed filter, then rolls the learned forecast
transition forward under submitted actions. Decoder weights stay fixed; reward/
continuation heads can learn. Student visual36 weights stay exact; their72
optimizer moments may change. The32 dynamics/head weights remain trainable.

For learning, the fixed filter observes the same32-arrival batch. Independent
per-lane sampling seeds depend only on the declared model seed and optimizer age,
never images or actions. Target draws vary during training; frozen means fixed
weights/coordinate functions, not an unchanged realization. Conditioned on the
same explicit seeds, targets must stay exact after student updates and restore.
Permute lane seeds with their source windows. Prefix targets must remain exact
when only future images/actions change. Reset masks and first clipped starts
match H0. Sampling streams are reproducible at a saved age, but ordinary restore
still does not reproduce all learner RNG or uninterrupted learning.
Teacher targets may use future images; no future teacher state enters a predicted
state. The student suffix seed is the fixed filter's state at observed index15.

At each observed arrival, one-step student priors condition on preceding fixed
teacher h/z and match the current fixed posterior. Current RGB and reward/
continuation readouts use fixed teacher h/z. From the causal seed, use existing
H16 categorical overshooting KL(scale1) and two-sample recurrent-state energy
(scale100), with stopped teacher targets. Keep base RGB100,KL1/.1 and heads1.
No free-nat floor is added to auxiliary KL. Separate prior draws are retained for
the two objectives; categorical straight-through derivatives remain biased.
No physical-coordinate loss or new analytic loss formula is introduced.

Average over full Bx16; mask the first future reset and subsequent rows, retaining
terminal arrivals. Evaluate unchanged1/4/8/16/32 futures. Teacher h/z are privileged
training targets, not successful forecasts. Internal coordinate consistency is
not physical competence or a calibrated distribution score.

The extra frozen filter adds stored parameters, GPU memory, forward passes and
host transfers. Report its parameter count/bytes and measured cost separately;
student count68/8,852,675/204 is not the total system footprint. Include inherited
H0 cost in learning cost. This is a warm start, not uninterrupted or independent
training. Checkpoints embed the immutable teacher file and SHA, fixed phase scope/
start age and objective in v6; old natives reject v6 before construction. Verify
teacher GPU weights against the anchor on save; restore validates its SHA before
GPU initialization. Preserve legacy/default/v3/v4/v5 behavior.

## Qualification and allocation

Run affected CPU values/derivatives, stopped-target/reset/ancestry, independent teacher-
seed causality, phase/header/digest, parameter geometry/gradient and packaged
Python tests, plus strict core/binding Clippy. Preserve failures and separate
repairs. No unchanged convolution/GRU/resize primitive qualification repeats.

Separately declare one native composed preflight(900s). Require204 exact inherited
tensors and18 cached initial arrays, fixed observed filter/visual outputs,
repeatable teacher targets before/after two updates and restore, prefix invariance
under changed suffix images, lane permutation equivariance, finite14 reports/
loss/mask identities, expected trainable families and exact36 visual weights.
Require18 trained restore arrays,teacher file/hash/GPU weights invariant,and
old-native v6 rejection before construction. Ordinary new construction must reject
the fixed-teacher objective without an anchor.

Only after reviewed qualification,separately declare4096 new updates(global12288)
(3600s) and final restore(600s), first4096 rows of the exact original4/2/1/1 plan.
Retain initial/intermediate/final checkpoints and teacher bytes, original periodic
cohorts,256 fitting/128 development matched dense-initial/final controls, complete
finite losses/exposures,204 frozen inference tensors,36 exact visual weights and
18 restore arrays. Check all observed RGB/encoder/posterior features remain equal
to actual initial. Include source H0 plus new phase costs and all timing scopes.
No new emulator actions/frames or data collection.

After reviewed learning/restore,separately declare only the new final checkpoint's
84 frozen oracle groups(1800s),history16 primary/32 secondary,eight draws,all
horizons. Reuse oracle/control arrays; no old-model assessment or collection
repeat. Keep observed>=90%,one>=89/109,eight>=28/107,primary energy<=.12712666431732942,
NOOP serve false-ball<=10%,ending32 Brier<=.10,ending16 Brier<=.01,RIGHT16 ending
<=.10,non-ending category-mean ending<=5%,and freeflight8 presence>=90% each plan.
Report every individual false alarm and all action/reward/paddle/brick regressions.

Use AMD/Auto/host guard,stop real faults/nonfinite state,no reset/recovery/NVML.
No unchanged extension,sweep,concurrent learner or automatic successor. Full
physical forecasts,event calibration,brick memory,late/tunnel/second-wall data,
untouched families and fresh confirmation remain required before goal completion.

## Construction failure and separate loader repair

The initial f19e3fec/nativea74334ab qualification stopped in construction before
updates or evaluations: the complete checkpoint's logical parameter map does
not equal the subset observation plan. Preserve its failed guard and all bytes
under the original root. No world learning result exists for that package.

The separate repair first loads/validates the full anchor against the complete
training plan, then copies matching weights through existing native transfer
into an independently allocated inference session and verifies it bitwise against
the anchor. No shared parameter allocation or backend change. Ordinary restore
subsequently loads current student state. Extra constructor loading/copying is
included in worker time. Root
`/mnt/data/kindle-breakout-world-fixed-teacher-loader-20261107`; reuse the exact
prepared data/plan, requalify the changed constructor with new pinned bytes,
and only then separately declare the original4096-update budget. This does not
repeat a learner. Original failed qualification remains failed.
