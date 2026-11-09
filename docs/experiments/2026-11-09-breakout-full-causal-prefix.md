# Four generated history frames during B8 training

The [B8 comparison](../results/2026-11-09-breakout-denoising-batch8.md) improves
actual-history one-step prediction to87–90%, while ordinary rollouts remain
25–30%. Select one fresh controlled training-history test using that B8 recipe.
The earlier B32 latest-only causal-prefix experiment failed and stays closed;
this test replaces all four history frames after a longer clean warmup.
Root `/mnt/data/kindle-breakout-full-causal-prefix-20261109`.

Reuse exact native60b9b672/source8dbf9ab5 and matching reward32 runner, no build.
Config equals B8 exactly:RGB64,inference/trainB8,Auto,seed41001,learning rate.001,
context-motion100,residual/half-max noise,sigma_data.05,reward32,three Euler
stages. Fresh8,192 T5 updates,65,536 targets,224 distinct source arrivals,
588 positive and32,768 forced-max-noise exposures. Source-target and native
training Gaussian order stay exact. Initial,first-update/2036/4084 tensors,
six first losses and20 initial/2036/4084 model arrays must match cached B8.

First4,096 updates use actual histories. Atages4096,4352,...7936, freeze current
weights and generate a cache for targetsT=9..31. For each T, query four future
frames from actual observations strictly before originT−4, passing recorded
commands atT−4..T−1 and seed50001+32*age+T. Only causal prior RGB enters the
cache, never observed denoising or targetT.16 refreshes/368 native queries
produce11,776 imagined frames; they are not new emulator experience. Keep every
cache,age,origin andseed. First cache135 tensors before/after must match exactly;
all cache queries must preserve model age and local inference RNG must not alter
native training RNG.

After warmup, the last two micro-updates of each four-repetition source group
use four cached generated history frames when T>=9. The other two remain actual.
This yields1,678 augmented calls/13,424 target exposures. Each lane has equal
forced/nonforced noise counts within actual andgenerated groups. TargetT RGB,
actions,reward,terminal labels stay unchanged. Arrivals4..8 are too short for
this five-observation API and stay clean; both source positive events stay clean.
Caches are atmost255 updates old. Preserve float32 RGB without CPU resize.

This is detached training from four generated frames initialized four actions
earlier from actual history. It is not differentiable unrolling or supervision
of an uninterrupted16-step generated training trajectory. Residual anchors,
past-derived importance andevent conditioning change together. It may regress;
no training-recipe success, isolated cause or natural calibration is assumed.

Seven CPU check groups cover exact budgets/config,all four generated rows/clean
current target,unchanged labels,warmup/refresh boundaries,noise balance,causal
origin/action/seed joins,short-history/positive exceptions. Native qualification
is reused,not repeated. Retain all8,192 finite six-loss rows,all45 changed
parameters,checkpoints0/2036/4084/6132/8192 andall four ordinary16-step forecasts
plus108 actual-history one-step queries. Require135 own initial/frozen tensors
and20 restored model arrays exact. Audit every cache andaugmented source join.

Keep fixed24 gates: every supported ordinary1/4/8/16 ball within1px,
all16>=90% of110 per draw,one-step>=90% of198. Missing balls fail; unsupported
endpoints unscored. All1,376 endpoint,paddle/brick/pixel/motion/event regressions
andfixed lane0/draw0 frames retained. Four reused draws are one learner/clipset,
not independent seeds or held-out testing. No object/RAM labels enter training.

Pin immutable source/native/runner/data/scripts beforeGPU;new CPU repairs use
new filenames/declarations. One service,two serial guards,900s train/300s restore,
existing AMD host guard,no NVML/reset/reload. No new game action/frame,actor,
critic,CDP,build orautomatic successor. Full physics/events,natural calibration,
changed-brick memory,late/tunnel/second-wall/above-top,untouched families and
fresh confirmation remain required;no fitting score completes thefull goal.
