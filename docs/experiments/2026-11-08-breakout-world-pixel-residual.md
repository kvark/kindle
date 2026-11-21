# Breakout: copy the latest image and learn a logit correction

The completed causal-prior learner fails retained physics and event gates.
At eight actions, 42/107 supported development balls are undetected, 38 more
are misplaced, and only 4/23 newly removed bricks are detected despite 99.765%
brick-cell accuracy. The existing spatial decoder rebuilds every RGB value from
learned global/history features. Source review finds no new alignment mismatch.
These observations motivate a persistence path, not proof that its absence
causes the failures.

Select optional `rgb_spatial_residual`, default off, in the standalone fused
spatial world with explicit `motion_event_rollout`, positive H16 and no phase.
Add the latest causal image's log odds to the decoder's existing output logits:
`logits = correction + log(p) - log(1-p)`, where
`p = clamp(latest_RGB + .5, 1/255, 254/255)`. A zero correction copies the latest
image to within one intensity level. The correction must learn motion, new and
removed content. Keep gradients through generated image history, including the
copy path; target images never enter that path. Reset/clipped history remains
the existing centered-zero history. No new parameters, heads, labels, collector,
encoder, replay geometry, decoder convolution or backend primitive.

Use the v13 suffix-event recipe as the comparator, not the failed v14 main-pass
causal objective. Retain posterior observed RGB/reward/continuation and KL,
H16 free RGB/reward/continuation losses, two-image history and full BPTT without
generated-image re-encoding. This changes output initialization, parameter
interpretation and gradient routes together. It is not an isolated memory test.
Preserve v1-v14/default semantics. Bind the new decoder in v15 and reject old
headers/new mode or incompatible objectives before GPU construction.

CPU checks: copy values/clamp, latest-versus-older history and lane isolation,
F64 finite differences through correction and generated history, production
decoder ancestry/parameter layout, causal training/rollout path and header/scope
rejection. Reuse existing fusion, sampling, event-loss and motion-weight math;
do not repeat completed native primitive fixtures.

After build/package review, separately declare one 900s guarded qualification:
new copy/correction values and derivatives versus F64, actual decoder copy
under zero correction and lane isolation; one full-shape update, all13 reports
finite with exact total/valid fraction, required changed parameter families;
all240 parent actual-initial tensors and16 cached non-image forecast arrays
match v13. RGB differs by construction and is never asserted parent-identical.
Check240 own frozen tensors and18 own initial/trained restored arrays, plus
old-native v15 rejection with GPU discovery disabled. No old-model query.

Only after reviewed qualification separately declare fresh8,192 updates from0
and final restore. RGB64/12M/B8/T32/H16/full BPTT/Auto, seed41001, rates4e-4,
mixing.01, main/future RGB100 and reward/continuation1, KL1/.1/free-nat1;
80 weights/8,871,955 scalars/240 tensors. Reuse the original full4/2/1/1 plan,
16 fitting/eight development streams and frozen evaluation locations. No new
emulator actions/frames. Retain actual initial/2048/4096/final and dense384
initial/final origins. Reused seed/families are development, not confirmation.

After reviewed learning/restore separately declare only the new final's84 frozen
groups, histories16 primary/32 secondary, eight draws, horizons1/4/8/16/32.
Reuse cached oracle/H0/fusion/suffix-event arrays and every original gate:
observed>=90%, one>=89/109, eight>=28/107, primary category-balanced
survival-weighted ball energy<=.12712666431732942, serve NOOP false balls<=10%,
ending32 Brier<=.10, ending16<=.01, RIGHT16 ending<=.10, non-ending category
means<=5%, and freeflight8 presence>=90% for each command. Retain individual
alarms and all per-action/horizon/paddle/reward/brick regressions. No best
checkpoint/draw, easier gate, unchanged extension, sweep or automatic successor.

Full all-horizon physics, event timing/calibration, persistent brick memory,
late/tunnel/second-wall coverage, untouched families and fresh confirmation still
govern the goal. The current oracle has no above-top-brick endpoints. Pixel
persistence or intermediate passes alone cannot complete the goal.

Root `/mnt/data/kindle-breakout-world-pixel-residual-20261108`. AMD/Auto/host
guard; stop on real faults or nonfinite state, no recovery/NVML. No GPU job is
allocated by this design declaration. Earlier learners/collectors stay closed.
