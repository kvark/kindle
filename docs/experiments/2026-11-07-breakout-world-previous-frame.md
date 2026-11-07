# Breakout: previous-frame input to the pixel world encoder

The retained H0 world reconstructs108/112 balls but tracks14/107 at eight steps.
The [H4 image learner](../results/2026-11-07-breakout-world-short-rollout.md)
also fails fitting forecasts:9/176 at eight,0/171 at sixteen. Completed H16,
latent,energy,freezing and observation-only code candidates remain closed and
negative. This does not identify a unique representation/optimizer cause.

Select optional `rgb_previous_frame` for an isolated H0 world. The recurrent
model already receives observed history; adjacent-frame input adds no information
to the full model. It gives the encoder a direct causal motion path instead of
requiring it to recover displacement only through stochastic recurrent history.
This is a new bounded architectural hypothesis, not a proved remedy.

Root `/mnt/data/kindle-breakout-world-previous-frame-20261107`. The earlier
[whole-agent CDP frame-pair test](../results/2026-11-03-cdp-rgb-frame-pair.md)
remains negative:114/39.208 versus143.667/70.750 with13.13% more time. Do not
repeat its learner, collector or six-channel convolution primitive qualification.
Its replay and supervision differ from this isolated pixel world.

## Architecture and invariants

Default off. Keep public replay and model arguments as three-channel RGB,
native max-pooled frames and one GPU resize. For each observed arrival, native
Rust selects the same lane's preceding resized image; at clipped starts and
reset actions(-1), duplicate current RGB. No prior-episode or future image.
Concatenate previous/current CHW images on GPU before the encoder. Only the
first convolution widens3→6,adding2,400 weights at12M. Keep encoder output1024,
RSSM feature width2560, categorical slots/classes, decoder RGB3 and all heads.
The68 logical parameters become8,855,075 scalar weights/204 checkpoint tensors.

Observed training uses the paired encoder; future priors receive actions and
latent state only, with no image argument. The decoder reconstructs current RGB
only. Motion weighting and all target shapes stay unchanged. New initial values
and predictions must be retained; shared seed is not matched architecture.
Version5 persists the flag and all-weight scope; reject disguised older formats
and reject v5 before GPU loading in old natives. Ordinary restore never changes
geometry/objective. The legacy frozen-visual API rejects this architecture.

No object/RAM/phase labels, pretrained encoder, actor/critic, new loss or backend
primitive. Native frame selection/copy adds host work; measure total time rather
than claiming a free speedup. Retained whole-agent settings stay unchanged.

## Qualification and bounded allocation

CPU checks: exact CHW/lane/reset/clip-start selection, no future dependence,
concatenation values/derivatives, encoder ancestry restricted to previous/current
images, image-free prior transition, only first-convolution geometry changed,
all world-family gradient connectivity, config/default/header rejection and
legacy behavior. Run affected Rust/Python checks and strict core/binding Clippy.
Preserve first formatting/compilation failures and subsequent repairs separately.
Build/package exact source/native/runner bytes, then release disposable cache
after qualification outputs are retained; no new backend qualification project.

Separately declare one full-shape native composition preflight(900s). Require
204 exact own-initial restored tensors,18 initial/trained restored arrays,finite
nine-report H0 update/total,65 expected first-update changed parameters,204 frozen
tensors and persisted v5 flag. At fixed current image, perturb previous image:
encoder features must change on a continuing lane, while a reset makes all
posterior/prior outputs identical. Verify lane independence. A new-native legacy
compatibility query must match18 cached arrays; do not rerun its learning or
assessment. Old native must reject v5 before construction. Reuse old convolution/
concat/resize primitive evidence, without repeating their native fixtures.

After reviewed qualification,separately declare one fresh8192-update H0 learner
(3600s) and restore(600s). Retain original RGB64/12M/B8/T32/full BPTT/Auto,mixing.01,
seed41001,rates4e-4,observed motion RGB100,KL1/.1/free-nat1 and reward/continuation1.
Use the same sixteen fitting/eight development streams and complete4/2/1/1 plan.
Verify all8192 locations/non-suffix exposures; no collector or new emulator actions/
frames. Retain initial/2048/4096/final checkpoints,original periodic cohorts and
256 fitting/128 development matched dense-initial/final controls. Check every
finite report,204 actual initial/frozen tensors and18 final restore arrays.
Dense-initial precedes the existing loop timer; report worker span separately.

After learning/restore review,separately declare new-final-only84-group oracle
assessment(1800s),history16 primary/32 secondary,eight draws,all1/4/8/16/32 horizons.
Reuse original oracle and retained controls. No old-model assessment or collection
repeat,chosen checkpoint,best-match draw,unchanged extension,sweep or automatic
successor. Use AMD/Auto and current host guard; stop faults/nonfinite state,no
reset/recovery/NVML. Single actor/learner allocation only.

Keep observed>=90%,one-step>=89/109,eight-step>=28/107,primary16/32 energy
<=.12712666431732942,NOOP serve false-ball<=10%,ending32 Brier<=.10,ending16
Brier<=.01,RIGHT16 ending<=.10,non-ending category-mean ending<=5%,and freeflight8
ball presence>=90% on every plan. Report every individual false alarm and all
action/horizon/reward/paddle/brick regressions. These repeatedly used families are
development,not fresh confirmation. Full causal physics,event calibration,brick
memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation
remain required; passing intermediate gates does not complete the goal.

## Actual qualification and allocation

All60 affected Rust/89 packaged Python checks and core/binding Clippy pass.
Preserve first float-syntax/import compilation and flat-weight-shape fixture
failures separately; repairs change no learning result. Native source/runner
70da4788,SHA96adefd2. First456-pin composition check completes in17.391s with
204 own-initial restored tensors,18 initial/trained restore arrays,18 exact legacy
arrays,65 expected changed parameters,204 frozen tensors,previous-frame influence,
lane independence,reset isolation and old-native v5 rejection. Build cache release
retains exact wheel/runtime/CPU executable and releases834,734,615 counted bytes.
The separately declared466-pin fresh8192 H0 learner/restore and918-pin new-final
84-group assessment are complete. The [closed result](../results/2026-11-07-breakout-world-previous-frame.md)
records all unchanged gates; full competence remains unproven. No GPU allocation,
unchanged extension,repeat or automatic successor remains selected. Source members
are copied byte-exactly from the pinned Git archive; original declarations remain
unchanged. Final-only branches execute after reviewed learning/restore completion and remain frozen.
