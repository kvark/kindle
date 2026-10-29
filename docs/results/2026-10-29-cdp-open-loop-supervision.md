# Three-step world supervision passes numerical and early-cost checks

Optional prior-rollout supervision is implemented, default off. The fresh
Qbert12M bounded check completes **8,192 actions /32,754 emulator frames /489
updates in70.207s**. Frozen restore completes1,024 actions /4,096 frames in.792s,
with zero updates and no completed natural episodes. This establishes execution
and early cost; task benefit remains unmeasured.

The retained same-size short check took69.886s. The new check is.46% longer,
but different experience and execution times prevent an isolated overhead claim.
Linear200k projection is28m34s; adding10% gives **31m25s**, below the fixed35-minute
selection threshold. This is not a guarantee of full-run duration. No full
learner, extension or retry is allocated by this result.

## What changes

From the first posterior state in each replay chunk/row, execute three native
prior transitions using recorded actions without intervening observations.
Add detached centered CNN targets at steps2–3, and reward/continuation targets
at steps1–3. The existing deterministic CDP first-step term is not duplicated.
Use the existing straight-through categorical estimator with an independent
random stream. No trainable parameters are added.

Components average all rows, including masked zeros, and their supervised
horizons; existing component weights500/1/1 and new global scale1 apply. Masks
retain a valid terminal arrival, exclude reset/end crossings and remain zero
after a boundary. Targets and centering mean are detached. The initial
posterior/CNN, dynamics, prior and prediction heads receive gradients; no new
behavior loss is added. The original teacher-forced/expert losses remain intact.

The option is serialized only when enabled and exposed through fresh-only
`--open-loop-length` / `--open-loop-scale`. Checkpoints retain it; restore rejects
overrides. See the [exact protocol](../experiments/2026-10-29-cdp-open-loop-supervision.md).

## Validation and bounded results

Native source `ab80486` passes135 Rust CPU tests,611 packaged Python tests and
strict native/binding Clippy. CPU checks cover configuration/default compatibility,
RNG isolation, boundary masks, action ancestry, detached gradients, unchanged
parameter layout/inference and ordinary/staged graphs. CPU tests make zero
native construction attempts;10 native-construction cases are excluded.

Two separately guarded GPU checks pass: masked continuation values/derivatives
against F64; and rollout world-gradient routing, a smooth reward-head finite
difference, future-target/state invariance, action sensitivity and zero masks.
All189 declaration pins remain unchanged at completion. Service exits07:30:58
UTC October29, whole3.222s. Straight-through paths are checked for gradient
routing, not claimed to be derivatives of discrete samples.

Legacy frozen compatibility restores the retained trained12M Qbert model with
the option absent:1,024 actions in.823s,zero updates,357 unchanged tensors.
All128 transition rows match the retained greedy trace after excluding only
finite wall-clock timestamps. The original service exits07:36:27 at a CPU
comparison that mistakenly included `unix_time`; a separate CPU-only audit
passes07:37:39 with all29 pins unchanged and no GPU repetition.

The fresh check keeps12M48/Auto/N8/B16/T16/H15/R16,coefficient1,actor-unimix0,
full18,original rates,seed4013,20%episode continuation/cap1920/expert100/all18
weights1/world gradients/all-state corrective labels. Only the optional learning
loss is intended to change; native/runner revisions are explicitly different.
Evaluation uses4000004013 with greedy actions and guidance disabled.

All44 declared pins,357 matching actual-initial tensors,357 invariant frozen
tensors,config/source/counter checks,finite reports,CNN/reward/actor/value changes,
complete training pixel/action/label replay and both host guards pass. First
update occurs at376 actions; residual scheduler credit is.5. All489 reports
contain finite,nonzero auxiliary losses with audited weighted reductions.

| Auxiliary metric | Mean across489 updates | Minimum–maximum |
| --- | ---: | ---: |
| Scaled total |220.1032|42.3664–519.3414|
| CDP cosine distance |.431149|.076504–1.026136|
| Reward cross entropy |4.40201|2.16797–5.54126|
| Continuation BCE |.126851|.020558–.795089|
| Valid target fraction |.997912|.916667–1|

All489 expert reports remain supervised;117,261 labeled replay rows equal their
weight sum. Maximum raw world/actor expert-loss difference is8e−7. The episode
guidance rule actually overrides4,341 actions (52.99%);3,851 actions have passive
labels. Guided online returns are not frozen competence. Both learner and
evaluation seeds are reused development inputs.

Main service starts07:41:36/source`bade642` and succeeds **07:43:21 UTC October29**,
whole1m45.501s including construction,evaluation and CPU replay/audits. The
terminal collector passes with no repeated GPU stages. CPU preflight fixes for
a future checkpoint stub and premature empty output paths are preserved in
their original scripts/logs; they performed no GPU work. Earlier CPU compile
and interpreter-path errors are also retained. No training settings or caps change.

## Runtime and evidence

Root: `/x/Temp/kindle-open-loop-20261029`. Use its `runtime` directory and
`native-check/atari_vector.py`; the packaging receipt pins native,wheel,Python
sources and validation executable. Native SHA256:
`b4cdf26c5334ea7fde0e85a708809d5af11552e8de6f7a321d81e65437e2638a`.
The separate posterior-mode draft is retained as a patch,exact source copies and
hash receipt under `parked-posterior`,excluded from this runtime. Both disposable
build caches are released (2,959,631,566 file bytes); wheels,validation executable,
checkpoints and evidence remain.

- Gradient review: `6806137d0b7eb40d0d910259dfcf9524d4691f7d0f1fa55ff910b79f0cebeff8`.
- Compatibility review: `ffd2140d5fe62be0521d6327ba6962d35d1a75bb373f32a5a257ac85d6a146c7`.
- Bounded review: `3be34e68fe15eb9873c74ea55b18a56ba885dbc2afde6310fc31c9cacd79067b`.
- Completion: `f72896741831138dc3fc9cbf92906d15b21a471da1c5bf8a0611ac07f3f2be9e`.
- Packaging: `4dea42a0229db9d8e0faf1ea08511b3a0318ded210ce62ca5a3eb7bd977ecb6a`.

No GPU allocation is active at completion. The five-game goal remains unfinished;
a fresh full-budget comparison is needed to measure whether this loss helps.
