# Breakout with a higher CDP encoder learning rate

The completed corrective200k learner scores4.292 versus its actual initial1.375,
below unaided full18's7.083; all have zero two-wall successes. Its weighted replay
expert cross-entropy averages1.000 over the final50k actions. That descriptive
training loss motivates an optimization trial, without proving an encoder bottleneck
or treating loss as task competence.

Select a fresh bounded8192-action/994-update check with encoder learning rate
6e-6→4e-5. Keep4M64/Auto/N8/B8/T16/H15/R16/coefficient1/actor-unimix.01/full18,
base/actor-critic rate4e-5,dynamics4e-4,warmup1000,20%episode guidance/cap1920/
expert10/all18 weights1/expert-world gradients/all-state corrective labels and
edge-reset feedback teacher. Seeds1009/eval4000001009 and the evaluation cohort
are reused development inputs. Native447fe50 and runtime remain unchanged.
This is one intended learner-setting change, not a proven causal explanation.

The optional fresh-CDP `--encoder-learning-rate` flag exposes existing native
configuration; omission preserves the old scaling. Nonpositive/nonfinite or
unrepresentable rates/ratios and non-CDP use are rejected before outputs or
environments. Restore rejects the override. Focused CPU checks pass348 tests,
10 native-construction tests excluded,zero construction attempts,in1.047s.
Historical CLI/test bytes are archived; completed experiment declarations and
runners remain unchanged. No native build or backend qualification is needed.

After8192 actions,restore for1024 greedy frozen actions with zero updates. Require
357 matched control initial tensors,357 finite/invariant frozen tensors,exact
config/source/counters,first update248/no residual debt,finite reports and changed
CNN/reward/actor/value. Audit expert row/weight accounting and aligned raw world/
actor losses,all-state labels,passive actions,complete label/input-pixel/action
replay and the existing AMD host guard. This short frozen check does not measure
natural-episode competence. Native max-pooled frames retain one GPU resize.

Training cap120s,frozen60s,CPU audit180s,owning service420s. Stop on real GPU faults
or nonfinite reports; no driver recovery,retry,cap increase,extension or automatic
promotion. Check experience will not enter any later learner. No full learner is
selected yet. Root`/x/Temp/kindle-breakout-encoder-rate-20261028`,service
`kindle-breakout-encoder-rate-check-20261028`. Live state was idle before preparation.

## Bounded check completed

Service starts22:22:31 UTC/source81a88d6 and succeeds22:23:22 on October28,
whole51.511s. Training8192 actions/32,722 frames/994 updates takes36.895s;
1024 frozen actions/4096 frames take.557s,zero updates/no natural episodes.
All42 pins,357 matched initial/invariant frozen tensors,finite reports,changed
CNN/reward/actor/value,complete guidance replay and host guards pass. No GPU stage
repeats. All994 reports are supervised;118,963 labeled rows equal the weight sum,
maximum raw world/actor expert-loss difference8e-7. Labels include2320NOOP,
1275FIRE and4597paddle actions;4296passive labels/3834disagreements. Actual
3896overrides are47.559% of actions. No competence or cap-release result follows.

Declaration`1505518f845389f3461f16a3960729d019517b3c567862ba5f2e4e92c57be852`,
review`cf346ed597289ab17190cd53a46ac1123c3e24042da68a484bcf4e50ca91cabb`,
replay`b5d795f4a7de04eb28a1dfadc19e3f6c70771556e3896e61595b2cfb01140b05`.
Early cost projects15m01s/200k; this is not a measured full-run time or speed gain.

## Separately selected fresh200k comparison

After the bounded service and all audits succeed, select one fresh200k-action/
24,970-update learner with exactly its declared settings. No bounded-check weights,
replay,games,belief or RNG are reused. Compare with the completed corrective200k
control at the same action/update budget. Encoder rate6e-6→4e-5 is the sole intended
learner-setting change; native/teacher/geometry stay fixed and the runner adds the
optional override. Reused1009 and the evaluation cohort remain development evidence.

Training cap1500s,100k recovery/12,470 updates,fixed final and own actual-initial
sampled evaluations,first3 natural episodes per8 lanes=24. Each frozen arm has
300s/600k-action caps,zero updates,guidance off,357 invariant tensors,no truncation,
full recorded-action/outcome replay/video and the unchanged90% two-wall gate.
Ordinary frozen logs lack live pixel hashes. Require357 matched control/check
initial tensors and paired initial semantic traces,finite reports,parameter changes,
source/config/counter audits,full training label/input-pixel/action replay,guards,
curves,exposure and completed comparison. No checkpoint selection,retry,cap increase,
extension or automatic successor. CPU audit1200s/whole main3360s caps remain fixed.

CPU preparation parses3 commands,compiles13 controllers,verifies42 bounded pins
and357 matched initial tensors in.126s without constructing an environment or
native agent. An initial copied-control-path error occurs before any full declaration
or GPU allocation; its failed script versions and receipt are preserved externally.
Correcting that path passes preparation without repeating GPU work. The launcher
now waits for a complete valid training header before starting the recovery observer;
six CPU readiness cases pass. Historical failed observer evidence remains intact.
Root`/x/Temp/kindle-breakout-encoder-rate-20261028/full`,service
`kindle-breakout-encoder-rate-200k-20261028`. Check live state before launch.

## Fresh200k launched and startup checked

Main starts22:25:10 UTC on October28/sourcecae0cf1,PID210609,
invocation`b46837dd8b0b430c90a5c4fc8a3ae4d9`. Declaration
`6a631febc2f09c67f26d47c5f3ddc95c998b5713f5f8b46d0800d5e084500bca`
pins64 inputs. Startup at5936 actions/711 finite reports verifies every pin,
exact config,357 initial tensors matching control and bounded check,and the
expected first updates248/256/264/272/280/288 with zero reported residual debt.
All supervision rows/weights match. The recovery observer starts22:25:14 after
the complete header; main,progress,recovery and postprocessing are active.
Startup review`584fcdb20c8b5ba10add8e0725112c84271b2b89e346ee9c5c1eefafc12b3f53`.
No frozen benefit has been measured and no additional allocation is selected.

## CPU comparison corrected while training continues

At22:28:43 UTC,read-only inspection finds a copied assertion in the future CPU
comparison still requiring the old config. It would reject the declared encoder
rate before producing a comparison. No comparison has executed. Preserve every
original input and replace only the waiting CPU postprocessor with a separately
staged version in`/x/Temp/kindle-breakout-encoder-rate-20261028/postprocess-repair`.
The main learner staysPID210609/invocation`b46837dd8b0b430c90a5c4fc8a3ae4d9`;
all64 declared inputs remain unchanged. No GPU stage is stopped or repeated.

Six CPU checks accept the actual bounded candidate config and reject the old
encoder rate or changed dynamics/base rates,replay ratio and actor mixture.
The corrected comparison changes the expected config only toencoder4e-5;
all other audit requirements remain. The original waiting observer is deliberately
stopped before comparison,with its journal and state retained. Replacement unit
`kindle-breakout-encoder-rate-postprocess-repair-20261028` starts22:28:43,
PID211059/invocation`8fd3c8100c0c403f8c1a99de5628e8bb`,with the original absolute
CPU deadline preserved. Training/evaluation caps and recovery observer are unchanged.

Preparation receipt`8a5e6b33edaf52efeae1b74867426657ef6da7fa1cacc2f53b6b62f1d100929c`
pins original and corrected CPU sources. Use the separate repair-directory
`collect_full.py` after terminal success; it additionally checks the replacement
and the recorded intentional stop. The original pinned collector remains untouched.
This repair constructs no agent or environment and supplies no learning result.
