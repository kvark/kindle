# Breakout: motion-weighted world model fits eight contact trajectories

The fixed final checkpoint **passes the declared fitting diagnostic**. All four draws predict all110 supported future ball positions within one resized pixel across actions1..16. Observed reconstructions and all supported1/4/8/16 endpoints also pass on every draw. Eighteen of128 future positions per draw are outside this detector's support and remain untested. This proves precise fitting on these eight trajectories, not a general Breakout world model. The full goal remains unproven.

[Protocol](../experiments/2026-11-08-breakout-world-contact-fit.md).
Root `/mnt/data/kindle-breakout-world-contact-fit-20261108`.

## What changes and what this establishes

The preceding broad-data image-feedback run fails physical/event gates. Its fitting-stream errors leave open whether the architecture can fit motion at all. Early small-clip fitting used uniform squared RGB error; later motion-weighted H16 training used broad data. This experiment combines the current qualified motion objective with the exact original eight fixed contact clips. It is a distinct data/objective combination, with no old learner or collector repeated.

Use the existing native CNN/categorical RSSM/RGB decoder/reward/continuation world only. No actor or critic learns. Fresh2,048 updates, RGB64/size12M/B8/T32/full BPTT/Auto, seed41001, rates4e-4, categorical mixing.01, observed/future H16 motion RGB cross-entropy100, KL1/.1/free-nat1 and observed reward/continuation1.68 logical/8,852,675 scalar weights/204 checkpoint tensors. No spatial decoder, image feedback, pretrained encoder, new backend primitive or native rebuild.

Fitting starts `[43,27,43,43,28,49,43,67]` match the original contact selection and manifest exactly, one clip from each of eight independent source environments.32 arrivals each,16 observed before causal rollout. All old source RGB/actions/episode alignment and cached context/future images match. RAM geometry chooses contact candidates but never enters model inputs or losses. The vertical-reversal predicate is not an exact collision oracle. Native max pooling and the saved single GPU RGB64 resize stay.

The batch has256 unique source arrivals;2,048 repeats expose524,288 replayed arrivals and262,144 future targets, not new experience. There are two positive observed-prefix rewards and zero future reward/terminal positives per update. No collection, new emulator actions/frames or event-calibration claim. Predictions receive past16 images and recorded future actions, without future image corrections. Fixed full trajectories can be memorized; no causal-rule or action-counterfactual generalization follows.

All204 fresh initial tensors match both the early MSE contact learner and the broad motion-H16 control. The early MSE comparison changes loss formulation and native runtime together; no isolated effect of motion weighting or CE follows. The broad-data comparison also changes data diversity and update budget. Native identities are kept separate.

## Fixed final precision

Within one resized pixel; missing predicted balls fail. The same below-brick detector and support apply to both cached old and new trajectories. Draws are fixed83001/83003/83009/83023, not independent learners. No best checkpoint or draw replaces final2,048.

| Model / draw | Observed | 1 | 4 | 8 | 16 | All1..16 supported |
|---|---:|---:|---:|---:|---:|---:|
| Earlier uniform MSE H16 / 0 |6/8|6/8|4/6|5/8|2/8|58/110|
| Earlier uniform MSE H16 / 1 |6/8|6/8|4/6|5/8|2/8|57/110|
| Earlier uniform MSE H16 / 2 |6/8|6/8|4/6|5/8|2/8|60/110|
| Earlier uniform MSE H16 / 3 |6/8|6/8|4/6|4/8|2/8|58/110|
| Motion CE H16 / 0 |8/8|8/8|6/6|8/8|8/8|110/110|
| Motion CE H16 / 1 |8/8|8/8|6/6|8/8|8/8|110/110|
| Motion CE H16 / 2 |8/8|8/8|6/6|8/8|8/8|110/110|
| Motion CE H16 / 3 |8/8|8/8|6/6|8/8|8/8|110/110|

Actual initial observed/1/4/8/16 counts, each fixed draw:

- Draw0: 0/8, 0/8, 0/6, 0/8, 0/8.
- Draw1: 0/8, 0/8, 0/6, 0/8, 0/8.
- Draw2: 0/8, 0/8, 0/6, 0/8, 0/8.
- Draw3: 0/8, 0/8, 0/6, 0/8, 0/8.

All24 fitting gates pass:every supported observed/1/4/8/16 ball on each draw and>=90% over all1..16 supported positions per draw. Actual all1..16 is110/110 on each draw. All endpoint errors/detections/unsupported cases, paddle metrics and brick additions/removals remain in `review.json`; raw RGB/features/event heads are retained. Ball precision alone does not qualify all image details or persistent brick memory.

Observed-prefix reconstruction uses real images for the ordinary world objective. The additional prior-image term and frozen causal predictions receive only the observed-prefix state and planned actions; future RGB is a stopped target. No object position target, event label or future reset corrects an imagined state.

## Events, scope and next implication

Future event results concern only negatives. At16 actions, mean modeled ending risk is 0.000853602, ending Brier 0.000000821, positive-reward Brier 0.000000000022 and reward MAE 0.000002310. Built-in discount is removed before survival products. These low errors on zero-positive support do not establish reward or ending calibration.

The existing architecture/objective can fit these local trajectories accurately. Its poor broad-data forecasts therefore coexist with demonstrated fitting capacity. The next useful question is how to retain that precision across diverse histories and action choices. This does not distinguish memorization, data balance, optimization interference or representation/transition generalization, and does not isolate a cause of earlier failures.

No original full-corpus gate is replaced by this easier support. Precise all-horizon causal physics, persistent brick memory, calibrated rewards/life/game-over timing, late/tunnel/second-wall coverage, untouched families and a fresh learner confirmation still govern the full goal. There is no32-action assessment here. No unchanged extension, old-stage repeat or automatic successor is selected.

## Execution and preserved launcher failure

Repository/source snapshot `5837ed9d8a4079765006d21ef8e0809bf378c221`; exact experimental runner/config are pinned in the48-input original declaration and57-input unstarted-stage repair. Reuse native/source `85dbe150aed1e87a42a3089ee612fc6ef2d88f86`, SHA256 `5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`. Existing motion-loss/derivative/mask/restore qualification is reused without rebuilding or repeating fixtures.

CPU preparation checks all eight source hashes, exact selection/context/future arrays, incoming-action alignment, episode/reset boundaries, all204 common initial tensors and declared native identity without constructing a world/emulator. The first launcher omitted `gpu_guard.py`; guard import failed before native-worker construction, with zero updates and no train output. Preserve the failed service, original queue/protocol and journal. This first launch remains failed.

A separate CPU import-closure check with GPU loader disabled passes after adding unchanged `gpu_guard.py` and `atari.py`. A new57-pin declaration allocates only still-unstarted train/restore jobs under a different service. No completed native stage is repeated. Both corrected native stages finish; two host guards/18 sealed files pass, no native numerical failure or host recovery. AMD/Auto throughout; no NVML/reset/reload/reboot.

All2,048 finite eleven-loss rows, full target mask/total identity,68 changed logical weights,204 initial/frozen tensors, unchanged optimizer age during forecasts and32 exact restored draw-arrays pass. CPU review verifies110 inputs in1.454s. Initial/512/1024/2048 and final frozen/restore checkpoints remain. Checkpoints omit full training RNG, live belief and replay; a restored extension would be a warm start, not an independent learner.

Native training calls 186.289s; main loop/checkpoint/evaluation 191.500s. Initial construction and actual-initial exports are outside this loop. The early MSE contact learner used186.404s native/216.029s loop at the same2,048 updates; distinct runtime/wrapper observations do not isolate a speed effect. Worker/restore spans are retained separately in guard events.

The comparison plot uses four fixed draws and all declared checkpoints, with mean/min–max rather than confidence intervals. The APNG shows all16 actions at fixed lane0/draw0, actual native frame versus cached early MSE versus new motion prediction. Display resizing never enters model inputs/metrics; all decoded clip frames match exactly. Analysis/figures/clips require no new model or emulator query. Source, runner, useful weights, failures and complete arrays remain outside Git.
