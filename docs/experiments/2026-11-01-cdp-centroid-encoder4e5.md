# Bounded position supervision with the stronger Breakout encoder rate

Prepare one fresh bounded check after the completed Qbert10007 closure.
All six original Qbert services succeed by10:55:36 UTC November1, with the
15k score gate missed and no stage repeated. No GPU work is active at this
protocol's preparation. Launch still requires CPU qualification and an idle
host. This protocol does not select a full learner.

The earlier scale100 centroid objective reduced position-readout error but did
not establish a gameplay gain; its greedy evaluation also recorded a timeout.
That comparison used encoder rate6e-6. The later4e-5 encoder recipe improves
Breakout much more, reaching sampled56.292/greedy104.667 in21m28s, although all
two-wall gates still fail. Tested response refitting and a settling-cost teacher
combination did not improve their respective screens. Keep the current teacher.

Test the existing centroid loss at scale100 with the stronger encoder rate4e-5.
Keep12M64/Auto/N8/B16/T16/H15/R16, base/actor4e-5, dynamics4e-4,
centeredCDP/action-effects1, sampled collection/posteriors, actor-unimix.01,
expert100/unit action weights/world gradients, all-state corrective labels,
20% guided episodes/cap1920, full18/sticky.25/repeat4 and one GPU resize from
native pooled frames. Reuse development learner1009/evaluation4000001009.
Start from scratch; no previous weights, replay, beliefs, games or experience.

Use qualified native2974160/SHAe7ba0d4 and the exact existing centroid runner
from `/x/Temp/kindle-color-centroid-20261029/full/atari_vector.py`, which already
supports `--encoder-learning-rate`. Runtime:
`/x/Temp/kindle-color-centroid-20261029/runtime`. No native rebuild or new loss
implementation. The head adds656,386 trainable parameters; total16,990,739 and
372 checkpoint tensors, including ten auxiliary Adam moments. Use the corrected
schema already checked against actual retained centroid tensors.

The detached target remains the normalized red-excess centroid in the fixed
source-coordinate region x[8,152),y[90,185), predicted from the posterior state
by the training-only head. No target or head is an inference input. Color/region
priors are explicit. The head and native/runner versions differ from the447fe50
control, so matching initial values and traces will not establish an isolated
loss effect or full default-learning parity.

Fix **8,192 training actions/489 updates**, first update376 and residual credit.5,
under180s. Then **1,024 greedy frozen actions**, zero updates, under60s.
Bound the CPU audit at180s and enclosing service at480s. Require all372 finite
initial tensors with matching retained centroid values;357 shared initial
tensors and the pre-update semantic trace must match the stronger no-centroid
control. Verify changed CNN/reward/actor/value and both centroid weight layers,
finite supervised/centroid reports, auxiliary mask/loss bounds, all372 tensors
invariant through frozen evaluation, full training pixel/action/label replay,
checkpoint/accounting checks and host guards. A short frozen run with no natural
episodes establishes execution only.

Use the fixed conservative cost screen
`training_seconds * 200000 / 8192 * 1.10 <= 1800`.
A pass permits considering a separately selected fresh200k comparison against
the retained4e-5 recipe, with100k recovery and final/actual-initial sampled/greedy
24-episode assessments. It does not select that learner automatically. Keep the
original90%-two-wall gate; no retries, cap increases, coefficient sweep,
checkpoint selection, automatic extension or check-experience reuse.

Before launch, qualify actual CLI/configuration paths without constructing a
native agent or emulator. Check schedule, full corrected schema, native/runtime/
runner/teacher hashes, control receipts and all source paths. Preserve the old
centroid numerical, schema and protocol failures; do not repeat their stages.

Root `/x/Temp/kindle-centroid-encoder4e5-20261101/native-check`, prospective unit
`kindle-breakout-centroid-encoder4e5-check-20261101`. At protocol preparation,
no preflight, declaration, queue or learner exists. Record each completed stage
once and retain all evidence outside Git. The five-game goal remains unfinished.
