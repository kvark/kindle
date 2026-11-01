# Bounded Qbert RGB64 check with dynamics-group rate1e-4

Select one fresh execution and early-cost check, changing only dynamics-group
learning rate4e-4→1e-4 from retained RGB64 learner12011. Encoder/base/actor rates
stay4e-5. This tests learning balance without adding parameters or update work.
The rate covers recurrent dynamics, posterior, future-prediction and exploration
prefixes together; it is not an isolated recurrent-core intervention.

Current Qbert critics and action forecasts contain useful information, but
better scalar prediction error does not consistently identify better actions.
The [shared-history ranking](../results/2026-11-01-cdp-qbert-rgb64-ranking.md)
does not identify a missing-loss fix or prove a rate mechanism. The earlier
Breakout dynamics-rate comparison is mixed. This Qbert experiment tests a new
game-specific rate choice directly by frozen gameplay at a fixed training
budget, if the bounded check qualifies. It does not assume the diagnosis has
established the cause of the gap.

## Fixed bounded stages

Keep centeredCDP/12M/RGB64/Auto/N8/B16/T16/fullBPTT/microbatch16/H5/R24,
encoder/base4e-5,behavior unset,warmup1000,greedy collection,sampled posteriors,
actor-unimix0,action-effects1,replay100k,expert100/unit action weights/world
gradients/all-state corrective labels,20% continuation-guided episodes/cap1920.
Keep full18/sticky.25/repeat4,native pooled RGB andone GPU resize.

Reuse learner12011/evaluation4000012011 for a development comparison with its
retained control, not independent confirmation. Start from fresh initialization;
reuse no training weights, replay, belief, environment or RNG state. Native
447fe50/SHA35ad0db and fast teacherfa261c53 remain unchanged at
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`. No native build.

The retained Qbert runneraba4300d receives only the12 already-qualified lines
exposing `--dynamics-learning-rate`. Require exact added-line parity with the
production runner and no removed or other changed lines. The existing flag is
fresh-CDP-only, validates positive finite F32 rates/multipliers and rejects
restore overrides. Its completed471-test production qualification is reused
after source hashes match; do not repeat it merely for another private recipe.

Train8,192 aggregate actions/733 updates, first update376/residual.75, under180s.
Then freeze for1,024 greedy actions, guidance off,zero updates, under60s.
Audit180s/main480s. Require357 finite initial tensors matching the retained
same-seed control, identical pre-update semantic events,16,334,353 parameters,
733 finite reports, changed CNN/reward/actor/value, aligned actor/world expert
losses,357 invariant frozen tensors, full guided pixel/action/label replay and
host guards. The short frozen stage establishes no task competence.

CPU qualification must parse both actual commands before agent/emulator
construction, verify the unchanged schema and8192/100k/200k cadence, reject
undeclared configs, verify the12-line runner delta, runtime package and original
completed control/preceding ranking evidence. Pin all inputs and preserve
original process identities and any errors. Launch only from a clean committed
tree after the prior main/collector are terminal and the host is idle.

Conservative cost projection is `training_seconds*200000/8192*1.10 <=1800`.
Only a passing closed bounded result permits separately selecting a fresh200k/
18,715-update comparison with100k/9,340 recovery and four final/actual-initial
sampled/greedy24-episode assessments. Compare against12011's12,678.125 greedy
primary and12,093.750 sampled, each24 first pyramids. Keep the original15k/90%
gate and original failed confirmation. A rate improvement on this reused seed
would still need independent confirmation; no reliability claim follows.

Root `/x/Temp/kindle-qbert-dynamics1e4-20261101/native-check`; trial
`qbert-12m-64-b16-h5-r24-dynamics1e4-expert100-check-seed12011`; main service
`kindle-qbert-dynamics1e4-check-20261101`, with one original CPU terminal
collector. No full learner is selected here. No retry, cap increase, rate sweep,
extension, automatic successor or reuse of check experience. The five-game goal
remains unfinished.


## CPU qualification complete

Preflight succeeds22:24:44.716 UTC November1/source213589e in1.542s (1.573s
wrapper): two actual commands,22 config cases,357 unchanged schema tensors,
562 retained-control pins,16 runtime files and733/9340/18715 cadence pass.
The preceding ranking's64 pins and original main/collector verify. No native
module, agent or emulator is constructed. Parameter count stays16,334,353.

Private runner SHAe161bf38c213a10b4e2719730969c2d953748c092dc849dedaaaad7d55222cf4
adds exactly the12 qualified lines to Qbert runneraba4300d, with no removals or
other edits. Production SHAfa124388 and its retained471-test qualification are
unchanged. The GPU check remains unlaunched until separate guarded selection.
