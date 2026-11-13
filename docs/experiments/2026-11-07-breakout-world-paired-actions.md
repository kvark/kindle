# Breakout: fitting-only paired action branches

The [action-gap audit](../results/2026-11-07-breakout-world-action-gap.md) finds
unwanted action sensitivity in the motion-rollout model: on eight freeflight
histories with identical oracle ball paths across four plans, modeled ball
presence at8 is100% under NOOP but64.06% under RIGHT. Existing fixed-command
training support is nonzero but unpaired and mostly outside that freeflight
slice. Select a data intervention, not another loss or sampling setting.

Root `/mnt/data/kindle-breakout-world-paired-actions-20261107`. Keep the existing
qualified native5b112866/runtime from the motion-rollout experiment. No native
rebuild, backend optimization, actor/critic, evaluation-origin reuse or collector
rerun. New branches are new fitting experience from retained fitting parents.

## Parent preparation and collection

Choose exactly eight parents in each of serve/freeflight/contact/brick/wall/
life_loss/ending from the original16 fitting streams only. Use the existing
source geometry predicates and32 observed arrivals within one episode; no future
outcome or model error selects parents. Selection seed651001 is a data-selection
seed, not a new emulator/learner seed. Balance original/supplement sources where
available, require eight source streams per category, unique native32-frame
histories and64-arrival spacing within a source episode across categories.
Preserve all selected indices, source seeds/hashes, incoming commands, rewards,
terminal/life/position references and native/RGB64 prefixes. No development or
oracle parent enters fitting. RAM/phase labels select data only, never model input
or loss. They remain imperfect analysis references.

Reconstruct every parent using its original seed and recorded commands, including
real episode resets. Validate source native images and exact selected prefix
hashes before snapshot capture. Reuse the qualified ALE snapshot/restore helper:
replay reconstructs the private sticky-action state omitted by serialization.
Never substitute a blind restore. Record source reconstruction and restore replay
actions/frames separately from new stored branch experience and reset counts.

Use eight independent emulator instances with eight distinct parents per category.
For each parent, execute four fixed32-command plans and two sticky RNG draws.
Non-serving plans are NOOP0,LEFT4,RIGHT3,and RIGHT4 decisions then LEFT28. Serving
plans are NOOP0,FIRE1,LEFTFIRE12,RIGHTFIRE11. Full18/sticky.25/repeat4/noops0 and
native max pooling stay fixed. Stop each branch at natural game-over, retaining
terminal arrival; do not invent reset or unavailable future images.

Maximum new experience14,336 actions/57,344 frames. Additional source/replay
emulator work is not new independent experience; precompute its bound from the
selected transcripts and cap it at1.1M actions/4.4M frames. One guarded collection
allocation,1800s cap, no automatic retry. Record actual counts/times and failures.
Use the existing AMD host guard/Auto; stop real faults, no reset/reload/reboot.

Restore the retained age0 H0 world only as a qualified camera. Batch eight native
pooled frames through its single GPU resize; preserve RGB64 replay. Zero world
predictions/updates and204 invariant weights/moments before/after. Match each
reconstructed prefix's saved RGB64 values. No CPU resize enters model data.

Store32 shared-prefix arrivals plus each branch's valid future, with actual
incoming-action alignment and labels. Prefix duplication across four plans/two
draws is not new action/frame count, independent data or a natural reset. Keep
actual source action/reward/terminal values; the normal training context cut
supplies the first belief reset. Hash every dataset/snapshot/config/source output.
Collection qualifies only after counts, alignment, restores, prefixes, masks,
guards and unchanged camera tensors pass. Late-game coverage remains unproved.

## Separately declared learning check

Only after collection review, implement/pin a fixed mixed sampler and one fresh
8,192-update learner(3600s), restore(600s) and new-checkpoint-only oracle assessment
(1800s), through separate declarations. Retain the motion-rollout objective,H16,
RGB64/12M/B8/T32/Auto,seed41001,rates4e-4,loss scales and matching native. Data and
sampling change; volume, coverage and within-batch correlations change together,
so this is not an isolated causal-confounding estimate or fresh confirmation.

Each batch: four paired plans from one fitting parent/repeat index/window offset,
two original-fit uniform windows, one supplement waiting window and one terminal
window. Repeat0 shares the parent RNG across plans; repeat1 uses evolving RNG
separately per plan, so its index is not a common-random-number or independence
claim. Choose the common paired offset within all four valid sequences; half
target the intervention origin where possible, half sample available offsets.
Short terminal branches shift the common cut earlier, preserving every terminal
and never padding fictitious data. Fix the exact sampling plan/counts before GPU
learning. Keep original16-stream fitting and original development assessments
unchanged for comparison; report paired-fitting diagnostics separately.

No native primitive reruns or rebuild. New data/runner CPU tests and one full-shape
native preflight(900s) verify initial204 tensors/18 arrays, eleven finite losses,
mask/total identities, frozen204 tensors and18 restored arrays. Preserve every
failure separately. New runner source and unchanged native identity must both be
pinned; never claim a rebuilt native from a later Python-only commit.

Retain all existing gates: observed>=90%,one-step>=89/109,eight-step>=28/107,
primary16/32 ball energy<=.12712666431732942,NOOP8 false balls<=10%,ending16 Brier
<=.01,RIGHT16 ending<=.10,ending32 Brier<=.10,non-ending group mean ending32<=5%.
Additionally require>=90% modeled ball presence in each of the four retained
freeflight plans at8; report action contrasts/position errors without converting
their internal units to a new competence claim. All horizons/categories/events/
paddle/bricks remain visible. No invalid FIRE90% matching gate, unchanged extension,
scalar/seed sweep or automatic successor. Passing remains intermediate: full-field
physics,calibrated events,brick memory,late/tunnel/second-wall coverage,untouched
families and fresh confirmation are still required for the goal.

## Launcher failure and unstarted-stage repair

The first launcher resolved the virtual-environment interpreter symlink and
invoked the base Python without NumPy. It failed before collector import, so
no emulator or native world was constructed and the output directory was never
created. Preserve its declaration, service log and failed guard. The separate
`repair_launcher.py` CPU check uses the virtual-environment path and verifies
NumPy, the packaged native binding and collector imports without constructing
an emulator/world. A new declaration pins those original failures and the
unchanged collector/native sources; `unstarted-collection-queue.json` allocates
only the still-unstarted first collection. Do not relabel the original launch
as passed or repeat any collected stage.

## Fixed sampler and runner implementation

Each update selects one of 112 parent/repeat groups uniformly. Its first four
lanes use the four plan streams at a common start in `[0, min(future lengths)]`.
Exactly half the 8,192 updates target start `min(16, limit)`; the other half use
uniform valid starts. Original-data lanes sample the original eight streams.
Waiting windows use the existing supplement pool. The terminal pool combines
existing original/supplement windows and each paired terminal's final complete
32-arrival window; cycle a fixed seeded permutation of that pool. This keeps
longer-plan terminal labels available when a short member limits the paired cut.
Declare actual terminal exposure; pool membership alone is not a learning result.

The Python runner adds an optional `evaluation_fitting` argument to the corpus
trainer. Its default retains previous behavior; this experiment explicitly uses
the original sixteen streams for periodic and dense fitting evaluation. Assert
both original fitting/development selections against the retained control before
training. A separate post-training diagnostic evaluates all 448 paired origins,
one fixed model draw, conditioning on the last sixteen source-prefix images.
Unavailable tails stay unscored under the qualified existing evaluator. Save and
verify 204 frozen tensors after that diagnostic. It is fitting evidence, not a
new held-out success gate or gameplay evaluation. Its time is reported separately
from the established training/evaluation loop timer.

The implementation changes Python data loading/sampling and evaluation routing
only. Native source `85dbe150` / SHA256 `5b112866` is reused unchanged. New CPU
sampler tests cover short branches, exact first-prefix actions, common cuts,
terminal availability, actual exposure counts and invalid groups/reset/terminal/
geometry inputs. Existing corpus/coverage tests also pass. Qualify one mixed
full-shape update with 204 matched initial tensors, eighteen initial/restore
arrays, eleven finite losses and exact suffix-mask/total identities before a
separately declared fresh learner. Do not rerun native primitive suites.
