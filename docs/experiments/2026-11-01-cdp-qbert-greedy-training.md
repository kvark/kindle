# Bounded Qbert check with greedy actions during training

The [frozen training-history readout](../results/2026-11-01-cdp-qbert-training-readout.md)
finds stronger movement fit on guided paths than later autonomous states.
Wholly unguided training episodes contain only159 actions at age1920+, while
greedy frozen play lasts longer and scores14,028.125 versus11,522.917 sampled.
Select one bounded check of greedy action collection, retaining the existing
labels and losses. This may deepen experience or reduce useful diversity;
neither benefit nor a causal coverage explanation is established.

Add `--training-action-mode {sample,greedy}` to the vector runner. The omitted
option preserves existing sampling and log fields. Explicit modes are fresh
training only, recorded as `training_action_mode`; frozen evaluation keeps its
existing `--greedy` option. The audit rejects invalid modes and frozen/restored
use. All three native action-dispatch paths receive the selected mode, while
teacher overrides, passive labels and actual-action feedback remain intact.
No native model, teacher, loss or checkpoint schema changes.

Use the retained447fe50 native runtime and an isolated copy of the exact retained
runner with only the same parser/validation/header/dispatch additions. Current
source passes637 targeted CPU checks, including plain/exploration/corrective
dispatch, override enforcement, reset/replay behavior and frozen modes. The
isolated runner passes34 targeted checks. Preserve two earlier test-environment
failures: missing import paths, then old packaged audit modules with newer tests.
Correcting test paths changes no production source or GPU stage.

Fresh8192 actions must produce489 updates, first update376 and residual credit.5.
Then run1024 frozen greedy actions withzero updates. Keep12M48/Auto/N8/B16/T16/
H15/R16/encoder4e-5/base4e-5/dynamics4e-4/expert100,world expert gradients,
all action weights1,all-state corrective labels,20% episode guidance/cap1920,
actor-unimix0,full18,sticky.25,repeat4 and sampled categorical posteriors.
Seeds4013/4000004013 are reused development inputs; no prior weights or experience
enter the fresh learner. Only training action selection changes as a setting.

Require357 control-matching initial tensors and357 invariant frozen tensors,
finite reports and parameter changes, matching raw world/actor expert losses,
full training input-pixel/action/teacher-label replay and both host guards.
Initial pre-update trajectories deliberately differ from sampled control;
record those differences rather than requiring a false trace match.

The.624s CPU preflight validates both actual CLI paths,13 audit-config cases,
357 tensor shapes,the8192/100k/200k schedule and all276 control pins without
constructing an emulator or native model. Training/frozen/audit caps are
180s/60s/180s,service480s. The conservative cost rule is
`check_training_seconds * 200000 / 8192 * 1.10 <= 2100`.

Root `/x/Temp/kindle-qbert-greedy-training-20261101/native-check`,trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-check`.
No full learner,extension,retry,cap increase or automatic promotion is selected.
A later separately selected comparison would need fresh200k training,100k
recovery and final/actual-initial sampled/greedy24-episode evaluation against
14,028.125 primary greedy /11,522.917 sampled,with the unchanged15k/90% gate.
