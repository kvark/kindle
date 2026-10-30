# Transfer the moderate encoder rate to cheaper Qbert learning

The completed [Breakout encoder2e-5 comparison](../results/2026-10-30-cdp-breakout-encoder-moderate.md)
improves frozen gameplay at nearly unchanged training cost. Its subsequent
[state and actor diagnostic](../results/2026-10-30-cdp-encoder-representation.md)
also improves fixed representation and teacher-action readouts. Select one
bounded Qbert execution/cost check using that same fixed encoder rate. This
tests a candidate for another game, without claiming cross-game transfer of
the benefit or that representation error causes Qbert's later-game weakness.

Start from the retained 12M48/Auto/N8/B16/T16/H15/R16/expert100 Qbert recipe.
Its 200k learner scores greedy11984.375 and sampled6759.375 in30m27s, both24/24
first pyramids but below mean15000. H5/R32 is the stronger40-minute point,
passing the full gate on one of two learners. The purpose here is to improve
the cheaper point. Change only encoder learning rate6e-6→2e-5 as a learner
setting; retain base/actor4e-5, dynamics4e-4, warmup1000, coefficient1,
actor-unimix0, full18 actions, replay100k, microbatch16, all-state corrective
labels, expert100/all action weights1/world gradients, and20% episode
continuation guidance/cap1920. No auxiliary centroid/open-loop loss is enabled.

Use unchanged native447fe50 and the qualified optional encoder-rate runner
from the completed Breakout comparison. Preserve the control's Qbert teacher
and native max-pooled frames with one GPU resize to48. Record the exact runner
diff and source identities. No production code change or build is required.
Seed4013/evaluation4000004013 and the evaluation cohort are reused development
inputs. No weights, replay, games, live belief, RNG or prior experience is reused.

Train8192 aggregate actions/489 updates under180s, with first update376 and
residual credit.5. Restore for1024 greedy frozen actions under60s, guidance and
labels off, zero updates. Require357 initial tensors matching the retained
12M control, matching pre-update semantic events,357 finite checkpoints and
invariant frozen tensors, exact config/source/counters, finite reports,
changed CNN/reward/actor/value, equal labeled-row/weight sums, aligned raw
world/actor expert losses, complete training pixel/action/label replay and
both existing AMD host guards. The short frozen segment measures execution,
not natural-episode competence. Keep120s inner/150s outer replay,180s CPU audit
and480s whole-service caps.

CPU preflight exercises both actual CLI paths before output/environment/native
construction, the actual audit config assertion with negative cases, checkpoint
schema, update accounting and all190 retained control pins. Pin exact scripts,
runtime, protocol and control evidence before launch. Stop on real faults or
nonfinite reports; no driver recovery, GPU retry, cap increase or experience
reuse. Preserve any failed stages and repair CPU-only defects without repeating
completed GPU work.

The cost rule is `check_training_seconds * 200000 / 8192 * 1.10 <= 2100`.
Only after correctness and early cost pass, separately consider one fresh
200k/12,477-update comparison under35 minutes, with100k/6227 recovery and fixed
final/own-initial greedy and sampled24-episode assessments. Greedy stays primary,
the original mean15000/90%-first-pyramid gate stays unchanged, and both modes
require zero updates/truncations. No full learner, extension, rate sweep,
checkpoint selection, seed confirmation or automatic successor is allocated
by this bounded protocol. The five-game goal remains unfinished.

Root `/x/Temp/kindle-qbert-encoder-moderate-20261030/native-check`;
service `kindle-qbert-encoder-moderate-check-20261030`.

## CPU preparation passes

Preflight passes in0.317s: both actual CLI paths,13 actual audit-config cases,
all190 retained control pins,357 checkpoint-schema tensors and the8192/489,
100k/6227,200k/12,477 update schedule verify. No native/environment construction
occurs. The exact runner diff is retained: besides the optional encoder-rate
flag/validation/assignment, it includes previously qualified Breakout corrective-
label plumbing. For this Qbert command the Breakout flag stays false and the
shared corrective-label branch is selected by the unchanged Qbert flag. The
Qbert teacher and native bytes stay fixed; matched initial traces and full
label/pixel/action replay are required, without claiming complete cross-version
learning parity. No native rebuild or GPU stage has occurred at preparation.
