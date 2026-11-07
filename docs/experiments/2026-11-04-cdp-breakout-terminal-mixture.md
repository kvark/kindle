# Bounded Breakout check with terminal-containing replay sequences

The retained learner sees terminal targets in only .1683% of replay rows;
76.352% of its final-window updates contain none. A fixed held-out readout finds
usable ending information in saved states, but that balanced cohort does not
establish calibrated probabilities. Increasing the whole continuation loss to10
lowers frozen gameplay. These results motivate a different, bounded development
test of replay exposure within the unfinished five-game goal.

Implement an optional native `terminal_replay_fraction`, absent by default.
Select .25 for the first bounded check. Fresh sequences keep their existing
priority. For each remaining sequence, choose with probability .25 uniformly
among all valid starts with a true terminal in the training rows; otherwise
use the existing uniform replay choice. Fall back to uniform when no eligible
sequence exists. Merge overlapping start ranges so sequences containing several
endings receive no multiplicity bonus. Prefix-only terminals, intermediate life
losses and timeouts do not qualify. Eviction removes expired terminal indices.

This changes the data distribution for every training objective, including
imitation and actor/critic targets. There is no importance correction; calibrated
ordinary-play continuation probabilities are not promised. Keep continuation
scale1. No new network, target definition, loss, parameter or per-update GPU
operation is introduced. Optional reports expose fresh/uniform/terminal route
counts, eligible starts and actual terminal training rows. Audit the row count
against the independently calculated continuation target mean.

## Qualification before GPU work

CPU tests must compare eligible ranges with exhaustive window enumeration over
multiple streams, contexts and evictions; test fresh priority, timeout exclusion,
uniform eligible-window sampling, the .25 mixture and exact samples/RNG for the
disabled/no-candidate paths. Validate configuration serialization, CLI propagation,
old-runtime rejection and malformed report detection. Run the Rust CPU suite,
strict Clippy and packaged Python adapter checks. Preserve any failure and its
correction. Native/emulator construction is forbidden during Python preparation.

Package a new native binary from the declared committed source, with pinned
wheel, source and runtime receipts. The retained447fe50 binary is unchanged.
Current source also includes earlier optional features; do not assume broad
numerical parity from CPU checks. Before candidate learning, require one bounded
default-off frozen compatibility capture using the retained checkpoint and
recorded evaluation prefix: 1,024 actions, zero updates, 357 invariant tensors,
matching semantic actions/outcomes and guards. No completed capture is repeated.

## Fixed bounded learner

Use a fresh Breakout12M/RGB64/Auto/N8/B16/T16/full-BPTT/micro16/H15/R16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, replay-value1, continuation1,
expert100/unit weights/world gradients, action-effects1, replay100k/context1,
sampled collection/posteriors, actor-unimix.01, full18/sticky.25/repeat4/noops0.
Preserve the native pooled images and one GPU resize, existing pixel teacher,
all-state labels and20% episode guidance/cap1920. Reuse development seed1009;
no experience or weights transfer into training.

Train8,192 aggregate actions/489 updates, first update376, residual credit.5,
with a180-second training cap. Retain actual-initial/final checkpoints; require
357 initial tensors and47 pre-update semantic events to match the retained
control. Then capture1,024 frozen greedy actions, seed4000001009, zero updates,
guidance/labels off,60-second cap and357 invariant tensors. Short frozen returns
are not a competence assessment. Require finite reports, correct weighted world
loss, actual terminal-route use, changed CNN/reward/continuation/actor/critic,
full training pixel/action/label replay and host guards. Record actual terminal
exposure and route counts, rather than assuming .25 of all batch rows are endings.
CPU audit cap180s; service cap600s. Do not restart successful stages or raise caps.

Conservative200k projection is bounded training seconds ×200000/8192×1.10;
require≤1800s before considering a separately selected full learner. A full
comparison would keep200k/12,477/30min,100k/6,227 recovery and four actual-initial/
final sampled/greedy24-natural-episode assessments against retained70.750/143.667
in21m27.910s, all0/24 two-wall. Judge frozen improvement per training minute and
the original864/90%-two-wall gate. No full learner, sweep, seed campaign, extension
or automatic successor is selected here.

Root: `/x/Temp/kindle-terminal-replay-20261104`. Development and CPU qualification
are in progress. GPU launch requires the package and actual queue/audit preflight.
