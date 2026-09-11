# Minimal-action Breakout: staged action-width qualification

The candidate at `exp/breakout-minimal` (`0591eda`) prepares the
[bounded comparison](2026-09-11-breakout-diagnostic.md#next-bounded-comparison)
without changing an active experiment. Its source-matched package passes
**620 Python CPU tests**. It has **no four-action GPU qualification, learning
result or adopted action protocol**. A separately declared hardware/synthetic
diagnostic now waits after the existing learning queue; it starts no learning.

The current Qbert → Freeway → Pong queue remains first. Boxing remains the
only game with completed fresh three-root confirmation; this work adds no wins.

## What changed

The existing `published-minimal` collection option already exposes Breakout's
NOOP/FIRE/RIGHT/LEFT vocabulary. The candidate does not change the collector,
world/behavior learning arithmetic, replay accounting or frozen episode stopping.
It adds three checks around that existing option:

- Explicit minimal-action Breakout declarations produce task replay **v2**.
  They must name `published-minimal`, action count 4 and the exact ordered
  vocabulary. Other games are not implicitly enabled. Published full-action
  replay remains **v1**, including historical declarations without the new fields.
- CPU reconstruction uses the verified action space when constructing ALE.
  The scorer rejects a replay version belonging to the other action protocol;
  all existing source, ROM, action, reward, reset and frame checks remain.
- Complete checkpoint verification requires an integer action count matching
  its schema, in addition to every existing tensor, optimizer, identity and
  final-save check. An eighteen-action schema cannot certify a four-action save.

The two-wall / 864-point Breakout gate is unchanged. Neither a replay task pass
nor these tests certifies campaign budgets, independent training roots or learning.

## Backend and package identity

Source: `/x/Code/.kindle-breakout-minimal`, commit
`0591eda794fa211c689dda8bd25ccfb4acd71d7e`, based on `24b2968`.

Only the four dependency/identity edits from `1e00e818` are carried into native
source: both lockfiles, `kindle/Cargo.toml` and `kindle/src/dreamer/mod.rs`.
All tracked native/build inputs match that qualified source exactly. Meganeura
is `ce80e9cd6056c230590b8b7e1eb9ffe9bbce08bc`, with its upstream LeVJEPA cache fixes.

The [fresh bundle](../../runs/breakout-minimal-package-20260911.fwfepW/package)
copies the qualified `abf4ae5d` native bytes and this candidate's Python modules.
Every imported module is checked against those sources in a fresh process.
No native compilation, GPU construction or historical editable-package overwrite
was needed. This is a source-matched bundle, not a newly built wheel.

**Both future learning arms must use this same new-backend bundle.** The old
`f6a2b6ad` Breakout pilot is historical context, not the matched eighteen-action
arm. Existing queues keep their own pinned packages. Prior upstream eighteen-action
[runtime qualification](2026-09-11-meganeura-runtime.md) does not qualify the
four-action graph or this composite package's frozen episode protocol.

## Completed CPU evidence

The [result](../../runs/breakout-minimal-package-20260911.fwfepW/result.json)
binds 102 inputs and outputs. All 580 inherited tests and 40 new cases pass,
with zero failures, errors or skips. All 1,622 active scheduler pins reverify
before and after; no queue input is changed.

New cases reject missing declarations, reordered actions, wrong counts/types,
changed preprocessing or identities, cross-protocol replay versions and
mismatched checkpoint schemas. Real ALE fixtures exercise both full and minimal
replay, including natural boundaries, independent resets and partial tails:
512 vector ticks × two streams × two protocols = **2,048 actual wrapper actions**.
The CLI tests deliberately stub the complete-run reader with CPU-generated data;
they are replay integration fixtures, not native training/evaluation certificates.
The unchanged ledger reader is covered separately by the inherited suite.

The earlier [physical mapping check](../../runs/breakout-minimal-cpu-20260911.NcHj9r/result.json)
remains complementary evidence: identical images/rewards/RAM/frames for the common
four-action subset, plus a wrong-left/right negative control. Neither check proves
that all eighteen action combinations are equivalent to four actions.

## Required before learning

After the existing queue, use a separately pinned runtime declaration:

1. Qualify four-action production-size losses and **all parameter gradients**,
   action/reset causality and full-recurrence behavior without relaxing tolerances.
   The existing production world-gradient fixture and synthetic canary hardcode
   eighteen actions; their old outputs are not four-action evidence. Prepare
   isolated fixtures without replacing those historical executables.
2. Verify zero-update and trained four-action saves/restores, complete finite
   tensors and optimizer moments, actual encoder identity, and rejection of a
   mismatched action schema. Record state from update 1 as well as later updates.
3. Check repeated four-action state/report/trace equality, unchanged default
   eighteen-action package behavior, frozen v4 stopping/prefixes and full CPU
   replays. Exact equality is required within controls, **not** between differently
   shaped learning arms. Verify same-name/same-shape initial parameters across
   arms; initialization is keyed by parameter name and seed, not a global draw order.
4. Measure complete N6/B16/T64/R256 learner-plus-perception memory and timing,
   with ≥2,048 MiB directly free. Fewer actions are not an assumed speedup.

Only then declare the paired learning pilot: same backend, encoder, budget,
replay ratio, reward and evaluation recipe, changing the action vocabulary alone.
Retain all final unassisted episodes and separately restored same-seed untrained
controls. Four actions necessarily change the actor output and RSSM action-input
widths; do not describe the comparison as merely renaming action indices.
Any provisional winner still needs a separately declared fresh three-root gate.

Preserve the completed CPU package and its pins. The bounded hardware/synthetic
declaration below covers only part of these requirements and changes no active
five-game campaign declaration.

## Prepared hardware fixtures — CPU checks only

The separate `exp/breakout-minimal-gates` worktree, commit
`18c7ffb751646ac42455aa44c777572ef5cf6337`, now provides the missing fixtures.
It does not edit the frozen `0591eda` source/package above. All changes are in
`cfg(test)` code and the synthetic canary example; production world/behavior
code, other native/build inputs, Python and backend identity stay unchanged.

- The existing temporal batching comparison has an explicit four-action
  **production B16/T64/full-recurrence** test. The eighteen-action/default and
  tiny controls retain their arithmetic, data, reset masks and tolerances.
- New actor/value comparisons cover **15,360 imagined rows / 1,008 replay rows**,
  not just a 1,024-row actor-layer probe. Both eighteen- and four-action fixtures
  compare the full row-independent graph against sixteen disjoint row slices.
  They verify matching initialization, all six loss/entropy outputs and every
  parameter gradient. The reference averages slice results in F64; it never
  partitions the RSSM or changes the deployed learner. Signed advantages,
  zero/nonuniform weights and a nonzero test value head exercise the gradient paths.
- The isolated canary accepts `--actions 4`, retaining eighteen by default.
  Existing `--updates 1` / `--updates 8` and `--checkpoint` options can capture
  full optimizer state from the first update. The two invalid-count CLI checks
  exit before GPU initialization. No positive four-action canary has run yet.

These are compiled test capabilities, **not passing GPU comparisons**. The
world fixture retains loss/gradient tolerances `3e-4` / `3e-3` at production size;
the new behavior fixture uses those respective bounds too, plus the same `1e-7`
gradient absolute allowance. Preserve a failure rather than weakening bounds
to obtain acceptance. Numerical agreement still needs actual hardware evidence,
complete state/restore/pixel gates and combined learner-plus-perception headroom.

The [completed CPU preparation](../../runs/breakout-minimal-fixtures-complete-20260911.dv2SqC/result.json)
passes **98 Rust workspace tests**, formatting and Clippy with warnings denied.
The suite counts are 81 library, 5 environment and 12 example tests; all 25 GPU
tests remain ignored. Exact listing confirms the three new GPU fixtures are in
the pinned executable. The synthetic canary also compiles. One build job ran
under an enforced one-core / 2 GiB host-memory / zero-swap scope, whose measured
peak was approximately **990 MiB**. This is host memory, not a VRAM measurement.
No production-size ML graph was compiled or executed during the live trainer.

The [initial preparation](../../runs/breakout-minimal-fixtures-20260911.EaDRLF/declaration.json)
successfully compiled and passed all 81 library tests, then stopped on an
incorrect assertion expecting the workspace-wide count of 98. Preserve that
failure and its complete logs. The separate completion verified the raw prefix,
ran the actual workspace/all-target suites and finished the remaining checks;
no learning/backend source fix or GPU retry was involved.

All **134 completion pins**, the 102 package pins, 1,622 active scheduler pins and
original qualified executables reverify. Preserve both artifact roots. The
compiled fixture identities are recorded in the completion result; do not use
stale root release binaries or replace historical executables. There is still
**no paired learning declaration**. Qbert → Freeway → Pong remains first, with
the later runtime gates required before adoption.

## Declared hardware/synthetic stage — waiting, not qualified

The new [declaration](../../runs/breakout-minimal-hardware-20260911.PLAL8H/manifest.json)
binds **1,870 pins** and **92 passing CPU checks**. Its
[live-parent refusal](../../runs/breakout-minimal-hardware-20260911.PLAL8H/refusal.json)
captures an actual CLI exit before hardware outputs, device queries or GPU work.
The [one-shot follower](../../runs/breakout-minimal-hardware-20260911.PLAL8H/launch.json)
started at **18:58:29 UTC on September 11** and is waiting for scheduler PID
42730 / start ticks 1021056. Its own verified process is PID 52404 / start ticks
1665628. Preserve this queue; do not manually launch the diagnostic or restart it.

The stage has exactly **16 native phases**, serialized after complete Qbert,
Freeway and Pong results:

1. Four production gradient comparisons: world and full actor/value heads for
   each of the eighteen- and four-action vocabularies. Force the full world
   fixture, preserve every gradient check and keep the declared tolerances.
2. Six complete synthetic state pairs: update 1 and two eight-update repeats per
   vocabulary, with AB/BA order for the latter. The eighteen-action pairs use the
   original qualified executable against the new fixture executable; four-action
   pairs repeat the new executable independently. Also retain the qualified
   eighteen-action checkpoint/report anchor. Require every logical tensor and
   optimizer moment to match exactly within each pair, plus reports except timing.

The new read-only whole-queue prerequisite auditor includes Pong's **24 commands**
(12 native phases plus replays, scoring and world-recording extraction), actual
source-matched v4 ledgers, complete checkpoints, paired frozen controls, replay/video
bindings, selected first-four-match recordings and raw GPU coverage. It recursively
rechecks Freeway, the breadth continuation and recovered-backend qualification.
Valid competence failures are retained, not mistaken for incomplete results;
complete CPU wrapper tests are **not** completed Pong evidence or native world
forecasts. The fixture-development list/tuple assertion failure is preserved in
the new root's development notes; no native result was involved.

The follower polls only the bound process while it is live, waits at most 14 days,
and allows at most 12 hours for this one diagnostic. Any changed pin/host,
incomplete prerequisite, native failure or unsafe/incomplete memory window stops
it without retry. Every phase requires ≥2,048 MiB **directly reported free**,
250 ms sampling and no gap above 1.5 seconds. No GPU phase has run yet.

Even a complete pass will leave four-action **zero-update initialization and
restore, native N6 pixel traces/v4 replay, and combined learner-plus-perception
memory** unqualified. Update 1 is not zero-update state: optimizer moments already
change. These remaining checks need a separate declaration before any action-width
learning pilot. There is no speedup, new Breakout win or seed-reliability claim.
