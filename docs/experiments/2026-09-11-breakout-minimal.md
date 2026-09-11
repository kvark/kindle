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

## Action-width state inspector — completed CPU preparation

The existing historical state helper binds its logical layout to eighteen
actions, so it cannot certify the new four-action saves. Keep that pinned helper
unchanged. The separate [CPU result](../../runs/breakout-action-state-20260911.lOcNzn/result.json)
provides a strict inspector for the future runtime gate, with **88 passing tests
and 53 evidence pins**. It accepts only the declared 12M recipes and these three
production topology changes:

| Parameter | Published shape | Minimal shape |
| --- | --- | --- |
| RSSM `world.dynamics.core.dynin2.weight` | `[18, 256]` | `[4, 256]` |
| Actor `behavior.actor.out.weight` | `[256, 18]` | `[256, 4]` |
| Actor `behavior.actor.out.bias` | `[18]` | `[4]` |

Their six existing optimizer moments follow those shapes. All **241 entries**
(95 parameter entries and 146 moments) remain mandatory; every other name,
shape, dtype and optimizer membership stays fixed. This removes only **7,182
parameter elements / 86,184 logical state bytes** including moments, not measured
VRAM. Action simplification is a learning ablation, not a presumed memory or
throughput fix.

The inspector exactly agrees with the qualified checker on three real
eighteen-action checkpoints: synthetic updates 1 and 8, and the N6 pixel save at
610 updates. It hashes and checks every tensor, complete metadata and actual
encoder identity. The real first-update save contains **122 nonzero optimizer
moment tensors**; it is correctly rejected as zero-update state. Initial state
requires all moments zero and the normalizer's actual native initial `(0, 0)`.

Clearly named, CPU-generated four/eighteen-action files exercise full-size I/O,
including all 92 same-shape initial parameter entries and 140 same-shape moments.
These are **synthetic fixtures, not native initializations or restores**. Wrong
action schemas, actual encoder files, torn payloads, old backend identity and
update-one-as-zero state are rejected. Frozen-state checks retain weights,
moments, normalizer, config, encoder and stream count, allowing only the declared
action-counter increment.

The one-core / 2 GiB / zero-swap CPU scope peaks at about **377.5 MiB host memory**.
All 1,870 waiting hardware pins reverify before and after. No source, package,
historical helper, checkpoint or queue changes; no GPU or production ML graph
executes. Preserve the completed preparation and generated fixture labels. A
future pixel declaration must bind this inspector to actual native captures and
complete ledger/replay/memory evidence **after** the queued hardware gate; there
is still no native four-action restore qualification or pixel follower.

## Capture harness — completed CPU preparation

The [isolated capture library](../../runs/breakout-pixel-capture-20260911.dJZZaz/result.json)
passes **71 CPU tests**, with **184 evidence pins**. It intercepts construction
or restore, saves immediately before the original runner's loop, returns the
actual object to that unchanged loop and saves again after normal return.
Fresh training therefore captures zero-action/zero-update initialization, not
the already-updated first learner state. Python bindings and arguments are
restored on exceptions. Reused paths, symlink aliases and overlap with the source
checkpoint are rejected.

Its request contract is limited to short N6/R256/B16/T64/full-BPTT64 12M Breakout
fixtures with four or eighteen actions: 3,840-action fresh training and fixed
6/768/18,000-action frozen checks, including the existing one-episode-per-stream
stopping/cap fixture. These are **runtime test budgets, not final game-evaluation
budgets**. No exploration, greedy policy or restored-training overrides are added.
Actual learner updates must come from the complete reset-dependent ledger, not
the fake objects' counters. The library explicitly requires separate complete
state and ledger audits; it cannot certify episode completion from counters alone.

The lifecycle tests use fake objects, not native agents. Separate fresh-process
checks verify real `0591eda`/`abf4ae5d` package/wrapper/runner imports and native
default configuration for both action widths. The historical editable `f663dd93`
package is actually imported and rejected as the wrong package, without changing
it. Library import loads no Kindle/native module; direct execution refuses because
there is **no declared GPU entrypoint**. The initial fake-factory recursion failure
is preserved in development notes and was corrected only in the CPU fixture.

The enforced one-core / 2 GiB / zero-swap scope peaks near **50.8 MiB host memory**;
all 1,870 waiting hardware pins reverify. No native agent, GPU graph or pixel
worker starts. A future runtime declaration must compare captured and uncaptured
native controls for exact state/report/trace parity: saves outside the loop are
not presumed neutral. Native initialization/restore, frozen v4 prefixes, full
replay and combined-memory qualification remain required after the queued
hardware gate. There is no additional follower or learning result.
