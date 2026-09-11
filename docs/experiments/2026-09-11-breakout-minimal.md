# Minimal-action Breakout: isolated CPU candidate

The candidate at `exp/breakout-minimal` (`0591eda`) prepares the
[bounded comparison](2026-09-11-breakout-diagnostic.md#next-bounded-comparison)
without changing an active experiment. Its source-matched package passes
**620 Python CPU tests**. It has **no four-action GPU qualification, learning
result, automatic follower or adopted action protocol**.

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

Preserve the completed CPU package and its pins. This report neither launches
those gates nor changes the active five-game campaign declarations.
