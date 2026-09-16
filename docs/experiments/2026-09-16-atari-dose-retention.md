# Checkpoint support for the Freeway/Qbert exposure comparisons

The [standalone snapshot helper](../../python/examples/retain_atari_checkpoint.py)
recovers the tested Qbert retention code and adds the existing Freeway .5/hold64
contract. It copies a completed save and its exact training-log prefix without
constructing, pausing, restoring or restarting an agent. This is artifact handling,
not CPU learning or a replacement native runtime. Main carries the identical
helper/tests from source `4032063`; active Pong inputs stay unchanged.

## Verified now

The [completed preparation](../../runs/atari-dose-retention-check-20260916.teQLiJ/capture/result.json)
passes **753 Python tests**, including 51 retention cases, using actual qualified
native `886bae68`. All **32 native/build inputs**, six packaged modules and the
acting/training runner are unchanged from `8dc0b98`; no native binary is rebuilt.
The scope is one core, 2 GiB and zero swap. Ten complete command records and
**5,214 input pins** reverify, including the active Pong pins before and after.

The [main-source check](../../runs/atari-dose-main-check-20260916.hfoYkE/capture/result.json)
then verifies exact native/Python source equality, actual main helper imports and
**753 passing main Python tests** on the same package. Preserve that completed
one-shot writer too; its generic evidence audit is read-only.

Real completed Qbert seed 0 and Freeway seed 1009 checkpoints are retained
byte-for-byte in separately labelled historical archives. Both remain their
original **200,004 actions / 49,651 updates**, old native/backend and outcomes.
Each archive checks the full prefix, all **241 tensors / 95 parameters / 146
optimizer moments**, actual encoder bytes and finite values. No log header,
checkpoint metadata, score or original artifact is rewritten.

The real current-runtime 3,840-action / 610-update checkpoint also passes the
complete-state inspection against its qualified same-backend schema. An old-
backend schema and the different .25-probability Freeway runtime fixture are
rejected. This is file-format/state-reader evidence, not new GPU execution.

Tests cover wrong counters/headers, missing or corrupt parameters/moments,
changed encoders, stale or moving saves, existing destinations and broken
exploration histories. Qbert cannot acquire Freeway's assistance; Freeway cannot
silently change its probability or persistence. A failed copy leaves an explicitly
unqualified archive without a completion marker or retry. A retained prefix still
fails the ordinary complete-run reader with `missing run_end`, as it should.

The independent audit passes. Preserve the completed writer and both historical
archives; only its `audit`/actual-import readers are reusable. No NVML, agent,
GPU job, follower or 400k learning experiment starts in this preparation.

## Retired incremental midpoint observer

The [preserved observer](https://github.com/kvark/kindle/blob/462702b21f3a39dc933b0ee5c51a088f0b43db39/python/examples/observe_atari_checkpoint.py) carried the
earlier Qbert incremental reader onto this shared helper, without its historical
source or recipe bindings. Source `462702b` changes only that module and its
tests. It binds an explicitly supplied child command, parent and process-start
identity; partial log writes wait, while changed headers, missed saves, replaced/
truncated logs, child exit or a deadline fail without retry. It never launches,
signals, pauses or restarts a process.

The [isolated preparation](../../runs/atari-midpoint-observer-check-20260916.eYU7Cx/capture/result.json)
and its read-only audit pass: **812 Python tests**, including 59 observer cases,
all 32 unchanged native/build inputs and 176 input pins. Main carries the exact
source and also passes [all 812 tests](../../runs/atari-midpoint-main-check-20260916.O4CMVU/tests.xml),
with actual main-module and unchanged native-package imports checked before/after.
Both checks use one CPU, 2 GiB and zero swap; no native binary is rebuilt. The
process fixture is a harmless stdin-waiting child and its checkpoint state is
explicitly synthetic—not a new GPU integration result. Active Pong pins remain
unchanged. Preserve both completed checks and the isolated source.

Its design required a caller to propagate observer failure through the direct-child
guard. That integration was never implemented or attached to Pong. The simpler
runner-owned approach below removes this requirement. Main retires the unused
module and its tests (412 lines); source `462702b`, commit `027e49f` and all
completed evidence remain recoverable. The [remaining main suite](../../runs/atari-observer-retirement-20260916.DK4f6a/tests.xml)
passes 843 tests with the runner/native bytes unchanged.

## Complete-study artifact reader

The [result reader](../../python/examples/audit_atari_dose.py), isolated at
`f467014`, adapts the historical Qbert stage reader to both games. It binds
midpoint/final saves to one complete 400,008-action history, verifies all saved
state and restored identities, retains separately initialized zero-moment controls,
and reuses the existing task scorer and whole-stream replay/video bindings.
Qbert keeps four complete episodes per stream with cap 600,000; Freeway keeps
75,000 unassisted frozen actions. The R256 recipe and task gates stay fixed.

The [completed check](../../runs/atari-dose-reader-check-20260916.dIT4dM/capture/result.json)
passes **902 Python tests**, including 90 new reader cases, and its read-only
audit reverifies 234 input pins and ten command records. All 32 native/build
inputs remain unchanged; no binary is rebuilt. Main's exact carry also passes
[all 902 tests](../../runs/atari-dose-reader-main-check-20260916.WyDKOL/tests.xml)
with actual module/package identities checked before and after. Preserve both
completed checks and the isolated source; only documented audit/import modes
are reusable.

The separate [historical component read](../../runs/atari-dose-reader-check-20260916.dIT4dM/integration.json)
reproduces Qbert's 17/24 successes and mean 3,754.17 versus 125, and Freeway's
16/36 successes and mean 24.5833 versus zero. Complete original/archived state,
replays, video identities and untrained moments verify. Both remain failed
**historical 200,004-action** experiments on their original backend. The new
complete-study reader rejects their old budgets; no header or result is relabelled.

This reader does not certify GPU execution or command lifecycles. The future
declaration must bind separate raw guard results for those prerequisites. A
successful pilot would still be one root, not three-root or five-game reliability.

## Preferred storage path: staged, not adopted

The runner already owns scheduled native saves. Isolated source
[`0a83c58`](https://github.com/kvark/kindle/blob/0a83c585898a625cafcdd30b0155bfd1bb229156/python/examples/atari_vector.py)
adds `--checkpoint-history`: save to `CHECKPOINT/<run-actions>` with exclusive
directory creation and the same native save call. Earlier slots remain intact;
the default still replaces its last save. No observer, thread, process signal,
pause/restart, extra native save or guard modification is needed. Retain and
analyze the two settled slots only after successful training exit.

The [CPU preparation](../../runs/atari-checkpoint-history-cpu-20260916.x3QpxQ/capture/result.json)
and audit pass: 856 tests, 178 input pins and eleven commands; all 32 native/build
inputs and six packaged modules remain unchanged. Thirteen explicit fake-agent
cases cover cadence, final saves, unchanged interaction/accounting, memory hooks,
existing-slot refusal and failed-save preservation. Preserve the initial two
test failures comparing different output paths; the corrected test verifies each
path separately and compares all remaining accounting exactly. No native failure
or numerical acceptance rule changed.

This flag is **not on main or GPU-qualified**. After the declared Pong campaign
and Breakout gates, require a small source-matched native default/history/restore
comparison before adoption and any dose study. Reuse the unchanged binary and
component results; no backend rebuild or broad requalification is implied.

## Next learning comparison, still undeclared

After storage qualification, use one fresh uninterrupted **400,008-action** history
per proposed pilot, with `--checkpoint-every 200004 --checkpoint-history`. Retain
the immutable midpoint/final slots after exit, then evaluate both snapshots with
the same frozen protocol and separately restored untrained control. Do not interrupt/restart at 200k:
checkpoints do not preserve live replay or belief equivalently.

Keep Qbert unassisted, with four complete frozen episodes per stream and cap
600,000. Keep Freeway training at probability .5/hold64 and both frozen stages at
75,000 unassisted actions. Keep N6/R256, the model, encoder, optimizer, full BPTT,
action vocabulary, rewards and competence criteria fixed. The midpoint and final
are one root, not two independent seeds or a historical-backend comparison.

A new declaration must still bind the actual GPU child, current immutable native/
adapter bundle, qualified history layout, complete training and frozen/control
evidence, native memory budget and whole-rollout replays/videos.
The previous Qbert stage/execution readers retain their historical source/recipe
bindings and must not be invoked as a current-runtime launcher. Preserve their
completed tests. Bind the source-matched storage and complete-study reader
explicitly in that future declaration. The existing guard owns the direct native
child and propagates save/runtime failures; the artifact reader is not a launcher.
Finish scheduled Pong and the Breakout action-width gates before launching more
game work. A successful pilot still needs fresh-root reliability confirmation.
