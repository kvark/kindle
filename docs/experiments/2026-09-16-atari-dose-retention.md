# Retaining checkpoints for the Freeway/Qbert exposure comparisons

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

## Next learning comparison, still undeclared

Use one fresh uninterrupted **400,008-action** history per proposed pilot, with
`--checkpoint-every 200004`. Retain the midpoint before the final save replaces
the live checkpoint, then evaluate both snapshots with the same frozen protocol
and separately restored untrained control. Do not interrupt/restart at 200k:
checkpoints do not preserve live replay or belief equivalently.

Keep Qbert unassisted, with four complete frozen episodes per stream and cap
600,000. Keep Freeway training at probability .5/hold64 and both frozen stages at
75,000 unassisted actions. Keep N6/R256, the model, encoder, optimizer, full BPTT,
action vocabulary, rewards and competence criteria fixed. The midpoint and final
are one root, not two independent seeds or a historical-backend comparison.

A new declaration must still bind the actual GPU child, current immutable native/
adapter bundle, midpoint observer and failure propagation, complete training and
frozen/control evidence, native memory budget and whole-rollout replays/videos.
The previous Qbert stage/execution readers retain their historical source/recipe
bindings and must not be invoked as a current-runtime launcher. Preserve their
completed tests. Source matching and the new Freeway frozen-stage contract need
explicit binding in that future declaration; this helper does not certify them.
Finish scheduled Pong and the Breakout action-width gates before launching more
game work. A successful pilot still needs fresh-root reliability confirmation.
