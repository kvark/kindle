# Completed observed returns for replay-value targets

Select one default-off native implementation and qualification. The retained
replay-value loss learns from real rewards with imagined bootstraps. Earlier
critic diagnostics found substantial complete-return error, and the completed
bonus-off actor experiment demonstrated that completed returns become available
online. Neither established that replacing replay-value targets helps. Existing
completed-return actor and life-boundary learners remain closed; do not repeat
them. This proposal changes replay-value supervision, not the actor auxiliary.

## Target semantics

Add `replay_value_completed=false`, omitted from default serialized configs.
When enabled, retain discounted observed returns through true game terminals,
using the existing return accumulator independently of the optional actor loss.
Use the configured scaled stored reward channels, the ordinary continuation
discount and F64 accumulation rounded to F32. Include the terminal arrival's
reward in the preceding outgoing state's return. Never cross resets or streams;
timeouts and live tails remain incomplete. Ring eviction may remove earlier
history without invalidating the known future of a retained suffix.

At valid replay state t, replace its scalar lambda target with the completed
return when available; otherwise keep the exact existing lambda target. Preserve
the final/reset masks, two-hot encoding, slow-value regularization, replay-value
coefficient and representation-gradient switch. The same selected target feeds
the behavior critic and existing world replay-value loss. Imagined actor/critic
targets, reward/continuation supervision, sampling and optimizer ownership stay
unchanged. No extra loss, head, parameters or graph operations are added.

Reject active model-derived disagreement: those bonuses are recomputed during
learning and are absent from stored complete returns. Reject auxiliary life
boundaries, which would change the target's game-level meaning. Ordinary stored
intrinsic reward remains supported. Require a positive replay-value coefficient
and CPU target construction; this restriction is compatible with AMD Auto.
Expose the number of valid replaced rows on every learner report when enabled,
including zero before any returns complete. This counts repeated replay exposure,
not unique experiences. Checkpoint metadata preserves the option; replay and
pending returns remain absent from checkpoints.

Use fresh CLI `--replay-value-completed`, require `--disagreement-scale 0`, and
reject explicit fresh overrides on restore/evaluation. Check a specific native
capability on both fresh selection and restore, before environment construction.
Default source/runtime behavior must remain compatible with retained checkpoints.

## Qualification and limits

Preserve pre-edit source bytes and exact receipts under
`/x/Temp/kindle-completed-value-20261106`. CPU checks cover configuration/default
serialization, incompatible reward/boundary modes, independent forward return
sums, terminal rewards, timeout/reset separation, stream/ring retention, sampled
context, fallback and label/row alignment. Python checks cover fresh/restore
validation, native capability, exact config propagation and report accounting,
without native or emulator construction. Run relevant Rust/Python suites and
strict Clippy, then package an isolated native runtime with source/wheel hashes.
Preserve failures and separately record repairs; never overwrite receipts.

After packaging, declare bounded native integration checks that verify selected
targets, unchanged initial parameters/default-off frozen behavior and finite
updates. The value-loss graph and its derivatives are unchanged; qualification
must demonstrate the new labels reach that graph rather than claim a new
derivative mechanism. No GPU allocation is selected by this implementation
declaration, and no full learner or automatic successor follows from CPU checks.

If qualification passes, a separately declared 8,192-action cost/activity check
can compare with the retained bonus-off control (36.677s bounded; 65.208 greedy /
46.458 sampled at 200k in 17m05.486s). Any later fixed-budget learner must also
beat the retained bonus-on learning-efficiency point before adoption. Removing
the bonus/ensemble changes the recipe relative to that primary control; it must
not be presented as an isolated critic comparison. Completed targets follow the
historical behavior/teacher mixture and can have greater variance and policy
mismatch than bootstrapped targets. Gameplay benefit remains unestablished.

Keep primary Breakout 143.667 greedy / 70.750 sampled, CDP scale 500 and all
original game protocols. Original 864/90% two-wall and the five-game goal remain
unfinished. No coefficient sweep, unchanged extension or old-stage repeat.
