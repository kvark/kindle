# Optimizer poison handling and cached-runtime qualification pass

October21,2026. [Prior state/Session evidence](2026-10-21-optimizer-state-boundaries.md).
[Meganeura draft PR275](https://github.com/kvark/meganeura/pull/275), commit
`4bdcb8d1ddccded8706f1fdad7a37fc60854bd2b`, contains one commit/two changed files.

The GPU repair now preserves NaN/Inf behavior as well as the previously tested
finite state. At beta2=0 it discards history only when that history is finite;
nonfinite history takes the original arithmetic path. A32-bit exponent check
implements the guard. Epsilon, variance storage, bias corrections and the
ordinary finite path remain unchanged. This is not general IEEE emulation.

## Current upstream

On Meganeuraa75c38e with Bladeeae0c6f7/Nagaa489554c, three GPU tests pass:
ordinary LaProp, finite/tiny-state checkpoint continuation, and sixteen mixed
poison/finite cases across Adam/LaProp and beta2=0/.999. The new192 checks compare
finite values at the original bounds and nonfinite values by classification/sign,
not NaN payload identity. Previous512 raw-gradient/1,536 state/960 exact restore-
continuation checks remain.53 synthetic Session updates, zero Dreamer updates.
The workstation headroom assertion moved out of the portable regression into
the local launcher. Rustfmt and direct Clippy pass with the repository's existing
needless-range-loop allowance. No other lint suppression or numerical relaxation.

The first queue failed before spawning a child because the Python wrapper's
executable hash was missing. The corrected queue retains that failure and launches
once. One GPU guard passes, zero warnings,342 receipts. Artifacts:
`runs/optimizer-poison-20261021.vMrW4YUp`.

**The declared serialized-checkpoint byte gate remains failed for three of four
files.** A separate CPU audit establishes that only metadata-map key order
changes: all192 F32 values and parsed metadata match, and eight tensor-byte/step
mutations are rejected. There is no GPU repeat and no numerical tolerance change.
Do not report whole-file identity. The draft PR describes this limitation.

## Retained Kindle runtime

The qualifiedd3ff4948 source/control was copied privately and all seals verified.
Its original optimizer shader matches upstreama75c38e's original exactly. The
only backend source change is replacing that shader with the qualified repair;
Kindle changes only its local Cargo path. No other kernels, compiler policies,
retained extensions, Blade/Naga versions or performance flags change. This is
distinct from the latest-upstream qualification above, not a silent source refresh.

The three tests pass again against this backend with command caching enabled.
Eight checks explicitly require both recordings and replays at updates4/8 across
the four finite cases. Optimizer commands remain dynamic; the graph commands
are what this extension caches. Checkpoint tensors and metadata match upstream
exactly after parsing,192 values.53 synthetic Session updates, zero Dreamer updates.
One passing guard, zero warnings,451 receipts. CPU audit passes; no worker remains.
Artifacts: `runs/optimizer-cached-20261021.2sOmnmzz`.

The full Kindle test executable is built as
`6b621e5cd7e784164097ad85ec6e555cdfcb5c6c2183cfd18d15cb2e793e09ee`, but **no full
Dreamer update has run on it yet**. The reference remainsd3ff4948. Across both
allocations: two passing GPU guards,106 synthetic Session updates, zero game
actions. These are component checks, not new Atari results or performance claims.

## Next

Qualify complete RGB/CDP updates and measure the repair's overhead before adoption.
Keep configurations, sparse rewards, exploration, precision and the six retained
flags fixed. Compare unchanged initial states/programs and account for optimizer-
state differences caused by preserving variance; do not demand reproduction of
the known-bad underflow or silently call any finite trajectory correct. Full-model
numerical evidence must precede a timing claim. No automatic production promotion
or Atari restart. The1.85× performance gap and five-game objective remain open.
