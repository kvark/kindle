# Native standalone world action features

The [CPU/emulator semantics check](../results/2026-11-09-breakout-world-action-semantics.md)
finds608 exact nonidentity comparisons across832 rollouts and distinguishes all
15 pairs of six canonical controls somewhere in32 fixed parent selections.
Select native implementation and a bounded composition qualification now. This
is a separate decision after reviewing semantics;no learner is selected here.

Root `/x/Temp/kindle-world-action-features-20261109/native`.

## Implementation

Add default-off `world_action_features`:an optional action_count×action_count
finite table with values in[-1,1]. Breakout rows encode LEFT,RIGHT,FIRE in the
first three channels and zero in the other15. Table lookup replaces raw one-hot
encoding only in the standalone world's shared action-input helper. Reset-1
keeps zero features and zero state/action keep masks;NOOP's zero features retain
one-valued keep masks. Raw action IDs remain in replay and boundary checks.

Do not change native graph geometry,weights,initialization,losses orsampling.
Require full-weight,positive-suffix categorical RGB worlds,either plain motion
cross-entropy or the existing local-action/event-rollout composition. Reject
unsupported auxiliary,mean/Gaussian,transport,feedback and frozen/teacher phases;
whole-agent construction/restore rejects the standalone option before GPU work.

Persist the table in world-only formatv25 with an explicit validated legacy base
formatv2 orv18. Never disguise this input representation as an older checkpoint;
reject table/format disagreement,invalid base format andscope before construction.
DefaultNone omits the field and preserves original config/checkpoint formats.
CPU checks covertable validation,reset/NOOP semantics,all18 aliases,serialization,
header guards andpre-GPU rejection,plus relevant existing tests andstrict Clippy.

## Bounded native qualification

Build/package from committed source,retain matching runner/wheel/test executable
andreceipts outside Git. Pin exact source/native/scripts/data/config andcached
controls before execution. Use Auto/RADV andthe existing host-only guard;stop on
real faults/nonfinite reports. No reset/reload orautomatic retry.

Qualify the retained small64 local-action1M/B8/T32/H16 composition,seed41001,
RGB100,event losses1,rates4e-4. First comparedefault-off240 initial checkpoint
arrays and18 initial forecast arrays against existing small64 preflight controls;
no old-model queries. Feature-enabled initial240 arrays must share the same
initial weights. No trainableparameters areadded (80 parameters,603,915 scalars).

Compare raw-alias versus canonical commands with fixed identical model seeds
andprediction noise in observed prefixes andimagined futures. Requireexact
predicted arrays,includingRGB,events andbeliefs. Train two fresh identical
feature-enabled instances forone update each on the same RGB/reward/terminal
batch,with raw aliases versus canonical action IDs. Requireall240 resulting
arrays and13 reports exact,finite totals/masks,expected parameter-family changes,
anddifferent predictions from disabled inputs somewhere. Native mapping is the
same shared helper in learning,observation andfutureprediction;no new backend
primitive or derivative rule isintroduced.

Require own initial/trained frozen240 arrays,table/formatv25 persistence andexact
trained restore forecasts with pinned noise. At mosttwo feature-enabled updates
andzero main learning updates;no emulator construction/new game actions. Useone
guarded qualification subprocess witha900s cap. Reviewall raw failures andlimits.
Release onlythe new disposable buildcache afterpackaging/qualification closure.

Passing this qualification establishes composition andinput semantics,not learned
dynamics. Onlyafter review may a separate bounded learning comparison be selected,
withunchanged fullworld gates. Fullphysics,events,brickmemory,lategame,untouched
families andfresh confirmation remain unproven. All earlier jobs stayclosed.

## Pre-GPU source correction

The initial strict core Clippy run rejects a nested conditional in table
validation. Preserve that failure and the first packaged wheel/runtime, which
has not been used for GPU qualification. Collapse the conditional without
changing its predicate. The focused feature/config tests and strict core Clippy
pass on the corrected source. Package the corrected source separately under
wheels-v2/runtime-v2 with new receipts; retain the initial package unchanged.
