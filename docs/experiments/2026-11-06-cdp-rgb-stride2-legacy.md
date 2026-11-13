# Default-off stride-two package compatibility

The [two new native fixtures](2026-11-06-cdp-rgb-stride2-native.md) pass once:
all Tiny encoder outputs/loss and four selected derivatives match the independent
reference; two synthetic CDP updates change encoder weights and restore all
parameters, moments and config exactly. Zero Atari actions,266 pins unchanged,
3.568s guarded controller. Both children exit0 and are gone; no render holders
or running Kindle services remain. Original CPU fixture failure stays retained.

Select two frozen1,024-action Breakout traces using the new package with
`rgb_stride2` omitted: retained final and its actual initial checkpoint, seed
4000001009, eight independent environments, greedy actions, native RGB,
published protocol, repeat4, full18 actions, sticky.25, no noops or training aid.
No learner updates. The reference is
`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009`.
Compare all128 transition rows per arm to the corresponding saved reference
trace; only finite wall timestamps may differ. Require exact config, architecture,
all357 checkpoint tensors/moments and zero learner updates. Normalize only the
old omitted `replay_value_bootstrap` to its existing `imagined` default. All
optional encoder/actor settings, including stride-two, remain omitted.

Use source22c262ea native SHA256
`cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`
and the exact current runner copied before execution. Pin runner, native,
package, reference checkpoints/traces, queue, controllers and guards. Use Auto
on the same AMD host; verify source bytes, live boot/driver and no other render
holder or Kindle service. Cap each GPU child at120s and queue at300s. Stop on
real faults/nonfinite results; preserve failure, no retry/reset/reload/reboot.
Require successful host audits, terminal child identity and exact frozen
invariance. These short traces establish compatibility, not natural-episode
competence or learning improvement.

Root `/x/Temp/kindle-rgb-stride2-20261106/legacy`. The3.9GiB disposable build
cache is released after packaging and native validation; wheel, library and
test executable remain. The exact original native-pinned build receipt is
archived separately; its current copy records cache release only.

Success permits a separate8,192-action cost declaration. No learning allocation
or full learner is selected here; original864/90% two-wall and the five-game
goal remain unfinished.
