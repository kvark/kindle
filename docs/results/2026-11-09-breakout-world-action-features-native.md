# Standalone action features: native qualification

The optional fixed action table passes composition qualification. This supports
starting a learning comparison; it establishes no forecasting improvement.
[Protocol](../experiments/2026-11-09-breakout-world-action-features-native.md).

The three submitted-command features LEFT, RIGHT and FIRE occupy the existing
18-wide input. One shared lookup supplies training, observed prefixes and
imagined futures. Replay keeps raw action IDs. Reset -1 clears both features
and belief masks; zero-valued NOOP keeps the belief. No graph, parameter or
backend derivative rule changes. Whole-agent paths reject this standalone option.

Enabled checkpoints use v25, persist the full table, and validate their v2/plain
or v18/local-action base. Default None omits the field and retains old formats.
Only the small64 local-action 1M composition receives GPU qualification here;
plain composition has CPU header coverage only.

- All 93 relevant world CPU tests, 29 configuration tests, five checkpoint tests
  and 44 packaged Python checks pass. The four new feature groups pass after the
  final style correction. Fourteen GPU tests in the world suite remain ignored.
- Default-off initial 240 tensors and 18 forecast arrays exactly match retained
  small64 controls. Feature-enabled initial weights share all 240 tensors;
  forecasts differ when action inputs differ, as intended.
- Fresh alias/canonical instances each learn once from the same RGB, reward and
  terminal batch with equivalent physical commands. All 13 reports and all 240
  resulting arrays are bitwise identical. All 18 raw commands appear in the
  fixed future plans. Each instance changes 77/80 parameters; the three initial
  reward hidden parameters remain unchanged at the first update.
- At initialization and after learning, alias versus canonical observed/future
  commands give 32 exact fixed-noise forecast arrays. Trained restore matches
  another 32 arrays. Own initial/trained/restored frozen tensors remain exact.
  The model retains 80 parameters / 603,915 scalars / 240 saved arrays.

Qualification takes 41.435s on Auto/RX7900XT/RADV. The one guarded job exits
successfully; the host audit verifies nine sealed files with no faults, resets
or unfinished children. All 182 execution pins and 41 closure pins pass. Two
qualification updates, zero main learner updates, zero new emulator actions or
frames. A 2,994,106,368-byte disposable build cache is released; wheels, both
packages, matching CPU executables, runner, source and evidence remain.

The initial strict Clippy nested-conditional failure is preserved. A separate
style correction passes; its separately packaged native bytes happen to match
the first unqualified package exactly. Neither package is mixed with another
runner. The earlier emulator verifier failure remains separately documented.
No GPU qualification stage fails or repeats.

Source `1331b70278c20253a5b3a4c6500a33d7a624e027`; native SHA256
`15ea18b55336bab5f159aa0af3fe6f366d62e4f2985aa7ce86176783d6eabada`.
Runtime `/x/Temp/kindle-world-action-features-20261109/native/runtime-v2`,
matching runner under the same root. Invocation
`f27bf10f64364f98acafe7db254cc622`; service is inactive/successful.
Root `/x/Temp/kindle-world-action-features-20261109/native` is closed.

The complete Breakout world-model goal remains unproven: precise physics and
action responses, natural event calibration, persistent brick memory, late-game
and above-top coverage, untouched families and fresh confirmation are still
required. A separately declared fresh learning comparison is the next selected
stage; no old learner is extended or repeated.
