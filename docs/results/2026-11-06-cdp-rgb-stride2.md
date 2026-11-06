# Optional stride-two RGB encoder: qualification

The [default-off experiment](../experiments/2026-11-06-cdp-rgb-stride2.md)
implements `rgb_stride2` and fresh CLI `--rgb-stride2`. Four stride-two5×5
convolutions replace stride-one convolution plus max-pool in the encoder.
Parameter names/shapes/count, observation/replay geometry, embedding width and
latent/control dimensions remain unchanged. Decoder graphs remain exact.

The encoder convolutions compute one quarter of their former output positions;
no encoder max-pool follows them. This changes sampling and nonlinear reduction,
so it is neither a behavior-preserving optimization nor a measured speedup.
Learning quality and total training cost remain unmeasured. Native max-pooled
RGB and the existing single GPU resize remain; no labels or emulator state enter
the model. The earlier pixel diagnostic already recovers precise ball positions
fromRGB64, so resizing alone is not a sufficient explanation of feature errors.

## Completed CPU checks

| Stage | Result | Wall time |
|---|---|---:|
| Initial Rust compile |Fails in the new standalone test fixture: missing imports|25.307s|
| Repaired Rust suite |216 Kindle +5 adapter tests pass;125 native/reference tests ignored|13.834s|
| Python checks |1,246 pass,10 constructor checks deselected;zero native construction attempts|4.923s|
| Workspace/all-target Clippy |Pass, warnings denied|16.305s|
| Binding/all-target Clippy |Pass, warnings denied|7.380s|

The initial failed fixture and its receipt remain retained. Source review also
corrects an overly broad fixture assertion: posterior graphs include the changed
encoder, so their parameter/output shapes match, while their operations differ.
Imagined dynamics and behavior graphs still match exactly. These repairs change
tests only; production implementation and tolerances remain unchanged. The
original failed stage is not relabeled passed.

CPU evidence covers config defaults/roundtrip and invalid encoders;32/48/64/96
geometry; unchanged parameter shapes and control/decoder graphs; quartered CNN
output elements; absence of encoder max-pooling; CDP/reconstruction backward
graph compilation and all12 encoder gradient owners. An independent scalar
fixture checks nonconstant spatial sampling, channel order and every token.

Fourteen new CLI cases cover fresh settings, default behavior, native capability,
restore preservation and fresh-override/evaluation rejection. Python tests use
current Python sources with retained native7704d25c for defaults; the new capability
is mocked where required and all native constructors are forbidden. This is not
qualification of a newly built package. Existing CPU adapter tests may exercise
emulators; no native learning or GPU work occurs.

## Completed native qualification

Two new ignored fixtures compile: TinyRGB64 encoder values/loss and four selected
parameter derivatives against an independent F64 strided reference; two synthetic
CDP updates with exact parameter/optimizer/config restore. Both pass once under the
[guarded native protocol](../experiments/2026-11-06-cdp-rgb-stride2-native.md):
encoder.77s, synthetic learner/restore.58s; complete guarded controller3.568s.
All266 pins remain unchanged. Both children exit0 and are gone, guards pass,
and no render holder or Kindle service remains. Zero Atari actions.
Existing native resize, CNN and learner fixtures are not selected for repetition.

Release test compilation passes in63.677s and wheel build in6.935s. Installed
package checks pass1,246 cases with10 deselected and zero native construction
attempts in4.572s. Source `22c262ea`, native SHA256
`cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`,
test executable SHA256
`da5dcc9901bba568e260d0db99f5546f3c373c0c163223e944ab4930a1687374`.
The receipt pins260 source files and all packaged artifacts.

Root `/x/Temp/kindle-rgb-stride2-20261106` retains source pins, exact commands,
logs and the original failed fixture. Native qualification is complete.
The [default-off compatibility protocol](../experiments/2026-11-06-cdp-rgb-stride2-legacy.md)
now selects two frozen1,024-action traces of retained/actual-initial checkpoints.
No learning updates. The fixed8,192-action cost check remains conditional and
requires a separate declaration. The3.9GiB build cache is released after
packaging/native validation; all packaged artifacts and the exact original
native-pinned build receipt remain retained. No full learner or gameplay benefit is selected or established. Retain the
current teacher, RGB64 recipe and143.667 greedy/70.750 sampled; original864/90%
two-wall and the five-game goal remain unfinished.
