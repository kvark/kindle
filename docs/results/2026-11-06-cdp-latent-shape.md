# Fixed-width categorical regrouping development

Native qualification is complete with a separate test readback repair. The
original qualification remains failed. The repaired categorical fixture and
previously unexecuted Tiny learning/restore fixture both pass; details and
limits follow below. No Atari learner or gameplay benefit is established.

The [development protocol](../experiments/2026-11-06-cdp-latent-shape.md) selects
an optional64×8 latent grouping at12M instead of32×16, with the same512 flattened
entries and network parameter shapes/counts. Sampling, active one-hot entries
and KL behavior change together. A representation benefit is unproven; no
Atari learner or GPU allocation is selected at this stage.

`DreamerConfig.latent_shape` defaults absent and preserves legacy serialization.
Validation rejects zero/one-class/overflow/mismatched-width configurations and
requires centered RGB CDP before construction. `--latent-shape VARIABLES CLASSES`
is fresh-only; checkpoint metadata preserves the shape. A capability check
rejects old native packages before fresh or restored construction. All existing
CNN/RSSM/actor/critic graph builders consume the selected shape; no new loss,
kernel, image transform or emulator-state input is added.

CPU qualification passes202 Kindle and5 adapter tests,36 focused Python tests,
and strict workspace/binding Clippy. Legacy and identity-override graphs match;
Tiny and12M RGB graph owners preserve parameter schemas while categorical input
geometry changes. The independent F64 KL reference covers clamped and active
states and stopped derivatives. Both synthetic streams'34 RGB observations
construct on CPU; declared world parameter names exist before GPU execution.

The first Python invocation fails during collection because `atari_vector` is
not on the import path. The explicitly configured repair passes without source
changes; original log/receipt remain. Review then catches a feature-encoder name
in the new RGB native fixture before its execution, corrects it to `cnn0.weight`,
and strengthens synchronization checks to all11 actor parameters. The affected
CPU suite and strict Clippy pass again. Original fixture copies are retained;
no native stage failed or ran during preparation.

Root `/x/Temp/kindle-latent-shape-20261106` retains exact stage declarations,
source hashes and preparation evidence. Release packaging is next, followed by
installed-package CPU checks. A separate guarded native declaration must precede
the two ignored fixtures; a separate bounded protocol must precede Atari learning.
Keep retained143.667 greedy/70.750 sampled, actor dynamics/residual/coordinates
off. Original864/90%-two-wall and five-game gates remain unfinished.

## Packaged native qualification selected

Release packaging passes on source536012d5, native SHA256 prefix2c9c6e6f and
executable73a18277, with246 source pins. The installed package passes1,139 Python
tests/10 deselected with zero native construction attempts. Both release artifacts
and the wheel are retained. The [separate native protocol](../experiments/2026-11-06-cdp-latent-shape-native.md)
now selects two guarded fixtures,120s each/controller300s: independent F64
categorical values/derivatives/sampling, then two synthetic vector updates and
exact restore. No Atari learner or bounded cost check is selected.

## Original native failure and separate readback repair

The original native qualification fails at11:00:22.953 UTC November6, source
befcbf0c, in2.169s wrapper. The first scalar-loss check passes, then the raw-KL
check reads the first output because `read_output` takes a length, not an index.
Derivative checks and the second learning fixture are not reached; zero learner
updates or Atari actions. The guard records child1462683 exit101 and no recovery
or unfinished children. Preserve the original failed qualification and binary.

The [separate readback repair](../experiments/2026-11-06-cdp-latent-shape-readback-repair.md)
changes only the new test fixture's raw-KL/sample reads to indexed reads. Graphs,
F64 oracle, thresholds, production math and original packaged native remain
unchanged. Five affected CPU tests and strict Clippy pass. A separate release
executable and new output/checkpoint paths will qualify the repaired derivative
fixture and previously unexecuted learning/restore fixture. No Atari learner.

## Completed separate repair and qualification closure

At11:04:01.865 UTC November6, the repaired controller succeeds in3.518s
(3.672s wrapper), sourceacc24213. Its executable SHA256 is
`550934ce0f77de6abcdb940ec7766a1a395021e72992381e58313a9a39b08035`.
Only `agent/latent_shape_tests.rs` differs among246 original source pins;
production native2c9c6e6f and all original package artifacts remain unchanged.

The categorical fixture passes in.75s: six declared forward/backward cases at
32×16/64×8 match independent F64 values and all parameter derivatives, including
free-nat clamping and stopped routes. Supplied-draw samples match exactly;
parameters and optimizer counters stay unchanged. The previously unexecuted
Tiny8×2 learning fixture passes in.59s: exactly two finite vector updates change
encoder/reward/actor/critic, all11 actor parameters share correctly, live groups
are one-hot, and checkpoint weights/moments/configuration/counters restore exactly
with empty replay. Both guards and255 pins pass. Zero Atari actions.

The separate CPU closure finishes11:05:07.150 in.037s, confirming original and
repaired child identities are terminal, no Kindle service/render holder remains,
and the original failed guard still records exit101. It verifies the isolated
fixture change and byte-identical production package. The subsequent cache
release preserves the wheel, original native, original test executable and
repaired executable while freeing6,411,078,903 logical bytes. No completed
native fixture or original failed queue was repeated unchanged.

`qualification-review.json` records `complete_with_test_readback_repair` and
`qualified=true`; the original qualification remains failed. Original execution
SHA256 `61f74b84aeb211095c7ff2494b09c239371db0c8d69b28333cd48893abbf2232`;
repaired execution `1118a7009213e5c47e982811c2ebbebd7e891b9ad5d021f44c6b993457dc3370`;
repaired native review `489fc055003e0591b7c35d7c292bb80c6267d3a0fb4bce4b1d00b07f10a5e5e9`.

These results establish the declared numerical and Tiny learning/restore fixtures,
not12M Atari learning, initialized-tensor equivalence or improved efficiency.
A separate fixed8,192-action cost/replay protocol is the next substantive check;
no such allocation or full learner is selected here. Keep retained143.667/70.750
and all original gates unchanged. Do not repeat the closed qualification stages.

## Selected bounded cost/replay check

The [fixed8,192-action protocol](../experiments/2026-11-06-cdp-latent-shape-bounded.md)
now selects one fresh12M64/64×8 Breakout learner and1,024 frozen greedy actions.
All other retained settings stay fixed. Require357 equal retained initial
weights/moments,489 finite updates, full training/frozen replays and357 invariant
frozen tensors. Initial action traces may differ under changed grouping.

CPU preflight passes in.167s wrapper with zero native/emulator construction:
both exact queued commands, missing-shape rejection, all357 equality and five
actor/world/shape/extra/missing negative cases. Same16,334,353 parameters and
5,939,200,000-byte full replay footprint. Guard/cost/headroom caps follow the
separate protocol; no full learner or gameplay benefit is established. Root
`/x/Temp/kindle-latent-shape-20261106/bounded`; original qualification failure
and successful test-only repair stay retained without repeating either stage.
