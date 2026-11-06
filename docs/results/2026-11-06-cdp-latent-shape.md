# Fixed-width categorical regrouping development

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
