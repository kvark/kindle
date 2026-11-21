# Fixed-width categorical regrouping development

Selected November 6 after the residual serving ablation closes. This is a new
representation experiment within the unfinished five-game goal. No GPU stage,
bounded Atari learner or full comparison is selected by this development protocol.
Root: `/x/Temp/kindle-latent-shape-20261106`.

The current image reaches the actor through sampled categorical posterior state.
The earlier [representation readouts](../results/2026-11-01-cdp-encoder4e5-representation.md)
show useful but imperfect position/motion information; readout error does not
prove information is absent, and those measurements precede the retained recipe.
The completed width, coordinates and lower-mixing comparisons did not improve
retained gameplay. This proposal changes categorical grouping at fixed flattened
width rather than repeating those interventions. The actor-gradient probes do
not establish a representation cause and are not its justification.

At12M, test64 variables with8 classes instead of32 with16. Both have512 flattened
entries and the same2048 deterministic state,2560 actor/critic feature width,
CNN, parameter shapes/counts and RGB replay. The theoretical maximum categorical
code size changes128→192 bits, but useful information is unmeasured. One-hot
activity changes32→64; sampling, KL sums and subsequent experience also change.
This is not an isolated capacity or noise test. Preserve existing unimix.01,
free nats, KL scales and all other retained settings in any later comparison.

Implement optional `latent_shape={variables,classes}`, omitted by default, and
fresh-only `--latent-shape VARIABLES CLASSES`. Validate positive variables,
classes>1, checked multiplication equal to the selected preset's flattened
width, and centered RGB CDP scope before constructing graphs or allocating GPU
resources. Restore takes shape from metadata and rejects fresh overrides. A
native capability flag prevents a new runner from silently using an old package.
Legacy configurations and identity overrides preserve graph semantics. No new
kernel, actor objective, image transform or emulator-state input is introduced.

CPU qualification covers legacy/roundtrip/invalid configuration, equal parameter
schemas for all graph owners, changed categorical geometry, and early CLI
failure/capability/restore handling. Construct every synthetic RGB observation
before native execution. Run workspace CPU tests, relevant Python CLI tests,
strict workspace/binding Clippy, and installed-package CPU tests. Package exact
native, runner and a release test executable with source hashes; retain failures
and use separately named repaired preparation stages if necessary.

Prepare two ignored native fixtures for a separate guarded declaration:

1. Compare production categorical KL values and derivatives against independent
   F64 calculations at32×16 and64×8, including mixed softmax and stop-gradient
   routes; check supplied-draw samples and zero parameter/optimizer changes.
2. Tiny4×4→8×2 RGB CDP: two finite synthetic vector updates, changed world/actor/
   critic weights, shared actor parameters, one-hot stream states, and exact
   weights/moments/configuration restore with empty replay. Zero Atari actions.

Only after native qualification may a separate fixed8,192-action check be
declared. A later fresh200k comparison needs its own fixed protocol, actual-initial
frozen controls and own-initial gain per training minute. Parameter initialization
should match retained because names/shapes are unchanged; sampled initial action
traces may differ. Never impose retained trace equality on changed grouping.
Do not resume a retained32×16 checkpoint as64×8. No extension, sweep, old-stage
repeat or automatic successor follows. Keep the retained143.667/70.750 recipe
and original864/90%-two-wall goal unchanged.
