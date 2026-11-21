# Direct world RGB color attention: native qualification

The optional replay-derived color attention passes implementation qualification.
This does not establish useful learning or Breakout competence. The full goal
remains active and unproven.

[Protocol](../experiments/2026-11-09-breakout-world-color-attention.md).
Root `/x/Temp/kindle-world-color-attention-20261109`.

Replay red excess over1.5×max(green,blue), region[4/64,28/64,60/64,56/64], gives
spatial weights1+1000×excess. Repeat across RGB channels, multiply existing motion
weights and normalize the combined weights to mean one per image. Apply to both
observed and causal imagined RGB cross-entropy. Replay/motion/reset weights stay
detached; future RGB/weights/later actions never enter earlier beliefs. No object
or emulator-state input, new head, learned parameter or inference path. This is
explicit game-specific color/region knowledge, not unguided representation.

The independent CPU audit on256 existing qualification arrivals gives finite,
positive weights, mean-one maximum error6.67e-16 inF64, range.2000–4336.82.
243 rows have positive color mass; their positive-color pixels receive69.49% of
RGB loss weight on average,72.71% median. This is not oracle presence, adequate
coverage or a learning score. The intervention is strong and can encourage
incorrect color fragments; all original physical/image/event gates remain.

144 world CPU tests pass,42 GPU tests ignored;29 config tests,strict core/binding
Clippy and25 packaged Python checks pass. Native primitive RGB-logit loss/weights/
derivatives match F64 withF32 graph constants; replay/motion/keep gradients are
detached and reset-masked gradients exactly zero. Checkpointv23 and constructor
checks reject incompatible headers/objectives/geometries beforeGPU. Defaults
remain off; whole-agent construction/restore rejects this world-only mode.

Full RGB64/12M/B8/T32/H16/Auto fresh one-update qualification passes11 finite
reports and total/mask accounting.68 logical weights/8,852,675 scalars/204 saved
weights+moments;65 weights change, including encoder/GRU/prior/decoder. Three
reward hidden weights retain first-step initialization. All204 common initial
and own initial/trained frozen tensors,32 actual-initial and32 trained restore
prediction arrays are exact. No old model is queried again.

First CPU fixture compilation failed because tuple string types were inferred
as unsized str. The corrected fixture then used an ideal F64 reciprocal instead
of the graph's F32 mean constant. Original sources/logs remain; the separate
corrected F64 reference passes. No native job or guard audit repeats for these
fixture repairs. Native qualification itself succeeds in19.477s, including the
primitive subprocess, one world optimizer update, readbacks and restore. One
host-only guard/nine sealed files pass; zero main learner updates, emulator
actions/frames, NVML, resets or recovery. This interval is not training speed.

Sourcec672292c/native SHA256
`ef45f798dcfc6e6714b17506ae5b73237ccf236362f81c1ab614b608e6c66695`.
CPU test SHA256
`932ead7fb51550ced6ee2e3448ecdaf84ef08d57f276e1a68c05028ad15b6a53`.
Invocation2991cdb346e04a14be6caf4931b49567. Wheel/native/test executable/source,
failed fixtures/logs and all checkpoints remain. Own disposable build cache is
released after packaging and qualification.

Only a separately declared fresh8,192-update broad learner may follow. It would
keep original H16/RGB100 settings and data, changing only this attention. It must
match these zero-step initial tensors, not resume the qualifier update. Ordinary
RGB observed≥90%,one≥89/109,eight≥28/107 and all horizon/paddle/brick/event metrics
stay. No learner or automatic successor is allocated by this receipt.

Full precise physics,natural event calibration,removed-brick memory,late/tunnel/
second-wall/above-top,untouched families andfresh confirmation remain required.
No implementation/fitting result completes the goal. Actor/critic,CDP comparison
andfive-game work stay parked;prior jobs stay closed andfullres protocol-failed.

Closure verifies158 pins andone terminal successful service/nine sealed files;811,401,216 bytes ofown build cache released.
