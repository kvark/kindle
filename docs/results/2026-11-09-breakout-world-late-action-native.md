# Late command conditioning: native qualification passes

The selected decoder path is implemented and qualified. This establishes its
values, derivatives, compatible default-off behavior and bounded learning/restore
mechanics. It does not establish better trajectories or prove the world model.
[Protocol](../experiments/2026-11-09-breakout-world-late-action.md).

Default-off `rgb_late_action` moves the18 command planes from motion0 to fusion,
after three image-only motion layers. Exactly7,200 weights move between the two
5x5 convolutions. At1M/RGB64 there are still80 logical parameters,603,915 scalar
weights and240 saved arrays. Two weight shapes differ; old checkpoints cannot be
reinterpreted as the new architecture. Checkpointv26 binds the flag and wraps
validatedv18 or physical-featurev25 metadata. Whole-agent and unsupported
objective/phase combinations fail before GPU. Legacy flags remain default-off.

## Verification

- CPU:96 world checks,29 configuration,14 checkpoint-filter checks and5 decoder
  checks pass. Native tests remain ignored in these suites. New checks cover
  header disguises/composition, exact parameter movement, causal graph ancestry,
  action/history derivatives and lane isolation. Direct commands enter fusion;
  motion0–2 use only causal image history. Strict core/binding Clippy pass.
- Packaged Python:56 pre-GPU checks pass against the new native package, including
  legacy action-feature/local-action and newv26 rejection cases.
- Native decoder: one declared test compares values and every parameter gradient
  against CPU F64 reference at three command feeds. Tolerance is2e-4+2e-3 relative.
  First-lane predictions remain exact when changing only the second lane. No
  optimizer update occurs in this decoder fixture.
- Default-off restore:32 cached forecast arrays and240 source/frozen tensors
  match the retained early action-feature qualifier exactly. No old runtime is
  queried or old physical assessment repeated.
- Two new one-update worlds use physically equivalent raw-command feeds. Their
  common240 initial tensors,240 trained tensors and all13 reports are exact.
  All32 initial and32 trained alias forecasts match.77/80 parameters change,
  covering encoder, posterior, core, prior, decoder/motion/fusion and event heads;
  the same three initially zero reward-hidden weights stay unchanged.
- Trained restore reproduces32 forecasts and240 own frozen tensors exactly.
  Metadata retainsv26, the late flag, physical feature table, base-format chain
  and optimizer/model age1. These test instances are not a main learner.

## Identity and closure

Source/runner **81a5e4e3cf68a582ec7cd39cbaebe2aa66afcf76**. Native SHA256
`4b0eb0b0b141ae576bda738a85afa275038b3da567711884d5e2dfdbb96986ba`.
Root `/x/Temp/kindle-world-late-action-20261109`, exact `runtime`/`runner` pair.
The standalone validation executable is retained with debug sections removed;
its packaged bytes are pinned and execute the actual native test.

One host-guarded qualification completes in52.355789s, including decoder fixture,
restore/construction, predictions, two updates, saves and input verification.
All128 execution pins and69 closure pins pass. Native service invocation
`0394725821b54f0983dd544e16059254` and observer both terminate successfully.
The observer verifies the same live invocation and terminal journal, including
after systemd unloads the unit. Guard audit verifies nine sealed files, no
unfinished children, no fault/recovery and zero NVML queries. No failure or retry
occurs in this qualification. Earlier experiments' failures remain retained.

The2.9GiB disposable compiler cache is released after packaging and terminal
qualification. Runtime, wheel, source archive, executable, logs, configurations,
initial/trained/frozen checkpoints and all forecast exports remain. No emulator,
collection, actor/critic or CDP work occurs.

The preceding real-history result motivates this hypothesis but does not establish
its cause. The earlier action-feature world still fails broader gates. Review
this qualification before separately declaring fresh fixed-budget learning;
unchanged dense and physical/event gates, full game coverage, brick memory,
untouched families and fresh confirmation remain required.
