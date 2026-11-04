# One-step prior continuation investigation

The next selected bounded test directly supervises continuation on sampled
one-step prior states. It uses the shared native head and real terminal flags,
equal row weights and scale1. The option defaults to off and adds no parameters.
The [protocol](../experiments/2026-11-04-cdp-breakout-prior-continuation.md) requires
numerical and execution checks before any separate full learner is considered.
No gameplay benefit is established.

## Continuation is used in both targets and loss weights

Native imagined lambda returns use continuation at the arrival state; actor and
critic losses use cumulative continuation including the posterior start. Replay
targets use actual termination flags but, in this recipe, bootstrap from imagined
returns. The imagined-return shader function and CPU lambda-return/weight helpers
are byte-identical between retained447fe50 and terminal-weight3672a21a. This is a
scoped source check, not broad runtime equivalence.

An exploratory CPU summary of all24,954 saved learner reports finds:

| Training actions | Control imagined continuation | Weight10 imagined continuation | Control advantage attenuation | Weight10 advantage attenuation |
|---|---:|---:|---:|---:|
|0–50k|.982457|.961853|.896782|.803722|
|50k–100k|.994997|.987439|.965450|.910465|
|100k–150k|.996035|.993549|.968834|.944760|
|150k–200k|.996546|.995194|.972274|.957095|

Attenuation is sum(weighted absolute normalized advantage) divided by sum(absolute
normalized advantage), not mean survival or a parameter-gradient norm. In the last
window, absolute advantage is.019345 control/.027131 weight10 and weighted absolute
advantage is.018808/.025967. The weighted signal is therefore larger in weight10
despite stronger proportional attenuation. Return scales differ33.744/30.092 and
mean imagined returns16.823/18.157. These data do not support a simple overall loss
of actor signal; they do not exclude a damaging change at particular states.

The runs have different experience and learned states. These descriptive means
are not paired causal estimates, calibration measurements or significance tests.
Analysis takes.697s CPU on one core at nice10, with no model/emulator/GPU work;
both full logs match the previously completed experiment's manifest before/after.
Exact values and source-function hashes remain at
`/x/Temp/kindle-terminal-weight-use-20261104`. No analysis is repeated.

## Selected implementation and qualification

The auxiliary term uses the current transition's deterministic state and sampled
prior categories, before the arrival image enters the posterior. It uses an
independent RNG stream, the same continuation head and aligned real targets.
Reset rows receive zero weight and stay in the ordinary B*T denominator. Gradients
reach dynamics/prior and earlier posterior states. Acting, forecast graphs, the
posterior loss and disabled configuration remain unchanged.

The first CPU fixture fails configuration validation because centered CDP was
paired with truncated BPTT. Its source and13.003s failure are retained. Correcting
only the fixture to batch2/BPTT2 preserves production code and assertions. All
184 Rust library/5 gym CPU tests then pass in13.334s wrapper;18 focused Python
tests pass in.203s. Graph dependency checks exclude the arrival observation and
posterior sample, verify prior/head gradient paths, unchanged parameters/inference
and compilation of ordinary/staged training. No GPU fixture has run yet.

Runtime packaging, guarded numerics and the declared8,192-action check remain
pending. No full learner is selected. Root`/x/Temp/kindle-prior-continuation-20261104`.
Retain143.667 greedy/70.750 sampled,all0/24 two-wall. Original864/90%-two-wall and
five-game goal remain unfinished.
