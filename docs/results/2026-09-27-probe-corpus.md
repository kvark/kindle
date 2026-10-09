# Phase 2 probe corpus: collected, quality comparison pending

**6,144 complete 16-arrival clips** across Pong, Breakout and Seaquest, from
100,959 actual random-policy actions. Four training, two validation and two
test trajectories per game; 256 clips each. Seeds, hashes, target coverage,
discarded terminal tails and emulator-frame counts are in the
[self-contained record](2026-09-27-probe-corpus.json).

Native max-pooled RGB210x160, full18/repeat4/sticky .25, no reset no-ops.
RAM is privileged **evaluation-only** data, not an agent input. This corpus
does not train the frozen encoders; PCA and supervised probes may use only
their declared training split. Seaquest was not in Tiny's video corpus.

Independent coarse sprite-color checks hit 100% of present Pong objects,
99.36% of Breakout balls and 100% of its paddles. Seaquest hits are 91.85% for
the player and 84.09–91.91% for fixed enemy slots. Visibility/palette/label
limitations need interpretation; these are not proven-perfect labels. Keep
all valid-target counts and unmatched cases, not just easy visible frames.
Coordinates describe the final arrival; velocities are backward differences,
not future forecasts. See the [protocol](../experiments/2026-09-27-representation-comparison.md).

A guarded Tiny extraction smoke passes all 24 recordings ×16 variants, capped
at 12 clips each. Exported tokens reproduce native JL/mean features exactly;
arrays are finite and nonconstant. It takes 42.81s and leaves >=15.84GB sampled
Vulkan estimated headroom. No NVML polling or recovery. This establishes an
exercised extraction path, **not representation quality, learning or Phase 2
completion**.

Full pretrained-Tiny extraction subsequently completes in 678.06s: all 24
recordings ×256 clips ×16 variants are finite and nonconstant, with exact
production projection parity. Its 48 game/variant ridge fits complete in
25.95s on one CPU core. These scores alone cannot show a pretraining benefit;
the initial-Tiny, Large and learned-CNN controls are still required.

The native GPU MLP fitter passes an independent scalar value/masked-gradient
reference, learns a known regression and leaves weights, optimizer moments and
step count unchanged during evaluation (guarded RTX 5080 test, 2.29s). Full
MLP fits and all RL comparisons remain required.

Raw artifacts: `runs/representation-probes-20260927.POCnif`.
