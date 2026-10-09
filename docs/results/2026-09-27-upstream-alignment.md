# Matched upstream protocol: CPU alignment passes, GPU comparison pending

The Phase 2 control uses the pinned upstream DreamerV3 agent/optimizer, with a
shared Kindle Atari collector and corrected interaction accounting. This is
**not a completed GPU learning comparison or an efficiency result**.
[Configuration and trace hashes](2026-09-27-upstream-alignment.json).

Both Python environments produce identical RGB64, reward, RAM and boundary
traces on Pong, Breakout and Seaquest, with six independent stream seeds:

- 960 actions per game with a deliberately short 256-frame cutoff: twelve
  cutoffs/game, correctly distinct from terminals.
- 9,216 actions per game with the real 100,000-frame cutoff: 6 Pong, 55
  Breakout and 16 Seaquest natural terminals. Early action repeats, resets and
  exact executed emulator-frame counts agree; no artificial cutoffs.

Gymnasium1.3.0 / ALE0.12.1 / Pillow12.3.0 match. Native Python uses NumPy2.5.2,
upstream uses1.26.4; the complete trace hashes still agree. CPU fixtures verify
prefill debt is discarded, reset observations earn no action/update credit,
partial-stream resets preserve terminal observations, and timeouts bootstrap.

The full comparison remains three learner seeds per method/game, **200,004
actual actions** each, N6/B16/T64/context1/H15/R256/F32, unassisted sticky .25.
Upstream retains its native policy synchronization delay. Its sequence capacity
99,616 corresponds to 100,000 arrivals including six 64-row context/tails;
sampling RNG, chunk storage and eviction implementations are not identical.
Upstream uses a learned RGB64 encoder/pixel reconstruction; Kindle uses native
input, frozen features and latent prediction. Those differences stay explicit.

The bounded matched-protocol GPU smoke is next, not a campaign learner seed.
Application NVML polling remains disabled; stock JAX backend initialization is
allowed. Every GPU process remains host-guarded and serialized. The automated
queue stops before any successor on a failed guard or incomplete native result.
