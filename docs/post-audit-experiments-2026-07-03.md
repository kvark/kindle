# Post-audit fix-out and first experiments — 2026-07-03

Follow-up to `docs/audit-2026-07-03.md`. All 11 critical/high findings,
all 24 mediums, and most lows were fixed in five commits (`agent:`,
`modules:`, `kindle-gym:`, `python:`, `docs/ci/hygiene:` on this
branch). This note records the verification and the first
post-fix training results on Atari and ARC-AGI-3.

Environment: remote container, lavapipe (CPU Vulkan) — all GPU numbers
below are software-rasterizer throughput, not real-GPU throughput.

## Verification

- `cargo fmt` / `clippy --workspace --all-targets -D warnings`: clean
- Unit tests: 86 + 20 pass; reward_signal: 6 pass
- GPU suites (now hard-gated in CI): opt_parity 2/2, canaries 2/2,
  wm_direction 1/1, kindle-gym stability 2/2
- Python: `pip install -e python/` works again (readme path fix);
  pytest 13/13 including the new error-path tests
- `bench_envs` under the corrected post-action observation convention:
  **14/14 WM convergence checks pass** across all 7 envs, L0 and L1 —
  the WM learns the true `(z_t, a_t) → z_{t+1}` problem and still
  converges everywhere.

## Atari (Pong, `atari_batch.py`)

Setup: gymnasium[atari] + ale-py + opencv installed; standard
preprocessing (frameskip 4, 84×84 gray, framestack 4), 8 lanes.

- Smoke (2k steps, bare): runs end-to-end on the fixed signed
  extrinsic-reward channel at 27 env-steps/s. 92% of wall time is the
  batched `observe()` (268 ms) on lavapipe.
- **Latent collapse found and instrumented**: bare CNN runs show WM
  loss ≈ 0.007 with policy entropy pinned at ln(6) — the encoder
  collapses to a near-constant z (the WM trivially predicts it; the
  policy is state-blind). The harness now exposes the WM graph's
  reconstruction loss (`--recon-loss-coef`, `--recon-visual-target` —
  the same fix the ARC GRPO recipe uses) and prints a cross-lane
  latent-std canary (`zstd`) every log line.
- 12k-step run with recon 10 + n-step 8 + value bootstrap + γ 0.99 +
  entropy β 0.01: recon holds the encoder non-degenerate
  (wm ≈ 2.09 doing real reconstruction work, zstd 0.064 stable), value
  loss converges (9.7 → ~1), but mean return stays at the random floor
  (−20.22 over 98 episodes at 96k env-steps). Expected: Pong needs
  several hundred thousand env-steps; at 9 env-steps/s (observe = 888
  ms with the recon decoder, 97.5% of wall) that is a multi-day
  lavapipe run. **On this hardware the Pong result is a correctness
  baseline, not a learning result.** Next lever is throughput (real
  GPU, or lighter recon target), not algorithm.

## ARC-AGI-3

The toolkit (arc-agi ≥ 0.9 + arcengine) needs Python ≥ 3.12 — set up in
a venv with the kindle bindings rebuilt for 3.12. Connects with an
anonymous API key; 25 games available; games download and run locally.

Baselines with `arc_agi3_batch.py` (6k steps each, ~15 steps/s):

| run | result |
|---|---|
| cd82, `--preset full` (validated stack) | level 1/6 reached, mean end-level 0.98 — reproduces the pre-audit baseline, but entropy is pinned at ln(6): L1 falls to randomized exploration, the policy never commits |
| ls20, `--preset full` | **0 level events** in 46 episodes — ls20's L1 does not fall to a random walk; entropy pinned at ln(4); WM loss again collapse-tiny (the `full` preset carries no recon loss) |

GRPO harness (`arc_agi3_grpo.py`, cd82+ls20, 4 forks/game, PPO +
masks + recon 10 — the run that exercises the audit's PPO-path fixes:
single-step PPO advantage staging, KL/π_old consistency, executed-
action feedback, mask pinning):

| macro step | events | entropy | notes |
|---|---|---|---|
| 25 | 0 | 1.79 (max) | |
| 50 | **1 (ls20)** | 1.72 | first fork-sync; **ls20 level event — the game where the preset baseline got zero in 6k steps** |
| 100 | **2 (cd82+ls20)** | 1.65 | recon-WM 0.17, policy loss down 16× |
| 175 | 2 | 1.56 | entropy still descending — the policy keeps committing |

Two qualitative changes vs. the pre-fix picture: (1) entropy actually
moves — the policy commits instead of staying uniform forever, and
(2) level events appear on a game random exploration doesn't crack.
Throughput is the binding constraint again: ~2 env/s aggregate for the
256-dim CNN-DQN stack on lavapipe, so a session-length run covers only
~4k micro-steps/lane. The historical wall ("1 game at L2 per 50k
steps") needs a 50k+-step run on real hardware to test properly.

## What to try next

1. Same GRPO run on a real GPU (or with `--latent-dim 128
   --hidden-dim 128`) for ≥ 50k micro-steps — the entropy trend and
   early events at 4k suggest the fixed PPO path scales past the
   pre-fix ceiling.
2. Atari observe() cost: the recon decoder tripled the observe time.
   A downsampled recon target (or reconstruction every k-th step)
   keeps the anti-collapse pressure at a fraction of the cost.
3. The `full` preset should probably include the recon block — both
   its games show the collapse signature without it.
