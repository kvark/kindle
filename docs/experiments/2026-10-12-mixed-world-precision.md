# One fixed mixed-stage precision check

Declared October12,2026 after the BF16 convolution audit and before GPU work.
Dense+convolution BF16 improves CDP146.31ms versus dense-only149.27ms, but
world training61.87ms still loses to the retained F16 path56.47ms. Test the
measured combination: native BF16 posterior/imagination/behavior and F16-input
world-gradient products with F32 storage, masters, optimizer and reductions.
No per-game branch, model/replay change or new backend kernel. Explicit `mixed`
selection; Auto and live acting stay unchanged. RGB keeps its separately
qualified BF16 candidate; this is a CDP decision, not a universal promotion.

Compare `bf16` and `mixed` CDP12M/B32/T16/H15 on the same17c9287 backend,
checkpoint/RNG,16 warmup+256 ordinary measured synthetic updates each. No layout
or hardware-row patch yet. Require exact initial346 tensors and first imagined
targets, finite272 reports/all checkpoints, native BF16 in the four retained
sessions, no BF16 in mixed world gradients and actual F16 cooperative selection.
CPU tests must verify the world policy is derived without mutating the caller.
No exact optimizer-trajectory or learning-quality claim; record full and stage
time. If slower, retain the failure and discard the mode; no policy grid.

Two guarded600-second jobs in one bounded non-restarting systemd service, expected
5080 and>=2GiB sampled headroom. No compilation during timing, standalone NVML,
game actions, automatic learning resume or recovery. Retain warnings. Artifacts:
`runs/mixed-world-20261012.63YIxyjc`. The user reviews the backend round before
fresh learning; full CDP world reduced storage remains a separate limitation.
