## v0.2 (14 Apr 2026)

- Split into a Cargo workspace: `kindle` (core), `kindle-gym` (envs +
  examples), `python` (optional pyo3 extension, maturin-built).
- Redesigned the order reward. Frozen random digest + sliding recent/
  reference windows over the universal obs token; `r_order = H_reference
  − H_recent`. Grounded, causal, and ungameable by latent collapse.
  Enabled by default (weight 0.5).
- Optional Python bindings: `kindle.Agent` with `train(env, steps)` and
  `diagnostics()`, drop-in for any gymnasium loop.

## v0.1 (15 Apr 2026)

- Phase 0 — Foundation
- Phase 1 — World Model & Reward Circuit
- Phase 2 — Credit Assigner
- Phase 3 — Policy & Full Loop
