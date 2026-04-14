## v0.2 (14 Apr 2026)

- **Rename**: `iris` → `kindle`. To kindle is to start a fire from nothing —
  the name makes the project's claim explicit. Crate name, repository URL,
  documentation, and internal identifiers all updated.
- **Workspace restructure**: root is now `[workspace]`-only. Core library
  lives in `kindle/` (publishable), built-in environments and runnable
  examples live in `kindle-gym/`, and an optional pyo3 extension lives in
  `python/` (maturin-built; excluded from `cargo build --workspace`).
- **Order reward redesign**. The old component-wise Shannon entropy of the
  raw observation vector (`−H(|o_i| / Σ|o_j|)`) was ungrounded and trivially
  gameable. Replaced with a causal, environment-agnostic formulation: a
  frozen random digest + bucket grid feeds two sliding windows (recent, W=64;
  reference, W_ref=512) of bucket IDs over the universal obs token, and
  `r_order = H_reference − H_recent`. Positive when the agent has
  concentrated its recent trajectory relative to its own historical
  baseline; negative when dispersing. Default weight now 0.5 (was 0.0).
- **Python bindings** (optional). A maturin-built pyo3 extension in `python/`
  exposes `kindle.Agent` with `train(env, steps)` / `diagnostics()` so a
  kindle agent can be dropped into any gymnasium loop.
- **CI**: workspace-aware build, reward-signal CPU tests on every run,
  GPU-gated tests split per-crate, new Python job that builds the
  extension and runs pytest.

## v0.1 (15 Apr 2026)

- Phase 0 — Foundation
- Phase 1 — World Model & Reward Circuit
- Phase 2 — Credit Assigner
- Phase 3 — Policy & Full Loop
