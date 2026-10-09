# Soft disagreement targets: qualified for the learning screen

One production change: ensemble regression uses detached posterior categorical
probabilities rather than sampled one-hot targets. It preserves the expected
squared-error parameter gradient, not bitwise optimizer trajectories. The
regression test verifies observation/previous-state dependence and independence
from the current categorical draw. No other learning setting changes.

Local114 Rust/1,137 Python tests, formatting and strict workspace/Python Clippy
pass. Four guarded GPU checks pass: independent F64 bonus/raw derivatives,
detached repeat-versus-novel learning, real/imagined transition alignment with
positive intrinsic advantages, and exact zero-scale baseline updates/RNG.

The excluded seed9001 smoke finishes2,048 actions/451 updates in12.28s,
mean update25.19ms. The separate1,024-action frozen restore performs no learning
and preserves all346 model/optimizer tensors exactly. These are plumbing and
numerical checks, not reward discovery or matched throughput evidence.
All six guards/seals pass; no new allocation warning, NVML polling or recovery.

Proceed with the [declared three fresh seeds](../experiments/2026-10-05-freeway-soft-disagreement.md)
at32,768 actions each, reusing the completed hard-target/extrinsic controls.
Freeway remains unsolved. [Compact qualification](2026-10-05-freeway-soft-qualification.json).
