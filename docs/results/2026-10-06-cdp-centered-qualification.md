# Centered-CDP loss qualifies; learning benefit remains untested

The [declared centering ablation](../experiments/2026-10-06-cdp-centered.md)
passes eight serialized GPU guards and the independent CPU audit. No new
allocation/validation warnings, NVML polling or recovery. This verifies the
exercised loss and runtime path, not improved Pong or five-game competence.

- F64 reference:128 centered distances and32,768 raw prediction derivatives
  pass existing tolerances. The target and its full-batch mean are detached.
  The original four-distance/1,024-gradient cosine fixture also passes.
- Unchanged CDP passes1,300 upstream value/gradient/optimizer/EMA comparisons
  across four synthetic learner updates. This protects the reference arm;
  centering is our ablation, not claimed upstream numerical parity.
- Control and centered production paths each run1,024 actions/195 updates,
  then frozen restore1,024 actions/zero updates. Finite checkpoints, encoder
  movement and zero training debt pass. Actual initial checkpoints have zero
  actions/updates and exactly identical346 tensors across arms. Each frozen
  export preserves all346 model/optimizer tensors byte-for-byte.
- Full local checks:115 Rust tests,1,171 Python tests, formatting and strict
  workspace/binding Clippy. The actual future campaign's training-audit
  function is rehearsed on both production smokes. The first CPU build
  service stops at exit127 because Cargo is absent from its PATH, before
  compilation/GPU work; an explicit PATH corrects it. That failed log remains.

Native`d32f3dc88f39c4d3e5341b396443db789b41cad5ef6f06304e1e2e2c365b5f08`,
Meganeura`b684ffd9`, Blade`e349cddf`, RTX5080/580.178.04. The only learner
mechanism change is the optional centered training loss. No new encoder,
parameters, EMA statistics, acting transform or CPU learner path.

The CLI also saves actual initial weights when explicitly requested; audit
tools reject a centered run labeled as unchanged CDP. Default CDP remains
unchanged. Full-batch BPTT is required for the centered loss so its target
population cannot silently vary with microbatching.

The two training smokes take6.67s and6.73s, respectively. These short checks
are not steady-state speed evidence. The upcoming matched learning runs measure
whole-agent cost.4,096 smoke actions and synthetic updates are excluded
diagnostic compute; GPU utilization remains unmeasured.

[Compact audit](2026-10-06-cdp-centered-qualification.json).
Artifacts: `runs/cdp-centered-20261006.n6XuQXmE`.
