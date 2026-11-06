# Stronger reward supervision passes bounded execution and cost checks

The [bounded protocol](../experiments/2026-11-06-cdp-breakout-reward10.md) changes
only the existing posterior reward-prediction loss scale from1 to10. One fresh
seed1009 run completes **8,192 actions /32,730 emulator frames /489 updates in
46.723357s**. Its padded200k projection is1,254.777659s (20m54.778s), below the
fixed30-minute screen. Historical bounded control takes47.689684s; this short
comparison does not establish a speedup or useful gameplay learning.

The1,024-action greedy restore takes.896658s with zero updates and357 invariant
tensors. All357 initial tensors and47 pre-update semantic events match the
retained1009 control. Encoder, reward, continuation, actor and critic parameters
change. All489 weighted world and unchanged behavior totals pass; an independent
saved-log recount finds maximum world-total absolute error.00013553 within the
declared relative tolerance. Full CPU replay reconstructs every training action,
reward, frame, boundary and corrective pixel label;3,788 overrides are recorded.

Both canonical and private runners pass34 targeted CPU cases with zero
native/emulator construction. Actual queued commands and audit rejection fixtures
pass in1.476s. Native447fe50/SHA35ad0db and the original teacher remain unchanged;
the private runner adds only the fresh-training reward override. Both GPU guards,
62 allocation pins and338 retained pins pass. No GPU stage fails or repeats.
Execution source080c8b94, root `/mnt/data/kindle-breakout-reward10-20261106`.

The main completes21:31:10.844 UTC November6; collector21:31:15.605 and separate
CPU closure21:31:44.531. Both original services terminate successfully. Closure
recounts losses, initial/frozen tensor bytes, pre-update events and cost arithmetic.

Initial preparation for the subsequent full comparison rejects one closure-log
hash: the closure script hashed its open stdout before printing its final result.
The original failed helper execution and source remain preserved. A separate
resolution verifies that only this log differs and its complete JSON equals the
terminal summary without the file map; every other sealed file is unchanged.
The corrected full-helper check passes, including11 corrupted-header cases,
comparison direction checks, loss rejection fixtures, retained checkpoint schema
and an isolated full plotting fixture. No native/GPU stage is repeated or
original receipt rewritten.

This qualifies cost and execution, not natural-episode competence. A separate
[full200k protocol](../experiments/2026-11-06-cdp-breakout-reward10-200k.md) now
selects development22003 with its matching retained final/actual-initial controls
on cohort4000022003. No experience from the bounded check is transferred. The
original864/90%-two-wall and full five-game goals remain unfinished.

## Full200k comparison running

Execution source3be21d90 launches21:37:11.656 UTC November6, original PID1552095 /
invocationf6402af01ae946019805b26479f206de. All263 allocation pins verify. Initial
357 tensors and47 pre-update actions exactly match retained22003; first update376.
The100k/6,227 recovery passes independent CPU verification at21:48:13.435 UTC
in1.569s:357 finite tensors/moments,263 unchanged pins,0555 directory/0444 files
and unchanged original learner identity. The verifier performs zero model
forwards or learner updates. The recovery observer terminates successfully;
training continues past104k and the terminal collector remains active. No
frozen gameplay result exists yet.

Full-helper checks pass in4.343s. The first actual-command preflight fails before
native/emulator construction because its forbidden-API test double lacks the
`act` signature inspected by the runner. The separate repaired fixture adds only
that signature, still forbids execution and is AST-identical when this method is
removed. All five commands and four wrong-command checks pass in1.110s with zero
native/emulator construction and no GPU descriptor. Original source/failure/logs
remain pinned. The separate launch qualification verifies repair provenance,
then calls the unchanged declared launcher; no training code, queue or cap changes.

A read-only live-publication reader initially used system Python, which lacks
NumPy; it stopped on import before verification or writes. The failure receipt
is preserved. The corrected reader uses the declared Python environment and
verifies the running identity and pins; no learner or collector is affected.
