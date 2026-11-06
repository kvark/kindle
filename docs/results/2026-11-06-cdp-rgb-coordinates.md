# RGB coordinate channels pass bounded execution and cost checks

The fresh check completes **8,192 actions /32,713 emulator frames /489 updates
in47.565s**. A1,024-action frozen restore takes.882s with zero updates and no
completed natural episodes. The padded200k projection is **21m17.383s**, below
the fixed30-minute cap. Gameplay benefit remains unmeasured.

The [protocol](../experiments/2026-11-04-cdp-rgb-coordinates.md) adds fixed x/y
pixel-center channels inside the native learned RGB CNN. First-layer weights
increase by1,600, giving16,335,953 total parameters; replay remains RGB and
receives one GPU64 resize. Other settings stay centeredCDP12M64/Auto/N8/B16/
T16/H15/R16, encoder4e-5/dynamics4e-4/behavior8e-5, critic1/1, expert100 and
20% episode guidance/cap1920. Initialization changes; retained-baseline tensor
equality is not claimed. Native and runner also differ from retained447fe50.

All47 original pins,357 finite checkpoint tensors,357 invariant frozen tensors,
correct cadence,weighted reports and changed CNN/reward/continuation/actor/value
weights pass. Full CPU replay verifies every recorded pixel/action/label and
3,823 overrides (46.6675%). All489 updates receive supervision. Guided scores
and own-replay fitting do not measure frozen competence.

## Qualification and preserved failures

Source2d651cb0/nativeSHA256
`b913646707ce01aa2c6f621ed27e56892296a5f8b6120eabd6b2fa53f750130d`
passes186 Kindle+5 gym CPU tests,1,031 packaged Python tests and strict Clippy.
One guarded independent F64 CNN fixture checks outputs,loss and RGB/x/y/later
weight derivatives. The default-off legacy final/initial restores each reproduce
128 transition rows with357 invariant tensors,2,048 total actions and zero
updates. Native and legacy qualification complete November4; they are not repeated.

Earlier installed-package import and restore-fixture failures remain preserved.
The build's final executable matcher incorrectly includes the gym binary; a
separate CPU packaging repair selects the already-built Kindle executable.
A bounded command fixture incorrectly expected no empty output directory;
its original failure and corrected expectation remain recorded.

The original bounded controllerPID1191398 completes both guarded GPU stages
on November4, then fails importing `kindle._breakout_guidance` in its CPU audit.
The user pause arrived before the launch and was noticed afterward; training
and frozen assessment finished before cancellation could reach them. No process
remained active, and no further experiment ran during the pause.

After explicit resume, a separate **6.995s CPU repair** supplies the matching
runtime on `PYTHONPATH` and completes the original unchanged audit, including
the previously unrun5.542s trajectory replay. Closure is
`complete_with_cpu_audit_repair` at **04:25:53.730 UTC November6**. All24 repair
pins and47 original pins remain unchanged; original audit return1 is preserved.
No successful GPU,build,test or replay stage repeats.

The conservative memory check subtracts the entire5,939,200,000-byte full replay
from minimum observed headroom, leaving12,066,185,216 bytes above the2GiB gate.
This is a projection, not proof of sustained full-run memory or speed. The
47.565s check versus retained47.690s does not establish a runtime improvement.

Evidence remains at `/x/Temp/kindle-rgb-coordinates-20261104`; the repaired
completion is `bounded/check/completion-repaired.json`, reviewSHA
`ee7900519838e606a451f1051fae6be362e164fd7e0d2f5ba22ab815230c67a3`.
Do not repeat completed stages. A separately declared
[fresh200k comparison](../experiments/2026-11-06-cdp-breakout-rgb-coordinates-200k.md)
will measure whether explicit image position improves frozen gameplay per minute.
The original864/90%-two-wall gate and five-game goal remain unfinished.

## Fresh comparison startup

The separately declared200k learner launches **04:30:28.229 UTC November6**
on sourcecb9a0499, with124 fixed pins. MainPID1386077, invocation
`c11b782f30804e6f951712401a64d0d8`; service
`kindle-breakout-rgb-coordinates-200k-20261106`. All357 initial tensors match the
bounded architecture and47 pre-update events match. The first update occurs
at376 actions. Recovery andterminal observers are active; no frozen full result
is available. Check live process identity before any further GPU work.

Full CPU preparation first fails only in its plot fixture, which retained the
previous experiment's path andlatent-mixing expectation. Both are corrected in
that fixture; a separate1.111s CPU continuation completes the remaining checks,
without repeating prior successful helper stages. The complete comparison entry,
config/schema,allfive mocked queued commands andreplay paths pass. Original
failure,fixture source andrepair receipts are pinned. The1.009GB disposable
build cache is released; wheel,native runtime andvalidation executable remain.

The [SuperMimic review](2026-11-06-supermimic-architecture-review.md) records a
separate connected-actor-graph proposal. It is not part of this comparison and
has no selected GPU allocation.
