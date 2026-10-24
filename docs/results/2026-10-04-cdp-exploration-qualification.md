# CDP disagreement: qualified for the Freeway screen

The four-head latent-disagreement mechanism passes numerical, alignment,
detachment, zero-scale and production/restore checks. This is not yet Freeway
reward discovery. The [declared six-run screen](../experiments/2026-10-04-cdp-freeway-exploration.md)
is the next learning test; the previous five-game matrix remains deferred.

## Implementation

Four small RMSNorm/SiLU MLPs predict detached next categorical RSSM samples from
detached previous features and executed actions. Reset arrivals have zero weight.
The existing world optimizer/checkpoint owns the heads; their loss cannot shape
the encoder or RSSM. Current predictions supply the same action-aligned bonus to
imagined and replay-value targets. Observed reward prediction stays extrinsic-only.
No second image encoder, acting-path readback or scripted action aid.

Use `atari_vector.py ALE/Freeway-v5 --cdp --disagreement-scale 1 ...`.
Coefficient zero removes heads and computation. Frozen restore uses saved config.

## Checks

- 113 Rust CPU tests,1,137 Python tests, formatting and strict workspace/binding
  Clippy pass. Five guarded GPU tests plus three production jobs pass.
- Independent F64 references cover3 bonus rows, masked loss and60 raw derivatives,
  including identical and nearly identical predictions. No tolerances relaxed.
- Repeated-state disagreement0.5303→approximately0 after256 synthetic updates;
  held-out-state disagreement0.5978→0.5809. State/action/target gradients are absent.
- Independently expanded return recurrences verify action-to-arrival alignment,
  replay reset masks, terminal/truncation boundaries and actor targets.
- Four zero-scale updates exactly match disabled control parameters, optimizer
  moments, reports and posterior/imagination RNG draws.
- N8 production checks:2,048 actions/451 updates per arm, then1,024 frozen
  candidate actions/zero updates. All346 saved model/optimizer tensors remain
  unchanged. Guards, sealed artifacts, counter ledgers and finite checkpoints pass.

The candidate has positive mean intrinsic reward (imagined0.2048/replay0.2027)
and absolute advantages0.5759 with zero real rewards. These are wiring checks,
not a game score. All5,120 production-check actions are excluded from learning.

## Cost and limits

Final1,024-action/256-update windows: control24.94ms/update and149.2 actions/s;
candidate24.43ms and153.4 actions/s. No measurable slowdown in this short sequential
check; do not call the small difference a speedup or precise overhead estimate.
No compilation overlapped timing. GPU utilization remains unmeasured.

All eight successful guards record zero new allocation/validation errors, no NVML
polling and no recovery. One earlier guarded graph-construction assertion is
retained: a scalar-only helper was called on a matrix. It failed before training;
switching to the tensor scaling API and adding CPU graph coverage fixed it.
Two earlier CPU preparation/schema failures are also retained, not hidden.

Backend: Meganeura592a2f5 / Bladee349cddf. Full identities, tolerances and results
are in the [compact data](2026-10-04-cdp-exploration-qualification.json).
Raw root: `/x/Code/kindle/runs/cdp-exploration-20261004.mE26yV2V`.
