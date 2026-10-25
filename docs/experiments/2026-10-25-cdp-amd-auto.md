# Breakout: protected Auto versus raw F16 training products

Declared after the completed AMD B2/T64 F16 life and paired early-update checks;
see [precision evidence](../results/2026-10-25-cdp-amd-precision.md).

Question: does protecting derivative precision improve learning on the same
AMD B2/T64 setup? Raw gradient operands have not been captured; the measured
early update difference is a reason to test learning, not its explanation.

- Fresh seed1009, centered CDP12M, B2/T64, full BPTT/microbatch2, H15,
  replay-ratio32,128 positions/update, replay100k, action-effects1.
- Change only `training_precision` from `f16` to `auto`. Keep the same native
  binary, RX7900XT/RADV, runtime packages, device flags and ordinary-target path.
  `KINDLE_DEVICE_TARGETS` remains omitted. No optimizer, objective or kernel edit.
- Eight independent Atari lanes, batched native inference/shared GPU context,
  native max-pooled frames and one GPU64 resize. Full18 actions, sticky.25,
  repeat4, no action aid.
- Budget500000 aggregate actions, expected124843 updates, first at632. This is
  a fresh life; neither the smoke nor a completed training checkpoint is extended.
  Save actual initial and fixed final checkpoints. No checkpoint selection.
- Evaluate final and actual initial using the first three natural sampled
  episodes per stream,24 per arm, seed4000001009. Each arm has600k-action/1800s
  caps, zero updates and exact frozen tensor checks. Replay all actions on CPU,
  retain whole stream0 videos, and report caps/truncations and two-wall successes.
- Compare with the completed AMD F16 fixed500k score14.7083 and initial1.4167,
  plus learning curves by actual actions, frames and wall time. The F16 control
  used the same configuration apart from precision and the same reused seed.
- Training wall cap4h; CPU audit1h; service cap6h5m includes both frozen caps.
  Existing AMD host guard,12GiB process-memory cap/no swap, stop on real GPU or
  numerical failure. No retries, reset/reload/reboot, or automatic extension.

Interpretation: this is one paired development comparison, not held-out or
multi-seed reliability. A better mean alone does not establish Breakout mastery,
CDP superiority, or the mechanism behind precision-dependent updates. Review
the fixed results before selecting another learning test or replication.

Artifacts/controller: `/x/Temp/kindle-amd-20261025/auto-life` and
`/x/Temp/kindle-amd-20261025/run_auto_life.py`. Machine-readable declaration,
controller hash, exact queue, native identity and retained-control hash are
recorded before launch. Service: `kindle-amd-auto500k-20261025`.
