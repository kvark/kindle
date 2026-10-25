# AMD precision: early updates differ despite matching losses

The completed F16 context life is weak: [500k/final/initial14.71/10.04/1.42](2026-10-25-cdp-amd-takeover.md).
Small paired production runs now show a precision-dependent update difference.
They do not prove the mechanism or its effect on long-run scores. A fresh
[500k Auto comparison](../experiments/2026-10-25-cdp-amd-auto.md) is running to
test learning while holding the AMD host, B2/T64 configuration and seed fixed.

## Paired production updates

Both arms use the unchanged installed native library at `f4bb2ca`, SHA256
`6cb2677f2bc33f0aaced090ed7b6f328533057893672434c6b30c0ef08dec7fd`, on the same
RX7900XT/RADV host. Only `training_precision` changes: `f16` selects `AllowF16`;
`auto` retains Meganeura's derivative-precision protection. No runtime rebuild,
kernel modification, loss scaling or driver operation was performed.

Fresh seed1009, centered CDP12M/B2/T64/H15/R32, full BPTT/microbatch2, replay100k,
action-effects1, eight independent lanes and native frames/one GPU resize stay
fixed. All346 initial tensors match each other and the completed life's actual
initial checkpoint. Both632-action trajectories (79 vector records) and both
640-action trajectories (80 records) match exactly, including actions, rewards,
termination flags and frame counters. The first two world/behavior reports also
match exactly across precision modes; timing fields are excluded.

The632-action diagnostic performs one optimizer step. Step-indexed learning-rate
warmup makes its learning rate zero, so parameters stay unchanged while moments
differ. That result is retained. A separately declared640-action diagnostic
performs three updates, including the first two nonzero-rate parameter changes.
All accounting, finite tensor and host-guard checks pass for all four arms.

After three updates, `world.reward.layer0.weight` has655360 elements:

| Observation | F16 | Auto |
| --- | ---: | ---: |
| Parameters changed from actual initial |0 |654080 |
| Nonzero stored variance entries |0 |654080 |

F16 also has zero momentum throughout this tensor. An independent tensor read
verifies these counts. This is early suppression/delay, not a permanently frozen
block: the earlier1M audit found all108 owned tensors had changed.

Across all owned tensors, the world parameter-delta vector differs from Auto by
1.626% in relative L2; the behavior delta differs by73.876%, cosine0.7303.
These compare the small warmup updates, not absolute parameter magnitudes;
maximum behavior-parameter difference is3.15e-7. Aggregate norms can hide the
completely unchanged reward projection. Known-vector checks validate the metric
calculation independently.

Raw derivatives and individual matrix operands were not captured. First-step
optimizer-input gradients can be estimated from zero-initialized LaProp moments,
but these are after clipping and F32 rounding. Later moments contain history;
the three-update analysis does not reconstruct individual gradients. Replay
indices and sampled latent draws are not exported; seeds and actual experience
match. Auto is a protected-policy control, not an independent mathematical oracle.
The observations support a controlled learning comparison, not a claim that
F16 underflow caused the completed score regression.

Evidence outside Git: `/x/Temp/kindle-amd-20261025/precision-update` and
`precision-update-three`, including declarations, guarded queues, checkpoints,
`update-review.json` and the latter's `independent-check.json`. The original
three-update declaration and its stale-prose correction are both retained;
operational budgets were640 actions/three updates throughout.

## Auto smoke and next learning test

Fresh4096-action Auto training produces16350 emulator frames and867 updates,
first at632, with zero debt. All reports are numeric, finite and sequential;
checkpoint configuration/counters/hashes and all346 tensors pass. Actual initial
tensors are identical to the F16 life. Training takes72.771s,56.29 actions/s,
80.889ms mean update. The earlier F16 smoke takes73.849s,55.46 actions/s,
82.164ms mean update. These single short runs have similar throughput; they do
not establish a speed advantage or steady-state utilization.

The1024-action frozen check produces4090 emulator frames, zero updates and exact
equality of all346 tensors. Both guards pass. Evidence: `auto-smoke/review.json`
under the same artifact root. Native identity remains unchanged.

The fresh500k Auto life is owned by `kindle-amd-auto500k-20261025`, with artifacts
in `auto-life`. It compares against the completed AMD F16 policy's fixed500k
score14.7083 and actual-initial1.4167. Only training precision changes. Fixed
final/initial24-natural-episode evaluations, tensor equality and full CPU
replay/videos follow automatically; no extension or replication follows without
review. Seed1009 is reused development evidence, not an independent seed.
