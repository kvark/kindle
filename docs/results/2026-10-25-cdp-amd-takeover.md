# AMD takeover: Breakout context experiment completes without improvement

The fresh seed1009 B2/T64 life and fixed evaluations completed. Frozen scores
are14.71 at500k,10.04 at2M and1.42 initially, with zero two-wall successes in
each24-episode selection. Retained B8/T16 scores are58.83/98.21/1.42. This AMD
F16-input run is slower and learns much less; context/batch, precision and host
remain confounded. The five-game reliability goal remains unfinished.

## Source and host

- Native library at `f4bb2ca`; all329 library files match the handover receipt.
  Meganeura423d3736, Bladebbbc93b7, Naga13f582dd. No learning-kernel changes.
- RX7900XT,20GiB VRAM, device0x744c, Mesa RADV26.0.8-1ubuntu0.3;
  Linux7.0.0-34-generic, Rust1.95.0, CPython3.13.13. Python packages match
  `tools/handover/requirements.txt`.
- Boot `09aba711-7628-4bb6-830f-4aff41048556`;
  amdgpu srcversion `4B803D5D5A2CA02E40497CD`.
- Release extension SHA256:
  `6cb2677f2bc33f0aaced090ed7b6f328533057893672434c6b30c0ef08dec7fd`.
- Archive SHA256 and all304 member hashes verified. Evidence, declarations,
  controller and weights: `/x/Temp/kindle-amd-20261025`, outside Git.

The user confirms `/mnt/data/kindle-handover-20261025` is the complete available
transfer. Its manifest contains only the context-smoke and B8/T16 Breakout
checkpoints. Boxing, Pong, Freeway and Qbert scores remain historical evidence;
their trained weights are not available on this host. Preserve that distinction
when planning verification after the current learning comparison.

The host guard now accepts explicit `driver_module: amdgpu`, checks the loaded
module srcversion, and recognizes AMD page faults, ring timeouts and GPU resets.
NVIDIA declarations keep their default.150 CPU guard/queue tests pass. No driver
reset, reload or reboot was performed.

## Compatibility and smoke results

Original BF16 restore exits at the capability assertion: this device/driver lacks
the required16x16x16 BF16/F32 products. A separate checkpoint copy changes only
`config.training_precision` to `f16`; all tensor files remain identical. Its
first restore retains `KINDLE_DEVICE_TARGETS=1` and exits because staged world
training is BF16-only. Both failed attempts are retained; neither is a GPU fault.

Omitting that flag uses the existing ordinary world/target path. Restore passes
1024 aggregate actions/4095 emulator frames with zero updates and all346 tensors
unchanged. Construction13.14s; acting/export0.827s. Export counters867/6144;
the original shared export remains867/5120. F16 permits reduced matrix inputs
with F32 parameters, optimizer state and accumulators; this is not BF16 parity.

Source review confirms `F16` selects Meganeura's `AllowF16`, which permits
uncompensated F16 derivative operands and bypasses its normal gradient-precision
protection. Kindle's session builder adds no numerical loss scaling. Finite
losses and F32 parameter storage therefore do not establish gradient fidelity.
No actual underflow or causal score effect has been measured. This remains an
unresolved precision limitation; see `precision-source-review.json` for pinned
source paths/hashes and the question to revisit after the fixed evaluation.

CPU inspection of the1M optimizer state finds all108 owned parameter tensors
changed since both the actual initial and500k checkpoints; every stored second
moment entry is positive. This rules out wholly inactive parameter blocks over
those intervals, but leaves partial gradient loss untested. Evidence:
`life/optimizer-activity-1000000.json`, including source hashes and scalar counts.

Fresh smoke: centered CDP12M/B2/T64/H15/R32, full-sequence/full-batch updates,
action-effects1, replay100k, eight independent environments with batched inference
and shared GPU context. Native max-pooled frames receive one GPU64 resize.
Full18 actions, sticky.25, repeat4; no action override.

4096 aggregate actions produce16353 emulator frames and867 updates, first at
action632, zero debt. Construction12.76s; run73.849s,55.46 actions/s. Mean update
82.16ms; learning71.30s. Historical BF16 B2/T64 took45.245s: this takes1.63 times
as long. GPU, precision, execution path, driver, compiler and Python differ.

Six10-second sysfs samples during learning show94% GPU busy; the final sample
after completion shows63%. Sampled VRAM peaks at2.074GiB. These are coarse
device-wide samples, not exclusive process utilization or achieved FLOPS.
High utilization does not establish optimal kernels or faster learning.

Independent accounting and finite checkpoint checks pass. Subsequent1024-action
frozen evaluation produces zero updates and preserves all346 tensors. Receipts:
`restore-review.json`, `smoke-review.json`, `smoke/utilization.jsonl`. These checks
establish execution/accounting/restore, not competence or a gradient oracle.

## Completed life and fixed evaluation

User service `kindle-amd-context64-20261025` finished successfully at15:06:49UTC.
It ran the fresh2M-action life, sequential frozen evaluations and CPU audits,
with12GiB process-memory cap and no swap. That learner has exited; the subsequent
[precision comparison](2026-10-25-cdp-amd-precision.md) is tracked separately.
Artifacts: `/x/Temp/kindle-amd-20261025/life`.

Training completed2,000,000 aggregate actions/7,993,225 emulator frames and499843
updates in42025.27s (11h40m25s,47.59 actions/s), excluding construction. First
update is632; replay finishes at100k and debt is zero. All499843 learner reports
are numeric, finite and sequential; final configuration/counters, tensor hashes
and all346 tensors pass. Native identity is unchanged. Evidence:
`checkpoint-2000000-review.json` and `learning-review.json`.

The completed curve is in `curve-final` (PNG/SVG/JSON/CSV), independently
reconciled against the existing completed-run reader. Online last-50 returns:

| Actions | AMD B2/T64 F16 | Retained B8/T16 BF16 |
| --- | ---: | ---: |
|500k |12.54 |65.76 |
|1M |11.32 |66.02 |
|1.5M |8.96 |112.68 |
|2M |10.48 |57.10 |

All fixed policies receive the first three natural sampled episodes per stream,
24 selected episodes per arm, seed4000001009. No checkpoint selection occurred.
Whole-run action/frame counts below include extra completed episodes and partial
tails; scores use only the fixed24. Acting/export times exclude construction.

| Policy | Selected mean | Two-wall successes | Aggregate actions | Emulator frames | Acting/export seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
|500k |14.7083 |0/24 |13472 |53856 |7.615 |
|2M |10.0417 |0/24 |10064 |40219 |5.773 |
|Actual initial |1.4167 |0/24 |5944 |23719 |3.564 |

Every arm reaches its natural-episode budget without an action-cap cutoff,
produces zero updates and preserves all346 tensors exactly. Complete CPU action
replays match; whole stream0 videos are retained as `prefix.mp4`, `final.mp4`
and `initial.mp4`. All four host guards pass, with no unsealed files or recovery
operation. The actual initial tensors match the smoke. None of the task gates
passes, and one reused development seed does not establish reliability.

Learning occupies41056.19s (97.7% of training wall time). Reported GPU memory
plateaus at7.923GiB; coarse learning-time busy samples are typically93–94%.
These measurements do not establish kernel efficiency. The historical B8/T16
2M life took8991s; this life takes4.67 times as long under the declared confounds.
The old8h cap was explicitly adapted to14h; it was not reached.

No unchanged extension or seed sweep follows this weak result. Subsequent paired
early updates show precision-dependent optimizer/parameter changes despite
matching initial tensors and experience; raw gradients were not captured.
[Precision evidence](2026-10-25-cdp-amd-precision.md) records that scope and the
fresh500k Auto learning test. A controlled learning comparison must establish
whether the update difference matters for scores; numerical checks alone cannot
satisfy the five-game goal.

## Intermediate evidence retained

Checkpoint audits also pass at500k (1,998,287 frames/124843 updates),1M
(3,996,815/249843) and1.5M (5,995,063/374843). Each checks exact configuration,
counters, tensor hashes, all346 finite tensors and numeric/sequential learner
reports; native identity stays unchanged. Receipts: `prefix-checkpoint-review.json`,
`checkpoint-1000000-review.json` and `checkpoint-1500000-review.json` under `life`.

The earlier curve through770048 actions/3,077,690 frames is retained in
`life/curve-live-first`; its last-50 return is10.76 after16145.98s. The completed
curve above supersedes that snapshot for final interpretation.
