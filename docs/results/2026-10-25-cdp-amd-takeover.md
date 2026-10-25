# AMD takeover: restore passes and fresh Breakout training starts

The fresh seed1009 B2/T64 life is running. The supported AMD precision path is
**slower than the historical NVIDIA BF16 smoke**; no new full-life score is
available. The five-game reliability goal remains unfinished.

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

## Running life and review boundary

User service `kindle-amd-context64-20261025` owns the fresh2M-action life and
sequential frozen evaluations, with12GiB process-memory cap and no swap.
Artifacts: `/x/Temp/kindle-amd-20261025/life`. Expected499843 final updates,
124843 at500k. The actual initial checkpoint is saved before experience.
Fixed initial/500k/2M policies receive24 selected natural sampled episodes each,
three per stream, seed4000001009; complete action replays and whole stream0
videos follow. Every frozen export is compared tensor-for-tensor. Failures stop
the queue. No automatic extension or seed sweep follows.

The first fixed500k checkpoint is saved after about2h55m:1,998,287 emulator
frames and124843 updates. CPU review verifies its exact configuration/counters,
all three tensor-file hashes and all346 finite tensors; all124843 learner
reports in the fixed prefix are finite and sequential. The installed native hash
is unchanged. Evidence: `life/prefix-checkpoint-review.json`. Training continues
past this checkpoint without restarting; its frozen evaluation remains queued
after the uninterrupted2M life. Online last-50 return is12.54, not a frozen score.

The nearest preceding progress record (499712 actions) reports10467.08s,
47.74 aggregate actions/s and zero debt. Replay is full at100k; reported GPU
memory has plateaued at7.923GiB, with recent device-wide busy samples93–94%.
These measurements do not establish kernel efficiency or playing reliability.

Measured throughput projects roughly12h. The receiving-host declaration changes
the old8h training wall cap to14h;2M actions, ratio32 and128 replay positions per
update stay fixed. Evaluation caps remain600k actions/1800s; CPU audits3600s,
2GiB/one CPU. Review `life/learning-review.json` before choosing another test.

Compare with retained B8/T16 scores58.83/98.21/1.42. Disclose both coupled
sequence-length/independent-batch changes and precision/hardware differences.
Seed1009 is reused development evidence, not held-out reliability.
