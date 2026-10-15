# Complete compact-channel convolution reuse policy

Declared October15,2026 after the [output-contract audit](../results/2026-10-15-output-contract.md).
Current complete RGB remains51.945ms/27.223ms inclusive JAX. This design targets
the measured feeding/reuse difference, including the nine expensive checked
occurrences; it is not another loader-count, precision or geometry sweep.

One shared typed algorithm covers all23 actual occurrences/20 contracts:

- Pack channels to multiples of four, matching the existing four-element vector
  transfer, instead of padding every channel group to32. Include all input/weight
  packs, canonical outputs and existing split reductions in complete operation
  cost. No im2col, halo cache, new encoder or library/CUDA learner.
- For forward/dX, choose M32 when output channels<=32, otherwise M64; N is
  min(spatial extent,8192/M). K stage remains32. This bounds logical accumulator
  state to64 F32 values/computing lane. For dW, use the same M and N=4096/M,
  bounding the three compensated states to96 logical values/lane. These are
  source-state bounds, not measured registers/spills. No empirical tile search.
- Four full32-lane subgroups share one padded input slot and4KiB fragment output
  scratch. Use the existing cooperative operations and ordinary whole-workgroup
  barriers. No producer/consumer split, async-copy claim or new Naga feature.
  Current packed-style16-byte padding fits all three scalar types with one slot.
- Keep original512 images, dW partition ranges,256-term compensation and final
  reduction order. Channel packing can reorder forward/dX terms relative to
  checked kernels; disclose that and retain the independent numerical gates.
  Do not assume bitwise whole-update trajectories. dW's spatial reduction order
  stays fixed. Preserve native BF16 rounding including exceptional-value handling.

First perform CPU preparation: exact current-selection reconciliation; all
typed shader compilation/validation; ownership and packed-address checks including
three/48-channel cases, tails and every partition; independent payload/MAC/live
state accounting. No GPU allocation until that result is reviewed. Preparation
is1CPU/2GiB/no swap; artifacts `runs/reuse-policy-20261015.Yy1qEpm6`.

If the CPU contract is sound, separately declare one complete numerical and
timing allocation against the unchanged native67559ce5 and original GLSL serial
controls. All23 occurrences remain in the weighted result, including regressions;
packing/reductions/canaries and changed/restored input checks are mandatory.
Original forward/dX/dW pointwise/L2 gates stay; old cancellation failures remain
separate evidence. Require aggregate serial calibration within10% and at least
20% lower complete-family GPU and queued-wall cost against both controls before
considering native integration. One fixed policy, no retiming, favorable subset,
gate relaxation or automatic full-update/learning campaign.
