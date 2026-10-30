# Intermediate-capacity Breakout with preserved ball detail

Prepare one fresh seed1009 after reviewing the active Qbert screen. Test whether
the intermediate model with T16 yields useful frozen learning per GPU-hour.
The completed12M/B2/T64 lives remain weak and are not extended. Historical1M
Breakout is also weak; use4M and preserve64x64 input because the visibility
samples attenuate the tiny ball at32. This is a new development recipe, not a
controlled attribution to model size or temporal context.

Centered CDP/action-effects1, Size4M/RGB64/Auto, N8/B8/T16/H15,
full BPTT/microbatch8, replay100k and ratio32. Full18/sticky.25/repeat4,
native max-pooled frames and one GPU resize. No action aid, RAM input, pretrained
encoder or reward change. Record the actual parameter count from the runtime.

Fixed200k aggregate actions, expected49939 updates, first update248, zero debt.
Training cap1800 seconds including construction. Save actual initial weights and
a100k checkpoint, replacing it with the fixed200k final. A cap cutoff preserves
the100k milestone without authorizing a resume or a successful-budget claim.
No best-checkpoint selection or automatic extension. The4M/RGB48
smoke supplies the complete config reference, changing only image geometry and
ratio. Its tensors are not an initial-weight equality control for4M/RGB64.
Check the new initial weights for finite values and zero counters.

Frozen evaluation uses seed4000001009, the first three natural sampled episodes
per lane (24 per arm), zero updates,600k-action/300-second caps per arm. Keep
excess episodes, tails and cutoffs. Retain the existing Breakout task gate:
at least20 completed episodes and90% two-wall (864-point) successes. A higher
score than the initial policy is a useful gain, not completion of that gate.

Audit every numeric report, complete config/source identity and action/frame/
update accounting. Check finite checkpoints and all346 unchanged tensors for
each frozen arm. Replay every recorded action on CPU and preserve stream0 videos;
audit cap1200 seconds, service cap3660 seconds. Any cutoff or failure is retained
without increasing the allocation or restarting the learner.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-200k`.
Controller: `run_breakout_200k.py`; intended service owner:
`kindle-cheap-breakout-r32-200k-20261025`. Commands, caps and source/native hashes
are declared before launch. Use the qualified native runtime without a rebuild.
Launch only after Qbert's completed review; keep one learner. Seed1009 is reused
development data, and promising recipes still require confirmation across seeds.
