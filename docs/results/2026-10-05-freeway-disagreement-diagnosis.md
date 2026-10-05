# Freeway: disagreement barely changes exploration

The six completed runs'196,608 logged actions replay with matching rewards,
boundaries, resets and frame counts. Neither arm reaches player height160;
the highest reached in any stream is139. UP and DOWN each account for roughly
one third of requested actions; sustained UP lasts at most nine decisions.
The candidate's coverage is not better than the extrinsic-only control.
RAM labels are diagnostic-only:3,062 of3,072 sampled raw frames independently
match player-sprite position; ten ambiguous/invisible samples remain unchecked.
Original pixels were not logged, so this is not original-image equality.

## The bonus does not strongly distinguish useful actions

Restore each candidate's final checkpoint and re-encode its recorded stream0;
256 states/seed, all18 actions. This is the current model on recorded experience,
not the original online beliefs or a causal return experiment. All346 saved
model/optimizer tensors per checkpoint remain unchanged; no learner updates.

| Seed | Mean bonus | Within-state action SD | UP bonus exceeds DOWN | Highest-bonus action is DOWN |
| --- | ---: | ---: | ---: | ---: |
|1009|.00993|.000186|33.6%|57.0%|
|2017|.01213|.000170|38.7%|32.0%|
|3019|.01063|.000159|38.7%|41.8%|

Between-state bonus SD is9.7–17.6 times the within-state action SD. The current
actor captures less than1.1% of the one-step improvement available by selecting
the highest-bonus action. That is not a claim that greedy novelty would solve
the game: mean UP-minus-DOWN bonus is negative in all three seeds.
Next-sampled-latent prediction MSE is.140/.135/.149 versus same-cohort constant
baselines.146/.151/.158. These narrow diagnostics suggest weak action contrast
and noisy targets, not a proven root cause or a justification to scale up blindly.

## Precision discrepancy isolated, not hidden

Default cooperative GPU readouts differ from independent F64 MLP calculations
by maxima1.46e-5/2.19e-5/1.49e-5, failing the original3e-6 absolute bound.
The same GPU graph with native-F32 operands agrees within2.65e-8, without
changing that bound. Default results reproduce exactly on rerun. Pipeline keys
confirm cooperative versus scalar GPU matmuls; production arithmetic is unchanged.
UP-versus-DOWN preference agrees at all768 states; best-action identity agrees
98.4%/100%/98.0%. This isolates reduced-input matrix precision as the discrepancy,
with no evidence that it caused zero reward discovery. No CPU learner workaround.

Nine frozen diagnostic guards and retained-log seals pass. One additional
readout stopped at graph construction because the new probe omitted explicit
inference mode; corrected before the new queue. The initial color-mask failure,
missing-Cargo build and failed F64 check remain retained. No recovery or NVML
polling; GPU utilization remains unmeasured. This is not new learning evidence.

## Next decision

Test one change: ensemble targets become detached posterior categorical
probabilities, replacing fresh one-hot latent draws. For squared error, this
preserves the expected predictor gradient while removing categorical sampling
noise. It adds no encoder, privileged state, action aid or world-model gradient.
Do not change the bonus coefficient, optimizer or interaction budget at the same
time. This is a falsifiable hypothesis, not a promise of reward discovery.

[Declaration](../experiments/2026-10-05-freeway-disagreement-diagnosis.md) ·
[Compact evidence](2026-10-05-freeway-disagreement-diagnosis.json).
Raw artifacts: `/x/Code/kindle/runs/freeway-disagreement-diagnosis-20261005.IJdvatPP`.
