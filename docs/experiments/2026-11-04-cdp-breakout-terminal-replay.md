# Terminal supervision in the retained Breakout learner

The [ending-feedback diagnostic](../results/2026-11-04-cdp-breakout-ending-feedback.md)
still predicts high continuation one step before selected game-overs. Before
changing training, audit the retained learner's terminal replay exposure and
what its existing continuation summaries can establish about replay fitting.
This is CPU-only analysis within the five-game goal; no model execution,
emulator replay, native change or new learner is selected.

Use only the original replay-value1 training log: 200,000 actions, 12,477
updates, B16/T16 and 256 reported replay rows per update. Verify the log against
its retained guided replay and completion evidence. Check the exact native447fe50
sources: terminal arrivals supply zero continuation; all other flags supply
F32(1−1/333). The reported prediction mean and mean absolute error use the same
replay-posterior prediction array and target array. The head is sigmoid-bounded.
Do not conflate these summaries with per-example world-training BCE values.

Recover the terminal count in each batch by enumerating the exact F32 sum of
identical nonterminal targets, divided by256, for counts0…256. Require a unique
bit match after converting the logged mean back toF32. Independently verify
the nearest integer implied by the F64 target mean, within a conservative
5e-5 absolute mean tolerance. The reward-class row counts must also sum to256.
Count observed terminal arrivals directly from source transitions and reconcile
episode records. Repeated replay rows are exposure, not unique examples or
evidence that every terminal was sampled.

For each report letN=256, n be terminal count, gamma the nonterminal target,
P the prediction mean and M the mean absolute error. For predictionsc in[0,1],

    A = (N*(P+M) - (N-n)*gamma)/2
      = sum_terminal(c) + sum_nonterminal(max(c-gamma, 0)).

Thus terminal prediction sums lie between

    max(0, A - N*epsilon - (N-n)*(1-gamma))
    min(n, A + N*epsilon),  epsilon = 5e-5.

The allowance covers F32 reductions/subtractions and JSON round-trip error in
each reported prediction/error mean. For256 nonnegative bounded rows, the usual
F32 sequential-sum error bound is below this allowance. Validate the bound
against direct per-row arithmetic, including F32 accumulation, overshoot above
gamma, zero/all-terminal batches and perturbed summaries. Reject inconsistent
or nonfinite reports; do not silently relax tolerances.

Report whole-run and four fixed50k-action windows: update and row counts,
observed terminal arrivals, replay terminal exposures/fraction, batches with
terminal rows, count histogram, mean predictions/targets/MAE/BCE, and exposure-
weighted lower/upper bounds on terminal prediction means. Also count batches
whose entire terminal-mean interval is above or below.5, retaining undecided
batches. These are deterministic bounds, not point estimates or confidence
intervals. No subgroup, threshold or coefficient search follows the result.

Changing models and replay mixtures across training limit comparisons between
windows. Cached/recomputed replay posteriors differ from the frozen prior
forecast histories. These statistics cannot identify terminal examples, prove
population calibration, isolate representation or optimization, or measure the
gradient effect of a new loss weight. Preserve the original training and frozen
scores, including zero of24 two-wall successes.

Pin the current sources, historical native sources, log, checkpoint metadata,
protocol and all preceding diagnostic closure inputs. Qualify synthetic and
malformed-input cases before one analysis, capped at60s on one low-priority CPU
core. Block native and emulator imports and check no GPU descriptor opens.
Perform one independent saved-row recount and verify terminal artifacts. Do not
repeat a completed training, capture, replay or earlier collector.

Root `/x/Temp/kindle-breakout-terminal-replay-20261104`. The original
864/90%-two-wall and five-game goal remain unfinished.
