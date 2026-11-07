# CPU reconstruction of Breakout replay-context age

The retained sampled70.750/greedy143.667 Breakout learner uses cached recurrent
context and refreshes sampled states from the pre-update posterior. The smaller
replay comparison regressed but did not measure cache age. The completed H3
comparison also regressed. Select one CPU reconstruction of the retained
200k-action control's replay bookkeeping before choosing another intervention.

Use only the original training log at
`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009`.
Preserve its338 declaration pins, completion, learning review and log hash.
Resolve exact native447fe50 replay, scheduler, RNG and vector-arrival sources
from Git, and the pinned cf4ea238 runner. No emulator, native learner, checkpoint
restore, replay video, GPU work or training occurs. Do not rerun prior readers.

Compile the original replay module in a private CPU harness. Use small scalar
markers in place of observations and latent values; selection, eviction,
fresh-first sampling, reset flags and context refresh execute the original
source. Every device function must fail if reached. Reuse pinned Rust rand0.9.4
and its ChaCha12 dependency, with seed1009 XOR the exact replay stream constant.
Verify source/dependency identity before analysis. Geometry-invariance fixtures
must establish identical sampling when marker widths change.

Stamp each arriving context with the current completed-update count and each
sampled refresh with the pre-update count, following the original runner's
transition→learn→reset order. Reproduce the original numeric scheduler. Reconcile
every logged learner step, replay length, labeled-row count and weight sum,
progress debt and final counters. Stop on any mismatch; preserve the failed
attempt and diagnose it separately. Matching bookkeeping does not recreate
unlogged GPU tensors or prove full execution equivalence.

Report all199,632 sampled sequence starts (12,477 updates ×16 rows). Fix action
windows1–50k,50k–100k,100k–150k and150k–200k before analysis. For all contexts and
the subset not masked by a first-frame reset, report mean,median,p90,p99,max
age in learner updates, exact counts at ages0,1–15,16–63,64–255,256–1023,1024+,
and whether the last computation was collection or replay refresh. Separately
count labeled actor rows potentially depending on the initial context before
the first reset in each sequence. Keep ignored contexts separate. No phase,
threshold, seed or outcome-based subset selection.

Fixtures must cover collection/refresh stamps, fresh-first selection, eviction,
stream separation, reset masking, outgoing-label alignment, geometry invariance
and rejection of inconsistent logs. Pin the harness, executable, dependencies,
historical sources, protocol and inputs before reading the full outcome.
Cap CPU compilation/qualification at180s and the one analysis at60s. Retain all
execution receipts and hashes under
`/x/Temp/kindle-breakout-replay-age-20261103`; keep large evidence outside Git.

This measures the age of the last state computation, not latent error, inherited
history age, parameter movement or a causal gameplay bottleneck. Live states
also carry history across updates. There is no automatic threshold for a new
loss, burn-in, buffer change or learner. Select any subsequent experiment
separately from this evidence. The five-game goal remains unfinished.
