# CDP BF16 loss boundary and actual staged-world qualification

Declared October15,2026, before new GPU work. The existing-policy comparison
finds78.206→50.122ms complete CDP updates, but finite trajectories do not qualify
its numerics. No timing repeat, new precision policy, optimization or learning.

Use the same categorical-GPU Kindle/general-ordering Meganeura/Blade/Naga
source lineage. Add test-only harnesses in an isolated source copy; no backend
algorithm, production source or compiler feature change. Verify source seals.

## Independent loss/gradient boundary

Exercise the production BF16 graph conversion/derivative options around a
learned projection followed by the actual CDP detached-mean/cosine loss. A
trainable per-row additive probe exposes the unrounded incoming derivative;
it is a diagnostic observer, not a proposed model parameter. Targets and their
batch mean must remain detached. Scale the mean loss by500, as in the recipe.

Four fixed shapes `(rows,input,output,centered)`: `(4,16,256,false)` for zero/
tiny-norm rows, `(3,17,19,true)` for tails, `(128,256,256,true)` and the actual
12M final-predictor shape `(512,1024,1024,true)`. Each has two deterministic
dyadic input/weight/target/probe sets, including a common target/probe component
for centering. CPU source/config checks establish the two model widths. GPU
exports all inputs, forward outputs, raw input/weight/probe gradients and plan
summaries. No optimizer or clipping; no actor updates in this boundary check.

Independent CPU F64 equations use explicit round-to-nearest-even BF16 operands
and forward-result rounding. Forward predictions must be within one BF16 ULP
of independently rounded contraction, plus F32 addition roundoff for the probe.
Cosine distances retain3e-6 absolute error; scaled loss uses3e-6×500. Probe
derivatives retain2e-5+2e-5×abs(reference), plus relativeL2<=2e-5. The scalar
reference uses the separately checked actual prediction to isolate this loss
boundary rather than compound a permitted forward rounding difference.

Backward matrix operands explicitly round the incoming derivative to BF16.
Their independent reference uses that rounded F64 derivative. Predeclare a
format-aware pointwise envelope: propagate one BF16 ULP of incoming derivative
plus32×F32 unit-roundoff×its row RMS through absolute operand contractions;
add the standard gamma_K F32 accumulation envelope and one F32 output ULP.
Also require relativeL2<=1e-2 for each complete input/weight gradient. This
allows quantizer-boundary sensitivity, not arbitrary changed summation error.
Report actual errors/envelope usage; do not change limits after seeing data.
These are explicit BF16-boundary checks, not F32 equivalence, a finite-difference
derivative of rounding, or a replacement for any earlier failed F64 gate.

## Same-graph actual world execution

Only after the independent boundary review passes, restore the exact CDP12M/
B32/T16 BF16 fixture and run one initializer update. Require exact346 initial/
first saved tensors and first targets/reports against the prior BF16 cost run.
Then freeze weights/moments and explicitly disable optimizer, clipping and
accumulation. Use the existing compiled prepared graph, allocations/dependencies
and cached execution; whole and staged calls must not rebuild different graphs.

Compare five backward passes: original whole/staged, changed-late-target whole/
staged, then original whole restored. Before each staged prefix poison the
reward/continuation/replay-value targets and weights with NaNs; the finite
prepared posterior/encoding outputs must equal the whole control exactly.
Restore the intended late inputs before finishing. Changed reward targets rotate
the existing distribution one bin and continuation labels flip; no early inputs
change. Full outputs and every raw trainable world gradient must match the
corresponding whole call exactly; changing late targets must change the tail,
not the prefix, and restoring must reproduce the original. All346 saved model/
optimizer tensors and counters remain unchanged across backward-only work.

This qualifies a BF16 loss/projection boundary and the actual graph's split
execution. It is not an independent full-RSSM/convolution oracle, a proof of
game competence or permission to erase earlier numerical failures. Review the
coverage before any production/default promotion; no automatic learning follows.

Two serialized host-guarded GPU processes maximum,120s boundary and180s world,
persistent service15min bound,Restart=no,KillMode=control-group. Expected5080/
580.178.04/current boot,>=2GiB sampled Vulkan estimated headroom, record standalone
allocation warnings. Numerical/API/hard/deadline failures stop for review; no
blind retry, separate NVML polling or recovery. CPU preparation/build/reviews
1CPU/2GiB/no swap; GPU workers no CPU quota. Preserve sources/binaries/plans/
raw outputs and every failed gate. Production and Atari stay unchanged.
