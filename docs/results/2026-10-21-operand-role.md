# Operand-role isolation identifies a B-preparation asymmetry

October21,2026. [Evidence](2026-10-21-operand-role.json) ·
[Preceding load/MMA diagnostic](2026-10-21-load-mma-isolation.md).

**In the calibrated diagnostic, B preparation accounts for twice the derived
shared-read work of A.** Independently reusing A or B breaks their previously
equal counts; reusing both passes a held-out prediction. This is not a speedup
or a general permission to reuse different convolution operands. A targeted
B-only staging candidate now passes CPU checks; its GPU numerics/cost are unrun.
Keep79bc7cec/RGB42.808ms versus inclusive JAX26.066ms,1.642×. Atari stays stopped.

## One shader, four predetermined settings

The same submitted module retains the BF16/F32 matrix contract, staging/global
reads, grid[1,1,256], inline uniforms, default subgroup settings and two MMA
sites. Runtime flags optionally reuse the first A/B matrix for the second
product. Each setting therefore has its own independently checked output.
The baseline reproduces32,768 derived shared reads and4,096 tensor instructions.

| Setting | Logical A loads | Logical B loads | Shared reads minus conflicts | LSU instructions |
| --- | ---: | ---: | ---: | ---: |
| Normal |2,048|2,048|32,768|64,512|
| Reuse A |1,024|2,048|28,672|64,512|
| Reuse B |2,048|1,024|24,576|56,320|
| Reuse both, held out |1,024|1,024|20,480|56,320|

Tensor instructions remain4,096. Fitting the first three settings gives
`4*A + 8*B + O`, where `O=8,192` logical output row groups. It predicts the
fourth exactly in both repeated capture ranges. A reuse changes shared reads
without changing issued LSU counts; these counters are not interchangeable.
Warp instructions are276,480/276,480/264,192/264,192 respectively.

The local result concerns operand loading/preparation, possibly including
first-use conversion. It is not per-source-instruction latency, SASS identity,
or evidence that B alone caused the historical driver difference. Historical
A/B coefficients were not separately measured. The broader cuDNN feeding/reuse
gap remains distinct.

## Actual layout audit and one candidate

All23 actual reconstructed SPIR-V modules reconcile. Nine typed occurrences use
row-major A/B at vector stride10; ten packed occurrences use row-major A and
column-major B at stride10; four packed derivatives use column-major A and
row-major B at stride18. All use BF16 matrix components. Thus a blanket
“all packed B is column-major” or “row-major alone explains the gap” is false.

The proposed candidate changes **only B's shared layout** in all nine typed
occurrences. It retains A layout, global gathers/iterators, tile sizes, splits,
K order, precision, barriers and shared extents. Existing vector B values are
scattered to column-major shared locations, then loaded with the corresponding
cooperative layout. No Naga feature is required.

All18 old/new modules validate offline; all nine old modules are byte-exact with
the retained actual controls. Reversing the declared B edits reconstructs every
original source.9,216 unique scalar writes/18,432 matrix-value addresses and
nine wrong-orientation negatives pass. The emitted loop preserves global memory
sites and contains direct16-bit shared stores, no shared read-modify-write.

**The cost tradeoff remains unmeasured:** two B-vector stores become eight
scalar stores per iteration. This differs from the rejected both-operand
column-major packet-axis rewrite, which also changed global gathering and
iterator work. That rejected family stays closed; this is one evidence-led
B-only candidate, not a layout/geometry search or a performance claim.

## Checks, failures and next gate

Eight GPU guards pass with zero allocation warnings. Exact integer/F64 outputs,
negative-half input/restoration, immutable inputs,256-byte tails and actual
subgroup membership pass:4,194,304 reference comparisons/2,097,152 paired values/
262,144 membership records include capture rechecks. All696 metrics reproduce
offline over200 passes.448 primitive dispatches, zero learner updates/actions/
ordinary timing samples. The qualified observer binary is reused unchanged.

The first CPU fixture check fails because inherited B has identical upper/lower
halves. New non-degenerate integer matrices pass all six pairwise output
distinctions and independent oracles, without changing exact gates. The earlier
load/MMA experiment remains valid for its original question. Separately, the
first layout reader's all-packed assumption fails on a derivative; the corrected
complete inventory preserves that failure. Neither repair repeats GPU work.

GPU ends11:52:27UTC; B-only CPU proof ends11:58:54; final216-receipt review ends
12:00:20. All workers are terminal. One CPU emitter helper builds, no backend/
compiler edit; upstream main remainsa75c38e.

Next separately qualify all nine B-only GPU primitives and model-calibrated
counters. Only a useful instruction result justifies one complete-operation
cost comparison, including unchanged packed products/transforms/dW reductions,
then any whole-update integration. No favorable subset, timing sweep, promotion
or Atari restart. Full backend parity and five-game mastery remain open.

Raw evidence: `runs/operand-role-distinct-20261021.cfEouL1R`; original failed
fixture preparation: `runs/operand-role-20261021.mJRdRz1v`.
