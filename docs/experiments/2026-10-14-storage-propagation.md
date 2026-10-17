# Qualify upstream Vulkan storage propagation before launch geometry

October14,2026. Upstream recheck at03:03 UTC finds Blade96c1dc7, newly pinning
Naga a97559b. Meganeura main remains2214412. This supersedes the unstarted
[dense-grid check](2026-10-14-dense-launch-grid.md) as the immediate next action.
Its empty artifact `runs/dense-launch-grid-20261014.Upb23FRK` contains no work;
no kernel/grid change, learner or failed run has been restarted.

Use `runs/storage-propagation-20261014.LAt4T0Zs`, reproducing the immutable
row-selection6d892a3d candidate's Meganeura/Kindle sources and graphs in isolated
worktrees. Only change Bladefe5f5b5→96c1dc7 and Nagaa4b4183→a97559b in both
dependency trees. Production pins, neighbors, arithmetic, RNG, split counts
and scheduling barriers remain untouched.

The Naga change removes per-access MakePointerVisible/MakePointerAvailable
from ordinary storage-buffer accesses under the Vulkan memory model and moves
propagation to storage barriers. NonPrivatePointer remains; workgroup/image
and explicitly decorated global accesses retain their special semantics.
It is not permission to remove Blade's real inter-dispatch barriers, a blanket
Vulkan-memory-model change or proof of an already-measured hardware speedup.
Review the exact diff and retained shader snapshots before claiming causality.

Build/Clippy and existing CPU shader/plan checks precede GPU work. Run existing
independent F64 typed dense/reduced derivatives, packed convolution derivatives
and grouped products, row selection, plus staged forward/backward/optimizer
lifetime controls. Keep all original bounds and per-output/partial checks.
Expected5080,>=2GiB sampled Vulkan headroom, guarded serial persistent services,
record allocation warnings; numerical/API/hard faults stop for review. No
NVML polling, recovery or CPU quota on GPU work. CPU preparation/analysis:
one CPU,2GiB/no swap, never during timing.

After numerical review, two24-update RGB/CDP smokes; require unchanged plans,
initial states and all first outputs/reports/saved tensors. Any difference
stops for investigation: this upstream change is synchronization lowering,
not an intended arithmetic intervention. Then one ordinary16-warmup+256-
measured control/new-compiler pair per workload, same exact-state checks,
all reports and complete stage costs, no GPU timestamps/concurrent preparation.
Use the retained JAX boundary separately; no direct CDP/RGB comparison.

If there is a material gain, attribute it to this isolated compiler change
and inspect which actual shader families select the Vulkan memory model.
Do not call MakePointer operands one-for-one hardware fence instructions without
machine-code/counter evidence. Preserve all snapshots/failures, audit the PR
dashboard and update the root-cause plan before returning to launch geometry.
No Atari training or automatic production promotion.

## First qualification failure: compare the immutable compiler control

Six groups pass: typed dense/reduced derivatives, packed convolution, grouped
products, staged lifetimes and first-maximum selection. Row-fusion precision
stops at128×2048, F32 input/BF16 output/default fusion: weight-gradient949 is
1.689396594883874e-4 versus F64 reference1.690247938686015e-4; error8.51344e-8
exceeds the unchanged3.63294e-8 allowance. Keep the failed guard and all logs;
no learner smoke or timing has started. One separately bounded300-second check
now runs this exact test on immutable old-compiler backend82795094, without
source/threshold changes, to distinguish a new compiler regression from an
inherited candidate/test issue. Review either outcome before follow-up.

The immutable old compiler reproduces the exact failing value and allowance.
No new-compiler regression is established. Before learner work, one bounded
diagnostic on the new compiler exports the original raw gradients and checks
whether exposing the existing quantized output preserves them. Evaluate the
same F64 backward graph conditioned on the observed quantization decision,
keeping the original error scales and tolerance; independently differentiate
the RMSNorm/SiLU/MSE weight path and account for any output-bin crossings.
This is a diagnostic, not a replacement passing qualification. No runtime,
learner, split, threshold or optimizer change; retain both original failures.

The diagnostic preserves every raw gradient when exposing the existing output.
Three/262,144 BF16 outputs choose adjacent bins near F64 prequantization values.
The independently differentiated weight shift at949 is-8.513310751958957e-8,
explaining the original8.513438021410624e-8 error to1.273e-12. Conditioning on
the observed bins passes every bias/weight gradient with the original error
scales and tolerance; independent analytic/F64 weight gradients agree to
2.17e-19 maximum absolute error. Zero optimizer updates/actions. The diagnostic
is not an unconditional F64 pass, and neither original failed guard is relabeled.
The first diagnostic build's Graph::clone error is retained; deep_clone fixes
only test construction, not runtime or numerics.

Review permits the already-declared unchanged-graph smokes and timing only:
this is a known, reproduced F32/BF16-versus-F64 reference boundary, not an
identified new compiler defect. Require exact old/new whole-learner reports,
outputs, plans and saved states as declared. Six original passing numerical
groups and the diagnostic remain scoped evidence, not a claim that the whole
original qualification passed. Do not broaden this exception to other failures
or relax any tolerance, and do not promote production/training from it.

Both smokes pass all24 numerical reports/first outputs/initial-first-final
292/346 tensors. Plans match after canonicalizing only ExecutionPlan's three
HashMap fields (param_types/input_types/weight_buffers); every other byte,
including constants, dispatch order, barriers and geometry, matches unchanged.
The original byte-order assertion and two CPU2GiB parser OOMs remain. The bounded
field-canonical audit passes in3.24s/1.7GiB peak without another GPU run.
Proceed with the declared ordinary timing, RGB control/new compiler followed
by CDP new compiler/control, no concurrent preparation.

Completed: [result](../results/2026-10-14-storage-propagation.md). RGB28.197→
28.314ms/CDP92.965→93.049ms gives no material gain. All272 reports/first
outputs/plans/initial-first-final tensors match exactly. Keep the new compiler
for isolated tests, not production.13 passing/two retained failed guards,
three warnings; both original numerical failures and CPU audit failures remain.
No worker or game actions. Return to the separately declared actual-grid check.
