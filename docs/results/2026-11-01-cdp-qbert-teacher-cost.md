# Qbert teacher color masks have substantial avoidable CPU cost

Direct channel comparisons preserve the teacher's decisions on the fixed59-frame
diagnostic and reduce its unprofiled472-call time **2.911s→.274s**. This10.64×
sequential snapshot timing is not a whole-training speedup:the inputs are selected
historical failure/goal images,each repeated eight times from a fresh teacher,
and the two timings occur at different times. Full candidate validation against
the retained200k-action training log is still running;no learner uses the change.

## Why inspect the Python teacher

The completed H3 learner spends561.302s of1601.182s (35.06%) in its combined
acting/guidance timer. That includes the native policy call,teacher work and
feedback,so it does not isolate teacher cost. The original H5 full label replay
takes633.898s. This motivates inspecting the adapter without changing native
learning,model size or objectives.

The [fixed CPU profile](../experiments/2026-11-01-cdp-qbert-teacher-profile.md)
uses all27/32 saved native RGB8 snapshots from the October27 hybrid and hybrid-
repeat lane0 diagnostics. Every image matches its original recorded input/output
pixel hash;source/diagnostic receipts and trace hashes match. No emulator is
constructed and no previous gameplay stage is repeated.

For each image,a fresh current `FeedbackPixelPolicy` proposes eight actions on
that same image and receives each proposed action as feedback. This synthetic
sequence exercises waiting and goal-color stabilization;it is not a real history
or decision-quality test. One unprofiled pass takes2.911s;one cProfile pass takes
3.131s. All472 proposals and final teacher states match,with unchanged images.
NumPy `all` accounts for1.684s cumulative,53.78% of3.131s profiler self time.
Connected components account for.160s cumulative and unique-color extraction
.235s;these overlapping cumulative entries must not be summed as disjoint costs.

Profile11.666s completes14:46:50 UTC November1/source377f98a,including provenance
and input checks. All17 profile inputs and506 then-active Qbert inputs remain
unchanged with originalPID803859/invocation798ad208226e40a3a73fa9e43fec8066.
This co-hosts only the CPU audit,after H3 training and frozen GPU stages end.
No training-time correction,native loading,GPU work or learning occurs.

## Narrow implementation and qualification

The [selected change](../experiments/2026-11-01-cdp-qbert-color-masks.md) adds one
`rgb_mask` helper and changes eight predicate calls from RGB equality followed
by `all(axis=-1)` to a conjunction of the three channel comparisons. Every color,
patch,component order,rule,feedback/reset and schedule stays fixed. Exact source
comparison verifies that no other teacher logic changes. The shared guidance
helper is byte-identical. The original packaged runtime is never edited.

All130 targeted CPU tests pass in.84s (.923s wrapper),including21 new cases that
exercise every0–255 single-channel value for seven colors and contiguous,
reversed and transposed views. Existing guidance and corrective-label tests pass;
the passing run attemptszero native/emulator constructions.

The first harness run passes129 tests and failsone because lazy native export
loading replaces a test-local stub with the no-GPU sentinel. The sentinel blocks
one call;no native object or GPU context is constructed. The failed source/log
are retained in`cpu-attempt1`. Eagerly resolving exports before installing guards
fixes only the harness;production and test bytes remain unchanged across runs.

Candidate qualification2.647s completes14:56:11 UTC/source3c11831. Its one timed
472-call pass takes.274s and matches every retained proposal and final teacher
state. All25 declared inputs and506 completed H3 inputs remain unchanged. H3's
original services are already terminal success,so this timing does not overlap
its training or audit. No original profiler or native/GPU stage repeats.

Evidence roots`/x/Temp/kindle-qbert-teacher-profile-20261101` and
`/x/Temp/kindle-qbert-color-masks-20261101` retain complete profiler functions,
input pixels' provenance,decisions,source pins,tests and qualification receipts.

- Profile reviewSHA256`d37fe2edae9417f1b6e392226b1733488bd99915ef73cb63157f4523114e87a5`.
- CPU tests`168fd95934b52fae808f4eb67085508745f42d73ff33778f41085269b886dc8b`.
- Qualification`524486ad64cd5d79c683a1872c194490da12172125f62063bc76f8e7ba0930da`.
- Original teacher`6a2c1dc3bc8fe1986da6984854410318d6d27cc1f41333c9b9eddf04ed536a6c`.
- Candidate teacher`fa261c53991c329236bcc167745f309fe6e568547fe2d8c6e29da3d5bc7c3e5d`.
