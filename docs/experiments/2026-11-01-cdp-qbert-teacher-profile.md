# Bounded CPU profile of the unchanged Qbert teacher

Qbert H3 training ends at200k actions in1601.182s;its combined acting/guidance
stage takes561.302s (35.06%). The retained H5 training-label replay takes633.898s.
Neither timer separates teacher cost from the surrounding work. Before considering
a Python adapter change,measure the unchanged teacher on existing lossless frames.
This is CPU analysis within the compute-constrained learning goal;it selects no
native backend work,new learner or teacher change.

Use all59 RGB8/native210x160 snapshots from the retained October27 hybrid and
hybrid-repeat lane0 diagnostics (27/32). Verify each saved image against its
original recorded input/output pixel hash,plus retained source/diagnostic receipt
and trace hashes. These are selected failure/goal snapshots,not a representative
sample of current training. No emulator is constructed or old replay repeated.

For each image,construct a fresh current `FeedbackPixelPolicy`,then make eight
proposals with the same image,feeding each proposed action back. This deliberately
synthetic fixed-input sequence exercises ordinary waiting and goal-color
stabilization;it is not a game trajectory or decision-quality assessment. Use
one unprofiled timed pass and one cProfile pass over all59 images (472 calls each).
Require exact proposals and final teacher states between passes,unchanged image
bytes and finite nonnegative timings. Keep complete profiler output and report
both timings,including profiling overhead. Do not extrapolate a whole-training
speedup or attribute the full561s acting timer to the teacher.

Use the exact447fe50-era packaged teacher and shared guidance module without
modification. Import through an isolated Python package with no native/ALE loading.
Pin inputs before execution;check all506 active Qbert inputs and original
PID803859/invocation798ad208226e40a3a73fa9e43fec8066 before/after. Its training
and frozen GPU work are finished;the original full replay/audit remains active.
One low-priority CPU core,60s total diagnostic cap. Preserve audit co-host timing;
it does not overlap the already finished training timer. No repeat or automatic
optimization follows. Root`/x/Temp/kindle-qbert-teacher-profile-20261101`.
