# Current pixel positions do not rescue the fixed Breakout action readout

The fixed pixel-augmented kernel readout fails its screen. Overall teacher
agreement rises only56.91%→57.40% versus the saved learned-feature kernel,
while near-descending agreement falls58.82%→55.15%. The retained actor remains
better at62.56% overall and66.91% near descending. No native spatial input,
replacement head, loss, teacher or learner is selected on this evidence.

| Readout | All601 test rows |136 near-descending rows | Lane6,329 rows | Lane7,272 rows |
| --- | ---: | ---: | ---: | ---: |
| Saved learned-feature kernel |342/56.91%|80/58.82%|174/52.89%|168/61.76%|
| Current pixels only |290/48.25%|59/43.38%|155/47.11%|135/49.63%|
| Learned plus pixel kernels |345/57.40%|75/55.15%|184/55.93%|161/59.19%|
| Saved actor |376/62.56%|91/66.91%|207/62.92%|169/62.13%|

Each cell gives correct decisions and percentage agreement. Augmentation beats
the learned kernel on one of two test lanes and trails the actor on both. It
does not meet the predeclared five-percentage-point improvement over both
baselines overall and near descending, or the requirement to beat both in each
test lane. The pixel-only result is retained separately rather than selected
after seeing the outcome. No input-set, coefficient or kernel retry follows.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-pixel-action-readout.md)
reuses2,064 aligned rows from the old6e-6 control's trajectories and the4e-5
learner's scalar re-observed features. Training/validation/test remain954/509/601
on lanes0–3/4–5/6–7. The new input contains only current RGB64 pixel predictions:
ball visibility,x,y,paddle visibility andpaddle x. There is no added motion,
history, teacher landing target or future observation. Teacher reference target
coordinates are never inputs; missing positions stay zero with visibility off.
Ball detections number845/472/559 by split; all2,064 paddle detections remain.
The original pixel detector's misses and extra detections are not filtered away.

The augmented readout averages equally weighted, separately normalized RBF
kernels over3,584 learned features andfive pixel columns. Training rows alone
set normalization and median-distance bandwidths. Validation MSE chooses among
the original seven ridge penalties per action target, without refitting on
validation or consulting test labels. Selected penalties for NOOP/FIRE/RIGHT/
LEFT are1/.01/.1/.1 for augmentation and1/1/.01/.1 for pixels alone. Exact scores,
normalization, selections and every phase/lane/class confusion count are retained.
Original learned readouts and actor predictions are reused, with their metrics
reproduced from saved predictions; no prior fit or native forward pass repeats.

This rejects one small action-readout construction. It does not show that
precise spatial information cannot help a jointly trained native model, prove
information absent from the recurrent features, or identify why the actor loses
lives. Kernel geometry changes alongside supplied information; color/crop
priors share information with the teacher, whose decisions are not optimal
action labels. The reused control states do not measure the stronger policy's
own trajectories or independent seed reliability. There is no new gameplay
score: retained encoder4e-5 greedy104.667 remains zero of24 two-wall successes.

Six CPU fixture groups and3.472s preflight pass: exact joining/rejections,
missing detections, independence from reference targets, training-only fitting,
one-block equivalence to the qualified kernel fitter, and direct matrix ridge
arithmetic. Analysis takes.760s; controller3.410s completes **20:30:08.770 UTC
November1**, source6a6ce27. All51 diagnostic and562 active Qbert inputs remain
unchanged. Original QbertPID842010/invocationd37840f258104f9caf8b6bc5c0316899
is preserved before and after. Keep CPU co-host timing without correction.
No failure, replay, native/emulator import, GPU work, learner update or repeated
fit occurs. Qbert confirmation continues under its original observers.

Root `/x/Temp/kindle-breakout-pixel-action-readout-20261101` is complete; do not
rerun qualification, fitting or collection. The five-game goal remains unfinished.

| Evidence | SHA256 |
| --- | --- |
| Declaration | `54f153c898ddf9b82f5368c953acb215517af1a0fda6e8c503c0a2f2bfb1af1c` |
| Review | `1fad9e77fdc9eda294dd575c0cb22eee5fbdb29b3d6a4932b36abca2aef31e3d` |
| Predictions | `4885cecdcdc4dffddc9e4ecc53c39dcc6ca470f5dce352023f7bb6bae4bfbe3d` |
| Completion | `363fdebb9e78f8811314ef989a21e43063f8eb3f18e5d8c8f63b0dcee106a834` |
