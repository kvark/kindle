# Explicit pixel motion does not rescue the fixed Breakout action readout

Adding immediate-predecessor pixel displacement to the fixed augmented readout
scores **55.91% overall /55.88% near descending** teacher agreement, below the
retained actor's **62.56% /66.91%**. It fails the predeclared screen. This supplies
no positive evidence for the proposed temporal-input prototype; retain the actor
and existing training recipe. It does not rule out jointly learned temporal
inputs or identify the cause of Breakout's remaining control gap.

| Readout | All601 test rows |136 near-descending rows | Lane6,329 rows | Lane7,272 rows |
| --- | ---: | ---: | ---: | ---: |
| Saved actor |376/62.56%|91/66.91%|207/62.92%|169/62.13%|
| Saved learned-feature kernel |342/56.91%|80/58.82%|174/52.89%|168/61.76%|
| Saved learned plus current pixels |345/57.40%|75/55.15%|184/55.93%|161/59.19%|
| Learned plus current pixels and motion |336/55.91%|76/55.88%|181/55.02%|155/56.99%|
| Saved current pixels alone |290/48.25%|59/43.38%|155/47.11%|135/49.63%|
| Current pixels and motion alone |325/54.08%|72/52.94%|176/53.50%|149/54.78%|

Each cell gives correct decisions and percentage agreement. Motion helps the
pixel-only construction, but the primary augmented construction loses nine
overall decisions versus its current-pixel baseline and gains only one near
descending. It trails the actor in both test lanes. Neither the five-percentage-
point overall/near margin nor the per-lane requirement passes. All outcomes and
class/phase/lane confusion counts remain retained; no input or kernel retry follows.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-pixel-motion-readout.md)
uses the same2,064 rows and954/509/601 lane-separated train/validation/test split
as the completed current-position screen. The ten new-block columns contain
current ball/paddle presence and position plus presence-masked current-minus-
previous displacement. The predecessor is exactly one recorded decision earlier
within the same natural episode. Missing objects and reset boundaries produce
zero displacement with presence off; no backward search fills gaps. Teacher
coordinates, phase, commands, landing targets and future observations are not
feature inputs. Test motion support is547 ball/595 paddle rows; missing cases
stay in the601-row denominator.

The new CPU capture collects predecessor observations absent from the original
stride8 pixel archive. All20,960 actions/83,799 frames, input pixel hashes,
rewards, natural boundaries and reset counters reproduce. All2,064 current
pixel predictions match the saved archive exactly. It reuses the qualified
RGB64 detector and byte fixture; no original pixel analysis, frozen feature
capture, model execution or baseline fit repeats.

The primary model averages equally weighted, independently normalized RBF
kernels over3,584 learned features and the ten pixel/motion columns. Training
rows alone determine normalization, bandwidth and centering. The unchanged
seven-penalty validation rule selects1/.1/.1/1 for augmented NOOP/FIRE/RIGHT/LEFT
and1/1/.1/1 for pixels/motion alone, without validation refitting. Saved actor,
learned-kernel and both current-pixel baseline metrics reproduce exactly from
their saved predictions; no baseline is retrained.

Six causal-join/missingness/bounds/arithmetic fixtures pass. All75 pinned inputs
remain unchanged. Capture takes12.177s, fitting.950s and the complete controller
13.330s, completing **03:56:34.670 UTC November3**, selection sourcec91dcfa.
No failure, GPU work, learner update or native learning import occurs. Fitting
also blocks emulator imports. No Kindle service is running before or after.

These are reused weak-control trajectories and an older encoder4e-5 checkpoint,
not the latest replay-value1 learner's own states. Color/crop priors share
information with the teacher, whose labels are not optimal-action truth.
The readout geometry changes alongside supplied information. Displacement from
pooled observations is not physical velocity. The screen supplies no gameplay
score, causal representation diagnosis or independent seed evidence.

Root `/x/Temp/kindle-breakout-pixel-motion-20261103` is complete. Do not repeat
the completed capture, fits, qualification or collection. No native architecture,
teacher, loss, learner or automatic successor is selected. The retained Breakout
70.750 sampled/143.667 greedy scores and0/24 two-wall outcomes are unchanged;
the five-game goal remains unfinished.

| Evidence | SHA256 |
| --- | --- |
| Declaration | `c812fcae22e1b5b8fa694a312892370b89c5159ed64c7fd2d50fcc75169a98ae` |
| Capture | `cbb5a5b73e9d7a221810027bcb0bc2490a15dfec3bf5d2c6a536e39d2deaf6f8` |
| Readout result | `38518e9ce83e8ace868ba758b5892e34c0a683cff71d276c46768a45e0e6e1e2` |
| Completion | `09cacc8d8622a58706dc48286b67853db71b33213688430bedea95a2d93a3424` |
