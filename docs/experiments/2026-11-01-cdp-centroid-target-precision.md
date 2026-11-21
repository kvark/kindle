# Measure the existing color-position target on fixed Breakout frames

The saved-head study identifies a crop difference between the actual auxiliary
target and the earlier precise pixel reference. Measure the CPU F64 reference
of the existing native2974160 target formula on the unchanged representation
corpus, using its saved checkpoint settings. This resolves target precision;
it does not select a new crop, loss, teacher, head or learner.

Replay all20,960 recorded actions/83,799 frames with every input pixel hash,
action outcome and boundary verified. On all2,064 selected rows, reconstruct
RGB64 with the previously qualified Pillow bilinear path. Reproduce all saved
pixel-reference predictions exactly. Keep the954/509/601 splits and the555
common teacher/pixel-visible test rows; report native-target support and misses
explicitly rather than dropping missing detections. Also report all test rows,
near-descending rows and both held-out lanes.

Use channel0,background1.5,region[.05,.42857143,.95,.88095236],minimum mass1e-6
and the native normalized-center mask,including rows27–55 and columns3–60.
Convert to source centers with(prediction+.5)*[160,210]-.5. Compare to the saved
teacher coordinates,old pixel reference and already computed old head outputs.
Measure head error against its own target on valid rows. Report every support
disagreement and boundary coordinates. No fitting,threshold/crop sweep or
retuning follows from this diagnostic. The old learner is encoder6e-6 and its
gameplay protocol failure stays; this is not the newer encoder4e-5 learner.

Before replay, require independent scalar target fixtures,mask boundary and
minimum-mass cases,coordinate conversion and the existing native RGB64 byte
fixture. Record CPU F32/F64 drift separately. No direct native GPU target/head
readback occurs; fixture agreement and source reconstruction do not establish
exact native outputs on these real frames. Teacher coordinates are estimates
with shared color priors,not independent physical truth.

Use one low-priority CPU core and one BLAS thread,180s analysis cap. Block native
imports and GPU handles; emulator replay is CPU-only. Verify all521 active
Qbert25M inputs and originalPID821599/invocation208ec1ae7b96453f927694fae11fa0ff
before/after. Preserve Qbert training and observers; no overlapping GPU work.
Retain CPU co-host time without correction. Preserve failures without replaying
completed stages. Root`/x/Temp/kindle-centroid-target-precision-20261101`.

## Completed execution

The first CPU replay completes17:00:05 UTC November1/source79701cc in11.769s
(11.876s wrapper). All27 inputs,521 active Qbert inputs and original identity
pass,with all20,960 actions/83,799 frames and2,064 saved pixel predictions exact.
No failure,GPU work,updates or repeated stage occurs. The [completed result](../results/2026-11-01-cdp-centroid-target-precision.md)
reports target error.058/.225 pixels versus learned-head error5.461/5.206 against
the target. No crop,loss or learner change is selected;do not repeat the analysis.
