# Saved centroid head position accuracy

Measure the CPU F64 reconstruction of the original trained centroid100 head on
its already captured posterior states. Use native2974160 source and the exact
old checkpoint/capture under`kindle-centroid-representation-20261029`. This is
the older encoder6e-6 model,whose full gameplay comparison remains protocol-
failed;it is not the newer encoder4e-5 learner or the active B8/T32 learner.

Use all2,064 saved rows with unchanged954/509/601 train/validation/test splits.
Report test ball x/y error on the same555 rows shared by the pixel reference and
teacher;also report557 teacher-visible rows,near-descending and held-out lanes.
Reuse existing linear recurrent and combined-kernel predictions without fitting.
The fixed head has2560→256 Linear/RMSNorm(epsilon1e-4)/SiLU→2 linear outputs.
Convert normalized predictions to source pixels as(prediction+.5)*[160,210]-.5.
No clipping,calibration,refit,subset search or target input into the model.

Verify checkpoint/capture/prediction hashes,exact indices/labels/splits,source
architecture and five parameter shapes. Test the vector expression against an
independent scalar implementation,coordinate conversion and invalid inputs.
Record F32-versus-F64 CPU drift separately. This head has no separate native
forward-agreement check;CPU drift is not a GPU-equivalence proof.

The native target is a red-excess centroid in a normalized crop. Its mask uses
resized pixel centers directly,while the old pixel decoder uses source centers
with a-.5 offset. Enumerate the resulting64-grid row/column bounds. Do not call
the saved pixel decoder an exact reproduction of the native target,or measure
unavailable native targets without replay. Teacher coordinates are estimates,
not independent physical truth. The scalar capture also differs from original
vector posterior draws;all state and gameplay conclusions retain that limit.

One low-priority CPU core,one BLAS thread,NumPy/safetensors only. Forbid native
and emulator imports;no new game replay,GPU work,learning or checkpoint writes.
Preserve active BreakoutPID812861/invocation258e9d9deb0b463a91d4206a198085e4 and
all292 input hashes before/after. Record CPU co-host time without correction.
No new loss,head replacement,learner or automatic successor is selected.
Root`/x/Temp/kindle-centroid-head-readout-20261101`.

