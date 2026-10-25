# Remove the one-dimensional packing launch cliff

Declared October15,2026 at11:03UTC before implementation/build/GPU work.
The CPU reproduction of all23 actual convolution selectors finds exactly four
falling back solely because packing needs65536 workgroups, one beyond65535:
decoder conv0 forward/dW and encoder cnn1 dX/dW. All other existing eligibility
bounds pass. Keep the32Mi-element allocation bound, channel/shared-memory/grid/
schedule checks and existing convolution algorithms, tiles, splits and precision.

Use a bounded two-dimensional packing dispatch and flatten its coordinates in
the existing typed pack shader. Retain an isolated diagnostic control selecting
the original eligibility. The current14 packed products remain; candidate18
includes all four newly reachable cases, not a hand-picked subset. The five
three-channel fallbacks remain. No new convolution kernel, geometry search,
Naga/Blade change, CDP recipe change or production promotion.

CPU first: exact coverage/no duplicates at the65535/65536 boundary, shader
validation for existing storage types, original fallback tests and actual-plan
selection assertions. Use fresh isolated source copies/distinct Cargo caches;
formatting/Clippy and source receipts precede GPU. Upstream99e3de27/e7c3f7b
rechecked unchanged October15 at10:47UTC.

One numerical control/candidate pair covers all four actual B512 shapes.
Independent direct F64 one-image convolution/derivatives extend analytically
to full B512 inputs: one active last image and repeated signed-dyadic images
with fixed batch scale factors. Check every output, including inactive zeros,
changed input and restoration. Forward reference is independently BF16-rounded;
retain existing pointwise1e-4+.008*abs(reference), relativeL2.004; derivatives
pointwise1e-4+1e-5*abs(reference), relativeL2.00002. Actual packing tail/padding
and selected heavy pipelines must match the declaration. No tolerance changes.
These structured inputs are scoped numerical evidence, not a universal oracle.

Only after CPU review of qualification: one ordinary full RGB and centered-CDP
B32/T16 control/candidate comparison on current fixed checkpoints, each16 warm+
256 measured updates. Keep all samples, exact initial292/346 tensors and targets,
finite reports/states; report first/final trajectory differences, not bitwise
parity if changing an existing kernel's reduction order changes rounding.
Current-control trajectories/programs must match retained076f3135. Require>=3%
complete RGB improvement with no>2% CDP regression for adoption consideration.
Otherwise retain/reject without a subset or geometry sweep. No extra family
timing needed; optimize complete updates. No learning restart or parity claim.

Guarded persistent systemd, Restart=no,KillMode=control-group. Numerical workers
<=300s; complete-update workers<=240s; service<=1800s. Expected5080/580.178.04/
boot3e89d55c-a9e5-472f-a18a-06508c5bafa7,>=2GiB sampled Vulkan estimated headroom.
Retain allocation warnings; API/numerical/hard/deadline failures stop for review,
not blind retry. No recovery or separate NVML polling. CPU prep/build/review
1CPU/2GiB/no swap, no compile/analysis during matched timings. Retain failures,
all extra compute and original evidence. Production remains unchanged.
