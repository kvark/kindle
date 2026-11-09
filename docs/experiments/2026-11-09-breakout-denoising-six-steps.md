# Frozen six-step sampler test

The retained [sigma_data.05 model](../results/2026-11-09-breakout-denoising-sigma-data0p05.md)
learns motion but omits47–58 of198 one-step balls per draw. An exploratory CPU
inspection of those missing predictions finds faint red near many true positions;
do not change detector thresholds. Its fixed three-step Euler grid jumps from
sigma.283 to.002, across the new data scale.05. Select a frozen six-step sampler
intervention to test a sampling contribution before choosing more training.
This is a new intervention, not another learner or an unchanged assessment.
Root `/mnt/data/kindle-breakout-denoising-six-steps-20261109`.

Add only an ignored native test collector and one CPU schedule check. Production
code, model, loss, training, default sampler, checkpoint format and Python API
remain byte-identical. Rebuild a standalone release test executable, not a wheel
or backend. Use six positive sigma nodes with the existing rho7 interpolation
between exactly5 and.002, followed by0; six Euler denoiser evaluations. Keep
RGB quantization, residual anchor, coefficients/history scale and equations.
No stage selection, threshold relaxation, label-based correction or new weights.

Before the intervention, require the test collector's three-step images to match
all108 cached actual-history batches and all64 ordinary rollout batches exactly.
Also require four complete production predict calls to match20 saved per-draw
image/event arrays exactly. The six-step mode must not start before these checks
finish. Generation receives only observed prefix pixels/actions and submitted
future commands; expected predictions and target pixels are not its arguments.
Use the same Gaussian stream/seed in three- and six-step comparisons; extra
denoiser stages consume no new random draws. Seeds83001/83003/83009/83023.

Then compare six versus three on all arrivals5..31 actual-history one-step and
all16 ordinary rollout steps from source16, eight lanes. Keep the original110/
198 supported endpoints per draw,18 unsupported cases, one-pixel detector/gates,
own-initial/copy controls and every ball/paddle/brick regression. No new event
competence claim: event outputs of the intervention are not collected. Retain
all final RGB arrays, per-endpoint outcomes and fixedlane0/draw0 visualization.
Observed-denoising is not a future task. No new emulator actions/frames or data.

Pin source snapshots/executable, qualified native source/archive, checkpoint,
config, cached references/data, protocol and runner before GPU allocation.
Run focused CPU tests and strict release Clippy; no derivative/backend fixture
needed for test-only inference. Native baseline bit parity is mandatory; stop
on any discrepancy and preserve failures. Existing AMD host guard, one300s job,
Auto, no reset/reload/NVML. Require terminal successful service, nine sealed
guard files and135 exact frozen tensors. Release only this disposable build
cache after packaging. No extension, learner, collection or automatic successor.

The test does not alter the full objective or claim that image contrast is the
only cause. Even passing fitting gates leaves full physics/events/natural
calibration, changed-brick memory, late/tunnel/second-wall/above-top, untouched
families and fresh confirmation required. Actor/critic, CDP comparison and
five-game work remain parked. All earlier trials/failures/repairs stay closed.

CPU qualification initially expects a sigma node within a factor of two of.05;
standard six-point rho7 does not satisfy that mistaken logical expectation.
Preserve the original source/executable/failing log. A separate CPU repair
checks exact independent F64 node values and narrower bracketing of.05. The
selected grid, generation code, original physical thresholds and native baseline
bit-parity requirements are unchanged. The original shell redirection also
fails before the build directory exists; preserve it, then create the directory
before the first actual compile. No GPU work precedes either correction.

Strict release Clippy over test targets also flags two pre-existing config
fixture initializations. Preserve its failing log/source, then use equivalent
struct initializers without changing values/assertions. The final collector
rebuild passes19 CPU checks/four ignored GPU tests and strict release test-target
Clippy. All production source bytes remain unchanged.
