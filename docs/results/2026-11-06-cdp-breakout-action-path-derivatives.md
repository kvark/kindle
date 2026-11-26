# Frozen action-path derivatives at saved rescue origins

The [action-path probe](../experiments/2026-11-06-cdp-breakout-action-path-derivatives.md)
and its separately declared [fixture repair](../experiments/2026-11-06-cdp-action-path-fixture-repair.md)
are complete. Return derivatives at the recorded plan favor the known short
rescue in 81/92 retained-control origins and 80/92 candidate origins. This does
not show generally reversed return gradients in the failed learner. Survival
directions are less convincing, and both models still predict high survival for
plans that actually end the game. Keep the extra actor dynamics loss off.

Each model uses all 120 saved origins, eight paired draw streams, two fixed
nine-action plans and two objectives. Positive means favor the teacher plan;
counts below use the declared absolute 1e-6 tie tolerance after averaging the
eight draws. Agreement means equal signs of local direction and finite plan
difference. There are no ties in these two primary slices.

| Measurement | Retained control | Dynamics.5 candidate |
|---|---:|---:|
| Survival: full teacher plan favored at the 18 differing game-over origins | 12/18 | 11/18 |
| Survival: local gradient at recorded plan favors teacher | 12/18 | 8/18 |
| Survival: local/finite signs agree | 10/18 | 7/18 |
| Return: full teacher plan favored at the 92 short rescues | 71/92 | 73/92 |
| Return: local gradient at recorded plan favors teacher | 81/92 | 80/92 |
| Return: local/finite signs agree | 68/92 | 71/92 |

At the 18 game-over origins, mean predicted discounted survival is
.970267/.967395 for recorded plans and .973851/.973785 for teacher plans
(control/candidate). Actual recorded survival is zero; teacher survival is
gamma^9=.973295. The local survival direction has negative mean
−.004495/−.021825 despite positive finite differences +.003584/+.006389.
Positive sign counts therefore do not imply favorable average slope magnitude.

Recorded-plan gradient signs vary across draws at 9/18 and 10/18 survival
origins, and 39/92 and 34/92 return origins. Mean within-origin draw standard
deviations are .010837/.032089 for survival and 4.047809/1.888621 for return.
Eight draws describe sensitivity; they do not establish Monte Carlo convergence.
At the teacher endpoint, the same teacher-minus-recorded direction is positive
only 3/18 and 13/18 for survival, and 36/92 and 43/92 for return. Endpoint
dependence can reflect ordinary nonlinear dynamics as well as surrogate bias;
this probe does not separate those causes.

The 19 identical-plan origins give exact zero finite differences and directions
for both objectives, both models and every draw. All-origin/distinct-plan
summaries, raw 7,680 objective/plan/draw gradient vectors and per-time directional
contributions remain in the artifacts. All 240 real short branches have zero
reward. The 92 rescues are a short-outcome proxy, not known higher complete
returns; the critic also reflects intrinsic-reward training.

Control execution takes **1.676s**, candidate **.949s**. Production controller
and audit take **6.029s** (9.077s with wrapper/initial pin verification), complete
**07:06:43.521 UTC November 6**, source `65f1a9acbeb3946d409e1f06b218f583c192b387`.
All 775 production pins, 43 unchanged frozen GPU input tensors per model,
checkpoint file hashes and both guards pass. Separate inference agrees on each
first batch/plan, maximum normalized error 3.390376e-7. The forward check uses
|actual−reference|/(1+|reference|), with the fixed 1e-3 tolerance. Minimum reported native
budget headroom is 21,234,376,704 bytes. This is not independent GPU telemetry.
There are **zero Atari actions, learner updates and optimizer steps**.

CPU analysis passes in .417s at 07:07:40.881; a separate closure recount verifies
the primary sign counts from saved draws, source zero rewards, terminal worker
identities and 775 pins. Closure at **07:11:32.503 UTC** seals 261 files in1.495s;
no Kindle service or render-device holder remains. Production and analysis have
no failed or repeated stage. The original qualification failure below remains.

These results limit a simple explanation that the candidate's return gradient
generally points away from rescue. They do show weak survival calibration and
local/finite disagreement in this selected cohort. Fixed plans omit actor
feedback, lambda weighting, disagreement bonuses, entropy, imitation and the
optimizer. They cannot establish the cause of the full learner's regression.
If pursuing this diagnosis, the unresolved question is how the auxiliary term
changes the actual actor update alongside existing losses; a coefficient sweep
or categorical-representation replacement is not justified by this probe.
No successor allocation or new learner is selected.

Original qualification at ceb2be51 fails before forward/backward execution:
the toy fixture attempts elementwise [1,1] × [1,3]. The repaired fixture uses
matrix multiplication and its CPU test now constructs and compiles the graph.
Production graph, objective, estimator, inputs and thresholds are unchanged.
The original failed guard, stderr, binary and fixture source remain under
`/x/Temp/kindle-breakout-action-path-20261106`; do not repeat that stage.
Earlier missing historical-receipt and closure index-type failures remain.

Before repair, 190 Kindle/5 gym CPU tests, four analysis tests and strict Clippy
pass. After the fixture change, both affected CPU checks, strict Clippy and
the release build pass. Native qualification checks all 18 F64 surrogate
derivatives, both scalar objectives, frozen inputs and zero optimizer steps.
Tiny forward values match separate existing transition/head/value inference
graphs, maximum relative error 1.180303e-6. Native test wall time is .81s;
guarded controller/audit is 3.193s, with 774 unchanged pins and a passing host
guard, completing **07:04:52.759 UTC November 6**. This repaired qualification
passes; the original qualification remains failed.

Repaired source `b7d45c32eda52bd400b29142ff5b3911469c7219`, test executable SHA256
`f564ec9027cd095a676e2a280c51faae8954641e40afea1177f73587baf7928b`.
Root `/x/Temp/kindle-breakout-action-path-20261106/shape-repair`; exact declaration,
source pins, inputs, binary and guard receipts are retained there. There is no
new Python package or production runtime change. The parent build cache is
released after validation (5,610,733,853 logical bytes); both packaged binaries
remain byte-identical, with separate cache-release receipts.

The two production jobs complete within their original 180s caps, serial, under
the existing host guard. This selected history and these two checkpoints do not
establish population calibration or gameplay gains. Keep the retained
Breakout control and actor dynamics off; original 864/90%-two-wall and five-game
gates remain unfinished.
