# Fixed-width categorical regrouping development

The completed fresh 200k comparison fails the primary retention criterion:
greedy **136.958** versus retained **143.667**, sampled **77.875** versus **70.750**,
in **21m25.849s** versus **21m27.910s**. Primary own-initial gain/min falls 4.295%;
both paired intervals span zero. Keep preset 32×16 and the retained recipe.
The full result is below. Original native qualification remains failed alongside
its separate successful test repair; no stage was repeated to obtain this result.

Native qualification is complete with a separate test readback repair. The
original qualification remains failed. The repaired categorical fixture and
previously unexecuted Tiny learning/restore fixture both pass; details and
limits follow below. These fixtures alone establish no gameplay benefit.

The [development protocol](../experiments/2026-11-06-cdp-latent-shape.md) selects
an optional64×8 latent grouping at12M instead of32×16, with the same512 flattened
entries and network parameter shapes/counts. Sampling, active one-hot entries
and KL behavior change together. A representation benefit is unproven; no
Atari learner or GPU allocation is selected at this stage.

`DreamerConfig.latent_shape` defaults absent and preserves legacy serialization.
Validation rejects zero/one-class/overflow/mismatched-width configurations and
requires centered RGB CDP before construction. `--latent-shape VARIABLES CLASSES`
is fresh-only; checkpoint metadata preserves the shape. A capability check
rejects old native packages before fresh or restored construction. All existing
CNN/RSSM/actor/critic graph builders consume the selected shape; no new loss,
kernel, image transform or emulator-state input is added.

CPU qualification passes202 Kindle and5 adapter tests,36 focused Python tests,
and strict workspace/binding Clippy. Legacy and identity-override graphs match;
Tiny and12M RGB graph owners preserve parameter schemas while categorical input
geometry changes. The independent F64 KL reference covers clamped and active
states and stopped derivatives. Both synthetic streams'34 RGB observations
construct on CPU; declared world parameter names exist before GPU execution.

The first Python invocation fails during collection because `atari_vector` is
not on the import path. The explicitly configured repair passes without source
changes; original log/receipt remain. Review then catches a feature-encoder name
in the new RGB native fixture before its execution, corrects it to `cnn0.weight`,
and strengthens synchronization checks to all11 actor parameters. The affected
CPU suite and strict Clippy pass again. Original fixture copies are retained;
no native stage failed or ran during preparation.

Root `/x/Temp/kindle-latent-shape-20261106` retains exact stage declarations,
source hashes and preparation evidence. Release packaging is next, followed by
installed-package CPU checks. A separate guarded native declaration must precede
the two ignored fixtures; a separate bounded protocol must precede Atari learning.
Keep retained143.667 greedy/70.750 sampled, actor dynamics/residual/coordinates
off. Original864/90%-two-wall and five-game gates remain unfinished.

## Packaged native qualification selected

Release packaging passes on source536012d5, native SHA256 prefix2c9c6e6f and
executable73a18277, with246 source pins. The installed package passes1,139 Python
tests/10 deselected with zero native construction attempts. Both release artifacts
and the wheel are retained. The [separate native protocol](../experiments/2026-11-06-cdp-latent-shape-native.md)
now selects two guarded fixtures,120s each/controller300s: independent F64
categorical values/derivatives/sampling, then two synthetic vector updates and
exact restore. No Atari learner or bounded cost check is selected.

## Original native failure and separate readback repair

The original native qualification fails at11:00:22.953 UTC November6, source
befcbf0c, in2.169s wrapper. The first scalar-loss check passes, then the raw-KL
check reads the first output because `read_output` takes a length, not an index.
Derivative checks and the second learning fixture are not reached; zero learner
updates or Atari actions. The guard records child1462683 exit101 and no recovery
or unfinished children. Preserve the original failed qualification and binary.

The [separate readback repair](../experiments/2026-11-06-cdp-latent-shape-readback-repair.md)
changes only the new test fixture's raw-KL/sample reads to indexed reads. Graphs,
F64 oracle, thresholds, production math and original packaged native remain
unchanged. Five affected CPU tests and strict Clippy pass. A separate release
executable and new output/checkpoint paths will qualify the repaired derivative
fixture and previously unexecuted learning/restore fixture. No Atari learner.

## Completed separate repair and qualification closure

At11:04:01.865 UTC November6, the repaired controller succeeds in3.518s
(3.672s wrapper), sourceacc24213. Its executable SHA256 is
`550934ce0f77de6abcdb940ec7766a1a395021e72992381e58313a9a39b08035`.
Only `agent/latent_shape_tests.rs` differs among246 original source pins;
production native2c9c6e6f and all original package artifacts remain unchanged.

The categorical fixture passes in.75s: six declared forward/backward cases at
32×16/64×8 match independent F64 values and all parameter derivatives, including
free-nat clamping and stopped routes. Supplied-draw samples match exactly;
parameters and optimizer counters stay unchanged. The previously unexecuted
Tiny8×2 learning fixture passes in.59s: exactly two finite vector updates change
encoder/reward/actor/critic, all11 actor parameters share correctly, live groups
are one-hot, and checkpoint weights/moments/configuration/counters restore exactly
with empty replay. Both guards and255 pins pass. Zero Atari actions.

The separate CPU closure finishes11:05:07.150 in.037s, confirming original and
repaired child identities are terminal, no Kindle service/render holder remains,
and the original failed guard still records exit101. It verifies the isolated
fixture change and byte-identical production package. The subsequent cache
release preserves the wheel, original native, original test executable and
repaired executable while freeing6,411,078,903 logical bytes. No completed
native fixture or original failed queue was repeated unchanged.

`qualification-review.json` records `complete_with_test_readback_repair` and
`qualified=true`; the original qualification remains failed. Original execution
SHA256 `61f74b84aeb211095c7ff2494b09c239371db0c8d69b28333cd48893abbf2232`;
repaired execution `1118a7009213e5c47e982811c2ebbebd7e891b9ad5d021f44c6b993457dc3370`;
repaired native review `489fc055003e0591b7c35d7c292bb80c6267d3a0fb4bce4b1d00b07f10a5e5e9`.

These results establish the declared numerical and Tiny learning/restore fixtures,
not12M Atari learning, initialized-tensor equivalence or improved efficiency.
A separate fixed8,192-action cost/replay protocol is the next substantive check;
no such allocation or full learner is selected here. Keep retained143.667/70.750
and all original gates unchanged. Do not repeat the closed qualification stages.

## Selected bounded cost/replay check

The [fixed8,192-action protocol](../experiments/2026-11-06-cdp-latent-shape-bounded.md)
now selects one fresh12M64/64×8 Breakout learner and1,024 frozen greedy actions.
All other retained settings stay fixed. Require357 equal retained initial
weights/moments,489 finite updates, full training/frozen replays and357 invariant
frozen tensors. Initial action traces may differ under changed grouping.

CPU preflight passes in.167s wrapper with zero native/emulator construction:
both exact queued commands, missing-shape rejection, all357 equality and five
actor/world/shape/extra/missing negative cases. Same16,334,353 parameters and
5,939,200,000-byte full replay footprint. Guard/cost/headroom caps follow the
separate protocol; no full learner or gameplay benefit is established. Root
`/x/Temp/kindle-latent-shape-20261106/bounded`; original qualification failure
and successful test-only repair stay retained without repeating either stage.

## Completed bounded check and selected fresh comparison

Sourcec65b68fb completes the bounded controller at11:11:18.958 UTC November6
in64.712s (64.745s wrapper). Training covers8,192 actions/32,734 emulator frames/
489 updates in46.655331s, versus retained47.689684s,2.169% less observed time.
Construction takes2.771416s. Historical timing and changed experience do not
establish an isolated speed improvement. Padded200k projection1,252.950787s
passes the1,800s gate; minimum observed headroom17,988,603,904 bytes across3,096
snapshots leaves12,049,403,904 after subtracting the entire5,939,200,000-byte full
replay footprint. The conservative2GiB gate passes.

All357 actual-initial tensors match retained and all schemas stay unchanged.
The1,024-action/4,096-frame frozen restore takes.907656s, preserves357 tensors
and performs zero updates. It completes zero natural episodes and provides no
competence evidence. Encoder/reward/continuation/actor/critic parameters change;
all489 finite reports and both weighted loss identities pass. Complete training
pixel/action/label/override replay and frozen action/outcome/frame/boundary replay
pass, as do both guards and51 allocation/246 source pins. Overrides cover47.046%
of actions;20% selects episodes. Learner time40.274s includes component sums
16.566s imagination,13.558s world training,6.145s posterior and2.501s behavior.

Independent CPU closure confirms original controller1464564 and both children
are terminal with51 unchanged allocation pins. No bounded stage fails or repeats.
Review SHA256 `8952e2ea167fdfc94f5537806d74deb7590edc16b787a7923f9b1f9226eaac67`;
training replay `ed4265a8690cc64e8d276e54530b828df5834bd5f6d8692adb840d650bd79598`.
Root `/x/Temp/kindle-latent-shape-20261106/bounded` is closed.

The [separate fresh200k protocol](../experiments/2026-11-06-cdp-breakout-latent-shape-200k.md)
selects one candidate with the same package/recipe, initialized afresh. CPU
helpers pass in3.554s with no native/emulator construction: full comparison
entry, own-initial lane estimates, corrupted losses/headers, exposure and plotting.
All357 bounded/retained initial tensors must match;47 pre-update events must
match the bounded candidate. Retained action traces may differ under grouping.
Four frozen24-natural-episode arms measure actual-initial and final policies.
No bounded experience is transferred; no full result or gameplay benefit existed
at declaration. Exact five-command preflight was required before launch.
Original864/90% and five-game gates remain unchanged.

## Full comparison launched

At11:16:40.300 UTC November6, sourceeda94b80 launches the fixed comparison under
`kindle-breakout-latent-shape-200k-20261106`. Main PID1464984/invocation
`eb874d2d6b8f4abd9ca4e97148e4f0de`, collector1464987/
`031409b24d4f4f4791cc97f2e44947ad`, recovery observer1464990/
`9e6a4019ce22401c9668689b3817c261` are independently recorded in launch receipts.
Exact five-command preflight passes in.437s with356 pins, three rejected training
changes, both replay argv paths and zero native/emulator construction.

Startup passes all357 bounded initial tensors and all357 retained initial tensors,
plus47 bounded pre-update semantic events. The descriptive retained comparison
also observes47 matching events; equality was not required for changed grouping.
At launch no frozen initial or final score had been measured. The original service
was active; no source/runtime edit, repeat, extension or successor was selected.
Follow its original identity and observers through completion rather than
restarting after an observation timeout. The fixed protocol remains unchanged.

The original recovery observer completes successfully at11:27:40.863 UTC in.357s:
100k actions/6,227 updates,357 finite tensors and356 unchanged pins, with original
main identity preserved throughout. Files/directories are read-only; verification
performs zero forwards or updates. Main training and terminal collection continued.
A manual status poll initially assumes the completed observer still exposes its
active invocation ID; systemd clears it on completion. Direct service/journal
inspection confirms success and the unchanged live learner. No job is restarted.

## Completed full 200k comparison

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-latent-shape-200k.md)
completes on source eda94b80 with native 536012d5/SHA256 prefix 2c9c6e6f.
One fresh seed 1009 learner finishes **200,000 aggregate actions /799,594 emulator
frames /12,477 updates** at **11:38:10.380 UTC November 6**, in **1,285.848676s**.
Construction takes 2.477081s. The observed training-time difference is only
−.160%; historical timing and changed experience do not establish a speedup.

| Measure | Retained 32×16 | Candidate 64×8 |
|---|---:|---:|
| Training wall time |21m27.910s|21m25.849s|
| Greedy return |143.666667|136.958333|
| Sampled return |70.750000|77.875000|
| Actual-initial greedy return |2.208333|1.791667|
| Actual-initial sampled return |1.416667|1.458333|
| Greedy own-initial gain/min |6.590132|6.307119|
| Sampled own-initial gain/min |3.230038|3.565738|
| Greedy/sample two-wall successes |0/24 each|0/24 each|

Greedy is the declared primary. Its raw difference is −6.708333 with paired
95% interval [−63.000000,56.796875]; own-initial gain/min falls **4.295%**, with
difference −.283013 and interval [−2.896655,2.688719]. Four of eight lanes improve.
Sampled raw difference is +7.125000 with interval [−8.833333,26.458333]; gain/min
rises **10.393%**, difference +.335700 with interval [−.406812,1.235388]. Five lanes
improve. Intervals resample the eight reused lanes 10,000 times at seed 0; they
are not training-seed reliability intervals. The candidate fails retention,
while neither mode establishes a reliable difference. Keep preset 32×16.

All 357 actual-initial tensors equal both bounded and retained, and all 47
bounded pre-update events match. Changed grouping nevertheless changes initial
behavior: sampled traces have 245 matching/538 differing paired events plus
52 candidate-only/16 retained-only events; greedy has 0 matching/743 differing
paired events plus 94/51 unpaired events. This is allowed by the fixed protocol,
and both comparisons subtract their own measured initial scores.

All four frozen arms select the first three natural episodes per eight lanes,
24 each, with zero updates or selected truncations and 357 unchanged tensors.
The four executions total **82,448 actions /329,636 frames /52.883s**; extra
completed episodes and partial tails outside the selected cohort are excluded.
Every action/outcome/frame/boundary replay, guard and video check passes. Frozen
logs lack live image hashes, so replay cannot prove unrecorded image equality.
The complete training pixel/action/label/override replay, finite reports, both
weighted losses, initial equality and 100k recovery all pass. Actor sharing
retains the earlier native fixture qualification; this full audit does not
independently remeasure internal graph ownership.

Guidance overrides 42.549% of training actions; the 20% setting selects episodes,
not actions. Training spends **96.182%** of actions below 216 points, with none
at 432 or above and maximum observed score 417 including partial tails. These
are cumulative-score bands, not wall events or frozen competence. Late labeled
imitation CE is .546367 versus retained .560514 on different own replay; lower
fitting loss does not establish better control. Learner stage time is 1,141.649s,
versus 63.117s acting, 62.570s environment and 12.393s observing. No representation
cause follows: categorical sampling, KL sums and active one-hot counts changed
together despite unchanged 16,334,353 parameters and flattened latent width.

Original main and collector finish **11:42:01.706 /11:42:12.947 UTC November 6**;
all three original services are inactive/success with their original PIDs gone.
All **356 current /338 retained pins** and sealed completion artifacts pass.
The existing read-only publication reader completes in **.047s**, with no helper
source change. No full preparation or production stage fails or repeats; the
manual read-only status-poll correction above is preserved separately. Original
native readback failure remains failed alongside its successful test repair.
No additional learner, extension, sweep or successor is selected.

Artifacts remain at `/x/Temp/kindle-latent-shape-20261106/full`:

- `comparison.json`: SHA256 `6e4839947bac060c145943285e63fce66dad0e01f219142df8d3653764b0b0f0`.
- `completion.json`: SHA256 `c090e2d70905266cdca004ddcb372e9c4d3d2ce958eeabc3b6c7f18376809176`.
- `terminal-summary.json`: SHA256 `be7b8e639230666b6139fe7a9fcc4a5579aa670c0122d6430984dd451f196b58`.
- `result-read-execution.json`: SHA256 `f3884c5ce13155a77ef36969ec3eae7a17df30884e625c5d8f2ad369df5e0f37`.

Retain Breakout **143.667 greedy /70.750 sampled**. The original mean-864 and
90%-two-wall gate, and the five-game goal, remain unfinished.
