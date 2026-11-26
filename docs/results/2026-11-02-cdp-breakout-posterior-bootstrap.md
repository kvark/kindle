# Posterior replay bootstrap lowers sampled Breakout and fails the paired protocol

Retain the imagined replay bootstrap. The candidate's accepted sampled score
falls **70.750→53.708 (−24.09%)**, while training takes **21m37.556s**,
**0.75% longer** than the control's21m27.910s. Initial-subtracted sampled return
per training minute falls **25.14%**. The actual-initial sampled mean1.417
matches. The greedy final arm records two25,000-action timeouts, so the full
comparison is **closed_protocol_failed**, not an accepted paired result.

| Fixed assessment | Retained imagined bootstrap | Posterior bootstrap |
| --- | ---: | ---: |
| Sampled final mean,24 natural episodes |70.750|53.708|
| Sampled actual-initial mean |1.417|1.417|
| Greedy final accepted mean |143.667|Not accepted: two timeouts|
| Greedy actual-initial mean |2.208|2.208|
| Training time |21m27.910s|21m37.556s|
| Sampled initial-subtracted return/min |3.230|2.418|

The greedy natural-only cohort mean117.917 is retained solely as a censored
diagnostic. All four natural cohorts have0/24 two-wall successes. Its two
truncated episodes occur in lanes4/5 at aggregate actions200,000/235,008,
with returns46/48 and lengths25,000 each. The arm ultimately finishes three
natural episodes perlane in238,952 actions/226.218s, within its original cap.
Do not discard either timeout or redefine acceptance after seeing the result.

The [full protocol](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap-200k.md)
keeps12M64/Auto/N8/B16/T16/H15/R16,encoder4e-5/behavior8e-5/dynamics4e-4,
replay-value1/expert100,sampled collection and teacher fixed. It changes the
intended replay bootstrap imagined→posterior for both critic and representation
losses. Native11373c1/runner8952f81f differ from retained447fe50/cf4ea238.
Reused1009/cohort and changed sources prevent isolated causal or seed-reliability
claims. This result does not support the target change or identify a single
cause of the critic's earlier calibration error.

## Completed evidence and preserved failures

Training completes **200,000 actions /799,562 frames /12,477 updates**. All
**412 original inputs**,100k/6,227 retained recovery,357 matching initial tensors,
357 invariant frozen tensors perarm,800/795 actual-initial semantic events,
finite reports,replays/videos,host guards,curves and exposure checks pass.
The four frozen arms total **285,280 actions /1,140,815 frames /261.765s**,
with zero updates. Guided training replay verifies full pixel/action/label
consistency; ordinary frozen replay verifies recorded outcomes/frames/resets,
without unavailable live pixel hashes.

Actual training overrides87,981 actions (**43.9905%**). Final50k replay imitation
CE is0.561238 versus control0.560514; these training-fit values do not establish
a policy mechanism. Guided online scores remain separate from unaided competence.
CPU startup,reader qualification and recovery co-host costs remain uncorrected.

Original main928259/invocation7aac1e71b2de47f9934cb67d2f7ee1ef fails
**09:54:47.194 UTC November2** at the declared zero-truncation assertion,
after allfive GPU stages finish. Collection fails**09:55:09.359**. Original
postprocessing,progress and recovery terminate successfully. Every host guard
passes. The original sampled-final and sampled-initial replays/videos are
complete before the audit rejects the greedy arm; neither replay is repeated.

The [separate CPU closure](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap-closure.md)
preserves534 inputs and runs only the three missing replays: final-greedy,
initial-greedy and guided training. They all succeed. The closure then fails
**10:06:17.630** on an overly strict assertion rejecting ALE imports:
`audit_atari_tasks.py` imports ALE for score/replay identity checks. Its source,
declaration,successful replay outputs and failed service remain unchanged.

A separate repair reuses **every completed replay**, preserves **553 inputs**,
and corrects that import-only check while still requiring no native Kindle
import. It finishes summaries,exposure and explicitly failure-labeled curves
in **10.403s**, completing **10:11:27.827 UTC**; the10.437s wrapper exits0.
Terminal confirmation passes once**10:12:01.748**, verifying22 artifacts and
that no Kindle allocation remains active. The curve is visually reviewed.
No GPU stage,successful replay or original collector repeats. The original
main,collection and first CPU-closure services remain failed; no original
passing learning review,postprocess review,comparison or completion is created.

Evidence root `/x/Temp/kindle-replay-bootstrap-20261102`; failed CPU attempt in
`cpu-closure`, completed failure closure in`cpu-closure-repair`. Do not repeat
training,frozen captures,replays,collectors,terminal readers or qualification.
No extension,cap increase,retry,confirmation or automatic successor is selected.
Check live state before new work. Breakout and the five-game goal remain unfinished.

| Closure artifact | SHA256 |
| --- | --- |
| Original CPU closure declaration |`a5fce034abd4ef7efcdccc98a1b2c9a7134b3b9794aed208c669b9753b0ec86e`|
| Repair declaration |`d736288bf8cae2437a5a87cc89b1e2d0591f68db82b5bca3ae33fc3e0386f464`|
| Closed protocol-failure review |`de4c7aaddd1a9b45f967f3c3578121c0af5a1a69e8ee99949b64105130abe173`|
| Diagnostic summary |`211847531eba0ee4a9a42d1f86d491ec82cd13e322c49fce9106d8ac83c6da25`|

## Earlier bounded qualification

The optional observed-state bootstrap completes **8,192 actions / 32,726 frames /
489 updates in46.900s**. Its conservative200k projection is **20m59.510s**,
below the30-minute training cap. The following1,024 frozen actions take.879s,
with zero updates and no natural completed episodes. This establishes execution
and early cost; no gameplay benefit is established. The separately selected
[fresh200k comparison](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap-200k.md)
subsequently measures frozen gameplay without reusing this bounded experience;
its protocol failure and lower sampled score are reported above.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap.md)
tests a specific value-target hypothesis after the [calibration diagnostic](2026-11-02-cdp-breakout-value-calibration.md).
Real-replay lambda targets now optionally bootstrap with values at observed
posterior states. The default still uses their first imagined returns. Real
rewards, intrinsic bonuses, episode boundaries, discount, lambda, replay weights,
slow-value regularization and imagined actor/critic targets stay unchanged.
The revised replay targets feed both the critic and world representation loss.
The change does not establish that imagined bootstraps caused the prior error.

## Implementation and package

Native `replay_value_bootstrap` accepts `imagined` or `posterior`. Old metadata
defaults to `imagined`; parameter names/shapes and checkpoint format remain
unchanged. The fresh-only `--replay-value-bootstrap` option records the choice,
rejects restore/evaluation overrides and rejects older native runtimes that lack
support. No dependency, backend optimization, precision or teacher change occurs.

Continuing, terminal and timeout target fixtures pass for both modes. Config
default/roundtrip/rejection and actual CLI forwarding/capability checks pass.
The package passes **145 Kindle / 5 Gym Rust CPU tests**, both strict Clippy
checks and **628 packaged Python checks** with zero native construction attempts.
The resident-target shader compiles offline; this is not execution validation
of that path. `KINDLE_DEVICE_TARGETS` remains unavailable on this AMD host.
Current learning uses host targets and Auto precision. Loss derivatives are
unchanged; no unrelated gradient attribution is repeated.

Package source is **11373c17d9254924f87014ffd02fd8a4e2a091ca**, native SHA256
`f1cf431721daa1bb35c2fc14f33feec6774c949fa35fac2e74c56deaebe47ee8`.
Runtime `/x/Temp/kindle-replay-bootstrap-20261102/runtime`; runner SHA8952f81f.
Packaging completes08:59:43.938 UTC November2. Wheel, native, validation
executable and source receipts remain; the2.989GiB disposable build cache is
released after the bounded check. The immutable build receipt records its
original cache state; separate `cache-release.json` records removal.

## Default compatibility and fresh learning

Before training, the new runtime restores the retained replay-value1 model and
reproduces all**128** semantic events in its first**1,024 sampled actions /
4,096 frames**, in**.889s** with zero updates. All**357** frozen tensors remain
identical; config differs only by the explicit new default field. All28 pins
and the host guard pass. This is bounded acting/restore compatibility, not a
new evaluation score or a default-learning equivalence claim.

Compatibility service `kindle-replay-bootstrap-compat-20261102`, original
PID927398/invocation49922050132944f3aa312194fe7cc765, launches09:02:51.043 and
succeeds09:02:57 UTC. Its saved review completes09:02:57.958. It is terminal;
do not repeat the capture.

The fresh candidate retains12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled collection,
learner1009 and the retained RGB-only teacher. Full18/sticky.25/repeat4/noops0,
native max-pooled frames and one GPU64 resize remain. Native and runner differ
from control447fe50/cf4ea238, so do not claim an isolated causal effect.

All**373 pins**, **357 matching initial tensors**, **47 pre-update semantic
events**, **489 finite reports**, complete guided pixel/action/label replay,
frozen tensors and host guards pass. CNN, reward, actor and value parameters
change. Actual guidance overrides3,811 actions; all8,192 receive labels.
Guided returns are not unaided competence. The bounded experience is not reused.

Main `kindle-breakout-posterior-bootstrap-check-20261102`, original
PID927572/invocationff26aa4f6e07412ab2254b7549a12080, launches
**09:06:55.856 UTC November2/source11373c1** and succeeds**09:08:00.345**.
Original collector PID927575/invocation19991e85029a4d70a5a2fcb728ff764e
succeeds**09:08:08.088**. The read-only terminal reader completes once
**09:08:31.094** in5.585s. All original stages succeed with no preparation,
build, GPU, replay or collector failure and no successful-stage repeat.

Root `/x/Temp/kindle-replay-bootstrap-20261102` retains package, compatibility
and bounded evidence. Do not repeat qualification, captures, learner, replay,
collector or terminal reader. No GPU allocation is active at closure; check
live state before separately selecting the fresh200k comparison. No extension,
cap increase, sweep or automatic successor is selected. The five-game goal
remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Package | `782e4895134086e15a422eab509d616e80189a076840b0f13b2e201fc29e668c` |
| Default compatibility review | `f0461a4e83ed06df5e1603e8093bd9103810a48ad0594bc5926406bbc28a41d3` |
| Bounded declaration | `c712da89c139a1032e06aa0c5b6d20719538e7d06c90a266e062d54641589050` |
| Original completion | `efbe8f3b78b3902cf84d33749310b9c57ce3112db35aabe4a8b886e3bfa0301e` |
| Terminal summary | `efc898c83bfa1f6719a64e9382bbd1f7629b61fb6d48caf37854a2abe043d235` |

## Fresh200k launch and initial checks

The separately selected [full protocol](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap-200k.md)
launches **09:28:02.911 UTC November2/sourceb40979f**. Fresh learner1009 keeps
12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4,
replay-value1/expert100, sampled collection and the retained teacher. The
intended setting change is imagined→posterior replay bootstrap; native and
runner differ, so no isolated causal claim follows. No prior experience is reused.

The fixed budget is200k actions/12,477 updates and a30-minute training cap,
with retained100k/6,227 recovery. Four final/actual-initial sampled/greedy arms
retain24 natural episodes and the original90%-two-wall gate. Compare against
retained70.750/143.667 in21m27.910s; sampled is primary. At launch, all full-run outcomes
remained pending.

Before launch, allfive actual commands,18 controllers,34 configuration cases,
15 startup-header cases,seven prelaunch cases,11 recovery cases and retention/
collector checks pass in2.366s. All373 bounded and338 control pins remain intact.
Selection pins412 inputs; declaration SHA
`681c5c61fee8ed9c7b5547ba3a19d467e27118ace46b16804b6b396bd76f550a`.
No agent or environment is constructed during CPU preparation.

Main service `kindle-breakout-posterior-bootstrap-200k-20261102` retains original
PID928259/invocation7aac1e71b2de47f9934cb67d2f7ee1ef. Postprocessing,progress,
recovery and collection observers are started. Startup review passes
**09:28:19.243** with412 unchanged pins,357 initial tensors matching both bounded
and retained-control models,47 pre-update semantic events and correct cadence.
Its complete prefix has1,880 actions and94 finite supervised reports. The1.512s
CPU startup wrapper co-hosts with learning; retain this caveat without correcting
training time. No failure, repeated stage or runtime change occurs at startup.

At startup, `/x/Temp/kindle-replay-bootstrap-20261102/full` is active. Check live state;
do not restart the learner or schedule overlapping GPU work. No extension,
cap increase,sweep,confirmation or automatic successor is selected. Breakout
and the five-game goal remain unfinished.

## Terminal reader prepared during training

The separate CPU reader passes its qualification **09:34:44.522 UTC November2**
in2.417s (2.438s wrapper), while the original learner remains active. All412
inputs remain unchanged. Five score cases and six configuration cases accept
the retained evidence and reject incorrect scores, native/runner identities,
teacher identity, bootstrap mode and behavior rate. It imports neither native
Kindle nor ALE and performs zero GPU work or learning updates.

The success-only reader was prepared to verify allfive original service identities and terminal states,
completion hashes,100k retained weights,initial/frozen invariants,complete
replays/videos and the exact comparison before summarizing results. Its SHA is
`81d834e5ffbe17ac1e5df1d5e017bc77426a4145bb5cadfc57d26425132e945f`.
The separately prepared recovery reader has SHA
`d2bfa62a51bc69ee43206e510018f2113315634a5562d583fb0e5617af3cfa0c`.
Both remain outside the active controller tree; no runtime,teacher,queue,
controller,checkpoint or observer is changed. Retain CPU co-host time without
correcting training time. At qualification, recovery and terminal execution remain pending. Do not repeat
their qualification or any completed stage. It is not executed after the original
comparison fails; the separate protocol-failure closure is described above.

## Halfway recovery is retained

The original recovery observer PID928398/invocation81257011a0fd44459734d819a75f5a9a
succeeds **09:39:09.306 UTC November2**. It preserves the100,000-action/
6,227-update checkpoint before final overwrite: metadata plus world,behavior
and slow-value tensors, **206,067,205 bytes**. All357 tensors are finite. Its
complete audited training prefix has102,504 actions and6,384 finite reports.
These are the observer's prefix counters, not the checkpoint's counters.

The separate read-only confirmation passes once **09:39:49.658** in2.418s
(2.437s wrapper), preserving412 inputs, all retained file hashes, matching
metadata/configuration and the original main process identity before and after.
The original journal contains exactly one checkpoint pass and a successful
watcher completion. Checkpoint review SHAe7ee50c9, retention SHAb17957d5,
metadata SHAc66c8d3c. The recovery observer is terminal; do not restart or repeat it.

At recovery confirmation, main PID928259/invocation7aac1e71b2de47f9934cb67d2f7ee1ef
continues unchanged; postprocessing,progress and terminal collection remain active. No GPU work,
learner update, replay or observer is repeated by the confirmation. Retain its
CPU co-host cost without correcting training time. The retained checkpoint
omits replay,belief,games and RNG and is a warm start, not an uninterrupted
snapshot. At that milestone, final200k learning and allfour frozen outcomes
remain pending; their terminal outcome is reported above.
