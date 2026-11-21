# Wider actor passes cost qualification but regresses frozen Breakout

The [fresh200k comparison](../experiments/2026-11-06-cdp-breakout-actor-width-200k.md)
is complete and negative: width512 scores **87.125 greedy /39.958 sampled** in
**24m14.355s**, versus width256's143.667/70.750 in21m27.910s. Primary own-initial
gain/min falls46.29%. Retain the preset256 actor, with the optional width absent.
All production stages and terminal collection pass; no successor is selected.
This is one reused development seed, not broad capacity or reliability evidence.

## Completed bounded cost check

The [width512 experiment](../experiments/2026-11-06-cdp-actor-width.md) and
[separate bounded protocol](../experiments/2026-11-06-cdp-actor-width-bounded.md)
pass their original executions. The fresh check completes **8,192 actions /
32,715 emulator frames /489 updates in53.270217s**,11.702% longer than the
retained bounded47.689684s. The padded200k projection is **23m50.597s**, below
30min. Conservative projected full-replay headroom is **11.214GiB**, above2GiB.
These are short-run projections; useful frozen learning remains unmeasured.
At bounded closure, no full learner, sweep, extension or automatic successor
was selected. The separately declared full comparison is reported below.

Only `actor_hidden_units=512` changes the recipe, from the12M actor's256.
This adds1,054,720 trainable actor parameters, for17,389,073 total. CNN, RSSM
and critic widths stay fixed, as do losses and optimizer ownership. Keep
12M64/Auto/N8/B16/T16/H15/R16, encoder/base4e-5, dynamics4e-4, behavior8e-5,
critic1/1, expert100/world gradients, all-state corrective labels and20% guided
episodes/cap1920. Coordinates and actor dynamics remain off. See the protocol
for the complete pinned recipe; seed1009 is reused development data.

## Implementation and qualification

Implementation80085208 passes194 Kindle/5 adapter CPU tests,1,116 packaged
Python checks with zero native construction and strict workspace/binding Clippy.
The default-absent width preserves legacy serialization; explicit preset width
preserves graph structure. All actor owners share the configured width, while
non-actor graph shapes remain unchanged. The CLI rejects invalid widths, restore
overrides and unsupported native packages before construction.

The original guarded Tiny qualification finishes **08:00:10.932 UTC November6**
in2.820s: two synthetic learner updates, changed actor/critic parameters,
synchronized actor copies and exact parameter/moment restore. All243 pins and
the host guard pass; zero Atari actions. Native SHA256
`eeaf699a84e341b99a4c9d2341fb9113c0edfe221de0569be7964bde67c9b00a`.

CPU bounded preflight completes in.167s with zero native/emulator construction.
Both actual queued CLI paths reach their expected mocked boundaries; omitting
the fresh width flag is rejected. Initial-comparison fixtures accept changed
actor shapes/values but reject a changed or missing non-actor tensor.

## Bounded measurements

Sourceeeebd450 launches the two original GPU jobs serially. Training begins
with fresh counters and first updates at action376, ending with489 updates and
debt.5. Construction takes2.404257s. The1,024-action/4,096-frame frozen restore
takes.892191s, with zero updates and all357 tensors unchanged. It completes no
natural episodes and provides no competence score.

All313 non-actor initial tensors match the retained initial checkpoint exactly.
The44 actor tensors/copies/moments are excluded;40 shapes widen and four output
biases retain18 elements. Initial actor values/actions are not required to match.
Encoder, reward, continuation, actor and critic weights change during learning;
actor copies agree in all three checkpoints.

All489 reports pass finite-value and weighted-world/behavior-loss checks.
Maximum absolute recount errors are1.111e-4 and2.296e-5. All489 updates contain
corrective supervision,116,987 repeated labeled replay rows with equal weight
sum. Maximum world/behavior imitation loss difference is7e-7. Full CPU replay
verifies every training action, reward, frame, boundary and native-RGB teacher
label;3,928 overrides are47.949% of actions. Separate frozen CPU replay verifies
all actions/outcomes/frames/boundaries without constructing an agent. Frozen
logs lack live image hashes, so unrecorded image equality is not established.

Across489 reports, summed imagination time is20.663s, world training13.910s
and behavior training4.703s; total reported learner time46.881s. Corresponding
retained bounded values are16.647s,13.613s,2.502s and40.438s. These are wrapper
timings against a historical runtime; changed experience and binaries prevent
an isolated kernel-cost or causal learning claim.

Minimum native budget headroom is17,980,190,720bytes across3,098 snapshots.
Subtracting the entire5,939,200,000-byte full replay footprint leaves
12,040,990,720bytes. This conservative projection counts allocated replay pages
twice and is not independent GPU telemetry. Both guards pass with no faults,
nonfinite reports, failed stage or repeated stage.

Controller/audit finish **08:14:02.459 UTC November6** in70.966s; outer execution
succeeds in70.994s. All49 allocation pins and239 original source pins remain
unchanged. Root `/x/Temp/kindle-actor-width-20261106/bounded` retains exact
commands, source receipts, initial/final/frozen checkpoints, logs and replays.
No completed GPU stage, replay or audit should repeat.

At08:15:25.188 UTC, no Kindle service or render-device holder remains. A separate
receipt releases4,806,734,453 logical bytes of disposable build cache while
verifying unchanged wheel, native extension and packaged test executable. The
original packaging receipt is preserved. Any full learner needs a separate
declaration with own-initial frozen controls and return gained per minute;
bounded weights/experience must not be reused as fresh training. Retain the
143.667/70.750 control and original864/90%-two-wall gate. The five-game goal is
unfinished.

## Full 200k comparison

Only actor width512 changes the configuration beyond explicitly serializing the
existing imagined bootstrap default. World/critic widths, all losses, guidance,
learning rates, environment protocol and200k budget stay fixed. The candidate
starts fresh, transferring neither bounded weights nor experience. Native and
runner identities differ from the historical control; the scoped qualification
does not establish broad runtime equivalence or an isolated capacity cause.

| Measure | Retained256 | Actor512 |
|---|---:|---:|
| Training time |21m27.910s|24m14.355s|
| Final greedy return |143.666667|87.125000|
| Final sampled return |70.750000|39.958333|
| Actual-initial greedy |2.208333|1.333333|
| Actual-initial sampled |1.416667|1.666667|
| Greedy own-initial gain/min |6.590132|3.539370|
| Sampled own-initial gain/min |3.230038|1.579738|
| Two-wall successes, either final mode |0/24|0/24|

The learner completes **200,000 actions /799,552 emulator frames /12,477
updates in1,454.354977s**,12.924% longer than retained. Primary gain/min falls
**46.293%**. Paired eight-lane difference is−3.050762,95% interval
[−5.029410,−.914726]; raw greedy difference is−56.541667,
[−100.209375,−8.873958]. Sampled gain/min falls51.092%, paired difference
−1.650300,[−2.303949,−1.001726]. These10,000-resample intervals use bootstrap
seed0 and describe the reused evaluation cohort, not training-seed reliability.
The predeclared retention rule and original864/90%-two-wall gate both fail.

Each of four frozen arms selects the first three natural episodes in each of
eight lanes,24 per arm, with zero selected truncations or learner updates.
The arms total76,296 actions/305,013 emulator frames; all357 tensors remain
unchanged in each. Actual-initial scores differ as expected, so gain/min uses
each policy's own initial score. Initial sampled traces have235 matching of771
paired events; greedy has0 matching of658. Candidate/control-only events are
retained, without requiring equality across different actor initialization.

All357 initial tensors and47 pre-update events match the bounded candidate
exactly. All313 non-actor initial tensors match retained;44 actor tensors/
copies/moments are excluded. Eleven pre-update events differ from retained.
Actor copies remain synchronized. The100k/6,227 recovery passes its original
audit08:39:19.911 UTC and independent CPU verification08:39:45.022 in.372s:
357 finite tensors,347 unchanged pins and read-only files. The verifier does
no forward or update, and original learner identity is unchanged. Recovery
omits replay, live games, beliefs and full RNG state.

All12,477 reports pass finite-value and both weighted-loss checks. Training
replay verifies every action, reward, frame, boundary, override and native-RGB
corrective label. All frozen replays/videos, guards, exposure, curves and score
coverage pass. Late own-replay labeled imitation CE changes.560514→.553252,
despite worse frozen scores; replay distributions differ. Guidance overrides
89,164 actions(44.582%). Training spends193,652 actions(96.826%) below216
points,6,348 at216–431 and none at432 or above. These are coverage/fitting
measurements, not evidence of a causal bottleneck or unaided competence.

Learning-stage time rises1,133.927→1,309.611s. The new run's summed report times
include553.418s imagination,364.783s world training and122.711s behavior
training. Wrapper times, historical binaries and changed experience limit
isolated performance attribution. No timing correction removes CPU observers.

CPU helper/comparison/plot checks pass in3.541s; all five actual queued commands
qualify in.414s without native/emulator construction. Source91647453 launches
08:27:14.431 UTC November6, original PID1432519/invocation
`6e0453060296495b8d1062b0dadc9b37`. Training finishes08:51:33.239. Original main
and terminal collector finish **08:55:25.209 /08:55:47.274 UTC**; all three
services succeed. All347 current/338 retained pins pass. The separate publication
reader succeeds in.049s without executing a stage. No preparation or production
stage fails or repeats; no extra learner, evaluation, replay or collector runs.

Root `/x/Temp/kindle-actor-width-20261106/full` is closed. Initial,100k,final and
frozen checkpoints, logs, videos and exact receipts remain outside Git. The
build cache was already released after bounded qualification; the packaged
runtime remains. Keep actor width absent, coordinates off, actor dynamics off
and143.667/70.750 retained. No extension, sweep, confirmation or automatic
successor is selected. This weakens the specific wider-actor recipe, not every
capacity hypothesis. The five-game goal remains unfinished.
