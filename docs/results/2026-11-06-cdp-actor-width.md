# Actor-only width passes the bounded cost check

The [width512 experiment](../experiments/2026-11-06-cdp-actor-width.md) and
[separate bounded protocol](../experiments/2026-11-06-cdp-actor-width-bounded.md)
pass their original executions. The fresh check completes **8,192 actions /
32,715 emulator frames /489 updates in53.270217s**,11.702% longer than the
retained bounded47.689684s. The padded200k projection is **23m50.597s**, below
30min. Conservative projected full-replay headroom is **11.214GiB**, above2GiB.
These are short-run projections; useful frozen learning remains unmeasured.
No full learner, sweep, extension or automatic successor is selected.

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
