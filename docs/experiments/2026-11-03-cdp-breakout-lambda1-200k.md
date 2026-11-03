# Fresh 200k Breakout comparison with return lambda1

Separately select one full learner after the [bounded lambda1 check](../results/2026-11-03-cdp-breakout-lambda1.md)
passes execution, restore, trajectory and cost validation. Its8,192 actions/489
updates take50.121s; conservative200k projection22m26.028s is below30minutes.
All385 pins,357 matched initial/invariant frozen tensors,47 pre-update events,
full guided replay and guards pass. Both original services are terminal success.
Do not repeat the bounded learner, frozen stage, replay, collector or reader.

The development hypothesis is that using the whole available return trace may
help when intermediate value estimates are inaccurate. Only lambda.95→1 changes
from the retained replay-value1 control. It changes both imagined actor/value
and real-replay targets; endpoint bootstrap, discount and actual last/terminal
handling remain. This may increase variance and imagined-reward bias. It is not
complete-episode Monte Carlo learning, a replay-only change or a proven defect.

Use a fresh200,000 aggregate actions, expected12,477 updates and a fixed
1,800-second training cap. Keep12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16,
encoder/base4e-5,behavior8e-5,dynamics4e-4,replay-value1,expert100/unit weights/
world gradients/all-state corrective labels, sampled collection/posteriors,
actor-unimix.01,action-effects1,replay100k,warmup1000,20% episode guidance/cap1920,
learnerseed1009/evaluationseed4000001009. Full18/sticky.25/repeat4/noops0,
native pooled frames/one GPU64 resize,native447fe50/SHA35ad0db and teacher stay
fixed. Reuse the qualified ten-line runner overlaySHA9eb21cca. No weights,
replay, live belief, games, RNG or bounded-check experience transfer.

Require357 initial tensors matching both the bounded candidate and retained
control, plus matching pre-update semantics. Retain100k/6,227 recovery weights
with exact metadata and keep the final checkpoint. Run fixed final and actual-
initial sampled/greedy arms: first three natural episodes in each of eight
lanes,24 per arm, zero updates/truncations, guidance and labels disabled.
Each arm has a300-second cap and the unchanged25,000-action episode limit.
Require357 invariant tensors per arm, exact paired actual-initial semantic
traces, full action/reward/frame/reset replays and videos, and host guards.

Compare sampled primary and greedy separately against retained70.750/143.667
and own initials1.417/2.208, allzero of24 two-wall. Report training wall time,
actual actions/frames/updates, raw and initial-subtracted return per minute,
natural episode outcomes, both seeds and actual guidance exposure. Keep the
original90% two-wall gate; a development score gain does not satisfy it or
establish seed reliability. Preserve losses, cutoffs, protocol failures and
inference-mode tradeoffs without selecting a checkpoint or cohort afterward.

Automate progress,100k recovery, postprocessing and original terminal collection.
CPU preflight must exercise actual command parsers, exact config assertions,
startup gates, observer arguments, recovery retention and all declaration paths.
The training service cap is4,500s, total CPU audit cap1,200s; observer/collector
deadlines retain their existing bounded envelopes. No overlapping GPU jobs,
cap increase, retry, extension, confirmation or automatic successor. Check live
state before launch. Root `/x/Temp/kindle-breakout-lambda1-20261103/full`.

## CPU preparation

The first.114s preflight stops before any model/environment construction because
the collector assembles November2 observer names while the new services use
November3. Its execution receipt and20 exact source/provenance files remain
under`full-preflight-failure`. A separate date-wiring repair changes no native
or runner code. Corrected preparation takes2.247s (2.346s wrapper), validating
five commands,34 config cases,18 startup-header cases,four observer argument
cases,12 recovery-body cases,four metadata cases,six retention andtwo collector
corruption cases,18 scripts and826 declaration path entries. All385 bounded/
338 control pins match. No successful experiment stage repeats; no GPU work
has occurred during this preparation.

## Launch and startup

Launch04:13:38.340 UTC November3/source958d622 selects448 pins; declarationSHA
`aa991176cb6cc06fd585f52ceb3c2d4ccdb96e915208a8d0c6268e892a0410c8`.
MainPID972800/invocation44e2a4cbb446493f9db029c194c27300,post972803/f4656a088f74416083015c395da92a10,
progress972806/379160324b634c1a937ef77b003b10f9,recovery972938/d4a54f459ce54965a211d9e7ceab0eab,
collection972941/c9711214ef3141c58edad6f03511000c are active atstartup review.
All357 initial tensors match both bounded andcontrol checkpoints;47 pre-update
transitions matchcontrol and241 finite reports cover4,232 actions. The first
six updates are376/392/408/424/440/456; logged4,096 actions/233 updates take
22.950s. StartupSHA`7e9c8e0411af08f6336264163b5fec9bdbe5eb984cba4c3885f6451785030bf8`.
Retain1.500s CPU wrapper co-host time without correction. No full frozen result
or100k recovery exists atthis review; follow the original services without
restarting oroverlapping GPU work.

## Read-only result preparation

The separate recovery and terminal readers are prepared while the original
learner runs. Terminal preflight completes04:19:53.388 UTC in2.422s (2.448s
wrapper), checking all448 pins, allfive original process identities, the sole
lambda configuration difference andfive score validation cases. Its summary
SHA is`214de6ebf14d204ea987c049bac08976c7b183389e92173e99c0cb2d822cfaa3`.
These readers do not rerun the recovery observer, training, assessments,
replays, postprocessing or terminal collector.

Before any frozen outcomes, a fixed CPU coverage follow-up is qualified at
04:20:27.325 UTC in2.441s (2.462s wrapper), with50 diagnostic/448 active pins
andfive configuration cases. It retains five already qualified boundary cases
without repeating them. After original completion and terminal confirmation,
count only the new candidate's training andfour frozen logs, using pre-action
score bins216/432/648/864. Reuse the completed replay-value1 control counts:
1,262 native actions in216–431 andnone at432+. Reconcile all candidate actions,
frames andepisodes against original replay outputs; keep fixed first-three-per-
lane frozen scores separate from extra episodes andtails. These bins describe
exposure, not verified wall boundaries or a causal learning bottleneck.

The selected coverage declarationSHA is
`544781deea3362c52024b7d9fea3515c025ed68e6aacda5bc713e5be0e041c15`,
under`/x/Temp/kindle-breakout-lambda1-20261103`. No new GPU work, learning,
emulator replay or successor is selected. CPU co-host times remain uncorrected.
Run the new terminal andcoverage readers once after their prerequisites;
the completed preflights andsource preparations must not repeat.

## Retained recovery

The original recovery observer succeeds **04:24:44.552 UTC November 3**.
The checkpoint contains exactly **100,000 actions / 6,227 updates**, 357 finite
tensors and matching metadata, with 206,067,161 bytes retained. Independent
confirmation passes **04:25:14.741** in 1.770s (1.788s wrapper), verifying all
448 pins and the original observer's journal identity. The complete log prefix
has 6,377 finite reports through 102,408 actions. Recovery-review SHA is
`e460e8299549cba433ef5e944909cf803ccb140ce60c1ca0ca942c74897779e7`;
retention SHA is`d7b719421c22422ebc2e4f240eee8497cf62310fc3df195577bfd41c83612266`.

Only weights and metadata are retained; replay, live belief, games and RNG are
omitted. The original main process remains active with the same PID/invocation;
frozen gameplay is pending. No recovery observer, training or GPU stage repeats.
Do not repeat the completed recovery confirmation. Retain CPU co-host cost
without correcting training time.
