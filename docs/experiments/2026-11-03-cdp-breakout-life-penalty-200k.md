# Fresh 200k Breakout comparison with one-point life-loss feedback

**Complete; retain the raw-reward control.** Frozen sampled42.542/greedy90.875
fall39.87%/36.75% from70.750/143.667, all0/24 two-wall. Training completes200k
/799,584 frames/12,477 updates in21m39.349s, within the original30min cap.
All453 pins, recovery, paired initial/frozen checks and original services pass.
Terminal and fixed coverage readers are complete once; no stage repeats or new
allocation follows. [Full result](../results/2026-11-03-cdp-breakout-life-penalty.md).
The five-game goal remains unfinished. The following is the fixed protocol
and historical preparation record; do not repeat completed work.

Separately select one fresh learner after the [bounded native check](../results/2026-11-03-cdp-breakout-life-penalty.md)
passes 8,192 actions / 489 updates in 47.317s. All 408 pins, 357 matching initial
and invariant frozen tensors, raw-score/life/reward accounting, full guided
replay and host guards pass. All 118 observed lost lives receive negative
training feedback. The conservative 200k projection is 21m10.713s, below 30min.
Both original services succeed; preserve the CPU fixture, path and terminal
reader failures and their corrections. Do not repeat the completed check.

The hypothesis is that explicit negative feedback at lost lives may help the
learner distinguish paddle failures sooner. Life loss already affects future
scoring opportunities; the saved zero immediate rewards do not establish a
missing-loss bug. This comparison measures one additional training aid, not
an isolated architectural mechanism or independent seed reliability.

Use **200,000 fresh aggregate actions / expected 12,477 updates**, with a fixed
**1,800-second training cap**. Keep centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/
microbatch16/H15/R16/lambda.95, encoder/base4e-5, behavior8e-5, dynamics4e-4,
replay-value1, expert100/unit weights/world gradients/all-state corrective
labels, sampled collection/posteriors, actor-unimix.01, action-effects1,
replay100k, warmup1000, 20% episode guidance/cap1920, learnerseed1009 and
evaluationseed4000001009. Full18/sticky.25/repeat4/noops0, native pooled RGB
and one GPU64 resize remain. No weights, replay, belief, games, RNG or bounded
experience transfer. Native447fe50/SHA35ad0db and the teacher remain exact.

Use the qualified runner SHA9ff2552b and isolated runtime under
`/x/Temp/kindle-breakout-life-penalty-20261103/runtime`. Only training enables
`--breakout-life-penalty`: F32 training reward is raw reward minus net lost lives.
Preserve the versioned reward contract, separate raw/training fields, real
termination/truncation flags and reset baselines. ALE life counts are reward
feedback, never an actor input. Gains have no penalty; game-over losses count.
All online and task scores remain raw. Full guided replay must independently
verify actual ALE life counts and penalty arithmetic alongside pixels, actions
and teacher labels. The original raw-score audit remains enforced separately.

Require 357 initial tensors matching bounded candidate and retained control,
correct first update at376, and paired pre-update semantics. Against control,
normalize only the declared extrinsic training difference, retaining raw reward,
action, boundary, teacher and intrinsic fields; the qualified full auditor checks
the declared difference. Candidate/bounded comparison retains all shaping fields.
Retain 100k/6,227 recovery weights with exact metadata and the final checkpoint.

Run fixed final and actual-initial sampled/greedy assessments: first three
natural episodes per eight lanes, **24 per arm**, zero updates/truncations,
guidance/labels/shaping off. Each arm has a 300-second cap and unchanged
25,000-action episode limit. Require 357 invariant tensors per arm, matching
actual-initial raw semantic traces, full recorded-action/reward/frame/reset
replays and videos, finite training, changed CNN/reward/actor/value and guards.

Compare sampled and greedy separately against retained **70.750 / 143.667**
and own initials **1.417 / 2.208**, all zero of24 two-wall. Report actual time,
actions/frames/updates, raw and initial-subtracted return per training minute,
natural episode outcomes, both seeds and guidance exposure. Retain the original
90% two-wall gate. A development score gain is not that gate or seed reliability.
No checkpoint/cohort selection after outcomes, penalty sweep, retry, extension,
cap increase, confirmation or automatic successor.

Automate progress, 100k recovery, postprocessing and original terminal collection.
CPU preflight must exercise actual training/frozen commands, exact configuration
and shaping contracts, startup gates, recovery retention, observer arguments,
source/runtime provenance and terminal journal identity. Use the existing
4,500s main service and 1,200s CPU audit envelopes. Check live state before launch;
allow no overlapping Kindle GPU jobs. Retain CPU co-host timing caveats without
correcting training time afterward.

Prospective root `/x/Temp/kindle-breakout-life-penalty-20261103/full`, trial
`breakout-12m-64-b16-r16-life-penalty-expert100-200k-seed1009`, service
`kindle-breakout-life-penalty-200k-20261103`. Full-run preparation and launch are
unfinished at selection. Gameplay benefit is unmeasured; the five-game goal
remains unfinished.

## CPU preparation

Full preflight passes **06:14:37.261 UTC November 3**, source `2eafbf2`, in
2.237s (2.340s wrapper). Five actual commands, 34 exact configuration cases,
24 startup-header cases, seven pre-update comparisons, 12 recovery-body cases,
four observer-parser cases, eight prelaunch command cases, four metadata cases,
six retention cases and two collector-retention cases pass. All 19 scripts
compile and 834 declaration path entries resolve. The 408 bounded and 338
control pins remain unchanged. Native/teacher/qualified runner remain exact;
no native or emulator is constructed. No preparation failure or repeated
successful stage occurs. Full selection and launch remain pending at this milestone.

## Startup

Startup review06:16:35.584 UTC passes453 pins,357 initial tensors matching both
bounded candidate andretained control,47 complete candidate/bounded pre-update
events and47 normalized raw-control events. All shaping fields remain in the
candidate/bounded comparison; control normalization changes only declared
extrinsic feedback. The firstsix update actions376/392/408/424/440/456 match.
There are113 finite supervised reports through2,184 actions and46 observed lost
lives with verified training reward accounting. No100k recovery orfrozen result
is available atstartup. Keep1.509s CPU startup wrapper co-hosting uncorrected.

Original observers:post983166/invocation3a2e4714c1bb463fb74b2928b021f4a1,
progress983169/82c8677727184601a907b308a44c42dc,
recovery983304/87e7d42b5eab48d288f5120d1c043403,
collection983307/235dab3abb3a4e189c05af8f6bb2cd84. Follow these identities.

## Read-only result preparation

Terminal-reader preflight passes06:19:26.520 UTC in2.457s (2.478s wrapper),
with all453 pins,allfive original identities,five score cases andsix feedback
cases. ProvenanceSHA148984df/preflightSHAce7f24e6. The separate recovery reader
is prepared; run it only after the original100k observer succeeds. Neither
reader changes active controllers orrepeats original stages.

Before frozen outcomes, fixed raw-score coverage qualifies06:20:09.659 in2.464s
(2.484s wrapper),with50 diagnostic/453 active pins,eight configuration cases
andfive retained boundary fixtures. DeclarationSHAafdab4f3. Afteroriginal
completion andterminal confirmation, count only the candidate train/four frozen
logs using pre-action raw-score bins216/432/648/864. Training penalties never
enter these bins. Reuse completed control coverage:1,262 native actions at216–431,
none432+. Reconcile all candidate actions,frames andepisodes against original
replays,keeping fixed first-three-natural-episode means separate from extras/tails.

These are exposure bins,not verified walls,visual coverage orcausal evidence.
No new replay,assessment,learner orGPU allocation isselected. Run each reader
once after its prerequisites; do not repeat completed preparation. Retain CPU
co-host timings without wall-time correction. Frozen gameplay was pending at preparation.

## Retained recovery

The original recovery observer succeeds **06:27:24.862 UTC November3**.
Independent confirmation passes **06:27:40.709**, in1.868s (1.890s wrapper),
verifying all453 pins,the original observer journal identity,and a retained
**100,000-action /6,227-update** checkpoint with357 finite tensors and
206,067,162 bytes. The audited complete log prefix contains6,421 finite reports
through103,112 actions. Recovery-reviewSHA41f86c4d/retentionSHAf8529f6f/
metadataSHA72a9068e. Weights andmetadata are retained; replay,belief,games andRNG
are omitted. The original main983163/invocation0191b9d7c0fe47acbe15c6db9845f519
was active at this recovery review. No stage or observer repeats; do not repeat
this confirmation. Frozen gameplay was pending then. Keep CPU co-host time
without correction.

## Original completion

Original main983163/invocation0191b9d7c0fe47acbe15c6db9845f519 succeeds
06:42:06.048 UTC November3; postprocessing06:42:24.088 and collection06:42:57.559
also succeed. All five original services are terminal success. Independent
terminal confirmation06:56:45.055 takes1.304s (1.323s wrapper); fixed coverage
06:56:53.724 takes2.995s (3.016s wrapper). All453 full/50 diagnostic pins,
100k/6,227 recovery,357 matching initial/invariant frozen tensors,47 pre-update
and800/795 paired initial events, full pixel/action/label/life/reward replay,
frozen replays/videos, guards, curves, exposure and comparison pass. All1,290
observed life losses produce independently verified negative training rewards;
raw game scores remain unchanged. Four frozen arms total77,592 actions/
310,203 frames/67.331s, zero updates/truncations. Fixed coverage reconciles
277,592 actions/1,109,787 frames and reuses the completed control without recount.

Sampled/greedy initial-subtracted return per minute falls41.21%/37.87%; observed
training time rises0.89%. Native training exposure at raw scores216–431 changes
1,262→1,297, none432+, despite guided online mean67.78→94.72. Retain raw rewards;
no penalty sweep, extension, confirmation or automatic successor. Earlier
bounded CPU failures remain preserved; no full-run failure or repeated stage,
collector or completed reader occurs. Keep co-host timings uncorrected.
CompletionSHA4d276268/terminalSHA4e848210/coverageSHA96930bc4. The package is
closed and no Kindle allocation is active at closure. Check live state before
separately selected work. Full hashes and interpretation are in the result.
