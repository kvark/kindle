# Fresh Breakout actor/critic-rate comparison at the fixed 200k budget

Select preparation of one fresh200k learner after the [bounded check](2026-11-02-cdp-breakout-behavior8e5.md)
passes8,192 actions/489 updates in47.149s, conservative21m06.197s. Its original
main andcollector complete successfully; all252 pins andaudits pass. Full
selection requires the actual full command/audit/observer qualification below,
an idle host andclean committed sources. Do not reuse bounded experience.

Only the shared behavior rate changes fromunset (effective4e-5) to8e-5. Keep
base/encoder4e-5,dynamics4e-4, centeredCDP/12M64/Auto/N8/B16/T16/full BPTT/
microbatch16/H15/R16, sampled collection/posteriors, actor-unimix.01,
action-effects1, replay100k, expert100/unit weights/world gradients/all-state
corrective labels,20% guided episodes/cap1920. Full18/sticky.25/repeat4/noops0,
native pooled RGB andone GPU resize stayfixed. Learner1009/evaluation4000001009
are reused development seeds. Native447fe50/SHA35ad0db, teacher andrunner6c1b69fb
are unchanged byte-for-byte. No build, loss change or additional per-update work.
Actor andcritic rates change together; their individual effects remain open.

Train exactly200,000 aggregate actions /12,477 updates with a1,800s cap. Retain
the100,000-action/6,227-update recovery checkpoint andsource metadata/hash using
the original observer. Save actual initialization andfinal weights. A checkpoint
omits replay, beliefs, games andRNG; it doesnot represent uninterrupted restart.
No checkpoint selection, warm-start extension, cap increase or rate sweep.

Run four frozen assessments: final/actual-initial, sampled/greedy. Guidance and
labels areoff; select the firstthree natural episodes per eight lanes=24 perarm,
with600k-action/300s caps, zero updates/truncations. Sampled remains primary.
Compare the retained4e-4 dynamics/base-behavior recipe's56.292 sampled/104.667
greedy, bothzero of24 two-wall successes, in21m28s. Keep the original90% two-wall
gate. A development score improvement alone doesnot complete the five-game goal.

Require357 initial tensors to match both bounded initialization andthe control;
match pre-update semantics andboth actual-initial frozen trajectories. Verify
the actual config differs only in`behavior_learning_rate`. Require finite
supervised reports andaligned actor/world losses, complete pixel/action/label
training replay, allfour frozen replays/videos,357 invariant tensors perarm,
host guards, recovery, curves, exposure andpaired comparison. Ordinary frozen
logs lack live pixel hashes; outcome/frame/reset replay doesnot establish
unavailable original live-image equality. Online guided returns andtraining
imitation loss are not unaided competence. Retain CPU co-host costs without
wall-time correction.

CPU qualification parses allfive actual commands, tests actual config/header/
recovery assertions andobserver arguments, checks retention andcollector fixtures,
resolves private paths andscreens audit/comparison/controller globals. It verifies
all bounded/control input hashes without native or emulator construction. Pin
all sources, schemas, protocols, queues, runtimes, checkpoints andevidence before
launch. Preserve original failures if any; never repeat completed GPU stages.

Root `/x/Temp/kindle-breakout-behavior8e5-20261102/full`; trial
`breakout-12m-64-b16-r16-behavior8e5-expert100-200k-seed1009`; main
`kindle-breakout-behavior8e5-200k-20261102`, with original progress,recovery,
postprocessing andterminal-collection observers. Main cap4500s,audit1200s;
observers retain their fixed declared deadlines. No automatic successor,
confirmation, extension, retry or cap increase. The first GPU launch and startup are recorded below.


## Full CPU qualification and separate selection

Qualification passes00:28:25.327 UTC November2 in1.399s (1.498s wrapper):five
commands,28 config/12 header/four observer/eight recovery-body/four metadata
cases,six retention/two collector fixtures,44 paths/18 scripts,seven global
screens andmatching collection unit lists. All252 bounded/227 control pins
pass; no native/emulator construction orfailure. PreflightSHA256
`3268fe898cf6cb569377f99115c2c5fa973900794bfc0cafd8c36a0759b1fc7b`.

Separately select the fixed200k learner after these checks andan idle-host
verification. The bounded original main/collector areterminal success andclosed.
Native,teacher,runner,settings andcaps remainfixed. The first GPU launch and startup are recorded below.


## CPU prelaunch failure and correction

The first launch invocation stops at 00:32:12.591 UTC November 2 before creating
any service or GPU allocation. The separate preflight still expected the copied
`--dynamics-learning-rate 1e-4` flag. The retained runner instead uses its fixed
4e-4 dynamics default and the selected `--behavior-learning-rate 8e-5`. The
original qualification parsed all actual commands but omitted this preflight
assertion; preserve that coverage gap and the original passing receipt.

Only this CPU assertion changes. A separate 2.911s repair qualification completes
00:33:24.500 UTC (3.012s wrapper), executes the actual corrected preflight and
rejects wrong and missing behavior-rate flags. The other 270 original pins,
queue, native, runner, teacher, settings and caps are unchanged. No native or
emulator construction, service start or repeated GPU stage occurs. Original
sources, declaration and failure remain in the package's `prelaunch-failure/`
directory and `full-launch-execution.json`. Repin the corrected controller and
repair evidence before the first actual GPU launch; do not rerun selection or
successful qualification stages. The fixed 200k comparison remains selected.


## First GPU launch and startup

The corrected, repinned declaration launches at **00:34:24.392 UTC November 2**,
source3099ff7. MainPID873316/invocation`56e692c47ef348ff8acd60089b7b46df` and all
four original observers are active. Declaration SHAd8d4f582 pins282 inputs,
including the original CPU failure and separate repair. This is the first actual
GPU launch; the earlier failed invocation created no service.

Startup passes at00:34:37.150 in1.019s wrapper: all282 pins,357 finite initial
tensors matching both control and bounded initialization,47 identical pre-update
transitions, updates at376/392/408/424/440/456 and77 finite supervised reports
through1,608 actions. Original process identity, native/runner/teacher, exact
configuration and fresh counters pass. No full frozen result is available.
Keep the original process and observers; do not restart or overlap GPU work.


## Recovery milestone

The original100k/6,227 recovery observer succeeds00:45:30.437 UTC November2.
357 finite tensors and6,419 finite reports through103,064 actions pass. Separate
0.870s CPU confirmation at00:46:33.271 verifies all282 inputs, retained files,
metadata and the original recovery journal while the same learner remains active.
No checkpoint creation, observer or GPU stage repeats. Continue the fixed run;
no full frozen result is available.
