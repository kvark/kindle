# A moderate encoder rate improves the cheaper Qbert recipe

The fresh12M48/B16/T16/H15/R16/expert100 learner with encoder rate2e-5 finishes
200k actions/799,728 frames/12,477 updates in30m16s. Its frozen greedy mean
13,785.417 improves the retained6e-6 control's11,984.375 by15.03%; sampled
9,987.500 improves6,759.375 by47.76%. Training is.58% shorter. Greedy clears24/24
first pyramids and sampled23/24, versus24/24 for both control modes. Both new
means still miss the original15,000/90%-first-pyramid gate.

| Encoder rate | Training wall | Frozen greedy / own initial | Frozen sampled / own initial | First pyramids, greedy / sampled |
|---|---:|---:|---:|---:|
| Retained6e-6 control |30m27s|11,984.375 /80.208|6,759.375 /177.083|24/24 /24/24|
| Fresh2e-5 candidate |30m16s|13,785.417 /80.208|9,987.500 /177.083|24/24 /23/24|

Each mode selects the first3 natural episodes in each of8 lanes=24. Excess
natural episodes and partial tails remain retained. The four new frozen arms
total106,424 actions/425,545 frames/77.763s,with zero updates and truncations.
Guidance and expert labels are disabled throughout evaluation.

Encoder6e-6→2e-5 is the only intended learner-setting change. Keep centered
CDP/12M48/Auto/N8/B16/T16/H15/R16/expert100,base/actor4e-5,dynamics4e-4,
warmup1000,coefficient1,actor-unimix0,full18 actions,microbatch16,replay100k,
all action weights1,world imitation gradients,20% episode guidance/cap1920,
all-state corrective labels and one GPU resize from native max-pooled frames.
Native447fe50 and Qbert teacher are unchanged. The qualified optional-rate
runner also contains Breakout corrective-label plumbing; its flag is false
here and the exact diff is retained. Matching initial tensors and traces do
not establish complete cross-version learning parity.

Seed4013/evaluation4000004013 and the cohort are reused development inputs;
no weights,replay,games,belief,RNG or bounded experience are reused. This is
one cheaper development learner,not independent confirmation or broad
cross-game reliability. The separate H5/R32 learners retain40m24s/40m22s costs
and greedy14,050/15,792.708; one of those two passes the original gate.
The present result does not measure combining H5/R32 with the moderate rate.

Actual79,334 guided overrides are39.667% of training actions. Guided online
returns and changing replay losses are not unaided competence or held-out
action accuracy. CPU observers,startup.535s,recovery confirmation.399s and
the completed Breakout own-policy analysis co-host with training; retain those
timing caveats without correcting wall time. Training wall excludes model
construction,evaluation,CPU replay/audits and prior development.

Root `/x/Temp/kindle-qbert-encoder-moderate-20261030/full`,trial
`qbert-12m-48-b16-r16-encoder2e5-expert100-200k-seed4013`. Source54b1b5d;
main invocation`33e82ad2f0dc4025bac23ba8383d9c6a`. No GPU stage is repeated.
The original pre-launch CPU recovery-assertion failure remains preserved.
Retain2e-5 as the stronger H15/R16 development point. No extension,rate sweep,
seed confirmation or automatic successor is selected. The five-game goal remains
unfinished.

## Completed validation

All235 declared pins,100k/6227 recovery,12,477 finite reports,357 finite and
control/bounded-matching initial tensors,357 invariant tensors in each frozen
arm,47 pre-update training events,1139 sampled/1101 greedy actual-initial
semantic events,changed CNN/reward/actor/value,full training input-pixel/action/
label replay,frozen outcome replays/videos,all five host guards,curves,coverage,
exposure and comparison pass. All2,991,367 labeled replay rows equal their
weight sum; maximum raw world/actor expert-loss difference is8e-7. Ordinary
frozen logs lack live pixel hashes; their outcome replays cannot compare
unavailable live images. Video hashes match their replay receipts. The comparison
plot is visually inspected.

Main succeeds02:17:51 UTC October30,postprocessing02:17:59,collection02:18:31.
Main,progress,recovery,postprocessing and collection all succeed; no GPU stage
repeats or host recovery occurs. Main service wall time is42m56.233s,including
CPU replay/audits; training itself remains30m16s. Terminal confirmation verifies
all235 pins,nine completion artifacts,four video hashes and all five states.
The journal retains the original PID/invocation after systemd removes the
completed transient units. No learner or GPU allocation remains active at closure.

Two ancillary CPU observation/metadata errors remain preserved separately from
the passing pipeline. An extra inline watcher expected a live invocation field
after transient-unit removal and raises an assertion; the original and terminal
journal entries verify the same invocation. The independent confirmation checks
pass,but its event-count metadata initially counts three dictionary keys. The
corrected confirmation uses the already-matched1139/1101 event counts and retains
the original receipt. Neither issue repeats an audit,collector or GPU stage.
The earlier pre-launch obsolete recovery assertion and its correction remain
preserved as well.

| Artifact | SHA256 |
|---|---|
| `completion-review.json` | `763b0f6ac082ecb9777179e1f36b5b7547b8213b1f07cd332dc7bc15a4a97e67` |
| `learning-review.json` | `81c483dd960500666aeb1f1448bfcc4beb9ef65b1f672397efa1073f89a88f84` |
| `postprocess-review.json` | `a154c07242a52462f931ca7df29885cadc0b4a4b74447a35cdc023599a0e8483` |
| `exposure-comparison/comparison.json` | `a12b399d23601e22147630872094b3233454aa475e6b5e34f80f9f9c47d96e11` |
| `guidance-replay.json` | `aab012ec4e57a6a4a566a7df62697aa83b085d284bab057b6c1306a748ff0256` |
| `terminal-confirmation-corrected.json` | `e93e84b54ec538c8ca90164a26ab6dd4b60ba6930c386cfd978a8c005a9c4825` |

The trial retains all four videos,`curve/`,`exposure-comparison/`,original and
corrected terminal confirmations. The parent root retains the ancillary
`terminal-observer-note.json` and verifier source.
[Fixed full protocol](../experiments/2026-10-30-cdp-qbert-encoder-moderate-200k.md).

## Earlier bounded execution and preparation

The fresh12M48/Auto/N8/B16/T16/H15/R16/expert100 check with encoder rate2e-5
completes8192 actions/32,754 emulator frames/489 updates in70.203s. Its1024
frozen actions/4096 frames take.806s,with zero updates and no natural episodes.
The conservative200k projection31m25s passes the35-minute cap. This establishes
execution and early cost only; Qbert gameplay benefit remains unmeasured.

The [bounded protocol](../experiments/2026-10-30-cdp-qbert-encoder-moderate.md)
transfers the fixed rate that improved compact Breakout to the cheaper Qbert
recipe. Only encoder6e-6→2e-5 changes as a learner setting. Native447fe50 and
Qbert teacher stay unchanged. The qualified optional-rate runner also includes
previously added Breakout corrective-label plumbing; its flag is false here,
and the exact diff is retained. No complete cross-version learning-parity or
causal representation claim follows. Seed4013/evaluation4000004013 are reused
development inputs; no weights or experience are reused.

All212 pins,357 initial tensors matching the retained12M control,47 pre-update
semantic events,357 finite/invariant frozen tensors,489 finite reports,changed
CNN/reward/actor/value,complete input-pixel/action/label replay and both host
guards pass. All489 reports are supervised:117,261 labeled rows equal weight
sums,with maximum raw world/actor loss difference8e-7. Labels include6631NOOP
and1561hops,3851 passive labels and3425 actual/teacher disagreements. Actual
4341 overrides are52.991% of actions; online guided returns are not frozen
competence. Trainable parameter count remains14,123,473.

CPU preflight passes both actual command paths,13 config assertions,190 control
pins and357 schema tensors in.317s without native/environment construction.
The bounded service starts01:29:14 UTC October30/source92bbc98,PID372626,
invocation`3c2f1004960d4d73b1b15d3ebc2995c8`,and succeeds01:31:01 after1m46.591s.
Terminal collection succeeds01:31:21. No bounded failure,GPU repeat or host
recovery occurs. Reading controller sources co-hosts briefly with the check;
no wall-time correction or isolated backend-speed claim is made.

A [separate fresh200k comparison](../experiments/2026-10-30-cdp-qbert-encoder-moderate-200k.md)
is selected after the bounded correctness/cost checks. It keeps the35-minute
training cap,100k recovery and four frozen final/actual-initial arms,greedy
primary against11984.375; the original mean15000/90%-first-pyramid gate is
unchanged. No check experience is reused. Its first CPU preflight catches an
obsolete copied recovery assertion; source and traceback remain preserved,
and the corrected complete-declaration check passes before full launch.

Root `/x/Temp/kindle-qbert-encoder-moderate-20261030/native-check`;
trial`qbert-12m-48-b16-r16-encoder2e5-expert100-check`. The bounded service is
terminal; check live state before GPU work. The five-game goal is unfinished.

| Artifact | SHA256 |
| --- | --- |
| `declaration.json` | `1b833779751cce2c8e3f887138989d4979b5cb2ae0f19268b96ca36c9752d3d0` |
| `review.json` | `2829d0cf6f088aacd6acb6ff2fa5d9e1a511f84dbdbf62863ed23c40e810e1ac` |
| `guidance-replay.json` | `e063a8751d7deda350a5779c88b17d2fe33be4efc8dc731ce553f14a368032fe` |
| `completion-review.json` | `221aa39abccfe248cffec139e541e585c1953f535a70591de2113ef04f046b2c` |

### Earlier launch and startup check

The fresh learner starts01:34:55 UTC October30/source54b1b5d,PID373087,invocation
`33e82ad2f0dc4025bac23ba8383d9c6a`. Startup at01:35:11 verifies all235 pins,
357 initial tensors matching control and bounded check,47 pre-update training
transitions,exact config and73 finite supervised reports through1544 actions.
First six updates376/392/408/424/440/456 match the declared cadence. No frozen
result is available. Progress,100k recovery,postprocessing and terminal-collection
observers are active. Do not restart this learner or schedule another GPU job
while it is active. The read-only.535s startup audit co-hosts with training;
retain this as a timing caveat without correcting training wall time.

Declaration`2da4231e56fac5f0382d2dd605ae7710fe49602e7a789abed1d15dfb9b77179b`;
startup`d0f8bd3998520219d9f525a742cd1bbd773b05d506969de3d8d8ff65e0ce8c85`.

### Earlier 100k recovery

At01:50:30 UTC October30,the original recovery observer verifies100,000 actions/
6227 updates and all357 finite tensors. It checks6381 finite reports in a
complete102,472-action prefix; last progress102,400/6377 retains credit.5. The
observer exits successfully. Read-only confirmation at01:51:16 rechecks all235
input pins and the original mainPID373087/invocation33e82ad2f0dc4025bac23ba8383d9c6a,
still active. Final200k and all four frozen assessments remain pending. No
learner interruption,new GPU work,cap change or frozen assessment occurs.

The.399s confirmation co-hosts with training; prior startup and CPU Breakout
analysis timing caveats also remain,with no wall-time correction. Recovery is
checkpoint-integrity evidence,not frozen competence. Keep the original35-minute
training cap and fixed evaluation protocol;do not restart or schedule a second
GPU job while this learner remains active.

Recovery review`052c2e69dc3d8c536950fcc0ebcc761912191520b3ae29f89a7eabf52b27eecd`;
checkpoint identity`7d00e39e549b47c442c16ca73082125985e193c9793ade8b4adfddb483ab5f6f`;
confirmation`cac442a2a92e3c119b58c0abbcafc00418116d026b93226bd819f2acb92831cf`.
