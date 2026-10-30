# Moderate Qbert encoder rate passes execution and early cost checks

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

## Launched and startup checked

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

## 100k recovery verified

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
