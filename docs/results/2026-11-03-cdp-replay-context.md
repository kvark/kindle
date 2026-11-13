# Replay-state recomputation passes execution and early cost checks

The separately selected [full comparison](2026-11-03-cdp-breakout-context9.md)
subsequently lowers sampled70.750→46.583 and greedy143.667→91.167, all0/24
two-wall, in23m30.959s (+9.55%). Retain context1. The completed bounded
qualification below establishes execution and early cost, not a gameplay gain.

An optional eight-step observed prefix completes a fresh Breakout check of
**8,192 actions / 32,730 frames / 485 updates in 50.564s**. The conservative
200k-action projection is **22m37.921s**, below the declared 30-minute gate.
This qualifies execution and early cost. It does not establish better recurrent
states, frozen gameplay or a full-run recipe.

The motivation is the [replay-context age reconstruction](2026-11-03-cdp-breakout-replay-age.md):
late sampled starting states had a median age of246 model updates. That age
does not measure latent error or identify the cause of weak Breakout control.
The retained gameplay control remains sampled70.750/greedy143.667, all0/24
two-wall successes. No full learner was selected at this bounded closure; the
separately selected full comparison is linked above.

## What the option changes

`replay_context=9` starts from one cached posterior, recomputes the following
eight observed steps with current world parameters, then trains on16 steps.
Recorded incoming actions and resets apply throughout. The prefix runs in a
separate inference session, uses an independent RNG, contributes no loss
targets, and detaches its final state before training. Only the16 training rows
refresh replay caches. There are no new parameters: checkpoints retain357
tensors and16,334,353 trainable parameters.

The default context1 keeps its prior path and omits prefix timing from reports.
Causal video replay still requires context1. Fresh-run CLI `--replay-context`
checks native capability and rejects restore overrides. Internal diagnostic
paths that have not recomputed a supplied prefix explicitly reject it.

The prefix changes replay eligibility, fresh-window stride, sampling and
warmup. The first update moves376→440 actions; there are485 updates at8,192,
and12,473 expected at200k, four fewer than the control. Training still has256
positions and at most240 labeled actor targets per update. This would not be
an isolated test of cache age or an equal-update comparison.

The check keeps centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4, policy1, critic1/1, constant actor/world
expert100, unit action weights, all-state corrective labels, replay100k and
seed1009. Guidance is20% of episodes, capped at1920 actions, using the retained
pixel teacher. Collection/posteriors stay sampled; H3, centroid, crop and decay
are off. Native pooled RGB receives one GPU resize to64. No prior experience
is reused.

## Measured cost and correctness

Prefix recomputation totals **3.234s**, averaging **6.668ms/update**, or6.40%
of measured training wall time. Mean per-update timing is not a causal estimate
of net overhead: trajectories, sampling, warmup and the native/runner versions
also differ. The largest accumulated learner stages are imagination16.417s
and world training13.505s; posterior sampling takes5.493s. Learner timings total
43.273s of50.564s wall time. These short-run host timings are not utilization
measurements or a full-run speed guarantee.

All485 reports are finite and contain positive prefix timing. CNN, reward,
actor and value parameters change. Raw world/actor expert losses align within
8e-7;116,066 replay rows are labeled. There are4,001 actual training overrides
(48.8403% of actions). Guided online returns are not unaided competence.

All357 initial tensors match the retained control, as do the47 common
pre-update semantic events. A context1 legacy capture reproduces128 retained
events and leaves357 tensors unchanged. The candidate's frozen1,024 actions
take **.914s**, perform zero updates and preserve357 tensors; no natural
episodes complete. Complete training pixel/action/label replay and all three
host guards pass. The optional prefix's last state also agrees with the
corresponding longer posterior under identical uniforms and an interior reset;
inference leaves parameters unchanged. Device replay matches known host rows
through eviction and cache refresh.

## Validation and preserved failures

Native source **65dd69b4**, SHA256
`7b15cb69bb20c3ac78f2853af3923d3b7db94fd88d67d8537ccd1b81a7415c70`,
passes155 Kindle+5 gym CPU tests,983 packaged Python checks and strict workspace
and binding Clippy. Two earlier CPU failures remain: a test incorrectly indexed
a padded tail target, and an unused reset helper was flagged after the shared
mask refactor. Their corrections and original logs are retained.

The posterior-prefix native fixture passes once. The first device fixture fails
before replay because it constructs RGB32 through the feature-only constructor.
Test-only source **694c64bd** corrects geometry and adds a CPU geometry check;
156 Kindle+5 gym CPU checks and strict Clippy pass. Production replay source is
byte-identical outside the test module, the packaged runtime is unchanged, and
only the failed device fixture is rerun. Both final native guards and235 pins
pass at13:05:46.605 UTC November3. The original failed fixture remains failed.

The bounded service launches13:07:09 UTC/source694c64bd, mainPID1028546,
invocation46eab86433784e3e9bf3456863ce64ed. Legacy and training finish successfully,
but the queue controller fails13:08:10: its copied result assertion still expects
489 updates despite the fixed protocol and declaration requiring485. A separate
continuation executes only the frozen stage that had not started, under its
original60s cap. No training or frozen stage is repeated.

The first CPU audit then expects decay-only scale fields that constant imitation
correctly omits. A separate CPU repair checks their absence and uses the exact
declared constant scale100. Full trajectory replay runs once after this repair.
All601 original and631 repair pins remain unchanged. Terminal collection passes
once at **13:11:47.928 UTC** in2.590s. Closure is
`complete_with_cpu_controller_repair`; the original service remains failed.
No driver recovery, cap increase or prior successful stage repeat occurs.

This runtime and runner differ from retained447fe50; only the explicit default
`replay_value_bootstrap=imagined` is normalized. Matching initial values and
legacy traces do not establish broad runtime equivalence or an isolated causal
effect. CPU co-host activity remains a timing caveat without correction.

[Fixed protocol](../experiments/2026-11-03-cdp-replay-context.md).
Root: `/x/Temp/kindle-replay-context-20261103`; packaged runtime: `runtime`.
Bounded evidence: `native-check/breakout-12m-64-b16-r16-context9-check`.
Wheels, test binaries, build cache, failed attempts and all receipts are retained.
The check and its readers are closed; do not repeat them. No GPU allocation
remains active at closure; check live state before new work. The five-game goal
remains unfinished.

- Bounded declaration: `5de0c4842abfe549dc94c201170b5fd410c92b40cbf3893b90dee2d43397175f`.
- Repaired audit: `61e6302a28751299c228c44426e26e5612e894d944b36e677ceb9e546813f053`.
- Full replay: `d0a1ecaa82d37ce058d99e83f432074f2e4e4bc2135aa1b80bcf8810e022f581`.
- Completion: `706c49025d6a91907829532fc458559a8647d4616115acf1a815ff87317b028a`.
