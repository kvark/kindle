# Life-bounded actor targets pass native and short-run qualification

The optional life-bounded observed-return actor completes 8,192 Breakout actions,
32,725 emulator frames and 489 updates in **49.039 seconds**. The retained
whole-game version took 50.754 seconds. This establishes execution and acceptable
early cost; it does not establish better gameplay. The 1,024-action frozen check
has no completed natural episodes and performs zero updates.

Follow the [fixed protocol](../experiments/2026-11-03-cdp-life-return-actor.md).
This remains within the five-game goal. Keep the retained 200k result,
sampled70.750/greedy143.667, and the original mean864/90%-two-wall gate.
No full learner, coefficient sweep or extension is selected by this qualification.

## What changed

The [preceding CPU audit](2026-11-03-cdp-breakout-return-life-audit.md) found that
later lives supplied much of the positive auxiliary actor weight on fixed
retained states. That was an arithmetic dependence, not evidence of a causal
learning defect. Full-game return remains a valid game-total objective.

`replay_actor_auxiliary_boundaries` is false by default and omitted from default
serialized configurations. It requires the completed-return actor and explicit
observed lower-bound mode with active disagreement. The vector observation API
accepts a boundary flag per arrival. A boundary closes only the auxiliary
observed return, including the reward on the arrival. Its outgoing target stays
pending until the next segment completes. Later completion fills that target
without overwriting the preceding segment. True terminals, timeouts, resets,
interleaved streams, ring eviction and replay context keep their existing rules.

The fresh-training flag `--breakout-return-life-boundaries` counts life decreases
at each raw emulator frame before four-frame pooling. Gains alone do not cut a
return; a loss followed by a gain within one action remains detectable. The
adapter forwards boolean boundaries and separately logged loss/gain counts.
Rewards, observations, episode flags, recurrent resets, critic targets and
continuation supervision remain unchanged. Frozen evaluation omits the adapter.

The auxiliary loss remains coefficient100 times executed-action cross-entropy,
weighted by `clip((G_segment − V_combined) / return_scale, 0, 1)`. Targets and
input representations are detached. No parameters or graph arithmetic were
added. Because the critic still estimates the full combined game return, this
is conservative positive reinforcement, not a per-life advantage estimate.

## Qualification

Source `ec12d41ec832498d097f989b1cd83aa1ccd32e33` passes 169 Kindle and five gym
Rust CPU tests, 1,044 packaged Python tests with ten native-construction tests
deselected, and strict workspace/binding Clippy. The packaged suite records zero
native construction attempts. Tests include adjacent boundaries, terminal rewards,
timeouts, reset separation, retained suffixes, forward F64 return agreement,
unchanged replay sampling, malformed API/log data, raw-frame loss/gain accounting,
the complete runner lifecycle and default/frozen omission.

The guarded native fixture passes 12 F64 loss/derivative/mask cases with the
new configuration in .83 seconds, preserving actor/critic parameters. The new
runtime then reproduces 128 retained frozen semantic events on 1,024 actions
and leaves all 357 tensors unchanged. No prior GPU stage was repeated.

Runtime `/x/Temp/kindle-life-return-actor-20261103/runtime` has native SHA256
`6a211fb5192cb5cd5a167b62925ae3494323a019a864234b2ba4aa831d6c3ffc`.
The wheel, validation binary, build cache and exact source receipts are retained.
Before editing, 214 source files from `5560be60` were preserved separately.

## Bounded training and independent replay

Keep 12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1,
encoder4e-5/dynamics4e-4/behavior8e-5, policy1/critic1/replay-value1,
expert100 with world gradients, disagreement1, actor100/lower-bound,
seed1009/full18/sticky.25/repeat4/noops0, sampled collection/posteriors,
teacher20%/cap1920 and one native-frame GPU resize. The new boundary flag is
enabled. No prior weights, replay, games or experience transfer.

| Measurement | Life-bounded candidate | Retained whole-game short check |
|---|---:|---:|
| Actions / updates | 8,192 / 489 | 8,192 / 489 |
| Training elapsed | 49.038937s | 50.754478s |
| Completed return rows, including repeat exposure | 88,128 | 55,469 |
| Positive rows, including repeat exposure | 34,261 | 33,588 |
| Updates with positive targets | 469 | 431 |
| First positive target action | 568 | 1,224 |

The candidate's clipped weight sum is31,844.706426 and unscaled auxiliary loss
sum348.540283. Imagined/replay bonus report sums10.031324/10.068238 stay positive.
All489 reports are finite, loss-component accounting passes, and the first
update remains action376. CNN, reward projection, actor and critic weights change.
Earlier target availability follows from closing segments sooner; counts across
different subsequent trajectories do not establish a learning improvement.

All357 initial tensors exactly match the preceding candidate and retained
baseline. The47 pre-update semantic events match after excluding only the new
boundary fields. An independent CPU replay reconstructs every pixel input,
executed action, teacher label, reward, frame count, episode and reset. It also
checks every boundary using separate raw before/after life-count pairs:
**123 losses, zero gains and123 boundary actions**. Replay takes5.217s.
There are3,686 guided overrides (44.995%); the21 training episodes include aid.

Frozen1,024 actions /4,096 frames take.902988s, with zero updates, no natural
completed episodes and all357 tensors invariant. Config metadata retains the
optional mode, while the frozen runner records no life-boundary aid fields.

The padded200k projection is1,316.964s, or21m56.964s, below the30-minute gate.
Observed training cost is3.38% lower than the retained short check, but this is
one changed trajectory on a different native revision, with normal host activity
and status reads. It is not an isolated speed estimate or a measured full-run
duration. Native learn calls consume40.539s (82.67%); these stage timers are not
a kernel profile. No timing correction is applied.

## Closure and retained failure

Main service `kindle-life-return-actor-check-20261103`, original PID1062560 and
invocation`8911df9127764f5b8e38a7c843ddd521`, completes19:15:26.762 UTC November3.
Legacy, training and frozen GPU stages each run once and all host guards pass.
All263 current and338 retained pins remain unchanged.

The original standalone CPU collector fails because it expects the preserved
source manifest's `files` to be a dictionary; the actual schema is a list. Its
source and failure receipt remain. A separate collector fixes that iteration,
verifies all214 preserved files, independently recounts actor reports/life events,
and closes19:16:56.676 UTC in1.213s (1.265s wrapper). No GPU stage or replay repeats.
Closure is `complete_with_cpu_collection_repair`; the original main service is
successful and the original collector remains failed. Two earlier atomic patch
application failures and the required formatting pass are also recorded; neither
changed production behavior. Build, CPU tests and native fixtures pass first try.

Evidence root: `/x/Temp/kindle-life-return-actor-20261103`, bounded subdirectory
`check`. No Kindle learner or GPU allocation remains active at closure. Do not
restart this check or its original collector. Check live state before new work.
Gameplay benefit needs a separately declared full comparison with the actual
initial controls; execution success and target counts cannot select a recipe.

| Evidence | SHA256 |
|---|---|
| Build receipt | `d5cca6faf4a62314181a1594ac8064c6569c44b1d993b3142e2115ba34ef269b` |
| Native fixture review | `2db9c33f2aaec11bf15c5699d9444cbe723dadddb1c19ede12c568c701323c93` |
| Bounded declaration | `2a40bb95985a66fe65d7e63713c05a6e97f1b9425e4559fc58e38fee9c144aee` |
| Independent pixel/life replay | `6946aded6dfe3c8e0ea25f662e5734abb647fdcb5d3ca02a52f510eee986b1d4` |
| Main completion | `83b72adc547d5a8227a06185e5b2ddb323e1aed4130d65b930778e6a82b715cd` |
| Repaired terminal summary | `3f2f56d300d3e9a707902f143cf50f38009d3a9f1f57ca9237803bc8a797e43e` |
