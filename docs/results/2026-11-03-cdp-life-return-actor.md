# Life-bounded actor does not improve retained Breakout at 200k

The [completed full comparison](#full-200k-comparison-does-not-improve-the-retained-recipe)
scores62.417 sampled/137.542 greedy in21m41.269s, below retained70.750/143.667
at slightly higher cost. Keep the retained recipe; the five-game goal is unfinished.

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

## Full 200k comparison does not improve the retained recipe

The separately selected [full protocol](../experiments/2026-11-03-cdp-life-return-actor-200k.md)
completes200,000 actions /799,589 frames /12,477 updates in **21m41.269s**.
Compared with the retained recipe, sampled62.417 falls11.78% and greedy137.542
falls4.26%, while training takes1.04% longer. Initial-subtracted score per
training minute falls12.92% sampled and5.31% greedy. All final arms still have
zero of24 two-wall successes. **Retain the existing200k recipe.**

| Recipe | Sampled mean | Greedy mean | Training time | Greedy gain/min |
|---|---:|---:|---:|---:|
| Retained, auxiliary actor off | 70.750 | 143.667 | 21m27.910s | 6.590132 |
| Whole-game actor100/lower-bound | 49.167 | 144.750 | 21m46.724s | 6.544994 |
| Life-bounded actor100/lower-bound | 62.417 | 137.542 | 21m41.269s | 6.240065 |

Each final arm selects the first three natural episodes in each of eight lanes.
Own actual-initial means are1.416667 sampled and2.208333 greedy, exactly matching
the retained and whole-game controls. Greedy is the primary mode. Its paired
candidate-minus-retained mean difference is−6.125, lane-bootstrap95% interval
[−53.168,44.917]; gain/min difference−.350067 has interval[−2.533782,2.011629].
Sampled differences are−8.333333 [−34.666667,21.208333] and−.417398 gain/min
[−1.636540,.950374]. These reused-cohort intervals span zero; the point estimates
and declared selection rule reject promotion, without establishing reliable harm.

Against the contextual whole-game actor, sampled mean rises26.95% while greedy
falls4.98%, for.42% less training time. Sampled return difference13.25 has
interval[−5,31], and gain/min difference.620134 has interval[−.219829,1.439705].
Greedy return difference−7.208333 has interval[−67.75,47.626042]; gain/min
difference−.304929 has interval[−3.088359,2.219088]. This contextual comparison
also fails its primary greedy selection rule. All bootstrap estimates use10,000
fixed paired lane resamples, seed0; none estimates training-seed reliability.

### Targets, fit and exposure

The optional loss executes throughout training with positive targets on11,932
updates, first at action568. There are2,729,915 completed-return rows and613,108
positive rows, counting repeated replay exposure; clipped weight sum336,860.554861
and unscaled auxiliary loss sum1,584.539574. Imagined/replay bonus report sums
80.933067/81.225762 remain positive. All12,477 finite reports pass loss-component
and expert-world accounting. CNN, reward projection, actor and critic weights
change. No prior experience or weights transfer.

Independent108.880s CPU replay verifies every training pixel input, action,
corrective label, reward, frame count, episode/reset and raw life change:
**1,288 losses, zero gains, 1,032 nonterminal boundaries and256 terminal
boundaries**, with zero timeout boundaries. The per50k loss counts are
517/322/234/215. These observed counts mix changing policy, guidance and episode
duration; they do not isolate a survival improvement caused by the new objective.

The whole-game actor had1,010,324 positive repeated rows and12,403 positive
updates. In the final50k window, the life-bounded actor has90,251 positive rows
and31,697.224460 clipped weight, versus220,960 and75,527.439039 for whole-game
targets. These are different learned trajectories and batches, not paired
counterfactual targets. More conservative credit is active, but better gameplay
does not follow at this budget.

Late teacher-label cross-entropy improves to.543307 from the retained.560514 and
whole-game.559403. Guided online last50 return is83.26, with91,607 actual action
overrides (45.8035%). These fitting and aided-play measures do not replace the
lower frozen scores. Training reaches a maximum raw score419;1,201 native
actions occur at pre-action scores216–431 (.6005% of training), and none at432+.
The retained control had1,262 such native actions and the whole-game actor939.
Score bands are not verified wall events, and these observations do not establish
the architecture's causal limitation.

Native learn calls consume1,147.828s, or88.21% of training elapsed time; acting
and environment stepping take63.266s/63.123s. These are stage timers, not a kernel
profile. Preserve the .306s startup check, .497s early CPU review, .294s recovery
observer, .681s independent recovery verification and ordinary status reads as
co-hosting caveats. No training-time correction is applied.

### Full evidence and closure

Use the same qualified native `ec12d41e`, runtime and byte-identical bounded
runner. Settings remain12M64/Auto/N8/B16/T16/H15/R16/context1, retained split
rates, expert100/world gradients, disagreement1 and actor100/lower-bound with
life boundaries. Development seed1009 and evaluation cohort4000001009 are
reused. Native/runner revisions differ from both historical full controls;
exact bounded/initial compatibility is narrower than full execution equivalence.

All330 current and338 retained pins pass. All357 initial tensors match the
bounded candidate and historical controls. The47 pre-update events match the
bounded candidate including life fields; comparison with older runs excludes
only those added boundary fields. Both actual-initial frozen traces reproduce
799/794 semantic events against both controls, excluding only wall clocks.

The original100k/6,227 checkpoint is read-only, with357 finite tensors and
51,502,659 elements. Independent verification succeeds19:35:44.899 UTC while
the original learner continues. Recovery preserves weights and optimizer
moments, but omits replay, game state, beliefs and full RNG state; it is not an
exact-resume checkpoint. Do not repeat the completed verification.

Four frozen arms total92,552 actions /370,036 frames /70.341s, with zero
updates or truncated episodes and357 invariant tensors per arm. All full
recorded-action replays, videos, source/ROM/host guards, learning curves,
guidance exposure, score coverage and paired comparisons pass. The completed
learning figure is visually checked; its curve is aided online return and its
caption reports the separate sampled frozen score.

Original main PID1063215, invocation`335925ab865f49d280efb8eaa211fc55`, completes
**19:50:54.540 UTC November3**, source`e6e5f4ba`. Original terminal collector
PID1063218/invocation`297599e0fc9f4180bb9adee0a4e9ef41` succeeds
**19:51:02.459** in3.231s. Main, recovery and collector services all terminate
successfully. The five GPU stages each execute once. There is no full-run
failure, repair, cap change or repeated training/evaluation/replay/collector.
The preceding bounded CPU collection failure and repair remain preserved.

Root `/x/Temp/kindle-life-return-actor-20261103/full` is closed. Do not restart
this learner, repeat its stages or rerun its observers. No Kindle allocation
remains active at closure; check live state before new work. No extension,
coefficient sweep, confirmation or automatic successor is selected. The original
mean864/90%-two-wall gate and five-game goal remain unfinished.

| Full-run evidence | SHA256 |
|---|---|
| Declaration | `5803f807e0a5a02e792eea42387e123ea2261c849606e1fcdc272a8c8f8d67d9` |
| Pixel/life replay | `178ade2922cfddeadfd3d605f1483c8972832078ba70f6b4848edd6238ab9afa` |
| Score coverage | `11092b98c5007640aea99e7c014adc3b1f9ff581c8abe70905e962a8901a5284` |
| Comparison | `57b587d433c1f49c16d4ebf49470d16602bae3f1223ca1cd44180cc26589e201` |
| Main completion | `2dd9f9ceaafcb7351ad83144720c6bcf938aef5ccca09d74ed28314ec8f1ba28` |
| Terminal summary | `1fa2c6912b18dad732ced185b14e8360209ccc1084d3da359c9e93f5f6652559` |
