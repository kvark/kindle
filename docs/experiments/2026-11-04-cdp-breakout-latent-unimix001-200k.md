# Fresh200k Breakout comparison with latent uniform mixing0.001

Select one fresh learner after the [bounded check](../results/2026-11-04-cdp-breakout-latent-unimix001.md)
passes execution and its fixed30-minute cost screen. The mixture interpretation
does not establish an empirical noise problem; this comparison measures whether
the changed training recipe produces useful frozen gameplay at the fixed budget.

Use exactly the bounded configuration and runner4070a0b8/native447fe50/SHA35ad0db.
Only latent `unimix` .01→.001 differs from the retained learner; actor-unimix.01,
sampled collection/posteriors,12M64/Auto/N8/B16/T16/H15/R16, rates, expert100,
20% episode guidance/cap1920 and Atari protocol stay fixed. Native pooled RGB
receives one GPU64 resize. No warm start or bounded-check experience is reused.
Seed1009 and evaluation4000001009 remain reused development choices.

Train200,000 aggregate actions/12,477 updates within1,800 seconds. Require the
100,000-action/6,227-update checkpoint, finite357 tensors and read-only retained
files. Verify357 initial tensors against both bounded and retained models, exact
47 pre-update semantic events against the same-configuration bounded run, declared
configuration/header changes, all finite weighted reports and changed learning
parameters. Use the original prepared recovery and terminal observers once.

Evaluate final/actual-initial sampled and greedy arms, each first3 natural
episodes per8 lanes=24. Disable all guidance/labels and updates; verify357
invariant tensors, exact counters and full action/outcome/frame/boundary replays
and videos. Each arm retains the existing action cap,100k-frame episode limit
and300-second wall cap. Do not increase caps, retry or substitute episodes.
Initial weights match but changed mixing may change initial trajectories:
record matched/different/unpaired initial semantic events without requiring
retained-control equality. Use the candidate's actual initial scores for gains.

Greedy is primary. Favor the candidate only if raw greedy return and
own-initial-subtracted return per training minute improve over143.667/6.590132,
and the10,000-resample paired eight-lane gain/min95% interval has a positive
lower bound. Seed0 for bootstrap; retain all paired results and sampled secondary
scores. The original task gate remains mean864 and90% two-wall success. These
reused development lanes do not measure reliability across training seeds.

Require complete training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting, actual-initial/final comparison and original
terminal collection. Guided online scores and own-replay fitting are separate
from frozen competence. Historical control timing and CPU co-host costs remain
uncorrected. Preserve the bounded CPU validation failures and terminal stdout
manifest repair; never change their original receipts or repeat completed stages.

Root `/x/Temp/kindle-breakout-latent-unimix001-20261104/full`; main service
`kindle-breakout-latent-unimix001-200k-20261104`. Only one GPU allocation may
run. Stop real faults/nonfinite values; no GPU reset/reload, extension, second
learner, coefficient sweep or automatic successor. Check live identity before
work, and never treat an observer timeout as a stopped learner.
