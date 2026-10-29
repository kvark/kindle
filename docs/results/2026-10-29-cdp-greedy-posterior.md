# Modal posterior categories lower frozen Qbert score

## Retain sampled posterior states with greedy actions

The separately selected full assessment lowers trained Qbert mean
**11984.375→10539.583 (−12.06%)** when live posterior categories use their modes
instead of samples. Both clear24/24 first pyramids,but both miss the original
mean15000 gate. Retain sampled posteriors with greedy actions for this learner;
no extension,seed confirmation or global inference-default change follows.

| Frozen checkpoint | Sampled posterior | Modal posterior |
| --- | ---: | ---: |
| Trained mean score |11984.375|10539.583|
| Trained first pyramids |24/24|24/24|
| Actual initial mean score |80.208|0.000|
| Actual initial first pyramids |0/24|0/24|
| Trained full-game gate |Fail|Fail|

Actions are greedy in every arm. Each collects exactly24 natural episodes,
first3 per8 lanes,with no truncations; partial tails remain recorded. Modal
posteriors lower the trained mean in five lanes and raise it in three. Per-lane
trained means are:

| Lane | Sampled posterior | Modal posterior |
| --- | ---: | ---: |
|0|10283.333|9008.333|
|1|12658.333|11550.000|
|2|11350.000|12658.333|
|3|13875.000|11208.333|
|4|11358.333|11416.667|
|5|11333.333|5550.000|
|6|12591.667|10116.667|
|7|12425.000|12808.333|

These lanes share one retained learner and are not independent learner-seed
confirmation. Removing posterior sampling changes the states supplied to a
policy trained with stochastic categories; the result does not identify why
the score falls. Different trajectories also prevent an isolated inference-speed
claim. No weights,training settings,action mode or game gate are changed.

Four arms total **110,112 actions /440,249 emulator frames /67.014s** of frozen
evaluation,zero updates. Trained sampled/modal arms take28.931s/28.802s for
48,256/46,376 actions; initials take5.119s/4.162s for8,432/7,048 actions.
The original200k-action/12,477-update/30m27s training cost remains unchanged.

All53 declared pins,configuration/source/counter checks,357 finite and invariant
tensors per arm,complete recorded-action/outcome replays,full lane0 videos and
host guards pass. The default final/initial arms reproduce **6080/1100 semantic
transition/episode/reset events** and both retained means exactly,excluding only
finite wall-clock timestamps/durations. Ordinary live logs contain no input
pixel hashes; replay verifies outcomes and frame/reset accounting within that
stated scope. No GPU stage in the full assessment repeats.

Service`kindle-posterior-qbert-eval-20261029`,source`f278e42`,runs
**10:10:12–10:12:26 UTC October29**,2m14.123s including audits and CPU replay.
Terminal collection passes with no active GPU allocation or host recovery.
Nativeb76f91c/SHA7555bf9 and the corrected bounded qualification below remain
the matching runtime. The five-game goal is unfinished.

[Fixed full-assessment protocol](../experiments/2026-10-29-cdp-greedy-posterior-qbert.md).
Evidence: `/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/full`,
including all four logs,exports,replays/videos,per-episode metrics and reviews.

- Declaration: `0fc84a11dd59a12e5671179a3d46d7706fa371b8485159d121e17ea9f377649e`.
- Comparison: `d2766558010c20c970671d29637567e1256fb604c611b7a36f91fb867abb0efe`.
- Completion: `c32605fdfea6435622896a266639fbd1d64b338a23d7aea2f2730015ebf93f93`.

## Bounded native and restore checks pass after correcting the test readback

Optional `--greedy-posterior` selects modal categorical RSSM states independently
of action sampling. It consumes no live-posterior RNG and disables subsequent
learning for that instance. The CLI accepts it only for frozen restores and logs
the mode explicitly. Default sampling, training settings, checkpoint config and
weights are unchanged; restoring a checkpoint defaults to sampled posteriors.

The guarded native test compares selected categories with their actual posterior
logits, verifies unchanged RNG state, exercises episode resets and rejects mode
selection after observations, including terminal arrivals. Both learning entry
points reject the evaluation mode. The existing vector/serial belief and policy
comparison also passes, covering sampled and greedy actions on the default path.

Two Qbert12M48/Auto/N8 restores of the retained final checkpoint complete
**1,024 actions /4,096 frames each**, with greedy actions: sampled posteriors
take.856s and modal posteriors.826s. Both have zero updates and no naturally
completed episodes. All357 tensors remain finite and unchanged;50 declared pins,
configuration/counters/source identity,complete action/outcome replay and host
guards pass. The sampled-posterior arm reproduces128 retained transition rows
exactly after removing only finite wall-clock timestamps. These short checks
establish execution and compatibility,not task competence or a speed advantage.

The first native test fails **10:01:30 UTC October29**, before default parity or
either restore. It used `Session::read_output_by_index` without the required
producer wait, allowing stale logits to be compared with a later state readback.
The corrected test waits for both observation and state-commit producers.
The revised validation passes under a separate declaration/service; production
native library bytes remain identical. Original49-pin declaration,failed guard,
logs,journal,source and validation executable remain preserved. The host guard
records native exit101,not a host fault. No driver recovery occurs. Default
parity and the two game restores execute once each.

Corrected source`b76f91c` passes **135 Rust CPU tests,625 packaged Python tests
and strict native/binding Clippy**. Ten native-construction tests are excluded
from the Python CPU count,whose eager construction guards recordzero attempts.
Native tests requiring a GPU are recorded separately. Service
`kindle-posterior-sync-check-20261029` succeeds **10:05:45 UTC October29**,with
17.232s total wall time,followed by successful terminal collection.

Use `/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/runtime` and its
matching packaging receipt. Native SHA256:
`7555bf94d94d34a28fe8c862436dc834e7852f305c657ecb756d5107c23fe1a6`.
The857,454,521-byte build cache is released after qualification; both original
and revised wheels/test executables and all evidence remain. The earlier parked
draft,its CPU logs and pre-change source archives are also preserved.

[Bounded protocol](../experiments/2026-10-29-cdp-greedy-posterior.md).
Evidence root: `/x/Temp/kindle-greedy-posterior-20261029/resumed`;
corrected checks and packaging are under `synchronized`.

- Corrected declaration: `d2683f9c2520fc944b6c253374c4bee898f104b3a9dbf327541792c2fd7595fc`.
- Corrected review: `2534e68a887fe8c98ef413bc59200480218ed51208e610f72a2c88b30a5c8cfe`.
- Corrected completion: `b9bb138846adda005b2f9ac5348a6e4f5e6acbba05840ccc46c33eda7bb7a192`.
- Packaging receipt: `322da00a101d41b50d7863252319f846ba10b1b6d154f3c08528d8aa3e498edf`.
