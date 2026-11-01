# Fresh Qbert 300k comparison at the retained 12M/H5/R24 settings

The [bounded check](2026-11-01-cdp-qbert-300k.md) completes at 17:39:38 UTC
November 1, with original terminal collection at 17:39:42. All 497 inputs,
357 matching initial/invariant frozen tensors, 47 pre-update semantic events,
733 finite updates, full pixel/action/corrective-label replay and host guards
pass. Training 8,192 actions takes 48.112s; 1,024 frozen actions take .819s with
zero updates and no completed episodes. No failure or stage repeat occurs.
The bounded conservative 300k estimate is 32m18.118s; the completed faster-teacher
cost reference gives 34m26.421s. Their maximum passes the fixed 35-minute cap.

Select one fresh **300,000-action / 28,090-update** learner. This tests whether
50% more experience and updates at the retained ratio improve Qbert's later
play, after replay ratio, shorter imagination, larger capacity and higher
behavior rate did not improve the primary result. It changes the budget,
not the retained native model or learning configuration. The faster Python
teacher also differs from the original 200k control, with qualified equivalent
decisions and lower CPU cost. Do not attribute total time changes to the native
backend or claim an isolated experience mechanism.

Use 12M/RGB48/Auto/N8/B16/T16/full BPTT/microbatch16/H5/R24, base and encoder
4e-5, dynamics4e-4, behavior unset (base4e-5), warmup1000, greedy collection,
sampled posteriors, actor-unimix0, action-effects1, expert100/unit weights/world
gradients and all-state corrective labels. Keep 20% episode continuation
guidance/cap1920, full18/sticky.25/repeat4 and one GPU resize from native pooled
frames. Native447fe50/SHA35ad0db, runneraba4300d and faster teacherfa261c53 stay
unchanged at `/x/Temp/kindle-qbert-color-masks-20261101/runtime`; no build.

Reuse learner10007/evaluation4000010007 as development choices. Start from
fresh initialization, games, replay, beliefs and RNG. Require all 357 initial
tensors and both bounded/control pre-update traces to match. Keep first update
at action376 and residual credit .75. Retain all four checkpoint files at
100k/9,340 and 200k/18,715 updates before later writes replace the working copy.
These are recovery artifacts, not intermediate evaluation or model selection.

Train to the fixed 300k endpoint within **2,100s**. Assess final and actual initial
models in greedy-primary and sampled modes, three natural episodes per each of
eight lanes (24 per arm), 300s per arm. Guidance is off; require zero updates,
zero truncations and all 357 frozen tensors unchanged. Retain the original
mean15,000/90%-first-pyramid gate. Compare with the completed 200k control's
greedy14,339.583/24 and sampled13,761.458/24, own initials1.042/171.875 and zero
pyramids. Report absolute scores, own-initial gains, time and gain per minute;
do not select whichever evaluation mode looks best afterward.

Require complete training pixel/action/corrective-label replay, all four frozen
outcome replays/videos, host guards, finite reports, changed CNN/reward/actor/
value, paired initial traces, curves, coverage, measured guidance exposure,
comparison and original terminal collection. Training replay/audit/service caps
remain 1,200/1,800/5,400s. Progress, recovery, postprocessing and collection
observers must retain original process identity and failures. CPU co-host time
is disclosed without wall-time correction.

No restart of a completed life, checkpoint extension, retry, cap increase,
budget sweep, checkpoint selection, automatic confirmation or successor.
Any failure stays recorded; do not repeat a completed GPU stage. This is one
reused development seed, not independent reliability or five-game completion.

Root `/x/Temp/kindle-qbert-300k-20261101/full`; trial
`qbert-12m-48-b16-h5-r24-encoder4e5-greedy-expert100-300k-seed10007`; service
`kindle-qbert-300k-20261101`. Qualify all five CLI paths, configuration/header
guards, both recovery milestones, collector integration, schedule, referenced
paths and pinned inputs on CPU before declaration and launch. Check live state;
no overlapping GPU job is allowed.

CPU qualification passes at 17:41:16 UTC November 1 in 3.393s: five CLI paths,
28 configuration checks, five teacher-identity cases, eight collection-mode
cases, 16 header cases, four observer parsers, 11 recovery-header cases, six
retention fixtures, four retention/collector integration cases at both
milestones, 51 resolved paths and 17 compiled scripts. All 497 bounded and
442 control inputs and 357 initial schemas pass. No native agent or environment
is constructed, and no preparation/qualification failure occurs.


The full learner launches at17:42:11 UTC November1/source7bd5001 with524 pinned
inputs, PID826399/invocation6947a8ed4d734296a1cff8e1af8113b6. Original progress,
recovery,postprocessing and collection observers are active. Startup passes at
17:42:48:357 initial tensors match both controls,all47 pre-update transitions
match,first six updates follow376/392/400/408/424/432,and485 finite supervised
reports cover5,544 actions. Declaration
SHA7bd6d5a12991fcc74d003f06c40aa0dac4908aee9fa11797624c40190979fe51.
The1.545s CPU startup wrapper co-hosts with training;retain without time correction.
No failure or GPU-stage repeat occurs;no frozen gameplay result exists yet.
Do not restart the learner or schedule another GPU allocation while it is active.
