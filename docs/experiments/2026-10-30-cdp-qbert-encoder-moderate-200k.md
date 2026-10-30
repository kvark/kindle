# Fresh Qbert comparison with the moderate encoder rate

The [bounded check](2026-10-30-cdp-qbert-encoder-moderate.md) passes8192 actions/
489 updates in70.203s, followed by1024 greedy frozen actions in.806s with zero
updates and no natural episodes. All212 pins,357 matched initial/invariant
frozen tensors,47 pre-update semantic events,full replay and both guards pass.
Its conservative200k projection31m25s passes the35-minute training cap. Select
one fresh200k/12,477-update learner to measure frozen gameplay benefit.

Keep the exact checked Qbert12M48/Auto/N8/B16/T16/H15/R16/expert100 recipe,
with encoder learning rate2e-5 instead of the retained control's6e-6. This is
the only intended learner-setting change. Keep native447fe50 and Qbert teacher,
base/actor4e-5,dynamics4e-4,warmup1000,coefficient1,actor-unimix0,full18 actions,
replay100k,microbatch16,all-state corrective labels,world imitation gradients,
all action weights1 and20% episode continuation guidance/cap1920. Preserve
native max-pooled frames and one GPU resize to48. No auxiliary loss is enabled.

The qualified optional-rate runner also contains previously added Breakout
corrective-label plumbing; the Qbert command leaves that flag false and selects
the original Qbert corrective path. Its exact diff and native/teacher identities
are retained. This is not a claim of complete cross-version learning parity.
Seed4013/evaluation4000004013 and the evaluation cohort are reused development
inputs. No weights,replay,belief,games,RNG or bounded experience are reused.

Train exactly200,000 aggregate actions/12,477 updates under2100s, first update
376,residual credit.5. Save and audit100k/6227 recovery. Evaluate fixed final
and own actual initial with sampled and greedy actions,greedy primary. Each
arm selects first3 natural episodes per8 lanes=24,with sampled posteriors,
guidance/labels off,zero updates/truncations,357 invariant tensors and300s/
600k-action caps. Keep the original mean15000/90%-first-pyramid gate. Retained
control greedy11984.375/sampled6759.375 clears24/24 first pyramids in both modes,
with own initials80.208/177.083. Its30m27s training cost remains the reference.

Require357 actual-initial tensors matching control and bounded check,matched
pre-update training events and both own-initial frozen semantic traces,exact
source/config/counters,finite reports,changed CNN/reward/actor/value,equal
labeled-row/weight sums,aligned raw expert losses,full training pixel/action/
label replay,frozen outcome replays/videos,guards,curves,coverage,exposure and
completed comparison. Ordinary frozen logs lack live pixel hashes; their
outcome replay cannot compare unavailable live images. Guided returns and
changing replay losses are not unaided competence or held-out accuracy.

Caps:training2100s,each frozen arm300s,training replay1200s/outer1230s,CPU audit
1800s,main5400s,postprocessing6000s,terminal collection6300s. Recovery's fixed
2700s observation window does not authorize restarting the learner. Stop on
real faults or nonfinite reports; no driver recovery,GPU retry,cap increase,
extension,checkpoint selection,rate sweep,seed confirmation or automatic
successor. Preserve failures and repair CPU-only defects without repeating
completed GPU stages. The five-game goal remains unfinished.

Root `/x/Temp/kindle-qbert-encoder-moderate-20261030/full`,trial
`qbert-12m-48-b16-r16-encoder2e5-expert100-200k-seed4013`,service
`kindle-qbert-encoder-moderate-200k-20261030`. Check live state before launch.

## CPU preparation passes

Corrected preflight passes in.942s: five actual CLI paths,24 audit/comparison
config cases,49 resolved paths,15 compiled controllers,six header-readiness
cases,four observer-parser cases and five recovery-body cases. All212 bounded
and190 control pins,357 initial tensors and checkpoint/update accounting pass,
with zero native/environment construction. The original copied recovery watcher
has two partial assertions including obsolete expert1000; the CPU preflight
rejects its structure before declaration or launch. Preserve that source and
traceback under`preparation-failure`. Reuse the qualified watcher that compares
the whole config and guidance against the declaration. No GPU stage repeats.
