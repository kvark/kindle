# Fresh Qbert H5/R32 comparison with encoder2e-5

The [bounded check](../results/2026-10-30-cdp-qbert-encoder-h5.md) passes
8,192 actions/978 updates in95.153s, then1,024 frozen actions in.798s with zero
updates and no natural episodes. All288 pins,357 matching initial/invariant
frozen tensors,47 pre-update semantic events,full replay and host guards pass.
Its conservative200k projection42m35s passes45min. Select one fresh200k learner
to measure the combination's frozen gameplay benefit.

The primary control is retained H5/R32/encoder6e-6 on seed4013: greedy14,050,
sampled13,331.250,23/24 first pyramids in both modes,training40m24s. Its own
initials80.208/177.083 clear none. Change only encoder6e-6→2e-5 as a learner
setting. Keep the exact checked12M48/Auto/N8/B16/T16/H5/R32/expert100 recipe,
world microbatch16,backprop16,base/actor4e-5,dynamics4e-4,warmup1000,
coefficient1,actor-unimix0,full18 actions,replay100k,all action weights1,world
imitation gradients,20% episode continuation/cap1920 and all-state corrective
labels. Preserve native max-pooled frames and one GPU resize to48.

Native447fe50,Qbert teacher and qualified optional-rate runner remain fixed.
The runner also contains Breakout corrective-label plumbing,disabled here;
the retained diff and bounded qualification do not establish complete
cross-version learning parity. Seed4013/evaluation4000004013 and the cohort
are reused development inputs. No weights,replay,games,belief,RNG or bounded
experience are reused. The cheaper H15/R16/encoder2e-5 result remains a separate
30m16s budget point at13,785.417 greedy/9,987.500 sampled. Relative to that
point,horizon,sampling,cadence,warmup duration and total work change together;
do not attribute a future difference to horizon alone.

Train exactly200,000 aggregate actions/24,954 updates under2700s (45min),
first update376,then every8 actions,residual credit0. Save and audit100k/12,454
recovery. Evaluate fixed final and own actual initial with sampled and greedy
actions,greedy primary and sampled posteriors. Each arm selects the first3
natural episodes per8 lanes=24,with guidance/labels off,zero updates and
truncations,357 invariant tensors and300s/600k-action caps. Keep the original
mean15,000/90%-first-pyramid gate. No checkpoint selection or extra evaluation
mode is part of this comparison.

Require357 actual-initial tensors matching control and bounded check,matched
pre-update training events and both actual-initial frozen semantic traces,
exact source/config/counters,finite reports,changed CNN/reward/actor/value,
equal labeled-row/weight sums,aligned raw expert losses,full training
pixel/action/label replay,frozen outcome replays/videos,guards,curves,coverage,
exposure and completed comparison. Ordinary frozen logs lack live pixel hashes;
their replay cannot compare unavailable live images. Guided returns and
changing replay losses are not unaided competence or held-out accuracy.

Caps:training2700s,each frozen arm300s,training replay1200s/outer1230s,CPU
audit1800s,main6000s,postprocessing6600s,terminal collection6900s. Recovery has
the fixed2700s observation window. An observation timeout never authorizes
restarting the learner. Stop on real faults/nonfinite reports;no driver recovery,
GPU retry,cap increase,extension,rate sweep,seed confirmation or automatic
successor. Preserve failures and repair CPU-only defects without repeating
completed GPU stages. The five-game goal remains unfinished.

Root `/x/Temp/kindle-qbert-encoder-h5-20261030/full`,trial
`qbert-12m-48-b16-h5-r32-encoder2e5-expert100-200k-seed4013`,service
`kindle-qbert-encoder-h5-200k-20261030`. The preceding bounded check is terminal
success; check live state before launch.

## CPU preparation passes

Final preflight passes in1.513s: five actual CLI paths,24 audit/comparison config
cases,49 resolved paths,15 compiled controllers,six header-readiness cases,four
observer-parser cases and five recovery-body cases. All288 bounded/229 control
pins,357 initial tensors and978/12,454/24,954 accounting pass,with zero native
or environment construction. The progress observation window is aligned to
6300s after the first passing1.550s preflight; both earlier artifacts remain
under`preparation-history`. Final preflight covers the adjusted sources before
selection. No failure or GPU repetition occurs during preparation.

## Fresh learner launched and startup verified

The fresh full comparison starts02:43:36 UTC October30/source254c1bd,main
PID378851/invocation`f89e34b891a84b959ff05cd156fc4697`. At02:44:08 startup
verifies all325 declared pins,357 initial tensors matching control and bounded
check,47 pre-update transitions and287 finite supervised reports through2664
actions. First six updates376/384/392/400/408/416 match the declared cadence.
Progress,100k recovery,postprocessing and terminal-collection observers are
active. No full frozen result is available. Do not restart this learner or
schedule another GPU job while it is active. The read-only.803s startup audit
co-hosts with learning;retain the timing caveat without wall-time correction.

Declaration`893fc0ce97c0284f703c59f68bdda362da5ccd86e4b17b02c268c80a750b72d1`;
startup`c7a04ee0d9c7626ecf0b0db178dfdb2c92995f292ebdbf220ce6fa4eb12b3c61`.
