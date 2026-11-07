# Fresh200k Breakout comparison: encoder4e-5 versus8e-5

After the [8e-5 bounded check](../results/2026-11-01-cdp-breakout-encoder8e5.md)
passes execution,restore,replay and a conservative20m55s cost projection,select
one fresh200k-action learner to measure gameplay. The retained4e-5 point scores
56.292 sampled/104.667 greedy in21m28s,up from25.292/42.625 at2e-5. All modes
still have0/24 two-wall successes. More precise state readouts and the earlier
control diagnostic motivate this one additional development point without
establishing a sole architectural bottleneck. The earlier4M/expert10 larger-rate
failure remains retained; no universal optimal rate or reliability is claimed.

Change only encoder4e-5→8e-5 in centered CDP/12M64/Auto/N8/B16/T16/H15/R16/
expert100. Keep base/actor4e-5,dynamics4e-4,warmup1000,actor-unimix.01,
action-effects1,full18 actions,sticky actions.25,action repeat4 and one GPU
resize from native frames to64. Keep20% episode guidance/cap1920,all-state
corrective labels,world imitation gradients and all action weights1. Native
447fe50,the exact retained runner and pixel teacher stay unchanged. No build
is needed. No control or bounded weights,replay,games,belief,RNG or experience
enter this fresh learner. Seeds1009/eval4000001009 remain reused development
inputs,not independent confirmation.

Train200k aggregate actions/12,477 updates under1800s. Require first update376,
residual credit.5 and100k/6227 recovery. Require357 finite initial tensors
matching control and bounded check,47 matching pre-update transitions,finite
reports,changed CNN/reward/actor/value and exact source/config/counter accounting.
Check expert row/weight totals and raw world/actor losses,complete training
input-pixel/action/teacher-label replay and all host guards. The recovery
observer now saves the exact audited100k metadata bytes and their hash before
the final checkpoint replaces them. This preserves metadata and tensor hashes,
not a second copy of the100k tensor files; it changes no learner behavior.

Assess final and actual-initial checkpoints with sampled and greedy actions.
Each arm has300s and600k-action caps and selects the first3 natural episodes
per8 lanes=24. Guidance/labels are off. Require zero updates/truncations,357
invariant tensors,paired initial semantic traces,complete recorded-action/
outcome/frame/reset replays and videos. Ordinary frozen logs do not contain
live pixel hashes; do not claim comparison with unavailable live images.
Preserve the90% two-wall gate. Sampled remains primary; report greedy separately,
both absolute means and own-initial-subtracted return per training minute.
Do not switch primary mode or select checkpoints after observing outcomes.

Main cap4500s includes the fixed stages and1200s CPU audit; postprocess5100s,
recovery3000s and terminal collection5400s. Progress/recovery observers only
inspect the original process and never restart GPU work. Retain curves,exposure,
paired comparison and original terminal states. No retry,extension,cap increase,
rate queue,seed confirmation or automatic successor follows this declaration.
Stop on real GPU faults/nonfinite reports; no driver recovery. CPU-only repairs
must preserve original failures and avoid repeating completed GPU/audit stages.

CPU preflight1.375s passes five actual CLI paths,26 audit/comparison config
cases,44 resolved paths,15 compiled controllers,nine header cases,four observer
parser cases,seven recovery-config cases,four exact-metadata preservation cases,
251 bounded/227 control pins and357 schema tensors. Previous encoder rates are
rejected by actual comparison,startup and recovery assertions. The metadata
helper rejects hash mismatches and existing output files. No native/environment
construction occurs; no preparation failure or GPU repeat occurs.

Root`/x/Temp/kindle-breakout-encoder8e5-20261101/full`,trial
`breakout-12m-64-b16-r16-encoder8e5-expert100-200k-seed1009`,service
`kindle-breakout-encoder8e5-200k-20261101`. Recheck live state,clean source and
boot/driver pins at declaration and launch. The five-game goal remains unfinished.

## Full comparison launch and startup

The fresh200k learner launches04:49:29 UTC November1/source3897e74,with265
pinned inputs,mainPID742893/invocationeccf5a76eed6458babd75b872c1704e4. All four
observers attach;the30-minute training cap,100k/6227 recovery and fixed final/
actual-initial sampled/greedy24-episode arms remain unchanged. Startup passes
357 initial tensors matching control and bounded check,47 pre-update transitions,
first updates376/392/408/424/440/456 and267 finite supervised reports through
4648 actions. Its1.007s CPU check co-hosts with training;retain timing caveats
without correcting wall time. No full frozen outcome is available.

Declaration`5009887d92a4e26c2ebe5836ef3734ec6027e888113beb61e1fc1729cb18d23e`.
Do not restart this learner or schedule another GPU job while it remains active;
inspect the original service and observers before further work.

The separate CPU terminal verifier is prepared at
`/x/Temp/kindle-breakout-encoder8e5-20261101/review_terminal.py`; its.053s
preflight recomputes the retained control's fixed episode scores/videos and
verifies all five original service invocations. A successful short-lived
collector need not have a systemd CPU-accounting line; require its exact
completion object under the original invocation instead. Both successful
preflight revisions are retained,with no failed check or GPU work. Run its
`final` mode only after the existing original pipeline and collector succeed.
Its co-host CPU time remains a caveat without training-time correction.

## Halfway recovery passes

The original observer passes05:00:35 UTC November1 at100k actions/6227 updates,
with357 finite tensors and6445 finite reports in the complete103,496-action
prefix. It saves exact100k metadata automatically and exits success. Independent
.835s CPU confirmation verifies metadata identity,all three current tensor-file
hashes,265 input pins and original mainPID/invocation;no GPU stage repeats.

- Recovery review`7c2b3f3d32c329ce6dd996bd398c71cbc69e7f9b0b700cb4f3ee492470f4c1dc`.
- Metadata`c9b354d3d69824724337420d26db9d0848b405eecfe3554c0977b82ff770dc67`.
- Confirmation`016bc65ab1e9756198ba5f2bc8d604278a5283f7eb034ce461b9d3cc55f34697`.

The separate CPU Qbert life/color diagnostic preparation co-hosts with this
learner:five pure tests.028s and preflight3.036s. Its300s bounded replay remains
CPU-only and preserves the original process and inputs. Retain timing caveats
without correcting training wall time;frozen Breakout gameplay is unmeasured.

The Qbert CPU replay completes 05:07:23 UTC on November1 in49.272s analysis /
50.619s controller wall,with all293 diagnostic inputs and this learner's265
inputs/original process identity unchanged. No native/GPU work or learner
update occurs in that diagnostic. Retain this CPU co-host timing caveat without
correcting training wall time;the original learner continues.

## GPU stages complete; original CPU closure pending

Training finishes200k actions/799,597 frames/12,477 updates in1284.126s
(21m24s). All four frozen arms finish,with provisional fixed24 means55.875
sampled/125.375 greedy and actual initials1.417/2.208. These are raw log
summaries;the original full replay,host/checkpoint audits,postprocessing and
collection must pass before accepting the comparison. The original main and
observers remain active;do not repeat any stage. The separately prepared Qbert
CPU intervention's eight tests take.035s and input preflight3.331s after these
GPU stages. Preserve timing context without training-time correction.

## Accepted completion — November 1

All original stages and five services succeed:main05:14:41,postprocessing05:15:03
and collection05:15:06 UTC. Separate.872s terminal confirmation passes05:15:35
without repeating any original stage. All265 pins,recovery,initial/frozen checks,
full replays/videos,guards,curves,exposure and comparison pass. Fixed sampled
55.875 is.74% below56.292;greedy125.375 is19.79% above104.667,withall0/24 two-wall.
Training21m24s is.28% shorter. Retain4e-5 as sampled point and8e-5 as a separate
greedy option;do not switch the primary mode or claim an overall win. See the
[completed result](../results/2026-11-01-cdp-breakout-encoder8e5.md). No failure,
GPU repeat,extension,rate queue,confirmation or automatic successor occurs.

The separate Qbert CPU intervention completes63.910s analysis/65.230s controller
after this main service is terminal,with309 diagnostic/265 protected inputs and
original invocation unchanged. Its brief overlap with CPU postprocessing and
terminal confirmation is retained without training-time correction.
