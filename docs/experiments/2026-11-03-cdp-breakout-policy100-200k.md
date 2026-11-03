# Fresh200k Breakout comparison with imagined-policy weight100

Select one full comparison after the [bounded check](2026-11-03-cdp-breakout-policy100.md)
passes8,192 actions/489 updates in46.910s and a20m59.778s conservative projection.
All396 pins,357 matching/invariant tensors,47 pre-update events,full guided replay,
policy100 report arithmetic and host guards pass. Both original services and
the independent terminal reader complete. No bounded experience transfers.

Only `loss_scales.policy`1→100 changes from the retained replay-value1 control.
This scales score-function and entropy together; expert100,world imitation100,
imagined-value1,replay-value1 and actor/critic representation gradients off stay
fixed. It tests greater imagined-policy pressure relative to imitation without
removing representation supervision. It does not establish current-gradient
dominance, isolate entropy from advantage learning or fix a demonstrated defect.

Keep CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/discount333,
base-encoder4e-5/behavior8e-5/dynamics4e-4, expert unit weights/all-state corrective
labels, sampled collection/posteriors, actor-unimix.01/entropy.0003,action-effects1,
replay100k,warmup1000,20% episode guidance/cap1920,seed1009/eval4000001009,
full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize/raw rewards.
Native447fe50/SHA35ad0db and teacher stay unchanged; use bounded runner5a86e224.

Fresh training starts with no reused weights,replay,RNG,belief,games or experience.
Stop at200,000 actions/12,477 updates or the fixed1800s training cap. Preserve
actual initial weights and100k/6,227 recovery with four retained files and357
finite tensors. First update376,residual.5 and all12,477 finite reports must pass.
Verify behavior total as `100*policy+value+replay_value+100*expert` using the same
qualified finite-error bound. Track actual overrides separately from policy skill.

Fixed frozen arms: final/actual-initial, each sampled and greedy; first three
natural episodes in each of eight lanes=24 per arm. Guidance and labels off,
zero updates and truncations,357 invariant tensors. Keep300s per arm; no retry,
episode-cap increase, early checkpoint selection or post-result mode selection.
Require original matched initial values and800/795 sampled/greedy initial semantic
events. Record all excess episodes and partial tails while selecting only the
declared cohort. Sampled remains primary; report both modes and the unchanged
90% two-wall gate against retained70.750 sampled/143.667 greedy,all0/24 successes.

Require full training pixel/action/label replay, all frozen action/outcome/frame/
boundary replays and videos,guards,curves,exposure and learning-per-minute comparison.
Ordinary frozen logs lack live pixel hashes; do not claim unavailable live-image
equality. The comparison must preserve exact configs except policy weight and
report coupled score-function/entropy scaling, reused seed/cohort and CPU co-host
timing without corrections. No causal attribution or broad reliability follows.

Qualify all five commands, audit/config assertions, observer wiring, paths,
checkpoint schema and retention on CPU before clean committed selection. Keep
one original main,progress,recovery,postprocessing and terminal-collection service;
no overlapping GPU work. Main cap4500s,audit1200s and the retained observer deadlines
remain fixed. Preserve failures and repair CPU-only stages separately if needed;
never repeat successful GPU stages. Inspect original identities and completion
once, then independently summarize recovery,terminal results and descriptive score
coverage from new logs. Reuse the retained control coverage without rerunning it.

No extension,coefficient sweep,confirmation learner or automatic successor is
selected. Root `/x/Temp/kindle-breakout-policy100-20261103/full`. The five-game
goal remains unfinished regardless of a passing execution audit.

## Launch and startup

The full comparison launches **09:34:23.440 UTC November3**, source
`0c40e657b3f9d9c7e79038ca0c81f8f1a011c852`, with436 pins. MainPID1000011/
invocation`dbf3484b87e947cfa048d424a144d958`; service
`kindle-breakout-policy100-200k-20261103`. Original postprocessing/progress/
recovery/collection services start once. Startup passes **09:34:53.761**
(1.505s wrapper):357 bounded/control-matching initial tensors,47 pre-update
transitions,first updates376/392/408/424/440/456,and250 finite reports through
4,360 actions. No full frozen result is available. No restart,overlap or extension.

DeclarationSHA`377c240b0aae298a1ff87300019d415a7deabce7aca3c65bf419c7127a685ebf`.
The100k recovery,all four assessments and original terminal collection remain
pending. No full preparation failure or repeated GPU stage occurs.

The independent terminal reader qualifies five score checks at09:35:51.569,
3.545s body/3.572s wrapper. Coverage qualification first fails in2.440s because
the copied reader pins a separate prepare_coverage.py, while this package uses
one combined prepare_readers.py. Preserve that source/failure and the.019s
CPU repair. The corrected reader pins the actual generator,original source and
failure;52 diagnostic/436 active pins,seven config cases and five retained
boundary fixtures pass09:36:34.672 in2.439s (2.460s wrapper). Original process
identity stays unchanged. No terminal analysis,emulator or GPU stage repeats.

Keep these CPU co-host timings without correcting learner wall time. Run the
recovery confirmation only after the original100k observer succeeds while the
learner remains active. Run terminal and coverage summaries once after their
required original completion artifacts exist. Their private sources/provenance
are fixed before frozen outcomes.
