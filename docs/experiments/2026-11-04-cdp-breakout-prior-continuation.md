# One-step prior continuation supervision: bounded qualification

Select a default-off auxiliary native continuation loss on sampled one-step prior
states. The retained control's ending information is more accessible after seeing
the arrival image; terminal weighting changes probability levels and lowers prior
ranking on the fixed cohort. Neither result proves a causal gameplay limitation.
The earlier three-step joint feature/reward/continuation objective regressed and
is not repeated. No full learner is selected by this bounded protocol.

Use the existing transition's deterministic state and prior logits, before the
arrival posterior consumes its image. Draw a separate categorical prior sample
with straight-through gradients. Apply the shared continuation head to that prior
feature and supervise the actual aligned target:0 at true game ends, gamma
otherwise. Mask reset observations with the existing keep-action mask. Average
over all B*T rows, preserving zeros for masked rows. Report raw auxiliary loss
and valid fraction separately. Add scale1 times this loss to the world objective;
there is no class weighting, terminal replay mixture, future-feature target,
reward auxiliary target, correction or teacher change.

Keep gradients through prior probabilities, dynamics and earlier posterior states;
the arrival image/posterior must not enter its own prior loss. Use a separate RNG
stream, leaving disabled behavior and the ordinary posterior stream unchanged.
No new parameters are added. Omitted option preserves the old graph and serialized
configuration. Reject incompatible open-loop/terminal-weight options and invalid
scales before environment or GPU creation. Checkpoint restore must detect missing
native support.

Before GPU work, test serialization/validation, graph compilation and disabled
compatibility, causal graph dependencies and gradient reachability, reset masking,
microbatch scaling and CLI capability checks. Require strict Clippy and packaged
CPU checks. Use a guarded F64 numerical fixture for masked BCE loss/logit gradients
at1,17,256,513 rows, including extreme logits and masked terminals. Separately
check a tiny complete graph sends finite nonzero gradients into the prior and
continuation head without a gradient path through the same-step observation.
Preserve failures; do not change tolerances to pass a failed fixture.

Then one fresh8,192-action Breakout check with200s training cap and matching
1,024-action frozen restore:12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,
posterior continuation1,prior continuation1,expert100/unit weights/world gradients,
CDP future500/disagreement1. Retain full18 actions,sticky.25,repeat4,noops0,
native pooled RGB/one GPU64 resize,sampled posteriors/collection,unimix.01,
replay100k,20% episode guidance/cap1920,corrective labels and seed1009.

Require357 initial tensors matching retained/bounded control,47 pre-update semantic
events,first update376,489 finite reports,changed CNN/reward/continuation/actor/value,
full recorded pixel/action/label replay,zero frozen updates and357 invariant frozen
tensors,host guards and auxiliary-loss/total arithmetic. Verify masked valid fractions
against replay boundaries. Check disabled frozen compatibility on the retained
checkpoint. Conservative200k projection is25 times bounded time times1.10; require
less than30 minutes before considering a separately declared full learner.

Root `/x/Temp/kindle-prior-continuation-20261104`. One GPU job at a time; check live
state, preserve fault stops, never restart solely after observer timeout. Retain
143.667 greedy/70.750 sampled,all0/24 two-wall until gameplay evidence changes that
decision. Original864/90%-two-wall and five-game goal remain unfinished.
