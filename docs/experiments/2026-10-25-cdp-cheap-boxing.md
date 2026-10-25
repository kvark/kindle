# Cheap Boxing development screen

Test whether a smaller model and image can learn useful Boxing behavior within
the user's compute budget. Historical1M results support small capacity; the
native-frame visibility check retains both fighters at32x32. Neither establishes
learning on this backend. Start only after the current Freeway review and bounded
RGB32/RGB48 GPU checks pass; only one shared learner runs at a time.

Fresh seed1009, centered CDP/action-effects1, Size1M, RGB32, Auto,
N8/B8/T16/H15, full BPTT/microbatch8, replay100k/R16. Full18 actions, sticky.25,
repeat4, native max-pooled frames and exactly one GPU resize. No action aid.
Resolution is part of checkpoint geometry; frozen restore must use that geometry.
This changes several settings versus the historical12M reference and does not
isolate the effect of resolution.

Fixed200000 aggregate actions, expected24970 learner updates with first at248,
and a1800-second training cap including construction. Save the actual initial
and fixed final checkpoints. Evaluate at seed4000001009, selecting the first
three natural sampled matches per lane (24 per arm), zero updates and no action
aid. Each frozen arm has a600k-action/300-second cap. Keep every attempted run
and cutoff; no best-checkpoint selection or automatic extension.

Use the existing Boxing match gate: at least20 natural matches, mean at least50,
natural win fraction at least.9 and no truncations. A win alone is not this gate.
Verify finite reports/checkpoints, complete update and frame ledgers, exact
frozen tensor equality, and full CPU replay with stream0 videos. CPU replay/audit
cap1200 seconds; service cap3660 seconds. A promising single development seed
still needs replication before a reliability claim.

Artifacts: `/x/Temp/kindle-cheap-20261025/boxing-1m-32-200k`.
Controller: `run_boxing_200k.py` in that root. Service owner, once launched:
`kindle-cheap-boxing-200k-20261025`. The machine declaration records native/source
hashes and the passing smoke receipt before launch. The prepared native build
uses the existing Meganeura/Blade dependency pins; no backend optimization or
driver change is included.
