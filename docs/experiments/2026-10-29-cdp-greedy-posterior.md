# Frozen evaluation with categorical posterior modes

The retained Qbert12M48/B16/R16/expert100 learner scores11984.375 with greedy
actions, versus6759.375 sampled, and still misses mean15000. Greedy actions do
not make its live RSSM deterministic: `VectorCore::ingest_prepared` draws from
each stream's posterior RNG, and `ActingGpu::posterior` selects sampled categories.
Test whether removing that additional inference randomness improves the retained
policy before allocating more training. Benefit is unmeasured; modes can also
hurt a model trained with stochastic states.

This draft was parked while the H3 loss and representation diagnostics ran.
It is now selected again after those completed comparisons. Preserve the old
patch, source copies and logs; use a fresh `resumed` directory for this work.
The H3 learner remains rejected, and the retained control is the source below.

## Selected implementation and bounded check

Expose `--greedy-posterior` only for frozen restore. Reuse the existing GPU
categorical argmax path independently of action sampling. Select it before the
first observation, consume no posterior RNG, and reject subsequent learning.
Do not change checkpoint configuration, weights, schemas, training defaults or
the default sampled RNG sequence. Record the option in run headers and reject
unknown or training uses in the auditor. Restores default to sampled posteriors;
the evaluation option is deliberately not checkpointed.

Use an isolated build/runtime under
`/x/Temp/kindle-greedy-posterior-20261029/resumed`; retain native447fe50 and its completed
experiments. Run the relevant Python CLI/auditor tests without native construction,
the Rust CPU suite and strict Clippy. A guarded GPU test must compare each selected
latent category with the actual posterior logits, cover episode resets, verify
posterior RNG preservation and reject both learning entry points. Also reject
late mode selection after terminal observations, before episode reset. Stream
`active` remains true after its first observation, including at terminal states.
Run the existing
vector/serial belief parity test to verify the unchanged default path. Limit each
native test to120s; stop on numerical or real host faults, without driver recovery.

Then perform two fixed1024-action frozen restores of the retained Qbert12M final
checkpoint, greedy actions in both: default sampled posteriors and the new mode.
Keep Auto/N8, native frames/one GPU48 resize, full18, sticky.25/repeat4,
seed4000004013 and all checkpoint settings. Allow120s per arm. Require zero updates,
all357 finite and invariant tensors, exact counters/source identity, full recorded
action/outcome replay and guards. The default arm's semantic prefix must match
the retained greedy assessment through1024 actions. The mode arm need not match
that trajectory. No task competence is inferred from a bounded action count.

The completed source learner is
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
Its original200k/12,477 training cost remains30m27s. These checks create no learner
experience for reuse and cannot establish a seed-independent result. Pin the
exact native, runner, checkpoint and audit inputs before each GPU service starts.

## Possible full assessment, not yet allocated

Only after the bounded checks pass, separately select one four-arm frozen Qbert
assessment: final and own actual initial, each with sampled or modal posteriors,
greedy actions throughout. Keep first3 natural episodes per8 lanes=24,600k-action
and300s caps per arm, guidance/labels off, zero updates/truncations and357 invariant
tensors. The new default arms must reproduce the retained default semantic traces
and scores. Report the original mean15000/90%-first-pyramid gate without changing
the cohort or selecting intermediate checkpoints.

This document selects implementation and bounded validation only. No training,
automatic full assessment, other-game allocation, extension or seed campaign is
selected. The five-game goal remains unfinished.
