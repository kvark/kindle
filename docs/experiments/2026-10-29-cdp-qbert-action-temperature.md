# Frozen Qbert evaluation with a sharper sampled action policy

The retained12M48/B16/R16/expert100 model scores11984.375 greedy and6759.375
sampled, below15000. The completed25M/R16 comparison improves sampled score
but lowers greedy score at greater training cost. Test one intermediate action
temperature,0.5, on the retained12M model before selecting another learner.
This is inference tuning on a reused development cohort; benefit is unmeasured.

## Selected implementation and bounded checks

Add `--action-temperature` for sampled frozen restores only. Native action
sampling uses `softmax(logits / temperature)` before the checkpoint's actor
uniform mixture and any existing action mask. Posterior sampling remains
unchanged. Require a positive normal finite F32 temperature, selected before
the first observation, and disable both native learning entry points after
selection. Reject greedy-action combinations in the CLI and log auditor.
Restores default to temperature1; omit the header field when unspecified.
No checkpoint configuration, weights, training default or schema changes.

Keep the legacy temperature1 shader arithmetic unchanged. Test known policy
logits, multiple temperatures and draws, masks, greedy selection, RNG behavior,
episode resets and evaluation-only guards. Run the Rust CPU suite, strict
native/binding Clippy and packaged Python CLI/audit tests without GPU construction.
Then run the temperature test and existing vector/serial parity test under the
host guard,120s each. Check producer completion before readback.

Build an isolated runtime under `/x/Temp/kindle-action-temperature-20261029`.
Preserve native447fe50 and every completed runtime. Record exact source, native,
wheel, tests and Python source hashes; release the disposable build cache after
packaging and validation. Native learning/inference remain on Meganeura+Blade.

After CPU checks pass, select two bounded1024-action frozen restores of the
retained final checkpoint: unspecified temperature and explicit0.5, sampled
actions/posteriors in both. Keep Auto/N8/RGB48/native frames/one GPU resize,
full18,sticky.25/repeat4,seed4000004013 and all checkpoint settings. Require357
finite/invariant tensors,zero updates,exact source/config/counters,complete
action/outcome replay and host guards. The default arm must reproduce the
retained sampled assessment through1024 actions.120s per arm;no competence
claim from this bounded action count. Pin inputs before GPU work, stop on real
faults or numerical failure, and perform no automatic driver recovery or retry.

Source learner:
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
Its200k/12,477-update training cost remains30m27s. No new training is selected.

## Possible full assessment, separately selected after the checks

Four fixed arms: retained final and actual initial, each with default sampled
actions or temperature0.5. Require the default final/initial to reproduce the
retained sampled trajectories and scores. Each arm selects the first3 natural
episodes per8 lanes=24,with600k-action/300s caps,guidance and labels off,zero
updates/truncations,357 invariant tensors and full replays/videos. Compare0.5
primarily with the retained greedy11984.375/24, and report default sampled
6759.375/24 separately. The original mean15000/90%-pyramid gate stays fixed.
Do not scan temperatures, choose checkpoints or switch the primary comparison
after seeing results. Reused seed/cohort are not independent confirmation.

This selects implementation and bounded validation, not the full assessment,
other games, a learner, an extension or an automatic successor. The five-game
goal remains unfinished. Check live state before every GPU allocation.
