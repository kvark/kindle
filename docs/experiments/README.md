# Current evidence and research archive

The [project plan](../kindle_single_life_dreamer_plan.md) is the only roadmap.
Logs, declarations, checkpoints and videos live in git-ignored `runs/`. Do not
overwrite completed/failed experiments or change their acceptance gates.

## Current Pong confirmation

[Declaration](../../runs/pong-block-confirmation-20260916.rBwdGF/declaration.md):
fresh roots 2017/3019/1009, qualified native `886bae68`, causal LeVJEPA, 400,008
training actions per root. Each root needs final-policy evaluation, fresh
initialization, restored untrained control and complete replay/video checks.
Every phase is individually invoked/reviewed; no automatic successor.

Root 2017 training completed September 17, independently re-audited September 19:
**400,008 actions / 99,652 updates**, complete state/moments and final checkpoint,
243 natural episodes, no cutoffs. All 200,269 native memory samples meet the
gate; minimum estimated headroom 7,695,106,048 bytes. The guard reports exit 0,
a reaped child, no fault and no NVML. Training takes 39,390.40s (10.94h),
10.155 actions/s, .677× aggregate real time. Stage totals: learning 25,823.95s,
observing 13,275.99s, emulator 176.39s. Online outcomes are not frozen competence.
Frozen evaluation is running; untrained control and other roots remain pending.

Evidence: [training result](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-training-result.json),
[guard](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-training/result.json),
[ledger](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-train.jsonl),
[live frozen output](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-evaluation/child.stdout).

From the original checkout, inspect live processes/results before acting:

```sh
python/.venv/bin/python -B runs/pong-block-confirmation-20260916.rBwdGF/run.py verify
python/.venv/bin/python -B runs/pong-block-confirmation-20260916.rBwdGF/run.py audit seed2017-training
```

Never repeat a `run` invocation. Audit completed phases, then individually invoke
the next. `audit.py pair SEED` is the one-shot CPU replay/video writer after all
four phases; phase audits are read-only. The legacy training initialization
reader is memory-heavy on its 3.3 GB stderr; avoid unnecessary repeat reads.

## Qualified runtime

The current backend passes 23 native tests and same-driver complete-state/pixel
checks. Block products improve fixed-recipe N6 throughput **27.3% in both orders**,
retaining all 241 tensors/146 moments, non-timing reports and action traces exactly.
Episode-budgeted evaluation passes six integration phases on the unchanged binary.

Keep the completed [block pixel/timing group](../../runs/current-block-pixels-20260916.M7whE0/),
[episode integration](../../runs/current-block-episode-runtime-20260916.CXHzRj/)
and [source adoption](../../runs/current-block-adoption-20260916.4Smhmw/).
Source cleanup does not qualify a new build or require another backend campaign.

## Archive, not deleted evidence

The complete old history is retained at
[`9ef9a16`](https://github.com/kvark/kindle/tree/9ef9a169d52ed36a912d578a89690f992c5982fc)
and pushed as `archive/dreamer-jepa-pre-cleanup-20260919`. Its
[64 reports](https://github.com/kvark/kindle/tree/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments)
contain the detailed measurements, failed readers, qualification and incident
chronology removed from this changeset. No raw experiment, historical worktree
or package is deleted or relabeled.

Useful entry points:

- [Kickoff, recovered baseline and ablations](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-05-kickoff.md).
- [Boxing three-root confirmation](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-10-boxing-confirmation.md).
- [Freeway failures and recovered Pong](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-11-recovered-confirmations.md).
- [Block correctness and timing](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-current-block-matmul.md).
- [Prepared Breakout hypothesis](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-breakout-current-runtime.md).
- [Prepared checkpoint-history/exposure tools](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-atari-dose-retention.md).

Unused exposure-study/midpoint-retention helpers and tests are removed from the
production tree. Their prepared branches and CPU evidence remain available;
bring back only what a declared experiment needs. Archiving changes no gate.
