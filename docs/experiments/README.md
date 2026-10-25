# Results and experiment archive

Start with the [current roadmap](../kindle_single_life_dreamer_plan.md) and
[machine handover](../handover.md). [PR31](https://github.com/kvark/kindle/pull/31)
is the live status dashboard.

Current experiment: [Breakout B2/T64](2026-10-23-cdp-breakout-context64.md).
Its smoke trained successfully, but ordinary frozen restore hit the original
host's [PMU fault](../results/2026-10-23-cdp-context64-gpu-fault.md).

Recent learning evidence:

- [Pong reliability](../results/2026-10-21-cdp-pong-budget.md)
- [Freeway replication and videos](../results/2026-10-23-cdp-freeway-replication.md)
- [Breakout budget](../results/2026-10-22-cdp-breakout-budget.md)
- [Breakout warm-start result](../results/2026-10-23-cdp-breakout-warmstart.md)
- [Breakout state diagnosis](../results/2026-10-23-cdp-breakout-state.md)
- [Qbert budget](../results/2026-10-11-cdp-qbert-budget.md)
- [Artifact cleanup](../results/2026-10-23-artifact-cleanup.md)

The dated files in this directory are historical declarations; `../results/`
contains their reports and compact data. Keep failed and superseded results,
but do not treat their old next steps, host pins, timing estimates or unstarted
queues as active instructions. Backend optimization and upstream porting are
parked. The former chronological index is preserved in Git at8d464e0.
