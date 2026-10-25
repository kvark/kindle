# Tiny's replay timer also waits for queued observation work

The [first compact Tiny run](2026-10-01-small-jepa-learning.md) spends 6.20ms/update
in world training versus RGB's 12.77ms, but its reported replay stage averages
13.70ms versus0.75ms. **Do not interpret that as isolated replay-copy latency.**

Split the completed seed1009 logs by update position within each observation
tick, excluding the first1,000 learner updates:

| Frontend | Update within tick | Samples | Mean replay timer ms | Median ms |
| --- | --- | ---: | ---: | ---: |
| RGB | First after observation | 24,469 | 0.794 | 0.790 |
| RGB | Later, no new observation | 24,470 | 0.708 | 0.697 |
| Pretrained Tiny | First after observation | 24,469 | 27.175 | 26.720 |
| Pretrained Tiny | Later, no new observation | 24,470 | 0.291 | 0.286 |

[Data, source paths and limitations](2026-10-01-jepa-timing-attribution.json).
This is CPU-only analysis of existing logs, not another gameplay run.

The code submits frozen perception without a feature readback
([vector ingestion](../../kindle/src/dreamer/agent/vector.rs),
[perception submission](../../kindle/src/vision/levjepa.rs)). Replay gathers
its regions in **one transfer and wait**, on the same queue; that wait also
completes earlier producers ([readback](../../kindle/src/dreamer/readback.rs)).

The pattern is consistent with queued perception/observation work being charged
to the first replay boundary. It does not identify each kernel's share or prove
that all27ms belongs to the encoder. Isolating that needs GPU timestamps or a
separate explicitly fenced diagnostic. Whole-run time remains valid; the
current whole agent is slower despite cheaper world training.

Do not optimize a presumed per-region wait: reads are already batched. Keep the
six-run learning comparison unchanged and investigate observation-side work
separately. October1's upstream check still finds Meganeura `7c29497`; no newer
main-branch fix was available. No runtime or learning setting changed here.
