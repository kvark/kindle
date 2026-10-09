# Qbert1009 reproduces its recorded500k prefix across the backend refresh

The fresh2M-action learner's first500k actions match the retained500k run's
recorded learning trajectory exactly after excluding timing fields. Both
start from the same346 saved tensors and reach an online last50 mean of1,809.
For this seed and prefix, the backend refresh did not change the recorded
early learning behavior. Continue the fixed budget study; this is not a final
policy result, three-seed stability evidence or completion of the five-game goal.

[Active allocation](../experiments/2026-10-09-cdp-qbert-budget.md) ·
[Compact evidence](2026-10-09-cdp-qbert-prefix.json) ·
[Raw comparison](../../runs/cdp-qbert-2m-20261009.992tJMEX/prefix-500k-1009.json) ·
[CPU comparison source](../../runs/cdp-qbert-2m-20261009.992tJMEX/prefix-500k-1009.py).
Review completes October9 at21:30 UTC; the learner remains running unchanged.

## What matches

| Evidence through500k actions | Compared | Different |
| --- | ---: | ---: |
| Saved initial tensors, exact bytes |346 |0 |
| Batched action/reward/terminal/cutoff/executed-frame-count records |62,500 |0 |
| Non-timing learner reports |124,939 |0 |
| Completed episode records, excluding elapsed time |671 |0 |
| Common512-action progress reports, excluding timing |976 |0 |

Every record key is present on both sides. The learner reports compare their
world/behavior metrics, update counters and replay lengths without a numerical
tolerance. Exact configuration, environment seeds, actor protocol, runner/wrapper
sources and eight host workers also match. Initial tensor comparison ignores
nondeterministic safetensors header ordering but checks each tensor's shape,
dtype and bytes. Finite reported metrics and initial tensors pass.

The old final partial progress report at500,000 has no corresponding scheduled
live report; the common progress grid ends at499,712. All transitions, learner
reports and episode outcomes through500,000 are still compared. Both online
means are computed from the same final50 completed episodes, not substituted
from a nearby progress bin. The old complete log hash matches its retained
training audit; the growing live file is identified by the byte length/hash
of its immutable prefix, not a racing whole-file hash.

## Limits and cost

The retained run uses native83be73bf/Meganeuraf104f354/Bladee349cddf; the fresh
run uses7311547d/31026833/56f0565. Agreement here does not make those binaries
identical, prove universal numerical parity, isolate every possible source
of stochastic variation, or retroactively repin historical evidence.

Training transition records contain executed frame counts, not pixel hashes.
This check therefore makes no independent pixel-equality claim. There is no
live500k checkpoint to compare final model/optimizer tensors; only the actual
initial tensors are compared. Later learning, frozen competence, the other
two seeds and12M prior forecasts remain untested by this check.

CPU reanalysis takes6.619s wall/6.572s CPU,403.4MiB peak and zero swap under
one-CPU/2GiB limits. It makes zero GPU calls, game actions or actor updates and
does not change the learner, source pins, allocation or numerical gates. A
separate3.835s CPU rehearsal also runs the unchanged end-of-seed reviewer on
all three retained Qbert pairs:144 natural episodes, nine historical guards,
all scores/first-pyramid outcomes reproduced. Its17 recounted allocation
warnings belong to those historical runs, not the active learner.
The [rehearsal receipt](../../runs/cdp-qbert-2m-20261009.992tJMEX/review-rehearsal.hPjEMhDi/result.json)
remains with the raw artifacts. Neither reanalysis collects new experience.
