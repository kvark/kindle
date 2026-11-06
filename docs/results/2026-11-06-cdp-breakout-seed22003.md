# One fresh-seed confirmation of the retained Breakout recipe

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-seed22003.md) selects
one fresh200k learner at seed22003 and a new shared evaluation cohort4000022003.
The strongest retained recipe was selected on learner1009 and repeatedly
assessed on its original cohort. Its400k extension uses an identical training
prefix and provides no independent learner confirmation. This test changes
only the learner seed, with all consequent training environment, sampling and
guidance randomness disclosed. It introduces no new recipe.

Retained native447fe50/SHA35ad0db, runnercf4ea238 and the exact original runtime
and teacher are reused. Actual-initial controls and four frozen arms for each
of the new and retained policies separate evaluation-cohort movement from the
combined new training-seed effect. The original864/90% two-wall gate remains;
positive learning or a new best development seed alone cannot complete the goal.
One fresh learner is now active as recorded below. No second learner, extension
or automatic successor is selected.

Preparation initially fails copying an archived replay helper from a stale path.
`preparation-failure.json` and `failed-preparation-source.py.txt` retain the error
and exact source. `continue_preparation.py` resumes at that copy using the existing
SHAf40da642 helper under `/x/Temp/kindle-breakout-cdp1000-20261106/full`.
Earlier copied controller helpers and runner are verified without recopying;
no native/emulator construction or completed experiment stage repeats. This is
a separate preparation repair, not a relabeling of the failed original script.

Root `/x/Temp/kindle-breakout-seed22003-20261106/full` retains scripts, receipts,
queues and subsequent evidence. Logs, weights and videos remain outside Git.

## Launch and startup

All nine actual queued command paths pass in1.745132s (wrapper1.817601s),
including native/emulator construction guards, actual environment seed lists,
six corrupt-header cases, three wrong-training-command cases, two incorrect
weighted losses and the complete comparison entry with synthetic outcomes.
There are90 active pins plus338 verified retained input pins. Initial/final
retained schemas are checked without native construction.

Execution sourcedcc4dd5d9a89a44cfecca9deaff1643e128c2d0c launches20:36:28.653 UTC
November6, controller1546932/invocation316ee06c85004cb68a061364822fdc3b. Startup
passes at20:36:34.266:357 finite initial tensors,47 pre-update transitions and
first update376. The new seed changes2,400 encoder first-layer weights and
655,360 actor first-layer weights relative to1009; exact initial equality is
not claimed. Config and header differ only in declared seed fields and metadata.
Training and independent recovery/terminal observers are active. No frozen
result exists yet; the original task gate is unchanged.
