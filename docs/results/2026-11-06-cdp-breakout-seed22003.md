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
No GPU allocation has started. No second learner, extension or automatic
successor is selected.

Preparation initially fails copying an archived replay helper from a stale path.
`preparation-failure.json` and `failed-preparation-source.py.txt` retain the error
and exact source. `continue_preparation.py` resumes at that copy using the existing
SHAf40da642 helper under `/x/Temp/kindle-breakout-cdp1000-20261106/full`.
Earlier copied controller helpers and runner are verified without recopying;
no native/emulator construction or completed experiment stage repeats. This is
a separate preparation repair, not a relabeling of the failed original script.

Root `/x/Temp/kindle-breakout-seed22003-20261106/full` retains scripts, receipts,
queues and subsequent evidence. Logs, weights and videos remain outside Git.
