# Rank separation after the terminal replay mixture

The matched arrival diagnostic finds lower error on actual game endings but more
false ending indications at intermediate life losses. Select one CPU-only ranking
analysis to test whether this is accompanied by weaker class separation. Use only
the existing candidate/control continuation outputs on the same120 endpoints:
24 game endings and96 intermediate life losses, three/twelve per lane. No model
execution, emulator replay, fit, checkpoint change or new learner occurs.

Primary is candidate-minus-control posterior ROC AUC; the one-step prior is
secondary. Lower continuation ranks a state as more likely to end the game. Compute
AUC from all24×96 terminal/continuing pairs, assigning a half win to exact ties.
Use a paired bootstrap of eight whole lanes,5,000 resamples, seed20261104. Recompute
the pooled AUC with lane multiplicities; do not average within-lane AUCs or treat
all2,304 pairs as independent observations.

Also record the3×3 transition table of control/candidate pair orders, exact number
of changed pair orders, and the previously fixed below-.5 counts for each class.
Do not choose a new threshold or transform, fit a calibration offset, or promote
the weaker gameplay candidate. A strict monotone shift preserves pair ordering;
changed orders rule out that transformation as a complete description of the
observed outputs, without identifying the underlying mechanism. An interval
spanning zero does not establish equal discrimination or a pure calibration shift.

Verify the completed candidate-arrival declaration,1,052 pins and52 final artifacts,
including the control outputs. Join every endpoint/outcome exactly and reject
duplicates, missing lanes, wrong class counts and nonfinite/unbounded scores.
Before analysis, test perfect/reversed/tied scores, monotone invariance, changed
orders and agreement of the lane-weight calculation with explicit resampling.
Block native-model/emulator imports and check no GPU descriptor is opened.

Run once on one CPU core at nice10 under60s. Independently recount every pair,
the pooled AUCs, lane bootstrap and original threshold counts from saved inputs;
verify immutable input hashes and final artifacts without rerunning the analysis.
Preserve failures and do not repeat completed stages. Record source, exact inputs,
process, timing and all results outside Git.

This is an outcome-conditioned, reused development cohort, with different native
versions and learned states. Ranking is not population calibration, an acting
threshold or evidence of a causal gameplay limitation. Retain the existing sampler,
continuation1 and70.750 sampled/143.667 greedy, all0/24 two-wall. No loss, correction,
mixture setting, learner or automatic successor is selected by this analysis.
Root `/x/Temp/kindle-terminal-ranking-20261104`; the original864/90%-two-wall
and five-game goal remain unfinished.
