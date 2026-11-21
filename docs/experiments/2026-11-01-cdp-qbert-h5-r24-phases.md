# Locate the Qbert H5/R24 score difference in retained frozen episodes

Compare completed seed8039 and10007 H5/R24 learners using their retained final
sampled and greedy logs and verified replays. Each already has24/24 first pyramids;
8039 greedy scores15,430.208 and10007 greedy14,339.583. These are different
learner/evaluation seeds, not paired interventions. The second learner misses
the unchanged15k mean gate. No new learner or evaluation is selected.

For the same first three natural episodes per eight lanes used by each completed
assessment, split logged rewards/actions into transitions entirely before the
first-pyramid milestone, the transition containing that milestone, and subsequent
transitions. Raw milestone frames come from the existing task observer. Four-frame
pooled transitions do not resolve reward timing within the crossing action;
keep that action separate and report the prefix including it. Reconcile every
selected episode's score, action count, absolute/relative frame counts and retained
selection. Use all96 selected episodes; do not drop short or weak outcomes.

Report first-milestone timing, prefix reward, later reward/duration, aggregate
later reward per100 actions, the episode table and per-lane means. A count of
individual scores at least15k is descriptive; the original mean/first-pyramid
gate stays unchanged. Do not fit a predictor, optimize a cutoff, infer verified
later game levels, or attribute causality to initialization, survival or policy.
This is an exploratory decomposition after score/coverage results are known;
some10007 greedy coverage rows were inspected before this protocol.

Use CPU JSON processing only, with fixed boundary arithmetic fixtures and input
hashes. No native agent, emulator, experience collection or update is constructed.
Run once in `/x/Temp/kindle-qbert-h5-r24-phases-20261101`, retaining output rows,
summary, wall time and the identity of the active Breakout centroid learner before
and after. Keep co-host time without correction. Do not modify its pinned scripts
or allocate another GPU job while it is active. No new recipe, confirmation or
automatic successor follows from this descriptive check alone.
