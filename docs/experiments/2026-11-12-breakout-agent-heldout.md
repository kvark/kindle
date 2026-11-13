# Held-out evaluation cohort for the retained and pixel-world Breakout agents

Every agent score so far sits on cohorts 4000001009 (the retained learner's
selection cohort) or 4000022003 (reused across all of this week's runs).
This protocol evaluates four existing checkpoints once on a fresh cohort,
seed 4000777001, never used before, three natural episodes per environment,
sampled and greedy, zero updates, no selection:

- retained CDP seed 1009 at 200k
  (`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/final`)
  and at 400k (`/x/Temp/kindle-breakout-budget400k-20261103/full/breakout-12m-64-b16-r16-replayvalue1-expert100-400k-seed1009/final`);
- pixel world 64x64 seed 1009 at 200k
  (`/x/Temp/kindle-breakout-world96-matched-20261110/w64/final`) and at 400k
  (`/x/Temp/kindle-breakout-pixel64-400k-20261110/candidate/final`).

Root `/x/Temp/kindle-breakout-agent-heldout-20261112`, `queue.json`, outputs
`out/`. One cohort of 24 natural episodes per evaluation; read as a check
on cohort reuse, not as qualification against the 864/90% gate.
