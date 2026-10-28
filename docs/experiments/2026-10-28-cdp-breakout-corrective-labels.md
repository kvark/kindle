# Optional Breakout labels on learner-visited states

The existing Breakout adapter labels only overridden actions. Stage an opt-in
`--breakout-corrective-labels` path that recommends an expert action for every
visited state while keeping capped episode selection solely for actual overrides.
The teacher receives the executed full18 action and uses its horizontal component
in the calibrated paddle-response model. It preserves observed ball/paddle history.
Feedback is the command sent to ALE; sticky execution is observed only in pixels.
The optional path also uses the measured stationary-edge correction; labels,
feedback and teacher correction change together. No isolated attribution follows.

The default teacher and behavior remain unchanged. The new flag requires fresh
Breakout/full18/native-pixel guidance with an explicit cap and expert imitation.
Guidance remains forbidden in frozen evaluation. Native expert-label/world-gradient
support already exists in447fe50; no native change or build is needed. Training
still uses one GPU resize from native max-pooled frames.

All staging is external in`/x/Temp/kindle-breakout-corrective-20261028`. The Qbert
expert100/200k/R16 allocation remains the only GPU learner. Do not modify its
declared inputs while it or its audits are active. This CPU work selects no
Breakout native check or full learner. Before any later application to the working
tree, preserve an exact snapshot of affected historical source paths and hashes;
retain original declarations, and use the matching runtime/source for old audits.

## Completed CPU checks

The staged runtime, CLI, vector accounting and replay changes pass510 CPU tests,
10 native-construction tests explicitly excluded, with zero construction attempts.
Tests cover full18 horizontal-command feedback, invalid-call atomicity, released
and unselected labels, game-specific label vocabulary, forged ledgers, CLI rejection,
and distinct recommendations/executed actions through runner, resets and replay.
The guarded affined test wrapper takes1.734s; all138 active Qbert inputs and the
same PID203447/invocation`e042da0cefc4457f826202e4d6a37661` remain unchanged.

The staged feedback teacher then reproduces all52,072 actions/208,254 frames of
the retained edge-reset reference, including policy debug values,input/output
pixel hashes,rewards,boundaries,resets and full outcomes. The24 selected natural
episodes retain225.083/zero two-wall successes; this is reused CPU reference
parity, not passive-label accuracy,native execution or Kindle learning.

CPU parity runs21:09:03–21:09:34 UTC on October28,30.982s measured analysis and
31.084s whole service. All13 parity inputs and138 active Qbert inputs remain
unchanged, with the same learner process. No native module is imported and no
agent is constructed. Preserve both co-host durations without adjusting Qbert
training time. No failed attempt or GPU work occurs in this preparation.

Test receipt`575182f7a2219fdf7ff944b4a9bd2597005645dc04da6be9bc4408e3f552fd1d`;
parity declaration`e11246ad090f8e236007255c34565a6567d1bc3966530a82d4f6906c90ac9bde`;
parity review`a757019d30edae74c219f1d6df32eeb3ee048c24c77ec4224bed282231d024f1`.
`cpu-completion-review.json` records terminal success and the staging-only scope.

An exact112-file snapshot of historical declared repository inputs and Python
sources is retained under`previous-source/`, with receipt
`0362a128bc6c0008a968f6f9e8e39ae50882f9505234f0894db2a27aed53b62d`.
The original declarations are not rewritten. The prepared application script
requires terminal success of every Qbert service,its complete receipt and every
current pin before copying the exact tested patch. It then verifies historical
hashes using explicit archived paths where working-tree sources change. The
script has not been executed at this stage; native validation remains unselected.
