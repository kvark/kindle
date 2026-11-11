# Pixel-world Breakout agent with the context-frame decoder

The standalone [4M context-frame world](../results/2026-11-10-breakout-world-context-frame.md)
is the best Breakout world model so far at every horizon and the first with
sharp bricks, but the whole-agent pixel world used the state-only decoder,
whose bricks blur (balanced error .32). This run ports the context frame to
the agent: the decoder also sees one observed frame from eight arrivals
back, held fixed through the prior image rollout, so the static scene comes
from the frame and the ball and paddle from the state.

## Code

`rgb_spatial_decoder` with `rgb_context_lag` is now allowed in whole agents
(the autoregressive image history remains standalone only); the agent's
staged graph is skipped for spatial decoders, the diagnostic observation
render feeds an empty context, and restore accepts the pair. Runner flag
`--world-context-lag N` (fresh learned-RGB pixel world without CDP). Tests:
config and graph (CPU) and a GPU tiny-agent update with the context frame.

## Recipe

The [matched pixel world 64x64](2026-11-10-breakout-agent-world96-matched.md)
200k command (seed 1009; retained recipe minus CDP, motion cross-entropy,
8-step prior image rollout, bonus kept, unimix .01, entropy 3e-4) plus
`--world-context-lag 8`. Differences from the standalone 4M context run that
produced the world-model result: 12M at 64x64 and B16/T16 (the retained
agent geometry), motion cross-entropy instead of `motion_event_rollout`, and
an 8-step rollout. Root `/x/Temp/kindle-breakout-pixel-context-20261111`,
`queue.json`, unit `kindle-breakout-pixel-context-20261111.service`. The
768-action smoke (`smoke2/`; `smoke/` is an earlier attempt stopped by a
vector-agent guard that still forbade the spatial decoder, relaxed in the
same commit) reports the prior-rollout loss with `rgb_spatial_decoder` and
lag 8 in the recorded config, 170 ms per update against 143 without the
context frame, VRAM 4.1 GiB in the smoke. Launched 07:50 UTC.

## Evaluation

Frozen evaluations of the 200k checkpoint on cohorts 4000001009 and
4000022003, sampled and greedy, three natural episodes per environment.
References at 200k (greedy / sampled): pixel world without the context
frame 156.840 / 37.680 and 151.080 / 40.286; retained CDP 143.667 / 70.750
and 94.625 / 70.500.

## Reading

One seed. A gain in either mode over the state-only pixel world would say
the sharper world helps the actor; parity would say the actor already gets
what it needs from the blurred decoder's state; a loss would point at the
fixed context frame confusing imagination (the rollout holds a stale frame
for eight steps). Report updates, wall time, VRAM and two-wall counts.
