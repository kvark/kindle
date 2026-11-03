# Eight-episode CPU comparison of consistent contact timing

The [saved contact screen](../results/2026-11-03-cdp-breakout-paddle-contact-timing.md)
passes with small average gains and some worse tails. Select one private CPU
teacher candidate applying the exact conversion when choosing a plan. No new
native learner, loss, encoder or training allocation is selected.

Start from the retained edge-reset PixelPolicy. Only for its existing
`pooled_contact_steps <= 5` branch, convert all243 five-action predicted paths:
`q[0] = p[0] + observed_displacement/8`,
`q[i] = p[i] + (p[i]-p[i-1])/8`, and interpolate q at
`max(0, pooled_contact_steps-.125)`. Minimize squared error to the original
landing target plus the existing.02 movement cost. Preserve detector, response
weights, clipping, cached landing, edge reset, fallback and far-horizon cost.
Record original and corrected paths and contact times. The runtime teacher stays
unchanged; this is a private reference implementation.

Use the retained eight-lane seed4000001009 protocol: full18 actions, sticky.25,
repeat4, noops0, native pooled RGB, natural episodes with100,000 raw-frame cap.
Select the first natural episode per lane, requiring episode0 and no truncation.
Reuse the original edge-reset control without replaying or collecting it again:
returns151/97/256/226/204/331/82/136, mean185.375, zero two-wall successes.

Collect at most200,000 actions/90s and cap the entire collection/replay process
at180s,2GiB, one low-priority CPU core. Preserve extra completed and partial
episodes generated while waiting for the eighth lane. Require a full exact
pixel/action/reward/termination replay, recomputed teacher debug and lane0 video.
For this CPU reference, gameplay success is still the existing two-wall task.

The predeclared screen requires at least10% higher mean return, at least five
improved lane returns, and no fewer two-wall successes. Passing permits a
separately fixed24-episode assessment; it does not automatically select that
assessment, teacher integration or training. Failure selects no extension,
coefficient search or successor. These reused eight episodes are a development
screen, not independent reliability evidence.

Before execution, require exact private policy diff, CPU fixtures for unchanged
branches, cancellation, time/segment boundaries and independent plan scoring,
clean committed selection, pinned source/protocol/control/ROM/runtime inputs,
and no active Kindle allocation. Preflight must construct no emulator or native
agent. Preserve all failures. Verify pins afterward and independently recount
the ledger, timing debug, eight returns and screen without repeating emulator
stages. Root `/x/Temp/kindle-breakout-contact-timing-policy-20261103`.
