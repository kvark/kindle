# Fixed paddle centering while the Breakout reference loses the ball

Test one CPU pixel-reference change:while the ball is unseen but the paddle is
visible,move toward the midpoint81.5 of the retained paddle limits15.5/147.5.
Use the existing damping.75 and deadband4 on `paddle_x + .75*paddle_velocity`.
Request RIGHTFIRE,LEFTFIRE orFIRE so serve behavior remains possible. Set the
previous requested command accordingly for the next visible-ball plan. Preserve
the detector,visible-ball planner,response coefficients,edge reset,clipping and
all other costs. The qualified native teacher remains unchanged.

The saved eight-episode trace contains1,114 missing-ball decisions out of12,914
selected decisions (8.63%);all requestFIRE. In697 the paddle is more than35 pixels
from center. Previous observed motion identifies839 as upper-region candidates,
including552 far-from-center decisions. These are descriptive proxies,not
verified occlusions or proof that centering helps. The change can affect upper
tracking gaps and serve phases together;no isolated causal claim is intended.

Reuse the retained first natural episode per eight lanes,seed4000001009,returns
151,97,256,226,204,331,82,136 (mean185.375),zero two-wall successes. Keep native
pooled RGB,full18,sticky.25,repeat4,no reset noops and100,000-frame episode cap.
Fixed200k aggregate-action/90-second collection and180-second subprocess caps.
Preserve all extra episodes/tails;select episode0 without truncation. Require
full exact pixel/action/debug/reward/reset/task replay and video. Do not rerun
the retained control.

Before outcomes,the development screen requires at least10% higher mean than
185.375,improvement in at least five of eight lanes,no fewer two-wall successes
and all audits passing. A pass only permits a separately selected24-episode
assessment. It does not replace the native teacher or select training. A failed
screen retains the original behavior without center/damping/deadband retries.
Keep the90%-two-wall gate unchanged. This heavily reused reference cohort is
not Kindle learning or independent reliability evidence.

Check eight fixed cases for direction,deadband,momentum,no paddle and unchanged
visible-ball behavior;verify the exact source change. Use one low-priority CPU
core. Pin inputs and verify active BreakoutPID798240/invocation
cc7e1c87ac4b478ab8d9f23644a5fa65 and315 inputs before/after. Retain co-host time
without correcting learner wall time. No native agent,GPU work,updates or RAM
input. Preserve failures and do not repeat passing stages.

Root `/x/Temp/kindle-breakout-unseen-center-20261101`. The preceding saved-log
analysis is `/x/Temp/kindle-breakout-unseen-ball-20261101`;no emulator replay
or gameplay modification occurred in that analysis.
