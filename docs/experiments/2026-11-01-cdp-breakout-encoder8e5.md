# One bounded Breakout encoder8e-5 check

The completed [encoder4e-5 comparison](../results/2026-11-01-cdp-breakout-encoder4e5.md)
raises frozen sampled/greedy scores to56.292/104.667 from25.292/42.625 at2e-5,
in21m28s versus22m27s. All modes still have0/24 two-wall successes. Fixed-corpus
ball and motion readouts also improve, while the [teacher-takeover diagnostic](../results/2026-11-01-cdp-breakout-recoverability.md)
shows that earlier feedback control avoids many losses that late takeover does
not rescue. Neither result proves that encoder learning is the sole remaining
bottleneck. The earlier4M/B8/expert10 rate4e-5 failure remains negative evidence
against a universal larger-rate rule.

Prepare one bounded execution and cost check at encoder8e-5, changing only
encoder4e-5→8e-5 from the retained12M64 control. This is a separately selected
development point, not a rate queue, full learner or new architecture/loss.
The active Qbert4e-5 pipeline must finish its original audits, postprocessing,
collection and independent terminal confirmation before selection or launch.
CPU preparation can proceed while its CPU audit runs; no GPU work overlaps.

Keep centered CDP/12M64/Auto/N8/B16/T16/H15/R16,base/actor4e-5,dynamics4e-4,
warmup1000,actor-unimix.01,action-effects coefficient1,full18 actions,sticky
actions.25,action repeat4 and one GPU resize from native frames to64. Keep
constant expert100,world imitation gradients,all action weights1,20% episode
guidance/cap1920 and all-state corrective labels. Native447fe50,the exact
retained runner and pixel teacher remain unchanged; use the qualified runtime
at`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build is needed.
Seeds1009/eval4000001009 and the cohort are reused development inputs. No prior
weights,replay,games,belief,RNG or check experience enter this fresh run.

Train8192 aggregate actions/489 updates,then restore for1024 greedy frozen
actions with zero updates. Require357 finite/control-matching initial tensors,
357 invariant frozen tensors,47 matching pre-update semantic events,exact
configuration and source hashes,first update376/residual credit.5,finite reports
and changed CNN/reward/actor/value parameters. Verify raw expert world/actor
agreement,row/weight accounting,passive labels/overrides,complete training
input-pixel/action/teacher-label replay and both AMD host guards. This short
frozen segment cannot establish task competence or a gameplay benefit.

Caps remain training180s,frozen60s,CPU audit180s,whole service480s. Require
`training_seconds * 200000 / 8192 * 1.10 <= 1800` for early cost qualification.
Any full200k comparison needs a separate decision after these results,with a
30-minute training cap and fixed final/actual-initial sampled/greedy24-episode
assessments against retained56.292/104.667. No full learner,extension,retry,
cap increase,broader rate sweep or automatic promotion follows this check.

CPU preflight1.625s passes both actual CLI paths,14 actual audit-config cases,
the357-tensor schema,the8192/100k/200k update schedule and227 retained control
pins. Both previous encoder rates are rejected; base/actor rates stay4e-5.
There are zero native construction attempts or environment creation. The runner
is byte-identical to the retained4e-5 control. Preparation occurs after Qbert
training and frozen evaluation finish,while its original CPU audit continues;
it does not change the recorded training time. No GPU check has launched yet.

Root`/x/Temp/kindle-breakout-encoder8e5-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder8e5-expert100-check`,service
`kindle-breakout-encoder8e5-check-20261101`. Selection requires an idle host,
clean source tree,matching boot/driver pins and completed original Qbert evidence.
The five-game goal remains unfinished.
