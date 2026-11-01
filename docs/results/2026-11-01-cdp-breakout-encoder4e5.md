# Breakout encoder4e-5 passes bounded execution and early cost checks

The fresh12M64/Auto/N8/B16/T16/H15/R16/expert100 check completes8192 actions/
32,721 emulator frames/489 updates in46.899s. Frozen1024 actions/4096 frames
take.899s with zero updates and no natural completed episodes. Conservative200k
projection20m59s passes the30-minute gate. This establishes execution and early
cost,not a gameplay gain or measured full-run time.

Only encoder2e-5→4e-5 changes from the retained compact control. Native447fe50,
the exact retained runner,pixel teacher and all other learner settings stay fixed.
The earlier4M/B8/expert10 rate4e-5 comparison remains negative evidence; this
distinct12M/B16/expert100 test does not establish a universal rate or causal
architecture diagnosis. Seed1009/evaluation cohort are reused development inputs.
No previous experience enters the check.

All194 declared inputs,357 control-matching initial and invariant frozen tensors,
47 pre-update events,exact config/counters,finite reports,changed CNN/reward/
actor/value,full training input-pixel/action/label replay and both host guards pass.
First update376/residual credit.5 match. All489 reports are supervised;
117,051 labeled replay rows equal the weight sum,with maximum raw world/actor
expert-loss difference8e-7. Actual3792 overrides are46.2890625%;4400 passive
labels and4065 action disagreements are exposure,not unaided competence.

Service`kindle-breakout-encoder4e5-check-20261101` starts03:03:57 UTC November1,
sourcee11932e,PID732514/invocation`9d574be690dc445290d0c14c0101cfd0`,and succeeds
03:05:01 after64.243s. Terminal collection succeeds03:05:09. No failure,GPU
repeat or host recovery occurs. CPU preflight passes two actual CLI paths,
13 audit-config cases,the357-tensor schema and173 control pins in.425s with
zero native/environment construction. No GPU allocation is active at closure.

Root`/x/Temp/kindle-breakout-encoder4e5-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder4e5-expert100-check`.

- Declaration`5837c966a748c36bbb7cba3b5bc0e5968c454b91378cf4266256dfcfb36d37a4`.
- Review`e9bf3082ad7400111d9c69bd28e35478207bc8d567640e4dc9708af33d41d2c2`.
- Completion`7a410ce63e46a48bfddb4595efd8bac2b031bdaf6b3aa4b010b7040a7022d803`.

The [bounded protocol](../experiments/2026-11-01-cdp-breakout-encoder4e5.md)
selects no automatic promotion. After its completed checks, a
[separate fresh200k comparison](../experiments/2026-11-01-cdp-breakout-encoder4e5-200k.md)
is selected to test frozen gameplay against the retained25.292/42.625 control.
The five-game goal remains unfinished.
