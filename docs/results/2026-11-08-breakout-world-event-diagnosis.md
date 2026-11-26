# Breakout event errors occur during rollout and on observed terminal examples

The CPU diagnosis is complete. It finds a large accumulation of false ending risk during imagined rollout, plus weak terminal recognition on a small observed fitting subset. Forcing the saved survival weights to one does not repair the full model. The world-model goal remains unproven; retain H0 as the broad reference.

[Protocol](../experiments/2026-11-08-breakout-world-event-diagnosis.md).
Root `/mnt/data/kindle-breakout-world-event-diagnosis-20261108`. Source base `a3e90e19fac85a8121208d700f48decbc8766a2f`.

## Where the errors appear

These quantities remove the configured continuation discount. Observed head means use the actual image-conditioned state. Prior hazards use saved model rollouts without future images. Branch cumulative ending probabilities multiply conditional survival along each sampled path before averaging draws. The table keeps these distinct; a small instantaneous error can accumulate over time.

| Model | Observed nonterminal mean, dev128 | Observed terminal mean, fit4 | Terminal-arrival prior mean, selected dev10 | Non-ending branch cumulative mean at8 /16 /32 |
|---|---:|---:|---:|---|
|Reference H0|0.0236%|82.824%|68.166%|0.049% / 0.106% / 0.256%|
|Categorical H16|0.0779%|49.896%|32.650%|0.070% / 0.232% / 1.971%|
|Spatial H16|0.1464%|38.785%|19.931%|0.750% / 2.239% / 13.554%|
|Image feedback|0.0418%|43.600%|13.073%|0.337% / 1.042% / 7.539%|
|Nonlinear fusion|0.2097%|23.278%|11.615%|0.750% / 5.860% / 34.180%|

All128 natural development context states are nonterminal. They cannot establish observed terminal recognition. The fitting natural cohort has four terminal contexts among256; these are few reused examples, not independent confirmation. The ten terminal-aligned development origins all terminate at their next arrival and are selected by outcome. Their mean hazard is neither a natural-frequency false-alarm rate nor population calibration.

Fusion has one observed development nonterminal hazard above5% (maximum14.631%), versus none for H0 (maximum0.563%). This observed error exists, but does not by itself explain the much larger cumulative branch risk. On non-ending planned branches, fusion mean instantaneous hazard grows from0.0070% at1 to4.4914% at32. Its cumulative risk grows from0.750% at8 to34.180% at32. The reference reaches0.256% at32. All per-origin/horizon values, both histories and individual alarms are retained; pooled means do not replace category gates.

Observed terminal fitting is weak too: fusion Brier .602037 versus H0 .039791 on those four contexts. On the selected terminal-arrival prior examples it is .826041 versus .231761. These observations contradict a diagnosis restricted to multiplying otherwise accurate heads. They do not identify whether shared representation, optimization, sampling or head capacity caused the regression.

## Ball errors remain after removing survival weighting

As a diagnostic, keep every saved decoded ball path but force survival to one. This is a different, generally incorrect forecast, particularly at real endings. It is not calibration, a valid replacement gate or evidence of a learned improvement. The original survival-weighted results and failed gates remain authoritative.

| Model | Original primary energy | Survival-one sensitivity | Non-ending categories only: original /survival-one |
|---|---:|---:|---|
|Reference H0|0.169502|0.197336|0.183739 / 0.184267|
|Categorical H16|0.311977|0.335400|0.289636 / 0.287705|
|Spatial H16|0.478844|0.494613|0.444349 / 0.434424|
|Image feedback|0.256859|0.264586|0.273031 / 0.261057|
|Nonlinear fusion|0.345425|0.271092|0.312616 / 0.220445|

Fusion improves under this intervention on the forecast weights, but its ball paths still score worse than H0 when both use survival one: .271092 versus .197336 across seven categories, or .220445 versus .184267 across six non-ending categories. Thus fixing only the event weights would not establish the required physical forecast quality. These differences are sensitivity results, not additive causal fractions of model error.

## What this changes next

The next justified learning intervention must preserve short-horizon visual fitting while supervising or stabilizing the states and event heads actually used in forecasts. Ordinary H16 motion-RGB training supervises observed reward/continuation heads, but not their imagined suffix outputs. The earlier image-feedback experiment also adds future event losses, while changing recurrent conditioning; its failure does not isolate event supervision alone. A separately declared event-supervised suffix without generated-image re-encoding remains an available hypothesis, not a selected or proven fix.

No gradient interference was measured here. Changing a decoder also changes learned states, gradient paths and fitting, so these comparisons do not prove an intrinsic recurrent defect, an event-head capacity limit or a particular loss cause. Do not replace proper event forecasts with an always-alive model. Do not resume actor/critic training on the strength of the recorded tracking improvement.

## Execution and limits

First declared CPU analysis passes 1,136 input pins and 2,300 exact context/corpus RGB joins in 19.360s. The independently reconstructed F64 continuation MLP agrees with every saved native endpoint within 9.48e-07, below the predeclared1e-4 tolerance. There are 11,500 parity values across five models/four cohorts. Per-pixel targets are not used in this MLP calculation.

All 860,160 survival values, 16,800 event means and 16,800 original per-origin energy scores agree within1e-12. Retain 9,665 supported head rows and 16,800 branch rows, plus all 1,180 non-ending alarms across five models and both histories. Their related observations are not independent trials.

No native extension or ALE module is imported; GPU loader discovery is disabled. There are zero new model queries, training updates, environment actions, emulator frames or hardware allocations. All old native stages and original failed comparisons remain closed. The source-only console aggregation initially used system Python without NumPy and failed before computing a result; the project interpreter completed that console check. The declared analysis itself passes on its first execution with unchanged inputs and tolerance.

The plot uses retained tables only. Fitting/development seeds and families are reused. Terminal support is explicitly limited. No scalar fit, threshold search, new loss, learner, collection or automatic GPU successor is selected. Full causal physics, event timing/calibration, persistent brick memory, late/tunnel/second-wall coverage, untouched families and fresh confirmation remain required.
