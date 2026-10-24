# Centering recovers useful recurrent Pong state in the weak seeds

The six paired frozen probes complete October6 09:02 UTC in9m40s. All six
guards and the independent8.4s CPU audit pass: identical environment traces
and GPU-resized pixels, zero actor updates, exact346 saved tensors/model and
the declared h1 deterministic-state alignment bound. No new GPU warnings.

**The learning gain has a corresponding world-state gain.** Both weak raw-CDP
seeds still lose ball information between CNN and recurrent state. Every
centered seed retains readable ball/enemy state and predicts negative reward
events. This supports keeping the loss change, not declaring the world solved.

## Readable state

Held-out readout R²; ball_x /ball_y. Labels and fitted heads never enter actor
training. Every arm has the same train/validation/test split and head budget.

| Model | CNN ball R² | Posterior ball R² | h15 prior ball R² | Posterior negative-event AUC |
| --- | ---: | ---: | ---: | ---: |
| Control1009 | .616 /.875 | -.502 /-.405 | -.427 /-.568 | .534 |
| Centered1009 | .657 /.833 | .794 /.795 | .582 /.392 | .998 |
| Control2017 | .604 /.834 | .779 /.849 | .593 /.531 | .996 |
| Centered2017 | .621 /.822 | .739 /.777 | .575 /.413 | .993 |
| Control3019 | .666 /.855 | -.352 /-.361 | -.420 /-.546 | .531 |
| Centered3019 | .600 /.839 | .729 /.818 | .467 /.526 | .994 |

The shared pixel positive control has ball R² .156/.440 and player/enemy
.570/.711; its small fitted head is not an optimal pixel decoder. Control2017
already has useful state and remains better than its centered match on several
coordinate probes, despite worse gameplay. This is not uniform representational
superiority or a claim that coordinate readability alone determines policy.

CNN centered variance /total energy is .232%,8.046%,.278% in raw controls and
8.465%,7.692%,6.523% in centered models. The visual input was not replaced or
downscaled then upscaled; the change is the detached training-loss centering.

## Actual prior forecasts and controls

Each centered prior beats the constant training mean, persistence and unrelated
actions in its own raw embedding space at both h1 and h15. Do not compare raw
cosine magnitudes across differently learned encoders or training objectives.

| Centered seed | h15 prior cosine distance | Constant mean | Persistence | Unrelated actions |
| --- | ---: | ---: | ---: | ---: |
| 1009 | .02771 | .04235 | .07236 | .05246 |
| 2017 | .02897 | .03780 | .06147 | .05237 |
| 3019 | .02053 | .03225 | .05771 | .03374 |

On the same visible h15 cohorts, all three centered models beat privileged
position persistence for every coordinate. Player-y RMSE is33.31/37.80/31.97,
versus54.02 for persistence and69.52/71.70/69.87 under unrelated actions.
Unrelated actions do not consistently worsen ball forecasts, which often cover
free flight. At h1, privileged position persistence still wins decisively; for
example its ball_x RMSE4.43 versus16.70/17.63/17.80 for centered priors.
There is no exact physics, complete-state or universal action-sensitivity claim.

## Sparse reward limits

The full held-out test has12,288 transitions/model,259 negative and9 positive
rewards. Centered posterior negative means are-.827/-.556/-.562, positive
means.483/.259/.076 and zero-event means about-.004 to-.006. Weak raw controls
predict near-.02 regardless of event. These posterior estimates are not forecasts.

The h15 prior cohort has758 forecasts, **16 negative, one positive and zero
terminal events**. Centered reward MAE.01702/.01512/.01794 beats the zero
baseline.02243 and unrelated-action.01883/.02144/.02091; negative-event AUC
.995/.990/.973. However the sole positive forecast is only.000012/.000125/.00534
for an actual+1. Positive-event sufficiency remains unproven. h1 has15 negative,
zero positive and one terminal event; do not claim terminal/positive reliability.

## Interpretation, accounting and reproduction

Together with the [three-seed gameplay comparison](2026-10-06-cdp-centered-learning.md),
this supports the specific diagnosis: a dominant shared embedding component
let the original loss coexist with poor recurrent task state. Centering improves
the failed seeds without a new encoder, larger model or pixel reconstruction.
It does not explain every remaining policy failure, and Pong still has only
4/72 frozen wins at200k actions. All five DreamerV3-quality targets remain open.

The [follow-up declaration](../experiments/2026-10-06-cdp-centered.md#reviewed-follow-up-october6)
uses the existing development trajectories, not a new unseen-game evaluation.
Each model collects32,768 random actions and fits five GPU readouts at2048
updates each: **196,608 extra diagnostic actions,61,440 readout updates, zero
actor updates**. Train-only normalization, validation-only head selection,
held-out readout tests; no actor checkpoint selection. All raw trajectories,
controls, event counts, readout files and failed/control outcomes are retained.

Both training and probing use natived32f3dc8 /Meganeurab684ffd9 /Bladee349cddf.
The newly publishedc6376542 attention-backward update was inspected but is not
substituted into this matched study. Earlier500k diagnostics retain their own
native/backend identities. No learning worker remains.

[Compact metrics and provenance](2026-10-06-cdp-centered-world.json);
[complete audited per-trajectory evidence](../../runs/cdp-centered-world-20261006.EYV6C9au/audit.json).
Artifacts: `runs/cdp-centered-world-20261006.EYV6C9au`.
