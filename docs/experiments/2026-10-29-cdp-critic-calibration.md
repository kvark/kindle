# Frozen live-belief critic calibration

The completed forecast diagnosis finds useful15-step reward information in the
retained Qbert12M48/B16/R16/expert100 model, with larger errors after its first
pyramid. Measure whether the critic estimates actual future returns under its
current policy. This is part of the existing five-game goal; no learning or new
loss is selected.

## Selected assessment

Use native447fe50 and the matching Breakout-era runtime. The retained runner
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/atari_vector.py` already
supports `--probe-behavior`. It reads the critic and action-conditioned intrinsic
bonus from the original vector actor's live state after action selection and
before the real transition. It does not sample another posterior or update any
weights. The separate unbuilt greedy-posterior implementation is unused.

Source trial:
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
First perform one1024-action greedy final-model execution check, capped at120s.
Require unchanged357 tensors, zero updates, finite values/bonuses, successful
host guard and exact transition-prefix equality with retained `final-greedy`.
Do not infer task competence from this bounded check.

After it passes, run the four preselected final/actual-initial × sampled/greedy
arms. Keep Auto/N8/RGB48, original seeds, full18, sticky.25, repeat4, native frames
and one GPU resize. Select the first3 natural episodes per lane,24 each, with
the original600k-action/300s caps. Guidance and labels remain disabled. Require
exact action/reward/boundary/frame and episode/reset equality with each retained
evaluation. Original aggregate counts are33248/8728 sampled final/initial and
48256/8432 greedy final/initial. Each source replay is reused only after this
complete semantic comparison and matching original-log hash verification.
This establishes outcome identity; it does not create new pixel-hash recordings.

Pin the runner, native, Python runtime, controller, checkpoint, source logs,
source replays and analysis inputs before GPU execution. Stop on a failed
numerical, source, trajectory or host check. No retries or host recovery. Preserve
all357 tensors in each arm and all checkpoint counters except the declared
evaluation observations. Retain complete logs and invariant frozen exports.

## Analysis

Use `probe_policy_values.py`: discounted realized returns from complete natural
episodes, including the frozen intrinsic bonus, without critic bootstrapping.
Gamma is the model's F32 `1 - 1/333`. First episode per lane calibrates a constant
baseline; the next two supply16 test episodes. Exclude unfinished/excess tails.
Report critic bias, MAE, RMSE, R², correlation, initial-state values and bonus
magnitude, against calibration-mean and zero baselines. Split before/after the
first pyramid using the verified replay outcomes.

The sampled final arm is primary because imagination samples actor actions.
Greedy evaluation changes the behavior policy and can change its true value.
Initial and final actors visit different states; their own-policy diagnostics
are controls, not a matched-state estimate of value-learning improvement.
These reused development seeds/cohorts cannot establish learner reliability.
Realized returns are noisy targets for an expectation; poor statewise fit alone
does not establish an incorrect policy gradient or a causal score bottleneck.
The critic's symlog-distribution decoding and training/replay policy differences
are further interpretation limits. Actual alternative-action ranking remains a
separate diagnostic; this assessment does not supply counterfactual outcomes.

Results will guide a separately selected action-ranking or learning experiment.
No new training, changed objective or automatic successor allocation is selected.
