# One extra response-history step does not pass the held-out screen

Retain the current Breakout teacher. The fixed six-coefficient model lowers
held-out RMSE but raises MAE, both overall and near descending balls. It fails
the predeclared requirement that both measures improve in both subsets. No
CPU gameplay comparison, teacher integration or learner follows this fit.

| Subset | Rows | MAE, control → history | RMSE, control → history | Bias, control → history |
| --- | ---: | ---: | ---: | ---: |
| Training interiors | 19,146 | 1.862086 → 1.889899 | 3.107795 → 3.031462 | -0.176465 → 0.000171 |
| Held-out all | 13,102 | 1.976923 → 1.997426 | 3.287560 → 3.203877 | -0.356574 → -0.189747 |
| Held-out near descending | 2,955 | 1.948802 → 1.958316 | 3.283574 → 3.191654 | -0.321382 → -0.154457 |
| Held-out stationary edge | 849 | 2.586030 → 2.423084 | 4.287527 → 4.063112 | -1.538543 → -1.400381 |

Six of eight held-out lanes have worse MAE; lanes1 and2 improve. The unclipped
velocity recurrence is stable, with spectral radius.165412, but stability does
not rescue the failed error screen. Lower squared error is consistent with the
OLS objective;the acceptance rule is not changed after observing the result.

The predictor adds one preceding observed displacement and one older command:
`[1, v_t, v_(t-1), u_t, u_(t-1), u_(t-2)]`. History is causal and resets on
fallbacks, episode boundaries and the existing stationary-edge condition. The
latter clears older velocity/commands while retaining current observed velocity.
Position clipping remains15.5–147.5. Exact F64 coefficients are retained:

```
[-0.04266994056141596, 0.26315845990162845, -0.02736124673038591,
  8.386187914149202, 10.987539445226144, -2.041187986544125]
```

All37,448 earlier selected rows,targets,splits and control predictions reproduce
exactly;46,122 source predictions are checked. One full-rank F64 fit uses the
same19,146 interior rows from episodes0–1. Episode2 supplies13,102 held-out rows,
including2,955 near descending balls and849 stationary-edge rows. There is no
row reweighting, regularization, feature search or alternate fit. The original
four-coefficient rejection and ambiguity evidence remain unchanged.

This reused episode holdout has already informed earlier diagnostics and is not
independent validation. The result rejects this specific linear observable-
history model. It does not rule out other history, pixel-timing corrections or
uncertainty models, establish an irreducible noise floor, or measure information
in Kindle's recurrent state. No gameplay benefit or architectural cause follows.

Seven CPU tests cover coefficient recovery/rank and nonfinite rejection,causal
history,fallback/episode/edge resets,old-model embedding and clipping. They pass
in.164s wrapper time. Preflight.138s preserves54 qualified inputs and confirms
all five preceding centroid services terminal. The fit/analysis takes **.853s**,
controller **1.087s**, completing **11:55:03 UTC November1**, source`885f9d1`.
All56 selected input hashes remain unchanged;normal-equation relative residual
is4.548e-16. There is no active learner,CPU co-host correction,native/emulator
import,GPU work,update,failure or repeated fit.

[Fixed protocol](../experiments/2026-11-01-cdp-breakout-response-history.md),
[earlier response refit](2026-11-01-cdp-breakout-response-refit.md),
[input ambiguity](2026-11-01-cdp-breakout-response-ambiguity.md).
Evidence `/x/Temp/kindle-breakout-response-history-20261101` retains all extended
rows,coefficients,per-lane metrics,tests and receipts. Complete;do not rerun the
fit or qualification. No coefficient sweep,gameplay test,native change or
automatic successor is selected. The five-game goal remains unfinished.

- `declaration.json`: `f7e4d91d5a5f3632633efdba66c2116d725cc62d6543c8fdae6f9381a3f27433`.

- `review.json`: `c3a2c6d0ee483828ee97e7686ae4b3939f346c5150a739d1197b0321620cbaf9`.

- `fit-rows.jsonl`: `668da1aca36a94ba630f7656fe45a6a1af134dc918db60a7fb426c6702352377`.

- `completion-review.json`: `5fb98192e4eb50b58a38558752ed33e129dfba7573fd1a14e19cb7208056e7d9`.
