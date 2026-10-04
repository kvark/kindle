# Probe normalization correction

The initial MLP results are superseded, not evidence that raw pixels cannot
represent game state. [Measured training-column statistics and affected result
identities](2026-09-27-probe-normalization.json) preserve the diagnosis.

Float32 axis-0 mean/variance accumulation invented standard deviations above
the existing `1e-6` threshold in truly constant columns. Dividing by those tiny
values magnified held-out changes. On the raw two-frame training features:

| Game | Constant columns | Incorrectly treated as varying | Maximum false std |
| --- | ---: | ---: | ---: |
| Pong | 10,081 | 7,498 | 1.109e-5 |
| Breakout | 11,556 | 3,136 | 7.629e-6 |
| Seaquest | 8,175 | 4,104 | 6.735e-6 |

F64 statistics give exactly zero variance in all these constant columns. The
correction uses F64 **training-only** feature/target statistics and then F32
inputs on the same native GPU MLP. The threshold, head seeds, optimizer, budget,
validation selection and held-out data are unchanged. No test-derived floor or
favorable model subset is introduced. Ridge already used F64 and is unaffected.

A CPU regression fixture checks constant-column behavior and held-out scaling.
All five MLP controls must be refitted into fresh outputs; the public report
reader refuses the superseded normalization. Original raw results stay intact.
