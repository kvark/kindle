# Consistent contact timing lowers the CPU teacher's gameplay score

The private timing correction fails its predeclared eight-episode gameplay
screen. Mean return falls **185.375→125.875 (−32.10%)**, with only **two of eight** lanes
improving and zero two-wall successes in either arm. Retain the original
teacher. No24-episode extension, integration, coefficient search or learner is
selected from this result.

| Lane | Original return | Corrected-timing return |
| --- | ---: | ---: |
| 0 | 151 | 185 |
| 1 | 97 | 57 |
| 2 | 256 | 108 |
| 3 | 226 | 196 |
| 4 | 204 | 125 |
| 5 | 331 | 102 |
| 6 | 82 | 173 |
| 7 | 136 | 61 |
| Mean | 185.375 | 125.875 |

The [saved-contact diagnostic](2026-11-03-cdp-breakout-paddle-contact-timing.md)
had modestly improved MAE/RMSE while worsening some tails. Using that exact
conversion to select actions gives lower gameplay quality in these reused
development episodes. This does not prove that timing information is useless,
identify a unique failure mechanism or measure the native learner's bottleneck.
The earlier teacher-state error reduction was insufficient evidence for changing
the guidance used during learning.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-contact-timing-policy.md)
changes only near-contact plan scoring. For each existing predicted path,
`q[0] = p[0] + observed_displacement/8` and
`q[i] = p[i] + (p[i]-p[i-1])/8`; interpolate at
`max(0, pooled_contact_steps-.125)` and score against the unchanged landing
target. Movement cost, response coefficients, clipping, detector, cached landing,
edge reset, far-horizon cost and fallback stay fixed. Native runtime and its
teacher file remain byte-identical. The candidate is a private CPU reference.

The screen required10% higher mean, at least five improved lanes and no fewer
two-wall successes. It fails the first two requirements. All eight selected
episodes are natural episode0, without truncation, under the retained seed
4000001009/full18/sticky.25/repeat4/noops0/native pooled RGB protocol. The original
control is reused without collecting or replaying it again. These are heavily
reused development episodes, not an independent reliability cohort.

All16 CPU fixture cases and211 declaration pins pass. Preflight constructs no
emulator or native agent. Its wrapper takes1.171s and completes
**09:19:18.039 UTC November3**; the inner receipt's earlier timestamp excludes
later hashing and is not the total preflight duration. Clean source
`334972c8950e20ebc44b38cff327fcd602446fdd` is selected09:19:33.136.

Collection completes13,352 actions/53,396 raw frames in8.838s. Full exact
pixel/action/reward/termination replay, recomputed policy debug and lane0 video
pass in10.789s. The controller completes **09:20:01.737** in19.811s. The independent
saved-ledger recount passes once at **09:20:24.966** in.085s, verifying all actions,
returns, criteria and2,552 corrected near-contact decisions; there are9,943
far-horizon decisions,857 fallback decisions and zero time clamps. All source
pins remain unchanged.

No failure, emulator-stage repeat, GPU work or learner update occurs. Root
`/x/Temp/kindle-breakout-contact-timing-policy-20261103` is closed; do not repeat
collection, replay or recount. No Kindle allocation is active at closure.
Breakout and the five-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `07cb8a56fa57c1aebef4787376f0c28b47e9159ee7a90442ac332949c189ce63` |
| Gameplay review | `17202d2d19421cbb0fcaa353b1278ff5403d10f03f134e3c66f6f81d37576fab` |
| Comparison | `b644a8e045abc4f91ee36c641075e883316a0f8157c45d53928a6913a87928a9` |
| Completion | `b915684ae5fadd4ab65cf463eb89ea20fa3a53122eb046ac40850fe28669fbc6` |
| Independent recount | `d9b3054283b5efef5e13b3bdfc4ec7a346569a68232a38e184984b38a52b90a6` |
