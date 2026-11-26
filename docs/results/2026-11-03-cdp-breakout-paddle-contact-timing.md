# Consistent timestamps modestly improve retained contact predictions

The fixed timing conversion passes its declared contact screen. On reused
episode2 contacts, paddle-position MAE falls **6.101→5.757 pixels** for losses
and **2.376→2.283** for successful contacts. Both RMSEs improve, and combined
held-out MAE improves in seven of eight lanes. This supports a separately
specified CPU controller comparison; it does not establish better gameplay.

| Accepted contacts | Count | Paddle MAE, original → converted | Paddle RMSE, original → converted | Timing MAE in raw frames, original → converted |
| --- | ---: | ---: | ---: | ---: |
| Episode2 losses | 18 | 6.100747 → 5.757079 | 7.544945 → 7.250068 | .396368 → .210364 |
| Episode2 successful | 245 | 2.375659 → 2.283485 | 3.047398 → 2.926359 | .557065 → .317579 |
| All losses | 65 | 4.943058 → 4.752385 | 6.518341 → 6.353575 | .460855 → .230607 |
| All successful | 674 | 2.317176 → 2.310334 | 3.036018 → 3.001300 | .561661 → .293622 |

The improvement is small and not pointwise. Held-out loss median error worsens
5.576→5.853 pixels and maximum worsens14.977→15.099. Held-out successful-contact
95th-percentile error worsens5.796→5.822 and maximum8.365→8.788; all-cohort
successful-contact maximum worsens13.868→16.982. Predicted-inside/actual-outside
counts fall38→35 for all losses and9→6 for all successful contacts, but held-out
counts change10→10 and2→1. These selected pixel fits are not exact collision
states or unbiased failure probabilities.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-paddle-contact-timing.md)
uses the recorded chosen five-action plan, unchanged response coefficients,
landing target and clipping. For its pooled path `p[0..5]`, the converted path is
`q[0] = p[0] + observed_displacement/8` and
`q[i] = p[i] + (p[i]-p[i-1])/8`. Interpolate q at
`max(0, pooled_contact_steps-.125)`. The ball and paddle timestamps shift
consistently. Under constant velocity these shifts cancel, away from the zero
clamp; the cancellation fixture passes. There are zero clamps in accepted rows.

All120 loss candidates and776 successful-contact candidates reconcile, retaining
65/674 accepted contacts and55/102 exclusions. Joins to the original52,072-action,
208,254-frame ledger verify source decision time, original predictions and errors.
Future fitted targets never enter a prediction. There is no fit, search, new
trajectory, replay, native/emulator import, GPU work or learner update. The
development holdout and original teacher trajectories are heavily reused.

All24 pins, five new fixtures and eight reused parsing/statistics fixtures pass.
Independent recount verifies739 rows and875 metrics. Source `b7bafe64`;
preflight takes.131s. Analysis completes **09:10:22.541 UTC November3** in.314s;
controller completes **09:10:22.591** in.407s. Recount completes
**09:11:34.480** in.036s. No failed or repeated stage occurs. Root
`/x/Temp/kindle-breakout-paddle-contact-timing-20261103` is closed.

The separately selected [eight-episode gameplay protocol](../experiments/2026-11-03-cdp-breakout-contact-timing-policy.md)
subsequently [lowers mean gameplay return185.375→125.875](2026-11-03-cdp-breakout-contact-timing-policy.md),
with only2/8 improved lanes. Retain the existing teacher and native runtime; no
extension, integration or learner is selected. The five-game goal stays open.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `e3bc6e85d99799c459224e0753316a9f99faf9f8c632f4f4a9dd07ddca35eeb0` |
| Review | `d1e457ca4149116222b9ed745bf5045a8df4989497932a9967aad70b0d020c98` |
| Completion | `872ae5de821e11a8647812bd6aefa47a9874cd801b365c9f5b95b65b5192f385` |
| Independent recount | `dd38befb75d1a33b08169c541113839e89eb92c1ab87e0135eb34c9bb14ee719` |
