# Breakout world model without the image-history shortcut

The [state probe](../results/2026-11-10-breakout-world-state-probe.md) showed
that the spatial-history recipe never encodes the ball in its recurrent state:
the decoder reads the ball from its two previous frames, so no gradient asks
the encoder for it, and the only ball pathway is autoregressive pixel feedback
that fades. This series removes the shortcut: the decoder renders from the
RSSM state alone, so the ball can only be reconstructed and forecast if the
encoder, posterior and prior carry it.

## Recipe

96x96 frames (multi-pixel ball), 1M preset, B8/T32/H16, Auto, seed 41001,
`motion_cross_entropy` (motion-weighted per-pixel cross-entropy on observed
reconstructions and on the 16-step prior rollout), RGB100, KL1/.1, rates 4e-4,
LEFT/RIGHT/FIRE action-feature table, `rgb_spatial_decoder`/`fusion`/`action`
all off. The same 524,288-action 96x96 corpus (`corpus/fit96`) and the
re-rendered November 7 development origins (`corpus/dev96`) as the other 96 runs.
The plain decoder costs 66.8 ms per update at 96 against 254 ms for the
spatial one, so a 16,384-update run takes about 20 minutes.

| Run | Output | Change |
|---|---|---|
| State-only 1M | `train-96-plain-v2` (first attempt `train-96-plain` lost to the host OOM killer at 3,200 updates) | 16,384 updates |
| State-only 4M | `train-96-plain-4m` | `size4_m` preset, 16,384 updates |
| State-only 1M, long | `train-96-plain-65k` | 65,536 updates, evaluation every 8,192 |

The 4M and 65k jobs run serially from `queue-plain-followups.json`
(`guards-plain-followups`) and replace the previously queued spatial-recipe
budget and capacity runs, which were stopped at startup (`guards-next-v2/superseded.json`).

## Scoring

The usual ball/paddle/brick/event metrics on the development natural, contact,
ending and 128 dense origins at every 2,048 updates, plus the linear state
probe on the saved encoder, posterior and prior features (ridge 300, fitting →
development, RAM positions in native pixels). The probe is the primary
diagnostic here: ball x/y RMSE from the current encoder features and from the
prior state at 1 / 4 / 8 / 16 / 32 steps, against a predict-the-mean baseline
of about 40 / 28 pixels and the spatial recipes' failure to beat it.

## Reading

If the ball is readable from the state and its prior-state RMSE grows slowly
with horizon, the latent dynamics carry the ball and the pixel renderer is the
remaining gap; if the pixel forecasts also beat the spatial recipes at eight
steps and beyond, the shortcut was the cause of the whole series. Capacity
(4M) and budget (65k) then show how far the state-only recipe scales before
any latent-prediction objective is added. Development screen on reused seed 41001.
