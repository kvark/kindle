# Breakout critic calibration on the retained policies

The two current Breakout critics have weak association with realized returns
under their sampled policies: correlation **0.105 / 0.097**, R² **−0.031 /
−0.267** for dynamics rates 4e-4 / 1e-4. Both beat the constant baseline fitted
on earlier episodes, whose mean differs from the test mean, but neither beats the
hindsight mean of the test targets in squared error. This is a measured critic
weakness, not proof that critic error causes the gameplay limitation.

The fixed [protocol](../experiments/2026-11-01-cdp-breakout-current-critic.md)
completes **145,632 actions / 582,380 frames in 154.699s**, with zero updates.
Every transition, episode and reset matches the retained source trajectory;
all 357 frozen tensors per arm remain unchanged. Original training times and
scores remain unchanged; all four arms still have zero of 24 two-wall successes.

## Complete-return calibration

Each arm uses 24 natural episodes: the first episode per lane fits a constant,
and the next two per lane provide 16 test episodes. Targets sum actual external
rewards and the native action-conditioned intrinsic bonus with the same F32
discount 1−1/333, through natural termination, with zero terminal bootstrap.
No value prediction is used to construct the target. State rows within an
episode are correlated; row counts are not independent sample counts.

| Dynamics rate / action mode | Test rows | Target mean | Predicted mean | Bias | Correlation | R² | Critic RMSE | Earlier-episode constant RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4e-4 sampled |16,145|12.718|13.359|+0.641|.105|−.031|6.682|8.847|
| 1e-4 sampled |17,532|12.841|14.593|+1.752|.097|−.267|6.691|7.661|
| 4e-4 greedy |21,361|30.063|18.954|−11.109|.445|.037|27.261|30.911|
| 1e-4 greedy |21,566|31.271|18.330|−12.941|.318|−.051|34.391|33.603|

Sampled arms are primary for calibration because the imagined actor samples
its actions. Greedy changes the target policy and visits different states;
its stronger returns and larger underprediction do not by themselves establish
a critic training defect. The two models also visit different states despite
sharing learner/evaluation seeds. Cross-model error differences are not matched-
state causal comparisons. Realized returns are noisy samples of expected value.

| Dynamics rate / action mode | Mean external discounted return | Mean intrinsic discounted return | Intrinsic fraction of mean total | Per-action bonus mean |
| --- | ---: | ---: | ---: | ---: |
| 4e-4 sampled |11.000|1.718|13.509%|.007417|
| 1e-4 sampled |11.365|1.476|11.496%|.006222|
| 4e-4 greedy |28.187|1.876|6.240%|.007405|
| 1e-4 greedy |29.787|1.484|4.744%|.005854|

The bonus contributes materially to the sampled target. These fractions do
not establish gradient dominance, harmful exploration or a reason to remove
the bonus. The saved final 50k training summaries in the protocol describe
imagined rewards and are distinct from these realized discounted returns.

The result supports investigating how the actor/critic use future outcomes.
It does not measure alternative-action ranking or policy-gradient correctness,
and does not select a new loss, coefficient or learner. Existing representation
and earlier-teacher-takeover evidence remains relevant; this diagnostic does
not identify a single cause of Breakout's weak control.

## Execution and retained evidence

Native 447fe50/SHA35ad0db and each exact original runner remain unchanged:
6c1b69fb for 4e-4, fe5cf347 for 1e-4. Runtime is
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. The live native probe reads
value and intrinsic bonus after act and before the environment step; it does not
sample another posterior or learn. Guidance and labels are off. Initial-policy
captures are unnecessary for this calibration of trained critics.

The preliminary 1,024-action / 4,096-frame check takes 1.022s, preserves 357 tensors,
and matches 128 semantic events. All 73 declaration pins, original checkpoint
counters/config/runner/native identity, finite values and bonuses, 24-episode
cohorts, zero truncations, host guards and source-log hashes pass. Full semantic
matches are 4,394 / 4,677 events for 4e-4 sampled/greedy and 3,878 / 5,469 for 1e-4.
Original frozen replays are reused only after source-log hashes and complete
semantic equality pass. They are not rerun. Source logs lack live pixel hashes;
no unavailable live-pixel equality is claimed. Qbert-specific phase slices are
excluded because Breakout's recorded milestone means two walls, not one wall.

| Capture | Actions | Frames | Seconds | Retained mean |
| --- | ---: | ---: | ---: | ---: |
| 4e-4 sampled |34,704|138,781|35.190|56.292|
| 4e-4 greedy |37,016|148,038|31.992|104.667|
| 1e-4 sampled |30,592|122,321|38.401|50.625|
| 1e-4 greedy |43,320|173,240|49.115|138.750|

Selection follows the fresh Qbert original closure and idle-host check. Main
PID 868111 / invocation `b1dacd51b17b40849c8345ec02504658` launches 00:03:12 UTC
November 2 / source1b6bed9 and succeeds **00:06:14.211**. Original collector PID 868114 / 
invocation`798fa2c7916e4965a10e810cef33a9f0` writes its completion at
**00:06:33.357** and is terminal success. Its journal has no separate resource-
consumption terminal timestamp; the completion payload andterminal systemd
state are verified. The .882s read-only summary (.901s wrapper) verifies original
identities, all pins, all completion files, all four analyses and target decomposition.
No original assessment stage or collector fails or repeats. Both services are
terminal,withno active Kindle allocation at closure.

CPU qualification and its two preserved preparation failures are detailed in
the protocol: missing runtime during test collection, then an absent Qbert-only
argparse field. Their separate corrections precede selection and do not change
native code or actual frozen commands. Nine value-target/ledger tests pass.

Evidence root: `/x/Temp/kindle-breakout-current-critic-20261101`.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `1d46e9bf0bf05f43a7f3ec2d779c38580489b6c64b675dc31cd481fece622160` |
| Comparison | `d0a29e58e05052495179c0a1c37e090156292b211759fdd2a43e762ecc0a7352` |
| Original completion | `ffc28bb4d333c59b4161a72cc9a4bdd620f332699c03d65443ec81e930289834` |
| Read-only summary | `e2d18cd35a1fde3d2539f97f7441c6d598d93bdbe30d30726677d7282b7dff4a` |

No learner, loss change, build, retry, extension or automatic successor is selected.
Do not repeat these completed captures or collectors. The five-game goal remains
unfinished; check live state before any separately selected GPU work.
