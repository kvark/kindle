# Late actor-only imitation decay worsens frozen Breakout

Fading actor imitation after 100k actions, while keeping world imitation at 100,
lowers sampled frozen score by **26.74%** and greedy score by **27.09%**, with
**1.10% more training time**. Retain constant actor/world imitation at 100 and
the existing 70.750/143.667 sampled/greedy control. The measured dominance of
imitation in two actor-gradient batches did not identify a useful change here.

| Measure | Retained constant imitation | Actor-only fade |
| --- | ---: | ---: |
| Sampled final / actual initial | 70.750 / 1.417 | 51.833 / 1.417 |
| Greedy final / actual initial | 143.667 / 2.208 | 104.750 / 2.208 |
| Two-wall successes, each final arm | 0/24 | 0/24 |
| Training time | 21m27.910s | 21m42.074s |
| Initial-subtracted sampled score/min | 3.230038 | 2.323217 |
| Initial-subtracted greedy score/min | 6.590132 | 4.725155 |

The candidate completes **200,000 actions / 799,600 frames / 12,477 updates**.
Initial-subtracted score per training minute falls **28.07% sampled / 28.30%
greedy**. Four frozen arms total 82,752 actions / 330,829 frames / 69.334s,
with zero updates and truncations. Each uses the first three natural episodes
per eight lanes, with guidance and labels off. Sampled remains primary; the
90% two-wall gate is unchanged and fails in both final arms.

The [full protocol](../experiments/2026-11-03-cdp-breakout-actor-only-decay-200k.md)
holds actor imitation at 100 through update 6,227, fades it linearly to zero
at 9,352, then keeps it zero through 12,477. All **3,126 zero-weight reports**
pass; world imitation remains 100 throughout. Teacher guidance, corrective
labels and optimizer history remain. This is not unaided training, and zero
current actor imitation does not erase its earlier optimizer contributions.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5, behavior8e-5,
dynamics4e-4, policy1, imagined-value1/replay-value1, actor/critic representation
gradients off and sampled collection. No crop, centroid or open-loop loss is
enabled. Seed1009/evaluation4000001009 are reused development choices; no
experience transfers from the bounded check or control. Nativec2d627f3/
SHA0015dff7 and runner8946f5f3 differ from retained447fe50/cf4ea238. The explicit
`replay_value_bootstrap=imagined` addition preserves the old implicit default.
Teacher and other learner settings stay fixed. This is not independent seed
confirmation or proof of an isolated mechanism.

## The comparison matches before the fade

The two saved-data comparisons were declared before the candidate's 100k
milestone and completed once, before frozen outcomes. All **12,500 transitions,
6,227 learner reports,170 episodes and167 resets** match exactly through 100k.
Exclusions are timestamps, episode wall time, learner timing and the candidate's
two scale fields, each required to equal 100. Actions, rewards, pixel hashes,
labels and overrides are included; there are no differing or unpaired records.

All **357 retained tensors / 51,502,659 elements** also match exactly at 100k,
including parameters, first/second optimizer moments and slow values. The
world/behavior/slow groups contain 280/66/11 tensors. Every difference count,
maximum absolute difference and L2 difference is zero. Serialized file bytes
are not the equality criterion. Checkpoints omit replay, belief, games and RNG;
these checks establish saved-evidence agreement, not full-state or broad runtime
equivalence. They strengthen the comparison before the actor schedule diverges.

The original recovery succeeds **11:13:07.765 UTC November3**, retaining
100k/6,227, four files and 357 finite tensors. Independent confirmation passes
11:13:49.130; trace/tensor comparisons complete 11:14:31.209/11:14:30.090.
Their fixed sources, selection records and receipts remain in the private
package; these readers are closed and must not be repeated.

## Learning and validation

Late replay expert CE rises .560514→.572009, while guided online last50 rises
67.78→74.68. Actual overrides are 85,588/200,000 (**42.794%**), versus 43.0585%
in the control. The improved guided online curve does not establish better
unaided play. The existing comparison figure is visually checked; its frozen
annotations correctly show the regression in both modes.

All **649 pins**, 100k recovery, 357 matching initial/invariant frozen tensors,
47 pre-update transitions, 800/795 matching initial frozen semantic events,
exact cadence, finite reports and changed CNN/reward/actor/value pass. Scheduled
behavior total reconstructs on every report, with maximum absolute error
3.276e-5. Raw world/actor imitation CE differs by at most 7e-7. Full training
pixel/action/label replay, frozen action/outcome/frame/boundary replays and
videos, host guards, curves, exposure and comparison pass. Ordinary frozen
logs do not contain live pixel hashes; no unavailable image equality is claimed.

The original main/postprocessing/collection services succeed **11:28:15.248 /
11:28:35.619 /11:28:40.720 UTC November3**, source
`18d493e6271e32036790a69b3f3b78fa928d1291`. All five original services succeed.
Independent terminal reading completes once **11:29:04.321**, taking 1.488s
CPU /1.512s wrapper, with no native agent or emulator constructed.

No full-run stage fails; no GPU stage, original collector or completed reader
repeats. The earlier packaged Boolean-message fixture failure and full-preflight
converter-import failure remain preserved with their CPU repairs. A pre-test
F32 fixture correction is also retained. These preparation issues do not change
the successful original full-service outcomes.

CPU co-host costs remain uncorrected: startup1.575s, terminal-reader qualification
2.524s, prefix selection1.334s, tensor selection3.519s, recovery confirmation
1.375s, and trace/tensor analysis wrappers4.442s/3.255s. The latter two overlap;
do not sum them as an exclusive interval or subtract them from training time.

Root `/x/Temp/kindle-actor-only-decay-20261103/full` is closed. Keep the qualified
optional implementation default off; this tested schedule is not selected for
further learning. No extension, sweep, confirmation learner or automatic
successor is selected. No Kindle allocation remains active at closure; check
live state before further GPU work. Breakout and the five-game goal remain
unfinished. The result rejects this schedule at this budget; it does not prove
that every actor-imitation schedule would fail or identify the remaining
limitation of imagined-policy learning.

| Artifact | SHA256 |
| --- | --- |
| Learning review | `bc7069f81fab3e98ff7cdf52159e417770e42124c16355b9c0867ff738702b9d` |
| Completion | `08a2da760b0c9502bca780f36568cbbe2c4c9aa82d76ad57d9d05c5301d98244` |
| Independent terminal summary | `abced6e3190e8b21b1d9bf01cd12fc2acb45437aee45f5ad872ef462ffa234b9` |
| Pre-fade trace comparison | `550f0fb18982c0074638c46901c941caff21d4ae39a06bc682e985ff5aeaab3e` |
| Pre-fade tensor comparison | `c11f24696b3fc9317f0bf41d439d5c1d7e9f6e8993577d8710933e6324ca904e` |
