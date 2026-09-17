# Pong confirmation on the adopted block runtime

Status: root **2017 training is running**, launched at 22:07 UTC on September 16.
No frozen score or new competence result exists yet. The direct native child is
131029, owned by the existing host-only guard. NVML is disabled.

## Declaration and fixed recipe

The [new declaration](../../runs/pong-block-confirmation-20260916.rBwdGF/declaration.md)
follows the completed [source/episode integration](2026-09-16-block-runtime-adoption.md).
It passes nine reader checks and binds 5,329 inputs. It uses the immutable
`8dc0b98` adapters, actual native `886bae68`, Meganeura `589d73ab` and shared
Blade `2accfeee`, with LeVJEPA on driver 580.178.04. The 22:07 upstream check
still finds `986f49a` / `92553493`; no backend is switched within the campaign.
Main source adoption is commit `d10cb2f`, with 702 passing Python tests and the
complete native qualification reused by exact source/binary identity.

Root order is **2017, 3019, 1009**, all fresh on this one bundle. This opens the
previously untested roots first, then repeats 1009 for a matched three-root result.
The older successful root 1009 remains separate historical evidence. The stopped
xPz5ud queue and its output hold remain untouched.

Each root retains the original four phases: 400,008 unassisted training actions,
final-checkpoint frozen evaluation, fresh six-action untrained initialization,
then restored untrained evaluation. Both evaluations sample actions on environment
root 100000, finish four episodes in every stream, and have a hard 600,000-action
cap. No frozen learning or exploration override is allowed. The unchanged recipe
is N6/R256/B16/T64/full-BPTT64/world-micro16, 12M/F32, LR 4e-5, warmup 1000,
AGC .3, reconstruction zero/future .25 and extrinsic rewards only.

All three trained policies must meet the original Pong gate: >=20 natural games,
>=90% wins, mean >=+15 and no cutoffs. Each untrained control must fail that gate,
and its trained mean must be higher. Retain every completed episode, faster-stream
extra and unfinished tail. A competence failure is not a retry or budget extension.

## Runtime and outcomes

The [training log](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-train.jsonl)
and [native stdout](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-training/child.stdout)
are live. Training has an 18-hour hard bound; the checkpoint does not preserve
replay and live belief for an equivalent interrupted resume.

Native Vulkan budget-minus-usage must remain >=2 GiB after every GPU stage.
This is an estimate, not physical free or peak memory. Kernel/boot/driver checks
run once per second without NVML; utilization and recovery action are unmeasured.
All phases are individually launched/reviewed. There is no automatic follower,
run-all, recovery operation or subsequent game declaration.

After each root's four GPU phases, the declared CPU analysis replays both entire
frozen logs, verifies every action/reward/boundary/reset/frame count and saves
whole stream-zero videos, including partial tails. The first four complete trained
stream-zero matches are preselected for later H1 world diagnostics, without score
filtering. Extraction is not GPU forecast validation. The roadmap remains the
single current game-status table and video index; five-game reliability is not
established by this campaign declaration or by short runtime checks.

## First-quarter online learning — September 17

Root 2017 reaches the 100,020-action save at about 00:51 UTC. The first five
fixed 20,004-action blocks show the following **online training** results from
the linked raw log. Counts include every match ending in each block, including
matches that began earlier; none ends by cutoff.

| Block ends at action | Completed matches | Natural wins | Mean return |
| --- | --- | --- | --- |
| 20,004 | 19 | 0 | -20.6842 |
| 40,008 | 20 | 0 | -20.6000 |
| 60,012 | 12 | 0 | -18.9167 |
| 80,016 | 9 | 2 | -10.4444 |
| 100,020 | 11 | 11 | +15.2727 |

All 24,655 learner reports through that save contain finite values. These matches
span an updating policy, not a frozen final policy or an independent control.
The training budget remains 400,008 actions; final evaluation, the untrained
control and the other fresh roots remain pending. No competence gate, source,
recipe or declaration changes because of this interim improvement.

## Session handoff — September 17, 03:36 UTC

The session ends with root 2017 still running at approximately 200,000 of
400,008 actions and 10.18 actions/s. Controller 130967 and direct native child
131029 remain alive; no terminal training result exists yet. Leave this guarded,
bounded invocation uninterrupted. NVML remains disabled, and no evaluation or
successor launches automatically.

The last separately checked checkpoint was at 180,036 actions / 44,659 updates:
all learner reports and complete state/moments were finite. Its preceding
20,004-action block contained 12 natural wins in 12 completed matches, mean
+20.0833, with no cutoffs. This remains online learning evidence, not frozen
competence. Checkpoints overwrite the same path; that intermediate state was
checked in place, not archived as a frozen-evaluation artifact.

On resumption, inspect the existing process and terminal results before any
launch. After training completes, verify the declaration and audit its result:

```sh
python/.venv/bin/python -B runs/pong-block-confirmation-20260916.rBwdGF/run.py verify
python/.venv/bin/python -B runs/pong-block-confirmation-20260916.rBwdGF/run.py audit seed2017-training
```

Only after both pass and the guard/native result is reviewed, individually
launch `run.py run seed2017-evaluation`, followed by the declared initial and
untrained phases with the same review boundaries. Do not rerun training or
`prepare`. Roots 3019 and fresh 1009 remain pending; the five-game reliability
goal is not complete. Breakout action-width gates and later checkpoint-history /
Freeway/Qbert dose experiments remain separate, after the declared Pong work.
