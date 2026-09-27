# Three-seed fast development screen

**Phase 1b's workflow completes in 8 minutes 18 seconds**, including the serial
guard/controller overhead. Each independent Size1M learner completes **32,768
actions / 8,135 updates in 2 minutes 44 seconds**. This establishes a practical
iteration loop, **not good gameplay or a learning advantage**.

[Complete configuration, seed curves, action/time summaries and uncertainty](2026-09-27-minatar-screen.json)
are self-contained. [Reusable recipe and adapter](../screening.md).

| Learner seed | First score at 2,048 actions | Final score at 32,768 | Complete episodes | Run time |
| --- | ---: | ---: | ---: | ---: |
| 1009 | 0.32 | 0.44 | 2,796 | 163.68 s |
| 2017 | 0.54 | 0.40 | 2,807 | 163.70 s |
| 3019 | 0.52 | 0.68 | 2,828 | 163.75 s |

Scores are rolling means of the last 50 completed **online training** episodes,
not frozen evaluations. Final equal-seed mean is **0.507**, pointwise 95%
bootstrap CI **[0.400, 0.680]**, versus **0.460 [0.320, 0.540]** at the first
measurement. The paired early-to-final change is **+0.047**, 95% seed-bootstrap
CI **[−0.140, +0.160]**. Resampling uses the three independent learners, never
their episodes/streams. This small experiment does **not establish a reliable score
improvement**. Full curves retain fluctuations and the uncompleted episode
tails; no best checkpoint or favorable window is selected. No untrained or
alternative-method control was run, so do not claim a method gain.

## Exact recipe and boundary

MinAtar **Breakout**, version 1.0.15, all six actions, action repeat one, default
0.1 sticky actions and difficulty ramping. Unmodified sparse game rewards;
**no pretraining, exploration override, intrinsic/shaped reward or privileged
signal**. MinAtar's public binary channels include game-object information;
this is a simplified benchmark, not RGB ALE or Atari human-normalized scores.

Size1M preset, **828,965 actual trainable parameters** (world 712,928; behavior
116,037), eight independent streams, B8/T16/full BPTT/M8/H15/R32, replay 16,384,
F32, prediction-only scale .25, default 4e−5 learning rate / 1,000-update warmup
and LaProp/AGC. All scheduled updates are drained; final learner debt is zero.
Configuration and all environment seeds are in the JSON.

The 10×10×4 public observation is packed **losslessly** into 5×5×16, zero-padded
to 7×7×64, then encoded by the jointly learned observation encoder. Every input
location/channel has an inverse-tested mapping. No frozen LeVJEPA, resize,
averaging or random projection. Pixel-agent speed tests still use their unchanged
pretrained Tiny frontend; this new recipe is a separate screening workflow.

The MinAtar simulator and packing run on CPU. Batched belief/policy, device
replay and learning run on **RTX 5080 / driver 580.178.04**, using Meganeura
`367e53d4` / Blade `7cca6377`. This is the explicitly allowed environment
fallback while CUDA/JAX-to-Vulkan interop remains separate work, **not a
GPU-resident game or a CPU learner**. R32/Size1M/MinAtar timings must not be
presented as an optimization gain over R256/12M/Atari.

All three guards pass, children exit zero and are reaped, no NVML polling or
recovery occurs. Sampled Vulkan estimated budget headroom stays above
15,892,021,248 bytes; this is neither physical free memory nor peak VRAM.
Every final checkpoint's recorded hashes and all **241 tensors / 146 moments**
reverify finite. A separate Tiny plumbing smoke completed 1,024 actions / 199
updates; it is excluded from the three-seed statistics.

The workflow is now ready for controlled representation/exploration experiments.
The separate [12M speed work](2026-09-27-fused-learner.md) reaches **1.15×**, not
the strategy's **3× target**; completing this screen does not erase that miss.
Raw runs, checkpoints and serial controller:
[local evidence](../../runs/minatar-screen-20260927.9pnoFC/).
