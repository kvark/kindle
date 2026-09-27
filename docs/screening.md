# Fast learning screens

Use `python/examples/screen_minatar.py` for early method experiments. This is
MinAtar, **not Atari/ALE**; scores do not share Atari's scale or mastery gates.
The first recipe is Breakout, then Freeway for exploration. Both use the public
game observations, unmodified sparse rewards, all six actions, 0.1 sticky-action
probability and default difficulty ramping. No exploration override, video
pretraining, intrinsic reward or privileged state is added.

The default run uses Dreamer **Size1M**, eight independent streams, B8/T16/full
BPTT/M8/H15, R32, replay capacity 16,384 and 32,768 aggregate actions. Use learner
seeds **1009, 2017 and 3019**, separately; drain all scheduled updates. R32 is an
explicit screening recipe, **not** a speed comparison against the R256 Atari
reference. Run each seed under `gpu_host_guard.py` in a bounded persistent user
service, with no concurrent heavy GPU work. The budget must finish within one
hour; report construction time and unfinished episodes too.

Install the `screening` Python extra. The runner takes an output JSONL path,
`--seed`, optional `--checkpoint`, and defaults to the recipe above. Native
bindings expose `FeatureVectorAgent`, using the same `begin_episodes`, `act`,
`observe`, `learn_scheduled` lifecycle as pixel collection. Terminal observations
are consumed before individual resets; streams never share recurrent histories.

## Observation and GPU boundary

MinAtar's 10×10×C public binary image is packed losslessly into 5×5×4C with
2×2 space-to-depth, then zero-padded to the existing 7×7×64 contract. Unit tests
invert the transform and recover every spatial/channel value exactly. There is
no resize, averaging, RGB rendering or projection. The existing observation
encoder is trained jointly with the RSSM. This separate frontend does not
change the frozen Tiny encoder used by the 12M speed comparison.

MinAtar stepping and observation packing are CPU work; features are uploaded.
Batched belief/policy, categorical sampling, replay collection and learning
remain on Meganeura/Blade. This is the strategy's explicit temporary CPU-env
fallback, **not** a GPU-resident environment or a CPU learner.

[Craftax](https://github.com/MichaelTMatthews/Craftax) is a better long-term
open-ended GPU environment. Its JAX arrays can be exchanged using
[DLPack](https://docs.jax.dev/en/latest/jax.dlpack.html), but that is not a direct
match for our external Vulkan memory/semaphore ownership API. The current
Blade path accepts externally allocated Vulkan resources and synchronization,
not arbitrary CUDA device pointers. A CUDA/Vulkan allocation and synchronization
bridge would be additional engineering, not a thin adapter; this is why the
first screen uses the permitted MinAtar fallback. Track GPU-native Craftax and
[CuLE](https://github.com/NVlabs/cule) separately before claiming TASK.md's full
GPU environment pipeline.

## Reusable curve format

`kindle-screening-v1` records exact config, actual model size/device, seed rules,
aids, native package hash, each completed episode, periodic learning metrics,
actions and elapsed wall time. The score curve is each learner seed's rolling
mean of its last 50 completed **online training** episodes. Unfinished tails
remain in the final record. This is not a frozen-policy evaluation.

`kindle._screening.summarize_curves` aligns the exact action grid and a common
wall-time interval (linear interpolation, no extrapolation), then computes mean
and 95% percentile bootstrap intervals across independent **learner seeds**.
Three-seed intervals are coarse; episodes and streams are not extra independent
replicates. Commit the full seed-level curves and summary in `docs/results/`.
For method comparisons, retain this recipe and change one factor at a time.
