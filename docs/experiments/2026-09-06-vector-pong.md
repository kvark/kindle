# Vectorized native LeVJEPA Pong

Declared 2026-09-06, before this experiment's control or training runs. This is
a fresh vector-collection protocol, not continuation of the interrupted serial
[mastery experiment](2026-09-06-levjepa-pong.md). Its stronger competence gate is
retained. No vectorized Pong mastery result is available at declaration.

## Fixed protocol

- One native F32 LeVJEPA frontend, one shared Dreamer world model/policy/learner,
  and independent ALE streams. Per-stream visual cache, belief, RNG and replay
  history remain separate; there are no independent learners or swarm services.
- Train three fresh models with seeds **0, 1, 2**, each for **200,000 aggregate
  executed actions**. A vector tick contributes N actions. No restored training
  state, forced-random coverage, demonstrations, intrinsic reward or scripted
  gameplay. Use explicit, unshaped Pong point rewards only.
- Keep the prediction-only 12M recipe: B16, T=BPTT64, row microbatch 16, replay
  ratio 256, replay capacity 100,000, world/behavior LR 4e-5, warmup 1,000 updates,
  AGC 0.3, reconstruction 0, future prediction 0.25. No B32/B64 cadence change.
  All other parameters are recorded and must match across training seeds.
- Select N once, before training, from the fixed-B16 N=2/4/8 throughput matrix
  in `runs/vector-20260906-temporal/`. Among complete, accounting-valid jobs that
  retain the GPU memory guard, choose the fewest streams within 3% of the fastest
  measured final-1,024-action window. This is a throughput choice, not selection
  by game score. Keep the same N for all three training seeds.
- Environment seeds are `(model_seed + stream * 1_000_003) mod 2^32`; live policy
  and posterior RNG seeds follow the recorded native rule, `config.seed + stream`
  modulo 2^64. Independently initialized games may die and reset naturally; no
  cloning/rewinding of a running stream is used.
- Use `ALE/Pong-v5`, ALE 0.12.1, the published wrapper, all 18 actions, repeat 4,
  zero reset no-ops/sticky actions, and a 100,000-frame episode cap. Consume each
  terminal observation before resetting its stream. Count actual emulator frames.
- Save rolling checkpoints every 20,000 aggregate actions for diagnosis/recovery.
  Only the declared 200,000-action final save is scored. A restored model lacks
  exact replay/RNG/environment/live-state continuation and cannot silently
  complete this uninterrupted protocol.
- Evaluate each final model for **75,000 frozen sampled actions in N=1**, with
  fresh visual history/belief and the corresponding environment seed. Exactly
  zero learner updates; no greedy fallback or best-checkpoint selection.
- Run a fresh seed 0, N=1, **20,000-action zero-update control** using the same
  executable/frontend/recipe except replay ratio 0. The earlier serial control
  remains contextual evidence, not this new executable's matched control.

## Unchanged mastery gate

For **every seed's final frozen evaluation**, require at least **20 complete
natural games**, mean return **at least +15**, and **at least 90% natural wins**.
A win has positive return, terminated=true and truncated=false. Mean return
and win fraction include **all completed games**, including timeouts; timeouts
are not wins. Report unfinished tails and timeouts separately. If any seed
fails, report failure and design a new experiment without lowering these gates.

`python/examples/audit_pong.py --vector-envs N` applies these gates on top of
the strict vector action/reward/replay/warmup/update ledger. It verifies fresh
training, frozen evaluation, source/frontend identity, the exact final checkpoint
hashes and finite tensors. The serial audit remains a separate supported mode.
Interrupted jobs stop the launcher; they must not quietly advance to the next
seed or be counted as complete.

## Implementation and measurement

The frontend remains `galilai-group/LeVJEPA-VideoMix-Large`, revision
`e831a0347737fcaa660b39c57d41c109de399845`, weight SHA-256
`da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d`, encoding
`levjepa-large-f32-chunk16-letterbox224-jl64-pool2-v1`. Its CC-BY-NC-4.0 weights
are separate from Kindle's license. This is not a DINO run.

Replay encoding and independent prediction/reward/value heads are now batched
across time. RSSM/posterior recurrence and full BPTT remain sequential. Numerical
checks and rejected candidates are recorded in the
[throughput report](2026-09-06-vectorization.md); this change is not bit-identical
to serial-head training. Meganeura remains pinned to 35a410c and Blade to b208f3b.

Artifacts go in `runs/levjepa-vector-pong-20260906/`; checkpoint targets use
`checkpoints/levjepa-vector-pong-{untrained,seed0,seed1,seed2}-20260906`.
Before launch, record selected N, source commit, native/runner/wrapper hashes
and the exact launch command. The launcher checks identities before every job.
Record construction separately, actual aggregate and per-stream simulated/wall
ratios, learner-stage time, debt, and a 1 Hz trace of GPU UUID
`GPU-6869e50d-83aa-bec7-6169-adc413f49b32` (RTX 5080). GPU-heavy jobs are serialized.
High busy percentage or fast frozen inference does not establish accelerated
playing plus training or game competence.
