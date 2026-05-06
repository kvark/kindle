# ARC-AGI-3 generalization plan (2026-05-06)

## Where we are

Best joint-training recipe (`arc_agi3_multi.py`, 25-game lanes, 20k steps):
```
--use-ppo 1 --entropy-beta 0.01 --recon-loss-coef 10 --recon-visual-target 1
--goal-bonus 10 --latent-dim 256 --hidden-dim 256
--adam-eps 1e-4 --grad-clip-norm 0.5
```
Result: 10/25 games hit ≥1 level event, 11 total events, **cd82 advanced to level 2 (max_lvl=2)** — the first level-2 progression on the joint-training task. Prior runs all topped out at level 1.

Open structural problems holding kindle back from a *general* ARC policy:
1. **Per-game centroid dominance**: encoder's between-game variance is ~15× the within-game state variance. The policy gradient sees "which game" first and "what state" only via faint within-game variation.
2. **Sparse reward**: ≈ 11 events / 500k env-steps. Policy can't commit on signal that rare.
3. **No real transfer signal**: the 20-train + 5-val run produced 3 val hits — same rate as random discovery on those games, no clear "trained policy is better than random" signal.
4. **Hyperparameter sweeps exhausted**: 13 distinct configs tried (8 in the 8h sweep + K+goal-bonus + K+options + train/val + a few earlier). All cap at the same general region. Architecture/representation has to change.

## Plan: 5 work items, ordered by likelihood × cost

### 1. Long horizon on best recipe
**Hypothesis**: cd82→L2 happened at step 20k. The policy may keep climbing (more games to L1, some reaching L2) if we run longer.
**Action**: 100k–200k step run with K+goal-bonus recipe. Save checkpoints every 50k for offline analysis.
**Cost**: 3.5–7 hours of GPU per run.
**Status**: starting now (item 1 of 5).
**Decision rule**: if at 100k we see ≥3 games at L2 OR ≥2 games with ≥3 events, the recipe is working — push to 200k. Otherwise plateau confirmed → move to items 2–4.

### 2. Curriculum: subset corpus → full corpus
**Hypothesis**: 15 of 25 games never produce events; their lanes contribute pure-zero advantage and dilute the policy gradient. Train on the unlockable subset first; transfer the trained encoder/policy to the full corpus.
**Action**: 50k steps on `--game-prefixes "cd82,sp80,ft09,lp85,ls20,m0r0,r11l,sk48,tr87,vc33"` (the 10 unlockable games), checkpoint, then 50k more on all 25 with `--load-state` from the checkpoint.
**Cost**: ~50 min subset run + ~70 min full-corpus run = ~2 hours of GPU.
**Decision rule**: if subset training pushes ≥3 games to L2 in 50k steps, the curriculum hypothesis is right. The "expand to 25" stage tests transfer.

### 3. Pre-train encoder as autoencoder, then RL-finetune
**Hypothesis**: kindle's encoder collapses to per-game centroids because the WM forward-prediction loss admits trivial low-rank z. A pure pre-training pass with a reconstruction loss (and no RL gradients) lets the encoder discover state-distinguishing features without per-game-centroid shortcuts.
**Action** (substantial — 1–2 days):
  a. Add an offline encoder-pretraining loop in kindle: feed visual frames, compute z, decode to obs/visual, MSE backward. No policy/value/WM losses.
  b. After ~M frames of pretraining, save checkpoint.
  c. Load the pretrained encoder weights into a fresh K+goal-bonus run; freeze (or slow-tune at 0.1× LR) the encoder.
**Cost**: design + implementation ~1 day; pretraining run ~few hours; RL fine-tune ~few hours.
**Decision rule**: encoder PCA top-2 should hold a noticeably smaller fraction of variance after pretraining vs after WM-only training.

### 4. Options with real capacity
**Hypothesis**: the options=8 + per-option-bias attempt failed because per-option-bias is just a single bias vector per option — almost no expressive power. With per-option-fc2 (full `[hidden, action]` matrix per option) and num_options=4 (so each option gets ~6 lanes on average), options become real sub-policies.
**Action**: K+goal-bonus + `--num-options 4 --per-option-heads 1` at 50k steps.
**Cost**: ~70 min GPU.
**Decision rule**: if any option specializes to a game-cluster (visible in option-selection statistics), and per-option breadth is comparable to single-policy, the architecture works. Then increase num_options and steps.

### 5. Proper train/val with middle-difficulty holdout
**Hypothesis**: prior train/val test held out {ar25, cd82, ft09, sp80, sk48} — a mix of "historically easy" + "historically hard" games. The 3 val hits were on the easy set (random discovery rate). We can't detect transfer without middle-difficulty games where the random rate is in (0, 1).
**Action**: identify games that random reaches occasionally but not always (look for games with eps_to_event between, say, 5k and 15k from prior data). Hold out 5 of those; train on the remaining 20; measure per-game val event rate vs train rate.
**Cost**: requires a baseline pass to characterize per-game random discovery rate (~30 min), then a train/val run (~70 min).
**Decision rule**: if val rate ≈ train rate on the held-out middle-difficulty games, generalization confirmed.

## Cross-cutting infrastructure (already done 2026-05-06)
- `Agent.save_state(dir)` / `Agent.load_state(dir)` — checkpoint API used by items 2 & 3
- `BatchAgent.latents()` + latent_probe.py — measures per-game centroid dominance
- `--val-prefixes` + `--val-steps` train/val split — used by items 2 & 5
- `--goal-bonus α` extrinsic pulse — best result so far
- PPO + L1 options compatibility — needed by item 4
- arc_agi3_multi.py multi-game harness — common base for items 1, 2, 4, 5

## Decision tree

```
Run item 1 (long horizon)
├── ≥ 3 games at L2 OR ≥ 2 with ≥3 events
│   → item 5 (validate generalization)
│   → if val rate ≈ train rate → publish, move to harder ARC sets
│   → else → items 2 + 4 to push further
└── plateau by 100k
    → items 2 + 4 in parallel (cheap)
    └── if neither breaks plateau
        → item 3 (encoder pretraining) — the deeper structural fix
```

## Notes
- Save_state captures only trainable weights — episode state (lane buffers, step counter) does NOT survive load. Acceptable for items 2 & 3 (we don't need episode continuity across pretraining/RL boundary).
- All experiments to date have used 25 lanes × 1 env each. Multi-env-per-game (more lanes per game) is another axis we haven't explored — could amplify learning signal on rare events.
- The kindle reward circuit fundamentally cannot represent "game won". Goal-bonus α=10 was the simplest extrinsic injection. A more principled path is to surface the env's win/loss/reward as a normal extrinsic — already supported via `set_extrinsic_reward`.
