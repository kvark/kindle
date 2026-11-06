# Guarded native qualification of categorical regrouping

Select two serial native fixtures after [CPU and package qualification](../results/2026-11-06-cdp-latent-shape.md):
202 Kindle/5 adapter tests,36 focused Python tests, strict workspace/binding
Clippy, and1,139 installed-package tests/10 deselected with zero construction
attempts. Preserve the initial Python import-path failure and corrected run.
No Atari learner, bounded cost check or full comparison is selected here.

Use package source536012d59fd25e7e3e95ac3b8d2fddbfd0390e60 under
`/x/Temp/kindle-latent-shape-20261106`. Native SHA256
`2c9c6e6f40bbcaf33b0c982d09e3a130057894acbe2115f54acd77d0d0a2e8bd`;
test executable SHA256
`73a18277762ebf9f483da4c86acf7dabe5b55baacf9973783bea4f93b6e67e99`.
The receipt pins246 source files. No overlay or rebuild is selected.

On idle AMD RX7900XT/RADV with Auto and the existing host guard:

1. `latent_shape_native_categorical_values_derivatives_and_samples`: two states
   at32×16 and64×8; unimix.01, free nats1. The first state has identical prior/
   posterior and is clamped; the second has active KL. Compare independent F64
   mixed-softmax KL loss/raw values (absolute tolerance2e-4) and all2,048 parameter
   derivatives (2e-5) in each of three routes: both active, posterior stopped,
   prior stopped. Check exact supplied-draw one-hot samples, unchanged parameters
   and zero optimizer steps. Six forward/backward executions; no learner.
2. `latent_shape_native_learning_sharing_and_restore`: Tiny8×2/RGB32/B2/T4/H3,
   two synthetic streams with16 transitions each and passive labels. Exactly two
   ordinary updates must have finite policy/imitation losses and change encoder,
   reward, actor and critic parameters. Check all11 actor parameters in scalar/
   vector acting and imagination, frozen world actor output sharing, exact live
   one-hot groups, and exact checkpoint configuration/weights/moments/counters.
   Restored replay is empty. Synthetic observations and parameter names were
   checked on CPU before this selection. Zero ALE instances or Atari actions.

Each original job has120s, controller300s. Record binary/source identity, immutable
queue/declaration, stdout, execution receipt and both guard audits. All246 source
pins and allocation inputs must stay unchanged. Require both jobs to succeed.
Stop on actual GPU faults, numerical errors or assertions; preserve failures and
do not automatically repeat. No reset/reload/reboot or old collector is involved.

These finite fixtures cannot demonstrate gameplay benefit or broad equivalence.
A separate8,192-action protocol must measure cost, replay and restore before
considering a fresh full learner. Keep retained143.667 greedy/70.750 sampled and
the original864/90%-two-wall/five-game gates unfinished.
