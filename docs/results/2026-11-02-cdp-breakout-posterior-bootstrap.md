# Posterior replay bootstraps pass execution and early cost checks

The optional observed-state bootstrap completes **8,192 actions / 32,726 frames /
489 updates in46.900s**. Its conservative200k projection is **20m59.510s**,
below the30-minute training cap. The following1,024 frozen actions take.879s,
with zero updates and no natural completed episodes. This establishes execution
and early cost; no gameplay benefit is established. The separately selected
[fresh200k comparison](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap-200k.md)
will measure frozen gameplay without reusing this bounded experience.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-posterior-bootstrap.md)
tests a specific value-target hypothesis after the [calibration diagnostic](2026-11-02-cdp-breakout-value-calibration.md).
Real-replay lambda targets now optionally bootstrap with values at observed
posterior states. The default still uses their first imagined returns. Real
rewards, intrinsic bonuses, episode boundaries, discount, lambda, replay weights,
slow-value regularization and imagined actor/critic targets stay unchanged.
The revised replay targets feed both the critic and world representation loss.
The change does not establish that imagined bootstraps caused the prior error.

## Implementation and package

Native `replay_value_bootstrap` accepts `imagined` or `posterior`. Old metadata
defaults to `imagined`; parameter names/shapes and checkpoint format remain
unchanged. The fresh-only `--replay-value-bootstrap` option records the choice,
rejects restore/evaluation overrides and rejects older native runtimes that lack
support. No dependency, backend optimization, precision or teacher change occurs.

Continuing, terminal and timeout target fixtures pass for both modes. Config
default/roundtrip/rejection and actual CLI forwarding/capability checks pass.
The package passes **145 Kindle / 5 Gym Rust CPU tests**, both strict Clippy
checks and **628 packaged Python checks** with zero native construction attempts.
The resident-target shader compiles offline; this is not execution validation
of that path. `KINDLE_DEVICE_TARGETS` remains unavailable on this AMD host.
Current learning uses host targets and Auto precision. Loss derivatives are
unchanged; no unrelated gradient attribution is repeated.

Package source is **11373c17d9254924f87014ffd02fd8a4e2a091ca**, native SHA256
`f1cf431721daa1bb35c2fc14f33feec6774c949fa35fac2e74c56deaebe47ee8`.
Runtime `/x/Temp/kindle-replay-bootstrap-20261102/runtime`; runner SHA8952f81f.
Packaging completes08:59:43.938 UTC November2. Wheel, native, validation
executable and source receipts remain; the2.989GiB disposable build cache is
released after the bounded check. The immutable build receipt records its
original cache state; separate `cache-release.json` records removal.

## Default compatibility and fresh learning

Before training, the new runtime restores the retained replay-value1 model and
reproduces all**128** semantic events in its first**1,024 sampled actions /
4,096 frames**, in**.889s** with zero updates. All**357** frozen tensors remain
identical; config differs only by the explicit new default field. All28 pins
and the host guard pass. This is bounded acting/restore compatibility, not a
new evaluation score or a default-learning equivalence claim.

Compatibility service `kindle-replay-bootstrap-compat-20261102`, original
PID927398/invocation49922050132944f3aa312194fe7cc765, launches09:02:51.043 and
succeeds09:02:57 UTC. Its saved review completes09:02:57.958. It is terminal;
do not repeat the capture.

The fresh candidate retains12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled collection,
learner1009 and the retained RGB-only teacher. Full18/sticky.25/repeat4/noops0,
native max-pooled frames and one GPU64 resize remain. Native and runner differ
from control447fe50/cf4ea238, so do not claim an isolated causal effect.

All**373 pins**, **357 matching initial tensors**, **47 pre-update semantic
events**, **489 finite reports**, complete guided pixel/action/label replay,
frozen tensors and host guards pass. CNN, reward, actor and value parameters
change. Actual guidance overrides3,811 actions; all8,192 receive labels.
Guided returns are not unaided competence. The bounded experience is not reused.

Main `kindle-breakout-posterior-bootstrap-check-20261102`, original
PID927572/invocationff26aa4f6e07412ab2254b7549a12080, launches
**09:06:55.856 UTC November2/source11373c1** and succeeds**09:08:00.345**.
Original collector PID927575/invocation19991e85029a4d70a5a2fcb728ff764e
succeeds**09:08:08.088**. The read-only terminal reader completes once
**09:08:31.094** in5.585s. All original stages succeed with no preparation,
build, GPU, replay or collector failure and no successful-stage repeat.

Root `/x/Temp/kindle-replay-bootstrap-20261102` retains package, compatibility
and bounded evidence. Do not repeat qualification, captures, learner, replay,
collector or terminal reader. No GPU allocation is active at closure; check
live state before separately selecting the fresh200k comparison. No extension,
cap increase, sweep or automatic successor is selected. The five-game goal
remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Package | `782e4895134086e15a422eab509d616e80189a076840b0f13b2e201fc29e668c` |
| Default compatibility review | `f0461a4e83ed06df5e1603e8093bd9103810a48ad0594bc5926406bbc28a41d3` |
| Bounded declaration | `c712da89c139a1032e06aa0c5b6d20719538e7d06c90a266e062d54641589050` |
| Original completion | `efbe8f3b78b3902cf84d33749310b9c57ce3112db35aabe4a8b886e3bfa0301e` |
| Terminal summary | `efc898c83bfa1f6719a64e9382bbd1f7629b61fb6d48caf37854a2abe043d235` |
