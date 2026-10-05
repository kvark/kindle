# Existing-embedding exploration target: qualified

The four action-conditioned predictors now regress the existing detached CNN
encoding on arrival. No new encoder or target-selection compatibility layer.
For Size1M, output width128→256 adds33,280 parameters; natural target scale also
changes. The world objective, actor, coefficient and action protocol do not.

Local114 Rust/1,137 Python tests, formatting and strict workspace/Python Clippy
pass. Five guarded GPU checks pass: full-graph world/encoder detachment, independent
F64 bonus/raw derivatives, repeat-versus-novel learning, real/imagined alignment
and exact zero-scale baseline updates/RNG.

The first detachment check failed by assuming all dead parameters lacked gradient
buffers. Meganeura emits scalar-zero sentinels; an unused scalar continuation
bias can have such a buffer. The corrected test checks every non-ensemble
autodiff result is constant zero and reads any scalar GPU gradient as exactly
zero. Non-scalar encoder/RSSM parameters have no gradient buffers. This changes
the diagnostic only, not the learner or backend. Failure logs remain retained.

The excluded seed9001 production smoke completes2,048 actions/451 updates in
12.36s, mean update25.32ms. Frozen restore adds1,024 actions/zero updates and
preserves all346 model/optimizer tensors. Seven successful guards/seals, no new
allocation warnings, NVML polling or recovery. These checks are not learning or
matched timing evidence. Upstream recheck still finds only non-runtime changes
over Meganeura592a2f5a; Blade main remains e349cddf.

Proceed with the [declared three fresh seeds](../experiments/2026-10-05-freeway-embedding-disagreement.md),
32,768 actions/8,131 updates each, against retained soft/extrinsic controls.
No automatic longer run. [Compact qualification](2026-10-05-freeway-embedding-qualification.json).
