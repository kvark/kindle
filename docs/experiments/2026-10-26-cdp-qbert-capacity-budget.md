# Qbert capacity and replay batch at200k actions

**Selected after the bounded12M/RGB48/B16 check.** It passes16,384 actions/
2,002 updates, ordinary restore and all346-tensor/guard checks in149.387s of
training. It does not establish competence. A rough scaling from the completed
4M control suggests about31 minutes for200k; this is not a measured full-budget
cost or a learning prediction. Declare a35-minute cap before starting.

Fresh1009/eval4000001009, centered CDP12M/RGB48/Auto,14,123,473 parameters,
N8/B16/T16/H15/R32, full BPTT/microbatch16, replay100k, intrinsic coefficient1,
actor_unimix0, full18/sticky.25/repeat4/noops0 and100000-frame episode limit.
Native max-pooled frames receive one GPU resize; no action aid, RAM input or
external reward shaping. Native`bfb8c31`, runner`1ed07af`, replay`e853818` unchanged.

Keep200k aggregate actions,24,954 expected updates and first376. Save100k recovery
at12,454 updates and the fixed200k final. Relative to the completed4M/RGB48/B8/
R32/200k control, only model_size4M→12M and batch/microbatch8→16 change in native
config. This jointly changes capacity, initial tensors, optimizer cadence,
centering and warmup per action. Sampled positions6,388,224 versus6,392,192 are
nearly equal, while optimizer steps24,954 versus49,939 are not. No isolated
capacity or batch effect, optimizer rescaling or matched-initial claim across sizes.

Save own actual initial and verify equality only against the same-shape bounded
check's initial. This is a fresh life with new replay/games/belief/RNG, not a
checkpoint extension. Both seeds are reused development data. The validation
life's cost is additional and it is not an independent seed.

Evaluate fixed final and own actual initial for the first three natural sampled
episodes per lane,24 per arm, under600k-action/300s caps. Require zero updates/
truncations and retain excess episodes/partial tails. Keep the original mean15000
and90%-first-pyramid gate with at least20 natural episodes; cube coverage is a
diagnostic, not a lowered target. Audit all reports, config/source/counters,
finite checkpoints,346 unchanged frozen tensors, full CPU replays/videos and
three guards. Generate curves, coverage and4M/B8-versus12M/B16 comparison.

Training cap2100s includes construction; audit1200s and service3960s. Retain any
cutoff explicitly; no cap increase, resume, checkpoint selection, extension or
automatic next queue. The recovery monitor already passes against completed B16
checkpoints and the full Pong trial; its shape/config checks admit this recipe.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-12m-48-b16-r32-200k`;
controller:`run_qbert_12m_48_b16_200k.py`; intended owner
`kindle-cheap-qbert-12m-48-b16-200k-20261026`. Complete queue normalization,
compilation and unselected refusal pass. Prepared declaration SHA256
`231561532523991eb9101f0247456f11e37d4f1ea04e0473fc320501261a6fd5`;
controller`2ac94ec4f411f3a60343ea0de8d1cc59c5cc75becfda467feedd4a5f436b512d`;
comparison`6e044784a7f32df0aa20f4bbfe9bf53715666b1e994f20ea39fb6c362110a418`.

Started **10:41:58 UTC**, source`be6c942`, on the same healthy RX7900XT/RADV host.
Startup verifies the full configuration,14,123,473 parameters, zero counters,
first updates376:1,384:2,392:3 and346 finite initial tensors. All346 match the
same-shape check's initial, without an equality claim against4M. Initial metadata
SHA256`147fc8966b9b6852a5a72aa131bf7bd9d2146964786c49a5eb5b712787d23909`;
selected declaration`28c43419c871b413650dbb2dc83eb3a49e57a5d843b38345b525c4deb6937f69`.
The100k recovery watcher and completed-run curve/coverage/comparison automation
are running. Frozen learning results remain pending.

The100k recovery audit passes at12,454 updates, with346 finite tensors and
12,637 strict numeric reports through101,472 actions. Captured98,304-action
progress takes938.656s with zero debt. Recovery metadata SHA256
`757ba7a53cb78000641c858ed906fffab04fe0ff69f4f3368f818a714bafaf7b`.
