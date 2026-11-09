# Fresh broad mean-latent world learning

[Qualification](../results/2026-11-09-breakout-world-mean-latents.md) passes:
149 world/29 config CPU,strict core/binding Clippy,31 packaged Python checks,
native F64 probabilities/true derivatives/CPU parity,204 tensors,32 default-off
initial/32 trained restore arrays andidentical requested draws. The qualifier
update isclosed;the main learner starts fresh,not from its age1 checkpoint.
Root `/x/Temp/kindle-world-mean-latents-learning-20261109`.

Select exactly8,192 fresh updates,RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,
rates4e-4,mixing.01,KL1/.1/freeNat1,observed reward/continuation1,RGB100. Change
only `world_mean_latents=true` relative to the original H16 recipe. Posterior
andprior state feeds are their mixed probability vectors during training and
inference,with no latent uniforms/Gumbel/one-hot/straight-through sampling.
No extra image weighting/position head/parameter/pretrained encoder/object-state
input.68 weights/8,852,675 scalars/204 saved tensors/11 reports,v24.

This changes representation andgradient estimator together,not merely inference
noise ormixing. The old Gaussian/lower-mixing runs stayclosed. Prior drift
evidence doesnot establish sampling as the cause. Four requested draws are
identical andmust notcount as independent stochastic evidence. Nonlinear
f(mean-state) isnot mean(sampled futures);stochastic calibration isnot assumed.
All original branch/distribution,event andfull-game requirements remain.

Reuse sourceaffedefa/native SHA256
`cc6022e36fbb0c3d145058fe85f5fee66c38172e5e54335bfc051cdcbcf07494`
andmatching qualification runtime/runner,no rebuild/overlay. Match all204
qualified zero-step tensors. Record new actual-initial predictions;they differ
from sampled controls bydesign. Reused41001/development families are not fresh
learner confirmation. No actor/critic/CDP ornew emulator actions/frames.

Exact original16 fitting/eight development streams andsealed4/2/1/1 plan:
2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,
16,008 positive suffix rewards. Preserve source max-pooling andgeometry.

Retain initial/2048/4096/final8192 checkpoints,curves0/2048/4096/6144/8192,
original384 dense initial/final origins andalltarget/command/reset/outcome arrays.
Preserve all four requested draw exports andverify their identical model arrays
in curve/contact exports. Require8,192 finite11-report rows,RGB100 total/source/
mask identities,all68 changed weights,204 own initial/frozen tensors exact,
18 development/32 original-contact restore arrays exact andv24 flag retention.
Additional contact clips remain reused diagnostics,not this learner fitting
data oruntouched generalization. New actual-initial tensors/outputs stayfrozen.

Fixed final ordinary RGB screen:observed≥90%,one-step≥89/109,eight-step≥28/107
withinone resized pixel. Compare retainedH0,originalH16 andclosed representation/
loss references without requerying old models. Keepall1/4/8/16/32,presence,paddle,
brick,changed-pixel andevent regressions. No bestcheckpoint/draw,changed tolerance
orcoordinate substitution. Require allthree gates before separately declaring
the existing84-group branch assessment;otherwise close negative. No conditional
assessment isallocated here. A deterministic point forecast doesnot discharge
natural-frequency calibration or uncertain-future requirements.

Pin source archive,native,runner,config,data,plan andscripts beforeGPU. Declared
scripts,including CPU analysis,areimmutable;repairs use new filenames/declarations.
Two serial host-guarded jobs:train cap2400s,restore cap600s. The independent
observer checks live PID/invocation/command andseals guard audits after success.
Observer timeout isnot astopped job;never restart solely because observation
expires. Stop realfaults/nonfinite state;no NVML/reset/reload/reboot orautomatic
retry. Original fixture style failure/corrected checks remain. The qualifier's
811,446,272-byte build cache isreleased;packages/executable/source remain.

Full precise physics,natural reward/life/game-over calibration,persistent
removed-brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required. No implementation/fitting/intermediate result
completes thegoal. No extension,collector orautomatic successor. Actor/critic,
CDP comparison andfive-game work stayparked;earlier jobs remain closed.
