# Breakout: diagonal Gaussian latent world reference

Select a distinct architecture after closed fixed-target and fixed-filter RGB
phases fail retention. Accurate observed reconstruction and poor causal futures
remain; categorical sampling/straight-through bias is a hypothesis,not an
established cause. Replace categorical latents with learned diagonal Gaussians
and reparameterized samples. Do not extend or repeat any completed learner.
Root `/mnt/data/kindle-breakout-world-gaussian-20261107`; no GPU allocation yet.

## Selected architecture and budget

Optional `world_gaussian`,default false,standalone plain RGB world only.
Keep512 stochastic scalars and2048 recurrent scalars at12M,encoder1024,state2560,
CNN/GRU/RGB decoder/reward-continuation heads unchanged. Prior/posterior output
both mean and raw scale,interleaved per original16-wide block. Scale is
`.1 + sigmoid(raw_scale)`,bounded[.1,1.1]; means unbounded. Reparameterize
`z=mean+std*epsilon`,independent standard normals,no categorical straight-through
operation. Sum exact diagonal-Gaussian KL over512 scalars;retain stopped-q
/dynamics1,stopped-p/representation.1,free-nat1. Unimix is unused by this family.
This changes distribution family,capacity,KL geometry,sampling and gradients;
do not claim an isolated estimator effect or reproduction of an external method.

Four output projection weights/biases double. Expected68 logical/9,115,843 scalar
weights/204 tensors,versus68/8,852,675/204. Same named initialization should preserve
64 common weights/192 weight-moment tensors; verify rather than assume. New draws
and output geometry require actual-initial and own restored controls; no cached
initial forecast equality claim. No frozen teacher session or warm initialization.

Select fresh8192 updates,H16 existing motion RGB cross-entropy100,base RGB100,
KL1/.1/free-nat1,reward/continuation1,RGB64/12M/B8/T32/full BPTT/Auto,seed41001,
rates4e-4,original sixteen fitting/eight development streams and complete original
8192-row4/2/1/1 plan. Gaussian priors receive RGB suffix gradients through fixed
normal draws;future images/motion weights are stopped targets only. Keep terminal
arrivals and reset masks/full Bx16 denominator. No new labels,data,pretrained
encoder,actor/critic or collection. Learning/inference stay native Meganeura/Blade.

Save architecture in v8 with explicit config flag;old formats cannot disguise it,
old native rejects v8 before GPU. Restore keeps geometry/objective and rejects
incompatible flags/unsupported world losses. Whole-agent entry points reject
this standalone architecture. Default categorical graphs,sampler RNG and versions
v1–v7 remain unchanged. Full learner RNG/replay/live game state are not saved;
any restored extension would be a warm start,not selected here.

## Qualification and evaluations

CPU F64 values/finite-difference derivatives for Gaussian sample/scale/KL including
stop sides/free-nat floor,standard-normal seeded stream/shape/moments,unique input
names,forward ancestry/reset masks,parameter geometry,new header/config/phase
rejection,default categorical behavior and packaged Python. Strict core/binding
Clippy. Reuse unchanged CNN/GRU/resize primitives;no backend port/qualification
campaign. Preserve first failures and separate corrections.

Separately declare one900s native preflight. Compare Gaussian math values/
derivatives against F64 at declared fixture sizes,then composed full-shape one
update with finite11 reports/total/masks,expected changed parameter families,
204 finite/frozen tensors,18 initial and18 trained restore arrays. Verify64
common weights/192 tensors against H0 actual initial,own204 initial restore,
new means/scales genuinely used,old-native v8 rejection with GPU loader disabled.
No prior successful primitive or old-model forecast rerun.

Only after reviewed qualification,separately declare8192 fresh updates(3600s)
and final restore(600s). Keep initial/2048/4096/final,384 matched dense-initial/
final origins,original periodic cohorts,finite11-report rows,exact full source
plan/exposures,204 initial/frozen tensors and18 final restore arrays. Report all
new costs and initialization/geometry differences. This development seed is not
fresh confirmation. No unchanged extension,sweep or automatic successor.

After reviewed learning/restore,separately declare new-final-only84 oracle groups
(1800s),history16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints. Reuse
oracle/H0/previous control arrays,not old-model queries. Keep observed>=90%,one
>=89/109,eight>=28/107,primary16/32 energy<=.12712666431732942,serve NOOP false-ball
<=10%,ending32 Brier<=.10,ending16 Brier<=.01,RIGHT16 ending<=.10,non-ending category
means<=5%,freeflight8 presence>=90% each command. Retain every individual false
alarm and per-action/horizon/reward/paddle/brick regression. No easier slice.

AMD/Auto/existing host guard;stop real faults/nonfinite state,no reset/recovery/
NVML. Full all-horizon physics,event calibration,persistent brick memory,late/
tunnel/second-wall coverage,untouched families and fresh confirmation remain
required. This oracle has no above-top-brick endpoint. Passing intermediate gates
does not prove the full goal;actor/critic/CDP campaign remains parked.
