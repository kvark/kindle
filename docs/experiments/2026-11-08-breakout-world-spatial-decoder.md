# Breakout: decoder with causal spatial image history

The Gaussian512 and Gaussian32 recipes fail forecasting despite strong observed
reconstruction. Select a distinct spatial-memory hypothesis, not another latent
size or unchanged extension. Keep the categorical RSSM and train a decoder that
also sees the two preceding RGB images through a small local convolution branch.
Past images preserve visible board detail and local motion without requiring
all of it to survive a global bottleneck. This is a hypothesis, not an isolated
cause or a claim that other architectures cannot work.

Optional default-off `rgb_spatial_decoder`, standalone plain RGB world only,
with motion_cross_entropy. Three5x5 stride1 ConvNorm layers, width8, process6
history channels at native model geometry; concatenate8 channels with the
upsampled decoder features before the final5x5 RGB output convolution. Effective
history receptive field17 pixels. No hand labels, motion rules, pretrained
encoder or new backend primitive. RGB64/12M categorical recurrent2048/stochastic
512/encoder1024 stay. Expected77 logical/8,857,723 scalar weights/231 tensors;
67 unchanged-shape initialized weights/201 tensors should match H0. Verify counts.
The history branch and expanded output projection change initialization,capacity,
state geometry,computational work and gradient paths together.

The full world state now includes two RGB64 images (24,576 scalars) in addition
to the RSSM feature2560. Legacy feature readbacks expose only the RSSM portion;
do not interpret those as the full new belief. Replay stays RGB with one GPU
resize. Posterior reconstruction reads only images before its current arrival,
zeroing history at reset and older history across the preceding reset. The
forecast seed uses the last two observed images, zeroing the older across a
last-observed reset. Future steps shift in decoded predictions, with full BPTT
through the image path. Future images, terminal/reset labels and motion weights
are targets/masks only; no future outcome resets or corrects imagined images.
Inference uses exactly the same history shift. No teacher forcing in futures.

v10 persists the explicit architecture. Old natives reject before GPU; no
old-format disguise or categorical phase/agent entry-point use. Defaults and
old formats remain unchanged. Checkpoints omit live image history/replay/RNG;
restore keeps architecture and geometry. Fresh initial/restored controls required.
No prior checkpoint warm start.

CPU qualification: config/header/parameter geometry, history lane/CHW layout,
reset/clip masking, F64 history values/derivatives, unique inputs, prior/core/
decoder gradient connectivity and future-image/reset-mask ancestry exclusions.
Packaged Python rejections and strict core/binding Clippy. Reuse existing Conv5/
RMS/upsample/categorical receipts; no backend qualification campaign. Separately
declare one900s guarded native composition: history values/derivatives,one full
update with finite11 reports/total/masks,all expected changed families including
spatial branch,201 common initial/231 own initial tensors,18 initial/trained
restore arrays,history perturbation affects decoding,231 frozen tensors and
old-native v10 rejection. Preserve original failures and separate repairs.

Only after reviewed native qualification separately declare fresh8,192 updates
(3600s) and final restore(600s): RGB64/12M/B8/T32/full BPTT/Auto,seed41001,
rates4e-4,H16 motion RGB100 plus observed RGB100,KL1/.1/free-nat1,reward/cont1.
Original full8,192-row4/2/1/1 plan,sixteen fitting/eight development streams,
975,941 valid suffix targets/9,570 suffix terminals and all source counts stay.
Retain initial/2048/4096/final,384 dense actual-initial/final origins,all finite
reports,full exposures,231 frozen tensors and18 restore arrays. Record the
additional image state/activation work and all new costs; no isolated speed claim.
No new collection,data,actor/critic,labels or old-model query.

After reviewed learning/restore separately declare only new-final84 frozen
oracle groups(1800s),history16 primary/32 secondary,eight draws,all1/4/8/16/32
endpoints. Reuse oracle/H0/categorical H16 arrays. Unchanged gates: observed>=90%,
one>=89/109,eight>=28/107,primary16/32 survival-weighted ball energy<=
.12712666431732942,serve NOOP false ball<=10%,ending32 Brier<=.10,ending16<=.01,
RIGHT16 ending<=.10,nonending category means<=5%,freeflight8 ball presence>=90%
each command. Keep every individual false alarm and action/horizon reward,paddle,
brick regression. No easier support or best-draw selection. Event heads still
read RSSM state; passing image metrics alone cannot qualify the world model.

Full physics,event timing/calibration,persistent removed-brick memory,late/tunnel/
second-wall coverage,untouched families and fresh-seed confirmation remain
required. Existing oracle has no above-top-brick endpoint. Intermediate gates
cannot complete the goal. No unchanged extension,sweep,completed stage repeat or
automatic successor. AMD/Auto/host guard; inspect live identity before GPU,stop
real faults/nonfinite state,no driver reset/recovery/reboot/NVML. Root
`/mnt/data/kindle-breakout-world-spatial-decoder-20261108`. No GPU allocation yet.
