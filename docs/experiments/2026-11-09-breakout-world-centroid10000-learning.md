# Fresh broad world learning with position weight10,000

The [frozen contract diagnosis](../results/2026-11-09-breakout-world-centroid-contract.md)
passes native/CPU readout,same-time target andgradient accounting. Onone recorded
mixed batch/two draws,position/RGB gradient norms are4.61–6.79% inthe encoder,
4.60–5.71% inthe prior and12.05–12.69% inthe core. Encoder directions are partly
opposed. Test10× stronger position supervision before adding another mechanism.
These measurements are not an isolated cause orAdam/AGC update prediction;
contact clips have stronger relative gradients andcould regress under this scale.

Root `/x/Temp/kindle-world-centroid10000-learning-20261109`. This is a new fresh
learner,not a restore/extension/retry ofthe closed weight1000 learner. Change only
`color_centroid_supervision.scale`1000→10000 relative tothat recipe. KeepRGB1,
RGB64/12M/B8/T32/H16/full BPTT/Auto,41001,allrates4e-4,mixing.01,KL1/.1/freeNat1,
observed reward/continuation1. Same native449361ae/SHAee44ec16,matching qualified
runtime/runner,no rebuild. Same70 parameters/8,857,797 scalars/210 tensors,v22,
shared linear head. Same game-specific RGB-derived color/region supervision;
no object/RAM state enters input orloss. Optional mode staysdefaultoff.

Reuse exact16 fitting/eight development streams andsealed4/2/1/1 plan. Exactly
8,192 fresh updates,2,097,152 replayed arrivals,975,941 valid suffix targets,
9,570 suffix terminals,16,008 positive suffix rewards. No emulator actions/frames,
actor,critic,CDP orcollection. Reused training/development seed families are not
fresh confirmation. The qualification'szero-step210 tensors must match exactly;
no qualification update isresumed. Budget,initialization,exports andgates match
the weight1000 learner;this is acoefficient development comparison,not areliability
orbackend performance claim.

Retain initial/2048/4096/final checkpoints,allcurves0/2048/4096/6144/8192,four
draws,original384 dense initial/final origins andallsource/target/action/reset/
outcome arrays. Require8,192 finite15-report rows,weighted total/mask/source
identities,all70 weights change,210 initial/own frozen tensors exact,18 development
and32 original-contact restore arrays exact. Additional contact clips remain
reused diagnostics,not fitting data ofthis learner oruntouched evaluation.

Primary comparison isfixed final ordinary RGB onoriginal dense development:
observed>=90%,one-step>=89/109,eight-step>=28/107 within one resized pixel.
Keepalloriginal horizon1/4/8/16/32,paddle,brick,image/event metrics andregressions.
Compare retained H0,original H16 andclosed weight1000 RGB forecasts without old
GPU queries. Do not select abetter checkpoint/draw,change tolerance orsubstitute
coordinate accuracy. Ifallthree intermediate gates pass,separately declare the
existing84-group oracle assessment;otherwise close negative. No conditional
assessment,extension,collector orautomatic successor isallocated here.

Pin source archive,native,runner,config,data,plan andscripts before GPU. Scripts,
including CPU analysis,stayimmutable;new repair filenames anddeclarations only.
Two serial guarded jobs:train cap2400s,restore cap600s. Verify live unit PID,
invocation andcommand;observer timeout isnot astopped job. Existing host-only
guard/Auto,no NVML orreset/reload/reboot. Stop realfaults/nonfinite state.
Automated observer closes guards andverifies pins after successful terminal
execution. Logs,weights,arrays andreceipts stayoutsideGit.

Precise all-horizon physics,natural event calibration,removed-brick memory,
late/tunnel/second-wall/above-top,untouched families andfresh learner confirmation
remain required. Even passing allthree intermediate gates wouldnot prove the
full world-model goal. Actor/critic,CDP comparison andfive-game work stayparked.
