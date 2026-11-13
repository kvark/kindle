# Full-resolution pixel predictor: protocol-failed run, CPU-only diagnosis

The goal **prove the world model training on Breakout** remains unproven.
The [fullres16 run](../experiments/2026-11-09-breakout-fullres16.md) is
**protocol-failed andineligible as aqualified comparison**. While the learner
ran,I edited analyse.py,which was already hash-pinned. It completed2,048 updates,
final forecasts/event checks andsaved final-frozen weights,then its final pin
check failed onthat script. The controller stopped before restore. Original
failed service,child stderr,observer failure,manifest anddeclared/edited analysis
bytes remain preserved. No learner,assessment,guard audit orGPU work isrepeated
to repair this failure. A separate CPU-only review scores saved predictions;
it doesnot relabel the original protocol aspassed. Restore remains unmeasured.

Descriptively,one-step matches improve while ordinary rollouts regress;all24
original fitting gates stillfail. Retain the qualified width16 U-Net/reward32
control;do not adopt this experimental recipe.

| Saved final2048 task | Qualified U-Net control | Fullres own initial | Fullres trained,protocol-failed |
|---|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|0/0/0/0|27/25/25/27|
| Actual-history one-step,198 supported per draw |140/149/140/140|See retained initial teacher outputs|149/152/152/151|
| Ordinary horizon1,support8 |6/6/5/6|See saved curves|7/7/7/7|
| Ordinary horizon4,support6 |3/3/2/2|See saved curves|5/4/4/4|
| Ordinary horizon8,support8 |2/6/4/3|See saved curves|1/0/0/0|
| Ordinary horizon16,support8 |0/1/0/0|See saved curves|0/0/0/0|

Native counters:2,048 updates/65,536 targets,all45 weights changed,finite losses/
tensors. train/result.json saysstatuscomplete because that counter file was
written before the final integrity assertion;it isnot a successful job receipt.
The failed main host guard reportschildexit1/host_guard_passed=false. Its audit
verifies27 sealed files with no unsealed files,whilepreserving the failed job
status. Legacy fixture audit verifiesnine files/pass. No kernel_fault event is
observed;this isnot an independent hardware-health qualification.

The declared analysis bytes are reconstructed exactly from the immutable creator/
source andmatch their original SHA256,archived in declared-inputs/analyse.py.
All144 declaration inputs match after resolving that one path toits archive;
the live-path mismatch remains explicit. The edited original path ispreserved.
Separate analyse_v2.py declares its inputs andindependently checks saved tensors,
loss rows,targets/config andscores. First preservation CPU script expected audit
CLIexit1;the auditor returns0 for avalid integrity audit evenwhen job status
isfailed. Original CPU assertion/log kept;preserve_failure_v2.py reads the existing
audit,without rerunning it. Original protocol remains failed throughout.

Architecture: optional full_resolution=false preserves U-Net/v1–v10. True uses
v11/conditional-fullres16-residual0p1-history4. Allnine5×5 pixel convolutions stay
64×64,16 hidden channels,seven residual adds at.1,no feature down/up sampling or
U-Net concatenations.81,459 pixel parameters versus606,707,event heads unchanged
1,181,136,total1,262,595 versus1,787,843. Maximum theoretical receptive field37×37.
Spatial layout,channel allocation,skip topology andparameter budget change
together;this doesnot isolate resampling orcapacity as acause. Four-frame
conditioning doesnot resolve missing-state memory. Normal native RGB resize stays.

Sourcec4f3279b/native1eb5d74265a9084951adad36d3e962085668a0b416a6cbdc1c413799d9faafcf.
33 Rust CPU/67 packaged configuration-restore/four runner/strict release test-target
Clippy pass. Pixel convolution sizes,event graph identity,train/sample sharing,
counts,legacy defaults andheader rejection checked. Existing conv/norm/SiLU/add/
scale derivative mechanisms reused;no new independent full-network float64
measurement. Guarded default U-Net fixture135 initial/frozen/first-update tensors,
20 arrays/six losses matched before main. Main57 initial/first/final event tensors,
15 shared-edge initial tensors/two first event losses match.135 own initial/final
frozen tensors match inCPU repair;GPU restore/20 restored outputs never run.

Same actual histories,RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,
residual/half-max,.05 scale,default noise mean−1.2,reward32,three Euler steps,
224 unique source arrivals,588 positive exposures andnative Gaussian/target
order. No history augmentation,target-motion,direct-frame,width32,new data,
emulator actions/frames,object labels,actor/critic/CDP orbackend primitive.

Counter intervals:learn-call165.783608s,loop173.606426s,
versus control161.420355s/169.299289s.
They cover completed loop work before the integrity failure,not aqualified
end-to-end comparison orisolated kernel speed. Loop includes packing,intermediate
exports/checkpoints,not all initial/final assessment. Native Meganeura+Blade
learn/infer;Python supplies corpus/analysis. Own792,853,659-byte
apparent disposable build cache released after authoritative failed terminal
state andCPU review;native,wheel,CPU executable,source,weights/logs retained.

Original gates unchanged:all supported ordinary1/4/8/16 balls within1px andall16
>=90%;actual-history one-step>=90%. Missing predictions fail;18 unsupported per
task/draw unscored. All1,376 endpoints,motion,pixel ambiguity,paddle/brick/events
andfixedlane0/draw0 lossless frames retained. No best-age,tolerance orscope change.
Four draws share one learner;one causal positive/no terminals cannot establish
natural calibration. Full physics/events,changed-brick memory,late/tunnel/second-
wall/above-top,untouched families andfresh confirmation remain required. No
fitting score completes the full world-model orfive-game goal.

Artifacts `/mnt/data/kindle-breakout-fullres16-20261109`,main invocation
30c3cfe1c7b84f759226622bab4bcbc9. No extension,collector,old-model repeat,restore
retry orautomatic successor selected. Actor/critic/CDP/five-game work stayparked.

CPU closure verifies277 archived/current evidence pins in0.636s. Original protocol/live-pin andmain guard remain failed;legacy guard passed,two actual sealed guards/36 files. Main/legacy services are respectively failed/successful andboth stopped. Restore remains unrun;no GPU/native/audit repeats. The recovered CPU assessment doesnot qualify the comparison.
