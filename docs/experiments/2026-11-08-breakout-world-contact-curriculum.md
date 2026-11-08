# Breakout: contact-first curriculum on the broader corpus

The [fixed-contact diagnostic](../results/2026-11-08-breakout-world-contact-fit.md)
fits all110 supported future ball positions across actions1..16 on each of four
draws. Broad-data H16 training remains inaccurate. Select a distinct staged-data
experiment: warm-start that precisely fitting checkpoint, then train the ordinary
all-weight world on the broader corpus. This tests a practical curriculum, not
an isolated cause of earlier failures. No completed stage is repeated.

Root `/mnt/data/kindle-breakout-world-contact-curriculum-20261108`. Reuse the
qualified native85dbe150/SHA5b112866 runtime and the existing H16 motion RGB
objective unchanged. No native rebuild, new loss, frozen subset, spatial decoder,
image feedback, pretrained encoder, object/RAM training target, actor or critic.

Restore exactly `/mnt/data/kindle-breakout-world-contact-fit-20261108/train/checkpoint-2048`,
including all204 weights/moments and optimizer age. Keep RGB64/12M/B8/T32/full
BPTT/Auto, seed41001, rates4e-4, mixing.01, observed/future motion RGB CE100,
KL1/.1/free-nat1 and observed reward/continuation1. Restore has no fresh-training
overrides. Checkpoints omit live belief/replay/full RNG, so this is a warm start,
not uninterrupted training or an independent learner.

Run exactly6,144 new updates, age2,048→8,192. Use the first6,144 rows of the
retained8,192-row4/2/1/1 plan:four original-fit uniform, two supplement uniform,
one waiting and one terminal window. These are the same sixteen broad fitting
streams as the retained H0/H16 controls. The eight pretraining sources are a
different retained family; verify all pretraining/broad fitting seeds are
disjoint from original development. No collection/emulator actions/frames.
The staged recipe changes data order, volume per family and optimizer history.
Total8,192 updates matches the fresh H16 budget, but broad-data exposure differs;
no isolated curriculum estimate or matched-experience claim follows.

CPU preparation independently verifies the prefix plan, all broad source hashes,
reset/action/outcome alignment, suffix masks, exact target exposure counts and
source-family separation. Pin the inherited checkpoint, its fitting-only closure,
native/runtime, complete runner import closure and source snapshot. Test imports
with GPU loader disabled, constructing neither a native world nor an emulator.
Reuse qualified native primitives/restore without repeating their fixtures.

One guarded learner,2400s cap, must first reproduce all204 inherited tensors and
all32 cached contact prediction arrays exactly. Save warm actual-initial controls
on the original384 dense fitting/development origins. Keep initial/age4096/6144/
final8192 checkpoints, all6,144 eleven-loss rows, plan/exposures and periodic
four-draw cohorts. Final dense origins/source arrays must match original controls.
Compare original contact predictions again after broad learning and retain every
1..16 endpoint, with no favorable draw or checkpoint selection. Contact retention
is descriptive and does not replace the full-corpus gates.

A separately guarded600s final restore must reproduce the18 original development
arrays and32 contact prediction arrays exactly, with all204 tensors unchanged.
No architecture/learning-setting override or frozen phase is introduced. Report
new training/evaluation/restore costs plus inherited186.289s native/191.500s loop.
Distinguish actual construction/dense-initial/export/pinning work from loop timers.

After reviewed training/restore, separately declare only the new final's84 frozen
oracle groups,1800s cap, history16 primary/32 secondary, eight draws, endpoints
1/4/8/16/32. Reuse cached oracle/H0/categorical-H16 arrays with no old-model query.
Keep gates:observed>=90%,one>=89/109,eight>=28/107,category-balanced survival-
weighted16/32 ball energy<=.12712666431732942,serve NOOP false ball<=10%,ending32
Brier<=.10,ending16<=.01,RIGHT16 ending<=.10,nonending category means<=5%,and
freeflight8 ball presence>=90% for each command. Keep individual false alarms and
all per-action/horizon ball/paddle/reward/ending/brick regressions visible.

Full all-horizon causal physics,event timing/calibration,persistent brick memory,
late/tunnel/second-wall coverage,untouched families and fresh confirmation remain
required. The retained oracle has no above-top-brick endpoint. Intermediate gates
cannot prove the full goal. No unchanged extension,repeat,sweep or automatic
successor. AMD/Auto/existing host guard;stop real faults/nonfinite state,no driver
reset/recovery/reboot/NVML. No GPU allocation has run at declaration time.
