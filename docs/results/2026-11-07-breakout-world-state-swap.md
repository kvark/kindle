# Breakout: recurrent-state substitutions repair much of the visible rollout

Readout-only replacement of the recurrent state h repairs far more ball tracking
than replacement of the current categorical sample z. On the newer model at
eight steps,accuracy rises8/108→90/108 with image-informed h,versus12/108 with
image-informed z; full teacher reaches104/108. At32 steps the corresponding
counts are0/108,87/108,0/108,107/108.

This supports investigating recurrent rollout supervision before more latent-KL
weight or capacity. It does **not** assign a causal fraction of error to the GRU,
exonerate categorical priors or demonstrate a learnable correction. The teacher
uses future observations; hybrid states can be off-manifold. No model improved
or trained in this diagnostic. The world-model goal remains open.

## Fixed intervention

The [protocol](../experiments/2026-11-07-breakout-world-state-swap.md) selects the
coverage and latent-overshooting final8,192 checkpoints,with saved RGB64/12M/B8/
Auto configurations. Take the first32 of the original128 dense development
origins in their original batch order,and all10 supported natural terminal
windows shifted so termination is future step17. No source-window exclusions;
six padding lanes are never scored. Four fixed seeds83001/83003/83009/83023.
These are reused and event-selected development data,not untouched tests.

Compute the canonical32-step prior once per batch/draw. Separately observe the
real48-frame sequence to obtain teacher states. Decode four combinations at
each step: prior h/z,teacher h with prior z,prior h with teacher z,and full teacher.
No hybrid feeds either branch. Teacher h uses real observations through the
previous arrival; teacher z also uses the current true image and teacher h.
It is not a posterior correction conditioned on the erroneous forecast h.

Initial observed features match exactly. All32 replayed prior transitions and
re-decoded prior RGB/reward/continuation match their canonical values. At step1,
h is identical and the paired h/z readout controls agree. Eight retained baseline
export comparisons,each containing eight arrays,match bitwise,including five
endpoint images/features and32-step head traces. Thus this new intermediate-
state/intervention query reproduces its shared old outputs; it is not another
closed assessment,learner or collector.

## Natural-origin ball readouts

Counts pool four draws per supported origin. For example108 means27 visible
origins×4 draws,not108 independent episodes. The within1px instrument is the
existing conservative RGB ball detector. Missing/ambiguous detections do not pass.

| Checkpoint / horizon | Prior | Teacher h,prior z | Prior h,teacher z | Full teacher |
|---|---:|---:|---:|---:|
|Coverage1|91/108|91/108|104/108|104/108|
|Coverage4|31/108|87/108|34/108|105/108|
|Coverage8|15/108|93/108|24/108|104/108|
|Coverage16|6/112|78/112|9/112|102/112|
|Coverage32|1/108|74/108|0/108|101/108|
|Latent overshooting1|98/108|98/108|103/108|103/108|
|Latent overshooting4|36/108|94/108|37/108|106/108|
|Latent overshooting8|8/108|90/108|12/108|104/108|
|Latent overshooting16|6/112|83/112|5/112|107/112|
|Latent overshooting32|0/108|87/108|0/108|107/108|

At eight steps,teacher-h substitution rescues83 prior misses and harms1 prior
success on the newer model; teacher-z rescues5 and harms1. Coverage's teacher-h
substitution rescues78 and harms0,versus10/1 for teacher-z. The pattern persists
beyond the checkpoint targeted by the last objective. All paddle,brick,detection,
error and paired counts remain in `analysis-review.json`; the table is not the
entire scorecard. Terminal-centered windows separately show the same broad ball
pattern at4/8; no visible balls support their16/32 within1px scores.

For the newer checkpoint,recurrent-state cosine with its own image-informed
state declines1→.9336→.8357→.6802→.5077 across1/4/8/16/32. Recurrent RMS error
reaches.1958 at8 and.3544 at32. Its categorical KL(q_teacher||p_prior) also grows
18.78→22.06→23.98→27.59→30.26. These are internal statistics within that model,
not a common physical quality scale across learned representations.

The last loss only matches categorical distributions,not h directly. Nonetheless,
prior sample errors at earlier times can themselves cause recurrent drift,and
the current-z substitution uses a posterior tied to teacher h. These experiments
do not isolate an intrinsically defective recurrence or prove z contains too
little information. Their strongest result is the readout repair with corrected
recurrent history,which motivates a separately justified training intervention.

## Timing remains imperfect on event-aligned windows

All10 terminal-centered trajectories terminate at17,not16. Probabilities below
first average four draws per origin,then average origins. Post-terminal/reset/
unavailable teacher states are excluded; none become32-step event evidence.

| Newer checkpoint readout | P(end by16),target0 | P(end by17),target1 | Instantaneous terminal probability at17,target1 |
|---|---:|---:|---:|
|Prior|.1453|.2335|.1850|
|Teacher h,prior z|.3706|.6369|.5756|
|Prior h,teacher z|.1764|.2690|.2115|
|Full teacher|.4465|.7347|.6585|

Full teacher improves terminal-arrival detection but still assigns substantial
probability before the actual event. Coverage's full teacher similarly gives
.4578 by16 and.7618 by17. These posterior-conditioned timing errors warrant
attention alongside free-rollout drift; correcting h at readout does not yield
exact event timing on these examples.

The windows were selected by their terminal outcome. These numbers are **not**
natural-frequency false-alarm rates or proof of population miscalibration; they
also do not rule out observational ambiguity. A separately sampled conditional/
natural-frequency assessment is still needed. The earlier prospective-oracle
timing failure remains its own stronger repeated-branch evidence,not a result
retroactively established by this aligned subset.

## Clips and execution

Root `/mnt/data/kindle-breakout-world-state-swap-20261107`. Lossless animated PNGs
`natural-state-readouts.png` and `ending17-state-readouts.png` show the fixed first
origin of each group,draw0,both checkpoints and all four readouts. They contain
32/17 decisions,128/68 original emulator frames,5fps display,action/frame labels
and explicit privileged-future warnings. They stop at the first unsupported
arrival. Static posters and all underlying F32 traces remain retained. No new
forecast is generated to make the clips,and no best draw/origin is chosen.

The original sourcece0dc547/executable2a5feb1d attempt **fails construction**:
strict raw JSON-double comparison differs by one F64 ULP for beta2/epsilon although
the values are identical native F32. It exits101 before any query/snapshot. Its
empty log,source,binary,declaration,failed guard and27 sealed diagnostic files
remain unchanged. It is never relabeled a passed protocol. The first CPU repair
audit's incorrect nine-pin assumption is also retained.

The corrected test parses the complete declaration into DreamerConfig and compares
native values exactly; a CPU regression accepts equivalent F32 values and rejects
real beta2,epsilon and geometry changes. CPU review checks112 configuration leaves
and rehashes the original38 data/14 execution pins. Only the48 unstarted queries
run in root/repair under source`6f0ae3095c200124decbde4b95b08716e6a601e1`,test
executable SHA256`e14de97be90f7457f8d9138540ba5e840c2a62bd98c68d688e5c85952f7f38ca`.
Production graphs/APIs/packages remain unchanged; no runtime overlay is mixed.

56 CPU world tests(25 GPU tests ignored in that CPU run) and strict Clippy pass.
The corrected48 queries complete42.977s; CPU scoring4.723s. Its guard/nine sealed
pins pass.20 corrected-execution,60 analysis-artifact and five clip pins verify;
a separate helper-source audit binds the analysis to byte-identical previously
qualified metrics/event code and records package versions. All816 snapshot tensors
(two checkpoints×before/after×204) match source weights/moments exactly. Optimizer
age stays8,192. Zero training updates,new environment actions or emulator frames.

The initial moved-value compile error and empty-support Boolean CPU-analysis
failure stay preserved with separate repairs. The latter only fixes the dtype
of empty masks; no native query repeats. `closure-review.json` distinguishes the
original failed attempt from the completed corrected measurement,and
`analysis-source-review.json` supplements its source provenance without changing
the pinned closure. No Kindle worker or successor remains active/selected.

The complete goal still requires trainable accurate1/4/8/16/32 forecasts,event/
reward calibration,brick memory,late-game/tunnel/second-wall coverage,untouched
families and fresh confirmation. No state swap is a usable forecast or policy,
and no new loss or learner is selected by this diagnostic alone.
