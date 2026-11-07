# Breakout: bounded latent-overshooting world learner

The [fixed-weight diagnostic](../results/2026-11-07-breakout-world-gradient-noise.md)
finds severe conditional score-function noise in the prior and outcome heads.
The coverage model reconstructs108/112 observed balls but reaches14/107 at eight
steps. Select a new objective addressing causal multi-step transition learning,
without the failed joint sampled energy. Do not tune a scalar from the diagnostic.

Root `/mnt/data/kindle-breakout-world-overshooting-20261107`. Actor/critic,
collectors,backend optimization and the full five-game campaign remain parked.
No old worker,collector or frozen model evaluation is repeated.

## Objective and scope

Add explicit optional `motion_latent_overshooting`,prediction length16,default
off. Preserve observed motion cross-entropy,one-step KL1/.1,observed reward/
continuation losses,all68 parameters and inference. From the posterior at index15,
detach deterministic/stochastic seed states,roll16 causal prior transitions under
the submitted actions at16..31,and match each prior distribution to the detached
posterior logits computed on the observed trajectory. Future observations and
posterior states are targets only; never inject them into the prior rollout.

Additional loss is mean across16×B of per-row sum of categorical
KL(sg(q_future)||p_rollout),with the existing .01 categorical mixing,no free-nat
floor,and coefficient equal to the existing dynamics scale1. Use the same
straight-through intermediate prior samples as ordinary Kindle rollouts; the
final KL integrates the current categorical probabilities exactly. No sampled
reward/continuation score,extra decoder execution or new head. This is not a
claim of globally unbiased stochastic BPTT or direct observable calibration.

Mask a lane from its first natural reset inside the suffix onward,including the
reset arrival. Include genuine terminal arrivals. Normalize by the declared
full B×16,not a variable valid count. Prefix contains ordinary natural resets.
Report base nine components plus latent KL and suffix-valid fraction; total
adds dynamics_scale×latent_KL,not reconstruction_scale×latent_KL.

This is a bounded PlaNet-inspired variant,not a reproduction: one suffix seed,
categorical RSSM,detached seed,existing pixel/motion loss. PlaNet's section4 uses
multi-step latent KL and stopped posterior targets; its RSSM did not consistently
benefit. [Primary source](https://arxiv.org/html/1811.04551v5#S4).
Learned latent agreement alone cannot qualify a world model.

## Qualification before learning

CPU checks cover F64 values/finite differences for mixed categorical masked KL,
detached target/mask gradients,causal rollout ancestry and gradient reach restricted
to dynamics for the added objective. Verify unchanged parameter names/shapes and
legacy objective/checkpoint readers. Python checks cover reports,total scaling,
H16 source-plan identity and terminal/reset masks. Run affected tests and strict
Clippy; package the exact native source and runner. Preserve failures separately.

After CPU/source pins and live-process/host review,one guarded native derivative
fixture(180s) and one full-shape preflight(900s): require204 identical initial
weights/moments versus coverage control,18 initial forecast/target arrays equal,
finite11-report update and correct loss identity,bitwise frozen tensors and
trained restore arrays,legacy coverage restore. Do not qualify by finite loss
alone. Stop on faults/nonfinite values;no automatic retry/reset/reload/reboot.

## Learning and external evaluation

Only after qualification passes,separately pin one fresh8,192-update learner
and restore. Keep RGB64/12M/B8/T32/Auto,seed41001,rates4e-4,exact coverage corpora
and4/2/1/1 locations for all8,192 updates. The reused development seed is not
confirmation. Recompute H16 suffix validity/terminal exposure on CPU before
execution; retain the original H0 counts separately. No new data or hidden-state
labels enter model training. Retain0/2048/4096/8192 checkpoints,fixed development
curves,256 fitting/128 development final exports and all1/4/8/16/32 horizons.

Declare3600s learner and600s restore limits. Then separately declare only the
new checkpoint's retained-oracle evaluation: history16 primary/32 secondary,
eight fixed draws,84 groups,1800s. No oracle recollection or old-model rerun.
Compare actual-initial and coverage controls using their retained arrays.

Development gates fixed now: observed balls>=90%; one-step within1px>=89/109;
eight-step within1px>=28/107(double coverage14/107); primary16/32 ball-feature
energy<=.12712666431732942(25% below coverage .16950221908977256). Also report
all event/class/horizon slices,brick balanced error,serving false balls,ending
calibration and action contrasts. Serving NOOP8 false-ball fraction<=10%,ending32
all-four-plan Brier<=.10 and non-ending group mean ending<=5% remain capability
checks. The old impossible FIRE90% pair-match criterion is not reinstated.

These are development retention checks,not full-goal completion. Forecasts are
stochastic; the single-draw tracking counts cannot alone establish calibrated
physics. Preserve regressions even if a gate passes. No unchanged extension,
scalar sweep,automatic fresh seed or actor/critic successor. Untouched trajectory
families,fresh confirmation,late-game/tunnel/second wall and brick memory remain
required before declaring the full world goal proved.

## Qualified allocation

Implementation/sourcee311b216,native SHA256
`965f2bd2c4b619efe24e841fb0cb325cc85d208743ccbe1aec4fc32728a3f1a0`,
matching root/runtime.54 CPU world tests,53 affected/53 packaged Python checks,
strict workspace/bindings Clippy pass. Preserve initial Rust literal syntax,
F32-mixture/mean-reference and scalar-zero-sentinel fixture failures; corrected
references match exact graph constants and scalar dead gradients are verified0.

One native derivative fixture and full-shape preflight pass under two guards
and nine sealed pins each. All204 initial tensors,18 initial arrays and18 trained
restore arrays match;204 tensors stay frozen in exports. First11-report update
matches shared observed components and total/validity identities;65 parameters
change,three reward-hidden tensors correctly remain fixed behind zero output
weights. Legacy coverage restores.87 qualification input pins pass.

CPU review reproduces all8,192 coverage locations exactly; H16 has9,570 valid
terminal arrivals and975,941 valid suffix rows. All other source counts match
H0:10,909 observed terminals,34,440 positive rewards,10,488 natural resets,
75,808 context resets,32,849 sampled unique arrivals of32,850 available.
Separately pinned96-input learning declaration selects only this fresh8,192-
update learner(3600s) and final restore(600s). No new actions/emulator frames
are collected. Qualification does not establish learning or full competence.
