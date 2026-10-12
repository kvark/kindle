# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Backend performance before more Kindle training:** follow the
  [implementation plan](docs/backend_training_performance_plan.md): cooperative
  convolution weight gradients, actual mixed-precision storage/execution
  (BF16 where Vulkan supports it), then measured fusion/layout/synchronization.
  No Atari learning or frozen rollouts during this round; bounded synthetic
  updates and backend GPU qualification are authorized. Preserve image detail,
  CDP/exploration semantics and strict-F32 fallback. Use isolated upstream
  worktrees; only the user merges. The Freeway screen below was stopped cleanly
  at39,568 actions/9,831 updates, no completed learner or later arm. Partial
  checkpoint/declaration remain; no automatic restart or same-lifetime claim.
- **Combined backend qualifies:** [result](docs/results/2026-10-12-backend-integration.md).
  CDP12M/B32 F16→mixed161.32→135.17ms (16.21% less time); RGB12M/B8
  F16→BF1672.69→61.15ms (15.88%). Same native workloads/initials,16+256 updates,
  not additive isolated gains or learning/time-to-score evidence. Fresh JAX
  RGB FP32/highest24.29ms/BF1612.47ms leaves native2.52x/4.90x slower; different
  weights/RNG/graphs. World training and imagination are now about40% each of
  CDP. No SM-utilization or universal memory-saving claim.
  Meganeurad68f1a5 combines BF16PR250/fd9608f, row addressingPR249 and
  hardware-row selectionPR252 on current maineb8c90f; Blade663afd2 is on
  main7909fc1; Naga forkc9fd6a85a was committed with explicit permission.
  Git pins only, both locks agree, no merge commits or upstream wgpu PR.
  All ten final guards pass/three allocation warnings retained; independent
  CPU audit passes in4.767s. Exact initial346/292 tensors,272 finite reports/
  targets/checkpoints; old controls exactly reproduce their previous fixtures.
  New RGB retains exact preceding BF16 metrics/tensors; new CDP's first imagined
  targets match, but its F16 alignment change alters the later trajectory.
  Thirteen backend/14 Kindle CPU checks, Clippy and193 installed-binding
  runner tests pass. Upstream CI965/966/961, Blade1172 and all Kindle runtime
  CI355 jobs pass. Native83a9227f, Pythonf2f7b999. No GPU worker remains.
- **Precision decision:** `mixed` keeps BF16 posterior/imagination/behavior and
  the measured faster F16-input/F32-storage CDP world-gradient path. RGB's
  `bf16` also reduces forward convolution storage. F32 masters/optimizer/
  reductions remain; convolution derivatives/grouped products are not all BF16.
  [Convolution storage](docs/results/2026-10-12-bf16-convolution.md) and
  [fixed mixed-stage](docs/results/2026-10-12-mixed-world-precision.md) reports
  preserve the isolated decisions. Auto default/live acting stay unchanged;
  obsolete `bf16_dense` CLI support is removed, historical evidence retained.
  Broader CDP world/backward reduced storage and JAX's speed remain open.
  No new precision sweep or automatic learning campaign.
- **Failures/limits retained:** [PR248 attribution](docs/results/2026-10-12-convolution-attribution.md)
  rejects the isolated-fast convolution weight-gradient candidate twice on
  complete updates; keep it unadopted. [Hardware-row checks](docs/results/2026-10-12-convolution-subtiles.md)
  qualify forward/input gradients, not the unchanged scalar weight-oracle failure
  reproduced on control. [BF16 foundations](docs/results/2026-10-11-native-bf16.md)
  and [dense integration](docs/results/2026-10-12-bf16-learner.md) retain odd-clear,
  coverage and CI timing-test failures. Timing-only harvesting now captures all
  four world submissions with~2.7% diagnostic overhead; ordinary GPU fields
  remain last-submission-only, not SM utilization. Full wgpu xtask lacks
  cargo-nextest; scoped Naga checks pass. Final CPU preparation's invalid-CLI
  fixture resets eight ALE environments before rejection, no actor/update,
  setup frames unrecorded; repaired tests prohibit environment construction.
  Missing service PATH and audit-field failures remain with their corrections.
- **October11 performance pivot:** the user authorizes upstream refresh,
  reduced-precision training, cheaper exploration, wider batches, a native/JAX
  profile and artifact cleanup. Optimize one5080 learner before concurrent seeds
  or more GPUs. Old long training queues remain cancelled; the five-game goal
  remains unachieved. The [current plan](docs/kindle_single_life_dreamer_plan.md#performance-pivot)
  supersedes historical “next”, strict-parity-only and no-factorization-retry
  directions for this new performance study, without rewriting their failures.
  Use numerical checks appropriate to the selected arithmetic plus finite
  training and three-seed learning/time evidence; reduced precision need not
  match the legacy F32 trajectory. Keep F32 weights/optimizer/reductions and
  record matrix precision explicitly. No blind retries or concurrent learners.
  Remove regenerable build caches, not user files or unique experiment evidence.
- **Performance candidate:** [report](docs/results/2026-10-11-cdp-performance.md).
  Cheaper all-action exploration, wide grouped RSSM, broader convolution
  splitting and explicit training precision. B32/F16 measures162.00ms per512
  replay positions versus original B8/Auto107.92ms per128:2.66x throughput,
  not learning quality. Latest Meganeurad7a7293/Blade56f0565; retain430f14b
  profiles and older controls. Auto remains default pending the separately
  declared, now stopped [three-seed Freeway comparison](docs/experiments/2026-10-11-cdp-performance-learning.md):
  optimized B8/Auto versus B32/F16,200k actions each, no old queue resumes.
  Half operands alone save little; cooperative convolution weight gradients
  and reduced-precision/fusion coverage are measured backend opportunities.
  No extraGPU or concurrent-seed assumption. SM utilization remains unmeasured.
- **October7 goal: reach DreamerV3 quality on Boxing, Pong, Freeway,
  Breakout and Qbert with CDP; retain the budget comparison.** All five
  and the budget question remain open. A completed screen, learned subset or
  smaller model does not complete this goal.
- **Five-game200k screen complete:** twelve new learners plus all three retained
  Freeway cohorts, seeds1009/2017/3019,49,939 updates each. Frozen means:
  Boxing69.74 vs.39 initial, Pong−18.53 vs−20.31, Freeway24.14 vs0,
  Breakout4.36 vs1.61, Qbert403.13 vs152.43. Four games improve in every seed;
  Pong improves in only one. All45 guards,720 selected natural frozen episodes,
  zero updates/cutoffs, exact346 saved tensors/model, full replay/video checks
  pass. New screen finishes12:36 UTC in5h43m45s; no worker remains. [Result,
  curves and whole videos](docs/results/2026-10-05-cdp-five-game-screen.md).
  Fifteen learners cost3M actions/749,085 updates/6h56m28s training, excluding
  earlier development work. None reaches the predeclared long-run reference.
- **Pong500k follow-up complete:** three fresh124,939-update learners finish
  October5 at17:42 UTC in3h32m29s. Frozen−21.00/−2.375/−20.9167 versus initial
  mean−20.375,7/72 wins. Online mean−15.2667 versus published−7.1622 near2M
  frames (protocol differences retained). Two weak seeds do not recover.
  All nine guards,144 natural frozen episodes, zero updates/cutoffs, exact
  tensors and full replays/videos pass. [Result and curves](docs/results/2026-10-06-cdp-pong-budget.md).
  That raw-cosine allocation stays closed; the new centered allocation below
  is a separately reviewed study, not an automatic extension.
- **Frozen Pong diagnosis complete:** all four guards/exact346 tensors pass,
  131,072 diagnostic actions and zero actor updates. Ball/paddle information is
  readable from every CNN, including initial weights, but ball state/reward
  prediction is weak in1009/3019's RSSM. Their tiny cosine errors mostly match
  a constant-mean prediction;2017 retains useful state and h15 forecasts.
  [Result and controls](docs/results/2026-10-06-cdp-pong-world.md).
  This is not complete encoder collapse. Batch-centered CDP cosine now
  [qualifies](docs/results/2026-10-06-cdp-centered-qualification.md): eight
  guards,32,768 independent centered derivatives,1,300 unchanged-CDP upstream
  comparisons and both production/frozen smokes pass. Natived32f3dc8.
  **Centered CDP's three-seed screen passes:** six fresh200k runs finish
  October6 08:41 UTC in2h55m56s. Frozen centered/control−8.4583/−18.6528,
  paired+10.1944 [7.5,12.375]; every centered seed improves over its actual
  initial and matched control. All18 guards/288 selected natural episodes,
  exact346 tensors/model and full replay/videos pass. Only4/72 centered wins,
  not mastery. [Result and all videos](docs/results/2026-10-06-cdp-centered-learning.md).
  Promote centered loss as the next candidate (`--cdp --cdp-centered`), not
  a solved architecture. No new encoder/capacity/decoder or blind extension.
  **Paired world probes complete:** six guards/exact346 tensors pass at09:02
  UTC in9m40s;196,608 diagnostic actions, zero actor updates. Every centered
  seed has readable recurrent ball state and useful h15 negative-event
  forecasts. Raw2017 remains stronger on some coordinate probes; positive /
  terminal evidence is sparse. [Result](docs/results/2026-10-06-cdp-centered-world.md).
  **Throughput trial passes:** grouped B128 imagination reduces ordinary full
  synthetic updates30.49->28.01ms (8.12%), dispatches3,100->1,210. Same upstream
  backend/control, exact first targets and272 metric reports,428,032 F64 block
  outputs,2,824 upstream comparisons, raw/centered train/frozen smokes and all
  14 guards pass. [Result](docs/results/2026-10-06-cdp-imagination-throughput.md).
  Keep it; no optimization sweep. This is not measured whole-game throughput
  or GPU utilization. Previous per-dispatch instrumentation perturbs timing.
- **Five-game centered500k allocation complete after evaluation repair:**
  [results, curves and whole videos](docs/results/2026-10-07-cdp-centered-five-game-budget.md).
  Fifteen learners,7.5M actions/1,874,085 updates/16h24m7s training; original
  service finishes October7 at02:30 UTC in16h45m23s. Frozen means Boxing77.69,
  Pong4.54, Freeway30.26, Breakout5.14, Qbert993.75; every seed beats its actual
  initial. Boxing/Freeway pass historical gates in all three seeds; Pong54/72
  wins remains unstable, Breakout/Qbert remain weak. None reaches its long-run
  reference. This is not a matched five-game centering/RGB ablation.
  All720 selected natural episodes, exact346 tensors/model, zero frozen updates,
  full replay/video checks and46 GPU guards pass;31 whole videos and all tails
  remain. Breakout3019's original8.0833 cutoff-containing cohort stays invalid.
  The corrected7.9167 reproduces its exact207,288-action prefix; extra209,504
  evaluation actions and the original timeout are retained. Host workers8->1
  in that rerun are disclosed; no unchanged-runtime speed claim. Vector v5
  counts natural episodes; v4 keeps its exact completed-episode receipt schema.
  Both CPU audit failures and pre-fix source snapshots remain. No learner retry.
- **Frozen Breakout diagnosis complete, interpretation limited:**
  [result and next bounded check](docs/results/2026-10-07-cdp-breakout-world.md).
  All six initial/final models finish in9m40s after a12.29s smoke. Seven guards,
  exact346 actor tensors/model and identical action/frame/pixel traces pass;
  196,992 diagnostic actions/61,600 readout updates/zero actor updates. No worker
  remains. Ball state is weakly readable and all three h15 ball readouts lose
  to constant mean; reward evidence is only7/8 positive events at h1/h15.
  The shared pixel control overfits: per-pixel whitening turns bounded RGB
  values into validation/test magnitudes up to4,021. F64 reproduces saved GPU
  predictions at9e-7 relative L2, implicating diagnostic conditioning, not an
  observed GPU prediction fault or proof of lost image information.
  The declared fixed-range follow-up finishes in5.01s:2,048 readout updates,
  selected1,792 by validation, one passing guard, zero new actions/actor updates.
  Paddle test R².979; ball near zero even in training, so this head is not an
  information ceiling. An independent fixed color centroid locates483/483
  eligible held-out open-playfield balls at.628/.542 source-pixel x/y RMSE.
  RGB64 retains those balls; no full-frame guarantee or actor-side detector/RAM.
  Keep original failed heads and source snapshots; no readout sweep.
  [Follow-up data](docs/results/2026-10-07-breakout-pixel-control.json), artifacts
  `runs/cdp-breakout-pixel-range-20261007.kQHipWpq`.
  **12M preflight passes:**1,024 training/195 finite updates in22.23s, then
  1,024 frozen actions/zero updates/exact346 tensors; two guards, no new
  warnings. Last128 updates average108.98ms. [Evidence](docs/results/2026-10-07-cdp-12m-preflight.json).
- **12M capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-07-cdp-capacity.md).
  Three fresh500k Breakout seeds score48.17/15.88/50.58, mean38.21 versus
  retained1M5.14; paired+33.07 [12.38,44.17]. Every seed improves, none reaches
  the381.81 long-run reference.1.5M actions/374,817 updates/11h39m37s training,
  3.58x the retained1M wall time. Service finishes15:39:48 UTC in11h42m14s;
  all nine guards/144 natural frozen episodes/exact346 tensors and six whole
  replays/videos pass. Three allocation warnings,18 excess episodes and all
  tails remain;88,120 frozen actions. CPU review passes; no worker remains.
  Only whole-model capacity changes (16.33M actual parameters including
  CDP/exploration), not an isolated encoder/RSSM/ensemble test. Native6d38eea2
  unchanged. Reused development seeds and old3019's8->1 host-worker repair
  remain disclosed.12M is the next Atari quality candidate,1M the fast control;
  12M prior forecasts remain untested; the separately allocated Pong result
  below extends policy evidence to a second game. No automatic extension.
  All five quality targets and the budget question remain open.
- **Projection-reuse throughput trial rejected:** [two numerical failures](docs/results/2026-10-07-cdp-exploration-throughput.md).
  Tiny/1M B3 F64 checks pass;12M head0 exceeds the pointwise gate with identical
  expanded/factored outputs. A reviewed unstarted1M B128 check then differs by
  4.909e-4 relative L2 versus2e-5. No tolerance relaxation, timing, game actions
  or actor updates; candidate source/binaries/failures retained, original graph
  restored. Precision-path selection is a hypothesis, not proven causality.
  No optimization sweep. [Latest-backend control refresh](docs/results/2026-10-07-meganeura-control-refresh.md)
  qualifies f104f35 on unchanged graphs:20 guards/2,824 upstream comparisons,
  three train/frozen smokes,13 retained allocation warnings. Native83be73bf.
  Paired12M timing107.85->108.00ms;272 metric reports/first outputs/final346
  tensors match exactly in this fixture. No speedup or universal parity claim.
  These scoped checks do not erase the failed stronger F64 check.
  12M updates previously averaged110.03ms,55.4% imagination/29.9% world training;
  utilization remains unmeasured.
- **Four-game allocation complete:** [declaration](docs/experiments/2026-10-07-cdp-12m-four-game-capacity.md),
  Pong, Qbert, Boxing, Freeway; three fresh500k-action seeds each, same centered
  CDP/action-effects recipe.6M actions/1,499,268 updates total; service finishes
  October9 at16:08:01 UTC in47h1m32s. All final/actual-initial natural cohorts,
  exact tensors/full replay/videos pass. CPU completion review finishes16:12:35;
  no learner or review worker remains. No allocation restarts.
  Keep completed12M Breakout and all retained1M controls with their original
  backend pins; no exact same-binary capacity ablation claim. No automatic
  budget extension; all five quality targets and the budget question stay open.
- **12M Pong capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-08-cdp-pong-capacity.md).
  Three fresh500k seeds score7.21/12.96/17.04, mean12.40 versus retained1M4.54;
  paired+7.86 [1.04,18.25],69/72 wins. Every seed improves, but only3019 passes
  the historical mastery gate; long-run20.45 remains unmet.1.5M actions/
  374,817 updates/11h40m7s training,3.57x retained1M cost. Cohort finishes
  October8 at04:52 UTC; Qbert1009 starts next in the unchanged allocation.
  Nine guards/144 natural frozen episodes/exact346 tensors/model and all six
  replays/videos pass;359,456 frozen actions, zero updates, eight excess episodes
  and eight allocation warnings retained. Independent CPU review passes in15.21s.
  A supplemental reporter's retained-control key error and failed source remain;
  its correction changes no runtime, cohort or actions. Native83be73bf versus
  historical controls6d38eea2 is disclosed. No Pong extension or mastery claim.
- **12M Qbert capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-08-cdp-qbert-capacity.md).
  Frozen1,790.63/1,328.13/4,855.21, mean2,657.99 versus retained1M993.75;
  paired+1,664.24 [435.42,3,823.96]. All three reward means improve, but first
  pyramids2/0/0 ->0/0/24:1009 loses its two completions; only3019 completes
  reliably. No seed passes mean15,000 or long-run193,220.77.1.5M actions/
  374,817 updates/11h39m35s training,3.51x retained1M cost. Cohort finishes
  October8 at16:34 UTC; Boxing/Freeway continue in the unchanged allocation.
  Nine guards/144 natural frozen episodes/exact346 tensors and all six replays/
  videos pass;114,408 frozen actions, zero updates, two excess episodes and17
  allocation warnings retained. CPU review passes in15.09s. Task completion
  is selected from existing replay outcomes, not positive-return counts; the
  old Pong reviewer source remains. Capacity helps three games' rewards, not
  yet reliable suite competence. No new Qbert budget or representation sweep.
- **12M Boxing capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-09-cdp-boxing-capacity.md).
  Frozen73.17/92.33/90.29, mean85.26 versus retained1M77.69; paired+7.57
  [−1.29,20.71] includes zero.1009 declines slightly; both sizes win72/72 and
  pass historical gates in all seeds. None reaches99.6133.1.5M actions/
  374,817 updates/11h41m33s training,3.57x retained1M cost; this does not
  establish a consistent larger-model advantage. Cohort finishes October9
  at04:20 UTC; only the declared three Freeway pairs remain. Nine guards/144
  natural frozen episodes/exact346 tensors and six whole replays/videos pass;
  212,288 frozen actions, zero updates, seven excess episodes and11 allocation
  warnings retained. CPU review passes in15.17s. Keep both sizes; the completed
  Freeway comparison below closes this allocation, not the quality goal.
- **12M Freeway and five-game capacity comparison complete:**
  [results, combined comparison and whole videos](docs/results/2026-10-09-cdp-freeway-capacity.md).
  Frozen33.58/17.25/31.79, mean27.54 versus retained1M30.26; paired−2.72
  [−13.29,4.08]. Crossing gates72/72 ->48/72;2017 regresses, with slow learning
  rather than an observed late collapse. All initials are0. The cohort costs
  1.5M actions/374,817 updates/11h41m22s training,3.57x retained1M. Nine guards/144 natural
  frozen episodes/exact346 tensors and six whole replay/videos pass;294,912
  frozen actions, zero updates, no excess episodes, all tails and13 warnings.
  All15 larger learners cost7.5M actions/1,874,085 updates/58h22m14s training,
  3.56x retained1M16h24m7s. All45 guards/720 selected natural episodes/30 full
  videos pass;1,069,184 frozen actions,35 excess episodes and52 warnings remain.
  Capacity helps every paired reward mean in Breakout/Pong/Qbert, not every
  title or task milestone. Keep12M as quality candidate and1M as fast control;
  no universal promotion, capacity sweep or automatic extension. All five
  long-run quality targets,12M prior forecasts and the RGB budget claim remain open.
- **Budget claims:**200k actions are~800k emulator frames, more than Atari100k's
  400k but less than Atari57's200M. Compare published online curves and our
  frozen controls separately. Retain all released seeds and the declared
  last10% long-run target, not convenient bins. Protocol/model/averaging windows
  differ; no exact parity or same-hardware compute-saving claim without a
  matched RGB control. At~800k frames the current means are similar to published
  Atari57 on Boxing/Pong/Breakout, ahead on Freeway, somewhat behind on Qbert.
- CDP is the roadmap main path, while the CLI still defaults to learned RGB:
  main-path declarations must pass `--cdp`. Joint small CNN + deterministic
  latent prediction, Dreamer RSSM/replay/imagined policy, qualified split rates,
  cosine loss and ac_grads=false. No frozen Tiny/DINO/Large requirement or
  visualization decoder. RGB remains the reference/fallback; causal Tiny is
  an optional later video/3D hypothesis, not a2D obligation.
- Freeway exploration is unlocked across all three unassisted200k seeds;
  only3019 passes its original mastery gate. No Freeway-only gate tuning.
  [Result and limitations](docs/results/2026-10-05-freeway-effects-200k.md).
  Action-effects disagreement subtracts each predictor's all-action mean before
  measuring ensemble variance; no new action hint, coefficient, encoder or
  externally shaped reward. Retain earlier all-zero/weak32k screens and their
  additional compute. There is no matched200k extrinsic-only superiority claim.
- The strategy reset remains iteration speed -> useful representation ->
  exploration/reward -> video priors -> native deployment. Phases0–2 and the
  Seaquest CDP/RGB comparison are complete. The3x12M speed target is an unmet,
  user-accepted stretch target. The completed five-game screen's31.62ms
  updates spend51% in imagination,27% world training; about8x aggregate /1x
  per-stream realtime. The new-backend Pong comparison measures32.11ms raw /
  32.13ms centered updates, about28min per200k learner; no new speed claim.
  GPU utilization remains unmeasured. Optimize measured whole-agent cost.
- No old queue restarts. The stopped45-run small CDP/RGB/Tiny comparison retains
  seven completed Freeway pairs plus interrupted CDP3019 at126,408 actions.
  The historical12M matrix retains24 completed and21 cancelled unstarted entries.
  Their controls/protocols are not the current intrinsic study. Venture, Tiny
  comparisons and new representation matrices are deferred by the five-game goal.
- **October6 backend refresh qualified:** Meganeura mainb684ffd9 adds bounded
  shader reads and primitive/composite gradient/recognition fixes over592a2f5a;
  Bladee349cddf unchanged. Native683ccd22 passes18 guards,2,824 upstream CDP/RGB
  comparisons, independent cosine/action-effects checks and train/frozen
  smokes. [Qualification](docs/results/2026-10-06-meganeura-refresh.md).
  Completed learning retains nativef4b6a5c7/592a2f5a; diagnostics disclose their
  new runtime separately. No claim that these fixes caused the old failures.
  Upstreamc6376542 adds attention-backward batching/layout search, not an
  identified CDP fix. It is now adopted/qualified for CDP/RGB with the grouped
  throughput trial; completed probes retain their training native. Tiny's
  attention path is not newly qualified by these CDP/RGB checks. Neighboring
  user work stays untouched.
  Rechecked October7 at02:44 UTC: Meganeurafc3a2fb adds f32 cooperative acceleration
  and checked geometry; Blade49ec60a changes presentation-damage hints. No
  relevant fix for the current small-CDP learning weakness was identified.
  Keep active jobs/frozen diagnosis on the qualified runtime; review a refresh
  before the next learner change, without repinning completed evidence.
- **October9 next-step review:** at approximately2M frames the completed12M
  online means are ahead of released DreamerV3 bins on Pong/Breakout and close
  on Boxing/Qbert; the completed Freeway online mean is also ahead of that bin.
  This is descriptive, not protocol-matched parity or a change to the five
  long-run targets. Freeway and the CPU completion review are complete.
  The final upstream recheck adds Meganeura31026833's native16 split-reduction
  stability fix; Blade56f0565 is unchanged. The [refresh qualifies](docs/results/2026-10-09-meganeura-refresh.md):
  20 guards,2,824 upstream comparisons and all three train/frozen smokes pass.
  Native7311547d; three allocation warnings and the corrected CPU provenance
  failure remain. Paired12M updates107.97->108.23ms: no speedup. First outputs,
  all272 metric reports and346 final tensors match exactly in this fixture.
  Qualification/timing finish16:38:45 UTC; no worker remains. No identified
  cause of earlier learning failures or historical repin. The [B8/T64 versus
  T16 preflight](docs/results/2026-10-09-cdp-sequence-preflight.md) completes:
  four guards, identical initial346 tensors and both frozen restores pass,
  zero warnings. Same65,536 replay positions cost.8580/.8185ms each:4.6% lower,
  below the declared10% gate. T64 has only2.05GiB sampled headroom,56MiB above
  the floor. Keep T16; no sequence sweep or T64 learning campaign.8,192 training
  actions/1,156 updates plus2,048 frozen actions; final T64 debt.5 is reconciled.
  Longer BPTT changes learning/batch statistics/scheduling, not just speed;
  keep capacity/exploration fixed. No sweep or automatic budget extension.
  Neighboring user work stays untouched.
- **Imagination cost attributed:** [one fixed-checkpoint capture](docs/results/2026-10-09-cdp-imagination-cost.md)
  finds the four all-action ensemble first projections at20.50ms/38.7% of
  instrumented imagination dispatch time,91.23G MACs. One guard, exact346 tensors
  across profiling, zero warnings/game actions, one synthetic update; no worker
  remains. Timestamp intervals are not ordinary latency or SM utilization.
  No shader defect, weak-seed cause or new optimization is established; rejected
  factorization stays rejected. Do not make optimization an indefinite learning
  gate. Keep12M/T16; all five quality targets and the RGB budget question remain
  open.
- **Qbert 2M budget study complete:** [results, curves and whole videos](docs/results/2026-10-11-cdp-qbert-budget.md),
  three fresh seeds1009/2017/3019 on unchanged12M/T16 centered CDP/action-effects.
  Frozen5,172.92/5,812.50/11,115.63, mean7,367.01 versus retained500k2,657.99;
  paired+4,709.03 [3,382.29,6,260.42]. First pyramids0/0/24 ->20/22/20:
  both weak seeds improve, but3019 loses four successes. No seed passes the full
  historical gate or long-run193,220.77. Online mean6,545.50 versus published
  8M-frame14,607.50 is descriptive, not protocol-matched parity/inferiority.
  Six million actions/1,499,817 updates/46h40m1s training,4.002x retained500k;
  service finishes October11 at16:10:24 UTC in46h44m25s. All nine guards/144
  natural frozen episodes/exact346 tensors/model and six whole replays/videos
  pass;221,648 frozen actions, zero updates, nine excess episodes and ten warnings
  remain. Final CPU review passes in52.02s; no worker remains. Native7311547d
  versus retained83be73bf is disclosed; no exact same-binary budget-ablation
  claim, historical repin, lifetime resume or automatic extension.
  Next inspect action-conditioned reward/terminal forecasts along recorded
  Qbert policy trajectories in a separately declared, bounded frozen diagnostic,
  before another learning allocation. No capacity/representation sweep or new
  five-game budget. All five quality targets,12M prior forecasts and the RGB
  budget claim remain open.
  **1009's recorded500k prefix matches exactly across runtimes:** [CPU comparison](docs/results/2026-10-09-cdp-qbert-prefix.md),
  initial346 tensors,62,500 transition records,124,939 non-timing learner reports,
  671 episodes and976 common progress reports; online last50 remains1,809.
  Review completes21:30 UTC in6.62s, zero extra actions/updates/GPU work. This
  supports prefix agreement for one seed, not universal parity, pixel identity,
  final500k tensor equality or three-seed/final2M competence. The independent
  end-of-seed reviewer also reproduces all three historical pairs in3.84s.
- **Qbert frozen forecasts complete:** [result and controls](docs/results/2026-10-11-cdp-qbert-forecasts.md).
  Six guards/exact346 tensors and identical traces pass at17:16:46 UTC;
  386,496 diagnostic actions/zero updates/24m16.84s capture,12 warnings.
  Every2M seed beats zero/mean/persistence/unrelated actions on h15 reward
  MAE/RMSE; positive AUC.951/.967/.973, paired MAE gain over500k−1.981
  [−2.854,−1.318]. Later-stage forecasts improve strongly;3019's first-pyramid
  forecast improves neither uniformly nor enough to establish stable control.
  Continuation's discount is explicitly removed for terminal probabilities;
  original metrics remain. Only16/seed h1 and3/2/2 h15 terminals: survival
  remains unresolved. One sampled latent path, final-policy state distribution,
  not exact vector belief replay or universal world-model quality. The failed
  40,789-action lifecycle capture,128-action smoke and original sources remain;
  427,413 total diagnostic actions,15 warnings across all attempts. No worker
  remains. Next a bounded greedy-versus-sampled Qbert check using existing action
  mode and actual initial controls; no new learner budget or representation
  sweep. All five quality targets and the RGB budget question remain open.
- **Greedy Qbert check complete:** [result and whole videos](docs/results/2026-10-11-cdp-qbert-action-mode.md).
  Scores5,444.79/6,392.71/9,022.92, mean6,953.47 versus sampled7,367.01;
  paired−413.54 [−2,092.71,580.21]. First pyramids20/22/20→22/24/23, so all
  improve that fraction, but3019 loses2,092.71 points and none passes the full
  mastery/long-run gate. Keep sampled evaluation. Six guards/144 natural
  episodes/exact346 tensors and whole videos pass;218,936 actions/zero updates,
  14 excess episodes/nine warnings retained. Finishes17:33:10 UTC in3m32.63s;
  independent CPU review passes2.33s. No worker remains. Preparation's lazy
  module error/source remain, before GPU work. Next review longer-horizon
  value/credit assignment and measured learning cost; account for the critic's
  combined extrinsic/intrinsic target. No entropy/representation sweep or
  automatic learner extension; all five quality targets remain open.
- After Atari: video priors for dynamics/behavior, mind-games vkQuake2 then
  TMNF, a small GOG/Wine panel, held-out cross-game adaptation/retention, then
  swarms. One effective actor first; no concurrent learner service now.

- **Qbert live-value check complete:** [result and whole videos](docs/results/2026-10-11-cdp-qbert-values.md).
  All three2M critics beat zero/calibration mean on held-out return MAE/RMSE;
  R².435/.927/.531. Bias+236/−62/−152 is not shared collapse;1009's first-pyramid
  optimism and3019's later underprediction remain. Intrinsic contributes<.03%
  of measured on-policy returns, not proof it was unnecessary during learning.
  Nativee9ebbf5d's lazy batched probe uses exact live features/actions; unchanged
  learning graphs/backend31026833/56f0565, historical training native7311547d.
  Five guards/exact346 tensors/72 natural episodes/full traces and videos pass.
  193,248 diagnostic actions plus1,024 smoke, zero updates;88.87s capture, five
  warnings, nine excess episodes/all tails. CPU timestamp-comparison failure
  and source remain; no GPU retry. Final independent CPU review passes18:06:57
  in2.11s; no worker remains. Next one bounded upstream measured-build check
  of unchanged learning graphs and whole-update cost, not a representation/
  entropy/critic sweep or indefinite optimization gate. Rejected factorization
  stays rejected. No new learner budget; all five quality/RGB budget goals open.

- **October11 measured-build check complete:** [result](docs/results/2026-10-11-cdp-measured-build.md).
  Ordinary Meganeura2c42b448/Blade56f0565/native0f26ee63 qualifies:21 guards,
  2,824 upstream comparisons, three train/frozen smokes and exact first outputs/
  272 metric reports/346 final tensors.107.93→108.30ms: no speedup. Original
  4GiB logical-memory preflight fails; one declared20GiB correction stops on
  value/bonus numerical tolerances at the third callback. Both failed guards,
  source/binaries and ten total warnings remain; zero search optimizer updates/
  game actions, no GPU recovery. Test-only hook removed; no further search,
  gate relaxation or rejected-factorization retry. Next separately declare a
  three-seed Pong experience-budget study on unchanged12M/T16, targeting69/72
  wins and inconsistent mastery. Keep actual initials, natural cohorts and20.45
  long-run target; no automatic suite extension. All five quality/RGB goals open.

- **Pong2M study stopped by user for the pivot:** the
  [fixed declaration](docs/experiments/2026-10-11-cdp-pong-budget.md) remains intact;
  [stop receipt](docs/results/2026-10-11-cdp-pong-stop.json). Seed1009 runs
  October11 18:56:09–20:25:19 UTC, ending gracefully at189,576 actions /
  47,333 updates, zero debt,1h29m6s runner time. The partial and actual-initial
  checkpoints remain under `runs/cdp-pong-2m-20261011.lb5DyI6h`; saved partial
  hashes match the final log. Native0f26ee63/2c42b448/56f0565 unchanged. No frozen
  evaluation or completed learner;2017/3019 and all evaluation jobs are cancelled.
  Service/control-group and game workers are gone, Restart=no, no timer. Systemd's
  failed exit records the intentional controller interruption, not a GPU fault;
  the interrupted guard has no terminal verdict. Zero new allocation warnings;
  143 baseline warning records remain. Do not restore this as an uninterrupted
  lifetime or resume the queue. All five quality/RGB goals remain open.

## Architecture and invariants

- CDP keeps Dreamer's categorical RSSM, replay and imagined actor/critic; it
  predicts detached CNN embeddings from the deterministic prior, not from the
  posterior containing the target. The CNN is jointly trained, not frozen.
  Keep the [qualified recipe](docs/experiments/2026-10-04-cdp.md) reproducible.
- Eight small-recipe environments share batched perception/policy and one
  learner, not causal histories. Preserve per-stream recurrent state, RNG,
  replay and resets. Shared mutable actor/learner weights require serialized
  access. Historical speed comparisons use N6/12M and their original settings;
  a smaller model or lower replay ratio is not an unchanged-learning speedup.
- The target is game GPU -> capture/preprocessing -> encoder -> belief/policy ->
  action readback. Only selected actions leave the acting hot path; rewards,
  checkpoints, diagnostics and sampled learner data may cross the host.
  Scalar targets/slow-critic work still partly uses the host. Do not claim a
  completely GPU-resident learner or environment from a buffer entry point.
- Preserve native image detail. CDP's learned RGB 64 path performs one GPU resize;
  never downscale to RGB 64 and upscale for JEPA. Optional causal Tiny uses
  independent 16-arrival chunk histories; a chunk reset is not an environment/
  RSSM reset. Do not slice Large weights or silently substitute random features.
- Capture must validate ownership, producer completion, memory visibility and
  ring reuse. Blade's existing `Memory::External(Fd(Some(fd)))` ->
  `create_buffer` path borrows/duplicates the FD and uses matching resource/
  allocation recipes at binding offset zero. Device/driver compatibility is the
  caller's responsibility. Acquire/release are safe whole-buffer encoder methods,
  separate from import; first-use ownership belongs to the buffer, not a slot.
  No parallel import constructor or exporter-metadata framework.
- Dullahan GPU_SYNC v4 rejects older protocol tags and transfers the whole ring.
  Its exact-byte ring passed functionally but logged an allocation warning.
  The v4 real-producer test is unrun; earlier v3 vkQuake success is not v4
  qualification or game competence. Keep sparse reward/terminal adapters and
  full mind-games GameSession integration distinct from transport success.

## Evidence to preserve

- [CDP evaluation](docs/results/2026-10-04-cdp-learning.md): three fresh Seaquest
  pairs,200k actions/49,939 updates each. CDP 543.6 versus RGB 318.1 online mean;
  paired+225.5 [62.8,330.4],13.6% less wall time and 35.9% less world-training time.
  All three pairs favor CDP; all guards/seals/counters/finite checkpoints pass.
  No frozen competence or Atari-wide claim. Keep all episodes/tails and failures.
- Frozen CDP/RGB diagnostics:196,608 extra random actions,24 GPU readouts,
  all 1,626 saved tensors unchanged, zero actor updates. CDP's RSSM has better
  readable player state; h15 latent forecasts beat persistence/constant mean/
  unrelated actions. One-step persistence, zero-reward MAE and matched privileged
  position persistence still win. Sparse reward events limit the result.
- [CDP qualification](docs/results/2026-10-04-cdp-qualification.md): independent
  cosine values/1,024 derivatives,1,300 CDP/1,524 RGB upstream comparisons and
  production replay/restore/smokes pass. Raw gradients use exact pre-step weights;
  optimizer/EMA is independently checked, not bitwise stochastic trajectory
  parity. Earlier sequential-weight/configuration/stack failures and warning
  stops remain retained; no numerical tolerances were relaxed.
- [Phase 2](docs/results/2026-10-01-frontend-decision.md) completed in seven of
  ten replication attempts plus six separately declared Tiny runs. Frozen Tiny
  halves world training but takes 18% longer end to end; no clear pretraining
  benefit. Initial Tiny was also frozen. Later
  [joint-Tiny](docs/results/2026-10-03-joint-tiny-learning.md) and
  [direct-policy](docs/results/2026-10-03-policy-tiny-learning.md) screens found
  no clear early advantage. Preserve failed diagnostics and extra offline/
  interrupted compute in their reports, not as new active work.
- Historical non-sticky reliability is3/5 (Boxing, Pong, assisted Freeway).
  Breakout/Qbert fail their original gates; sticky Pong fails (2/24 wins,
  mean−7.1667). Freeway training used random-action probability .5/hold 64.
  Tiny checkpoint `7fe9b252` used 250k same-title random-play observations,
  45k train +5k validation/game. These are not fresh CDP or online-only results.
  See the roadmap's game/video table and [archive](docs/experiments/README.md).
- Phase 1 implementation is complete; the3x12M speed target is an unmet,
  user-accepted stretch target. MinAtar's three-seed screen takes 8m18s but weak
  scores do not establish competence. It is a separate CPU-environment/small
  public-observation recipe, not the CDP Atari control.
- Historical pre-performance production backend: Meganeura main `2c42b448`, Blade `56f0565`,
  native `0f26ee63`; the [October11 ordinary refresh](docs/results/2026-10-11-cdp-measured-build.md)
  qualifies unchanged graphs without speedup. Its measured-construction trial
  fails and is not adopted. Historical `31026833`/`7311547d` retains the Qbert2M
  learners and [October9 qualification](docs/results/2026-10-09-meganeura-refresh.md).
  Historical `f104f354`/
  `e349cddf`/`83be73bf` retains the completed four-game learners and its
  [October7 qualification](docs/results/2026-10-07-meganeura-control-refresh.md).
  Previous `c6376542`/`6d38eea2` remains the historical
  grouped-imagination and completed capacity-study runtime.
  The [October6 refresh](docs/results/2026-10-06-meganeura-refresh.md) qualifies
  CDP/exploration, RGB and frozen Tiny on its historical `b684ffd9`. The historical
  [October4 refresh](docs/results/2026-10-04-meganeura-main-qualification.md)
  passes 2,824 upstream CDP/RGB comparisons, causal Tiny streaming and all three
  production/frozen-restore smokes; 14 guards pass with no new warnings. No
  learning campaign resumes. The previous `13b19d33` already included the
  attention-value alias fix and passed all 148 independent F64 Tiny gradients;
  do not relabel that historical joint-Tiny result as new-backend qualification.
  Check upstream before diagnosing already-fixed issues. Historical Phase 2/
  capture results retain their original pins; no recursive repinning or migration.
  The [capture report](docs/results/2026-10-02-matching-external-allocations.md)
  is separate from ordinary learner qualification.

## Research and status

- Keep one decision-focused roadmap with one game-status table and direct
  rollout/world-report links. Detailed results go in compact JSON + Markdown
  under `docs/results/`; large raw artifacts stay in `runs/`.
- Use the active PR description for dated done/running/next status, results and
  limits. [PR31](https://github.com/kvark/kindle/pull/31) is the current dashboard;
  PR29 is merged history. No STATUS.md. Update at meaningful boundaries, not
  every poll. Preserve the user's untracked `TASK.md`.
- Compare one changed mechanism, at least three learner seeds, matched actual
  interactions and score-vs-actions/time curves with seed-bootstrap uncertainty
  or suite IQM. Episodes/streams are not independent learner replicates.
  Numerical smokes are not learning evidence. One matched timing plus numerical/
  learning check suffices for a speed change; no micro-campaign.
- Frozen evaluation never updates model/optimizer tensors. Predeclare cohorts,
  retain unfinished tails and compare untrained controls for competence claims.
  New controllers must select `cohort(..., natural_only=True)` explicitly; its
  completed-episode default is retained only for historical reports. Do not
  reuse an old controller's complete flag as proof of a natural cohort.
  Separately evaluate prior forecasts with persistence/constant/unrelated-action/
  reward controls and event counts. Posterior estimates are not forecasts,
  features are not imagined RGB, and privileged observers never enter policy.
- A restore without replay/RNG/live belief is not an uninterrupted lifetime.
  Natural deaths/respawns are allowed; cloning/rewinding a live game for learning
  is not. Disclose all aids, shaped rewards and pretraining experience.

## GPU operation

- Ordinary bounded GPU work is authorized on driver580.178.04. Normal JAX/CUDA
  initialization, including internal NVML, is allowed. Separate NVML polling/
  legacy health loggers stay off. No NVML-free backend or CPU learner workaround.
  Unmeasured utilization is not zero; timings are not SM-utilization readings.
- Serialize heavy GPU jobs under `python/examples/gpu_host_guard.py` in persistent
  systemd user services: `Restart=no`, `KillMode=control-group`, bounded deadline.
  Explicitly own/clean up game processes too. Review every failure before follow-up.
- Require the expected native device and >=2 GiB sampled Vulkan estimated budget
  headroom. Budget-minus-usage is not physical free or peak VRAM. Last observed
  boot: `3e89d55c-a9e5-472f-a18a-06508c5bafa7`.
- **October4 user permission: proceed with GPU use unless it is wedged.**
  New declarations set `record_allocation_warnings=true`: retain standalone
  allocation warnings without aborting or requesting approval for each one.
  This supersedes the earlier120-second and proposed startup-only restrictions.
  API/numerical failures, hard faults and deadlines still stop the affected job
  for review; do not confuse a failed job with a wedged GPU or an approval gate.
  No blind retries. A wedge needs recovery review. The known
  `VUID-StandaloneSpirv-None-10684` is non-blocking by explicit user direction;
  other validation errors remain job failures, not evidence of a wedge by themselves.
- No GPU reset, driver reload/change, reboot or power-cycle without new user
  approval. Historical Xid62/154 incidents remain unexplained; successful
  no-NVML jobs prove neither causality nor safety. Never retry quarantined
  75dfe, 0a98775/02b600a1 or 070f4b51/7db0d05c bundles.
  Follow [incident response](docs/gpu_incident_response.md), not old followers.

## Implementation discipline

Keep exercised production code small. Delete obsolete encoding/compatibility
code instead of extending it; do not preserve old checkpoints at the cost of the
new path. Preserve unrelated user edits, especially in Meganeura/Blade/mind-games.
Use proportionate tests, formatting and Clippy. Do not rebuild unchanged native
source for docs, or compile during matched timings.

Automate routine validation; inspect long training about every 30 minutes or on
completion, not counters on every poll. Heavy preparation uses one CPU,2 GiB and
zero swap. Apply CPU limits to preparation/analysis, not GPU workers in matched
runs: the native host-worker count follows the available CPU quota. Review CPU
test filters: Meganeura has unignored GPU tests.
Only the user merges; commits/pushes are allowed. Keep history linear.
