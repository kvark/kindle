# Kindle working direction

Kindle is a Rust agent that learns while acting in one continuing stream of
experience. Games are the first testbed. Intrinsic motivation and experience
sharing between independent Kindles are long-term goals; game rewards and human
guidance are allowed while establishing reliable learning.

- Favor minimalism, expressiveness, safety, and speed. Keep the native learning
  and inference path on Meganeura and Blade. Python is for adapters, controls,
  and analysis.
- Maintain one authoritative research plan at
  `docs/kindle_single_life_dreamer_plan.md`. Keep its claims tied to code and
  measured results. Experiment logs belong in `runs/`, not in an ever-growing
  chronological plan.
- Prioritize one learning actor: Atari breadth, video/world pretraining and fast
  serial playing plus training, then mind-games (vkQuake2/TMNF), GOG/Wine games,
  cross-game adaptation and retention. Pong's initial-learning gate is achieved,
  not consistent mastery. Require strong single-actor GOG and transfer results
  before swarm learning. Actor/learner concurrency is not the next milestone;
  GPU kernel parallelism and batched replay remain useful.
- Preserve a measured Dreamer control. Add JEPA-style prediction as a causal,
  action-conditioned objective that predicts an observation before consuming
  it. Predicting the current frozen DINO features from the posterior is already
  the existing feature-reconstruction control.
  Current positive runs use DINOv3 plus causal prediction, not LeVJEPA. Complete
  a bounded, matched causal-video frontend experiment with LeVJEPA as the first
  candidate; do not describe the DINO stepping stone as the full video pivot.
- Change one scientific variable per comparison. Report real interactions,
  learner updates, wall time, model/data provenance, all seeds, and failures.
  A short integration test or an historical score is not a matched benchmark.
  Match head structure and initialization when comparing objectives, and version
  changed heads or intrinsic hash schemes instead of reinterpreting old state.
  Verify the actual encoder file on restore; matching shapes are not identity.
  Require complete checkpoint tensors; a detected torn save is not atomic recovery.
- Profile learner stages, synchronization, and GPU idle time before committing
  days of compute. Check existing branches and local run artifacts before
  repeating old experiments. Preserve corrected full-precision gradients and
  full-recurrence row microbatching when integrating backend work.
- Test dense Atari, sparse Atari, and a small native persistent environment.
  Keep external reward and intrinsic reward separate. Retain an extrinsic-only
  control for every intrinsic-reward experiment.
  Record human guidance and the action actually executed; distinguish assisted
  behavior from unguided evaluation and game rewards from human feedback.
  Distinguish agent-collected online learning from forced-random coverage tests;
  verify learned behavior under frozen evaluation against untrained controls.
  Evaluate the declared final checkpoint and confirm independent training seeds;
  do not select a winning checkpoint or weaken acceptance after observing results.
  Record sampled/greedy action mode and recurrent-state initialization; isolate
  their effects when diagnosing a frozen-policy failure.
  Visual novelty is not task competence. Evaluate intrinsic exploration through
  held-out dynamics and later guided adaptation, with explicit reward provenance.
- Gate video encoders and pretraining on evidence: a usable pinned checkpoint,
  causal streaming semantics, native numerical parity, latency, and improved
  held-out control-relevant probes. A paper alone is not an implementation plan.
  Probe the trained recurrent belief before inferring a need for more temporal
  input from single-frame feature probes.
- Distinguish video-encoder initialization, action-conditioned world pretraining
  and policy-skill transfer. Missing action/reward labels are not NOOP/zero.
  Hold target titles out of source data and tuning; measure adaptation and
  forgetting. Retain the source policy when testing full-policy transfer, while
  declaring head, optimizer, normalizer, replay and recurrent-state resets.
- Reuse mind-games' launch, time-control, capture and input infrastructure.
  Verify its Kindle revision/API before integration; legacy BatchAgent adapters
  are not the current Dreamer path. Keep privileged reward/task observers outside
  policy inputs, and do not inherit unreported shaping or scripted gameplay.
- Respect the single stream of external consequences. Natural deaths and
  respawns are allowed; cloning or rewinding the game for training is not.
  Distinguish uncapped stepping, super-real-time playing plus training, and a
  free-running game without time control. Measure simulated/wall time with
  learning enabled; fast frozen inference is not training throughput. Preserve
  arrival order, actual action durations and observation gaps, and bound training
  debt. Try measured serial scheduling before any actor/learner separation.
- Delete superseded code and redundant documentation when they have no current
  purpose. Git retains history. Prefer small concrete modules over speculative
  frameworks, broad configuration surfaces, or premature swarm infrastructure.
- Work autonomously on authorized implementation, diagnostics, and experiments.
  Follow `/mnt/data/GUIDELINES.md` when available: attention is expensive, keep
  code self-describing, and only the user merges pull requests. Commit and push
  are permitted; do not send messages to other people without authorization.
- Preserve unrelated working-tree changes. Serialize GPU-heavy tests on a
  shared device and record the selected adapter. Run relevant formatting,
  Clippy, Rust/Python tests, and numerical checks for learning/backend changes.
