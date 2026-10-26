# Breakout actor/critic gradients: staged option

**CPU preparation only; no GPU allocation is selected.** Six-action Pong remains
the active learner. Its pinned runner, runtime and analysis files are unchanged.
Any application or bounded GPU check follows review of that completed run.

The compact four-action Breakout control scores9.542 at200k actions in20m59s,
with0/24 two-wall successes. Shorter imagination and smaller intrinsic reward
have not improved the completed results. Historical state probes find ball and
paddle information, without proving sufficient precision for control. Test a
different hypothesis: more task-directed representation learning might help
the policy use the available state within a small experience budget.

Native`actor_critic_gradient` already adds policy and imagined-value losses at
the initial posterior states to world training, using stopped targets and
frozen behavior heads. Later imagined states remain detached; the contribution
is normalized by imagination length. This is not the first task supervision:
reward, continuation and replay-value gradients are already active. No useful
learning effect or cause of Breakout's weakness is established by reading code.

The CLI currently rejects CDP with`--actor-critic-gradient`. A staged two-line
change allows that existing opt-in combination with the learned CNN. Default
CDP remains false; frozen-encoder and restored-run overrides still reject.
No native edit or rebuild. The staged CPU suite passes313 tests, including
default/opt-in forwarding across CDP centering, image sizes and batch shapes,
and the existing invalid-config/restore refusals. These are CLI tests, not GPU
execution, gradient-value or learning evidence. The exact patch passes
`git apply --check`; production files remain byte-identical to their originals.

Source review predicts11 extra frozen`behavior.actor.*` tensors in the world
checkpoint: world280/behavior66/slow11,357 total, versus346 in the control.
They have the existing actor shapes and no world-optimizer moments. The current
346-tensor audit helpers must not be reused blindly or weakened. A later smoke
must verify the exact expected keys/shapes/dtypes against saved exports, all
finite values, behavior-copy consistency and complete frozen invariance.
This count remains an unmeasured expectation until that smoke runs.

If selected after Pong review, start with one bounded fresh4096-action check:
centered CDP4M/RGB64/Auto/N8/B8/T16/H15/R32, minimal4, coefficient1,
actor mixture.01, original rates, seed1009 and no aid. Only native
actor_critic_gradient changes. Expected963 updates, first248, zero debt and
300-second training cap. Save actual initial/final and perform one ordinary
1024-action frozen restore with zero updates, full tensor invariance and a
60-second cap. Require finite/nonzero initial policy/value loss reports and
changed trainable world parameters; those checks do not prove correct gradients
or competence. Record all shared initial-tensor comparisons without presuming
equality. Use existing host guards and stop on faults/numerical failures.

No full learning life follows automatically. A possible later fresh200k/30-minute
candidate would retain the fixed final/actual-initial24-natural-episode protocol
and original864-point/90%-two-wall gate, with complete replays and the explicitly
verified357-tensor schema. That run would be separate from validation experience.
No cap increases, checkpoint selection, warm extension or broad campaign.

Staging:`/x/Temp/kindle-cheap-20261025/cdp-actor-critic-cli`.
Production runner SHA256`1e4b7c83768c82226200cb41b166543a75ab5993b750593cc3e88a44f3be54da`;
staged runner`d190fd9ad314c6d242d68163fcdf89db3d406fab0f6fe9e51cd9cd03eddc1ee5`;
patch`d56aa285959b0b9ef461826b780ea6e1ec2ef2868dc976c93d975cca76eb93ce`;
validation`2e6364f2aaa5b8072d98cfb53e8ecfc3265aaea2685fe94261a5d4487f7c80f6`.
Schema`e60b9e6d2ba653872cc4bfdf55fd0ae42ca91de08718c0183c72e5ba2372d9d1`
(`expected-checkpoint-schema.json`) retains the control-derived complete key/shape
map and its source/checkpoint hashes. Preparation source`88ec353`; native remains
`bfb8c31`, production runner`22bd61b`, CPU replay`5043b7a`.
