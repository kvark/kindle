# Qbert: explicit expert actor imitation at the existing 200k budget

**Completed:**200k actions/799,509 frames/49,939 updates in20m11s. Fixed unaided
frozen1083.333/2of24 pyramids versus809.375/0 for unaided training and750.000/0
for capped guidance. All audits/comparison pass; service exits23:23:23 UTC.
Coverage18.625/21; original mean15000/90% gate still fails. Retain the candidate
for a separately selected fresh-seed check; no extension or successor queued.
See [the completed result](../results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate).

Selected after the completed native gradient, actor-only update and bounded
Qbert checks. The question is whether learning directly from the trusted pixel
rule's executed actions improves **unaided frozen play**. Capped guidance alone
scored 750.000, below the unaided 809.375; both cleared 0/24 first pyramids.

The optional native path passes 127 Rust CPU tests, strict Clippy, 477 packaged
Python tests, analytic weighted/masked GPU gradients and an actor update with
the imagined policy loss disabled. Fresh 4096-action learning and 1024-action
frozen restore pass full audits. These checks establish execution and gradient
behavior, not game competence. See [the bounded result](../results/2026-10-25-cdp-cheap-qbert.md#native-expert-imitation-smoke-passes-task-benefit-remains-unmeasured).

## Selected protocol

- Fresh seed1009/evaluation4000001009; reuse neither smoke weights nor experience.
- 4M preset (3,578,225 parameters), RGB48 from native max-pooled frames with one
  GPU resize, Auto, eight environments, B8/T16/H15/R32, full18 actions, centered
  CDP coefficient1, actor0 and original learning rates; actor/critic gradients
  remain disabled. Native learning and inference stay on Meganeura/Blade.
- Select 5% of episodes for the pixel-only Qbert rule, capped at640 actions per
  selected episode. Actual guided-action share is measured separately. RAM
  task observers run only in post-hoc analysis, never as policy inputs.
- Add explicit expert imitation at scale1. NOOP target weight is .05; other
  actions have weight1. Replay state[t] targets the trusted incoming action at
  [t+1], respecting boundaries. The loss averages over all replay rows; it does
  not normalize away NOOP downweighting. No additional model parameters.
- Exactly200k aggregate actions, expected49,939 updates, first update248 and
  zero residual credit. Keep a100k/24,939-update recovery checkpoint audit.
- Hard training cap1500seconds (25minutes). No resume, extension, cap increase,
  earlier-checkpoint selection or automatic next allocation.
- Fixed final and actual-initial assessments: first3 natural episodes per lane,
  24 selected per arm, guidance disabled, zero learning and no truncations.
  Preserve excess episodes/tails and require346 unchanged tensors per arm.
  The original mean15000/90%-first-pyramid gate stays unchanged.
- Each frozen arm has a300second/600k-action ceiling; CPU audit cap1200seconds,
  service cap3360seconds. Run complete saved-action/pixel replays, videos,
  host guards, curves, coverage and comparison against both completed controls.

## Runtime and evidence

Owner: `kindle-expert-qbert-cap640-200k-20261026`.
Trial: `/x/Temp/kindle-expert-20261026/qbert-4m-48-r32-expert1-cap640-200k`.
The declaration pins the queue, controller, comparison, postprocessor, recovery
watcher, analysis sources, runtime modules, control reviews and completed checks.
Read-only preflight passes34 pins and both completed control readers; no agent
is constructed. A mistaken smoke completion filename in CPU preparation was
corrected before launch and is retained in `preparation-correction.json`.

Native/CLI source`79c9a7f`, isolated runtime
`/x/Temp/kindle-expert-20261026/runtime`, native SHA256
`7c07d6702e37ca11aa20c1b35b48951f0ed01e0e1dfe72e187eac52253f8455e`.
The release is unchanged by the subsequent actor-only test strengthening.
Original native/checkpoints remain retained; the packaged build cache is released.
Host: RX7900XT/RADV NAVI31, boot`a8648671`, kernel7.0.0-38,
AMD module`9BDBC866F55BF296F1B7FA2`, Mesa26.0.8. Use the existing host guard;
stop on genuine GPU faults or numerical failure, without automatic recovery.

## Interpretation

This compares a recipe, including the new auxiliary objective and native build,
against capped guidance and unaided training at matched action/update budgets.
Require all346 actual initial tensors and initial semantic trajectories to
match. The unaided control used an older host kernel; capped guidance used the
current one. Timing and outcome differences do not isolate a mechanism.
Seed1009 and the teacher cap are reused development choices, not held-out tests.
Teacher-assisted online scores cannot establish unaided learning. Any useful
result needs a separately selected seed confirmation; none is queued here.

## Launch and startup

Service starts23:01:34 UTC/source`bae5adc`, invocation
`0c38a5bbd50441a9b5a89e0f59c2ac8f`, main PID55861. The immutable selected
declaration SHA256 is
`6d4669e507036a109aac4257e232b9582acfdc7fe1e3637ef488c85bb53ae319`.
Read-only startup review matches all346 actual initial tensors and the first six
updates against both retained control and qualified smoke:
248:1/256:2/256:3/264:4/264:5/272:6. At8480 complete logged actions,2057 learner
reports are finite,755 actions are explicitly trusted and1220 reports contain
expert labels. These are execution checks; frozen competence remains unmeasured.

A handwritten startup expectation incorrectly placed the second update at248.
The CPU checker was corrected to compare the saved control/smoke traces, which
match the live run. `startup-checker-correction.json` retains the original
expectation and correction. The learner and GPU stages were not restarted.

Separate read-only units are attached under the same `kindle-expert-qbert-cap640`
prefix: `recovery-20261026` audits100k before replacement, `progress-20261026`
observes completed log prefixes, and `post-20261026` waits for terminal success
before plotting/comparison. Their deadlines do not extend the learner cap or
launch another GPU allocation. Consult live service state before any GPU work.

## Completed evidence

The100k/24,939 recovery passes at23:12:14 UTC with346 finite tensors and26,161
finite reports in its complete104,896-action log prefix. Review SHA256
`8b2c5a8311ab24e1c81916c34696de2a9042dae526b5733b12376c13531dd899`.
The fixed final and own-initial assessments complete with zero learning and
truncations,346 unchanged tensors each, full replays/videos and guards.
Both controls match all346 actual-initial tensors and initial semantic traces.
All34 source/evidence pins remain unchanged after completion.

The main service exits successfully23:23:23 UTC, total21m48.267s; the completed
CPU comparison exits23:23:46 and the additional CPU exposure analysis exits
23:24:02. All observer handles are terminal. The original CPU preparation and
startup-checker corrections remain; no GPU stage is repeated.

Full results and interpretation are in the linked result report. Completion
SHA256`bad2bcff8bd51ba954ee1bfefe457216e30ae3b0211fd3a97713e3ae9e00056b`;
comparison`7b4edd990733ac7754ec49a5c693d9af66058ba9344c5c2244da4e01776dab39`.
