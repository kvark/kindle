# Some first Qbert movement disagreements cause losses under teacher continuation

Changing only the first command at a recorded policy–teacher disagreement
changes survival through the original loss endpoint in 5 of 25 selected cases.
The teacher-first branch survives all 25; policy-first survives 20. Both branches
follow the same feedback-aware teacher after that first command. This establishes
a local action effect under that controller, not an architectural cause.

| Paired outcome | Original loss endpoint | 64 actions from disagreement |
|---|---:|---:|
| Both survive | 20 | 16 |
| Teacher-first only survives | 5 | 7 |
| Policy-first only survives | 0 | 0 |
| Neither survives | 0 | 2 |

The 20 recoverable policy-first choices show that the first disagreement alone
does not account for most recorded losses at the primary endpoint. Subsequent
control matters. The longer endpoint is separate: one of the five primary
teacher-only cases later loses under both branches, while three additional
cases separate later. These interventions do not locate every harmful command
or establish whether a different controller could recover the five cases.

Four of the five primary differences substitute one movement command for
another; one substitutes a teacher movement for policy NOOP. The gap is therefore
not simply excessive waiting. At the 25 origins, the recorded actor assigns mean
probability .7031 to its own choice and .1687 to the teacher's. There are 12
policy NOOPs and one teacher NOOP. These probabilities come from the already
validated exact native readout, without new model inference. The summed
teacher-minus-policy reward through the matched endpoint is +25; rewards stop
at an earlier branch loss and are not new episode scores.

The [fixed protocol](../experiments/2026-11-01-cdp-qbert-first-disagreement.md)
selects the first disagreement in every one of the 32 existing failure windows
that contains one. All seven remaining windows contain no disagreement.
Selection uses commands, not rescue outcomes; in this corpus the 25 selected
windows coincide with the earlier teacher-rescued windows. All come from the
first natural episode in each of eight retained lanes. This is reused,
failure-selected development evidence, not average-policy accuracy or independent
reliability. Teacher-first survival at the original endpoint also reproduces
the earlier takeover suffix and is not an independent teacher-success test.

The initial action can change both the environment and the teacher's movement
memory through actual-action feedback. Subsequent commands are recomputed from
each branch's pixels. The experiment compares the same closed-loop controller
after different initial commands, not a fixed identical future command sequence
or the native policy continuing after a correction. Full native observations,
action repeat and sticky-action memory are preserved by process copies.

All 16,613 source actions / 66,440 frames reproduce, including pixels, rewards,
boundaries, score, life changes and first-pyramid events. Recorded children
match their source trajectories. The teacher branches reproduce all 1,373
available suffix rows from the previous interventions. Eight reverse-order
checks and parent emulator/teacher invariance pass. Intervention branches use
11,341 raw frames; recorded and reverse-order validation uses 6,276. Life gains
and losses are counted per raw frame so bonuses cannot mask losses.

Eleven CPU tests, the 1.415-second preflight and all 552 input hashes pass.
Analysis takes 65.045 seconds; the controller finishes in 65.779 seconds at
05:44:58 UTC on November 1, source `8a8b433`. No failure, retry, native model,
GPU allocation or learner update occurs. Existing training remains terminal.

The result supports further investigation of later movement choices and their
training coverage. It does not yet choose a new loss, model size, encoder,
teacher weight or learner. In particular, it supplies no reason to repeat the
previous unsuccessful lower-NOOP-weight recipe. The five-game goal remains
unfinished; no automatic successor is selected.

Evidence: `/x/Temp/kindle-qbert-first-disagreement-20261101`.

| Artifact | SHA256 |
|---|---|
| Declaration | `7872980e769a889d0a3a3c104eac73ce2545e2ef58ecc5beb1f4a17bad12525a` |
| Review | `a37af3cc30d771e0a7c91b1560316a1ce2e86f090162d4a6c433e5c22e9966c5` |
| Branch trajectories | `dc86cc5f478c42d6e944482de751b62ffcbd8c193fb3d32458af4d8dea6cf541` |
| Completion | `20c01b537cc62de8feb5eb324dd7d52c521e71366b7421e8c1059bf075d13af2` |
