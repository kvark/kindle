# Frozen Qbert assessment with modal posterior categories

The corrected bounded validation passes the guarded posterior argmax/RNG/reset/
learning-entry test, unchanged default vector/serial parity, and two1024-action
frozen restores. Both preserve357 tensors; the default reproduces128 original
transition rows exactly. No natural episodes complete. Select one full frozen
assessment to measure task quality, with no training or weight update.

Use native sourceb76f91c/SHA7555bf9 from
`/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/runtime`.
The original7fea347 test read logits without waiting for their producer and
failed; its service/declaration/logs/executable remain preserved. The revised
test adds waits, and the production native library hash is identical.135 Rust
CPU tests,625 packaged Python tests and strict Clippy pass. The matching
packaging receipt pins the wheel,source and corrected validation executable;
the857,454,521-byte build cache is released.

Retained source:
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
Use `final` and its own `initial`. Both retain12M48/Auto/N8/B16/T16/H15/R16,
centered CDP,expert100/world gradients/all-state labels in their saved config,
but frozen evaluation supplies no guidance or labels and never calls learning.
Keep native frames/one GPU48 resize,full18,sticky.25/repeat4 and seed4000004013.

Four fixed arms, in order: final/sample posterior, final/greedy posterior,
initial/sample posterior, initial/greedy posterior. Actions are greedy in every
arm. Select the first3 natural episodes per8 lanes=24; retain extra episodes
and partial tails. Each arm has a600k-action/300-second cap. No truncation counts
toward the quota; an incomplete cohort fails the assessment and is not retried.
Do not change caps, checkpoints, seeds or action modes after seeing results.

Before each GPU job, verify declared source/runtime/checkpoint/adapter pins and
the existing host guard. Stop on real host faults or numerical failures; no
driver recovery or automatic retries. Require zero updates,357 finite/invariant
tensors,exact restore/source/config/counters and complete recorded-action/outcome
replay plus a full lane0 video for every arm. Default arms must reproduce the
retained greedy semantic transition/episode/reset traces and scores11984.375
final/80.208 initial after excluding only finite wall-clock timestamps/durations.
The other posterior mode intentionally changes trajectories. Ordinary live logs
do not provide input pixel hashes.

Report the original mean15000 and90%-first-pyramid gate,paired initial gains,
per-lane/episode scores and evaluation wall cost. Keep the original200k-action/
12,477-update/30m27s training cost. This is a changed inference rule on one reused
learner/cohort,not new training or independent seed confirmation. Selecting
modal categories can hurt a policy trained with sampled states. Neither a gain
nor a passing cohort establishes a global default or five-game reliability.

Evidence belongs under
`/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/full`.
No extension,new learner,other-game allocation or automatic confirmation is
selected by this protocol.
